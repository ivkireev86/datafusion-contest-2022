import pytorch_lightning as pl
import torch
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.seq_encoder import create_encoder
from dltranz.seq_encoder.utils import NormEncoder
from dltranz.trx_encoder import PaddedBatch

from vtb_code.metrics import PrecisionK, MeanReciprocalRankK
from vtb_code.models import L2Scorer

from dltranz.trx_encoder import TrxEncoder, PaddedBatch
from dltranz.seq_encoder.rnn_encoder import RnnEncoder
from dltranz.seq_encoder.utils import LastStepEncoder


class CustomTrxTransform(torch.nn.Module):
    def forward(self, x):
        #         x.payload['mcc_code'] = torch.clamp(x.payload['mcc_code'], 0, 300)
        #         x.payload['c_cnt_clamp'] = torch.clamp(x.payload['c_cnt'], 0, 20).int()
        return x


class CustomClickTransform(torch.nn.Module):
    def forward(self, x):
        #         x.payload['cat_id'] = torch.clamp(x.payload['cat_id'], 0, 300)
        #         x.payload['level_0'] = torch.clamp(x.payload['level_0'], 0, 200)
        #         x.payload['level_1'] = torch.clamp(x.payload['level_1'], 0, 200)
        #         x.payload['level_2'] = torch.clamp(x.payload['level_2'], 0, 200)
        #         x.payload['c_cnt_clamp'] = torch.clamp(x.payload['c_cnt'], 0, 20).int()
        return x


class DateFeaturesTransform(torch.nn.Module):
    def forward(self, x):
        et = x.payload['event_time'].int()
        et_day = et.div(24 * 60 * 60, rounding_mode='floor').int()
        x.payload['hour'] = et.div(60 * 60, rounding_mode='floor') % 24 + 1
#         x.payload['weekday'] = et.div(60 * 60 * 24, rounding_mode='floor') % 7 + 1
#         x.payload['hour_s'] = torch.sin(2 * np.pi * (et % (60 * 60 * 24)) / (60 * 60 * 24))
#         x.payload['hour_c'] = torch.cos(2 * np.pi * (et % (60 * 60 * 24)) / (60 * 60 * 24))
#         x.payload['day_diff'] = torch.clamp(torch.diff(et_day, prepend=et_day[:, :1], dim=1), 0, 14)
#         x.payload['day_diff_c'] = torch.clamp(torch.diff(et, prepend=et[:, :1], dim=1) / (60 * 60 * 24), 0, 14)
        return x


class PBLinear(torch.nn.Linear):
    def forward(self, x: PaddedBatch):
        return PaddedBatch(super().forward(x.payload), x.seq_lens)


class PairedColesModule(pl.LightningModule):
    def __init__(self, params, sampling_strategy_params, loss_params, k,
                 lr, weight_decay,
                 step_size, gamma,
                 base_lr, max_lr, step_size_up, step_size_down,
                 ):
        super().__init__()
        self.save_hyperparameters()

        t = TrxEncoder(params['trx_seq.trx_encoder'])
        print(t.output_size)
        self.rnn_enc = torch.nn.Sequential(
            RnnEncoder(params['common_trx_size'], params['rnn']),
            LastStepEncoder(),
            NormEncoder(),
        )
        self.seq_encoder_trx_size = params['rnn.hidden_size']
        self._seq_encoder_trx = torch.nn.Sequential(
            CustomTrxTransform(),
            DateFeaturesTransform(),
            t, PBLinear(t.output_size, params['common_trx_size']),
        )
        t = TrxEncoder(params['click_seq.trx_encoder'])
        print(t.output_size)
        self._seq_encoder_click = torch.nn.Sequential(
            CustomClickTransform(),
            DateFeaturesTransform(),
            t, PBLinear(t.output_size, params['common_trx_size']),
        )

        self.cls = torch.nn.Sequential(
            L2Scorer(),
        )

        sampling_strategy = get_sampling_strategy(sampling_strategy_params)
        self.loss_fn = get_loss(loss_params, sampling_strategy)

        self.train_precision = PrecisionK(k=k, compute_on_step=False)
        self.train_mrr = MeanReciprocalRankK(k=k, compute_on_step=False)
        self.valid_precision = PrecisionK(k=k, compute_on_step=False)
        self.valid_mrr = MeanReciprocalRankK(k=k, compute_on_step=False)

    def seq_encoder_trx(self, x):
        x = self._seq_encoder_trx(x)
        return self.rnn_enc(x)

    def seq_encoder_click(self, x):
        x = self._seq_encoder_click(x)
        return self.rnn_enc(x)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        else:
            sheduler = torch.optim.lr_scheduler.CyclicLR(
                optim,
                base_lr=self.hparams.base_lr, max_lr=self.hparams.max_lr,
                step_size_up=self.hparams.step_size_up,
                step_size_down=self.hparams.step_size_down,
                cycle_momentum=False,
            )
            scheduler = {'scheduler': sheduler, 'interval': 'step'}
        return [optim], [scheduler]

    #     def forward(self, batch):
    #         return logits

    def training_step(self, batch, batch_idx):
        x_trx, l_trx, m_trx, x_click, l_click, m_click = batch

        self.log('seq_le/trx_mean', x_trx.seq_lens.float().mean())
        self.log('seq_len/click_mean', x_click.seq_lens.float().mean())

        z_trx = self.seq_encoder_trx(x_trx)  # B, H
        z_click = self.seq_encoder_click(x_click)  # B, H

        B = z_trx.size(0)
        device = z_trx.device

        loss = self.loss_fn(
            torch.cat([z_trx, z_click], dim=0),
            torch.cat([l_trx, l_click], dim=0),
        )
        self.log('loss/loss_ml', loss, prog_bar=True)

        with torch.no_grad():
            out = -(z_trx.unsqueeze(1) - z_click.unsqueeze(0)).pow(2).sum(dim=2)
            out = out[m_trx == 0][:, m_click == 0]
            T, C = out.size()
            assert T == C
            n_samples = z_trx.size(0) // (l_trx.max().item() + 1)
            for i in range(n_samples):
                l2 = out[i::n_samples, i::n_samples]
                self.train_precision(l2)
                self.train_mrr(l2)

        return loss

    def training_epoch_end(self, _):
        self.log('train_metrics/precision', self.train_precision, prog_bar=True)
        self.log('train_metrics/mrr', self.train_mrr, prog_bar=False)
