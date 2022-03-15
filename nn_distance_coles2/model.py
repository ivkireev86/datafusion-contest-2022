import pytorch_lightning as pl
import torch
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.seq_encoder import create_encoder
from dltranz.seq_encoder.utils import NormEncoder
from dltranz.trx_encoder import PaddedBatch

from vtb_code.metrics import PrecisionK, MeanReciprocalRankK
from vtb_code.models import L2Scorer


class CustomTrxTransform(torch.nn.Module):
    def forward(self, x: PaddedBatch):
        #         x.payload['mcc_code'] = torch.clamp(x.payload['mcc_code'], 0, 300)
        et = x.payload['event_time'].int()
        x.payload['hour'] = et.div(60 * 60, rounding_mode='floor') % 24 + 1
        x.payload['weekday'] = et.div(60 * 60 * 24, rounding_mode='floor') % 7 + 1
        return x


class CustomClickTransform(torch.nn.Module):
    def forward(self, x: PaddedBatch):
        #         x.payload['cat_id'] = torch.clamp(x.payload['cat_id'], 0, 300)
        #         x.payload['level_0'] = torch.clamp(x.payload['level_0'], 0, 200)
        #         x.payload['level_1'] = torch.clamp(x.payload['level_1'], 0, 200)
        #         x.payload['level_2'] = torch.clamp(x.payload['level_2'], 0, 200)
        et = x.payload['event_time'].int()
        x.payload['hour'] = et.div(60 * 60, rounding_mode='floor') % 24 + 1
        x.payload['weekday'] = et.div(60 * 60 * 24, rounding_mode='floor') % 7 + 1
        return x


class PairedColesModule(pl.LightningModule):
    def __init__(self, params, sampling_strategy_params, loss_params, k,
                 lr, weight_decay,
                 step_size, gamma,
                 base_lr, max_lr, step_size_up, step_size_down,
                 ):
        super().__init__()
        self.save_hyperparameters()

        m = create_encoder(params['trx_seq'], is_reduce_sequence=True)
        self.seq_encoder_trx_size = m.embedding_size
        self.seq_encoder_trx = torch.nn.Sequential(
            CustomTrxTransform(),
            m,
            NormEncoder(),
        )
        m = create_encoder(params['click_seq'], is_reduce_sequence=True)
        self.seq_encoder_click_size = m.embedding_size
        self.seq_encoder_click = torch.nn.Sequential(
            CustomClickTransform(),
            m,
            NormEncoder(),
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

    def training_step(self, batch, batch_idx):
        x_trx, l_trx, out_trx, x_click, l_click, out_click = batch

        self.log('seq_len/trx_mean', x_trx.seq_lens.float().mean())
        self.log('seq_len/click_mean', x_click.seq_lens.float().mean())

        z_trx = self.seq_encoder_trx(x_trx)  # B, H
        z_click = self.seq_encoder_click(x_click)  # B, H

        loss = self.loss_fn(
            torch.cat([z_trx, z_click], dim=0),
            torch.cat([l_trx, l_click], dim=0),
        )
        self.log('loss/loss_ml', loss, prog_bar=True)

        #         cross_l2 = ((z_trx.unsqueeze(1) - z_click.unsqueeze(0)).pow(2).sum(dim=2) + 1e-6).pow(0.5)

        #         l_0 = torch.sigmoid(self.l_0) * 2
        #         # out_of_match
        #         oom_loss = torch.relu(l_0 - cross_l2[out_trx.bool()]).sum()
        #         # with match
        #         l_match = l_trx.unsqueeze(1) == l_click.unsqueeze(0)
        #         match_loss = torch.relu(cross_l2[l_match] - l_0).sum()
        #         l_0_loss = oom_loss + match_loss
        #         self.log('loss/loss_l0', l_0_loss, prog_bar=False)

        out = -(z_trx[~out_trx.bool()].unsqueeze(1) - z_click[~out_click.bool()].unsqueeze(0)).pow(2).sum(dim=2)
        n_samples = z_trx.size(0) // (l_trx.max().item() + 1)
        for i in range(n_samples):
            l2 = out[i::n_samples, i::n_samples]
            self.train_precision(l2)
            self.train_mrr(l2)

        return loss  # + l_0_loss

    def training_epoch_end(self, _):
        self.log('train_metrics/precision', self.train_precision, prog_bar=True)
        self.log('train_metrics/mrr', self.train_mrr, prog_bar=False)


