import numpy as np
import pytorch_lightning as pl
import torch

from src.vtb_code.metrics import PrecisionK, MeanReciprocalRankK
from src.vtb_code import L2Scorer, MeanLoss

from dltranz.trx_encoder import TrxEncoder, PaddedBatch
from dltranz.seq_encoder.rnn_encoder import RnnEncoder
from dltranz.seq_encoder.utils import LastStepEncoder


class CustomTrxTransform(torch.nn.Module):
    def __init__(self, trx_amnt_quantiles):
        super().__init__()
        self.trx_amnt_quantiles = torch.nn.Parameter(trx_amnt_quantiles, requires_grad=False)

    def forward(self, x):
        x.payload['transaction_amt_q'] = torch.bucketize(x.payload['transaction_amt'], self.trx_amnt_quantiles) + 1
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
        x.payload['weekday'] = et.div(60 * 60 * 24, rounding_mode='floor') % 7 + 1
        x.payload['day_diff'] = torch.clamp(torch.diff(et_day, prepend=et_day[:, :1], dim=1), 0, 14)
        return x


class PBLinear(torch.nn.Linear):
    def forward(self, x: PaddedBatch):
        return PaddedBatch(super().forward(x.payload), x.seq_lens)


class PBL2Norm(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return PaddedBatch(self.beta * x.payload / (x.payload.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5),
                           x.seq_lens)


class MLMPretrainModule(pl.LightningModule):
    def __init__(self, data_type, params,
                 lr, weight_decay,
                 max_lr, pct_start, total_steps,
                 ):
        super().__init__()
        self.save_hyperparameters()

        common_trx_size = params['common_trx_size']
        self.seq_encoder = None

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, common_trx_size), requires_grad=True)
        self.transf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=common_trx_size,
                nhead=params['transf.nhead'],
                dim_feedforward=params['transf.dim_feedforward'],
                dropout=params['transf.dropout'],
                batch_first=True,
            ),
            num_layers=params['transf.num_layers'],
            norm=torch.nn.LayerNorm(common_trx_size) if params['transf.norm'] else None,
        )

        if params['transf.use_pe']:
            self.pe = torch.nn.Parameter(self.get_pe(), requires_grad=False)
        else:
            self.pe = None
        self.padding_mask = torch.nn.Parameter(torch.tensor([True, False]).bool(), requires_grad=False)

        self.train_mlm_loss_all = MeanLoss(compute_on_step=False)
        self.valid_mlm_loss_all = MeanLoss(compute_on_step=False)
        self.train_mlm_loss_self = MeanLoss(compute_on_step=False)
        self.valid_mlm_loss_self = MeanLoss(compute_on_step=False)

    def get_pe(self):
        max_len = self.hparams.params['transf.max_len']
        H = self.hparams.params['common_trx_size']
        f = 2 * np.pi * torch.arange(max_len).view(1, -1, 1) / \
            torch.exp(torch.linspace(*np.log([4, max_len]), H // 2)).view(1, 1, -1)
        return torch.cat([torch.sin(f), torch.cos(f)], dim=2)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=True,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]

    def get_mask(self, x: PaddedBatch):
        return torch.bernoulli(x.seq_len_mask.float() * self.hparams.params['mlm.replace_proba']).bool()

    def mask_x(self, x: PaddedBatch, mask):
        return torch.where(mask.unsqueeze(2).expand_as(x.payload),
                           self.token_mask.expand_as(x.payload), x.payload)

    def get_neg_ix(self, mask, neg_type):
        """Sample from predicts, where `mask == True`, without self element.
        For `neg_type='all'` - sample from predicted tokens from batch
        For `neg_type='self'` - sample from predicted tokens from row
        """
        if neg_type == 'all':
            mn = mask.float().view(1, -1) - \
                 torch.eye(mask.numel(), device=mask.device)[mask.flatten()]
            neg_ix = torch.multinomial(mn, self.hparams.params['mlm.neg_count_all'])
            b_ix = neg_ix.div(mask.size(1), rounding_mode='trunc')
            neg_ix = neg_ix % mask.size(1)
            return b_ix, neg_ix
        if neg_type == 'self':
            mask_ix = mask.nonzero(as_tuple=False)
            one_pos = torch.eye(mask.size(1), device=mask.device)[mask_ix[:, 1]]
            mn = mask[mask_ix[:, 0]].float() - one_pos
            mn = mn + 1e-9 * (1 - one_pos)
            neg_ix = torch.multinomial(mn, self.hparams.params['mlm.neg_count_self'], replacement=True)
            b_ix = mask_ix[:, 0].view(-1, 1).expand_as(neg_ix)
            return b_ix, neg_ix
        raise AttributeError(f'Unknown neg_type: {neg_type}')

    def sentence_encoding(self, x: PaddedBatch):
        return None

    def mlm_loss(self, x: PaddedBatch, neg_type, x_orig: PaddedBatch):
        mask = self.get_mask(x)
        masked_x = self.mask_x(x, mask)
        B, T, H = masked_x.size()

        if self.pe is not None:
            if self.training:
                start_pos = np.random.randint(0, self.hparams.params['transf.max_len'] - T, 1)[0]
            else:
                start_pos = 0
            pe = self.pe[:, start_pos:start_pos + T]
            masked_x = masked_x + pe

        se = self.sentence_encoding(x_orig)
        if se is not None:
            masked_x = masked_x + se

        out = self.transf(masked_x, src_key_padding_mask=self.padding_mask[x.seq_len_mask])

        if self.pe is not None:
            out = out - pe
        if se is not None:
            out = out - se

        target = x.payload[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask, neg_type)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        out_samples = torch.cat([predict, negative], dim=1)
        probas = torch.softmax((target * out_samples).sum(dim=2), dim=1)
        loss = -torch.log(probas[:, 0])
        return loss

    def training_step(self, batch, batch_idx):
        (x_trx, _), = batch

        z_trx = self.seq_encoder(x_trx)  # PB: B, T, H

        loss_mlm = self.mlm_loss(z_trx, neg_type='all', x_orig=x_trx)
        self.train_mlm_loss_all(loss_mlm)
        loss_mlm_all = loss_mlm.mean()
        self.log(f'loss/mlm_{self.hparams.data_type}', loss_mlm_all)

        loss_mlm = self.mlm_loss(z_trx, neg_type='self', x_orig=x_trx)
        self.train_mlm_loss_self(loss_mlm)
        loss_mlm_self = loss_mlm.mean()
        self.log(f'loss/mlm_{self.hparams.data_type}_self', loss_mlm_self)

        return loss_mlm_all + loss_mlm_self

    def validation_step(self, batch, batch_idx):
        (x_trx, _), = batch
        z_trx = self.seq_encoder(x_trx)  # PB: B, T, H

        loss_mlm = self.mlm_loss(z_trx, neg_type='all', x_orig=x_trx)
        self.valid_mlm_loss_all(loss_mlm)

        loss_mlm = self.mlm_loss(z_trx, neg_type='self', x_orig=x_trx)
        self.valid_mlm_loss_self(loss_mlm)

    def training_epoch_end(self, _):
        self.log(f'metrics/train_{self.hparams.data_type}_mlm', self.train_mlm_loss_all, prog_bar=False)
        self.log(f'metrics/train_{self.hparams.data_type}_mlm_self', self.train_mlm_loss_self, prog_bar=False)

    def validation_epoch_end(self, _):
        self.log(f'metrics/valid_{self.hparams.data_type}_mlm', self.valid_mlm_loss_all, prog_bar=True)
        self.log(f'metrics/valid_{self.hparams.data_type}_mlm_self', self.valid_mlm_loss_self, prog_bar=True)


class MLMPretrainModuleTrx(MLMPretrainModule):
    def __init__(self,
                 trx_amnt_quantiles,
                 params,
                 lr, weight_decay,
                 max_lr, pct_start, total_steps,
                 ):
        super().__init__(data_type='trx',
                         params=params,
                         lr=lr, weight_decay=weight_decay,
                         max_lr=max_lr, pct_start=pct_start, total_steps=total_steps,
                         )
        self.save_hyperparameters()

        common_trx_size = self.hparams.params['common_trx_size']
        t = TrxEncoder(self.hparams.params['trx_seq.trx_encoder'])
        self.seq_encoder = torch.nn.Sequential(
            CustomTrxTransform(trx_amnt_quantiles=trx_amnt_quantiles),
            DateFeaturesTransform(),
            t, PBLinear(t.output_size, common_trx_size),
            PBL2Norm(self.hparams.params['mlm.beta']),
        )


class MLMPretrainModuleClick(MLMPretrainModule):
    def __init__(self, params,
                 lr, weight_decay,
                 max_lr, pct_start, total_steps,
                 ):
        super().__init__(data_type='click',
                         params=params,
                         lr=lr, weight_decay=weight_decay,
                         max_lr=max_lr, pct_start=pct_start, total_steps=total_steps,
                         )
        self.save_hyperparameters()

        common_trx_size = self.hparams.params['common_trx_size']
        t = TrxEncoder(self.hparams.params['click_seq.trx_encoder'])
        self.seq_encoder = torch.nn.Sequential(
            CustomClickTransform(),
            DateFeaturesTransform(),
            t, PBLinear(t.output_size, common_trx_size),
            PBL2Norm(self.hparams.params['mlm.beta']),
        )


class PBLayerNorm(torch.nn.LayerNorm):
    def forward(self, x: PaddedBatch):
        return PaddedBatch(super().forward(x.payload), x.seq_lens)


class PairedModule(pl.LightningModule):
    def __init__(self, params, k,
                 lr, weight_decay,
                 max_lr, pct_start, total_steps,
                 beta, neg_count,
                 mlm_model_trx, mlm_model_click,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['mlm_model_trx', 'mlm_model_click'])

        common_trx_size = mlm_model_trx.hparams.params['common_trx_size']
        self.rnn_enc = torch.nn.Sequential(
            RnnEncoder(common_trx_size, params['rnn']),
            LastStepEncoder(),
            #             NormEncoder(),
        )
        self._seq_encoder_trx = torch.nn.Sequential(
            mlm_model_trx.seq_encoder,
            PBLayerNorm(common_trx_size),
        )
        self._seq_encoder_click = torch.nn.Sequential(
            mlm_model_click.seq_encoder,
            PBLayerNorm(common_trx_size),
        )
        # self.mlm_model_click = mlm_model_click

        self.cls = torch.nn.Sequential(
            L2Scorer(),
        )

        self.train_precision = PrecisionK(k=k, compute_on_step=False)
        self.train_mrr = MeanReciprocalRankK(k=k, compute_on_step=False)
        self.valid_precision = PrecisionK(k=k, compute_on_step=False)
        self.valid_mrr = MeanReciprocalRankK(k=k, compute_on_step=False)

    def seq_encoder_trx(self, x):
        x = self._seq_encoder_trx(x)
        return self.rnn_enc(x)

    def seq_encoder_click(self, x_orig):
        x = self._seq_encoder_click(x_orig)
        #         x = PaddedBatch(
        #             x.payload + self.mlm_model_click.sentence_encoding(x_orig),
        #             x.seq_lens,
        #         )
        return self.rnn_enc(x)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=True,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]

    def loss_fn_p(self, embeddings, labels, ref_emb, ref_labels):
        beta = self.hparams.beta
        neg_count = self.hparams.neg_count

        pos_ix = (labels.view(-1, 1) == ref_labels.view(1, -1)).nonzero(as_tuple=False)
        pos_labels = labels[pos_ix[:, 0]]
        neg_w = ((pos_labels.view(-1, 1) != ref_labels.view(1, -1))).float()
        neg_ix = torch.multinomial(neg_w, neg_count - 1)
        all_ix = torch.cat([pos_ix[:, [1]], neg_ix], dim=1)
        logits = -(embeddings[pos_ix[:, [0]]] - ref_emb[all_ix]).pow(2).sum(dim=2)
        logits = logits * beta
        logs = -torch.log(torch.softmax(logits, dim=1))[:, 0]
        #         logs = torch.relu(logs + np.log(0.1))
        return logs.mean()

    def training_step(self, batch, batch_idx):
        # pairs
        x_trx, l_trx, m_trx, x_click, l_click, m_click = batch
        z_trx = self.seq_encoder_trx(x_trx)  # B, H
        z_click = self.seq_encoder_click(x_click)  # B, H
        loss_pt = self.loss_fn_p(embeddings=z_trx, labels=l_trx, ref_emb=z_click, ref_labels=l_click)
        self.log('loss/loss_pt', loss_pt)

        loss_pc = self.loss_fn_p(embeddings=z_click, labels=l_click, ref_emb=z_trx, ref_labels=l_trx)
        self.log('loss/loss_pc', loss_pc)

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

        return loss_pt + 0.1 * loss_pc  # loss_pc

    def training_epoch_end(self, _):
        self.log('train_metrics/precision', self.train_precision, prog_bar=True)
        self.log('train_metrics/mrr', self.train_mrr, prog_bar=True)
