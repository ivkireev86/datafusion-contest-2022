import pytorch_lightning as pl
import torch
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.seq_encoder import create_encoder
from dltranz.seq_encoder.utils import NormEncoder
from dltranz.trx_encoder import PaddedBatch

from vtb_code.metrics import PrecisionK, MeanReciprocalRankK


class VICRegLoss(torch.nn.Module):
    def __init__(self,
                 invariance_lambda=25,
                 variance_mu=25,
                 covariance_v=1,
                 eps=1e-4,
                 ):
        super().__init__()
        self.invariance_lambda = invariance_lambda
        self.variance_mu = variance_mu
        self.covariance_v = covariance_v
        self.eps = eps

    def forward(self, z, l):
        pos_ix = torch.triu(l.view(-1, 1) == l.view(1, -1), diagonal=1).nonzero(as_tuple=True)
        s = torch.nn.functional.pairwise_distance(z[pos_ix[0]], z[pos_ix[1]]).pow(2).mean()

        v = torch.relu(self.variance_mu - (z.var(dim=1) + self.eps).pow(0.5)).mean()

        c = (z - z.mean(dim=0, keepdim=True)) / (z.var(dim=0, keepdim=True) + self.eps)
        c = self.off_diag(torch.mm(c.T, c).div_(c.size(0) // 2)).pow(2).mean()

        return self.invariance_lambda * s + self.variance_mu * v + self.covariance_v * c

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ChannelDropout(torch.nn.Module):
    def __init__(self, p, with_upscale):
        super().__init__()

        self.p = p
        self.with_upscale = with_upscale

    def forward(self, x: PaddedBatch):
        if not self.training:
            return x

        B, T, H = x.payload.size()
        mask = torch.bernoulli(torch.ones(B, 1, H, device=x.payload.device) * (1 - self.p))
        t = x.payload * mask
        if self.with_upscale:
            t = t * H / mask.sum(dim=2, keepdim=True)
        return PaddedBatch(t, x.seq_lens)


class FTT(torch.nn.Module):
    """FT-Transformer

    """

    def __init__(self, input_size, hidden_size, nhead, num_layers, norm, dim_feedforward, c_dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_trans_w = torch.nn.Parameter(torch.randn(1, input_size, hidden_size), requires_grad=True)
        self.num_trans_b = torch.nn.Parameter(torch.randn(1, input_size, hidden_size), requires_grad=True)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.transf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm() if norm else None,
        )
        self.c_dropout = c_dropout

    def forward(self, x):
        B, H = x.size()
        x_in = x.unsqueeze(2) * self.num_trans_w + self.num_trans_b  # B, H(T), F
        if self.training and self.c_dropout > 0:
            mask = torch.bernoulli(torch.ones(H, device=x.device) * (1 - self.c_dropout)).bool()
            x_in = x_in[:, mask]
        x_in = torch.cat([self.cls.expand(B, 1, self.hidden_size), x_in], dim=1)

        x_out = self.transf(x_in)
        cls_out = x_out[:, 0, :]
        return cls_out


class CutModule(torch.nn.Module):
    def __init__(self, ix):
        super().__init__()

        self.ix = ix

    def forward(self, x):
        return x[:, :self.ix]


class MarginLoss:
    def __init__(self, margin):
        self.margin = margin

    def sampling(self, x, labels):
        pos_ix = labels.nonzero(as_tuple=True)
        exp_x = x[pos_ix[0]]
        exp_pos = x[pos_ix[0], pos_ix[1]].unsqueeze(1).expand_as(exp_x)

        loss_ix = ((exp_x + self.margin - exp_pos) * (1 - labels[pos_ix[0]]) > 0).nonzero(as_tuple=True)

        row_id = pos_ix[0][loss_ix[0]]
        pos_ix = pos_ix[1][loss_ix[0]]
        neg_ix = loss_ix[1]
        return row_id, pos_ix, neg_ix

    def get_loss(self, pos, neg):
        loss = torch.relu(neg + self.margin - pos)
        return loss.mean()


class PairedModule(pl.LightningModule):
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
            m,
            NormEncoder(),
        )
        m = create_encoder(params['click_seq'], is_reduce_sequence=True)
        self.seq_encoder_click_size = m.embedding_size
        self.seq_encoder_click = torch.nn.Sequential(
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
        (x_trx, l_trx), (x_click, l_click) = batch

        self.log('seq_len/trx_min', x_trx.seq_lens.float().min())
        self.log('seq_len/trx_max', x_trx.seq_lens.float().max())
        self.log('seq_len/trx_mean', x_trx.seq_lens.float().mean())
        self.log('seq_len/click_min', x_click.seq_lens.float().min())
        self.log('seq_len/click_max', x_click.seq_lens.float().max())
        self.log('seq_len/click_mean', x_click.seq_lens.float().mean())

        z_trx = self.seq_encoder_trx(x_trx)  # B, H
        z_click = self.seq_encoder_click(x_click)  # B, H

        loss = self.loss_fn(
            torch.cat([z_trx, z_click], dim=0),
            torch.cat([l_trx, l_click], dim=0),
        )
        self.log('loss/loss_ml', loss, prog_bar=True)

        with torch.no_grad():
            out = -(z_trx.unsqueeze(1) - z_click.unsqueeze(0)).pow(2).sum(dim=2)
            n_samples = z_trx.size(0) // (l_trx.max().item() + 1)
            for i in range(n_samples):
                l2 = out[i::n_samples, i::n_samples]
                self.train_precision(l2)
                self.train_mrr(l2)

        return loss

    def training_epoch_end(self, _):
        self.log('train_metrics/precision', self.train_precision, prog_bar=True)
        self.log('train_metrics/mrr', self.train_mrr, prog_bar=False)


class L2Scorer(torch.nn.Module):
    def forward(self, x):
        B, H = x.size()
        a, b =x[:, :H // 2], x[:, H // 2:]
        return -(a - b).pow(2).sum(dim=1)