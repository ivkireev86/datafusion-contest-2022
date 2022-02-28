import numpy as np
import torch
from torchmetrics import Metric
import pytorch_lightning as pl


class PrecisionK(Metric):
    def __init__(self, k, **params):
        super().__init__(**params)

        self.add_state('_sum', torch.tensor(0))
        self.add_state('_cnt', torch.tensor(0))
        self.k = k

    def update(self, preds, target=None):
        B, _ = preds.size()
        ix_sort = torch.argsort(preds, dim=1, descending=True)
        ix_sort = ix_sort == torch.arange(B, device=preds.device, dtype=torch.long).view(-1, 1)
        k = min(self.k, B)
        ix_sort = (ix_sort[:, :k].int().sum(dim=1) > 0).int().sum()
        self._sum = self._sum + ix_sort
        self._cnt = self._cnt + B

    def compute(self):
        return self._sum.float() / self._cnt.float()


class MeanReciprocalRankK(Metric):
    def __init__(self, k, max_k=100, **params):
        super().__init__(**params)

        self.add_state('_sum', torch.tensor(0))
        self.add_state('_cnt', torch.tensor(0))
        self.k = k
        self.max_k = max_k

    def update(self, preds, target=None):
        B, _ = preds.size()
        ix_sort = torch.argsort(preds, dim=1, descending=True)
        ix_sort = ix_sort == torch.arange(B, device=preds.device, dtype=torch.long).view(-1, 1)
        k = min(self.k, B)
        ix_sort = ix_sort[:, :k]
        ranks = self.k / self.max_k / (1 + torch.arange(k, device=preds.device).view(1, -1).expand(B, k))
        ranks = ranks[ix_sort]

        self._sum = self._sum + ranks.sum()
        self._cnt = self._cnt + B

    def compute(self):
        return self._sum.float() / self._cnt.float()


class ValidationCallback(pl.Callback):
    def __init__(self, v_trx, v_click, target, device, device_main, k=100, batch_size=1024):
        self.v_trx = v_trx
        self.v_click = v_click
        self.target = target
        self.device = device
        self.device_main = device_main
        self.k = k
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        was_traning = False
        if pl_module.training:
            pl_module.eval()
            was_traning = True

        pl_module.to(self.device)
        with torch.no_grad():
            z_trx = []
            for ((x_trx, _),) in self.v_trx:
                z_trx.append(torch.nn.functional.normalize(
                    pl_module.seq_encoder_trx(x_trx.to(self.device)), dim=1))
            z_trx = torch.cat(z_trx, dim=0)
            z_click = []
            for ((x_click, _),) in self.v_click:
                z_click.append(torch.nn.functional.normalize(
                    pl_module.seq_encoder_click(x_click.to(self.device)), dim=1))
            z_click = torch.cat(z_click, dim=0)

            T = z_trx.size(0)
            C = z_click.size(0)
            device = z_trx.device
            ix_t = torch.arange(T, device=device).view(-1, 1).expand(T, C).flatten()
            ix_c = torch.arange(C, device=device).view(1, -1).expand(T, C).flatten()

            z_out = []
            for i in range(0, len(ix_t), self.batch_size):
                z_pairs = torch.cat([
                    z_trx[ix_t[i:i + self.batch_size]],
                    z_click[ix_c[i:i + self.batch_size]],
                ], dim=1)
                z_out.append(pl_module.cls(z_pairs).unsqueeze(1))
            z_out = torch.cat(z_out, dim=0).view(T, C)

            precision, mrr, r1 = self.logits_to_metrics(z_out)

            pl_module.log('valid_full_metrics/precision', precision, prog_bar=True)
            pl_module.log('valid_full_metrics/mrr', mrr, prog_bar=False)
            pl_module.log('valid_full_metrics/r1', r1, prog_bar=False)

        pl_module.to(self.device_main)
        if was_traning:
            pl_module.train()

    def logits_to_metrics(self, z_out):
        T, C = z_out.size()
        z_ranks = torch.zeros_like(z_out)
        z_ranks[
            torch.arange(T, device=self.device).view(-1, 1).expand(T, C),
            torch.argsort(z_out, dim=1, descending=True),
        ] = torch.arange(C, device=self.device).float().view(1, -1).expand(T, C) + 1
        true_ranks = z_ranks[
            np.arange(T),
            np.searchsorted(self.v_click.dataset.pairs[:, 0],
                            self.target.set_index('bank')['rtk'].loc[self.v_trx.dataset.pairs[:, 0]].values)
        ]
        precision = torch.where(true_ranks <= self.k,
                                torch.ones(1, device=self.device), torch.zeros(1, device=self.device)).mean()
        mrr = torch.where(true_ranks <= self.k, 1 / true_ranks, torch.zeros(1, device=self.device)).mean()
        r1 = 2 * mrr * precision / (mrr + precision)
        return precision, mrr, r1
