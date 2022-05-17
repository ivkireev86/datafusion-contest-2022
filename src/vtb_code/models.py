import pytorch_lightning as pl
import torch
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.seq_encoder import create_encoder
from dltranz.seq_encoder.utils import NormEncoder
from dltranz.trx_encoder import PaddedBatch
from dltranz.metric_learn.sampling_strategies import PairSelector, TripletSelector
from dltranz.metric_learn.metric import outer_pairwise_distance

from src.vtb_code.metrics import PrecisionK, MeanReciprocalRankK
import torchmetrics

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


class ResNetBlock(torch.nn.Module):
    def __init__(self, size, internal_size=1024, dropout=0.1):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.BatchNorm1d(size),
            torch.nn.Linear(size, internal_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(internal_size, size),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


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


class SemiHardTripletDualSelector(TripletSelector):
    """
        Generate triplets with semihard sampling strategy

        "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
        https://arxiv.org/abs/1503.03832
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count
        assert neg_count == 1

    def get_triplets(self, embeddings, labels):
        ix_split = (torch.diff(labels) < 0).nonzero().flatten() + 1
        assert len(ix_split) == 1

        embeddings0, embeddings1 = embeddings[:ix_split], embeddings[ix_split:]
        labels0, labels1 = labels[:ix_split], labels[ix_split:]

        n0, n1 = labels0.size(0), labels1.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels0.view(n0, 1).expand(n0, n1) - labels1.view(1, n1).expand(n0, n1)

        positive_pairs = (x == 0).int().nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings0[positive_pairs[:, 0]].detach()
        anchor_labels = labels0[positive_pairs[:, 0]]

        pos_embed = embeddings1[positive_pairs[:, 1]].detach()

        D_ap = torch.nn.functional.pairwise_distance(anchor_embed, pos_embed)

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = anchor_labels.view(m, 1).expand(m, n1) == labels1.view(1, n1).expand(m, n1)

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings1.detach())  # pairwise_distance anchors x all

        neg_mat_distances = mat_distances * (x == 0).type(mat_distances.dtype)  # filter: get only negative pairs

        # negatives_outside: smallest D_an where D_an > D_ap.
        upper_bound = 3
        negatives_outside = (upper_bound - neg_mat_distances) * \
                            (neg_mat_distances > D_ap.view(m, 1).expand(m, n1)).type(neg_mat_distances.dtype)
        values, negatives_outside = negatives_outside.topk(k=1, dim=1, largest=True)

        # negatives_inside: largest D_an
        values, negatives_inside = neg_mat_distances.topk(k=1, dim=1, largest=True)

        # whether exist negative n, such that D_an > D_ap.
        semihard_exist = ((neg_mat_distances > D_ap.view(m, 1).expand(m, n1)).sum(dim=1) > 0).view(-1, 1)

        negatives_indeces = torch.where(semihard_exist, negatives_outside, negatives_inside)

        triplets = torch.cat([positive_pairs, negatives_indeces], dim=1)

        # shift 2nd index
        triplets = triplets + torch.tensor([0, 1, 1]).to(device=embeddings.device).view(1, 3) * len(labels0)

        return triplets


class HardTripletDualSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and the neg_count hardest negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        ix_split = (torch.diff(labels) < 0).nonzero().flatten() + 1
        assert len(ix_split) == 1

        embeddings0, embeddings1 = embeddings[:ix_split], embeddings[ix_split:]
        labels0, labels1 = labels[:ix_split], labels[ix_split:]

        n0, n1 = labels0.size(0), labels1.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels0.view(n0, 1).expand(n0, n1) - labels1.view(1, n1).expand(n0, n1)

        positive_pairs = (x == 0).int().nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings0[positive_pairs[:, 0]].detach()
        anchor_labels = labels0[positive_pairs[:, 0]]

        # pos_embed = embeddings[positive_pairs[:,0]].detach()

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = anchor_labels.view(m, 1).expand(m, n1) == labels1.view(1, n1).expand(m, n1)

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings1.detach())  # pairwise_distance anchors x all

        upper_bound = 3
        mat_distances = ((upper_bound - mat_distances) * (x == 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        values, indices = mat_distances.topk(k=self.neg_count, dim=1, largest=True)

        triplets = torch.cat([
            positive_pairs.repeat(self.neg_count, 1),
            torch.cat(indices.unbind(dim=0)).view(-1, 1)
        ], dim=1)

        # shift 2nd index
        triplets = triplets + torch.tensor([0, 1, 1]).to(device=embeddings.device).view(1, 3) * len(labels0)

        return triplets


class RandomNegativeTripletDualSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and random negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        ix_split = (torch.diff(labels) < 0).nonzero().flatten() + 1
        assert len(ix_split) == 1

        embeddings0, embeddings1 = embeddings[:ix_split], embeddings[ix_split:]
        labels0, labels1 = labels[:ix_split], labels[ix_split:]

        n0, n1 = labels0.size(0), labels1.size(0)

        # construct matrix x, such as x_ij == 1 <==> labels[i] == labels[j]
        x = labels0.view(n0, 1).expand(n0, n1) - labels1.view(1, n1).expand(n0, n1)

        positive_pairs = (x == 0).int().nonzero(as_tuple=False)

        m = positive_pairs.size(0)
        anchor_labels = labels0[positive_pairs[:, 0]]

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = anchor_labels.view(m, 1).expand(m, n1) == labels1.view(1, n1).expand(m, n1)

        negative_pairs = (x == 0).type(embeddings.dtype)
        negative_pairs_prob = (negative_pairs.t() / negative_pairs.sum(dim=1)).t()
        negative_pairs_rand = torch.multinomial(negative_pairs_prob, 1)

        triplets = torch.cat([positive_pairs, negative_pairs_rand], dim=1)

        triplets = triplets + torch.tensor([0, 1, 1]).to(device=embeddings.device).view(1, 3) * len(labels0)

        return triplets


class RandomDistanceNegativeTripletDualSelector(TripletSelector):
    """
        Generate triplets with all positive pairs and random negative example for each anchor
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        ix_split = (torch.diff(labels) < 0).nonzero().flatten() + 1
        assert len(ix_split) == 1

        embeddings0, embeddings1 = embeddings[:ix_split], embeddings[ix_split:]
        labels0, labels1 = labels[:ix_split], labels[ix_split:]

        n0, n1 = labels0.size(0), labels1.size(0)

        # construct matrix x, such as x_ij == 1 <==> labels[i] == labels[j]
        x = labels0.view(n0, 1).expand(n0, n1) - labels1.view(1, n1).expand(n0, n1)

        positive_pairs = (x == 0).int().nonzero(as_tuple=False)

        m = positive_pairs.size(0)
        anchor_embed = embeddings0[positive_pairs[:, 0]].detach()
        anchor_labels = labels0[positive_pairs[:, 0]]
        pos_embed = embeddings1[positive_pairs[:, 1]].detach()

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = anchor_labels.view(m, 1).expand(m, n1) == labels1.view(1, n1).expand(m, n1)

        D_ap = torch.nn.functional.pairwise_distance(anchor_embed, pos_embed)

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings1.detach())  # pairwise_distance anchors x all
        neg_mat_mask = ((mat_distances < D_ap.view(m, 1).expand(m, n1)) & (x == 0)).float()

        negative_pairs_rand = torch.multinomial(neg_mat_mask.float() + 1e-4, 1)

        triplets = torch.cat([positive_pairs, negative_pairs_rand], dim=1)

        triplets = triplets + torch.tensor([0, 1, 1]).to(device=embeddings.device).view(1, 3) * len(labels0)

        return triplets


class SemiHardTripletSelector2(TripletSelector):
    """
        Generate triplets with semihard sampling strategy

        "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
        https://arxiv.org/abs/1503.03832
    """

    def __init__(self, neg_count=1):
        super(SemiHardTripletSelector2, self).__init__()
        self.neg_count = neg_count

    def get_triplets(self, embeddings, labels):
        n = labels.size(0)

        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        x = labels.expand(n, n) - labels.expand(n, n).t()

        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        m = positive_pairs.size(0)

        anchor_embed = embeddings[positive_pairs[:, 0]].detach()
        anchor_labels = labels[positive_pairs[:, 0]]

        pos_embed = embeddings[positive_pairs[:, 1]].detach()

        D_ap = torch.nn.functional.pairwise_distance(anchor_embed, pos_embed)

        # construct matrix x (size m x n), such as x_ij == 1 <==> anchor_labels[i] == labels[j]
        x = (labels.expand(m, n) == anchor_labels.expand(n, m).t())

        mat_distances = outer_pairwise_distance(anchor_embed, embeddings.detach())  # pairwise_distance anchors x all

        neg_mat_distances = mat_distances * (x == 0).type(mat_distances.dtype)  # filter: get only negative pairs

        # negatives_outside: smallest D_an where D_an > D_ap.
        upper_bound = int((2 * n) ** 0.5) + 1
        negatives_outside = neg_mat_distances * \
                            (neg_mat_distances < D_ap.expand(n, m).t()).type(neg_mat_distances.dtype)
        values, negatives_outside = negatives_outside.topk(k=1, dim=1, largest=True)

        # negatives_inside: largest D_an
        values, negatives_inside = neg_mat_distances.topk(k=1, dim=1, largest=False)

        # whether exist negative n, such that D_an > D_ap.
        semihard_exist = ((neg_mat_distances < D_ap.expand(n, m).t()).sum(dim=1) > 0).view(-1, 1)

        negatives_indeces = torch.where(semihard_exist, negatives_outside, negatives_inside)

        triplets = torch.cat([positive_pairs, negatives_indeces], dim=1)

        return triplets


class UniformNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """

    def __init__(self, step=16):
        super().__init__()
        self.step = step

    def get_pairs(self, embeddings, labels):
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = labels.size(0)
        x = labels.expand(n, n) - labels.expand(n, n).t()

        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        # hard negative minning
        mat_distances = outer_pairwise_distance(embeddings.detach())  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype))  # filter: get only negative pairs

        neg_ix1 = torch.argsort(mat_distances, dim=1)[:, ::self.step]
        neg_ix0 = torch.arange(neg_ix1.size(0), device=embeddings.device).view(-1, 1).expand(*neg_ix1.size())
        neg_ix = torch.stack([neg_ix0.flatten(), neg_ix1.flatten()], dim=1)
        selected_neg_dist = mat_distances[neg_ix[:, 0], neg_ix[:, 1]]
        negative_pairs = neg_ix[selected_neg_dist > 0.0]

        return positive_pairs, negative_pairs


class MeanLoss(torchmetrics.Metric):
    def __init__(self, **params):
        super().__init__(**params)

        self.add_state('_sum', torch.tensor([0.0]))
        self.add_state('_cnt', torch.tensor([0]))

    def update(self, x):
        self._sum += x.sum()
        self._cnt += x.numel()

    def compute(self):
        return self._sum / self._cnt.float()