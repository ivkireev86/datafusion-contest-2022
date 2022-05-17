from functools import reduce
from operator import imul

import numpy as np
import torch

from dltranz.data_load import padded_collate_wo_target
from dltranz.data_load.data_module.coles_data_module import coles_collate_fn
from dltranz.metric_learn.dataset.split_strategy import AbsSplit


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, data, augmentations, n_sample):
        super().__init__()

        self.pairs = pairs
        self.data = data
        self.augmentations = augmentations
        self.n_sample = n_sample

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        ids = self.pairs[item]
        return tuple([[a(d[i]) for _ in range(self.n_sample)]
                      for i, d, a in zip(ids, self.data, self.augmentations)])


class PairedZeroDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, data, augmentations, n_sample):
        super().__init__()

        self.pairs = pairs
        self.data = data
        self.augmentations = augmentations
        self.n_sample = n_sample
        self.full_pairs = self.get_full_pairs()

    @staticmethod
    def collate_fn(batch):
        """
        In:
        [1, 2 ,3], [4, 5]
        [6, 7], [None, None]
        [None, None], [8, 9]

        Out:
        x, labels, out_of_match

        PaddedBatch([1, 2, 3, 6, 7]), [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]
        PaddedBatch([4, 5, 8, 9]), [0, 0, 2, 2], [0, 0, 1, 1]
        """
        data_1 = [t for p, _ in batch for t in p if t is not None]
        data_2 = [t for _, p in batch for t in p if t is not None]

        labels = torch.arange(len(batch), dtype=torch.int32)
        labels_1 = torch.repeat_interleave(labels,
                                           torch.tensor([len([t for t in p if t is not None]) for p, _ in batch]))
        labels_2 = torch.repeat_interleave(labels,
                                           torch.tensor([len([t for t in p if t is not None]) for _, p in batch]))
        out_of_match_1 = torch.tensor([1 if p2[0] is None else 0 for p1, p2 in batch for t in p1 if t is not None])
        out_of_match_2 = torch.tensor([1 if p1[0] is None else 0 for p1, p2 in batch for t in p2 if t is not None])
        return (
            padded_collate_wo_target(data_1), labels_1, out_of_match_1.int(),
            padded_collate_wo_target(data_2), labels_2, out_of_match_2.int(),
        )

    @staticmethod
    def not_in(v, items_to_exclude):
        a = np.sort(items_to_exclude)
        return v[np.pad(a, pad_width=(0, 1), constant_values='')[np.searchsorted(a, v)] != v]

    def get_full_pairs(self):
        free_trx = self.not_in(np.array(list(self.data[0].keys())), self.pairs[:, 0]).reshape(-1, 1)
        free_clicks = self.not_in(np.array(list(self.data[1].keys())), self.pairs[:, 1]).reshape(-1, 1)

        return np.concatenate([
            self.pairs,
            np.concatenate([free_trx, np.full((len(free_trx), 1), '0')], axis=1),
            np.concatenate([np.full((len(free_clicks), 1), '0'), free_clicks], axis=1),
        ], axis=0)

    def __len__(self):
        return len(self.full_pairs)

    def __getitem__(self, item):
        ids = self.full_pairs[item]
        return tuple([[a(d[i]) if i != '0' else None for _ in range(self.n_sample)]
                      for i, d, a in zip(ids, self.data, self.augmentations)])


class PairedNegativeDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, data, augmentations, neg_rate, is_shuffle=True):
        super().__init__()

        self.pairs = pairs
        self.data = data
        self.augmentations = augmentations
        self.neg_rate = neg_rate
        self.is_shuffle = is_shuffle

        self.full_pairs = self.get_full_pairs()
        self.sample_ixs, self.labels = self.get_sample_ixs()

    @staticmethod
    def not_in(v, items_to_exclude):
        a = np.sort(items_to_exclude)
        return v[np.pad(a, pad_width=(0, 1), constant_values='')[np.searchsorted(a, v)] != v]

    def get_full_pairs(self):
        free_trx = self.not_in(np.array(list(self.data[0].keys())), self.pairs[:, 0]).reshape(-1, 1)
        free_clicks = self.not_in(np.array(list(self.data[1].keys())), self.pairs[:, 1]).reshape(-1, 1)

        return np.concatenate([
            self.pairs,
            np.concatenate([free_trx, np.full((len(free_trx), 1), '0')], axis=1),
            np.concatenate([np.full((len(free_clicks), 1), '0'), free_clicks], axis=1),
        ], axis=0)

    def get_sample_ixs(self):
        n = len(self.full_pairs)
        full_labels = np.arange(n)
        pos_ix = full_labels[(self.full_pairs != '0').all(axis=1)].reshape(-1, 1).repeat(2, axis=1)

        labels0 = full_labels[self.full_pairs[:, 0] != '0']
        labels1 = full_labels[self.full_pairs[:, 1] != '0']
        neg_ix = (labels0.reshape(-1, 1) != labels1.reshape(1, -1)).nonzero()
        neg_ix = [l[i] for l, i in zip([labels0, labels1], neg_ix)]
        neg_ix = np.stack(neg_ix, axis=1)
        n_neg = int(len(pos_ix) * self.neg_rate)
        if len(neg_ix) > n_neg:
            sampled_neg_ix = np.random.choice(len(neg_ix), size=n_neg, replace=False)
            neg_ix = neg_ix[sampled_neg_ix]

        ixs = np.concatenate([pos_ix, neg_ix])
        labels = torch.cat([torch.ones(len(pos_ix)), torch.zeros(len(neg_ix))])
        if self.is_shuffle:
            _i = np.arange(len(ixs))
            np.random.shuffle(_i)
            ixs = ixs[_i]
            labels = labels[_i]
        return ixs, labels

    def __len__(self):
        return len(self.sample_ixs)

    def __getitem__(self, item):
        ixs = self.sample_ixs[item]
        ids = self.full_pairs[ixs[0], 0], self.pairs[ixs[1], 1]
        return tuple([a(d[i])
                      for i, d, a in zip(ids, self.data, self.augmentations)]), self.labels[item]

    @ staticmethod
    def collate_fn(batch):
        s_trx = [i for (i, _), _ in batch]
        s_click = [i for (_, i), _ in batch]
        labels = [i for (_, _), i in batch]
        return (padded_collate_wo_target(s_trx), padded_collate_wo_target(s_click)), torch.tensor(labels)


class InferenceSplittingDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data, augmentations, splitter: AbsSplit):
        super().__init__()

        self.ids = ids
        self.data = data
        self.augmentations = augmentations
        self.splitter = splitter

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        _id = self.ids[item]
        rec = self.data[_id]
        dt_ixs = self.splitter.split(rec['event_time'])
        return [self.augmentations({k: v[ix] for k, v in rec.items()}) for ix in dt_ixs],


class CrossDataset(torch.utils.data.Dataset):
    def __init__(self, unique_ids, data, augmentations):
        super().__init__()

        self.unique_ids = unique_ids
        self.lens = [len(d) for d in self.unique_ids]
        self.data = data
        self.augmentations = augmentations

    def __len__(self):
        return reduce(imul, [len(d) for d in self.unique_ids])

    def ix_flat_to_multi_dimension(self, item):
        ixs = []
        for p in self.lens[::-1]:
            ixs.append(item % p)
            item = item // p
        return ixs[::-1]

    def __getitem__(self, item):
        ids = self.ix_flat_to_multi_dimension(item)
        return tuple([a(d[u[i]]) for i, u, d, a in zip(ids, self.unique_ids, self.data, self.augmentations)])


def paired_collate_fn(batch):
    return [coles_collate_fn(c) for c in zip(*batch)]


def paired_negative_collate_fn(batch):
    s_trx = [i for (i, _), _ in batch]
    s_click = [i for (_, i), _ in batch]
    labels = [i for (_, _), i in batch]
    return (padded_collate_wo_target(s_trx), padded_collate_wo_target(s_click)), torch.tensor(labels)


def paired_collate_fn_flat(batch):
    return [padded_collate_wo_target(c) for c in zip(*batch)]


class DropDuplicate:
    def __init__(self, col_check, col_new_cnt=None, keep='first'):
        super().__init__()

        self.col_check = col_check
        self.col_new_cnt = col_new_cnt
        if keep != 'first':
            raise NotImplementedError()

    def __call__(self, x):
        idx, new_cnt = self.get_idx(x[self.col_check])
        new_x = {k: v[idx] for k, v in x.items()}
        if self.col_new_cnt is not None:
            new_x[self.col_new_cnt] = torch.from_numpy(new_cnt)
        return new_x

    def get_idx(self, x):
        diff = np.diff(x, prepend=x[0] - 1)
        new_ix = np.where(diff != 0)[0]
        new_cnt = np.diff(new_ix, append=len(x))
        return new_ix, new_cnt


def frequency_encoder(x, mask=None, pad_value=0):
    a = x if mask is None else x[mask]
    u, c = torch.unique(a, sorted=True, return_counts=True)
    ix = torch.cat([torch.argsort(c, descending=True).long() + 1, torch.zeros(1, dtype=torch.long, device=x.device)])
    if mask is None:
        return ix[torch.searchsorted(u, x)]
    return torch.where(mask, ix[torch.searchsorted(u, x)], torch.full_like(x, fill_value=pad_value, dtype=torch.long))


class WeekTicks:
    def __init__(self, step_days):
        super().__init__()

        self.step_days = step_days
        self.step_sec = step_days * 24 * 60 * 60

    def __call__(self, x):
        et = x['event_time']
        t_from = torch.ceil(et.min() / self.step_sec) * self.step_sec
        t_to = torch.floor(et.max() / self.step_sec) * self.step_sec + 1
        if t_to <= t_from:
            new_x = {k: v for k, v in x.items()}
            new_x['ticks'] = torch.zeros(len(et)).long()
            return new_x

        tick_times = torch.arange(t_from, t_to, self.step_sec)
        tick_zeros = torch.zeros_like(tick_times)
        new_x = {k: torch.cat([v, tick_times if k == 'event_time' else tick_zeros.to(dtype=v.dtype)])
                 for k, v in x.items()}
        new_x['ticks'] = torch.cat([torch.zeros(len(et)), torch.ones(len(tick_times))]).long()

        et = new_x['event_time']
        ix = torch.argsort(et)
        new_x = {k: v[ix] for k, v in new_x.items()}

        return new_x


class PairedNegDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, data, augmentations, neg_count):
        super().__init__()

        self.pairs = pairs
        self.data = data
        self.augmentations = augmentations
        self.neg_count = neg_count

        self.all_clicks = np.sort(np.array(list(self.data[1].keys())))

    @staticmethod
    def collate_fn(batch):
        return [coles_collate_fn(c) for c in zip(*batch)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        trx_id, click_id = self.pairs[item]
        c_ids = np.random.choice([i for i in self.all_clicks if i != click_id], size=self.neg_count - 1, replace=False)
        return (
            [self.augmentations[0](self.data[0][trx_id])],
            [self.augmentations[1](self.data[1][click_id])] + [self.augmentations[1](self.data[1][c]) for c in c_ids]
        )
