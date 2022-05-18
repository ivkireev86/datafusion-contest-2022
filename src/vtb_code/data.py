import numpy as np
import torch

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.data_module.coles_data_module import coles_collate_fn


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

    @staticmethod
    def collate_fn(batch):
        return [coles_collate_fn(c) for c in zip(*batch)]


class PairedFullDataset(torch.utils.data.Dataset):
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
