from functools import reduce
from operator import imul

import numpy as np
import torch

from dltranz.data_load import padded_collate_wo_target
from dltranz.data_load.data_module.coles_data_module import coles_collate_fn


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
