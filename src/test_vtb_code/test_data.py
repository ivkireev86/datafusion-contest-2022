import numpy as np
import torch
from vtb_code.data import PairedDataset, PairedFullDataset


def get_data():
    pairs = np.array([
        ['A1', 'B1'],
        ['A2', 'B3'],
        ['A3', 'B2'],
        ['A4', '0'],
        ['A5', '0'],
    ])
    ds1 = {
        'A1': 1,
        'A2': 2,
        'A3': 3,
        'A4': 4,
        'A5': 5,
        'A6': 6,
    }
    ds2 = {
        'B1': 10,
        'B2': 20,
        'B3': 30,
        'B4': 40,
     }
    return pairs, ds1, ds2


def get_no_augmentation(n=2):
    return [lambda x: x] * n


def test_paired_dataset():
    pairs, ds1, ds2 = get_data()
    data = PairedDataset(pairs[np.all(pairs != '0', axis=1)], [ds1, ds2], get_no_augmentation(), n_sample=1)
    assert len(data) == 3
    exp = {
        0: ([1], [10]),
        1: ([2], [30]),
        2: ([3], [20]),
    }
    for i in range(len(data)):
        assert data[i] == exp[i]


def test_paired_zero_dataset():
    pairs, ds1, ds2 = get_data()
    data = PairedFullDataset(pairs[np.all(pairs != '0', axis=1)], [ds1, ds2], get_no_augmentation(), n_sample=1)
    assert len(data) == 7
    exp = {
        0: ([1], [10]),
        1: ([2], [30]),
        2: ([3], [20]),
        3: ([4], [None]),
        4: ([5], [None]),
        5: ([6], [None]),
        6: ([None], [40]),
    }
    for i in range(len(data)):
        assert data[i] == exp[i]


def test_paired_zero_dataset_collate_fn():
    batch = [
        ([{'col1': torch.arange(4)}, {'col1': torch.arange(3)}], [{'feat1': torch.arange(2)}]),
        ([{'col1': torch.arange(3)}], [None]),
        ([None], [{'feat1': torch.arange(6)}, {'feat1': torch.arange(2)}]),
    ]
    out = PairedFullDataset.collate_fn(batch)
    torch.testing.assert_close(out[1], torch.tensor([0, 0, 1]).int())
    torch.testing.assert_close(out[4], torch.tensor([0, 2, 2]).int())
    torch.testing.assert_close(out[2], torch.tensor([0, 0, 1]).int())
    torch.testing.assert_close(out[5], torch.tensor([0, 1, 1]).int())


def test_paired_dataset_collate_fn():
    batch = [
        ([{'col1': torch.arange(4)}], [{'feat1': torch.arange(2)}]),
        ([{'col1': torch.arange(3)}], [{'feat1': torch.arange(3)}]),
        ([{'col1': torch.arange(5)}], [{'feat1': torch.arange(6)}]),
    ]
    out = PairedDataset.collate_fn(batch)
    exp0p = torch.tensor([
        [0, 1, 2, 3, 0],
        [0, 1, 2, 0, 0],
        [0, 1, 2, 3, 4],
    ]).long()
    exp0s = torch.tensor([4, 3, 5]).int()
    exp1p = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0],
        [0, 1, 2, 3, 4, 5],
    ]).long()
    exp1s = torch.tensor([2, 3, 6]).int()

    torch.testing.assert_close(out[0][0].payload['col1'], exp0p)
    torch.testing.assert_close(out[0][0].seq_lens, exp0s)
    torch.testing.assert_close(out[0][1], torch.arange(3))
    torch.testing.assert_close(out[1][0].payload['feat1'], exp1p)
    torch.testing.assert_close(out[1][0].seq_lens, exp1s)
    torch.testing.assert_close(out[1][1], torch.arange(3))
