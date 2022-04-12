import numpy as np
import torch
from vtb_code.data import PairedDataset, PairedZeroDataset, PairedNegativeDataset, CrossDataset, paired_collate_fn
from vtb_code.data import frequency_encoder


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
    data = PairedZeroDataset(pairs[np.all(pairs != '0', axis=1)], [ds1, ds2], get_no_augmentation(), n_sample=1)
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
    out = PairedZeroDataset.collate_fn(batch)
    torch.testing.assert_close(out[1], torch.tensor([0, 0, 1]).int())
    torch.testing.assert_close(out[4], torch.tensor([0, 2, 2]).int())
    torch.testing.assert_close(out[2], torch.tensor([0, 0, 1]).int())
    torch.testing.assert_close(out[5], torch.tensor([0, 1, 1]).int())


def test_cross_dataset_ids():
    unique_ids = [
        np.arange(4),
        np.arange(5),
        np.arange(3),
    ]
    data = CrossDataset(unique_ids, [], [])
    assert len(data) == 60

    exp = {
        n: t for n, t in enumerate([
            [i, j, k]
            for i in range(4)
            for j in range(5)
            for k in range(3)
        ])
    }
    for i in range(len(data)):
        assert data.ix_flat_to_multi_dimension(i) == exp[i]


def test_cross_dataset():
    _, ds1, ds2 = get_data()
    unique_ids = [np.unique(list(df.keys())) for df in [ds1, ds2]]
    data = CrossDataset(unique_ids, [ds1, ds2], get_no_augmentation())
    assert len(data) == 24

    out = np.array([data[i] for i in range(len(data))])
    out = out.sum(axis=1).reshape(6, 4)

    exp = np.array([
        [11, 21, 31, 41],
        [12, 22, 32, 42],
        [13, 23, 33, 43],
        [14, 24, 34, 44],
        [15, 25, 35, 45],
        [16, 26, 36, 46],
    ])
    np.testing.assert_equal(out, exp)


def test_paired_collate_fn():
    batch = [
        ([{'col1': torch.arange(4)}], [{'feat1': torch.arange(2)}]),
        ([{'col1': torch.arange(3)}], [{'feat1': torch.arange(3)}]),
        ([{'col1': torch.arange(5)}], [{'feat1': torch.arange(6)}]),
    ]
    out = paired_collate_fn(batch)
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


def test_paired_negative_dataset():
    ds = PairedNegativeDataset(
        pairs=np.array([
            ['a', 'b', 'c'],
            ['A', 'B', 'C'],
        ]).T,
        data=[
            {'a': 0, 'b': 1, 'c': 2, 'd': 3},
            {'A': 5, 'B': 6, 'C': 7, 'D': 8, 'E': 9},
        ],
        augmentations=get_no_augmentation(),
        neg_rate=(4 * 5 - 3) / 3,
        is_shuffle=False,
    )
    np.testing.assert_equal(
        ds.full_pairs,
        np.array([
            ['a', 'b', 'c', 'd', '0', '0'],
            ['A', 'B', 'C', '0', 'D', 'E'],
        ]).T
    )
    np.testing.assert_equal(
        ds.sample_ixs,
        np.array([
            [0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [0, 1, 2, 1, 2, 4, 5, 0, 2, 4, 5, 0, 1, 4, 5, 0, 1, 2, 4, 5],
        ]).T
        # np.array([
        #     ['a', 'b', 'c', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd'],
        #     ['A', 'B', 'C', 'B', 'C', 'D', 'E', 'A', 'C', 'D', 'E', 'A', 'B', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
        # ]).T
    )
    np.testing.assert_equal(
        ds.labels,
        np.array([1] * 3 + [0] * 17)
    )


def test_frequency_encoder():
    x = torch.tensor([11, 13, 11, 10, 12, 10, 11, 11, 10, 12])
    y = frequency_encoder(x)
    exp = torch.tensor([1, 4, 1, 2, 3, 2, 1, 1, 2, 3])
    torch.testing.assert_close(y, exp)


def test_frequency_encoder_with_pad():
    x = torch.tensor([11, 13, 11, 10, 12, 10, 11, 11, 10, 12, 0, 0, 0])
    y = frequency_encoder(x, x != 0)
    exp = torch.tensor([1, 4, 1, 2, 3, 2, 1, 1, 2, 3, 0, 0, 0])
    torch.testing.assert_close(y, exp)


def test_frequency_encoder_with_mask():
    x = torch.tensor([11, 13, 11, 10, 12, 10, 11, 11, 10, 12, 100, 100, 100])
    y = frequency_encoder(x, x != 100)
    exp = torch.tensor([1, 4, 1, 2, 3, 2, 1, 1, 2, 3, 0, 0, 0])
    torch.testing.assert_close(y, exp)
