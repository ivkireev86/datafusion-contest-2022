import torch

from dltranz.trx_encoder import PaddedBatch
from vtb_code.models import ChannelDropout, MarginLoss
from vtb_code.models import SemiHardTripletDualSelector


def test_channel_dropout():
    x = PaddedBatch(torch.randn(3, 5, 4), None)
    out = ChannelDropout(0.1, True)(x)
    print(x.payload, out.payload)


def test_margin_loss_1():
    x = torch.tensor([
        [2, 3, 1, 2, 4],
        [1, 2, 0, 3, 4],
        [1, 2, 3, 4, 5],

    ])
    labels = torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ])
    m = MarginLoss(0.5)
    row_id, pos_ix, neg_ix = m.sampling(x, labels)
    torch.testing.assert_close(row_id, torch.tensor([0, 0, 0, 1, 1, 1, 1]))
    torch.testing.assert_close(pos_ix, torch.tensor([0, 0, 1, 2, 2, 2, 2]))
    torch.testing.assert_close(neg_ix, torch.tensor([3, 4, 4, 0, 1, 3, 4]))


def test_margin_loss_2():
    x = torch.tensor([
        [0, 1, 2, 3, 4],
        [2, 3, 0, 4, 1],
    ])
    labels = torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
    ])
    m = MarginLoss(0.5)
    row_id, pos_ix, neg_ix = m.sampling(x, labels)
    torch.testing.assert_close(row_id, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    torch.testing.assert_close(pos_ix, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4]))
    torch.testing.assert_close(neg_ix, torch.tensor([2, 3, 4, 2, 3, 4, 0, 1, 3, 0, 1, 3]))


def test_semi_hard_triplet_dual_selector():
    embeddings = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 0, 0.9],
        [0, 0, 1, 0],
        [0, 0, 0.9, 0],
        [0, 1, 0, 0],
        [0, 0.9, 0, 0],
        [0, 0, 0, 1.1],
        [0, 0, 0, 0.9],
        [0, 0, 1.1, 0],
        [0, 0, 0.9, 0],
    ])
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
    out = SemiHardTripletDualSelector().get_triplets(embeddings, labels)
    exp_pos = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [6, 7, 6, 7, 8, 9, 8, 9],
    ]).T
    exp_neg = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
    ]).T
    torch.testing.assert_close(out[:, :2], exp_pos)
    print(out[:, [2]])
