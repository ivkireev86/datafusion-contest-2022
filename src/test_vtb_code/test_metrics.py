import unittest.mock

import numpy as np
import pandas as pd
import torch
from vtb_code.metrics import PrecisionK, MeanReciprocalRankK, ValidationCallback


def get_data1():
    return (
        torch.tensor([
            [3, 0, 1, 2],
            [0, 3, 1, 2],
            [2, 1, 0, 3],
            [0, 1, 2, 3],
        ]),
        torch.tensor([
            [3, 0],
            [0, 3],
        ]),
    )


def get_data2():
    return (
        torch.tensor([
            [2, 0, 1, 3],
            [0, 2, 1, 3],
            [2, 0, 1, 3],
            [0, 1, 3, 2],
        ]),
        torch.tensor([
            [3, 0],
            [3, 0],
        ]),
    )


def test_metrics_1():
    mp = PrecisionK(k=3, compute_on_step=False)
    mm = MeanReciprocalRankK(k=3, max_k=3, compute_on_step=False)
    for logits in get_data1():
        mp(logits)
        mm(logits)
    torch.testing.assert_close(mp.compute(), torch.tensor(5 / 6))
    torch.testing.assert_close(mm.compute(), torch.tensor(5 / 6))


def test_metrics_2():
    mp = PrecisionK(k=3, compute_on_step=False)
    mm = MeanReciprocalRankK(k=3, max_k=3, compute_on_step=False)
    for logits in get_data2():
        mp(logits)
        mm(logits)
    torch.testing.assert_close(mp.compute(), torch.tensor(1.0))
    torch.testing.assert_close(mm.compute(), torch.tensor((0.5 * 4 + 1 + 1 / 3) / 6))


def test_metrics_3():
    mp = PrecisionK(k=64, compute_on_step=False)
    logits  = torch.randn(64, 64)
    mp(logits)
    torch.testing.assert_close(mp.compute(), torch.tensor(1.0))


def test_validation_callback_logits_to_metrics():
    z_out = torch.tensor([
        [5, 1, 2, 3, 4],  # 0 0 1 0 0
        [1, 5, 2, 3, 4],  # 1 0 0 0 0
        [1, 2, 5, 3, 4],  # 0 0 0 1 0
        [1, 2, 3, 5, 4],  # 0 0 0 0 1
    ]).float()
    vc = ValidationCallback(
        v_trx=unittest.mock.Mock(),
        v_click=unittest.mock.Mock(),
        target=pd.DataFrame({
            'bank': ['a1', 'a2', 'a3', 'a4'],
            'rtk': ['b3', 'b1', 'b4', 'b5'],
        }),
        device='cpu', device_main='cpu',
        k=3,
    )
    vc.v_trx.dataset.pairs = np.array(['a1', 'a2', 'a3', 'a4']).reshape(-1, 1)
    vc.v_click.dataset.pairs = np.array(['b1', 'b2', 'b3', 'b4', 'b5']).reshape(-1, 1)

    precision, mrr, r1 = vc.logits_to_metrics(z_out)

    np.testing.assert_allclose(precision, 1 / 2)
    np.testing.assert_allclose(mrr, 5 / 24)
    np.testing.assert_allclose(r1, 5 / 17)


def test_validation_callback_with_samples():
    z_out = torch.tensor([
        [2, 2, 2, 2, 2, 2, 2, 2],  # 0 0   0 0   1 1   0 0
        [2, 2, 2, 2, 2, 2, 2, 2],  # 0 0   0 0   1 1   0 0
        [2, 2, 2, 2, 2, 2, 2, 2],  # 1 1   0 0   0 0   0 0
        [2, 2, 2, 2, 2, 2, 2, 2],  # 1 1   0 0   0 0   0 0
        [2, 2, 2, 2, 2, 2, 2, 2],  # 0 0   0 0   0 0   1 1
        [2, 2, 2, 2, 2, 2, 2, 2],  # 0 0   0 0   0 0   1 1
    ]).float()
    vc = ValidationCallback(
        v_trx=unittest.mock.Mock(),
        v_click=unittest.mock.Mock(),
        target=pd.DataFrame({
            'bank': ['a1', 'a2', 'a3'],
            'rtk': ['b3', 'b1', 'b4'],
        }),
        device='cpu', device_main='cpu',
        k=3,
    )
    vc.v_trx.dataset.pairs = np.array(['a1', 'a1', 'a2', 'a2', 'a3', 'a3']).reshape(-1, 1)
    vc.v_click.dataset.pairs = np.array(['b1', 'b1', 'b2', 'b2', 'b3', 'b3', 'b4', 'b4']).reshape(-1, 1)

    precision, mrr, r1 = vc.logits_to_metrics(z_out)

    np.testing.assert_allclose(precision, 1 / 2)
    np.testing.assert_allclose(mrr, 5 / 24)
    np.testing.assert_allclose(r1, 5 / 17)
