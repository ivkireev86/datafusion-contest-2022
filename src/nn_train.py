import gc
import pickle
import random
import sys
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ptls.data_load import augmentation_chain
from ptls.data_load.augmentations.random_slice import RandomSlice
from ptls.data_preprocessing.pandas_preprocessor import PandasDataPreprocessor
from pyhocon import ConfigFactory

from vtb_code.model import MLMPretrainModuleTrx, MLMPretrainModuleClick, PairedModule
from vtb_code.data import PairedDataset, PairedFullDataset, DropDuplicate
from vtb_code.preprocessing import trx_types, click_types, trx_to_torch, click_to_torch


ENSEMBLE_SIZE = 11
CONFIG_MLM = ConfigFactory.parse_string('''
    common_trx_size: 256
    transf: {
        nhead: 4
        dim_feedforward: 1024
        dropout: 0.1
        num_layers: 3
        norm: false
        max_len: 6000
        use_pe: true
    }
    mlm: {
        replace_proba: 0.1
        neg_count_all: 64
        neg_count_self: 8
        beta: 10
    }
    trx_seq: {
        trx_encoder: {
          use_batch_norm_with_lens: false
          norm_embeddings: false,
          embeddings_noise: 0.000,
          embeddings: {
            mcc_code: {in: 350, out: 64},
            currency_rk: {in: 10, out: 4}
            transaction_amt_q: {in: 110, out: 8}

            hour: {in: 30, out: 16}
            weekday: {in: 10, out: 4}
            day_diff: {in: 15, out: 8}
          },
          numeric_values: {
            transaction_amt: identity
            c_cnt: log
          }
          was_logified: false
          log_scale_factor: 1.0
        },
    }
    click_seq: {
        trx_encoder: {
          use_batch_norm_with_lens: false
          norm_embeddings: false,
          embeddings_noise: 0.000,
          embeddings: {
            cat_id: {in: 400, out: 64},
            level_0: {in: 400, out: 16}
            level_1: {in: 400, out: 8}
            level_2: {in: 400, out: 4}

            hour: {in: 30, out: 16}
            weekday: {in: 10, out: 4}
            day_diff: {in: 15, out: 8}
          },
          numeric_values: {
            c_cnt: log
          }
          was_logified: false
          log_scale_factor: 1.0
        },
    }
''')
CONFIG_QSM = ConfigFactory.parse_string('''
        common_trx_size: 128
        rnn: {
          type: gru,
          hidden_size: 256,
          bidir: false,
          trainable_starter: static
        }
    ''')


def load_data(valid_fold_id):
    folds_count = len(glob('../data/train_matching_*.csv'))
    train_folds = [i for i in range(folds_count) if valid_fold_id is not None and i != valid_fold_id]
    print(f'Total folds_count = {folds_count}, used {len(train_folds)}')
    print(f'Loading...')

    df_matching_train = pd.concat([pd.read_csv(f'../data/train_matching_{i}.csv') for i in train_folds])
    df_trx_train = pd.concat([trx_types(pd.read_csv(f'../data/transactions_{i}.csv')) for i in train_folds])
    df_click_train = pd.concat([click_types(pd.read_csv(f'../data/clickstream_{i}.csv')) for i in train_folds])
    print(f'Loaded csv files')
    preprocessor_trx = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='event_time',
        time_transformation='none',
        cols_category=["mcc_code", "currency_rk"],
        cols_log_norm=["transaction_amt"],
        cols_identity=[],
        print_dataset_info=False,
    )
    preprocessor_click = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='event_time',
        time_transformation='none',
        cols_category=['cat_id', 'level_0', 'level_1', 'level_2'],
        cols_log_norm=[],
        cols_identity=['new_uid'],
        print_dataset_info=False,
    )
    features_trx_train = dict(trx_to_torch(preprocessor_trx.fit_transform(df_trx_train)))
    print(f'Trx features prepared')
    features_click_train = dict(click_to_torch(preprocessor_click.fit_transform(df_click_train)))
    print(f'Click features prepared')
    del df_trx_train
    del df_click_train
    gc.collect()
    # trainer.save_checkpoint('nn_distance_coles_model.cpt', weights_only=True)
    with open('preprocessor_trx.p', 'wb') as f:
        pickle.dump(preprocessor_trx, f)
    with open('preprocessor_click.p', 'wb') as f:
        pickle.dump(preprocessor_click, f)
    return df_matching_train, features_click_train, features_trx_train


def pretrain_mlm_trx(features_trx_train):
    train_dl_mlm_trx = torch.utils.data.DataLoader(
        PairedDataset(
            np.sort(np.array(list(features_trx_train.keys()))).reshape(-1, 1),
            data=[features_trx_train],
            augmentations=[augmentation_chain(
                DropDuplicate('mcc_code', col_new_cnt='c_cnt'),
                RandomSlice(32, 128)
            )],
            n_sample=1,
        ),
        collate_fn=PairedDataset.collate_fn,
        shuffle=False,
        num_workers=12,
        batch_size=128,
        persistent_workers=True,
    )

    # calculate trx_amnt_quantiles
    v = []
    for batch in train_dl_mlm_trx:
        v.append(batch[0][0].payload['transaction_amt'][batch[0][0].seq_len_mask.bool()])
    v = torch.cat(v)
    trx_amnt_quantiles = torch.quantile(torch.unique(v), torch.linspace(0, 1, 100))

    mlm_model_trx = MLMPretrainModuleTrx(
        params=CONFIG_MLM,
        lr=0.001, weight_decay=0,
        max_lr=0.001, pct_start=9000 / 2 / 10000, total_steps=10000,
        trx_amnt_quantiles=trx_amnt_quantiles,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_steps=8000,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=2000, save_top_k=-1,
            ),
        ]
    )
    model_version_trx = trainer.logger.version
    print('Trx pretrain start')
    print('baseline all:  {:.3f}'.format(np.log(mlm_model_trx.hparams.params['mlm.neg_count_all'] + 1)))
    print('baseline self: {:.3f}'.format(np.log(mlm_model_trx.hparams.params['mlm.neg_count_self'] + 1)))
    print(f'version = {model_version_trx}')
    trainer.fit(mlm_model_trx, train_dl_mlm_trx)
    trainer.save_checkpoint('pretrain_trx.cpt', weights_only=True)
    print('Trx pretrain done')


def pretrain_mlm_click(features_click_train):
    train_dl_mlm_click = torch.utils.data.DataLoader(
        PairedDataset(
            np.sort(np.array(list(features_click_train.keys()))).reshape(-1, 1),
            data=[features_click_train],
            augmentations=[augmentation_chain(
                DropDuplicate('cat_id', col_new_cnt='c_cnt'),
                RandomSlice(32, 128)
            )],
            n_sample=1,
        ),
        collate_fn=PairedDataset.collate_fn,
        shuffle=False,
        num_workers=12,
        batch_size=128,
        persistent_workers=True,
    )

    mlm_model_click = MLMPretrainModuleClick(
        params=CONFIG_MLM,
        lr=0.001, weight_decay=0,
        max_lr=0.001, pct_start=9000 / 2 / 10000, total_steps=10000,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_steps=6000,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=2000, save_top_k=-1,
            ),
        ]
    )
    model_version_click = trainer.logger.version
    print('Click pretrain start')
    print('baseline all:  {:.3f}'.format(np.log(mlm_model_click.hparams.params['mlm.neg_count_all'] + 1)))
    print('baseline self: {:.3f}'.format(np.log(mlm_model_click.hparams.params['mlm.neg_count_self'] + 1)))
    print(f'version = {model_version_click}')
    trainer.fit(mlm_model_click, train_dl_mlm_click)
    trainer.save_checkpoint('pretrain_click.cpt', weights_only=True)
    print('Click pretrain done')


def train_qsm(df_matching_train, features_trx_train, features_click_train, model_n):
    batch_size = 128
    train_dl = torch.utils.data.DataLoader(
        PairedFullDataset(
            df_matching_train[lambda x: x['rtk'].ne('0')].values,
            data=[
                features_trx_train,
                features_click_train,
            ],
            augmentations=[
                augmentation_chain(DropDuplicate('mcc_code', col_new_cnt='c_cnt'), RandomSlice(32, 1024)),  # 1024
                augmentation_chain(DropDuplicate('cat_id', col_new_cnt='c_cnt'), RandomSlice(64, 2048)),  # 2048
            ],
            n_sample=2,
        ),
        collate_fn=PairedFullDataset.collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=24,
        batch_size=batch_size,
        persistent_workers=True,
    )

    mlm_model_trx = MLMPretrainModuleTrx.load_from_checkpoint('pretrain_trx.cpt')
    mlm_model_click = MLMPretrainModuleClick.load_from_checkpoint('pretrain_click.cpt')
    pl.seed_everything(random.randint(1, 2**16 - 1))
    sup_model = PairedModule(
        CONFIG_QSM,
        k=100 * batch_size // 3000,
        lr=0.0022, weight_decay=0,
        max_lr=0.0018, pct_start=1100 / 6000, total_steps=6000,
        beta=0.2 / 1.4, neg_count=120,
        mlm_model_trx=mlm_model_trx, mlm_model_click=mlm_model_click,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_steps=3300,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=1000, save_top_k=-1,
            ),
        ]
    )
    print('Train qsm start')
    trainer.fit(sup_model, train_dl)
    trainer.save_checkpoint(f'nn_distance_coles_model_{model_n}.cpt', weights_only=True)
    print(f'Train qsm [{model_n}] done')


def main():
    valid_fold_id = int(sys.argv[1]) if len(sys.argv) == 2 else None
    df_matching_train, features_click_train, features_trx_train = load_data(valid_fold_id)

    pretrain_mlm_trx(features_trx_train)
    pretrain_mlm_click(features_click_train)
    for i in range(ENSEMBLE_SIZE):
        train_qsm(df_matching_train, features_trx_train, features_click_train, i)


if __name__ == '__main__':
    main()
