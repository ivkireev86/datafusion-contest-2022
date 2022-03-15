import pickle
from glob import glob
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pyhocon import ConfigFactory
from dltranz.data_load import augmentation_chain
from dltranz.data_load.augmentations.random_slice import RandomSlice

import gc

from dltranz.data_preprocessing.pandas_preprocessor import PandasDataPreprocessor

from vtb_code.data import PairedZeroDataset, DropDuplicate

from model import PairedColesModule
from vtb_code.preprocessing import trx_types, click_types, trx_to_torch, click_to_torch


def get_model(batch_size):
    sup_model = PairedColesModule(
        ConfigFactory.parse_string('''
        head_size: 128
        trx_seq: {
            trx_encoder: {
              use_batch_norm_with_lens: false
              norm_embeddings: false,
              embeddings_noise: 0.000,
              embeddings: {
                mcc_code: {in: 350, out: 64},
                currency_rk: {in: 10, out: 4}
                hour: {in: 30, out: 16}
              },
              numeric_values: {
                transaction_amt: identity
                c_cnt: log
              }
              was_logified: false
              log_scale_factor: 1.0
            },
            encoder_type: rnn,
            rnn: {
              type: gru,
              hidden_size: 128,
              bidir: false,
              trainable_starter: static
            }
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
              },
              numeric_values: {
                  c_cnt: log
              }
              was_logified: false
              log_scale_factor: 1.0
            },
            encoder_type: rnn,
            rnn: {
              type: gru,
              hidden_size: 128,
              bidir: false,
              trainable_starter: static
            }    
        }
    '''),
        sampling_strategy_params=ConfigFactory.parse_string('''
            train.sampling_strategy: SemiHardTriplets
            # train.balanced: true
            # train.neg_count: 10
        '''),
        loss_params=ConfigFactory.parse_string('''
            train.loss: TripletLoss
            # train.num_steps: 50
            train.margin: 0.5
        '''),
        k=100 * batch_size // 3000,
        lr=0.004, weight_decay=0,
        step_size=7, gamma=0.4,
        base_lr=0.0005, max_lr=0.004, step_size_up=300, step_size_down=900,
    )
    return sup_model


def main():
    folds_count = len(glob('../data/train_matching_*.csv'))
    print(f'folds_count = {folds_count}')

    df_matching_train = pd.concat([pd.read_csv(f'../data/train_matching_{i}.csv') for i in range(folds_count)])
    df_trx_train = pd.concat([trx_types(pd.read_csv(f'../data/transactions_{i}.csv')) for i in range(folds_count)])
    df_click_train = pd.concat([click_types(pd.read_csv(f'../data/clickstream_{i}.csv')) for i in range(folds_count)])
    print(f'Loaded csv files')

    preprocessor_trx = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='event_time',
        time_transformation='none',
        cols_category=["mcc_code", "currency_rk"],
        cols_log_norm=["transaction_amt"],
        print_dataset_info=False,
    )
    preprocessor_click = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='event_time',
        time_transformation='none',
        cols_category=['cat_id', 'level_0', 'level_1', 'level_2'],
        cols_log_norm=[],
        print_dataset_info=False,
    )

    features_trx_train = dict(trx_to_torch(preprocessor_trx.fit_transform(df_trx_train)))
    print(f'Trx features prepared')
    features_click_train = dict(click_to_torch(preprocessor_click.fit_transform(df_click_train)))
    print(f'Click features prepared')

    del df_trx_train
    del df_click_train
    gc.collect()

    batch_size = 128
    train_dl = torch.utils.data.DataLoader(
        PairedZeroDataset(
            np.concatenate([
                df_matching_train[lambda x: x['rtk'].ne('0')].values,
            ], axis=1),
            data=[
                features_trx_train,
                features_click_train,
            ],
            augmentations=[
                augmentation_chain(DropDuplicate('mcc_code', col_new_cnt='c_cnt'), RandomSlice(32, 1024)),  # 2000
                augmentation_chain(DropDuplicate('cat_id', col_new_cnt='c_cnt'), RandomSlice(64, 2048)),  # 5000
            ],
            n_sample=2,
        ),
        collate_fn=PairedZeroDataset.collate_fn,
        shuffle=True,
        num_workers=12,
        batch_size=batch_size,
        persistent_workers=True,
    )

    sup_model = get_model(batch_size)

    trainer = pl.Trainer(
        gpus=1,
        max_steps=4000,
    )
    trainer.fit(sup_model, train_dl)

    trainer.save_checkpoint('nn_distance_coles_model.cpt', weights_only=True)
    with open('preprocessor_trx.p', 'wb') as f:
        pickle.dump(preprocessor_trx, f)
    with open('preprocessor_click.p', 'wb') as f:
        pickle.dump(preprocessor_click, f)


if __name__ == '__main__':
    main()
