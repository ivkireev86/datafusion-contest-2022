import logging

import numpy as np
import pandas as pd
from zipfile import ZipFile

from sklearn.model_selection import StratifiedKFold

from vtb_code.utils import logging_init

N_SPLITS = 6
CNT_BIN_COUNT = 4
SHUFFLE_RANDOM_STATE = 42

logger = logging.getLogger(__name__)


def main():
    logger.info('Data loading...')
    df_train_matching = pd.read_csv('data/train_matching.csv')

    with ZipFile('data/transactions.zip') as z:
        df_transactions = pd.read_csv(z.open('transactions.csv'))
    with ZipFile('data/clickstream.zip') as z:
        df_clickstream = pd.read_csv(z.open('clickstream.csv'))

    logger.info(f'Loaded {len(df_train_matching)} pairs, '
                f'{len(df_transactions)} transactions, {len(df_clickstream)} clicks')

    vc = df_transactions['user_id'].value_counts()
    s_trx_cnt_bins = pd.cut(
        vc,
        vc.quantile(np.linspace(0, 1, CNT_BIN_COUNT + 1)),
        labels=np.arange(CNT_BIN_COUNT),
    ).fillna(0).astype(str).rename('trx_cnt_bins')
    vc = df_clickstream['user_id'].value_counts()
    s_click_cnt_bins = pd.cut(
        vc,
        vc.quantile(np.linspace(0, 1, CNT_BIN_COUNT + 1)),
        labels=np.arange(CNT_BIN_COUNT),
    ).fillna(0).astype(str).rename('click_cnt_bins')
    logger.info(f'Prepared {CNT_BIN_COUNT} bins for trx and clicks')

    df_train_matching = pd.merge(df_train_matching, s_trx_cnt_bins, left_on='bank', right_index=True, how='left')
    df_train_matching = pd.merge(df_train_matching, s_click_cnt_bins, left_on='rtk', right_index=True, how='left')
    df_train_matching['trx_cnt_bins'] = df_train_matching['trx_cnt_bins'].fillna(str(CNT_BIN_COUNT))
    df_train_matching['click_cnt_bins'] = df_train_matching['click_cnt_bins'].fillna(str(CNT_BIN_COUNT))
    df_train_matching['cnt_bins'] = df_train_matching['trx_cnt_bins'] + df_train_matching['click_cnt_bins']
    df_train_matching = df_train_matching[['bank', 'rtk', 'cnt_bins']]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SHUFFLE_RANDOM_STATE)
    for fold_id, (_, ix_folds) in enumerate(skf.split(df_train_matching, df_train_matching['cnt_bins'])):
        df_match = df_train_matching[['bank', 'rtk']].iloc[ix_folds]
        df_match.to_csv(f'data/train_matching_{fold_id}.csv', index=False)

        df_trx = df_transactions[lambda x: x['user_id'].isin(df_match['bank'].values)]
        df_trx.to_csv(f'data/transactions_{fold_id}.csv', index=False)

        df_click = df_clickstream[lambda x: x['user_id'].isin(df_match['rtk'].values)]
        df_click.to_csv(f'data/clickstream_{fold_id}.csv', index=False)

        logger.info(f'Saved data for fold {fold_id}: {len(df_match)} pairs, '
                    f'{len(df_trx)} transactions, {len(df_click)} clicks')

    logger.info(f'All splits saved')


if __name__ == '__main__':
    logging_init(handlers=[logging.StreamHandler(), logging.FileHandler('results/split_folds.log')])
    main()
