import pandas as pd
import torch
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip

CATEGORY_MAX_SIZE_TRX = {
    'mcc_code': 350,
    'currency_rk': 5,
}
CATEGORY_MAX_SIZE_CLICK = {
    'cat_id': 400,
    'level_0': 400,
    'level_1': 400,
    'level_2': 400,
}


def trx_types(df):
    df['mcc_code'] = df['mcc_code'].astype(str)
    df['currency_rk'] = df['currency_rk'].astype(str)
    df['event_time'] = pd.to_datetime(df['transaction_dttm']).astype(int) / 1e9
    return df[['user_id', 'event_time', 'mcc_code', 'currency_rk', 'transaction_amt']]


def click_types(df):
    df['event_time'] = pd.to_datetime(df['timestamp']).astype(int) / 1e9
    df = pd.merge(df, pd.read_csv('click_categories.csv'), on='cat_id')
    df['cat_id'] = df['cat_id'].astype(str)
    return df[['user_id', 'event_time', 'cat_id', 'level_0', 'level_1', 'level_2', 'new_uid']]

def trx_to_torch(seq):
    seq = CategorySizeClip(CATEGORY_MAX_SIZE_TRX)(seq)
    for x in seq:
        yield x['user_id'], {
            'event_time': torch.from_numpy(x['event_time']).float(),
            'mcc_code': torch.from_numpy(x['mcc_code']).int(),
            'currency_rk': torch.from_numpy(x['currency_rk']).int(),
            'transaction_amt': torch.from_numpy(x['transaction_amt']).float(),
        }


def click_to_torch(seq):
    seq = CategorySizeClip(CATEGORY_MAX_SIZE_CLICK)(seq)
    for x in seq:
        yield x['user_id'], {
            'event_time': torch.from_numpy(x['event_time']).float(),
            'cat_id': torch.from_numpy(x['cat_id']).int(),
            'level_0': torch.from_numpy(x['level_0']).int(),
            'level_1': torch.from_numpy(x['level_1']).int(),
            'level_2': torch.from_numpy(x['level_2']).int(),
            'new_uid': torch.from_numpy(x['new_uid']).int(),
        }
