import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc


def click_types(df):
    df['cat_id'] = df['cat_id'].astype(str)
    return df[['user_id', 'timestamp', 'cat_id']]


def click_pivot(df):
    clickstream_embed = df.pivot_table(index = 'user_id',
                            values=['timestamp'],
                            columns=['cat_id'],
                            aggfunc=['count']).fillna(0)
    clickstream_embed.columns = [f'{str(i[0])}-{str(i[2])}' for i in clickstream_embed.columns]
    clickstream_embed.loc['0'] = np.empty(len(clickstream_embed.columns))

    dtype_clickstream = list()
    for x in clickstream_embed.dtypes.tolist():
        if x=='int64':
            dtype_clickstream.append('int16')
        elif(x=='float64'):
            dtype_clickstream.append('float32')
        else:
            dtype_clickstream.append('object')

    dtype_clickstream = dict(zip(clickstream_embed.columns.tolist(), dtype_clickstream))
    clickstream_embed = clickstream_embed.astype(dtype_clickstream)
    clickstream_embed.reset_index(drop=False, inplace=True)
    return clickstream_embed


def trx_types(df):
    df['mcc_code'] = df['mcc_code'].astype(str)
    df['currency_rk'] = df['currency_rk'].astype(str)
    df['event_time'] = pd.to_datetime(df['transaction_dttm']).astype(int) / 1e9
    return df[['user_id', 'event_time', 'mcc_code', 'currency_rk', 'transaction_amt']]


def trx_pivot(df):
    bankclient_embed = df.pivot_table(index='user_id',
                                      values=['transaction_amt'],
                                      columns=['mcc_code'],
                                      aggfunc=['sum', 'mean', 'count']).fillna(0)
    bankclient_embed.columns = [f'{str(i[0])}-{str(i[2])}' for i in bankclient_embed.columns]

    dtype_bankclient = list()
    for x in bankclient_embed.dtypes.tolist():
        if x == 'int64':
            dtype_bankclient.append('int16')
        elif (x == 'float64'):
            dtype_bankclient.append('float32')
        else:
            dtype_bankclient.append('object')

    dtype_bankclient = dict(zip(bankclient_embed.columns.tolist(), dtype_bankclient))
    bankclient_embed = bankclient_embed.astype(dtype_bankclient)
    bankclient_embed.reset_index(drop=False, inplace=True)
    return bankclient_embed


def inference(df_trx, df_click, model, model_features, batch_size=100):

    list_of_rtk = list(df_click.index.unique())
    list_of_bank = list(df_trx.index.unique())

    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)

    num_of_batches = int((len(list_of_bank))/batch_size)+1
    submission_ready = []

    for i in range(num_of_batches):
        bank_ids = list_of_bank[(i*batch_size):((i+1)*batch_size)]
        if len(bank_ids) != 0:
            part_of_submit = submission[submission['bank'].isin(bank_ids)].explode('rtk')
            part_of_submit = part_of_submit.merge(df_trx, how='left', left_on='bank', right_index=True
                                        ).merge(df_click, how='left', left_on='rtk', right_index=True).fillna(0)

            for feature in model_features:
                if feature not in part_of_submit.columns:
                    part_of_submit[feature] = 0

            part_of_submit['predicts'] = model.predict(part_of_submit[model_features])
            part_of_submit = part_of_submit[['bank', 'rtk', 'predicts']]

            part_of_submit = part_of_submit.sort_values(by=['bank', 'predicts'], ascending=False).reset_index(drop=True)
            part_of_submit = part_of_submit.pivot_table(index='bank', values='rtk', aggfunc=list)
            part_of_submit['rtk'] = part_of_submit['rtk'].apply(lambda x: x[:100])
            part_of_submit['bank'] = part_of_submit.index
            part_of_submit = part_of_submit[['bank', 'rtk']]
            submission_ready.extend(part_of_submit.values)

    submission_final = np.array(submission_ready, dtype=object)
    return submission_final


def main():
    data, output_path = sys.argv[1:]

    df_trx = pd.read_csv(f'{data}/transactions.csv')
    df_trx = trx_types(df_trx)
    df_trx = trx_pivot(df_trx)
    df_trx.set_index('user_id', inplace=True)

    df_click = pd.read_csv(f'{data}/clickstream.csv')
    df_click = click_types(df_click)
    df_click = click_pivot(df_click)
    df_click.set_index('user_id', inplace=True)

    with open('lgb_model.p', 'rb') as f:
        lgb_model = pickle.load(f)

    submission_final = inference(df_trx, df_click, lgb_model, lgb_model.feature_name())

    print(submission_final.shape)
    np.savez(output_path, submission_final)


if __name__ == "__main__":
    main()
