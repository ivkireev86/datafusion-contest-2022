import numpy as np
import pandas as pd
import sys


def main():
    valid_fold_id = int(sys.argv[1])

    df_valid_match = pd.read_csv(f'../data/train_matching_{valid_fold_id}.csv').set_index('bank')['rtk']
    with np.load('submission_final.npz', allow_pickle=True) as f:
        submission_final = f['arr_0']

    mrr = 0
    pre = 0
    for bank_uid, top_rtk_list in submission_final:
        rtk = df_valid_match.loc[bank_uid]
        if rtk == '0':
            rtk = 0
        ix = np.where(top_rtk_list == rtk)[0]
        if len(ix) > 0:
            pre += 1
            mrr += 1 / (ix[0] + 1)
    cnt = len(submission_final)
    mrr /= cnt
    pre /= cnt
    r1 = 2 * mrr * pre / (mrr + pre)

    print(f'Fold {valid_fold_id} results: r1={r1:.4f}, mrr={mrr:.4f}, pre={pre:4f}. Scored {cnt} lines')


if __name__ == '__main__':
    main()
