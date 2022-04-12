import gc
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from dltranz.data_load import augmentation_chain
from dltranz.data_load.augmentations.seq_len_limit import SeqLenLimit

from model import MLMPretrainModuleTrx, MLMPretrainModuleClick, PairedModule
from vtb_code.data import PairedDataset, paired_collate_fn, DropDuplicate
from vtb_code.preprocessing import trx_types, click_types, trx_to_torch, click_to_torch


def main():
    data, output_path = sys.argv[1:]

    print(f'Loading...')
    os.system(f'ls -l {data}/*')

    df_trx = pd.read_csv(f'{data}/transactions.csv')
    df_click = pd.read_csv(f'{data}/clickstream.csv')
    print(f'Loaded csv files from "{data}". {len(df_trx)} transactions, {len(df_click)} clicks')

    df_trx = trx_types(df_trx)
    df_click = click_types(df_click)
    print(f'Loaded csv files from "{data}". {len(df_trx)} transactions, {len(df_click)} clicks')

    with open('preprocessor_trx.p', 'rb') as f:
        preprocessor_trx = pickle.load(f)
    with open('preprocessor_click.p', 'rb') as f:
        preprocessor_click = pickle.load(f)

    print(f'Loaded preprocessor files')

    features_trx = dict(trx_to_torch(preprocessor_trx.transform(df_trx)))
    print(f'Trx features prepared: {len(features_trx)} users')
    features_click = dict(click_to_torch(preprocessor_click.transform(df_click)))
    print(f'Click features prepared: {len(features_click)} users')

    uid_banks = np.sort(df_trx['user_id'].unique())
    uid_rtk = np.sort(df_click['user_id'].unique())
    print(f'uid_banks: {uid_banks.shape}, uid_rtk: {uid_rtk.shape}')

    del df_trx
    del df_click
    gc.collect()

    valid_dl_trx = torch.utils.data.DataLoader(
        PairedDataset(
            uid_banks.reshape(-1, 1),
            data=[
                features_trx,
            ],
            augmentations=[
                augmentation_chain(DropDuplicate('mcc_code', col_new_cnt='c_cnt'), SeqLenLimit(2000)),  # 2000
            ],
            n_sample=1,
        ),
        collate_fn=paired_collate_fn,
        shuffle=False,
        num_workers=4,
        batch_size=512,
        persistent_workers=True,
    )

    valid_dl_click = torch.utils.data.DataLoader(
        PairedDataset(
            uid_rtk.reshape(-1, 1),
            data=[
                features_click,
            ],
            augmentations=[
                augmentation_chain(DropDuplicate('cat_id', col_new_cnt='c_cnt'), SeqLenLimit(5000)),  # 5000
            ],
            n_sample=1,
        ),
        collate_fn=paired_collate_fn,
        shuffle=False,
        num_workers=4,
        batch_size=512,
        persistent_workers=True,
    )

    mlm_model_trx = MLMPretrainModuleTrx.load_from_checkpoint('pretrain_trx.cpt')
    mlm_model_click = MLMPretrainModuleClick.load_from_checkpoint('pretrain_click.cpt')
    pl_module = PairedModule.load_from_checkpoint('nn_distance_coles_model.cpt',
                                                  mlm_model_trx=mlm_model_trx,
                                                  mlm_model_click=mlm_model_click,)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pl_module.to(device)
    pl_module.eval()

    print('Scoring...')
    with torch.no_grad():
        z_trx = []
        for ((x_trx, _),) in valid_dl_trx:
            z_trx.append(pl_module.seq_encoder_trx(x_trx.to(device)))
        z_trx = torch.cat(z_trx, dim=0)
        print('Trx embeddings done')
        z_click = []
        for ((x_click, _),) in valid_dl_click:
            z_click.append(pl_module.seq_encoder_click(x_click.to(device)))
        z_click = torch.cat(z_click, dim=0)
        print('Click embeddings done')

        T = z_trx.size(0)
        C = z_click.size(0)
        ix_t = torch.arange(T, device=device).view(-1, 1).expand(T, C).flatten()
        ix_c = torch.arange(C, device=device).view(1, -1).expand(T, C).flatten()

        z_out = []
        batch_size = 1024
        for i in range(0, len(ix_t), batch_size):
            z_pairs = torch.cat([
                z_trx[ix_t[i:i + batch_size]],
                z_click[ix_c[i:i + batch_size]],
            ], dim=1)
            z_out.append(pl_module.cls(z_pairs).unsqueeze(1))
        z_out = torch.cat(z_out, dim=0).view(T, C)
        z_out = torch.cat([
            torch.zeros((T, 1), device=device),
            z_out,
        ], dim=1)
        print('Cross scores done')

        uid_rtk = np.concatenate([[0], uid_rtk])

        k = min(100, z_out.size(1))
        # k = 100
        z_out = torch.topk(z_out, k=k, dim=1, largest=True, sorted=True).indices
        z_out = z_out.cpu()

    submission_final = []
    for i, l in zip(uid_banks, uid_rtk[z_out]):
        submission_final.append([i, l])

    submission_final = np.array(submission_final, dtype=object)
    print(submission_final.shape)
    np.savez(output_path, submission_final)


if __name__ == "__main__":
    main()
