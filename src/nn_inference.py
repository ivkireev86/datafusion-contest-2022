import gc
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
from ptls.data_load import augmentation_chain
from ptls.data_load.augmentations.seq_len_limit import SeqLenLimit

from vtb_code.data import PairedDataset, DropDuplicate
from vtb_code.model import MLMPretrainModuleTrx, MLMPretrainModuleClick, PairedModule
from vtb_code.preprocessing import trx_types, click_types, trx_to_torch, click_to_torch


def load_data(valid_fold_id):
    print(f'Loading...')
    df_trx = pd.read_csv(f'../data/transactions_{valid_fold_id}.csv')
    df_click = pd.read_csv(f'../data/clickstream_{valid_fold_id}.csv')
    print(f'Loaded csv files. {len(df_trx)} transactions, {len(df_click)} clicks')
    df_trx = trx_types(df_trx)
    df_click = click_types(df_click)
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
        collate_fn=PairedDataset.collate_fn,
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
        collate_fn=PairedDataset.collate_fn,
        shuffle=False,
        num_workers=4,
        batch_size=512,
        persistent_workers=True,
    )
    return uid_banks, uid_rtk, valid_dl_trx, valid_dl_click


def get_pairvise_distance_with_model(model_path, valid_dl_trx, valid_dl_click):
    mlm_model_trx = MLMPretrainModuleTrx.load_from_checkpoint('pretrain_trx.cpt')
    mlm_model_click = MLMPretrainModuleClick.load_from_checkpoint('pretrain_click.cpt')
    pl_module = PairedModule.load_from_checkpoint(model_path,
                                                  mlm_model_trx=mlm_model_trx,
                                                  mlm_model_click=mlm_model_click,
                                                  )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

        # scores for rtk='0'
        # 0 score is maximum, z_out has only negative values
        z_out = torch.cat([
            torch.zeros((T, 1), device=device),
            z_out,
        ], dim=1)
    return z_out


def main():
    valid_fold_id = int(sys.argv[1])
    uid_banks, uid_rtk, valid_dl_trx, valid_dl_click = load_data(valid_fold_id)

    res = []
    model_list = sorted(glob('nn_distance_coles_model_*.cpt'))
    if len(model_list) == 0:
        raise FileNotFoundError('No models found')
    for i, model_path in enumerate(model_list):
        z_out = get_pairvise_distance_with_model(model_path, valid_dl_trx, valid_dl_click)
        res.append(z_out)
        print(f'Cross scores done [{i}: "{model_path}"]')

    # merge ensemble
    z_out = torch.stack(res, dim=0).sum(dim=0)
    print('Merge ensemble done')

    # add '0' rtk at 1st position
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
    np.savez('submission_final.npz', submission_final)


if __name__ == "__main__":
    main()
