# Experiments

0. Prepare data by `../README.md` 
1. Explore the data with `view_data.ipynb`
2. Run `prepare_features.ipynb` to prepare features
3. Train end estimate models

| Model                             | ValPre | ValMRR | Val R1  |  Description |
| --------------------------------- | ------ | ------ | ------- | ------------ |
| train_distance-coles-hour.ipynb   | 0.4458 [0.4328, 0.4416, 0.4416, 0.4497, 0.4529, 0.4564] | 0.1886 [0.1817, 0.1888, 0.1891, 0.1900, 0.1909, 0.1910] | 0.2660 [0.2619, 0.2646, 0.2666, 0.2668, 0.2672, 0.2693] | Distance between embeddings means similarity. New features added |
| train_shared_rnn.ipynb            | 0.4618 [0.4601, 0.4601, 0.4638, 0.4611, 0.4720, 0.4538] | 0.1925 [0.1918, 0.1938, 0.1938, 0.1920, 0.1931, 0.1907] | 0.2718 [0.2708, 0.2727, 0.2733, 0.2711, 0.2741, 0.2686] | Shared rnn model |
| train_pretrain_qsm.ipynb          | 0.5007 [0.5174, 0.5014, 0.5038, 0.4877, 0.4966, 0.4971] | 0.1954 [0.1972, 0.1955, 0.1959, 0.1936, 0.1946, 0.1957] | 0.2811 [0.2856, 0.2813, 0.2821, 0.2772, 0.2796, 0.2808] | Pretrained with MLM TrxEncoders with extra features, shared rnn, QuerySoftmaxLoss with dual anchors |
| train_pretrain_qsm.ipynb (clip)   | 0.5025 [0.514 , 0.497 , 0.506 , 0.496 , 0.498 , 0.504 ] | 0.1947 [0.194 , 0.198 , 0.194 , 0.194 , 0.193 , 0.195 ] | 0.2808 [0.282 , 0.283 , 0.281 , 0.279 , 0.279 , 0.281 ] | |
| train_pretrain_qsm_ensemble.ipynb | 0.5322 [0.544 , 0.540 , 0.535 , 0.495 , 0.540 , 0.539 ] | 0.2005 [0.201 , 0.203 , 0.202 , 0.191 , 0.202 , 0.204 ] | 0.2913 [0.294 , 0.295 , 0.293 , 0.276 , 0.294 , 0.296 ] | Based on `train_pretrain_qsm.ipynb`, pretrain once on full data, next train 5 models with different random seeds. Sum paired distances as ensemble predict merge |
| train_pretrain_qsm_ensmbl (clip)  | 0.5425 [0.549 , 0.533 , 0.546 , 0.541 , 0.541 , 0.545 ] | 0.2028 [0.202 , 0.204 , 0.204 , 0.202 , 0.202 , 0.203 ] | 0.2952 [0.296 , 0.295 , 0.297 , 0.294 , 0.294 , 0.295 ] | |
| train_concat.ipynb                |  0.278 |  0.174 |  0.214  |  Concat embeddings and predict match as binary task  |
| train_distance.ipynb              |  0.335 |  0.179 |  0.233  |  Distance between embeddings means similarity. Pairs are sampled from whole sequence |
| train_distance_window.ipynb       |  0.15  |  0.013 |  0.024  |  Distance between embeddings means similarity. Pairs are sampled from the specific time window  |

4. (not fully implemented) Valid pretrained model with `valid.ipynb` on any dataset.
