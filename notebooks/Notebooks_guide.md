# Experiments

0. Prepare data by `../README.md` 
1. Explore the data with `view_data.ipynb`
2. Run `prepare_features.ipynb` to prepare features
3. Train end estimate models

| Model                            | ValPre | ValMRR | Val R1  |  Description |
| -------------------------------- | ------ | ------ | ------- | ------------ |
| train_distance-coles-hour.ipynb  | 0.4458 [0.4328, 0.4416, 0.4416, 0.4497, 0.4529, 0.4564] | 0.1886 [0.1817, 0.1888, 0.1891, 0.1900, 0.1909, 0.1910] | 0.2660 [0.2619, 0.2646, 0.2666, 0.2668, 0.2672, 0.2693] | Distance between embeddings means similarity. New features added |
| train_shared_rnn.ipynb           | 0.4618 [0.4601, 0.4601, 0.4638, 0.4611, 0.4720, 0.4538] | 0.1925 [0.1918, 0.1938, 0.1938, 0.1920, 0.1931, 0.1907] | 0.2718 [0.2708, 0.2727, 0.2733, 0.2711, 0.2741, 0.2686] | Shared rnn model |
| train_pretrain_qsm.ipynb (*)     | 0.5007 [0.5174, 0.5014, 0.5038, 0.4877, 0.4966, 0.4971] | 0.1954 [0.1972, 0.1955, 0.1959, 0.1936, 0.1946, 0.1957] | 0.2811 [0.2856, 0.2813, 0.2821, 0.2772, 0.2796, 0.2808] | Pretrained with MLM TrxEncoders with extra features, shared rnn, QuerySoftmaxLoss with dual anchors |
| train_pretrain_qsm.ipynb         | 0.5001 [0.4891, 0.4925, 0.5090, 0.5058, 0.4986, 0.5058] | 0.1946 [0.1947, 0.1950, 0.1946, 0.1941, 0.1945, 0.1948] | 0.2802 [0.2785, 0.2794, 0.2816, 0.2805, 0.2799, 0.2813] | Pretrained with MLM TrxEncoders without extra features, shared rnn, QuerySoftmaxLoss with dual anchors |
| train_concat.ipynb               |  0.278 |  0.174 |  0.214  |  Concat embeddings and predict match as binary task  |
| train_distance.ipynb             |  0.335 |  0.179 |  0.233  |  Distance between embeddings means similarity. Pairs are sampled from whole sequence |
| train_distance_window.ipynb      |  0.15  |  0.013 |  0.024  |  Distance between embeddings means similarity. Pairs are sampled from the specific time window  |

4. (not fully implemented) Valid pretrained model with `valid.ipynb` on any dataset.
