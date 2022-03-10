# Experiments

0. Prepare data by `../README.md` 
1. Explore the data with `view_data.ipynb`
2. Run `prepare_features.ipynb` to prepare features
3. Train end estimate models

| Model                       | ValPre | ValMRR | Val R1  |  Description |
| --------------------------- | ------ | ------ | ------- | ------------ |
| train_concat.ipynb          |  0.278 |  0.174 |  0.214  |  Concat embeddings and predict match as binary task  |
| train_distance.ipynb        |  0.335 |  0.179 |  0.233  |  Distance between embeddings means similarity. Pairs are sampled from whole sequence |
| train_distance_window.ipynb |  0.15  |  0.013 |  0.024  |  Distance between embeddings means similarity. Pairs are sampled from the specific time window  |

4. (not fully implemented) Valid pretrained model with `valid.ipynb` on any dataset.
