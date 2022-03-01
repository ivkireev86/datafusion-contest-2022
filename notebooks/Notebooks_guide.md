# Experiments

0. Prepare data by `../README.md` 
1. Explore the data with `view_data.ipynb`
2. Run `prepare_features.ipynb` to prepare features
3. Train end estimate models

| Model                   | ValPre | ValMRR | Val R1  |  Description |
| ----------------------- | ------ | ------ | ------- | ------------ |
| train_concat.ipynb      |  0.12  |  0.009 |  0.017  |  Concat embeddings and predict match as binary task  |
| train_distance.ipynb    |  0.17  |  0.014 |  0.026  |  Distance between embeddings means similarity  |


4. (not fully implemented) Valid pretrained model with `valid.ipynb` on any dataset.
