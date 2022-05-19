# Data Fusion Contest 2022. 1-st place on the Matching Task.

[Link to competition](https://ods.ai/competitions/data-fusion2022-main-challenge)

Our solution acheived **1-st** place on the private leaderboard.

# Overview

## Model

This is neural network based solution. We use [pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream)
library to simplify work with sequences data.

We create neural network to encode transaction sequences and clickstream to vector representation.
Matching and ranking based on L2 distance between transaction sequence embedding and clickstream embedding.

Network include two stages:
- TrxEncoder, which transform each individual transaction or click into vector representation
- SequenceEncoder, which reduce sequence of vectors from TrxEncoder into one final vector

Training batch contains 128 pairs of bank-rtk users. Short transaction subsequence sampled twice from each bank user
and the same for clickstream. So we have 256 trx sequences samples for 128 bank users 
and 256 click sequences samples for 128 rtk users.
Then we make 256*256 matrix with pairwise L2 distances, and the matrix with match-unmatched labels.
Labels used to sample positive and negative samples for Softmax Loss.

SequenceEncoder is RNN network, shared for trx and clicks. Weights initialised randomly.
TrxEncoder is a mix of embedding for categorical features and normalised numerical.
We make TrxEncoder for transactions and TrxEncoder for clicks. We pretrain weights for both TrxEncoder.

We were inspired BERT architecture for pretrain.
We use TrxEncoder + TransformerEncoder and MLM (Masked Language Model) task.
TransformerEncoder predict masked TrxEncoder vectors.

We use ensembling in final solution. We train 11 models and average paired distances from output.

## Validation
We split train data at 6 folds. So we have about 3000 pairs in fold. This is similar as contest validation and test set.

# How to reproduce 

## Install and use environment

```
pipenv sync --dev

pipenv shell
```

## Run
```
# Load data
sh get_data.sh

# Split data
python src/split_folds.py

cd src/

# Pipeline with validation, valid_fold_id from [0, 1, 2, 3, 4, 5] shold be provided as parameter
python nn_train.py valid_fold_id=4
python nn_inference.py valid_fold_id=4
python nn_evaluation.py valid_fold_id=4


# Pipeline without validation, train model before submit
python nn_distance_train.py valid_fold_id=None
# there are no data for validation, all data was in train
# python nn_distance_inference.py

```
