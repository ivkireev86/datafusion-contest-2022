https://ods.ai/tracks/data-fusion-2022-competitions/

# Install

```
pipenv sync --dev
pipenv install "setuptools==59.5.0"
```

# Run
```
# split data
python split_folds.py

# Then preproces data, rerun `notebooks/prepare_features.ipynb` six times, changing `FOLD_ID = 0, 1, ..., 5`
# this prepares pickles with prepared features by folds, witch are used in other notebooks.

```

# Pack
See `build.md` in submit folder
