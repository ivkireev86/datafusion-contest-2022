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

```

# Pack
```
zip -r nn_distance.zip dltranz vtb_code metadata.json nn_distance_inference.py click_categories.csv nn_distance_model.cpt preprocessor_click.p preprocessor_trx.p

```
