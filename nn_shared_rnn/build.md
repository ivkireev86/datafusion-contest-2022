# Pack
```
cp -r ../../pytorch-lifestream/dltranz/ ./
cp -r ../vtb_code ./
cp ../data/click_categories.csv ./  

python nn_distance_train.py

zip -r nn_shared_rnn.zip dltranz vtb_code metadata.json model.py nn_distance_inference.py click_categories.csv nn_distance_coles_model.cpt preprocessor_click.p preprocessor_trx.p

```
