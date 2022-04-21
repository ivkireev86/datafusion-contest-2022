# Pack
```
cp -r ../../pytorch-lifestream/dltranz/ ./
cp -r ../vtb_code ./
cp ../data/click_categories.csv ./  

python nn_distance_train.py

zip -r nn_pretrain_qsm_ensemble.zip dltranz vtb_code metadata.json model.py nn_distance_inference.py click_categories.csv \
  pretrain_trx.cpt pretrain_click.cpt \
  nn_distance_coles_model_0.cpt nn_distance_coles_model_1.cpt nn_distance_coles_model_2.cpt nn_distance_coles_model_3.cpt nn_distance_coles_model_4.cpt \
  preprocessor_click.p preprocessor_trx.p

```
