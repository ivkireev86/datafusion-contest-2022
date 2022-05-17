# Pack
```
cp -r ../../pytorch-lifestream/dltranz/ ./
cp -r ../vtb_code ./
cp ../data/click_categories.csv ./  

python nn_distance_train.py

# for 5 models
zip -r nn_pretrain_qsm_ensemble.zip dltranz vtb_code metadata.json model.py nn_distance_inference.py click_categories.csv \
  pretrain_trx.cpt pretrain_click.cpt \
  nn_distance_coles_model_0.cpt nn_distance_coles_model_1.cpt nn_distance_coles_model_2.cpt nn_distance_coles_model_3.cpt nn_distance_coles_model_4.cpt \
  preprocessor_click.p preprocessor_trx.p

# for 11 models
zip -r nn_pretrain_qsm_ensemble11-upd.zip dltranz vtb_code metadata.json model.py nn_distance_inference.py click_categories.csv \
  pretrain_trx.cpt pretrain_click.cpt \
  nn_distance_coles_model_0.cpt nn_distance_coles_model_1.cpt nn_distance_coles_model_2.cpt nn_distance_coles_model_3.cpt \
  nn_distance_coles_model_4.cpt nn_distance_coles_model_5.cpt nn_distance_coles_model_6.cpt nn_distance_coles_model_7.cpt \
  nn_distance_coles_model_8.cpt nn_distance_coles_model_9.cpt nn_distance_coles_model_10.cpt \
  preprocessor_click.p preprocessor_trx.p

```
