#!/usr/bin/env bash

mkdir data
cd data/

curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/0433a4ca/transactions.zip
curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/0554f0cf/clickstream.zip
curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/acfacf11/train_matching.csv

curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/b949c04c/mcc_codes.csv
curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/705abbab/click_categories.csv
curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/e33f2201/currency_rk.csv

curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/a3643657/sample_submission.zip
curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/24687252/baseline_catboost.zip

curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/b99fed70/puzzle.csv

cd ../
