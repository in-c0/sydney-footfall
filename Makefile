PY ?= python3
.PHONY: all data dataset eval train web
all: data dataset train web

data:          ## download raw counts, weather and OpenStreetMap context (~120 MB)
	cd src && $(PY) fetch_data.py && $(PY) fetch_osm.py
dataset:       ## harmonise sources into data/processed/*.parquet
	cd src && $(PY) build_dataset.py
eval:          ## site-grouped 5-fold CV + temporal holdout vs baselines -> models/eval_footfall.json
	cd src && $(PY) train_footfall.py --eval
train:         ## final 3-seed FootfallNet ensemble + 24 h forecaster -> models/
	cd src && $(PY) train_footfall.py --final && $(PY) train_forecaster.py
web:           ## street predictions + single-file demo -> web/index.html
	cd src && $(PY) predict.py --export-web && $(PY) build_web.py && cp ../web/index.html ../docs/index.html
