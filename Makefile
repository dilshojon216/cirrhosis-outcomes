PYTHON_VERSION := python3.12
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data train predict test lint pipeline clean

## Virtual muhit yaratadi va bog'liqliklarni o'rnatadi
setup:
	$(PYTHON_VERSION) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## Ma'lumotlarni yuklab oladi: Kaggle (API kalit kerak) + ochiq UCI dataset
data:
	mkdir -p data/raw
	$(PY) -m kaggle competitions download -c multiclassificationtask -p data/raw
	cd data/raw && unzip -o multiclassificationtask.zip && rm -f multiclassificationtask.zip
	curl -sSL -o /tmp/cirrhosis-uci.zip \
		"https://archive.ics.uci.edu/static/public/878/cirrhosis+patient+survival+prediction+dataset-1.zip"
	unzip -o /tmp/cirrhosis-uci.zip -d data/raw
	rm -f /tmp/cirrhosis-uci.zip

## Yakuniy production modelni o'qitadi va models/ ga saqlaydi
train:
	$(PY) -m src.train

## Saqlangan modellardan outputs/submissions/submission.csv yaratadi
predict:
	$(PY) -m src.predict

## Testlarni ishga tushiradi
test:
	$(PY) -m pytest tests/ -v

## Lint (ruff)
lint:
	$(PY) -m ruff check src/ tests/

## To'liq pipeline: data -> train -> predict (bitta buyruq bilan tekshirish uchun)
pipeline: data train predict

## Generatsiya qilingan artefaktlarni tozalaydi (xom ma'lumot va venv saqlanadi)
clean:
	rm -f models/*.pkl
	rm -f outputs/oof/*.npy
	rm -f outputs/submissions/*.csv
	rm -f data/processed/*.parquet
