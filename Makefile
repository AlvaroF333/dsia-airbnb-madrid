.PHONY: setup lint format test cov download sample data train run all hf-smoke

PY ?= python
# Por defecto se trabaja con la muestra versionada; para los datos completos:
#   make data LISTINGS=data/raw/listings.csv.gz
LISTINGS ?= data/sample/listings_sample.csv
REVIEWS ?= data/sample/reviews_sample.csv
TOP_N ?= 20

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

cov:
	pytest -q --cov --cov-report=term-missing --cov-fail-under=80

# Datos completos de Inside Airbnb (necesita red) y regeneración de la muestra
download:
	$(PY) scripts/download_data.py

sample:
	$(PY) scripts/make_sample.py

# Flujo del Proyecto II: ingesta -> entrenamiento -> pipeline con inferencia e IA
data:
	$(PY) -m airbnb_price ingest --input $(LISTINGS)

train:
	$(PY) -m airbnb_price train

run:
	$(PY) -m airbnb_price run --reviews $(REVIEWS) --top-n $(TOP_N)

all: data train run

# Prueba del proveedor real (necesita HF_TOKEN en .env; nunca en CI)
hf-smoke:
	AI_PROVIDER=hf $(PY) scripts/hf_smoke.py
