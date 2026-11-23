.PHONY: setup lint format test cov download sample data tune train run all ollama-up ai-smoke hf-smoke

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

# Búsqueda de hiperparámetros con CV sobre train (lenta; escribe models/hparams.json)
tune:
	$(PY) -m airbnb_price tune

train:
	$(PY) -m airbnb_price train

run:
	$(PY) -m airbnb_price run --reviews $(REVIEWS) --top-n $(TOP_N)

all: data train run

# Proveedor real sin claves: Ollama en Docker con un LLM abierto (Qwen2.5 1.5B, ~1 GB)
OLLAMA_MODEL ?= qwen2.5:1.5b
ollama-up:
	docker start dsia-ollama 2>/dev/null || docker run -d --name dsia-ollama -p 11434:11434 -v dsia-ollama:/root/.ollama ollama/ollama:0.12.6
	docker exec dsia-ollama ollama pull $(OLLAMA_MODEL)

# Pruebas de humo de los proveedores reales (dejan evidencia en reports/; nunca en CI)
ai-smoke:
	$(PY) scripts/ai_smoke.py --provider ollama

hf-smoke:
	$(PY) scripts/ai_smoke.py --provider hf
