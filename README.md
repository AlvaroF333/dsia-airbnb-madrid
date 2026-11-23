# Estimador de precio de alojamientos en Madrid

Trabajo Final de **Desarrollo de Soluciones de IA** (DSIA 2026-2027). Este repositorio contiene el **Proyecto II** (automatización e integración de servicios), base del Proyecto III (API, dashboard y despliegue).

Estima el precio por noche de un alojamiento turístico en Madrid con un modelo de **scikit-learn** entrenado con datos de Inside Airbnb. Después **enriquece con un LLM abierto servido por API** (Ollama con Qwen2.5; Hugging Face como alternativa) los alojamientos cuyo precio anunciado más se desvía del estimado: calcula el sentimiento de sus reseñas y genera una explicación en lenguaje natural.

> **Estado (v0.3.0):** ingesta, modelo, pipeline `run` con informe, cliente de IA (`mock`, `ollama` y `hf`), tests y CI terminados. El modelo versionado está entrenado con los datos reales completos. La v0.3.0 añade 13 variables y una búsqueda de hiperparámetros con CV: el MAE baja de 54,21 a 47,25 €. La IA generativa está **verificada con un LLM real vía API** (Ollama con `qwen2.5:1.5b`): 20 de 20 alojamientos enriquecidos y 0 errores. Hugging Face queda como alternativa, porque su API exige crédito de pago (detalle en [`AI_USAGE.md`](AI_USAGE.md)).

## Datos

| Campo | Valor |
| --- | --- |
| Origen | [Inside Airbnb, Madrid](https://insideairbnb.com/get-the-data/): `listings.csv.gz` (detallado) y `reviews.csv.gz` |
| Licencia | [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), según la página de origen |
| Instantánea | 17 de septiembre de 2026 (descargada el 11 nov 2026) |
| Tamaño | 23.109 alojamientos × 90 columnas (11,8 MB comprimido); reseñas: 174 MB comprimido |
| Tras la ingesta | 18.620 alojamientos válidos: 4.324 sin precio y 165 con precio fuera de 10–1.000 € |
| Motivación | Datos reales y abiertos, objetivo supervisado claro (precio) y texto de reseñas en el que un modelo del Hub aporta una señal que el tabular no tiene |

En el repositorio solo se versiona una **muestra** (`data/sample/`, 2,9 MB):

- `listings_sample.csv`: 2.000 alojamientos al azar (semilla 42), solo con las 30 columnas del contrato de entrada.
- `reviews_sample.csv`: hasta 5 reseñas recientes por alojamiento de la muestra (7.531 reseñas de 1.678 alojamientos). El texto está limpio y recortado a 500 caracteres. **Se eliminan `reviewer_id` y `reviewer_name`**, porque no hacen falta y son datos personales.

```bash
python scripts/download_data.py --date 2026-09-17   # datos completos a data/raw/ (no se versionan)
python scripts/make_sample.py                       # regenera data/sample/
```

## Arquitectura

```text
Entradas finas:   CLI + cron (II)   ·   API FastAPI (III)   ·   Dashboard (III)
                            │
                     pipeline.py  (orquesta y mide)
                            │
  data/ (loader + validator + ingest) → features/ (puro) → model/ (train / predict)
                            │
     ai/: AIClient ── MockClient (por defecto) · OllamaClient (API local) · HFClient (HF_TOKEN) + caché
```

```text
src/airbnb_price/
  config.py        Settings.from_env(): proveedor, token, modelos de HF, rutas
  exceptions.py    excepciones de dominio con su código de salida
  cli.py           info · ingest · tune · train · predict · sentiment · run
  pipeline.py      run_pipeline: ingesta → predicción → desviación → IA → informes
  report.py        render_report (puro): informe Markdown de una ejecución
  data/            loader (único acceso a disco), validator (contratos, puro), ingest
  features/        build_features (puro: DataFrame → DataFrame)
  model/           train (hiperparámetros, entrenamiento, evaluación, análisis de error, guardado), predict
  ai/              base (Protocol), common (prompt, normalización, reintentos), mock, ollama, hf, factory, cache
```

Reglas de frontera: la CLI no limpia datos ni llama a la IA; `features` no toca disco ni red; el cliente de IA no conoce pandas. Gracias a ellas, la API del Proyecto III reutilizará `validate_prediction_input`, `load_model`, `predict` y `get_client` sin cambios.

## Puesta en marcha

Requiere Python 3.11 o superior (la CI usa 3.12). En un clon limpio, **sin `.env` ni claves**:

```bash
git clone https://github.com/AlvaroF333/dsia-airbnb-madrid.git && cd dsia-airbnb-madrid
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make setup                         # pip install -r requirements.txt
make all                           # data → train → run sobre la muestra, con IA mock
```

Sin `make` (por ejemplo, en Windows):

```bash
pip install -r requirements.txt
python -m airbnb_price ingest --input data/sample/listings_sample.csv
python -m airbnb_price train
python -m airbnb_price run
```

`make all` reentrena con la muestra y sobrescribe `models/model.joblib`. Para reproducir el modelo versionado con los datos completos:

```bash
make download && make data LISTINGS=data/raw/listings.csv.gz && make train   # antes make tune, para repetir la búsqueda
```

## 1. Ingesta

`python -m airbnb_price ingest --input <csv>` valida el **contrato de entrada** (30 columnas obligatorias de `data/validator.py::REQUIRED_COLUMNS`) y aplica reglas de limpieza: precio presente y entre 10 y 1.000 €, sin ids duplicados, capacidad entre 1 y 16, coordenadas dentro del municipio. Escribe `data/processed/listings_clean.csv` y `reports/ingest_metrics.json`, con las filas rechazadas por cada regla. `--max-error-rate` corta el job con exit 2 si se rechazan demasiadas filas.

## 2. Entrenamiento

`python -m airbnb_price train` entrena un `HistGradientBoostingRegressor` sobre `log(1 + precio)` con **27 variables**: tamaño, equipamiento, condiciones de reserva, disponibilidad, reseñas, anfitrión y ubicación. Las 4 categóricas (tipo de habitación, tipo de inmueble, distrito y barrio) usan el soporte nativo del boosting. Todo va dentro de un único `Pipeline`. La partición es 80/20 con `random_state=42`. Se compara con dos referencias sobre el mismo test y se hace validación cruzada de 5 pliegues sobre train.

Se excluyen a propósito las columnas que **filtran el precio**: `estimated_revenue_l365d` (ocupación × precio) y `price_quote_*`. También se excluyen las que esta instantánea publica vacías (`host_since` e `instant_bookable`). Si una variable llega vacía, `train` lo detecta y termina con exit 2.

**Hiperparámetros:** `python -m airbnb_price tune` (o `make tune`, unos 2 minutos) hace una búsqueda aleatoria con 30 combinaciones y 5 pliegues **solo sobre train** y guarda el resultado en `models/hparams.json`, que está versionado. `train` usa ese fichero si existe. El test nunca interviene en la elección.

Salidas:

- `models/model.joblib`: 2,6 MB comprimido, versionado.
- `models/model_meta.json`: versión, fecha UTC, versión de scikit-learn, filas, SHA-256 de los datos, variables e hiperparámetros.
- `reports/model_metrics.json`: métricas, más el **análisis de error por tramo de precio y por distrito** y la **importancia de cada variable por permutación**.

Métricas con los datos completos (test de 3.724 alojamientos, copiadas de `reports/model_metrics.json`):

| Modelo | MAE (€) | RMSE (€) | R² | Error abs. mediano (€) |
| --- | --- | --- | --- | --- |
| **HistGradientBoosting v0.3.0** | **47,25** | **86,29** | **0,6224** | **25,65** |
| Ridge (mismas variables, one-hot + escalado) | 59,96 | 107,40 | 0,4150 | 32,32 |
| Mediana (`DummyRegressor`) | 92,57 | 144,10 | −0,0531 | 63,85 |
| *v0.2.0 (14 variables, hiperparámetros de serie)* | *54,21* | *96,70* | *0,5257* | *29,21* |

Validación cruzada: MAE **46,84 ± 1,34 €**. Según la CV, la mayor parte de la mejora frente a v0.2.0 viene de las variables nuevas (de 53,28 a 48,48 €) y el resto de la búsqueda de hiperparámetros (de 48,48 a 46,84 €). Las variables que más pesan son el tipo de inmueble, las noches mínimas (separan el alquiler de media estancia, mucho más barato por noche), el barrio y la capacidad. El mayor límite sigue siendo que **infraestima los alojamientos caros** (−204 € de media por encima de 400 €). Detalle, usos previstos y límites en la [model card](models/model_card.md).

Predicción puntual con un JSON (el mismo contrato que tendrá `POST /predict` en el III):

```bash
echo '{"id": "demo", "latitude": 40.4168, "longitude": -3.7038, "room_type": "Entire home/apt", "accommodates": 4}' > piso.json
python -m airbnb_price predict --input piso.json
```

Son obligatorios la ubicación, `room_type` y `accommodates`; el resto de variables se imputa.

## 3. Pipeline con inferencia

`python -m airbnb_price run [--input] [--reviews] [--top-n 20]`:

1. **Ingesta** de la entrada (por defecto, la muestra), en un fichero propio que no pisa los datos de entrenamiento.
2. **Predicción** de todos los alojamientos con el modelo guardado.
3. **Desviación** = precio anunciado − precio estimado; se eligen los `top_n` con mayor desviación absoluta.
4. **IA** solo para esos `top_n`: sentimiento medio de hasta 5 reseñas recientes y explicación breve.
5. **Informes**: `reports/predictions.csv` (todas las filas), `reports/run_metrics.json` (filas, duración por etapa, proveedor, modelos, llamadas y errores de IA) y `reports/report.md`, un **informe legible** con el resumen de la ejecución, la tabla de los alojamientos más desviados (con su sentimiento y su explicación), los barrios con mayor desviación y la duración de cada etapa.

Cada fila de `predictions.csv` sigue el contrato de salida:

```json
{"id": "27111785", "barrio": "Universidad", "precio": 736.0, "precio_estimado": 231.6,
 "desviacion": 504.4, "modelo_version": "0.3.0", "sentimiento": 0.8333, "n_resenas": 3,
 "explicacion": "[mock] …", "provider": "mock", "model": "mock-sentiment"}
```

`sentimiento` va de 0 a 1 y queda vacío si no hay reseñas. `provider` y `model` solo se rellenan en las filas en las que realmente se llamó a la IA.

**Degradación controlada:** el precio es el producto y la IA lo enriquece. Si el proveedor falla, el pipeline termina igualmente con exit 0 y todas las predicciones, deja vacíos el sentimiento y la explicación, cuenta los errores en `run_metrics.json` y avisa en el log. Tras 3 errores seguidos deja de llamar al proveedor, para no encadenar reintentos y timeouts.

**Automatización con cron:** `scripts/run_pipeline.sh` activa el entorno, ejecuta `run`, añade la salida a `logs/pipeline.log` y propaga el código de salida.

```cron
0 6 * * * /ruta/a/dsia-airbnb-madrid/scripts/run_pipeline.sh --top-n 20
```

### Códigos de salida

| Código | Significado | Ejemplos |
| --- | --- | --- |
| `0` | Ejecución correcta (también con la IA degradada) | `run` con HF caído |
| `1` | Fallo del sistema o de una dependencia externa | `sentiment` con HF caído, error inesperado |
| `2` | La entrada o la configuración no cumplen el contrato | faltan columnas, no hay modelo, `AI_PROVIDER=hf` sin token, una variable vacía al entrenar |

## 4. Borde de IA generativa

El resto del proyecto solo conoce el `Protocol` `AIClient` (`ai/base.py`), con dos métodos: `sentiment(texts)` y `explain(context)`. `ai/factory.py::get_client` elige la implementación según `AI_PROVIDER`. Cambiar de proveedor no toca ni el pipeline ni el modelo.

| | `mock` (por defecto) | **`ollama`** (proveedor real verificado) | `hf` |
| --- | --- | --- | --- |
| Servicio | Ninguno: código local | API REST de [Ollama](https://ollama.com) (`POST /api/chat`), en Docker | Hugging Face Inference Providers |
| Modelo | — | **`qwen2.5:1.5b`** (Qwen2.5 1.5B Instruct, Apache 2.0), configurable con `OLLAMA_MODEL` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` y `Qwen/Qwen2.5-72B-Instruct` |
| Claves y coste | No | No: modelo abierto, sin cuenta | `HF_TOKEN`; requiere crédito de pago |
| Sentimiento | Cuenta palabras positivas y negativas | Chat con **salida estructurada** (JSON Schema): `{label, confidence}` → 0..1 | **`text_classification`** → 0..1 |
| Explicación | Plantilla fija con prefijo `[mock]` | Chat, temperatura 0 y semilla 42, `num_predict=150` | **`chat_completion`**, `max_tokens=150` |
| Uso | Tests, CI y demo sin red | Ejecución real y demo | Alternativa en la nube (sin crédito en esta cuenta) |

Las dos implementaciones reales comparten `ai/common.py`:

- el mismo prompt de explicación: dos frases en español, solo con los datos del alojamiento y con importes en €;
- la normalización del sentimiento a 0..1, con funciones puras y probadas. Un LLM devuelve "negativo con confianza 0,9", que se traduce como 0,5 − 0,5 · 0,9 = 0,05. Un clasificador devuelve probabilidades por etiqueta;
- la política de robustez: reintentos con espera de 0,5 y 1 s **solo** ante fallos transitorios (timeout, 429, 5xx), nunca ante 4xx. Todo fallo final se convierte en `AIClientError`, sin el token en el mensaje.

Además hay una caché JSONL en `cache/`, con clave SHA-256 de modelo + texto. El log registra el proveedor, el modelo, la latencia y el resultado, nunca el texto ni el token. Si Ollama no está arrancado, el error lo dice y no se reintenta. El pipeline entrega los precios igualmente.

Para usar el proveedor real (solo hace falta Docker):

```bash
make ollama-up                     # contenedor ollama/ollama:0.12.6 + descarga de qwen2.5:1.5b (~1 GB)
make ai-smoke                      # 4 reseñas + 1 explicación; evidencia en reports/ai_smoke_ollama.json
AI_PROVIDER=ollama python -m airbnb_price run --top-n 20
```

`hf` sigue disponible con `HF_TOKEN` en `.env` y `make hf-smoke`. Con el token se verificó la autenticación, pero la API responde `402 Payment Required` porque la cuenta no tiene crédito. Por eso el proveedor real verificado es Ollama (detalle en [`AI_USAGE.md`](AI_USAGE.md)).

## Tests y calidad

```bash
pytest -q                                                   # todo (165 tests)
pytest -q -m integration                                    # integración: CLI y extremo a extremo
pytest -q --cov --cov-report=term-missing --cov-fail-under=80
make lint                                                   # ruff check + format
```

- **Sin red ni claves**: un fixture limpia las variables de entorno y el proveedor es siempre `mock`. `HFClient` se prueba con un `InferenceClient` falso: normalización de etiquetas, reintento tras un fallo transitorio, sin reintento ante 401, caché y token ausente de los errores. `OllamaClient` se prueba con un transporte HTTP falso: salida estructurada, normalización de `{label, confidence}`, respuestas no válidas, reintentos, servidor apagado y caché.
- **Modelo** (`tests/unit/test_model.py`): entrenamiento con datos sintéticos, guardado, carga y predicción, reproducibilidad, `ModelNotFoundError`, análisis de error, búsqueda de hiperparámetros sin filas de test, uso de `hparams.json` y error claro si una variable llega vacía.
- **Informe** (`tests/unit/test_report.py`): secciones, formato de números, escape de celdas, recorte de textos largos y aviso si la IA se desactivó.
- **Extremo a extremo** (`tests/integration/test_pipeline_e2e.py`): `ingest → tune → train → run → predict` por la CLI en un directorio temporal, con un proveedor que siempre falla y sin fichero de reseñas.
- Cobertura actual: 98,9 %.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) se ejecuta en cada push a `main` y en cada pull request, siempre con `AI_PROVIDER=mock`. Hace lint, tests con una cobertura mínima del 80 %, tests de integración, el flujo completo `make all` sobre la muestra (comprobando que hay predicciones e informe, y que el proveedor fue `mock`) y el envoltorio de cron.

## Qué llega en el Proyecto III

- **API FastAPI**: `POST /predict` con el contrato de `validate_prediction_input` (errores de dominio → 4xx y fallos externos → 5xx, igual que los códigos 2 y 1), `POST /explain` y `GET /health`.
- **Dashboard** sobre `reports/predictions.csv` y `run_metrics.json`: mapa de desviaciones, sentimiento y explicaciones.
- **Despliegue** en contenedor, con el modelo versionado y los secretos fuera de la imagen.
- **Mejoras del modelo** detectadas aquí: el sentimiento de las reseñas como variable, la estacionalidad (varias instantáneas o `calendar.csv.gz`) y la corrección del sesgo en precios altos.

## Uso de IA

Ver [`AI_USAGE.md`](AI_USAGE.md).
