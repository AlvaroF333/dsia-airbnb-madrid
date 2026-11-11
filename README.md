# Estimador de precio de alojamientos en Madrid

Trabajo Final de **Desarrollo de Soluciones de IA** (DSIA 2026-2027). Este repositorio contiene el **Proyecto II** (automatización e integración de servicios), base del Proyecto III (API, dashboard y despliegue).

Estima el precio por noche de un alojamiento turístico en Madrid con un modelo de **scikit-learn** entrenado con datos de Inside Airbnb. Después **enriquece con modelos del Hub de Hugging Face** los alojamientos cuyo precio anunciado más se desvía del estimado: calcula el sentimiento de sus reseñas y genera una explicación en lenguaje natural.

> **Estado (v0.2.0):** ingesta, modelo, pipeline `run`, cliente de IA `mock`/`hf`, tests y CI terminados. El modelo versionado está entrenado con los datos reales completos. El proveedor real de Hugging Face está implementado y probado con un cliente falso. Contra la API real se ha verificado la autenticación y la degradación del pipeline, pero no se han obtenido respuestas porque la cuenta no tiene crédito de Inference Providers (detalle en [`AI_USAGE.md`](AI_USAGE.md#evidencia-del-proveedor-real-11-nov-2026)).

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

- `listings_sample.csv`: 2.000 alojamientos al azar (semilla 42), solo con las 16 columnas del contrato de entrada.
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
              ai/: AIClient ── MockClient (por defecto) · HFClient (HF_TOKEN) + caché
```

```text
src/airbnb_price/
  config.py        Settings.from_env(): proveedor, token, modelos de HF, rutas
  exceptions.py    excepciones de dominio con su código de salida
  cli.py           info · ingest · train · predict · sentiment · run
  pipeline.py      run_pipeline: ingesta → predicción → desviación → IA → informes
  data/            loader (único acceso a disco), validator (contratos, puro), ingest
  features/        build_features (puro: DataFrame → DataFrame)
  model/           train (entrenamiento + evaluación + guardado), predict
  ai/              base (Protocol), mock, hf, factory, cache
```

Reglas de frontera: la CLI no limpia datos ni llama a Hugging Face; `features` no toca disco ni red; el cliente de IA no conoce pandas. Gracias a ellas, la API del Proyecto III reutilizará `validate_prediction_input`, `load_model`, `predict` y `get_client` sin cambios.

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
make download && make data LISTINGS=data/raw/listings.csv.gz && make train
```

## 1. Ingesta

`python -m airbnb_price ingest --input <csv>` valida el **contrato de entrada** (16 columnas obligatorias de `data/validator.py::REQUIRED_COLUMNS`) y aplica reglas de limpieza: precio presente y entre 10 y 1.000 €, sin ids duplicados, capacidad entre 1 y 16, coordenadas dentro del municipio. Escribe `data/processed/listings_clean.csv` y `reports/ingest_metrics.json`, con las filas rechazadas por cada regla. `--max-error-rate` corta el job con exit 2 si se rechazan demasiadas filas.

## 2. Entrenamiento

`python -m airbnb_price train` entrena un `HistGradientBoostingRegressor` sobre `log(1 + precio)`, con imputación por mediana y one-hot de `room_type` y distrito, dentro de un único `Pipeline`. La partición es 80/20 con `random_state=42`. Se compara con dos referencias sobre el mismo test y se hace validación cruzada de 5 pliegues sobre train.

Salidas: `models/model.joblib` (340 KB, versionado), `models/model_meta.json` (versión, fecha UTC, versión de scikit-learn, filas, SHA-256 de los datos, lista de variables) y `reports/model_metrics.json`.

Métricas con los datos completos (test de 3.724 alojamientos, copiadas de `reports/model_metrics.json`):

| Modelo | MAE (€) | RMSE (€) | R² | Error abs. mediano (€) |
| --- | --- | --- | --- | --- |
| **HistGradientBoosting** | **54,21** | **96,70** | **0,5257** | **29,21** |
| Ridge (mismo preprocesado + escalado) | 63,65 | 112,89 | 0,3537 | 35,61 |
| Mediana (`DummyRegressor`) | 92,57 | 144,10 | −0,0531 | 63,85 |

Validación cruzada: MAE **53,28 ± 1,56 €**. El modelo supera a las dos referencias. Su mayor límite es que **infraestima los alojamientos caros** (−241 € de media por encima de 400 €). Detalle, usos previstos y límites en la [model card](models/model_card.md).

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
5. **Informes**: `reports/predictions.csv` (todas las filas) y `reports/run_metrics.json` (filas, duración por etapa, proveedor, modelos, llamadas y errores de IA).

Cada fila de `predictions.csv` sigue el contrato de salida:

```json
{"id": "1068196650380263312", "barrio": "Cortes", "precio": 998.0, "precio_estimado": 315.87,
 "desviacion": 682.13, "modelo_version": "0.2.0", "sentimiento": 0.9, "n_resenas": 5,
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
| `2` | La entrada o la configuración no cumplen el contrato | faltan columnas, no hay modelo, `AI_PROVIDER=hf` sin token |

## 4. Borde de Hugging Face

El resto del proyecto solo conoce el `Protocol` `AIClient` (`ai/base.py`), con dos métodos: `sentiment(texts)` y `explain(context)`. `ai/factory.py::get_client` elige la implementación según `AI_PROVIDER`:

| | `mock` (por defecto) | `hf` |
| --- | --- | --- |
| Red y claves | No | `HF_TOKEN` en `.env` |
| Sentimiento | Cuenta palabras positivas y negativas (español e inglés) | **`text_classification`** con `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingüe; negative/neutral/positive) |
| Explicación | Plantilla fija con prefijo `[mock]` | **`chat_completion`** con `Qwen/Qwen2.5-72B-Instruct`, `max_tokens=150`, dos frases en español solo con datos del alojamiento |
| Uso | Tests, CI y demo sin claves | Ejecución real |

Ambos modelos se pueden cambiar con `HF_SENTIMENT_MODEL` y `HF_CHAT_MODEL`. El score se normaliza a 0..1 con una función pura que admite etiquetas `negative/neutral/positive`, `LABEL_0..2` y `1–5 stars` (por ejemplo, `nlptown/bert-base-multilingual-uncased-sentiment`).

Robustez de `HFClient`: timeout de 30 s; reintentos con espera de 0,5 y 1 s **solo** ante fallos transitorios (timeout, 429, 5xx), nunca ante 401, 403 o 404; todo fallo final se convierte en `AIClientError` **sin el token en el mensaje**. Hay una caché JSONL en `cache/`, con clave SHA-256 de modelo + texto, para no repetir llamadas. El log registra el proveedor, el modelo, la latencia y el resultado, nunca el texto ni el token.

Para usar el proveedor real:

```bash
cp .env.example .env               # y pon tu token en HF_TOKEN= (https://huggingface.co/settings/tokens)
make hf-smoke                      # o: AI_PROVIDER=hf python scripts/hf_smoke.py
AI_PROVIDER=hf python -m airbnb_price run --top-n 5
```

`make hf-smoke` clasifica dos frases y pide una explicación. Si un modelo deja de estar servido por algún proveedor de Inference, se cambia la variable y se anota en [`AI_USAGE.md`](AI_USAGE.md). Ya ocurrió con `mistralai/Mistral-7B-Instruct-v0.3`, el modelo de chat inicial, sustituido por `Qwen/Qwen2.5-72B-Instruct`. Un `402 Payment Required` indica que la cuenta no tiene crédito: el pipeline lo trata como un fallo de la IA y entrega igualmente los precios.

## Tests y calidad

```bash
pytest -q                                                   # todo (123 tests)
pytest -q -m integration                                    # integración: CLI y extremo a extremo
pytest -q --cov --cov-report=term-missing --cov-fail-under=80
make lint                                                   # ruff check + format
```

- **Sin red ni claves**: un fixture limpia las variables de entorno y el proveedor es siempre `mock`. `HFClient` se prueba con un `InferenceClient` falso: normalización de etiquetas, reintento tras un fallo transitorio, sin reintento ante 401, caché y token ausente de los errores.
- **Modelo** (`tests/unit/test_model.py`): entrenamiento con datos sintéticos, guardado, carga y predicción, reproducibilidad y `ModelNotFoundError`.
- **Extremo a extremo** (`tests/integration/test_pipeline_e2e.py`): `ingest → train → run → predict` por la CLI en un directorio temporal, con un proveedor que siempre falla y sin fichero de reseñas.
- Cobertura actual: 98,8 %.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) se ejecuta en cada push a `main` y en cada pull request, siempre con `AI_PROVIDER=mock`. Hace lint, tests con una cobertura mínima del 80 %, tests de integración, el flujo completo `make all` sobre la muestra (comprobando que hay predicciones y que el proveedor fue `mock`) y el envoltorio de cron.

## Qué llega en el Proyecto III

- **API FastAPI**: `POST /predict` con el contrato de `validate_prediction_input` (errores de dominio → 4xx y fallos externos → 5xx, igual que los códigos 2 y 1), `POST /explain` y `GET /health`.
- **Dashboard** sobre `reports/predictions.csv` y `run_metrics.json`: mapa de desviaciones, sentimiento y explicaciones.
- **Despliegue** en contenedor, con el modelo versionado y los secretos fuera de la imagen.
- **Mejoras del modelo** detectadas aquí: el sentimiento de las reseñas como variable, la estacionalidad (varias instantáneas o `calendar.csv.gz`) y la corrección del sesgo en precios altos.

## Uso de IA

Ver [`AI_USAGE.md`](AI_USAGE.md).
