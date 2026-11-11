# Plan de ejecución del Proyecto II (para un agente de IA)

Este documento es una especificación autocontenida. Un agente de IA que lo lea, sin más contexto, debe poder terminar el Proyecto II de principio a fin. Está escrito el 11 nov 2026; **la entrega es el 12 nov 2026**.

## 1. Contexto

- Asignatura: Desarrollo de Soluciones de IA (máster), curso 2026-27. Alumno: Alvaro Figueroa.
- Proyecto II (20 % de la nota): "Automatización e integración de servicios". Es la base del Proyecto III (API, dashboard y despliegue), que **no** forma parte de este plan.
- Repositorio: `AlvaroF333/dsia-airbnb-madrid`. Paquete Python: `airbnb_price` (carpeta `src/`).
- Producto: estimar el precio por noche de un alojamiento de Airbnb en Madrid.
- Idioma: todo el texto visible (README, mensajes de la CLI, docstrings, commits) va en español.

## 2. Qué exige el enunciado

Cada punto es obligatorio. La columna "Evidencia" dice cómo se comprueba.

| # | Requisito | Evidencia |
| --- | --- | --- |
| R1 | Dominio y datos documentados: origen, licencia, motivación | Sección "Datos" del README |
| R2 | Pipeline reproducible por CLI, con logs | `python -m airbnb_price run` |
| R3 | Entrenamiento de un modelo scikit-learn reproducible | `python -m airbnb_price train` |
| R4 | Modelo persistido | `models/model.joblib` + `models/model_meta.json` |
| R5 | Inferencia integrada en el pipeline (no en un notebook) | `run` genera `reports/predictions.csv` |
| R6 | Métricas de evaluación reportadas | `reports/model_metrics.json` y tabla en el README |
| R7 | Tests pytest: unitarios + al menos uno de integración | `pytest -q` y `pytest -q -m integration` |
| R8 | Al menos un test de carga e inferencia del modelo | `tests/unit/test_model.py` |
| R9 | Cliente de Hugging Face con modo `mock` (obligatorio en tests y CI) | `src/airbnb_price/ai/` |
| R10 | Proveedor real documentado: tarea Inference + modelo `owner/name` + para qué se usa | README y `AI_USAGE.md` |
| R11 | `.env.example` con `HF_TOKEN=` y `requirements.txt` con `scikit-learn` y `huggingface_hub` | Ficheros en la raíz |
| R12 | `AI_USAGE.md`: proveedor, tarea, modelo, qué código de asistentes se aceptó | Fichero en la raíz |
| R13 | Estructura modular que admita API y despliegue sin reescribir el núcleo | Capas de la sección 5 |
| R14 | README: datos → entrenamiento → pipeline con inferencia → borde HF → qué se ampliará en el III | README |

Rúbrica: pipeline y automatización 20 %, modelo sklearn 20 %, integración HF 20 %, tests y calidad 15 %, robustez 10 %, documentación y continuidad 15 %.

## 3. Restricciones que no se negocian

1. **Sin secretos en el repo.** `HF_TOKEN` solo en `.env` local. El agente nunca pide ni escribe el token.
2. **`mock` por defecto.** Sin configurar nada, todo funciona sin red ni claves. La CI usa siempre `AI_PROVIDER=mock`.
3. **Tests sin red.** Ningún test llama a internet.
4. **No inventar resultados.** Las métricas del README salen de `reports/model_metrics.json` generado con datos reales. Si no hay datos reales, se dice y no se rellena con números supuestos.
5. **Fronteras entre capas** (sección 5). La CLI no limpia datos; `features` no toca disco ni red; el cliente de IA no conoce pandas.
6. **No tocar el Proyecto III**: nada de API, dashboard ni Docker.
7. Cada tarea termina con `ruff check .`, `ruff format --check .` y `pytest -q --cov --cov-fail-under=70` en verde.

## 4. Límites del entorno del agente y qué hace la persona

El agente en la nube **no tiene red** hacia `insideairbnb.com` ni `huggingface.co`. Solo la persona puede:

| Acción de la persona | Cuándo | Qué desbloquea |
| --- | --- | --- |
| Descargar `listings.csv.gz` (Madrid, detallado) de <https://insideairbnb.com/get-the-data/> y subirlo al hilo | Ya | Entrenar con datos reales (T3, T8) |
| Descargar `reviews.csv.gz` y ejecutar `python scripts/make_sample.py` en local, o subirlo si pesa poco | Hoy | Muestra de reseñas reales (T2) |
| Crear un token en <https://huggingface.co/settings/tokens>, ponerlo en `.env` y ejecutar `make hf-smoke` | Hoy | Evidencia del proveedor real (T6, R10) |
| Revisar y fusionar el PR, y entregar | Mañana | Entrega |

Si los datos reales no llegan a tiempo, el agente entrega todo el código funcionando sobre la muestra sintética de los tests y deja el entrenamiento real como un único comando (`make data train run`) que la persona ejecuta. Esto se indica en el README sin disimularlo.

## 5. Arquitectura objetivo

```text
src/airbnb_price/
  config.py          # hecho: Settings.from_env()
  exceptions.py      # hecho: excepciones de dominio + códigos de salida 0/1/2
  logging_conf.py    # hecho
  cli.py             # hecho: info, ingest  | falta: train, predict, sentiment, run
  data/
    loader.py        # hecho: load_listings  | falta: load_reviews
    validator.py     # hecho: REQUIRED_COLUMNS, clean_listings
    ingest.py        # hecho: run_ingest
  features/
    build.py         # falta: build_features (puro)
  model/
    train.py         # falta: train_model, evaluación, guardado
    predict.py       # falta: load_model, predict
  ai/
    base.py          # falta: Protocol AIClient, SentimentResult, AIClientError
    mock.py          # falta: MockClient determinista
    hf.py            # falta: HFClient (InferenceClient)
    factory.py       # falta: get_client(settings)
    cache.py         # falta: caché en disco
  pipeline.py        # falta: run_pipeline
scripts/
  download_data.py   # falta
  make_sample.py     # falta
  run_pipeline.sh    # falta
tests/unit, tests/integration
data/sample/         # muestra versionada
models/              # model.joblib, model_meta.json, model_card.md
reports/             # *.json, predictions.csv (no se versionan)
```

Códigos de salida de la CLI: `0` correcto, `1` fallo del sistema o de una dependencia externa, `2` entrada o configuración inválida.

## 6. Contratos

**Entrada** (`listings.csv` de Inside Airbnb). Columnas obligatorias en `data/validator.py::REQUIRED_COLUMNS`. Si falta alguna: `DataValidationError`, exit 2. El primer paso al recibir el fichero real es comprobar que estas columnas existen y ajustar la lista si el esquema real difiere.

**Salida de una predicción** (`predict` y cada fila de `predictions.csv`):

```json
{
  "id": "12345",
  "precio_estimado": 96.4,
  "modelo_version": "0.2.0",
  "sentimiento": 0.82,
  "n_resenas": 5,
  "explicacion": "…",
  "provider": "mock",
  "model": "mock-sentiment"
}
```

`sentimiento` va de 0 (negativo) a 1 (positivo); es `null` si el alojamiento no tiene reseñas. `provider` dice siempre la verdad: `mock` o `hf`.

**Cliente de IA** (`ai/base.py`):

```python
@dataclass(frozen=True)
class SentimentResult:
    score: float        # 0..1, probabilidad de sentimiento positivo
    label: str          # "positive" | "neutral" | "negative"
    provider: str
    model: str

class AIClient(Protocol):
    name: str
    def sentiment(self, texts: Sequence[str]) -> list[SentimentResult]: ...
    def explain(self, context: Mapping[str, object]) -> str: ...
```

## 7. Tareas

Se ejecutan en orden. Cada una es un commit (o varios pequeños) en la rama `proyecto-ii`.

### T1 · Ingesta — HECHA
`ingest` valida y limpia, escribe `listings_clean.csv` y `ingest_metrics.json`, con umbral `--max-error-rate`.

### T2 · Datos reales y muestra
- `scripts/download_data.py --date AAAA-MM-DD`: descarga `listings.csv.gz` y `reviews.csv.gz` a `data/raw/`. Solo biblioteca estándar. Lo ejecuta la persona.
- `scripts/make_sample.py`: de `data/raw/` genera `data/sample/listings_sample.csv` (2.000 alojamientos, semilla 42) y `data/sample/reviews_sample.csv` (hasta 5 reseñas por alojamiento de la muestra, texto recortado a 500 caracteres).
- `data/loader.py::load_reviews(path)`: columnas `listing_id`, `date`, `comments`.
- README, sección "Datos": URL, licencia, fecha de la instantánea, tamaño.
- Hecho cuando: `python -m airbnb_price ingest --input data/sample/listings_sample.csv` termina con exit 0.

### T3 · Features y modelo sklearn
- `features/build.py::build_features(df) -> DataFrame` puro. Variables: `accommodates`, `bedrooms`, `beds`, `bathrooms` (número extraído de `bathrooms_text`; "half-bath" = 0,5), `n_amenities` (elementos de la lista JSON `amenities`), `minimum_nights`, `number_of_reviews`, `review_scores_rating`, `is_superhost` (`"t"` → 1), `dist_sol_km` (haversine a 40.4168, -3.7038), `latitude`, `longitude`, y categóricas `room_type` y `neighbourhood_group_cleansed`.
- `model/train.py::train_model(df, seed=42)`: `Pipeline` con `ColumnTransformer` (imputación por mediana en numéricas; `OneHotEncoder(handle_unknown="ignore")` en categóricas) + `HistGradientBoostingRegressor`, dentro de `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
- Evaluación: partición 80/20 con `random_state=42`. Dos referencias sobre el mismo test: `DummyRegressor(strategy="median")` y `Ridge` con el mismo preprocesado (más `StandardScaler`). Métricas de las tres: MAE, RMSE, R², error absoluto mediano. Validación cruzada de 5 pliegues del modelo final sobre train (MAE medio y desviación).
- Salidas: `models/model.joblib`, `models/model_meta.json` (versión, fecha UTC, versión de scikit-learn, n filas, hash SHA-256 del CSV de entrenamiento, lista de features), `reports/model_metrics.json`.
- `model/predict.py`: `load_model(path)` (lanza `ModelNotFoundError`, exit 2, si no existe) y `predict(model, df) -> ndarray`.
- CLI: `train [--input] [--model-dir]` y `predict --input fichero.json`.
- Tests (R8): features con DataFrame en memoria; entrenamiento con ~200 filas sintéticas generadas con semilla, guardado, carga y predicción positiva; reproducibilidad (dos entrenamientos dan las mismas métricas); `ModelNotFoundError`.
- Hecho cuando: `train` sobre la muestra escribe los tres ficheros y el log muestra las métricas de modelo y referencias.

### T4 · Cliente de IA: interfaz y mock
- `ai/base.py` (sección 6) y `AIClientError` (exit 1).
- `ai/mock.py::MockClient`: determinista y sin red. `sentiment`: cuenta palabras positivas y negativas de una lista corta en español e inglés; score = 0,5 + 0,5·(pos − neg)/(pos + neg), 0,5 si no hay ninguna. `explain`: plantilla fija con los datos del contexto, prefijo `[mock]`.
- `ai/factory.py::get_client(settings)`.
- Tests: determinismo, rango 0..1, lista vacía, factoría según `AI_PROVIDER`.

### T5 · Cliente de Hugging Face
- `ai/hf.py::HFClient(token, sentiment_model, chat_model, timeout=30, max_retries=2)` con `huggingface_hub.InferenceClient`, importado dentro del constructor para que el mock no dependa de la librería en tiempo de ejecución.
- Tarea 1, `text_classification`. Modelo por defecto: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingüe; etiquetas negative/neutral/positive). Alternativa: `nlptown/bert-base-multilingual-uncased-sentiment` (1 a 5 estrellas). Configurable con `HF_SENTIMENT_MODEL`. El score se normaliza a 0..1 con una función pura testeada que admite los dos formatos de etiquetas.
- Tarea 2, `chat_completion`. Modelo configurable con `HF_CHAT_MODEL`; por defecto el del material del curso, `mistralai/Mistral-7B-Instruct-v0.3`. `max_tokens=150`. El prompt pide 2 frases en español y solo incluye datos del alojamiento.
- Reintentos solo ante fallos transitorios (timeout, 429, 5xx), con espera 0,5 s y 1 s. Nunca ante 401, 403 o 404. Cualquier fallo final se convierte en `AIClientError` sin incluir el token.
- `ai/cache.py`: JSONL en `cache/` con clave SHA-256 de `modelo + texto`. Solo para el proveedor real.
- Log por llamada: proveedor, modelo, duración, resultado. Nunca el texto ni el token.
- Los modelos por defecto no están verificados contra Inference desde el entorno del agente. `make hf-smoke` (lo ejecuta la persona) clasifica 2 frases y pide 1 explicación; si un modelo responde 404, se cambia la variable y se anota en `AI_USAGE.md`.
- Tests con un `InferenceClient` falso inyectado: normalización de etiquetas, reintento tras un fallo transitorio, sin reintento ante 401, caché, token ausente del mensaje de error.

### T6 · Pipeline integrado
- `pipeline.py::run_pipeline(settings, input_path, reviews_path, top_n=20)`:
  1. ingesta (reutiliza `run_ingest`);
  2. features y predicción con el modelo guardado;
  3. `desviacion = precio − precio_estimado`; se eligen los `top_n` alojamientos con mayor desviación absoluta;
  4. para esos, sentimiento medio de sus reseñas (hasta 5) y explicación;
  5. escribe `reports/predictions.csv` (todas las filas; sentimiento y explicación solo en las `top_n`) y `reports/run_metrics.json` (filas, duración por etapa, proveedor, modelo, llamadas a IA, errores de IA).
- Degradación controlada: si la IA falla, el pipeline termina con las predicciones, deja `sentimiento` y `explicacion` vacíos, cuenta el error y devuelve exit 0 con un aviso en el log. El precio es el producto; la IA lo enriquece.
- CLI: `run [--input] [--reviews] [--top-n]` y `sentiment --text "…"`.
- `scripts/run_pipeline.sh`: activa el venv, ejecuta `run`, añade la salida a `logs/pipeline.log` y propaga el código de salida. Línea de crontab de ejemplo en el README.
- `Makefile`: `data` (ingesta de la muestra), `train`, `run`, `hf-smoke`.
- Test de integración de extremo a extremo con datos sintéticos en `tmp_path` y proveedor mock; otro con un cliente de IA que siempre falla.
- Hecho cuando: en un clon limpio, `make setup && make data && make train && make run` termina sin claves.

### T7 · CI
El workflow existente ya ejecuta lint, tests y cobertura. Subir el umbral a 80 % si la cobertura real lo permite. Añadir un paso que ejecute `make data train run` sobre la muestra para probar el flujo completo en CI.

### T8 · Entrenamiento real y documentación
Requiere los datos reales (sección 4).
- Entrenar con el `listings` completo. Versionar `models/model.joblib` si pesa menos de 5 MB; si no, solo `model_meta.json` y las instrucciones.
- `models/model_card.md`: datos, variables, métricas frente a las referencias, uso previsto, límites (una sola instantánea, precio anunciado y no precio pagado, sin estacionalidad).
- README final con las secciones de R14, la tabla de métricas reales, los comandos, el esquema de capas y "Qué llega en el Proyecto III".
- `AI_USAGE.md`: tareas y modelos de HF realmente usados, comandos de evidencia, y qué partes generó el asistente y qué revisó Alvaro.
- Etiqueta `v0.2.0`.

## 8. Definición de terminado

- [ ] R1 a R14 con su evidencia.
- [ ] `make setup && make data && make train && make run` en un clon limpio, sin `.env`.
- [ ] CI en verde en `main`.
- [ ] Ningún secreto en el repo ni en el historial.
- [ ] Métricas del README copiadas de `reports/model_metrics.json`, con el modelo por encima de las dos referencias o, si no, explicado.
- [ ] `AI_USAGE.md` completo y veraz.

## 9. Orden de recorte si falta tiempo

1. Explicación con `chat_completion` (se queda el sentimiento, que ya cumple R9 y R10).
2. Caché en disco.
3. Paso de CI con el flujo completo.

No se recortan: el modo `mock`, los tests del modelo, el pipeline `run`, ni la veracidad de las métricas.
