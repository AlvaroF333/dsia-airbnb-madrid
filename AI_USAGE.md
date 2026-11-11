# Uso de IA

Registro del uso de herramientas de IA en este proyecto, según la política de la asignatura.

## IA dentro del producto

| Campo | Valor |
| --- | --- |
| Proveedor por defecto | `mock` (`ai/mock.py`): determinista, sin red ni claves. Lo usan siempre los tests y la CI |
| Proveedor real | Hugging Face Inference (`huggingface_hub.InferenceClient`, `provider="auto"`), activado con `AI_PROVIDER=hf` y `HF_TOKEN` en `.env` |
| Tarea 1 | **`text_classification`** con `cardiffnlp/twitter-xlm-roberta-base-sentiment` (`HF_SENTIMENT_MODEL`). Calcula el sentimiento de hasta 5 reseñas recientes de los alojamientos más desviados; la puntuación se normaliza a 0..1 |
| Tarea 2 | **`chat_completion`** con `mistralai/Mistral-7B-Instruct-v0.3` (`HF_CHAT_MODEL`), `max_tokens=150`. Explica en dos frases, en español, por qué el precio anunciado se desvía del estimado, usando solo datos del alojamiento |
| Para qué se usa | Enriquecer el resultado del modelo de scikit-learn, no sustituirlo: el precio lo estima siempre el modelo propio, y si la IA falla el pipeline entrega igualmente los precios |
| Datos enviados | Texto de reseñas (recortado a 500 caracteres) y características del alojamiento. Nunca nombres de quien reseña, que se eliminan en la muestra |

### Evidencia del proveedor real

**Pendiente.** Los dos modelos son los del plan y no se han verificado todavía contra la API real, porque requiere un token personal. Para obtener la evidencia:

```bash
cp .env.example .env     # poner el token en HF_TOKEN=
make hf-smoke            # o: AI_PROVIDER=hf python scripts/hf_smoke.py
AI_PROVIDER=hf python -m airbnb_price run --top-n 5
```

| Fecha | Comando | Resultado | Modelos realmente usados |
| --- | --- | --- | --- |
| _pendiente_ | `make hf-smoke` | _pendiente_ | _pendiente_ |

Si un modelo responde 404 (no servido por ningún proveedor de Inference), se cambia `HF_SENTIMENT_MODEL` o `HF_CHAT_MODEL` en `.env` y se anota aquí.

## Asistentes de programación

Todo el código y la documentación de este repositorio se han escrito con **Claude Code** (modelos Claude Opus 5.5 y Claude Fable 5.2 de Anthropic). Los commits lo indican con `Co-Authored-By`.

| Fase | Qué generó el asistente | Decisiones y comprobaciones durante la generación | Qué revisó o cambió Alvaro |
| --- | --- | --- | --- |
| Plan y fase 0 | Análisis de los enunciados, roadmap, esqueleto del paquete (configuración, CLI, excepciones, tests, CI, README) | — | _Completar_ |
| Fase 1 (T1) | Ingesta: loader, validator con 5 reglas de limpieza, job `ingest`, tests | — | _Completar_ |
| Plan del Proyecto II | `docs/PLAN_PROYECTO_II.md`: requisitos R1–R14, contratos y tareas T1–T8 | — | _Completar_ |
| T3–T5 | `features/build.py`, `model/train.py` y `model/predict.py`, `ai/` (base, mock, hf, factory, cache) y sus tests | — | _Completar_ |
| T2 | `download_data.py`, `make_sample.py`, `load_reviews`, `clean_reviews` | Esquema real comprobado contra el contrato (las 16 columnas existen); eliminados `reviewer_id` y `reviewer_name` de la muestra; descartadas reseñas sin letras (p. ej. `"."`) | _Completar_ |
| T3 y T6 | Comandos `train`, `predict`, `sentiment` y `run`; `pipeline.py`; `validate_prediction_input`; tests unitarios y de extremo a extremo | Cortacircuitos tras 3 errores de IA seguidos; el pipeline escribe su propio CSV limpio para no pisar los datos de entrenamiento (cubierto por un test) | _Completar_ |
| Automatización y T7 | `Makefile`, `run_pipeline.sh`, `hf_smoke.py`, CI con cobertura ≥ 80 % y flujo completo | `.gitattributes` para forzar saltos de línea LF en `.sh` y `Makefile` | _Completar_ |
| T8 | Entrenamiento con los datos completos, `model_card.md`, README y este fichero | Métricas copiadas de `reports/model_metrics.json`; análisis de error por distrito y tramo de precio calculado antes de redactar los límites (corrigió una afirmación inicial falsa sobre la periferia); `model_meta.json` sin rutas locales | _Completar_ |
