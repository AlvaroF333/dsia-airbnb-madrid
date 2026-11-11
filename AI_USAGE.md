# Uso de IA

Registro del uso de herramientas de IA en este proyecto, según la política de la asignatura.

## IA dentro del producto

| Campo | Valor |
| --- | --- |
| Proveedor por defecto | `mock` (`ai/mock.py`): determinista, sin red ni claves. Lo usan siempre los tests y la CI |
| Proveedor real | Hugging Face Inference (`huggingface_hub.InferenceClient`, `provider="auto"`), activado con `AI_PROVIDER=hf` y `HF_TOKEN` en `.env` |
| Tarea 1 | **`text_classification`** con `cardiffnlp/twitter-xlm-roberta-base-sentiment` (`HF_SENTIMENT_MODEL`). Calcula el sentimiento de hasta 5 reseñas recientes de los alojamientos más desviados; la puntuación se normaliza a 0..1 |
| Tarea 2 | **`chat_completion`** con `Qwen/Qwen2.5-72B-Instruct` (`HF_CHAT_MODEL`; sustituye a `mistralai/Mistral-7B-Instruct-v0.3`, ver abajo), `max_tokens=150`. Explica en dos frases, en español, por qué el precio anunciado se desvía del estimado, usando solo datos del alojamiento |
| Para qué se usa | Enriquecer el resultado del modelo de scikit-learn, no sustituirlo: el precio lo estima siempre el modelo propio, y si la IA falla el pipeline entrega igualmente los precios |
| Datos enviados | Texto de reseñas (recortado a 500 caracteres) y características del alojamiento. Nunca nombres de quien reseña, que se eliminan en la muestra |

### Evidencia del proveedor real (11 nov 2026)

Prueba ejecutada por Alvaro con su token en `.env` (`python scripts/hf_smoke.py` y `AI_PROVIDER=hf python -m airbnb_price run --top-n 5`).

| Comprobación | Resultado |
| --- | --- |
| Autenticación | Token aceptado: la API responde `402 Payment Required` y no `401` |
| `text_classification` con `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Modelo **activo** en el proveedor `hf-inference`; la llamada devuelve `402`: la cuenta no tiene crédito para Inference Providers |
| `chat_completion` con `mistralai/Mistral-7B-Instruct-v0.3` | `400 Bad Request`: su único proveedor (`novita`) figura en estado `error`, así que **el modelo ya no se sirve**, haya crédito o no |
| Cambio de modelo de chat | Se sustituye por `Qwen/Qwen2.5-72B-Instruct`: sin acceso restringido, 2 proveedores activos (`featherless-ai`, `deepinfra`) y no es un modelo de razonamiento (con `max_tokens=150` no se queda sin respuesta). Elegido consultando `inferenceProviderMapping` en la API pública del Hub; **no se ha probado con una llamada real** por falta de crédito |
| Comportamiento del cliente | Sin reintentos ante `402` y `400` (no son transitorios); error convertido en `AIClientError`; el token no aparece en el log, en `reports/` ni en `cache/` |
| Pipeline con el proveedor real fallando | Exit `0`, 1.623 predicciones entregadas, 0 filas enriquecidas, cortacircuitos activado tras 3 errores, 1,5 s en total; `run_metrics.json` registra `ai_errores: 3` |

**Pendiente:** obtener respuestas reales de los dos modelos requiere añadir crédito a la cuenta de Hugging Face (prepago o PRO). Con crédito, basta con repetir `python scripts/hf_smoke.py`; no hace falta cambiar código.

## Asistentes de programación

He usado **Claude Code** (modelos Claude Opus 5.5 y Claude Fable 5.2 de Anthropic) para generar la mayor parte del código y de la documentación de este repositorio; los commits lo indican con `Co-Authored-By`. Mi papel ha sido definir el proyecto y el esquema del pipeline, dirigir el trabajo por fases, pedir que cada paso se me explicara para entenderlo, revisar los resultados y tomar las decisiones que recoge la última columna.

| Fase | Qué generó el asistente | Decisiones y comprobaciones durante la generación | Qué revisó o cambió Alvaro |
| --- | --- | --- | --- |
| Plan y fase 0 | Análisis de los enunciados, roadmap, esqueleto del paquete (configuración, CLI, excepciones, tests, CI, README) | — | Diseñé un diagrama del pipeline completo como punto de partida, que sirvió de base para el plan y la estructura del paquete |
| Fase 1 (T1) | Ingesta: loader, validator con 5 reglas de limpieza, job `ingest`, tests | — | Tomé como referencia la arquitectura de otros trabajos para orientar la separación entre lectura, validación y limpieza |
| Plan del Proyecto II | `docs/PLAN_PROYECTO_II.md`: requisitos R1–R14, contratos y tareas T1–T8 | — | Generé el plan con un modelo de IA a partir del enunciado y lo usé como especificación de las tareas |
| T3–T5 | `features/build.py`, `model/train.py` y `model/predict.py`, `ai/` (base, mock, hf, factory, cache) y sus tests | — | Iteré con el asistente sobre varias versiones hasta quedarme con la actual |
| T2 | `download_data.py`, `make_sample.py`, `load_reviews`, `clean_reviews` | Esquema real comprobado contra el contrato (las 16 columnas existen); eliminados `reviewer_id` y `reviewer_name` de la muestra; descartadas reseñas sin letras (p. ej. `"."`) | Revisé el código y autoricé la descarga de los datos reales completos (23.109 alojamientos y sus reseñas) en lugar de trabajar solo con datos sintéticos |
| T3 y T6 | Comandos `train`, `predict`, `sentiment` y `run`; `pipeline.py`; `validate_prediction_input`; tests unitarios y de extremo a extremo | Cortacircuitos tras 3 errores de IA seguidos; el pipeline escribe su propio CSV limpio para no pisar los datos de entrenamiento (cubierto por un test) | Revisé el código |
| Automatización y T7 | `Makefile`, `run_pipeline.sh`, `hf_smoke.py`, CI con cobertura ≥ 80 % y flujo completo | `.gitattributes` para forzar saltos de línea LF en `.sh` y `Makefile` | Fusioné el PR en `main` después de comprobar que la CI pasaba |
| T8 | Entrenamiento con los datos completos, `model_card.md`, README y este fichero | Métricas copiadas de `reports/model_metrics.json`; análisis de error por distrito y tramo de precio calculado antes de redactar los límites (corrigió una afirmación inicial falsa sobre la periferia); `model_meta.json` sin rutas locales | Creé el token de Hugging Face, ejecuté la prueba con el proveedor real y decidí documentar el resultado (`402`) sin pagar crédito; revisé esta tabla |
