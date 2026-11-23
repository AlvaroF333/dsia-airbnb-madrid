# Uso de IA

Registro del uso de IA en este proyecto, según la política de la asignatura.

## 1. IA dentro del producto

| | |
| --- | --- |
| Proveedor por defecto | `mock`: determinista, sin red ni claves. Lo usan siempre los tests y la CI |
| **Proveedor real** | **Ollama** (API REST `POST /api/chat`, en Docker) con el modelo abierto **`qwen2.5:1.5b`** (Apache 2.0). Sin claves ni coste. `AI_PROVIDER=ollama` |
| Alternativa | Hugging Face Inference (`AI_PROVIDER=hf` + `HF_TOKEN`): implementado, pero la cuenta no tiene crédito (`402 Payment Required`) |
| Tareas | 1) **Sentimiento** de hasta 5 reseñas de cada alojamiento muy desviado (0 negativo, 1 positivo). 2) **Explicación** en dos frases de por qué su precio se desvía del estimado |
| Para qué | Enriquecer el modelo de scikit-learn, no sustituirlo: el precio lo estima siempre el modelo propio, y si la IA falla, se entregan igualmente los precios |

**Evidencia (23 nov 2026).** `make ai-smoke` clasifica bien 4 reseñas de prueba (positiva → 1,00, negativas → 0,05, neutra → 0,50) y genera una explicación en español (`reports/ai_smoke_ollama.json`). `AI_PROVIDER=ollama python -m airbnb_price run` enriqueció 20 de 20 alojamientos con 0 errores (`reports/run_metrics.json` y `reports/report.md`).

**Límite.** Con un modelo tan pequeño, las explicaciones son correctas pero genéricas. Se puede usar uno mayor con `OLLAMA_MODEL`.

## 2. Asistente de programación

**Qué hice o decidí yo:**

- Diseñé un diagrama del pipeline completo como punto de partida. Sirvió de base para el plan y la estructura del paquete.
- Tomé como referencia la arquitectura de otros trabajos para separar lectura, validación y limpieza de datos.
- Generé con IA el plan del proyecto (`docs/PLAN_PROYECTO_II.md`) a partir del enunciado y lo usé como especificación.
- Iteré con el asistente sobre varias versiones del modelo y del cliente de IA hasta quedarme con las actuales.
- Autoricé descargar los datos reales completos (23.109 alojamientos) en lugar de trabajar solo con datos sintéticos.
- Creé el token de Hugging Face y probé el proveedor real. Ante el `402`, decidí no pagar y usar un proveedor gratuito vía API (Ollama).
- Elegí el alcance de la mejora v0.3.0: más variables, búsqueda de hiperparámetros e informe generado.


**Qué se aceptó y qué se rechazó o corrigió:**

| Propuesta o resultado inicial | Decisión |
| --- | --- |
| Modelo de chat `mistralai/Mistral-7B-Instruct-v0.3` | **Rechazado**: ya no lo sirve ningún proveedor. Sustituido por `Qwen/Qwen2.5-72B-Instruct` |
| Pedir al LLM un `score` de positividad directamente | **Rechazado**: devolvía su confianza en la etiqueta ("negative" con 0,9). Ahora pide `{label, confidence}` y el código hace la conversión |
| Usar `estimated_revenue_l365d` y `price_quote_*` como variables | **Rechazado**: se calculan a partir del precio, así que el test daría resultados falsamente buenos |
| Usar `host_since` e `instant_bookable` | **Rechazado**: están vacías en esta instantánea. Se usa `hosts_time_as_host_*` y `train` avisa si una variable llega vacía |
| Elegir hiperparámetros mirando el test | **Rechazado**: la búsqueda se hace solo con validación cruzada sobre train |
| Una afirmación inicial sobre el error en la periferia | **Corregida**: se calculó el error por distrito antes de redactar los límites del modelo |
| `reviewer_id` y `reviewer_name` en la muestra de reseñas | **Eliminados**: son datos personales y no hacen falta |
| Métricas, `model_card.md` y README | **Aceptados** tras comprobar que las cifras salen de `reports/model_metrics.json` |

