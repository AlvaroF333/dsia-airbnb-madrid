# Estimador de precio de alojamientos en Madrid

Trabajo Final de **Desarrollo de Soluciones de IA** (DSIA 2026-2027): Proyecto II (núcleo) y Proyecto III (E2E).

Estima el precio por noche de un alojamiento turístico en Madrid con un modelo de **scikit-learn** entrenado sobre datos de Inside Airbnb, y lo enriquece con modelos del Hub de **Hugging Face**: sentimiento de las reseñas (como variable del modelo) y una explicación en lenguaje natural de la estimación.

> **Estado:** Fase 0 (cimientos): paquete, configuración, CLI mínima, tests y CI. Ingesta, modelo y borde de IA llegan en las fases siguientes.

## Datos

| Campo | Valor |
| --- | --- |
| Origen | [Inside Airbnb, Madrid](https://insideairbnb.com/get-the-data/) (`listings.csv.gz`, `reviews.csv.gz`) |
| Licencia | Creative Commons Attribution 4.0 (CC BY 4.0), a confirmar en la página de origen |
| Fecha de descarga | Pendiente (Fase 1) |
| Motivación | Datos reales y abiertos, objetivo supervisado claro (precio) y texto de reseñas donde un modelo del Hub aporta una señal útil |

En el repositorio solo se versiona una muestra pequeña (`data/sample/`); los datos completos se descargan con un script documentado.

## Arquitectura

```text
Entradas finas:  CLI + cron (II)   ·   API FastAPI (III)   ·   Dashboard (III)
                          │
                   pipeline.py  (orquesta)
                          │
  Ingesta (loader + validator) → Features (puro) → Modelo sklearn (train/predict)
                          │
                 Cliente IA: mock por defecto · hf con HF_TOKEN
```

Reglas de frontera: la CLI y la API no limpian datos ni llaman a Hugging Face directamente; `features` no toca disco ni red; el cliente de IA no conoce pandas.

## Los 4 contratos

1. **Schema de entrada:** columnas obligatorias, tipos y rangos del CSV de listados (se fija en la Fase 1). Si no se cumple, el job termina con código `2`.
2. **Schema de salida de la predicción:** `{"precio_estimado": float, "modelo_version": str, "sentimiento": float, "explicacion": str, "provider": str}`.
3. **Proveedor de IA por defecto:** `mock`. Hugging Face solo con `AI_PROVIDER=hf` y `HF_TOKEN`.
4. **Demo lista:** en un clon limpio y sin claves, `make setup`, después `make test`, y la CLI responde (se amplía a `data → train → run` en la Fase 4).

### Códigos de salida

| Código | Significado |
| --- | --- |
| `0` | Ejecución correcta |
| `1` | Fallo del sistema o de una dependencia externa |
| `2` | La entrada o la configuración no cumplen el contrato |

## Puesta en marcha

Requiere Python 3.11 o superior (la CI usa 3.12).

```bash
git clone <url-del-repo> && cd dsia-airbnb-madrid
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make setup                         # pip install -r requirements.txt (paquete editable + dev)
cp .env.example .env               # opcional: solo hace falta para usar Hugging Face
python -m airbnb_price info        # muestra la configuración efectiva, sin secretos
```

## Tests y calidad

```bash
pytest -q                                                   # todo
pytest -q -m "not integration"                              # solo unitarios
pytest -q --cov --cov-report=term-missing --cov-fail-under=80
make lint                                                   # ruff check + format
pre-commit install                                          # hooks antes de cada commit
```

Los tests no usan red ni claves: el proveedor de IA es siempre `mock` y un fixture limpia las variables de entorno.

## CI

El workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) ejecuta lint, tests y cobertura (mínimo 70 %) en cada push a `main` y en cada pull request.

## Roadmap

| Fase | Fechas | Entregable |
| --- | --- | --- |
| 0. Cimientos | 26–27 oct 2026 | Paquete, configuración, CLI, tests, CI |
| 1. Ingesta y validación | 28–30 oct | `ingest`, schema de entrada, `metrics.json` |
| 2. Modelo sklearn | 31 oct–3 nov | `train`, `joblib`, métricas frente a un baseline |
| 3. Borde Hugging Face | 4–6 nov | Cliente `mock`/`hf`: sentimiento + explicación |
| 4. Pipeline integrado | 7–8 nov | `run` de extremo a extremo, wrapper para cron |
| 5. Entrega del Proyecto II | 9 nov | README final, `AI_USAGE.md`, tag `v0.2.0` |
| Proyecto III | 10–23 nov | API, dashboard, robustez, despliegue |

## Uso de IA

Ver [`AI_USAGE.md`](AI_USAGE.md).
