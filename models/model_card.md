# Model card: estimador del precio por noche (Airbnb Madrid)

| Campo | Valor |
| --- | --- |
| Versión | `0.3.0` (`model_meta.json`) |
| Tipo | `TransformedTargetRegressor(log1p/expm1)` sobre `Pipeline(ColumnTransformer + HistGradientBoostingRegressor)` con categóricas nativas |
| Librería | scikit-learn 1.9.1, semilla 42 |
| Entrenado | 2026-11-23 (UTC), `python -m airbnb_price tune` + `python -m airbnb_price train` |
| Datos | Inside Airbnb, Madrid, instantánea del 17 sep 2026 (CC BY 4.0) |
| Filas | 23.109 en origen → 18.620 tras la ingesta (14.896 train, 3.724 test) |
| Huella de los datos | SHA-256 `88d606e5…a08a0` de `listings_clean.csv` (la misma que en v0.2.0) |
| Fichero | `model.joblib`, 2,6 MB (comprimido con `compress=3`) |

Todas las cifras de esta ficha salen de `reports/model_metrics.json` y `models/hparams.json`, generados por el código. No hay ninguna calculada a mano.

## Variables (27)

- **Numéricas** (el boosting trata los nulos de forma nativa):
  - Tamaño: `accommodates`, `bedrooms`, `beds` y `bathrooms` (extraído de `bathrooms_text`; "half-bath" = 0,5).
  - Equipamiento y anfitrión: `n_amenities`, `is_superhost`, `host_years` (`hosts_time_as_host_years` + meses / 12), `host_listings_count` y `calculated_host_listings_count`.
  - Reservas: `minimum_nights`, `maximum_nights`, `availability_30` y `availability_365`.
  - Reseñas: `number_of_reviews`, `number_of_reviews_ltm`, `reviews_per_month` y las puntuaciones `review_scores_rating`, `_cleanliness`, `_location` y `_value`.
  - Ubicación: `latitude`, `longitude` y `dist_sol_km` (distancia haversine a la Puerta del Sol).
- **Categóricas** (codificación ordinal y soporte categórico nativo del boosting; las categorías con menos de 10 casos se agrupan, y las nuevas se tratan como ausentes): `room_type`, `property_type` (55 tipos), `neighbourhood_group_cleansed` (21 distritos) y `neighbourhood_cleansed` (127 barrios).
- **Objetivo**: el `price` anunciado, en euros por noche, entre 10 y 1.000 €. El modelo aprende `log(1 + precio)` porque la distribución es muy asimétrica.

**Columnas excluidas a propósito:**

- `estimated_revenue_l365d` y `price_quote_*` se calculan **a partir del precio**. Con ellas el test saldría casi perfecto, pero en producción no existen, porque el precio es justo lo que se busca.
- `host_since` e `instant_bookable` están **vacías en el 100 %** de las filas de esta instantánea. Si una variable llega vacía, `train` lo detecta y se detiene con un error claro (exit 2).

## Métricas

Partición de test (3.724 alojamientos, `random_state=42`):

| Modelo | MAE (€) | RMSE (€) | R² | Error absoluto mediano (€) |
| --- | --- | --- | --- | --- |
| **HistGradientBoosting v0.3.0** | **47,25** | **86,29** | **0,6224** | **25,65** |
| Ridge con el mismo conjunto de variables (one-hot + escalado) | 59,96 | 107,40 | 0,4150 | 32,32 |
| Mediana (`DummyRegressor`) | 92,57 | 144,10 | −0,0531 | 63,85 |
| *v0.2.0, como referencia (14 variables, hiperparámetros de serie)* | *54,21* | *96,70* | *0,5257* | *29,21* |

Validación cruzada de 5 pliegues sobre train: MAE medio **46,84 €**, desviación **1,34 €**.

Lectura:

- El modelo reduce el MAE un 49 % frente a la mediana, un 21 % frente a Ridge y un **12,8 % frente a v0.2.0**.
- La CV (46,84 ± 1,34) coincide con el test (47,25), así que el resultado no depende de una partición afortunada.
- Según la CV, la mejora frente a v0.2.0 viene de dos fuentes. Las **variables nuevas** bajan el MAE de 53,28 a 48,48 € (con los hiperparámetros de serie). Los **hiperparámetros** lo bajan después de 48,48 a 46,84 €.

### Búsqueda de hiperparámetros

`python -m airbnb_price tune`: `RandomizedSearchCV` con 30 combinaciones y 5 pliegues, **solo sobre la partición de train**. El test no interviene en ninguna elección. El resultado se guarda en `models/hparams.json` (versionado), y `train` lo usa si existe.

| Hiperparámetro | De serie | Elegido | Espacio de búsqueda |
| --- | --- | --- | --- |
| `learning_rate` | 0,1 | 0,03 | 0,02 · 0,03 · 0,05 · 0,1 |
| `max_iter` | 100 | 600 | 300 · 600 · 1000 · 2000 (tope; hay parada temprana) |
| `max_leaf_nodes` | 31 | 127 | 15 · 31 · 63 · 127 |
| `min_samples_leaf` | 20 | 5 | 5 · 10 · 20 · 50 |
| `l2_regularization` | 0 | 1,0 | 0 · 0,1 · 1,0 |

Un primer espacio más estrecho dio un óptimo en su borde (CV 47,09 €). Al ampliarlo, la mejora fue de solo 0,25 €, muy por debajo de la desviación entre pliegues (1,34 €). Por eso no se siguió ampliando.

### Variables más importantes

Importancia por permutación en test: cuánto empeora el MAE (en €) al desordenar cada variable. Las 8 primeras:

| Variable | Aumento del MAE (€) |
| --- | --- |
| `property_type` | 17,32 |
| `minimum_nights` | 14,69 |
| `neighbourhood_cleansed` | 12,73 |
| `accommodates` | 11,27 |
| `bathrooms` | 5,13 |
| `bedrooms` | 4,89 |
| `availability_30` | 4,73 |
| `host_listings_count` | 2,01 |

`minimum_nights` pesa tanto porque separa dos mercados. El 2,4 % de los anuncios exige 32 noches o más (alquiler de media estancia), y su precio mediano por noche es 57,5 €, frente a 152 € en el resto. El barrio fino (`neighbourhood_cleansed`) aporta mucho más que el distrito (0,66 €), que queda casi redundante. Los pesos de las variables muy correlacionadas (por ejemplo, las tres de ubicación) se reparten entre ellas, así que cada una por separado parece menos importante de lo que es.

### Error por tramo de precio (mismo test)

| Precio anunciado | n | MAE (€) | Sesgo medio (estimado − anunciado, €) | v0.2.0: MAE / sesgo |
| --- | --- | --- | --- | --- |
| ≤ 100 € | 1.052 | 16,54 | +10,50 | 23,6 / +18,6 |
| 100–200 € | 1.520 | 31,72 | +12,60 | 33,6 / +13,6 |
| 200–400 € | 891 | 62,00 | −33,04 | 69,5 / −45,5 |
| > 400 € | 261 | 211,09 | −203,56 | 245,4 / −241,3 |

El modelo mejora en todos los tramos, pero mantiene la **regresión a la media**: sobreestima los alojamientos baratos e infraestima mucho los caros. Por distrito, el MAE es mayor donde los precios son altos (Salamanca 66,2 € y Centro 55,7 €) y queda entre 25 y 45 € en la mayoría de distritos periféricos. La tabla completa por distrito está en `model_metrics.json` (`analisis_error.por_distrito`).

## Uso previsto

- Estimar un precio de referencia por noche para un alojamiento de Madrid a partir de sus características.
- Detectar anuncios cuyo precio se desvía mucho de lo esperable, como apoyo a la revisión humana. El pipeline `run` lo hace con los 20 más desviados y lo resume en `reports/report.md`.

## Usos no previstos

- Fijar precios de forma automática o tomar decisiones sobre personas (anfitriones o huéspedes).
- Estimar ingresos reales: el dato es el precio **anunciado**, no el pagado.
- Usarlo fuera del municipio de Madrid o con precios fuera de 10–1.000 €.

## Límites conocidos

- **Una sola instantánea** (sep 2026): no hay estacionalidad, eventos ni evolución temporal.
- **Precio anunciado**, no pagado. El 18,7 % de los anuncios no publica precio y queda fuera.
- **Sin texto ni fotos**: la calidad, las reformas o las vistas no están en las variables. Por eso los pisos de lujo se infraestiman (−204 € de media por encima de 400 €).
- **Datos desequilibrados**: Centro reúne el 40,5 % del test. Distritos como Vicálvaro (9 casos) o Villa de Vallecas (17) tienen tan pocos que su error no es fiable.
- **Una predicción puntual con pocos campos es menos precisa.** `predict` solo exige ubicación, tipo y capacidad, e imputa el resto. Las métricas de arriba corresponden a anuncios con todas las variables.
- El ranking de desviaciones del pipeline favorece por construcción los anuncios **caros**: casi todos los más desviados tienen precios por encima de lo estimado.
- La muestra de `data/sample/` sale de la misma instantánea. Al ejecutar `run` sobre ella, la mayoría de filas están **dentro de la muestra de entrenamiento**, y las desviaciones salen menores que con datos nuevos.
- El sentimiento de las reseñas **no** es una variable del modelo en esta versión: solo enriquece los alojamientos más desviados.
