# Model card: estimador del precio por noche (Airbnb Madrid)

| Campo | Valor |
| --- | --- |
| Versión | `0.2.0` (`model_meta.json`) |
| Tipo | `TransformedTargetRegressor(log1p/expm1)` sobre `Pipeline(ColumnTransformer + HistGradientBoostingRegressor)` |
| Librería | scikit-learn 1.9.1, semilla 42 |
| Entrenado | 2026-11-11 (UTC), `python -m airbnb_price train` |
| Datos | Inside Airbnb, Madrid, instantánea del 17 sep 2026 (CC BY 4.0) |
| Filas | 23.109 en origen → 18.620 tras la ingesta (80 % train, 20 % test) |
| Huella de los datos | SHA-256 `88d606e5…a08a0` de `listings_clean.csv` |

## Variables

- **Numéricas** (imputación por mediana): `accommodates`, `bedrooms`, `beds`, `bathrooms` (extraído de `bathrooms_text`; "half-bath" = 0,5), `n_amenities`, `minimum_nights`, `number_of_reviews`, `review_scores_rating`, `is_superhost`, `dist_sol_km` (distancia haversine a la Puerta del Sol), `latitude`, `longitude`.
- **Categóricas** (one-hot, categorías nuevas ignoradas): `room_type`, `neighbourhood_group_cleansed` (21 distritos).
- **Objetivo**: `price` anunciado, en euros por noche, entre 10 y 1.000 €. El modelo aprende `log(1 + precio)` porque la distribución es muy asimétrica (mediana 150 €, media 180,8 €).

## Métricas

Partición de test (3.724 alojamientos), copiadas de `reports/model_metrics.json`:

| Modelo | MAE (€) | RMSE (€) | R² | Error absoluto mediano (€) |
| --- | --- | --- | --- | --- |
| **HistGradientBoosting (modelo)** | **54,21** | **96,70** | **0,5257** | **29,21** |
| Ridge con el mismo preprocesado | 63,65 | 112,89 | 0,3537 | 35,61 |
| Mediana (`DummyRegressor`) | 92,57 | 144,10 | −0,0531 | 63,85 |

Validación cruzada de 5 pliegues sobre train: MAE medio **53,28 €**, desviación **1,56 €**.

Lectura: el modelo reduce el MAE un 41 % frente a la mediana y un 15 % frente a Ridge. La CV coincide con el test, así que el resultado no depende de una partición afortunada. El RMSE casi duplica el MAE: hay pocos errores muy grandes, casi siempre en alojamientos caros.

### Error por tramo de precio (mismo test)

| Precio anunciado | n | MAE (€) | Sesgo medio (estimado − anunciado, €) |
| --- | --- | --- | --- |
| ≤ 100 € | 1.052 | 23,6 | +18,6 |
| 100–200 € | 1.520 | 33,6 | +13,6 |
| 200–400 € | 891 | 69,5 | −45,5 |
| > 400 € | 261 | 245,4 | −241,3 |

El modelo **sobreestima los alojamientos baratos e infraestima mucho los caros** (regresión a la media). Por distrito, el MAE es mayor donde los precios son altos: Salamanca 72,5 € y Centro 63,4 €, frente a 30–45 € en la mayoría de distritos periféricos.

## Uso previsto

- Estimar un precio de referencia por noche para un alojamiento de Madrid a partir de sus características.
- Detectar anuncios cuyo precio se desvía mucho de lo esperable, como apoyo a la revisión humana (el pipeline `run` lo hace con los 20 más desviados).

## Usos no previstos

- Fijar precios de forma automática o tomar decisiones sobre personas (anfitriones o huéspedes).
- Estimar ingresos reales: el dato es el precio **anunciado**, no el pagado.
- Usarlo fuera del municipio de Madrid o con precios fuera de 10–1.000 €.

## Límites conocidos

- **Una sola instantánea** (sep 2026): sin estacionalidad, eventos ni evolución temporal.
- **Precio anunciado**, no pagado; el 18,7 % de los anuncios no publica precio y queda fuera.
- **Sin texto ni fotos**: calidad, reformas o vistas no están en las variables; por eso los pisos de lujo se infraestiman (−241 € de media por encima de 400 €).
- **Datos desequilibrados**: Centro reúne el 41,7 % de los anuncios; distritos como Vicálvaro (9 casos en test) o Moratalaz (23) tienen tan pocos que su error no es fiable.
- El ranking de desviaciones del pipeline favorece por construcción los anuncios **caros**: casi todos los más desviados son precios por encima de lo estimado.
- La muestra de `data/sample/` sale de la misma instantánea, de modo que al ejecutar `run` sobre ella la mayoría de filas son **dentro de muestra** y las desviaciones salen algo menores que con datos nuevos.
- El sentimiento de las reseñas **no** es una variable del modelo en esta versión: solo enriquece los alojamientos más desviados.
