"""Variables del modelo a partir de los alojamientos limpios.

Funciones puras: DataFrame -> DataFrame, sin disco ni red.

Columnas del ``listings`` excluidas a propósito porque **filtran el precio**
(se calculan a partir de él): ``estimated_revenue_l365d`` (ocupación × precio) y
``price_quote_*`` (presupuesto de una estancia concreta). Usarlas daría métricas
excelentes en test e inútiles en producción, donde el precio es lo que se busca.

También se excluyen ``host_since`` e ``instant_bookable``: en la instantánea del
17 sep 2026 están vacías en el 100 % de las filas. La antigüedad del anfitrión se
toma de ``hosts_time_as_host_years`` y ``hosts_time_as_host_months``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Columnas que se leen como número sin transformar
RAW_NUMERIC = (
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "availability_30",
    "availability_365",
    "host_listings_count",
    "calculated_host_listings_count",
    "latitude",
    "longitude",
)
NUMERIC_FEATURES = RAW_NUMERIC + (
    "bathrooms",
    "n_amenities",
    "is_superhost",
    "host_years",
    "dist_sol_km",
)
CATEGORICAL_FEATURES = (
    "room_type",
    "property_type",
    "neighbourhood_group_cleansed",
    "neighbourhood_cleansed",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"
UNKNOWN_CATEGORY = "desconocido"

SOL_LAT, SOL_LON = 40.4168, -3.7038
EARTH_RADIUS_KM = 6371.0
MONTHS_PER_YEAR = 12


def parse_bathrooms(values: pd.Series) -> pd.Series:
    """Extrae el número de baños de textos como ``"1.5 shared baths"``.

    ``"Half-bath"`` (sin número) cuenta como 0,5.
    """
    text = values.astype("string").str.lower()
    number = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    return number.fillna(text.str.contains("half", na=False).map({True: 0.5, False: np.nan}))


def count_amenities(value: object) -> int:
    """Número de servicios de la lista JSON de ``amenities``; 0 si no se puede leer."""
    if not isinstance(value, str):
        return 0
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return 0
    return len(items) if isinstance(items, list) else 0


def haversine_km(lat: pd.Series, lon: pd.Series, lat0: float, lon0: float) -> pd.Series:
    """Distancia en km de cada punto a ``(lat0, lon0)``."""
    lat_r, lon_r = np.radians(lat.astype(float)), np.radians(lon.astype(float))
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    a = (
        np.sin((lat_r - lat0_r) / 2) ** 2
        + np.cos(lat_r) * np.cos(lat0_r) * np.sin((lon_r - lon0_r) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def parse_flag(values: pd.Series) -> pd.Series:
    """``"t"``/``"true"`` → 1; cualquier otro valor (incluido nulo) → 0."""
    return values.astype("string").str.lower().isin(["t", "true"]).astype(int)


def host_years(years: pd.Series, months: pd.Series) -> pd.Series:
    """Antigüedad del anfitrión en años: ``años + meses / 12`` (meses = resto del año).

    Nula si faltan los años; si solo faltan los meses, cuentan como 0.
    """
    whole = pd.to_numeric(years, errors="coerce")
    extra = pd.to_numeric(months, errors="coerce").fillna(0)
    return whole + extra / MONTHS_PER_YEAR


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """La columna ``name`` o, si no está (p. ej. en una predicción puntual), una serie nula."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las columnas de :data:`FEATURES`, en ese orden y con el mismo índice.

    Las columnas que falten en la entrada quedan nulas y el modelo las imputa.
    """
    out = pd.DataFrame(index=frame.index)
    for column in RAW_NUMERIC:
        out[column] = pd.to_numeric(_column(frame, column), errors="coerce")
    out["bathrooms"] = parse_bathrooms(_column(frame, "bathrooms_text"))
    out["n_amenities"] = _column(frame, "amenities").map(count_amenities)
    out["is_superhost"] = parse_flag(_column(frame, "host_is_superhost"))
    out["host_years"] = host_years(
        _column(frame, "hosts_time_as_host_years"), _column(frame, "hosts_time_as_host_months")
    )
    out["dist_sol_km"] = haversine_km(out["latitude"], out["longitude"], SOL_LAT, SOL_LON)
    for column in CATEGORICAL_FEATURES:
        values = _column(frame, column)
        out[column] = values.astype("object").where(values.notna(), UNKNOWN_CATEGORY)
    return out[list(FEATURES)].astype({column: float for column in NUMERIC_FEATURES})
