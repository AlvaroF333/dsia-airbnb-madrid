"""Variables del modelo a partir de los alojamientos limpios.

Funciones puras: DataFrame -> DataFrame, sin disco ni red.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "n_amenities",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "is_superhost",
    "dist_sol_km",
    "latitude",
    "longitude",
)
CATEGORICAL_FEATURES = ("room_type", "neighbourhood_group_cleansed")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"

SOL_LAT, SOL_LON = 40.4168, -3.7038
EARTH_RADIUS_KM = 6371.0


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


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las columnas de :data:`FEATURES`, en ese orden y con el mismo índice."""
    out = pd.DataFrame(index=frame.index)
    for column in ("accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews"):
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    out["review_scores_rating"] = pd.to_numeric(frame["review_scores_rating"], errors="coerce")
    out["bathrooms"] = parse_bathrooms(frame["bathrooms_text"])
    out["n_amenities"] = frame["amenities"].map(count_amenities)
    out["is_superhost"] = (
        frame["host_is_superhost"].astype("string").str.lower().isin(["t", "true"]).astype(int)
    )
    out["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    out["dist_sol_km"] = haversine_km(out["latitude"], out["longitude"], SOL_LAT, SOL_LON)
    for column in CATEGORICAL_FEATURES:
        out[column] = frame[column].astype("object").where(frame[column].notna(), "desconocido")
    return out[list(FEATURES)].astype({column: float for column in NUMERIC_FEATURES})
