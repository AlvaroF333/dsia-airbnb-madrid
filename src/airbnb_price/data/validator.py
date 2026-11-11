"""Contrato de entrada y reglas de limpieza de los alojamientos.

Funciones puras: reciben un DataFrame y devuelven otro, sin disco ni red.
"""

from __future__ import annotations

import pandas as pd

from airbnb_price.exceptions import DataValidationError

REQUIRED_COLUMNS = (
    "id",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "amenities",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
    "price",
)

PRICE_MIN, PRICE_MAX = 10.0, 1000.0
ACCOMMODATES_MIN, ACCOMMODATES_MAX = 1, 16
# Caja que contiene el municipio de Madrid
LAT_MIN, LAT_MAX = 40.30, 40.65
LON_MIN, LON_MAX = -3.90, -3.50

# Orden de evaluación: cada fila rechazada se cuenta en la primera regla que incumple.
REJECT_REASONS = (
    "price_missing",
    "duplicate_id",
    "price_out_of_range",
    "accommodates_out_of_range",
    "coords_out_of_bounds",
)


def validate_schema(frame: pd.DataFrame) -> None:
    """Comprueba que están todas las columnas obligatorias."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas obligatorias: {missing}")


def parse_price(values: pd.Series) -> pd.Series:
    """Convierte precios como ``"$1,234.00"`` a número; lo no interpretable queda nulo."""
    text = values.astype("string").str.replace(r"[$€,\s]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def clean_listings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica las reglas de limpieza.

    Devuelve los alojamientos válidos (con ``price`` numérico) y el número de
    filas rechazadas por cada regla de :data:`REJECT_REASONS`.
    """
    validate_schema(frame)
    data = frame.copy()
    data["price"] = parse_price(data["price"])
    accommodates = pd.to_numeric(data["accommodates"], errors="coerce")
    latitude = pd.to_numeric(data["latitude"], errors="coerce")
    longitude = pd.to_numeric(data["longitude"], errors="coerce")

    failures = {
        "price_missing": data["price"].isna(),
        "duplicate_id": data["id"].duplicated(keep="first"),
        "price_out_of_range": ~data["price"].between(PRICE_MIN, PRICE_MAX),
        "accommodates_out_of_range": ~accommodates.between(ACCOMMODATES_MIN, ACCOMMODATES_MAX),
        "coords_out_of_bounds": ~(
            latitude.between(LAT_MIN, LAT_MAX) & longitude.between(LON_MIN, LON_MAX)
        ),
    }

    rejected = pd.Series(False, index=data.index)
    rejects: dict[str, int] = {}
    for reason in REJECT_REASONS:
        newly_rejected = failures[reason] & ~rejected
        rejects[reason] = int(newly_rejected.sum())
        rejected |= newly_rejected

    return data.loc[~rejected].reset_index(drop=True), rejects


PREDICT_REQUIRED = ("latitude", "longitude", "room_type", "accommodates")
PREDICT_OPTIONAL = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column not in PREDICT_REQUIRED and column not in ("price", "neighbourhood_cleansed")
)


def validate_prediction_input(records: list[dict]) -> pd.DataFrame:
    """Contrato de entrada de una predicción (CLI ``predict`` y, en el III, la API).

    Exige ubicación, tipo y capacidad; el resto de variables es opcional y el
    modelo las imputa. Rechaza capacidades o coordenadas fuera de rango.
    """
    if not records or not all(isinstance(record, dict) for record in records):
        raise DataValidationError("La entrada debe ser un objeto JSON o una lista de objetos")
    frame = pd.DataFrame(records)
    missing = [column for column in PREDICT_REQUIRED if column not in frame.columns]
    if missing or frame[list(PREDICT_REQUIRED)].isna().any().any():
        raise DataValidationError(f"Faltan campos obligatorios: {list(PREDICT_REQUIRED)}")
    for column in PREDICT_OPTIONAL:
        if column not in frame.columns:
            frame[column] = None
    if "id" not in frame.columns or frame["id"].isna().any():
        frame["id"] = [str(i) for i in range(len(frame))]
    frame["id"] = frame["id"].astype(str)

    accommodates = pd.to_numeric(frame["accommodates"], errors="coerce")
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    if not accommodates.between(ACCOMMODATES_MIN, ACCOMMODATES_MAX).all():
        raise DataValidationError(
            f"accommodates debe estar entre {ACCOMMODATES_MIN} y {ACCOMMODATES_MAX}"
        )
    if not (latitude.between(LAT_MIN, LAT_MAX) & longitude.between(LON_MIN, LON_MAX)).all():
        raise DataValidationError("Las coordenadas están fuera del municipio de Madrid")
    return frame


MAX_REVIEW_CHARS = 500


def clean_review_text(values: pd.Series) -> pd.Series:
    """Quita etiquetas HTML (``<br/>``) y espacios repetidos, y recorta a 500 caracteres."""
    text = values.astype("string").str.replace(r"<[^>]+>", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text.str.slice(0, MAX_REVIEW_CHARS)


def clean_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    """Reseñas con texto útil, de la más reciente a la más antigua dentro de cada alojamiento."""
    data = frame.copy()
    data["comments"] = clean_review_text(data["comments"])
    data = data[data["comments"].fillna("").str.contains(r"\w", regex=True)]
    data = data.sort_values(["listing_id", "date"], ascending=[True, False], kind="stable")
    return data.reset_index(drop=True)
