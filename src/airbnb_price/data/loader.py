"""Lectura de los ficheros de Inside Airbnb (única capa de datos que toca disco)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from airbnb_price.exceptions import DataLoadError, DataValidationError

REVIEW_COLUMNS = ("listing_id", "date", "comments")


def load_listings(path: Path) -> pd.DataFrame:
    """Lee el CSV de alojamientos (``.csv`` o ``.csv.gz``).

    ``id`` se lee como texto para no perder precisión en identificadores largos.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"No existe el fichero de entrada: {path}")
    try:
        return pd.read_csv(path, dtype={"id": "string"}, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
        raise DataLoadError(f"No se puede leer {path}: {exc}") from exc


def load_json_records(path: Path) -> list[dict]:
    """Lee un objeto JSON o una lista de objetos (entrada de ``predict``)."""
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"No existe el fichero de entrada: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"{path} no es un JSON válido: {exc}") from exc
    return [data] if isinstance(data, dict) else data


def load_reviews(path: Path) -> pd.DataFrame:
    """Lee el CSV de reseñas y devuelve solo :data:`REVIEW_COLUMNS`.

    ``listing_id`` se lee como texto para poder cruzarlo con ``id`` de los alojamientos.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"No existe el fichero de reseñas: {path}")
    try:
        frame = pd.read_csv(path, dtype={"listing_id": "string", "comments": "string"})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
        raise DataLoadError(f"No se puede leer {path}: {exc}") from exc
    missing = [column for column in REVIEW_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas en las reseñas: {missing}")
    return frame[list(REVIEW_COLUMNS)]
