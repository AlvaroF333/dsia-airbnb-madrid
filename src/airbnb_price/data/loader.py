"""Lectura de los ficheros de Inside Airbnb (única capa de datos que toca disco)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from airbnb_price.exceptions import DataLoadError


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
