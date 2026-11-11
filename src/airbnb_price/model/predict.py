"""Carga del modelo guardado e inferencia."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airbnb_price.exceptions import ModelNotFoundError
from airbnb_price.features.build import build_features
from airbnb_price.model.train import META_FILENAME, MODEL_FILENAME


def load_model(model_dir: Path):
    """Carga el modelo y sus metadatos de ``model_dir``."""
    model_path = Path(model_dir) / MODEL_FILENAME
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"No hay modelo en {model_path}. Entrena antes con: python -m airbnb_price train"
        )
    meta_path = Path(model_dir) / META_FILENAME
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return joblib.load(model_path), meta


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    """Precio estimado por noche (en euros, 2 decimales) de cada alojamiento."""
    return np.round(model.predict(build_features(frame)), 2)
