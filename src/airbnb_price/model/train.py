"""Entrenamiento reproducible del modelo de precio y su evaluación."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airbnb_price import __version__
from airbnb_price.exceptions import DataValidationError
from airbnb_price.features.build import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
)

logger = logging.getLogger("airbnb_price.train")

MODEL_FILENAME = "model.joblib"
META_FILENAME = "model_meta.json"
MIN_ROWS = 50
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class TrainResult:
    model: TransformedTargetRegressor
    metrics: dict
    meta: dict


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=0.0,
    )


def _log_target(regressor) -> TransformedTargetRegressor:
    """Predice log(1 + precio): los precios son muy asimétricos."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def build_model(seed: int) -> TransformedTargetRegressor:
    """Modelo final: preprocesado + gradient boosting sobre el logaritmo del precio."""
    return _log_target(
        Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                ("reg", HistGradientBoostingRegressor(random_state=seed)),
            ]
        )
    )


def _baselines() -> dict[str, object]:
    return {
        "baseline_mediana": DummyRegressor(strategy="median"),
        "baseline_ridge": _log_target(
            Pipeline([("prep", _preprocessor(scale=True)), ("reg", Ridge(alpha=1.0))])
        ),
    }


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "medae": round(float(median_absolute_error(y_true, y_pred)), 2),
    }


def train_model(frame: pd.DataFrame, seed: int = 42) -> TrainResult:
    """Entrena y evalúa sobre alojamientos limpios (con ``price`` numérico).

    El modelo devuelto está ajustado solo con la partición de entrenamiento, de
    modo que las métricas de test describen exactamente el modelo que se guarda.
    """
    if len(frame) < MIN_ROWS:
        raise DataValidationError(
            f"Hacen falta al menos {MIN_ROWS} filas para entrenar (hay {len(frame)})"
        )
    features, target = build_features(frame), frame[TARGET].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=seed
    )

    model = build_model(seed).fit(x_train, y_train)
    metrics: dict = {"modelo": _scores(y_test, model.predict(x_test))}
    for name, baseline in _baselines().items():
        metrics[name] = _scores(y_test, baseline.fit(x_train, y_train).predict(x_test))

    cv_mae = -cross_val_score(
        build_model(seed), x_train, y_train, cv=CV_FOLDS, scoring="neg_mean_absolute_error"
    )
    metrics["cv_mae_media"] = round(float(cv_mae.mean()), 2)
    metrics["cv_mae_desviacion"] = round(float(cv_mae.std()), 2)
    metrics["n_train"], metrics["n_test"] = len(x_train), len(x_test)

    meta = {
        "modelo_version": __version__,
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "n_filas": len(frame),
        "features": list(FEATURES),
        "target": TARGET,
    }
    return TrainResult(model=model, metrics=metrics, meta=meta)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_train(input_path: Path, model_dir: Path, metrics_path: Path, seed: int = 42) -> dict:
    """Job de entrenamiento: lee el CSV limpio y guarda modelo, metadatos y métricas."""
    from airbnb_price.data.loader import load_listings

    frame = load_listings(input_path)
    if TARGET not in frame.columns:
        raise DataValidationError(f"El fichero de entrenamiento no tiene la columna {TARGET!r}")
    result = train_model(frame, seed=seed)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / MODEL_FILENAME)
    meta = {
        **result.meta,
        "entrenado_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "datos": str(input_path),
        "datos_sha256": _sha256(input_path),
        "metricas_test": result.metrics["modelo"],
    }
    (model_dir / META_FILENAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False) + "\n")

    for name in ("modelo", "baseline_ridge", "baseline_mediana"):
        logger.info("%s %s", name, result.metrics[name])
    logger.info("model=%s metrics=%s", model_dir / MODEL_FILENAME, metrics_path)
    return result.metrics
