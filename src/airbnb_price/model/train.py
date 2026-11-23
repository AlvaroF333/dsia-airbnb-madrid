"""Entrenamiento reproducible del modelo de precio, su evaluación y la búsqueda de hiperparámetros.

Flujo:

- ``tune`` (opcional, lento): búsqueda aleatoria con validación cruzada **solo sobre la
  partición de entrenamiento**. Guarda los mejores hiperparámetros en
  ``models/hparams.json`` (versionado).
- ``train``: entrena con esos hiperparámetros (o con los de serie si no hay fichero),
  evalúa en test frente a dos referencias y analiza el error.

El test nunca participa en la elección de hiperparámetros.
"""

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

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
HPARAMS_FILENAME = "hparams.json"
MIN_ROWS = 50
TEST_SIZE = 0.2
CV_FOLDS = 5
# Categorías con menos casos se agrupan en una sola ("infrecuente")
MIN_CATEGORY_FREQUENCY = 10
PRICE_BANDS = (0, 100, 200, 400, np.inf)
PERMUTATION_REPEATS = 5

DEFAULT_HPARAMS: dict[str, float | int] = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}
# Con más de 10.000 filas el boosting usa parada temprana: max_iter es un tope.
SEARCH_SPACE: dict[str, list] = {
    "learning_rate": [0.02, 0.03, 0.05, 0.1],
    "max_iter": [300, 600, 1000, 2000],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [5, 10, 20, 50],
    "l2_regularization": [0.0, 0.1, 1.0],
}
# Ruta de los hiperparámetros del regresor dentro de TransformedTargetRegressor(Pipeline)
_REG_PREFIX = "regressor__reg__"


@dataclass(frozen=True)
class TrainResult:
    model: TransformedTargetRegressor
    metrics: dict
    meta: dict


def _ridge_preprocessor() -> ColumnTransformer:
    """Imputación + escalado y one-hot: lo que necesita un modelo lineal."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                list(NUMERIC_FEATURES),
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0.0,
    )


def _tree_preprocessor() -> ColumnTransformer:
    """Categóricas como enteros para el soporte nativo del gradient boosting.

    Las numéricas pasan tal cual: el boosting trata los nulos por sí mismo. Las
    categorías nuevas o nulas se codifican como ``NaN``, que el modelo trata como
    valor ausente.
    """
    return ColumnTransformer(
        [
            ("num", "passthrough", list(NUMERIC_FEATURES)),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0.0,
    )


def _log_target(regressor) -> TransformedTargetRegressor:
    """Predice log(1 + precio): los precios son muy asimétricos."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def build_model(seed: int, hparams: dict | None = None) -> TransformedTargetRegressor:
    """Modelo final: preprocesado + gradient boosting sobre el logaritmo del precio."""
    params = {**DEFAULT_HPARAMS, **(hparams or {})}
    n_numeric = len(NUMERIC_FEATURES)
    categorical = list(range(n_numeric, n_numeric + len(CATEGORICAL_FEATURES)))
    regressor = HistGradientBoostingRegressor(
        random_state=seed, categorical_features=categorical, **params
    )
    return _log_target(Pipeline([("prep", _tree_preprocessor()), ("reg", regressor)]))


def _baselines() -> dict[str, object]:
    return {
        "baseline_mediana": DummyRegressor(strategy="median"),
        "baseline_ridge": _log_target(
            Pipeline([("prep", _ridge_preprocessor()), ("reg", Ridge(alpha=1.0))])
        ),
    }


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "medae": round(float(median_absolute_error(y_true, y_pred)), 2),
    }


def _split(frame: pd.DataFrame, seed: int):
    if len(frame) < MIN_ROWS:
        raise DataValidationError(
            f"Hacen falta al menos {MIN_ROWS} filas para entrenar (hay {len(frame)})"
        )
    features, target = build_features(frame), frame[TARGET].astype(float)
    empty = [column for column in NUMERIC_FEATURES if features[column].isna().all()]
    if empty:
        raise DataValidationError(
            f"Variables sin ningún valor en los datos de entrenamiento: {empty}. "
            "Revisa si la instantánea de Inside Airbnb ha dejado de publicar esas columnas"
        )
    return train_test_split(features, target, test_size=TEST_SIZE, random_state=seed)


def _band_label(low: float, high: float) -> str:
    return f"> {low:g} €" if np.isinf(high) else f"{low:g}–{high:g} €"


def error_analysis(x_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """Error por tramo de precio anunciado y por distrito, sobre la partición de test.

    El sesgo es la media de (estimado − anunciado): positivo si el modelo sobreestima.
    """
    errors = pd.DataFrame(
        {
            "real": y_test.to_numpy(),
            "error": y_pred - y_test.to_numpy(),
            "distrito": x_test["neighbourhood_group_cleansed"].to_numpy(),
        }
    )
    errors["abs"] = errors["error"].abs()
    errors["tramo"] = pd.cut(errors["real"], PRICE_BANDS, right=True)

    by_band = []
    for interval, group in errors.groupby("tramo", observed=True, sort=True):
        by_band.append(
            {
                "tramo": _band_label(interval.left, interval.right),
                "n": len(group),
                "mae": round(float(group["abs"].mean()), 2),
                "sesgo": round(float(group["error"].mean()), 2),
            }
        )
    by_district = (
        errors.groupby("distrito")
        .agg(n=("abs", "size"), mae=("abs", "mean"), sesgo=("error", "mean"))
        .round(2)
        .sort_values("mae", ascending=False)
        .reset_index()
    )
    return {
        "por_tramo_precio": by_band,
        "por_distrito": [
            {k: (int(v) if k == "n" else v) for k, v in row.items()}
            for row in by_district.to_dict(orient="records")
        ],
    }


def feature_importance(model, x_test: pd.DataFrame, y_test: pd.Series, seed: int) -> list[dict]:
    """Importancia por permutación: cuántos € de MAE se pierden al desordenar cada variable."""
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
    )
    ranking = sorted(
        zip(x_test.columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"variable": name, "aumento_mae": round(float(mean), 2), "desviacion": round(float(std), 2)}
        for name, mean, std in ranking
    ]


def train_model(frame: pd.DataFrame, seed: int = 42, hparams: dict | None = None) -> TrainResult:
    """Entrena y evalúa sobre alojamientos limpios (con ``price`` numérico).

    El modelo devuelto está ajustado solo con la partición de entrenamiento, de
    modo que las métricas de test describen exactamente el modelo que se guarda.
    """
    x_train, x_test, y_train, y_test = _split(frame, seed)
    params = {**DEFAULT_HPARAMS, **(hparams or {})}

    model = build_model(seed, params).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics: dict = {"modelo": _scores(y_test, y_pred)}
    for name, baseline in _baselines().items():
        metrics[name] = _scores(y_test, baseline.fit(x_train, y_train).predict(x_test))

    cv_mae = -cross_val_score(
        build_model(seed, params),
        x_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
    )
    metrics["cv_mae_media"] = round(float(cv_mae.mean()), 2)
    metrics["cv_mae_desviacion"] = round(float(cv_mae.std()), 2)
    metrics["n_train"], metrics["n_test"] = len(x_train), len(x_test)
    metrics["hiperparametros"] = params
    metrics["analisis_error"] = error_analysis(x_test, y_test, y_pred)
    metrics["importancia_variables"] = feature_importance(model, x_test, y_test, seed)

    meta = {
        "modelo_version": __version__,
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "n_filas": len(frame),
        "features": list(FEATURES),
        "target": TARGET,
        "hiperparametros": params,
    }
    return TrainResult(model=model, metrics=metrics, meta=meta)


def tune_hparams(
    frame: pd.DataFrame, seed: int = 42, n_iter: int = 30, cv: int = CV_FOLDS, n_jobs: int = -1
) -> dict:
    """Búsqueda aleatoria de hiperparámetros con validación cruzada sobre train.

    Usa la misma partición que :func:`train_model`, así que el test queda fuera.
    Devuelve los mejores hiperparámetros y un resumen de la búsqueda.
    """
    x_train, _, y_train, _ = _split(frame, seed)
    search = RandomizedSearchCV(
        build_model(seed),
        {_REG_PREFIX + name: values for name, values in SEARCH_SPACE.items()},
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=seed,
        n_jobs=n_jobs,
    )
    search.fit(x_train, y_train)
    best = {name.removeprefix(_REG_PREFIX): value for name, value in search.best_params_.items()}
    baseline_mae = -cross_val_score(
        build_model(seed), x_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    ).mean()
    return {
        "hiperparametros": {name: _plain(value) for name, value in best.items()},
        "cv_mae": round(float(-search.best_score_), 2),
        "cv_mae_por_defecto": round(float(baseline_mae), 2),
        "n_iter": n_iter,
        "cv": cv,
        "seed": seed,
        "espacio": SEARCH_SPACE,
        "n_train": len(x_train),
    }


def _plain(value):
    """Convierte escalares de numpy a tipos de Python (para JSON)."""
    return value.item() if hasattr(value, "item") else value


def load_hparams(model_dir: Path) -> dict | None:
    """Hiperparámetros guardados por ``tune``; ``None`` si no hay fichero."""
    path = Path(model_dir) / HPARAMS_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["hiperparametros"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_tune(
    input_path: Path, model_dir: Path, seed: int = 42, n_iter: int = 30, n_jobs: int = -1
) -> dict:
    """Job de búsqueda: guarda ``hparams.json`` junto al modelo."""
    from airbnb_price.data.loader import load_listings

    frame = load_listings(input_path)
    if TARGET not in frame.columns:
        raise DataValidationError(f"El fichero de entrenamiento no tiene la columna {TARGET!r}")
    result = tune_hparams(frame, seed=seed, n_iter=n_iter, n_jobs=n_jobs)
    result["buscado_en"] = datetime.now(UTC).isoformat(timespec="seconds")
    result["datos_sha256"] = _sha256(input_path)
    _write_json(Path(model_dir) / HPARAMS_FILENAME, result)
    logger.info(
        "tune cv_mae=%s (por defecto %s) hparams=%s",
        result["cv_mae"],
        result["cv_mae_por_defecto"],
        result["hiperparametros"],
    )
    return result


def run_train(input_path: Path, model_dir: Path, metrics_path: Path, seed: int = 42) -> dict:
    """Job de entrenamiento: lee el CSV limpio y guarda modelo, metadatos y métricas."""
    from airbnb_price.data.loader import load_listings

    frame = load_listings(input_path)
    if TARGET not in frame.columns:
        raise DataValidationError(f"El fichero de entrenamiento no tiene la columna {TARGET!r}")
    hparams = load_hparams(model_dir)
    logger.info("hiperparametros=%s", "hparams.json" if hparams else "por defecto")
    result = train_model(frame, seed=seed, hparams=hparams)

    model_dir.mkdir(parents=True, exist_ok=True)
    # Comprimido: el modelo de árboles pasa de ~6 MB a una fracción y se puede versionar
    joblib.dump(result.model, model_dir / MODEL_FILENAME, compress=3)
    meta = {
        **result.meta,
        "entrenado_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "datos": Path(input_path).name,
        "datos_sha256": _sha256(input_path),
        "metricas_test": result.metrics["modelo"],
    }
    _write_json(model_dir / META_FILENAME, meta)
    _write_json(metrics_path, result.metrics)

    for name in ("modelo", "baseline_ridge", "baseline_mediana"):
        logger.info("%s %s", name, result.metrics[name])
    logger.info("model=%s metrics=%s", model_dir / MODEL_FILENAME, metrics_path)
    return result.metrics
