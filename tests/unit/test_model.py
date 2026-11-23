import json

import numpy as np
import pytest

from airbnb_price.exceptions import DataValidationError, ModelNotFoundError
from airbnb_price.model.predict import load_model, predict
from airbnb_price.model.train import (
    DEFAULT_HPARAMS,
    META_FILENAME,
    MODEL_FILENAME,
    SEARCH_SPACE,
    load_hparams,
    run_train,
    run_tune,
    train_model,
    tune_hparams,
)
from tests.conftest import make_listing


@pytest.fixture(scope="module")
def trained(synthetic_listings):
    return train_model(synthetic_listings, seed=42)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, synthetic_listings):
    """Modelo pequeño entrenado y guardado en disco con el job real."""
    base = tmp_path_factory.mktemp("modelo")
    csv = base / "listings_clean.csv"
    synthetic_listings.to_csv(csv, index=False)
    run_train(csv, base / "models", base / "reports" / "model_metrics.json")
    return base


def test_model_beats_both_baselines(trained):
    metrics = trained.metrics

    assert metrics["modelo"]["mae"] < metrics["baseline_mediana"]["mae"]
    assert metrics["modelo"]["r2"] > 0.5
    assert {"baseline_ridge", "cv_mae_media", "n_train", "n_test"} <= metrics.keys()


def test_training_is_reproducible(synthetic_listings, trained):
    again = train_model(synthetic_listings, seed=42)

    assert again.metrics == trained.metrics


def test_too_few_rows_raises_domain_error(synthetic_listings):
    with pytest.raises(DataValidationError):
        train_model(synthetic_listings.head(10))


def test_run_train_writes_model_meta_and_metrics(model_dir):
    assert (model_dir / "models" / MODEL_FILENAME).is_file()
    meta = json.loads((model_dir / "models" / META_FILENAME).read_text())
    assert len(meta["datos_sha256"]) == 64
    assert meta["n_filas"] == 300
    metrics = json.loads((model_dir / "reports" / "model_metrics.json").read_text())
    assert metrics["modelo"] == meta["metricas_test"]


def test_load_saved_model_and_predict(model_dir, synthetic_listings):
    model, meta = load_model(model_dir / "models")

    prices = predict(model, synthetic_listings.head(5).drop(columns=["price"]))

    assert prices.shape == (5,)
    assert np.all(prices > 0)
    assert meta["features"]


def test_bigger_listing_costs_more(model_dir):
    import pandas as pd

    model, _ = load_model(model_dir / "models")
    frame = pd.DataFrame([make_listing(accommodates=1), make_listing(accommodates=8)])

    small, big = predict(model, frame)

    assert big > small


def test_predict_handles_unknown_category_and_nulls(model_dir):
    import pandas as pd

    model, _ = load_model(model_dir / "models")
    frame = pd.DataFrame([make_listing(room_type="Hotel room", bedrooms=None)])

    assert predict(model, frame)[0] > 0


def test_load_model_missing_raises_domain_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train"):
        load_model(tmp_path)


def test_metrics_include_error_analysis_and_importance(trained):
    analysis = trained.metrics["analisis_error"]

    assert sum(band["n"] for band in analysis["por_tramo_precio"]) == trained.metrics["n_test"]
    assert {"tramo", "n", "mae", "sesgo"} <= analysis["por_tramo_precio"][0].keys()
    maes = [row["mae"] for row in analysis["por_distrito"]]
    assert maes == sorted(maes, reverse=True)

    importance = trained.metrics["importancia_variables"]
    assert len(importance) == len(trained.meta["features"])
    # El precio sintético depende sobre todo de la capacidad
    assert importance[0]["variable"] == "accommodates"


def test_custom_hparams_are_used_and_recorded(synthetic_listings):
    result = train_model(synthetic_listings, seed=42, hparams={"max_iter": 20})

    assert result.model.regressor_.named_steps["reg"].max_iter == 20
    assert result.meta["hiperparametros"]["max_iter"] == 20
    assert result.metrics["hiperparametros"]["learning_rate"] == DEFAULT_HPARAMS["learning_rate"]


def test_tune_never_uses_test_rows(synthetic_listings):
    result = tune_hparams(synthetic_listings, seed=42, n_iter=2, cv=2, n_jobs=1)

    assert result["n_train"] == int(len(synthetic_listings) * 0.8)
    assert set(result["hiperparametros"]) == set(SEARCH_SPACE)
    for name, value in result["hiperparametros"].items():
        assert value in SEARCH_SPACE[name]


def test_run_tune_writes_hparams_used_by_train(tmp_path, synthetic_listings, monkeypatch):
    csv = tmp_path / "listings_clean.csv"
    synthetic_listings.to_csv(csv, index=False)
    models = tmp_path / "models"
    monkeypatch.setattr("airbnb_price.model.train.SEARCH_SPACE", {"max_iter": [30]})

    run_tune(csv, models, n_iter=1, n_jobs=1)
    assert load_hparams(models) == {"max_iter": 30}

    run_train(csv, models, tmp_path / "reports" / "model_metrics.json")
    meta = json.loads((models / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["hiperparametros"]["max_iter"] == 30


def test_load_hparams_without_file_returns_none(tmp_path):
    assert load_hparams(tmp_path) is None


def test_empty_feature_column_raises_domain_error(synthetic_listings):
    """Inside Airbnb a veces deja de publicar una columna: error claro, no un traceback."""
    frame = synthetic_listings.assign(availability_365=None)

    with pytest.raises(DataValidationError, match="availability_365"):
        train_model(frame)
