import json

import numpy as np
import pytest

from airbnb_price.exceptions import DataValidationError, ModelNotFoundError
from airbnb_price.model.predict import load_model, predict
from airbnb_price.model.train import META_FILENAME, MODEL_FILENAME, run_train, train_model
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
