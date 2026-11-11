"""Extremo a extremo por la CLI, con datos sintéticos en ``tmp_path`` y sin red."""

import json

import pandas as pd
import pytest

from airbnb_price.cli import main
from airbnb_price.exceptions import EXIT_INVALID_INPUT, EXIT_OK
from airbnb_price.pipeline import PREDICTION_COLUMNS
from tests.conftest import make_synthetic_listings
from tests.unit.test_pipeline import FailingClient

pytestmark = pytest.mark.integration


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Proyecto mínimo: muestra de alojamientos y reseñas, y modelo entrenado."""
    sample = tmp_path / "data" / "sample"
    sample.mkdir(parents=True)
    listings = make_synthetic_listings(n=200, seed=1)
    listings.to_csv(sample / "listings_sample.csv", index=False)
    pd.DataFrame(
        {
            "listing_id": [str(i) for i in range(0, 200, 2)],
            "date": ["2026-09-01"] * 100,
            "comments": ["Piso limpio y genial", "Muy ruidoso y sucio"] * 50,
        }
    ).to_csv(sample / "reviews_sample.csv", index=False)
    monkeypatch.setenv("AIRBNB_PRICE_HOME", str(tmp_path))

    assert main(["ingest", "--input", str(sample / "listings_sample.csv")]) == EXIT_OK
    assert main(["train"]) == EXIT_OK
    return tmp_path


def test_run_end_to_end_with_mock(home):
    assert main(["run", "--top-n", "10"]) == EXIT_OK

    predictions = pd.read_csv(home / "reports" / "predictions.csv", dtype={"id": str})
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 200
    assert (predictions["precio_estimado"] > 0).all()
    enriched = predictions.dropna(subset=["explicacion"])
    assert len(enriched) == 10
    assert set(enriched["provider"]) == {"mock"}
    assert enriched["sentimiento"].dropna().between(0, 1).all()
    assert enriched["sentimiento"].notna().any()

    metrics = json.loads((home / "reports" / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["ai_provider"] == "mock"
    assert metrics["ai_errores"] == 0
    assert metrics["filas_enriquecidas"] == 10
    assert {"ingesta_s", "prediccion_s", "ia_s", "total_s"} <= metrics["duracion"].keys()


def test_run_degrades_when_ai_always_fails(home, monkeypatch, capsys):
    monkeypatch.setattr("airbnb_price.pipeline.get_client", lambda settings: FailingClient())

    assert main(["run", "--top-n", "10"]) == EXIT_OK

    predictions = pd.read_csv(home / "reports" / "predictions.csv")
    assert len(predictions) == 200
    assert predictions["explicacion"].isna().all()
    metrics = json.loads((home / "reports" / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["ai_errores"] > 0
    assert metrics["ai_desactivada"] is True
    assert "errores de IA" in capsys.readouterr().err


def test_predict_from_json(home, tmp_path, capsys):
    request = tmp_path / "piso.json"
    request.write_text(
        json.dumps(
            {
                "id": "demo",
                "latitude": 40.4168,
                "longitude": -3.7038,
                "room_type": "Entire home/apt",
                "accommodates": 4,
            }
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert main(["predict", "--input", str(request)]) == EXIT_OK

    out = json.loads(capsys.readouterr().out)
    assert out[0]["id"] == "demo"
    assert out[0]["precio_estimado"] > 0
    assert out[0]["modelo_version"]


def test_predict_without_model_returns_exit_2(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRBNB_PRICE_HOME", str(tmp_path))
    request = tmp_path / "piso.json"
    request.write_text(
        '{"latitude": 40.42, "longitude": -3.7, "room_type": "Private room", "accommodates": 1}',
        encoding="utf-8",
    )

    assert main(["predict", "--input", str(request)]) == EXIT_INVALID_INPUT


def test_run_with_missing_reviews_file_returns_exit_2(home):
    assert main(["run", "--reviews", str(home / "no_existe.csv")]) == EXIT_INVALID_INPUT


def test_run_without_reviews_file_still_predicts(home):
    (home / "data" / "sample" / "reviews_sample.csv").unlink()

    assert main(["run", "--top-n", "3"]) == EXIT_OK

    predictions = pd.read_csv(home / "reports" / "predictions.csv")
    assert predictions["sentimiento"].isna().all()
    assert predictions["explicacion"].notna().sum() == 3


def test_sentiment_command_uses_mock(capsys):
    assert main(["sentiment", "--text", "Muy limpio", "--text", "Horrible"]) == EXIT_OK

    out = json.loads(capsys.readouterr().out)
    assert [r["label"] for r in out] == ["positive", "negative"]
    assert {r["provider"] for r in out} == {"mock"}
