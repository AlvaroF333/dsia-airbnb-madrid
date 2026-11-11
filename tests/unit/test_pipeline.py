import pandas as pd
import pytest

from airbnb_price.ai.mock import MockClient
from airbnb_price.exceptions import AIClientError
from airbnb_price.pipeline import (
    MAX_CONSECUTIVE_AI_ERRORS,
    enrich,
    reviews_by_listing,
    select_outliers,
)


class FailingClient(MockClient):
    """Cliente de IA que siempre falla, como un proveedor caído."""

    name = "hf"

    def __init__(self) -> None:
        self.calls = 0

    def sentiment(self, texts):
        self.calls += 1
        raise AIClientError("proveedor caído")

    def explain(self, context):
        self.calls += 1
        raise AIClientError("proveedor caído")


@pytest.fixture
def predictions():
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "barrio": ["Sol"] * 4,
            "precio": [100.0, 300.0, 50.0, 120.0],
            "precio_estimado": [100.0, 100.0, 150.0, 100.0],
            "room_type": ["Entire home/apt"] * 4,
            "accommodates": [2] * 4,
            "review_scores_rating": [4.5] * 4,
        }
    )
    frame["desviacion"] = frame["precio"] - frame["precio_estimado"]
    for column in ("sentimiento", "n_resenas", "explicacion", "provider", "model"):
        frame[column] = pd.Series([None] * len(frame), dtype="object")
    return frame


def test_select_outliers_uses_absolute_deviation(predictions):
    outliers = select_outliers(predictions, top_n=2)

    assert predictions.loc[outliers, "id"].tolist() == ["b", "c"]


def test_select_outliers_handles_large_and_negative_top_n(predictions):
    assert len(select_outliers(predictions, top_n=99)) == 4
    assert len(select_outliers(predictions, top_n=-1)) == 0


def test_reviews_by_listing_keeps_five_most_recent():
    reviews = pd.DataFrame(
        {
            "listing_id": ["b"] * 7 + ["x"],
            "date": [f"2026-01-0{i}" for i in range(1, 8)] + ["2026-01-01"],
            "comments": [f"texto {i}" for i in range(1, 8)] + ["otro"],
        }
    )

    texts = reviews_by_listing(reviews, {"b"})

    assert texts == {"b": ["texto 7", "texto 6", "texto 5", "texto 4", "texto 3"]}
    assert reviews_by_listing(None, {"b"}) == {}


def test_enrich_fills_only_outliers_and_tells_the_provider(predictions):
    outliers = select_outliers(predictions, top_n=2)

    stats = enrich(predictions, outliers, {"b": ["Genial y limpio"]}, MockClient())

    enriched = predictions.dropna(subset=["explicacion"])
    assert enriched["id"].tolist() == ["b", "c"]
    assert set(enriched["provider"]) == {"mock"}
    assert predictions.loc[1, "sentimiento"] == 1.0
    assert predictions.loc[2, "sentimiento"] is None
    assert stats.calls == 3
    assert stats.errors == 0


def test_enrich_degrades_and_stops_after_consecutive_errors(predictions):
    client = FailingClient()

    stats = enrich(predictions, predictions.index, {}, client)

    assert predictions["explicacion"].isna().all()
    assert stats.errors == MAX_CONSECUTIVE_AI_ERRORS
    assert stats.disabled
    assert client.calls == MAX_CONSECUTIVE_AI_ERRORS
    assert stats.error_types == {"AIClientError": MAX_CONSECUTIVE_AI_ERRORS}
