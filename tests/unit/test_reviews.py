import pandas as pd
import pytest

from airbnb_price.data.loader import REVIEW_COLUMNS, load_reviews
from airbnb_price.data.validator import MAX_REVIEW_CHARS, clean_reviews
from airbnb_price.exceptions import DataLoadError, DataValidationError


@pytest.fixture
def reviews_frame():
    return pd.DataFrame(
        {
            "listing_id": ["1", "1", "2", "2", "3"],
            "id": [10, 11, 12, 13, 14],
            "date": ["2026-01-01", "2026-05-01", "2026-02-01", "2026-03-01", "2026-04-01"],
            "reviewer_name": ["Ana", "Luis", "Eva", "Juan", "Rosa"],
            "comments": ["Muy limpio<br/>y céntrico", "Genial", None, ".", "x" * 900],
        }
    )


def test_load_reviews_keeps_only_contract_columns(tmp_path, reviews_frame):
    path = tmp_path / "reviews.csv.gz"
    reviews_frame.to_csv(path, index=False)

    loaded = load_reviews(path)

    assert tuple(loaded.columns) == REVIEW_COLUMNS
    assert "reviewer_name" not in loaded.columns
    assert loaded["listing_id"].tolist()[:2] == ["1", "1"]


def test_load_reviews_missing_file(tmp_path):
    with pytest.raises(DataLoadError):
        load_reviews(tmp_path / "no_existe.csv")


def test_load_reviews_missing_columns(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"listing_id": ["1"], "texto": ["hola"]}).to_csv(path, index=False)

    with pytest.raises(DataValidationError):
        load_reviews(path)


def test_load_reviews_unreadable_file(tmp_path):
    path = tmp_path / "vacio.csv"
    path.write_text("")

    with pytest.raises(DataLoadError):
        load_reviews(path)


def test_clean_reviews_strips_html_drops_empty_and_sorts_recent_first(reviews_frame):
    clean = clean_reviews(reviews_frame)

    assert clean["listing_id"].tolist() == ["1", "1", "3"]
    assert clean.loc[0, "comments"] == "Genial"
    assert clean.loc[1, "comments"] == "Muy limpio y céntrico"
    assert len(clean.loc[2, "comments"]) == MAX_REVIEW_CHARS
