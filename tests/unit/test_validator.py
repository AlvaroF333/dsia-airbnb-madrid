import pandas as pd
import pytest

from airbnb_price.data.validator import (
    REJECT_REASONS,
    REQUIRED_COLUMNS,
    clean_listings,
    parse_price,
    validate_schema,
)
from airbnb_price.exceptions import DataValidationError
from tests.conftest import make_listing


def test_valid_rows_are_kept_with_numeric_price(listings_frame):
    clean, rejects = clean_listings(listings_frame)

    assert len(clean) == 3
    assert clean["price"].tolist() == [120.0, 1000.0, 45.5]
    assert rejects == dict.fromkeys(REJECT_REASONS, 0)


def test_missing_column_raises_domain_error(listings_frame):
    with pytest.raises(DataValidationError, match="price"):
        validate_schema(listings_frame.drop(columns=["price"]))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$120.00", 120.0), ("$1,234.50", 1234.5), ("85", 85.0), (" 60 € ", 60.0)],
)
def test_parse_price_formats(raw, expected):
    assert parse_price(pd.Series([raw])).iloc[0] == expected


@pytest.mark.parametrize("raw", [None, "", "gratis"])
def test_parse_price_invalid_is_null(raw):
    assert parse_price(pd.Series([raw], dtype="object")).isna().all()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"price": None}, "price_missing"),
        ({"price": "$9.99"}, "price_out_of_range"),
        ({"price": "$1,000.01"}, "price_out_of_range"),
        ({"accommodates": 0}, "accommodates_out_of_range"),
        ({"accommodates": 17}, "accommodates_out_of_range"),
        ({"accommodates": None}, "accommodates_out_of_range"),
        ({"latitude": 41.38, "longitude": 2.17}, "coords_out_of_bounds"),
        ({"latitude": None}, "coords_out_of_bounds"),
    ],
)
def test_invalid_row_is_rejected_with_its_reason(overrides, reason):
    frame = pd.DataFrame([make_listing(id="ok"), make_listing(id="bad", **overrides)])

    clean, rejects = clean_listings(frame)

    assert clean["id"].tolist() == ["ok"]
    assert rejects[reason] == 1
    assert sum(rejects.values()) == 1


@pytest.mark.parametrize("price", ["$10.00", "$1,000.00"])
def test_price_limits_are_inclusive(price):
    clean, _ = clean_listings(pd.DataFrame([make_listing(price=price)]))

    assert len(clean) == 1


def test_duplicate_ids_keep_first():
    frame = pd.DataFrame([make_listing(id="7", price="$50.00"), make_listing(id="7")])

    clean, rejects = clean_listings(frame)

    assert clean["price"].tolist() == [50.0]
    assert rejects["duplicate_id"] == 1


def test_row_failing_several_rules_is_counted_once():
    frame = pd.DataFrame([make_listing(price=None, accommodates=99, latitude=0.0)])

    clean, rejects = clean_listings(frame)

    assert clean.empty
    assert rejects["price_missing"] == 1
    assert sum(rejects.values()) == 1


def test_input_frame_is_not_modified(listings_frame):
    before = listings_frame.copy()

    clean_listings(listings_frame)

    pd.testing.assert_frame_equal(listings_frame, before)


def test_extra_columns_are_preserved(listings_frame):
    clean, _ = clean_listings(listings_frame.assign(name="Piso céntrico"))

    assert "name" in clean.columns
    assert set(REQUIRED_COLUMNS) <= set(clean.columns)
