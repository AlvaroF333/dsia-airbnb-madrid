import pandas as pd
import pytest

from airbnb_price.features.build import (
    FEATURES,
    SOL_LAT,
    SOL_LON,
    build_features,
    count_amenities,
    haversine_km,
    host_years,
    parse_bathrooms,
)
from tests.conftest import make_listing


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1 bath", 1.0), ("1.5 shared baths", 1.5), ("Half-bath", 0.5), ("Shared half-bath", 0.5)],
)
def test_parse_bathrooms(text, expected):
    assert parse_bathrooms(pd.Series([text])).iloc[0] == expected


def test_parse_bathrooms_missing_is_null():
    assert parse_bathrooms(pd.Series([None], dtype="object")).isna().all()


@pytest.mark.parametrize(
    ("value", "expected"),
    [('["Wifi", "Kitchen"]', 2), ("[]", 0), ("no es json", 0), (None, 0), ('{"a": 1}', 0)],
)
def test_count_amenities(value, expected):
    assert count_amenities(value) == expected


def test_haversine_zero_at_origin_and_known_distance():
    lat, lon = pd.Series([SOL_LAT, 40.4530]), pd.Series([SOL_LON, -3.6883])
    distances = haversine_km(lat, lon, SOL_LAT, SOL_LON)

    assert distances.iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Puerta del Sol -> estadio Santiago Bernabéu: unos 4,2 km
    assert distances.iloc[1] == pytest.approx(4.2, abs=0.2)


def test_build_features_columns_and_values(listings_frame):
    features = build_features(listings_frame)

    assert list(features.columns) == list(FEATURES)
    assert len(features) == len(listings_frame)
    first = features.iloc[0]
    assert first["bathrooms"] == 1.0
    assert first["n_amenities"] == 2
    assert first["is_superhost"] == 1
    assert first["dist_sol_km"] == pytest.approx(0.0, abs=1e-6)


def test_build_features_does_not_need_price(listings_frame):
    features = build_features(listings_frame.drop(columns=["price"]))

    assert "price" not in features.columns


def test_build_features_handles_missing_values():
    frame = pd.DataFrame(
        [make_listing(bedrooms=None, review_scores_rating=None, room_type=None, amenities=None)]
    )

    features = build_features(frame)

    assert pd.isna(features.loc[0, "bedrooms"])
    assert features.loc[0, "room_type"] == "desconocido"
    assert features.loc[0, "n_amenities"] == 0


def test_build_features_new_variables(listings_frame):
    features = build_features(listings_frame)
    first = features.iloc[0]

    assert first["host_years"] == pytest.approx(8.5)
    assert first["property_type"] == "Entire rental unit"
    assert first["neighbourhood_cleansed"] == "Sol"


def test_build_features_tolerates_missing_optional_columns():
    """Una predicción puntual solo trae lo obligatorio: el resto queda nulo y se imputa."""
    frame = pd.DataFrame(
        [{"latitude": 40.42, "longitude": -3.70, "room_type": "Private room", "accommodates": 2}]
    )

    features = build_features(frame)

    assert list(features.columns) == list(FEATURES)
    assert pd.isna(features.loc[0, "availability_365"])
    assert pd.isna(features.loc[0, "host_years"])
    assert features.loc[0, "property_type"] == "desconocido"


def test_host_years_combines_years_and_months():
    years = host_years(pd.Series([3, None, 2]), pd.Series([6, 4, None]))

    assert years.iloc[0] == pytest.approx(3.5)
    assert pd.isna(years.iloc[1])
    assert years.iloc[2] == pytest.approx(2.0)


def test_leaky_columns_are_not_features():
    leaky_or_empty = {"estimated_revenue_l365d", "price", "price_quote_total_price", "host_since"}
    assert not leaky_or_empty & set(FEATURES)
