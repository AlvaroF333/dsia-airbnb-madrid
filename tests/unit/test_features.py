import pandas as pd
import pytest

from airbnb_price.features.build import (
    FEATURES,
    SOL_LAT,
    SOL_LON,
    build_features,
    count_amenities,
    haversine_km,
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
