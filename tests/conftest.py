"""Fixtures compartidas: los tests nunca dependen de red, claves ni de un .env local."""

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AI_PROVIDER", "HF_TOKEN", "LOG_LEVEL", "AIRBNB_PRICE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("airbnb_price.cli.load_dotenv", lambda: False)


def make_listing(**overrides):
    """Fila válida de alojamiento con el esquema de Inside Airbnb."""
    row = {
        "id": "1",
        "neighbourhood_cleansed": "Sol",
        "neighbourhood_group_cleansed": "Centro",
        "latitude": 40.4168,
        "longitude": -3.7038,
        "room_type": "Entire home/apt",
        "accommodates": 4,
        "bedrooms": 2,
        "beds": 3,
        "bathrooms_text": "1 bath",
        "amenities": '["Wifi", "Kitchen"]',
        "minimum_nights": 2,
        "number_of_reviews": 25,
        "review_scores_rating": 4.7,
        "host_is_superhost": "t",
        "price": "$120.00",
    }
    row.update(overrides)
    return row


@pytest.fixture
def listings_frame():
    """Tres alojamientos válidos en memoria."""
    import pandas as pd

    return pd.DataFrame(
        [
            make_listing(id="1"),
            make_listing(id="2", price="$1,000.00", neighbourhood_cleansed="Embajadores"),
            make_listing(id="3", price="$45.50", room_type="Private room", accommodates=1),
        ]
    )
