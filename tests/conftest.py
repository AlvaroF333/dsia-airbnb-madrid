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


def make_synthetic_listings(n: int = 300, seed: int = 0):
    """Alojamientos sintéticos y limpios cuyo precio depende de capacidad y cercanía a Sol."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    accommodates = rng.integers(1, 9, n)
    latitude = 40.4168 + rng.normal(0, 0.03, n)
    longitude = -3.7038 + rng.normal(0, 0.03, n)
    room_type = rng.choice(["Entire home/apt", "Private room"], n, p=[0.7, 0.3])
    distance = np.hypot(latitude - 40.4168, longitude + 3.7038) * 111
    price = (
        25 * accommodates
        + np.where(room_type == "Entire home/apt", 40, 0)
        - 4 * distance
        + rng.normal(0, 8, n)
    ).clip(15, 900)
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "neighbourhood_cleansed": rng.choice(["Sol", "Embajadores", "Goya"], n),
            "neighbourhood_group_cleansed": rng.choice(["Centro", "Salamanca", "Retiro"], n),
            "latitude": latitude,
            "longitude": longitude,
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": np.ceil(accommodates / 2),
            "beds": accommodates,
            "bathrooms_text": rng.choice(["1 bath", "2 baths", "1.5 shared baths"], n),
            "amenities": rng.choice(['["Wifi"]', '["Wifi", "Kitchen", "Heating"]'], n),
            "minimum_nights": rng.integers(1, 6, n),
            "number_of_reviews": rng.integers(0, 200, n),
            "review_scores_rating": np.round(rng.uniform(3.5, 5.0, n), 2),
            "host_is_superhost": rng.choice(["t", "f"], n),
            "price": np.round(price, 2),
        }
    )


@pytest.fixture(scope="session")
def synthetic_listings():
    return make_synthetic_listings()
