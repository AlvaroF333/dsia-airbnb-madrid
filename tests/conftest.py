"""Fixtures compartidas: los tests nunca dependen de red, claves ni de un .env local."""

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AI_PROVIDER", "HF_TOKEN", "LOG_LEVEL", "AIRBNB_PRICE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("airbnb_price.cli.load_dotenv", lambda: False)
