"""Elige el cliente de IA según la configuración (``AI_PROVIDER``)."""

from __future__ import annotations

from airbnb_price.ai.base import AIClient
from airbnb_price.ai.mock import MockClient
from airbnb_price.config import Settings


def get_client(settings: Settings) -> AIClient:
    if settings.ai_provider == "hf":
        from airbnb_price.ai.cache import SentimentCache
        from airbnb_price.ai.hf import HFClient

        return HFClient(
            token=settings.hf_token or "",
            sentiment_model=settings.hf_sentiment_model,
            chat_model=settings.hf_chat_model,
            cache=SentimentCache(settings.cache_dir / "sentiment.jsonl"),
        )
    return MockClient()
