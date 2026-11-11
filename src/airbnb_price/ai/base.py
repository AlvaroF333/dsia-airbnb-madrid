"""Contrato del cliente de IA. El resto del proyecto solo conoce esta interfaz."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD = 0.6, 0.4


@dataclass(frozen=True)
class SentimentResult:
    """Sentimiento de un texto: ``score`` va de 0 (negativo) a 1 (positivo)."""

    score: float
    label: str
    provider: str
    model: str


def label_for(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


class AIClient(Protocol):
    name: str
    sentiment_model: str
    chat_model: str

    def sentiment(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Un resultado por texto, en el mismo orden."""
        ...

    def explain(self, context: Mapping[str, object]) -> str:
        """Explicación breve, en español, de una estimación de precio."""
        ...
