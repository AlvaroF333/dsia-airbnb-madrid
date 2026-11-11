"""Cliente de IA simulado: determinista, sin red ni claves. Es el proveedor por defecto."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from airbnb_price.ai.base import SentimentResult, label_for

POSITIVE_WORDS = frozenset(
    [
        "bueno",
        "buena",
        "genial",
        "excelente",
        "perfecto",
        "perfecta",
        "limpio",
        "limpia",
        "recomendable",
        "encantador",
        "encantadora",
        "fantástico",
        "fantástica",
        "cómodo",
        "cómoda",
        "amable",
        "increíble",
        "good",
        "great",
        "excellent",
        "perfect",
        "clean",
        "lovely",
        "amazing",
        "nice",
        "comfortable",
        "friendly",
        "wonderful",
    ]
)
NEGATIVE_WORDS = frozenset(
    [
        "malo",
        "mala",
        "sucio",
        "sucia",
        "horrible",
        "ruidoso",
        "ruidosa",
        "terrible",
        "decepcionante",
        "pésimo",
        "pésima",
        "incómodo",
        "incómoda",
        "caro",
        "cara",
        "bad",
        "dirty",
        "awful",
        "noisy",
        "terrible",
        "disappointing",
        "poor",
        "rude",
        "uncomfortable",
        "worst",
    ]
)


class MockClient:
    name = "mock"
    sentiment_model = "mock-sentiment"
    chat_model = "mock-chat"

    def sentiment(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> SentimentResult:
        words = re.findall(r"\w+", str(text).lower())
        positive = sum(word in POSITIVE_WORDS for word in words)
        negative = sum(word in NEGATIVE_WORDS for word in words)
        total = positive + negative
        score = 0.5 if total == 0 else round(0.5 + 0.5 * (positive - negative) / total, 4)
        return SentimentResult(score, label_for(score), self.name, self.sentiment_model)

    def explain(self, context: Mapping[str, object]) -> str:
        return (
            f"[mock] Alojamiento en {context.get('barrio', 'Madrid')} para "
            f"{context.get('accommodates', '?')} personas: precio estimado "
            f"{context.get('precio_estimado', '?')} € frente a {context.get('precio', '?')} € "
            "anunciados."
        )
