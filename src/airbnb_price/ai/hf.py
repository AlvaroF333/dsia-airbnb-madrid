"""Cliente de Hugging Face Inference: sentimiento (text_classification) y explicación (chat)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence

from airbnb_price.ai.base import SentimentResult, label_for
from airbnb_price.ai.cache import SentimentCache
from airbnb_price.ai.common import (
    EXPLAIN_MAX_TOKENS,
    MAX_TEXT_CHARS,
    call_with_retries,
    explain_messages,
    is_transient,
    label_value,
    score_from_labels,
)
from airbnb_price.exceptions import AIClientError

__all__ = ["HFClient", "is_transient", "label_value", "score_from_labels"]

logger = logging.getLogger("airbnb_price.ai.hf")


class HFClient:
    name = "hf"

    def __init__(
        self,
        token: str,
        sentiment_model: str,
        chat_model: str,
        cache: SentimentCache | None = None,
        timeout: float = 30.0,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise AIClientError("Falta HF_TOKEN para usar el proveedor hf")
        if client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise AIClientError("Instala la dependencia: pip install huggingface_hub") from exc
            client = InferenceClient(api_key=token, provider="auto", timeout=timeout)
        self._client = client
        self._token = token
        self._cache = cache
        self._sleep = sleep
        self.sentiment_model = sentiment_model
        self.chat_model = chat_model

    def _call(self, task: str, model: str, func: Callable[[], object]):
        return call_with_retries(
            func,
            provider=self.name,
            task=task,
            model=model,
            logger=logger,
            sleep=self._sleep,
            secret=self._token,
        )

    def sentiment(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [self._one(str(text)[:MAX_TEXT_CHARS]) for text in texts]

    def _one(self, text: str) -> SentimentResult:
        model = self.sentiment_model
        score = self._cache.get(model, text) if self._cache else None
        if score is None:
            output = self._call(
                "text_classification",
                model,
                lambda: self._client.text_classification(text, model=model, top_k=5),
            )
            score = score_from_labels([(item.label, float(item.score)) for item in output])
            if self._cache:
                self._cache.put(model, text, score)
        return SentimentResult(score, label_for(score), self.name, model)

    def explain(self, context: Mapping[str, object]) -> str:
        output = self._call(
            "chat_completion",
            self.chat_model,
            lambda: self._client.chat_completion(
                messages=explain_messages(context),
                model=self.chat_model,
                max_tokens=EXPLAIN_MAX_TOKENS,
            ),
        )
        return (output.choices[0].message.content or "").strip()
