"""Cliente de Hugging Face Inference: sentimiento (text_classification) y explicación (chat)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence

from airbnb_price.ai.base import SentimentResult, label_for
from airbnb_price.ai.cache import SentimentCache
from airbnb_price.exceptions import AIClientError

logger = logging.getLogger("airbnb_price.ai.hf")

MAX_TEXT_CHARS = 500
RETRY_WAITS = (0.5, 1.0)
LABEL_VALUES = {"label_0": 0.0, "label_1": 0.5, "label_2": 1.0}


def label_value(label: str) -> float | None:
    """Valor 0..1 de una etiqueta de sentimiento; ``None`` si no se reconoce.

    Admite ``negative/neutral/positive``, ``LABEL_0..2`` y ``"N stars"`` (1 a 5).
    """
    text = label.strip().lower()
    stars = re.match(r"(\d)\s*star", text)
    if stars:
        return (int(stars.group(1)) - 1) / 4
    if text in LABEL_VALUES:
        return LABEL_VALUES[text]
    for prefix, value in (("pos", 1.0), ("neu", 0.5), ("neg", 0.0)):
        if text.startswith(prefix):
            return value
    return None


def score_from_labels(items: Sequence[tuple[str, float]]) -> float:
    """Positividad esperada (0..1) a partir de pares ``(etiqueta, probabilidad)``."""
    weighted = [(label_value(label), prob) for label, prob in items]
    known = [(value, prob) for value, prob in weighted if value is not None]
    total = sum(prob for _, prob in known)
    if not known or total <= 0:
        raise AIClientError(f"Etiquetas de sentimiento no reconocidas: {[i[0] for i in items]}")
    return round(sum(value * prob for value, prob in known) / total, 4)


def is_transient(exc: Exception) -> bool:
    """Solo se reintenta lo transitorio: timeouts, cortes de red, 429 y 5xx."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, TimeoutError | ConnectionError) or "Timeout" in type(exc).__name__


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
        """Ejecuta una llamada con reintentos y la traduce a ``AIClientError`` si falla."""
        for attempt in range(len(RETRY_WAITS) + 1):
            start = time.perf_counter()
            try:
                result = func()
            except Exception as exc:
                latency_ms = round((time.perf_counter() - start) * 1000)
                retry = is_transient(exc) and attempt < len(RETRY_WAITS)
                logger.warning(
                    "hf_error task=%s model=%s attempt=%s latency_ms=%s error=%s retry=%s",
                    task,
                    model,
                    attempt + 1,
                    latency_ms,
                    type(exc).__name__,
                    retry,
                )
                if not retry:
                    detail = str(exc).replace(self._token, "***")
                    raise AIClientError(
                        f"Hugging Face falló ({task}, {model}): {type(exc).__name__}: {detail}"
                    ) from None
                self._sleep(RETRY_WAITS[attempt])
            else:
                latency_ms = round((time.perf_counter() - start) * 1000)
                logger.info("hf_ok task=%s model=%s latency_ms=%s", task, model, latency_ms)
                return result
        raise AssertionError("inalcanzable")  # pragma: no cover

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
        datos = "; ".join(f"{key}: {value}" for key, value in context.items())
        prompt = (
            "Eres analista de alquiler turístico en Madrid. Explica en dos frases, en español "
            "y sin inventar datos, por qué el precio anunciado de este alojamiento se desvía "
            f"del precio estimado por el modelo. Datos: {datos}"
        )
        output = self._call(
            "chat_completion",
            self.chat_model,
            lambda: self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.chat_model,
                max_tokens=150,
            ),
        )
        return (output.choices[0].message.content or "").strip()
