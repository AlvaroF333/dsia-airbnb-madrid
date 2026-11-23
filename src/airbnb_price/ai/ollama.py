"""Cliente de Ollama: un LLM abierto servido por API REST, sin claves ni coste.

Ollama (https://ollama.com) sirve modelos abiertos (por defecto ``qwen2.5:1.5b``,
de Alibaba, licencia Apache 2.0) en ``http://localhost:11434``. Se usa su endpoint
``POST /api/chat`` para las dos tareas:

- **sentimiento**: salida estructurada con un JSON Schema (``format``), que obliga
  al modelo a responder ``{"label": ..., "confidence": ...}``;
- **explicación**: chat con el mismo prompt que el proveedor de Hugging Face.

Temperatura 0 y semilla fija: la misma entrada da la misma salida. Solo usa la
biblioteca estándar (``urllib``), así que no añade dependencias.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

from airbnb_price.ai.base import SentimentResult, label_for
from airbnb_price.ai.cache import SentimentCache
from airbnb_price.ai.common import (
    EXPLAIN_MAX_TOKENS,
    MAX_TEXT_CHARS,
    call_with_retries,
    explain_messages,
    score_from_label_confidence,
)
from airbnb_price.exceptions import AIClientError

logger = logging.getLogger("airbnb_price.ai.ollama")

SEED = 42
SENTIMENT_SYSTEM_PROMPT = (
    "Clasifica el sentimiento de una reseña de un alojamiento turístico (en cualquier "
    "idioma). Responde solo con JSON: label (positive, neutral o negative) y confidence "
    "(de 0 a 1, tu confianza en esa etiqueta)."
)
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
}

Transport = Callable[[str, dict], dict]


class OllamaHTTPError(Exception):
    """Error HTTP de Ollama con ``response.status_code`` (lo que mira ``is_transient``)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.response = SimpleNamespace(status_code=status)


def http_transport(base_url: str, timeout: float) -> Transport:
    """POST JSON a ``base_url + path`` con ``urllib``; traduce los errores de red."""

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise OllamaHTTPError(exc.code, detail) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(str(exc.reason)) from None
            # Servidor apagado o inaccesible: no es transitorio, no se reintenta
            raise AIClientError(
                f"No se puede conectar con Ollama en {base_url} ({exc.reason}). "
                "Arráncalo con: make ollama-up"
            ) from None

    return post


def parse_sentiment(content: str) -> float:
    """Positividad 0..1 a partir de la respuesta JSON del modelo."""
    try:
        data = json.loads(content)
        return score_from_label_confidence(str(data["label"]), float(data["confidence"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AIClientError(f"Respuesta de sentimiento no válida: {content[:100]!r}") from exc


class OllamaClient:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        cache: SentimentCache | None = None,
        timeout: float = 120.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._post = transport or http_transport(base_url, timeout)
        self._cache = cache
        self._sleep = sleep
        self.sentiment_model = model
        self.chat_model = model

    def _chat(self, task: str, messages: list[dict], max_tokens: int, fmt: dict | None = None):
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_predict": max_tokens},
        }
        if fmt is not None:
            payload["format"] = fmt
        response = call_with_retries(
            lambda: self._post("/api/chat", payload),
            provider=self.name,
            task=task,
            model=self.chat_model,
            logger=logger,
            sleep=self._sleep,
        )
        try:
            return str(response["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise AIClientError("Respuesta de Ollama sin message.content") from exc

    def sentiment(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [self._one(str(text)[:MAX_TEXT_CHARS]) for text in texts]

    def _one(self, text: str) -> SentimentResult:
        model = self.sentiment_model
        score = self._cache.get(model, text) if self._cache else None
        if score is None:
            content = self._chat(
                "sentiment",
                [
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=40,
                fmt=SENTIMENT_SCHEMA,
            )
            score = parse_sentiment(content)
            if self._cache:
                self._cache.put(model, text, score)
        return SentimentResult(score, label_for(score), self.name, model)

    def explain(self, context: Mapping[str, object]) -> str:
        return self._chat("chat", explain_messages(context), EXPLAIN_MAX_TOKENS).strip()
