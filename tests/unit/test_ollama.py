"""Cliente de Ollama con un transporte falso: sin red ni servidor."""

import io
import json
import urllib.error

import pytest

from airbnb_price.ai.cache import SentimentCache
from airbnb_price.ai.common import explain_messages, score_from_label_confidence
from airbnb_price.ai.factory import get_client
from airbnb_price.ai.ollama import (
    SENTIMENT_SCHEMA,
    OllamaClient,
    OllamaHTTPError,
    http_transport,
    parse_sentiment,
)
from airbnb_price.config import Settings
from airbnb_price.exceptions import AIClientError


class FakeOllama:
    """Sustituto de ``POST /api/chat``: falla con los errores indicados y luego responde."""

    def __init__(self, errors=(), sentiment='{"label": "negative", "confidence": 0.8}'):
        self.errors = list(errors)
        self.sentiment = sentiment
        self.payloads = []

    def __call__(self, path, payload):
        assert path == "/api/chat"
        self.payloads.append(payload)
        if self.errors:
            raise self.errors.pop(0)
        content = self.sentiment if "format" in payload else "  Está por encima de su zona.  "
        return {"message": {"role": "assistant", "content": content}}


def make_client(fake, cache=None):
    return OllamaClient(
        "http://ollama:11434", "qwen-test", cache=cache, transport=fake, sleep=lambda _: None
    )


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("positive", 1.0, 1.0),
        ("positive", 0.6, 0.8),
        ("negative", 0.9, 0.05),
        ("neutral", 0.7, 0.5),
        ("negative", 7.0, 0.0),  # confianza fuera de rango: se acota a 1
    ],
)
def test_score_from_label_confidence(label, confidence, expected):
    assert score_from_label_confidence(label, confidence) == pytest.approx(expected)


def test_score_from_unknown_label_raises():
    with pytest.raises(AIClientError):
        score_from_label_confidence("enfadado", 0.9)


@pytest.mark.parametrize(
    "content", ["no es json", '{"label": "positive"}', '{"label": 1, "confidence": "x"}']
)
def test_parse_sentiment_rejects_bad_output(content):
    with pytest.raises(AIClientError, match="no válida"):
        parse_sentiment(content)


def test_sentiment_uses_structured_output_and_normalizes():
    fake = FakeOllama()

    [result] = make_client(fake).sentiment(["Sucio y ruidoso"])

    assert result.score == pytest.approx(0.1)
    assert (result.label, result.provider, result.model) == ("negative", "ollama", "qwen-test")
    payload = fake.payloads[0]
    assert payload["format"] == SENTIMENT_SCHEMA
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["messages"][-1]["content"] == "Sucio y ruidoso"


def test_sentiment_truncates_long_texts():
    fake = FakeOllama()

    make_client(fake).sentiment(["x" * 2000])

    assert len(fake.payloads[0]["messages"][-1]["content"]) == 500


def test_explain_returns_clean_text_without_format():
    fake = FakeOllama()

    text = make_client(fake).explain({"barrio": "Sol", "precio": 200.0, "precio_estimado": 120.0})

    assert text == "Está por encima de su zona."
    assert "format" not in fake.payloads[0]


def test_explain_messages_use_readable_labels_and_skip_nulls():
    messages = explain_messages({"precio": 200.0, "sentimiento_resenas": None, "otro": 1})

    system, user = messages
    assert system["role"] == "system" and "dos frases" in system["content"]
    assert "Precio anunciado (€/noche): 200.0" in user["content"]
    assert "Sentimiento" not in user["content"]
    assert "- otro: 1" in user["content"]


@pytest.mark.parametrize("error", [TimeoutError("lento"), OllamaHTTPError(503, "cargando")])
def test_retries_transient_errors(error):
    fake = FakeOllama(errors=[error])

    assert make_client(fake).explain({}) == "Está por encima de su zona."
    assert len(fake.payloads) == 2


def test_does_not_retry_client_errors():
    fake = FakeOllama(errors=[OllamaHTTPError(404, "model not found")])

    with pytest.raises(AIClientError, match="404"):
        make_client(fake).explain({})
    assert len(fake.payloads) == 1


def test_response_without_content_raises():
    client = OllamaClient("http://x", "m", transport=lambda path, payload: {"error": "?"})

    with pytest.raises(AIClientError, match="message.content"):
        client.explain({})


def test_cache_avoids_repeated_calls(tmp_path):
    fake = FakeOllama()
    client = make_client(fake, cache=SentimentCache(tmp_path / "s.jsonl"))

    first = client.sentiment(["Muy limpio"])
    second = client.sentiment(["Muy limpio"])

    assert first == second
    assert len(fake.payloads) == 1


def test_factory_returns_ollama_with_settings(tmp_path):
    settings = Settings.from_env(
        {"AI_PROVIDER": "ollama", "OLLAMA_MODEL": "llama3.2:1b", "AIRBNB_PRICE_HOME": str(tmp_path)}
    )

    client = get_client(settings)

    assert isinstance(client, OllamaClient)
    assert client.name == "ollama"
    assert client.chat_model == client.sentiment_model == "llama3.2:1b"
    assert settings.ollama_url == "http://localhost:11434"


# --- transporte HTTP real, con urlopen sustituido ---------------------------


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_transport_posts_json(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        seen["body"] = json.loads(request.data)
        return FakeResponse(b'{"message": {"content": "hola"}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = http_transport("http://ollama:11434/", 7)("/api/chat", {"a": 1})

    assert out == {"message": {"content": "hola"}}
    assert seen == {"url": "http://ollama:11434/api/chat", "timeout": 7, "body": {"a": 1}}


def test_http_transport_maps_http_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 500, "boom", {}, io.BytesIO(b"fallo interno")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(OllamaHTTPError) as info:
        http_transport("http://x", 1)("/api/chat", {})
    assert info.value.response.status_code == 500


def test_http_transport_server_down_is_clear_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(AIClientError, match="make ollama-up"):
        http_transport("http://x", 1)("/api/chat", {})


def test_http_transport_timeout_is_transient(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(TimeoutError):
        http_transport("http://x", 1)("/api/chat", {})


def test_server_down_through_client_is_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient("http://x", "m", sleep=lambda _: None)

    with pytest.raises(AIClientError, match="ollama falló"):
        client.explain({})
    assert len(calls) == 1
