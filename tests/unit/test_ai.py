from types import SimpleNamespace

import pytest

from airbnb_price.ai.cache import SentimentCache
from airbnb_price.ai.factory import get_client
from airbnb_price.ai.hf import HFClient, is_transient, label_value, score_from_labels
from airbnb_price.ai.mock import MockClient
from airbnb_price.config import Settings
from airbnb_price.exceptions import AIClientError

TOKEN = "hf_token_secreto"


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status} con token {TOKEN}")
        self.response = SimpleNamespace(status_code=status)


class FakeInference:
    """Sustituto de InferenceClient: falla con los errores indicados y luego responde."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = 0

    def _maybe_fail(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)

    def text_classification(self, text, model, top_k):
        self._maybe_fail()
        return [
            SimpleNamespace(label="positive", score=0.8),
            SimpleNamespace(label="neutral", score=0.1),
            SimpleNamespace(label="negative", score=0.1),
        ]

    def chat_completion(self, messages, model, max_tokens):
        self._maybe_fail()
        message = SimpleNamespace(content="  Está por encima de su zona.  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_hf(fake, cache=None):
    return HFClient(TOKEN, "org/sent", "org/chat", cache=cache, client=fake, sleep=lambda _: None)


# --- mock -----------------------------------------------------------------


def test_mock_sentiment_is_deterministic_and_ordered():
    client = MockClient()
    texts = ["Piso genial y muy limpio", "Horrible, sucio y ruidoso", "Estuvimos tres noches"]

    first, second = client.sentiment(texts), client.sentiment(texts)

    assert first == second
    assert [r.label for r in first] == ["positive", "negative", "neutral"]
    assert all(0.0 <= r.score <= 1.0 and r.provider == "mock" for r in first)


def test_mock_sentiment_empty_list():
    assert MockClient().sentiment([]) == []


def test_mock_explain_mentions_context_and_provider():
    text = MockClient().explain({"barrio": "Sol", "precio_estimado": 90.0, "precio": 150.0})

    assert text.startswith("[mock]")
    assert "Sol" in text and "90.0" in text


# --- factoría ---------------------------------------------------------------


def test_factory_returns_mock_by_default():
    assert get_client(Settings.from_env({})).name == "mock"


def test_factory_returns_hf_when_configured(tmp_path):
    pytest.importorskip("huggingface_hub")
    env = {"AI_PROVIDER": "hf", "HF_TOKEN": TOKEN, "AIRBNB_PRICE_HOME": str(tmp_path)}

    client = get_client(Settings.from_env(env))

    assert client.name == "hf"
    assert client.sentiment_model == "cardiffnlp/twitter-xlm-roberta-base-sentiment"


# --- normalización de etiquetas --------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("positive", 1.0),
        ("NEGATIVE", 0.0),
        ("Neutral", 0.5),
        ("LABEL_2", 1.0),
        ("1 star", 0.0),
        ("5 stars", 1.0),
        ("3 stars", 0.5),
        ("otra", None),
    ],
)
def test_label_value(label, expected):
    assert label_value(label) == expected


def test_score_from_labels_is_expected_positivity():
    assert score_from_labels([("positive", 0.8), ("neutral", 0.1), ("negative", 0.1)]) == 0.85
    assert score_from_labels([("5 stars", 0.5), ("1 star", 0.5)]) == 0.5


def test_score_from_unknown_labels_raises():
    with pytest.raises(AIClientError):
        score_from_labels([("spam", 0.9)])


# --- cliente HF ---------------------------------------------------------------


def test_hf_sentiment_normalizes_output():
    result = make_hf(FakeInference()).sentiment(["Great flat"])[0]

    assert (result.score, result.label, result.provider, result.model) == (
        0.85,
        "positive",
        "hf",
        "org/sent",
    )


def test_hf_explain_returns_clean_text():
    assert make_hf(FakeInference()).explain({"barrio": "Sol"}) == "Está por encima de su zona."


@pytest.mark.parametrize("error", [HTTPError(503), HTTPError(429), TimeoutError("lento")])
def test_hf_retries_transient_errors(error):
    fake = FakeInference(errors=[error])

    result = make_hf(fake).sentiment(["ok"])

    assert fake.calls == 2
    assert result[0].score == 0.85


@pytest.mark.parametrize("status", [401, 403, 404])
def test_hf_does_not_retry_client_errors_and_hides_token(status):
    fake = FakeInference(errors=[HTTPError(status)])

    with pytest.raises(AIClientError) as exc_info:
        make_hf(fake).sentiment(["ok"])

    assert fake.calls == 1
    assert TOKEN not in str(exc_info.value)


def test_hf_gives_up_after_max_retries():
    fake = FakeInference(errors=[HTTPError(500)] * 3)

    with pytest.raises(AIClientError):
        make_hf(fake).explain({})

    assert fake.calls == 3


def test_hf_requires_token():
    with pytest.raises(AIClientError):
        HFClient("", "a/b", "c/d", client=FakeInference())


def test_is_transient():
    assert is_transient(HTTPError(502))
    assert is_transient(ConnectionError())
    assert not is_transient(HTTPError(404))
    assert not is_transient(ValueError("otro"))


# --- caché ------------------------------------------------------------------


def test_cache_avoids_repeated_calls_and_persists(tmp_path):
    path = tmp_path / "cache" / "sentiment.jsonl"
    fake = FakeInference()
    client = make_hf(fake, cache=SentimentCache(path))

    client.sentiment(["mismo texto", "mismo texto"])
    assert fake.calls == 1

    other = FakeInference()
    result = make_hf(other, cache=SentimentCache(path)).sentiment(["mismo texto"])
    assert other.calls == 0
    assert result[0].score == 0.85


def test_cache_key_depends_on_model(tmp_path):
    cache = SentimentCache(tmp_path / "c.jsonl")
    cache.put("modelo/a", "texto", 0.9)

    assert cache.get("modelo/a", "texto") == 0.9
    assert cache.get("modelo/b", "texto") is None
