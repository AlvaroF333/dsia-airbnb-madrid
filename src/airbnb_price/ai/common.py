"""Piezas comunes a los proveedores reales de IA (Hugging Face y Ollama).

Funciones puras (normalización del sentimiento, prompt de la explicación) y la
política de reintentos. Ninguna conoce pandas ni el resto del proyecto.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence

from airbnb_price.exceptions import AIClientError

RETRY_WAITS = (0.5, 1.0)
LABEL_VALUES = {"label_0": 0.0, "label_1": 0.5, "label_2": 1.0}
MAX_TEXT_CHARS = 500
EXPLAIN_MAX_TOKENS = 150

EXPLAIN_SYSTEM_PROMPT = (
    "Eres analista de alquiler turístico en Madrid. Respondes siempre en español, en "
    "exactamente dos frases y sin inventar datos. Los importes son euros por noche: "
    "escríbelos con el símbolo €."
)
_CONTEXT_LABELS = {
    "barrio": "Barrio",
    "tipo": "Tipo de alojamiento",
    "accommodates": "Capacidad (personas)",
    "precio": "Precio anunciado (€/noche)",
    "precio_estimado": "Precio estimado por el modelo (€/noche)",
    "desviacion": "Desviación: anunciado − estimado (€)",
    "valoracion_media": "Valoración media (sobre 5)",
    "sentimiento_resenas": "Sentimiento de las reseñas (0 negativo, 1 positivo)",
    "n_resenas_analizadas": "Reseñas analizadas",
}


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


def score_from_label_confidence(label: str, confidence: float) -> float:
    """Positividad (0..1) a partir de una sola etiqueta y la confianza del modelo en ella.

    Un LLM devuelve "negative con confianza 0,9", no "probabilidad de positivo".
    Se traduce: positivo → 0,5 + 0,5·c; negativo → 0,5 − 0,5·c; neutral → 0,5.
    """
    value = label_value(label)
    if value is None:
        raise AIClientError(f"Etiqueta de sentimiento no reconocida: {label!r}")
    confidence = min(max(float(confidence), 0.0), 1.0)
    return round(0.5 + (value - 0.5) * confidence, 4)


def explain_messages(context: Mapping[str, object]) -> list[dict[str, str]]:
    """Mensajes de chat para explicar la desviación de precio de un alojamiento."""
    datos = "\n".join(
        f"- {_CONTEXT_LABELS.get(key, key)}: {value}"
        for key, value in context.items()
        if value is not None
    )
    user = (
        "Explica por qué el precio anunciado de este alojamiento se desvía del precio "
        "estimado por el modelo. Primera frase: compara ambos precios. Segunda frase: "
        f"un motivo plausible, usando solo estos datos.\n{datos}"
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def is_transient(exc: Exception) -> bool:
    """Solo se reintenta lo transitorio: timeouts, cortes de red, 429 y 5xx."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, TimeoutError | ConnectionError) or "Timeout" in type(exc).__name__


def call_with_retries(
    func: Callable[[], object],
    *,
    provider: str,
    task: str,
    model: str,
    logger: logging.Logger,
    sleep: Callable[[float], None],
    secret: str | None = None,
):
    """Ejecuta ``func`` con reintentos ante fallos transitorios.

    Cualquier fallo final se convierte en ``AIClientError``, sin el ``secret`` en el mensaje.
    El log registra proveedor, tarea, modelo, intento y latencia; nunca el texto.
    """
    for attempt in range(len(RETRY_WAITS) + 1):
        start = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000)
            retry = is_transient(exc) and attempt < len(RETRY_WAITS)
            logger.warning(
                "%s_error task=%s model=%s attempt=%s latency_ms=%s error=%s retry=%s",
                provider,
                task,
                model,
                attempt + 1,
                latency_ms,
                type(exc).__name__,
                retry,
            )
            if not retry:
                detail = str(exc).replace(secret, "***") if secret else str(exc)
                raise AIClientError(
                    f"{provider} falló ({task}, {model}): {type(exc).__name__}: {detail}"
                ) from None
            sleep(RETRY_WAITS[attempt])
        else:
            latency_ms = round((time.perf_counter() - start) * 1000)
            logger.info("%s_ok task=%s model=%s latency_ms=%s", provider, task, model, latency_ms)
            return result
    raise AssertionError("inalcanzable")  # pragma: no cover
