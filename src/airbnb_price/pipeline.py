"""Pipeline de extremo a extremo: ingesta → predicción → enriquecimiento con IA → informes.

El precio estimado es el producto; la IA lo enriquece. Si el proveedor de IA
falla, el pipeline termina igualmente con todas las predicciones y deja vacíos
el sentimiento y la explicación de los alojamientos afectados.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from airbnb_price.ai.base import AIClient
from airbnb_price.ai.factory import get_client
from airbnb_price.config import Settings
from airbnb_price.data.ingest import run_ingest
from airbnb_price.data.loader import load_listings, load_reviews
from airbnb_price.data.validator import clean_reviews
from airbnb_price.exceptions import AIClientError
from airbnb_price.model.predict import load_model, predict

logger = logging.getLogger("airbnb_price.pipeline")

REVIEWS_PER_LISTING = 5
MAX_CONSECUTIVE_AI_ERRORS = 3
PREDICTION_COLUMNS = (
    "id",
    "barrio",
    "precio",
    "precio_estimado",
    "desviacion",
    "modelo_version",
    "sentimiento",
    "n_resenas",
    "explicacion",
    "provider",
    "model",
)


@dataclass
class AIStats:
    """Contadores del borde de IA durante una ejecución."""

    calls: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    disabled: bool = False
    error_types: dict[str, int] = field(default_factory=dict)

    def record_error(self, exc: Exception) -> None:
        self.errors += 1
        self.consecutive_errors += 1
        name = type(exc).__name__
        self.error_types[name] = self.error_types.get(name, 0) + 1
        if self.consecutive_errors >= MAX_CONSECUTIVE_AI_ERRORS and not self.disabled:
            self.disabled = True
            logger.warning(
                "ai_disabled tras %s errores seguidos; se continúa sin enriquecer",
                self.consecutive_errors,
            )


def select_outliers(frame: pd.DataFrame, top_n: int) -> pd.Index:
    """Índices de los ``top_n`` alojamientos con mayor desviación absoluta (desempate por id)."""
    order = frame.assign(_abs=frame["desviacion"].abs()).sort_values(
        ["_abs", "id"], ascending=[False, True], kind="stable"
    )
    return order.index[: max(top_n, 0)]


def reviews_by_listing(reviews: pd.DataFrame | None, ids: set[str]) -> dict[str, list[str]]:
    """Hasta :data:`REVIEWS_PER_LISTING` textos (los más recientes) por alojamiento."""
    if reviews is None or reviews.empty:
        return {}
    subset = clean_reviews(reviews[reviews["listing_id"].astype(str).isin(ids)])
    grouped = subset.groupby("listing_id", sort=False).head(REVIEWS_PER_LISTING)
    return grouped.groupby("listing_id")["comments"].apply(list).to_dict()


def _explain_context(row: pd.Series, sentiment: float | None, n_reviews: int) -> dict:
    return {
        "barrio": row.get("barrio"),
        "tipo": row.get("room_type"),
        "accommodates": row.get("accommodates"),
        "precio": row["precio"],
        "precio_estimado": row["precio_estimado"],
        "desviacion": row["desviacion"],
        "valoracion_media": row.get("review_scores_rating"),
        "sentimiento_resenas": sentiment,
        "n_resenas_analizadas": n_reviews,
    }


def enrich(
    frame: pd.DataFrame, outliers: pd.Index, texts: dict[str, list[str]], client: AIClient
) -> AIStats:
    """Rellena sentimiento, explicación, proveedor y modelo de los ``outliers`` (in situ)."""
    stats = AIStats()
    for index in outliers:
        if stats.disabled:
            break
        row = frame.loc[index]
        reviews = texts.get(str(row["id"]), [])
        sentiment = None
        try:
            if reviews:
                stats.calls += 1
                results = client.sentiment(reviews)
                sentiment = round(sum(r.score for r in results) / len(results), 4)
            stats.calls += 1
            explanation = client.explain(_explain_context(row, sentiment, len(reviews)))
        except AIClientError as exc:
            stats.record_error(exc)
            logger.warning("ai_error id=%s error=%s", row["id"], type(exc).__name__)
            continue
        stats.consecutive_errors = 0
        frame.loc[index, ["sentimiento", "n_resenas", "explicacion", "provider", "model"]] = [
            sentiment,
            len(reviews),
            explanation,
            client.name,
            client.sentiment_model,
        ]
    return stats


def run_pipeline(
    settings: Settings,
    input_path: Path,
    reviews_path: Path | None = None,
    top_n: int = 20,
    max_error_rate: float = 1.0,
    client: AIClient | None = None,
) -> dict:
    """Ejecuta el pipeline completo y devuelve las métricas de la ejecución."""
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[stage] = round(now - since, 3)
        return now

    clean_path = settings.data_dir / "processed" / "listings_clean.csv"
    ingest_metrics = run_ingest(
        input_path=input_path,
        output_path=clean_path,
        metrics_path=settings.reports_dir / "ingest_metrics.json",
        max_error_rate=max_error_rate,
    )
    listings = load_listings(clean_path)
    mark = lap("ingesta_s", started)

    model, meta = load_model(settings.models_dir)
    frame = pd.DataFrame(
        {
            "id": listings["id"].astype(str),
            "barrio": listings["neighbourhood_cleansed"],
            "precio": listings["price"].astype(float),
            "precio_estimado": predict(model, listings),
        }
    )
    frame = frame.join(listings[["room_type", "accommodates", "review_scores_rating"]])
    frame["desviacion"] = (frame["precio"] - frame["precio_estimado"]).round(2)
    frame["modelo_version"] = meta.get("modelo_version", "desconocida")
    for column in ("sentimiento", "n_resenas", "explicacion", "provider", "model"):
        frame[column] = pd.Series([None] * len(frame), dtype="object")
    mark = lap("prediccion_s", mark)

    outliers = select_outliers(frame, top_n)
    reviews = load_reviews(reviews_path) if reviews_path is not None else None
    if reviews is None:
        logger.warning("sin fichero de reseñas: el sentimiento quedará vacío")
    texts = reviews_by_listing(reviews, set(frame.loc[outliers, "id"]))
    client = client or get_client(settings)
    stats = enrich(frame, outliers, texts, client)
    frame["n_resenas"] = pd.to_numeric(frame["n_resenas"]).astype("Int64")
    mark = lap("ia_s", mark)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = settings.reports_dir / "predictions.csv"
    frame[list(PREDICTION_COLUMNS)].to_csv(predictions_path, index=False, encoding="utf-8")
    enriched = int(frame["explicacion"].notna().sum())
    metrics = {
        "input": str(input_path),
        "reviews": str(reviews_path) if reviews_path else None,
        "filas_entrada": ingest_metrics["rows_in"],
        "filas_predichas": len(frame),
        "top_n": top_n,
        "filas_enriquecidas": enriched,
        "modelo_version": frame["modelo_version"].iloc[0] if len(frame) else None,
        "desviacion_abs_media": (
            round(float(frame["desviacion"].abs().mean()), 2) if len(frame) else None
        ),
        "ai_provider": client.name,
        "ai_sentiment_model": client.sentiment_model,
        "ai_chat_model": client.chat_model,
        "ai_llamadas": stats.calls,
        "ai_errores": stats.errors,
        "ai_errores_por_tipo": stats.error_types,
        "ai_desactivada": stats.disabled,
    }
    lap("escritura_s", mark)
    timings["total_s"] = round(time.perf_counter() - started, 3)
    metrics["duracion"] = timings
    metrics_path = settings.reports_dir / "run_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if stats.errors:
        logger.warning(
            "pipeline terminado con %s errores de IA; %s/%s filas enriquecidas",
            stats.errors,
            enriched,
            min(top_n, len(frame)),
        )
    logger.info(
        "filas=%s enriquecidas=%s provider=%s llamadas_ia=%s total_s=%s",
        len(frame),
        enriched,
        client.name,
        stats.calls,
        timings["total_s"],
    )
    logger.info("predictions=%s metrics=%s", predictions_path, metrics_path)
    return metrics
