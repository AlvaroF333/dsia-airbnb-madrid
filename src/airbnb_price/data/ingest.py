"""Job de ingesta: lee, limpia y deja datos limpios + métricas en disco."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from airbnb_price.data.loader import load_listings
from airbnb_price.data.validator import clean_listings
from airbnb_price.exceptions import DataValidationError

logger = logging.getLogger("airbnb_price.ingest")


def run_ingest(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    max_error_rate: float = 1.0,
) -> dict:
    """Ejecuta la ingesta y devuelve las métricas.

    Las métricas se escriben siempre. Si la tasa de filas rechazadas supera
    ``max_error_rate`` no se escriben los datos limpios y se lanza
    :class:`DataValidationError`. Reejecutar con la misma entrada produce los
    mismos ficheros (idempotente).
    """
    raw = load_listings(input_path)
    clean, rejects = clean_listings(raw)

    rows_in, rows_out = len(raw), len(clean)
    error_rate = (rows_in - rows_out) / rows_in if rows_in else 0.0
    metrics = {
        "input": str(input_path),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "error_rate": round(error_rate, 4),
        "max_error_rate": max_error_rate,
        "rejects": rejects,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    logger.info("rows_in=%s rows_out=%s error_rate=%.4f", rows_in, rows_out, error_rate)

    if error_rate > max_error_rate:
        raise DataValidationError(
            f"error_rate {error_rate:.4f} supera el máximo permitido {max_error_rate}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    logger.info("output=%s metrics=%s", output_path, metrics_path)
    return metrics
