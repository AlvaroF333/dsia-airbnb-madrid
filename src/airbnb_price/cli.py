"""Punto de entrada de línea de comandos: ``python -m airbnb_price``.

Comandos: ``info``, ``ingest``, ``train``, ``predict``, ``sentiment`` y ``run``.
La CLI es una entrada fina: traduce argumentos y delega en el núcleo; no limpia
datos ni llama a Hugging Face directamente.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from airbnb_price import __version__
from airbnb_price.config import Settings
from airbnb_price.exceptions import EXIT_ERROR, EXIT_OK, AirbnbPriceError
from airbnb_price.logging_conf import setup_logging

logger = logging.getLogger("airbnb_price.cli")

SAMPLE_REVIEWS = Path("data") / "sample" / "reviews_sample.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airbnb_price",
        description="Estimador del precio por noche de alojamientos en Madrid.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Muestra la configuración efectiva (sin secretos)")

    ingest = sub.add_parser("ingest", help="Valida y limpia el CSV de alojamientos")
    ingest.add_argument("--input", type=Path, help="CSV de Inside Airbnb (.csv o .csv.gz)")
    ingest.add_argument("--output", type=Path, help="CSV limpio de salida")
    ingest.add_argument("--metrics", type=Path, help="JSON de métricas de la ingesta")
    _add_max_error_rate(ingest)

    train = sub.add_parser("train", help="Entrena y evalúa el modelo de precio")
    train.add_argument("--input", type=Path, help="CSV limpio (salida de ingest)")
    train.add_argument("--model-dir", type=Path, help="Directorio del modelo")
    train.add_argument("--metrics", type=Path, help="JSON de métricas del modelo")
    train.add_argument("--seed", type=int, default=42, help="Semilla (por defecto 42)")

    predict = sub.add_parser("predict", help="Estima el precio de alojamientos en JSON")
    predict.add_argument("--input", type=Path, required=True, help="Objeto o lista JSON")
    predict.add_argument("--model-dir", type=Path, help="Directorio del modelo")

    sentiment = sub.add_parser("sentiment", help="Sentimiento de uno o varios textos")
    sentiment.add_argument(
        "--text", action="append", required=True, help="Texto (se puede repetir)"
    )

    run = sub.add_parser("run", help="Pipeline completo: ingesta, predicción e IA")
    run.add_argument("--input", type=Path, help="CSV de alojamientos (por defecto, la muestra)")
    run.add_argument("--reviews", type=Path, help="CSV de reseñas (por defecto, la muestra)")
    run.add_argument("--top-n", type=int, default=20, help="Alojamientos que enriquecer con IA")
    _add_max_error_rate(run)
    return parser


def _add_max_error_rate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=1.0,
        help="Tasa máxima de filas rechazadas (0-1); si se supera, exit 2",
    )


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_info(settings: Settings, args: argparse.Namespace) -> int:
    print(f"airbnb_price {__version__}")
    print(f"ai_provider: {settings.ai_provider}")
    print(f"hf_token:    {'configurado' if settings.hf_token else 'no configurado'}")
    print(f"log_level:   {settings.log_level}")
    print(f"home:        {settings.home}")
    return EXIT_OK


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    from airbnb_price.data.ingest import run_ingest

    metrics = run_ingest(
        input_path=args.input or settings.data_dir / "raw" / "listings.csv.gz",
        output_path=args.output or settings.data_dir / "processed" / "listings_clean.csv",
        metrics_path=args.metrics or settings.reports_dir / "ingest_metrics.json",
        max_error_rate=args.max_error_rate,
    )
    print(
        f"rows_in={metrics['rows_in']} rows_out={metrics['rows_out']} "
        f"error_rate={metrics['error_rate']}"
    )
    return EXIT_OK


def cmd_train(settings: Settings, args: argparse.Namespace) -> int:
    from airbnb_price.model.train import run_train

    metrics = run_train(
        input_path=args.input or settings.data_dir / "processed" / "listings_clean.csv",
        model_dir=args.model_dir or settings.models_dir,
        metrics_path=args.metrics or settings.reports_dir / "model_metrics.json",
        seed=args.seed,
    )
    _print_json({name: metrics[name] for name in ("modelo", "baseline_ridge", "baseline_mediana")})
    return EXIT_OK


def cmd_predict(settings: Settings, args: argparse.Namespace) -> int:
    from airbnb_price.data.loader import load_json_records
    from airbnb_price.data.validator import validate_prediction_input
    from airbnb_price.model.predict import load_model, predict

    frame = validate_prediction_input(load_json_records(args.input))
    model, meta = load_model(args.model_dir or settings.models_dir)
    prices = predict(model, frame)
    version = meta.get("modelo_version", "desconocida")
    _print_json(
        [
            {"id": id_, "precio_estimado": float(price), "modelo_version": version}
            for id_, price in zip(frame["id"], prices, strict=True)
        ]
    )
    return EXIT_OK


def cmd_sentiment(settings: Settings, args: argparse.Namespace) -> int:
    from airbnb_price.ai.factory import get_client

    results = get_client(settings).sentiment(args.text)
    _print_json(
        [
            {"score": r.score, "label": r.label, "provider": r.provider, "model": r.model}
            for r in results
        ]
    )
    return EXIT_OK


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    from airbnb_price.pipeline import run_pipeline

    reviews = args.reviews
    if reviews is None and (settings.home / SAMPLE_REVIEWS).is_file():
        reviews = settings.home / SAMPLE_REVIEWS
    metrics = run_pipeline(
        settings,
        input_path=args.input or settings.data_dir / "sample" / "listings_sample.csv",
        reviews_path=reviews,
        top_n=args.top_n,
        max_error_rate=args.max_error_rate,
    )
    print(
        f"filas={metrics['filas_predichas']} enriquecidas={metrics['filas_enriquecidas']} "
        f"provider={metrics['ai_provider']} errores_ia={metrics['ai_errores']}"
    )
    return EXIT_OK


COMMANDS = {
    "info": cmd_info,
    "ingest": cmd_ingest,
    "train": cmd_train,
    "predict": cmd_predict,
    "sentiment": cmd_sentiment,
    "run": cmd_run,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        settings = Settings.from_env()
    except AirbnbPriceError as exc:
        setup_logging()
        logger.error("config_error %s", exc)
        return exc.exit_code

    setup_logging(settings.log_level)
    logger.info("command=%s provider=%s", args.command, settings.ai_provider)
    try:
        return COMMANDS[args.command](settings, args)
    except AirbnbPriceError as exc:
        logger.error("%s %s", type(exc).__name__, exc)
        return exc.exit_code
    except Exception:
        logger.exception("unexpected_error command=%s", args.command)
        return EXIT_ERROR
