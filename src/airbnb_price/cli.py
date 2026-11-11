"""Punto de entrada de línea de comandos: ``python -m airbnb_price``.

Comandos: ``info`` e ``ingest``. ``train`` y ``run`` llegan en las fases
siguientes.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from airbnb_price import __version__
from airbnb_price.config import Settings
from airbnb_price.data.ingest import run_ingest
from airbnb_price.exceptions import EXIT_ERROR, EXIT_OK, AirbnbPriceError
from airbnb_price.logging_conf import setup_logging

logger = logging.getLogger("airbnb_price.cli")


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
    ingest.add_argument(
        "--max-error-rate",
        type=float,
        default=1.0,
        help="Tasa máxima de filas rechazadas (0-1); si se supera, exit 2",
    )
    return parser


def cmd_info(settings: Settings) -> int:
    print(f"airbnb_price {__version__}")
    print(f"ai_provider: {settings.ai_provider}")
    print(f"hf_token:    {'configurado' if settings.hf_token else 'no configurado'}")
    print(f"log_level:   {settings.log_level}")
    print(f"home:        {settings.home}")
    return EXIT_OK


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
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
        if args.command == "info":
            return cmd_info(settings)
        return cmd_ingest(settings, args)
    except AirbnbPriceError as exc:
        logger.error("%s %s", type(exc).__name__, exc)
        return exc.exit_code
    except Exception:
        logger.exception("unexpected_error command=%s", args.command)
        return EXIT_ERROR
