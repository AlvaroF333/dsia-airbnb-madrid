"""Punto de entrada de línea de comandos: ``python -m airbnb_price``.

En la Fase 0 solo existe ``info``; ``ingest``, ``train`` y ``run`` llegan en
las fases siguientes.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from dotenv import load_dotenv

from airbnb_price import __version__
from airbnb_price.config import Settings
from airbnb_price.exceptions import EXIT_OK, AirbnbPriceError
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
    return parser


def cmd_info(settings: Settings) -> int:
    print(f"airbnb_price {__version__}")
    print(f"ai_provider: {settings.ai_provider}")
    print(f"hf_token:    {'configurado' if settings.hf_token else 'no configurado'}")
    print(f"log_level:   {settings.log_level}")
    print(f"home:        {settings.home}")
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
    if args.command == "info":
        return cmd_info(settings)
    return EXIT_OK
