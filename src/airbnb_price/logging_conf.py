"""Formato de log único para todas las capas."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configura el logger raíz; llamar una sola vez desde el punto de entrada."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
