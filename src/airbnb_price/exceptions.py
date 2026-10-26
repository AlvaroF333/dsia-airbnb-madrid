"""Excepciones de dominio y códigos de salida del job.

Los códigos distinguen la culpa del dato de la culpa del sistema, igual que
el job de la sesión 5 y los 4xx/5xx de la futura API:

- ``EXIT_OK`` (0): ejecución correcta.
- ``EXIT_ERROR`` (1): fallo del sistema o de una dependencia externa.
- ``EXIT_INVALID_INPUT`` (2): la entrada no cumple el contrato.
"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2


class AirbnbPriceError(Exception):
    """Base de todas las excepciones del proyecto."""

    exit_code = EXIT_ERROR


class ConfigError(AirbnbPriceError):
    """Configuración inválida o incompleta (variables de entorno)."""

    exit_code = EXIT_INVALID_INPUT


class DataValidationError(AirbnbPriceError):
    """Los datos de entrada no cumplen el schema acordado."""

    exit_code = EXIT_INVALID_INPUT
