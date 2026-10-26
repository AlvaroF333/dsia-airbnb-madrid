"""Configuración del proyecto leída del entorno (nunca del código)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from airbnb_price.exceptions import ConfigError

AI_PROVIDERS = ("mock", "hf")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Parámetros de ejecución. Se construyen con :meth:`from_env`."""

    ai_provider: str
    hf_token: str | None
    log_level: str
    home: Path

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Lee y valida la configuración.

        ``env`` permite inyectar un diccionario en los tests; por defecto se
        usa ``os.environ``. El proveedor de IA es ``mock`` salvo que se pida
        otro explícitamente.
        """
        env = os.environ if env is None else env

        provider = env.get("AI_PROVIDER", "mock").strip().lower() or "mock"
        if provider not in AI_PROVIDERS:
            raise ConfigError(f"AI_PROVIDER no soportado: {provider!r} (usa {AI_PROVIDERS})")

        token = env.get("HF_TOKEN", "").strip() or None
        if provider == "hf" and token is None:
            raise ConfigError("AI_PROVIDER=hf requiere HF_TOKEN en el entorno")

        level = env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
        if level not in LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL no válido: {level!r} (usa {LOG_LEVELS})")

        home = Path(env.get("AIRBNB_PRICE_HOME", "") or PROJECT_ROOT)
        return cls(ai_provider=provider, hf_token=token, log_level=level, home=home)
