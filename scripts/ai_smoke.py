"""Prueba de humo del proveedor real (Hugging Face Inference).

Clasifica dos frases y pide una explicación. Necesita ``HF_TOKEN`` en ``.env``.
Nunca imprime el token ni se ejecuta en CI. Uso::

    AI_PROVIDER=hf python scripts/hf_smoke.py      # o: make hf-smoke
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from airbnb_price.ai.factory import get_client
from airbnb_price.config import Settings
from airbnb_price.exceptions import AirbnbPriceError
from airbnb_price.logging_conf import setup_logging

FRASES = (
    "El piso estaba limpísimo y la ubicación es perfecta, volveríamos sin dudarlo.",
    "The apartment was dirty and very noisy at night, really disappointing.",
)
CONTEXTO = {
    "barrio": "Embajadores",
    "tipo": "Entire home/apt",
    "accommodates": 4,
    "precio": 180.0,
    "precio_estimado": 110.0,
    "desviacion": 70.0,
    "valoracion_media": 4.8,
    "sentimiento_resenas": 0.9,
    "n_resenas_analizadas": 5,
}


def main() -> int:
    load_dotenv()
    os.environ["AI_PROVIDER"] = "hf"
    try:
        settings = Settings.from_env()
        setup_logging(settings.log_level)
        client = get_client(settings)

        start = time.perf_counter()
        for frase, result in zip(FRASES, client.sentiment(FRASES), strict=True):
            print(f"[{result.provider} {result.model}] {result.score:.3f} {result.label}: {frase}")
        print(f"sentimiento: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        print(f"[{client.name} {client.chat_model}] {client.explain(CONTEXTO)}")
        print(f"explicación: {time.perf_counter() - start:.2f} s")
    except AirbnbPriceError as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
