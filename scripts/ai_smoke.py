"""Prueba de humo de un proveedor real de IA (``ollama`` o ``hf``).

Clasifica cuatro reseñas (en español y en inglés) y pide una explicación. Deja la
evidencia en ``reports/ai_smoke_<proveedor>.json``: fecha, modelos, latencias,
resultados o error. Nunca imprime ni guarda claves, y no se ejecuta en CI. Uso::

    python scripts/ai_smoke.py --provider ollama    # o: make ai-smoke
    python scripts/ai_smoke.py --provider hf        # necesita HF_TOKEN en .env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime

from dotenv import load_dotenv

from airbnb_price.ai.factory import get_client
from airbnb_price.config import Settings
from airbnb_price.exceptions import EXIT_OK, AirbnbPriceError
from airbnb_price.logging_conf import setup_logging

FRASES = (
    "El piso estaba limpísimo y la ubicación es perfecta, volveríamos sin dudarlo.",
    "The apartment was dirty and very noisy at night, really disappointing.",
    "Estuvo bien, nada especial.",
    "Great location but the bed was uncomfortable and the host never answered.",
)
CONTEXTO = {
    "barrio": "Universidad",
    "tipo": "Entire home/apt",
    "accommodates": 3,
    "precio": 736.0,
    "precio_estimado": 231.6,
    "desviacion": 504.4,
    "valoracion_media": 4.9,
    "sentimiento_resenas": 0.83,
    "n_resenas_analizadas": 3,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("ollama", "hf"), default="ollama")
    args = parser.parse_args(argv)

    load_dotenv()
    os.environ["AI_PROVIDER"] = args.provider
    evidence: dict = {
        "fecha_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": args.provider,
    }
    code = EXIT_OK
    settings = None
    try:
        settings = Settings.from_env()
        setup_logging(settings.log_level)
        client = get_client(settings)
        evidence["sentiment_model"] = client.sentiment_model
        evidence["chat_model"] = client.chat_model

        start = time.perf_counter()
        results = client.sentiment(FRASES)
        evidence["sentimiento_s"] = round(time.perf_counter() - start, 2)
        evidence["sentimiento"] = [
            {"texto": frase, "score": r.score, "label": r.label}
            for frase, r in zip(FRASES, results, strict=True)
        ]
        for item in evidence["sentimiento"]:
            print(f"[{client.name}] {item['score']:.3f} {item['label']:<8} {item['texto']}")

        start = time.perf_counter()
        evidence["explicacion"] = client.explain(CONTEXTO)
        evidence["explicacion_s"] = round(time.perf_counter() - start, 2)
        evidence["contexto"] = CONTEXTO
        print(f"[{client.name} {client.chat_model}] {evidence['explicacion']}")
        evidence["resultado"] = "ok"
    except AirbnbPriceError as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        evidence["resultado"] = "error"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        code = exc.exit_code

    if settings is not None:
        path = settings.reports_dir / f"ai_smoke_{args.provider}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"evidencia: {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
