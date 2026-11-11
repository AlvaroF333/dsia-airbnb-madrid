"""Descarga los datos de Inside Airbnb (Madrid) a ``data/raw/``.

Solo usa la biblioteca estándar para poder ejecutarse antes de instalar el
paquete. Los ficheros completos no se versionan (ver ``.gitignore``).

Uso::

    python scripts/download_data.py --date 2026-09-17
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/{date}/data/{name}"
FILES = ("listings.csv.gz", "reviews.csv.gz")
DEFAULT_DATE = "2026-09-17"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = "dsia-airbnb-madrid/0.2 (proyecto docente)"


def download(url: str, target: Path) -> int:
    """Descarga ``url`` en ``target`` (vía un fichero temporal) y devuelve los bytes escritos."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)
    return target.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=DEFAULT_DATE, help="Fecha de la instantánea AAAA-MM-DD")
    parser.add_argument("--out", type=Path, default=RAW_DIR, help="Directorio de destino")
    parser.add_argument("--force", action="store_true", help="Descarga aunque ya exista")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        print(f"Fecha no válida: {args.date!r} (formato AAAA-MM-DD)", file=sys.stderr)
        return 2

    for name in FILES:
        target = args.out / name
        if target.is_file() and not args.force:
            print(f"ya existe {target} ({target.stat().st_size:,} bytes); usa --force")
            continue
        url = BASE_URL.format(date=args.date, name=name)
        print(f"descargando {url}")
        try:
            size = download(url, target)
        except OSError as exc:
            print(f"error al descargar {url}: {exc}", file=sys.stderr)
            return 1
        print(f"ok {target} ({size:,} bytes)")
    (args.out / "SNAPSHOT").write_text(args.date + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
