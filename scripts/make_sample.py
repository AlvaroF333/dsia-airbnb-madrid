"""Genera la muestra versionada ``data/sample/`` a partir de ``data/raw/``.

- ``listings_sample.csv``: 2.000 alojamientos al azar (semilla 42), solo con las
  columnas del contrato de entrada, para que el repositorio pese poco.
- ``reviews_sample.csv``: hasta 5 reseñas (las más recientes) de cada alojamiento
  de la muestra, con el texto limpio y recortado a 500 caracteres. Se quitan
  ``reviewer_id`` y ``reviewer_name``: no hacen falta y son datos personales.

Uso::

    python scripts/make_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from airbnb_price.data.loader import REVIEW_COLUMNS, load_listings
from airbnb_price.data.validator import REQUIRED_COLUMNS, clean_reviews

ROOT = Path(__file__).resolve().parents[1]
N_LISTINGS = 2000
REVIEWS_PER_LISTING = 5
SEED = 42


def sample_listings(listings: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    sample = listings.sample(n=min(n, len(listings)), random_state=seed)
    return sample[list(REQUIRED_COLUMNS)].sort_values("id").reset_index(drop=True)


def sample_reviews(reviews_path: Path, ids: set[str], per_listing: int) -> pd.DataFrame:
    """Lee el fichero grande por trozos y se queda con las reseñas de ``ids``."""
    parts = []
    chunks = pd.read_csv(
        reviews_path,
        usecols=list(REVIEW_COLUMNS),
        dtype={"listing_id": "string", "comments": "string"},
        chunksize=200_000,
    )
    for chunk in chunks:
        parts.append(chunk[chunk["listing_id"].isin(ids)])
    reviews = clean_reviews(pd.concat(parts, ignore_index=True))
    return reviews.groupby("listing_id", sort=True).head(per_listing).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sample")
    parser.add_argument("--n", type=int, default=N_LISTINGS)
    args = parser.parse_args(argv)

    listings = sample_listings(load_listings(args.raw / "listings.csv.gz"), args.n, SEED)
    reviews = sample_reviews(
        args.raw / "reviews.csv.gz", set(listings["id"].astype(str)), REVIEWS_PER_LISTING
    )

    args.out.mkdir(parents=True, exist_ok=True)
    listings.to_csv(args.out / "listings_sample.csv", index=False)
    reviews.to_csv(args.out / "reviews_sample.csv", index=False)
    print(f"listings_sample.csv: {len(listings)} filas")
    n_with_reviews = reviews["listing_id"].nunique()
    print(f"reviews_sample.csv: {len(reviews)} reseñas de {n_with_reviews} alojamientos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
