"""Caché en disco de resultados de sentimiento, para no repetir llamadas de pago o con cuota."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class SentimentCache:
    """Fichero JSONL ``{"key": ..., "score": ...}``; la clave es SHA-256 de modelo + texto."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._scores: dict[str, float] = {}
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._scores[entry["key"]] = entry["score"]

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> float | None:
        return self._scores.get(self.key(model, text))

    def put(self, model: str, text: str, score: float) -> None:
        key = self.key(model, text)
        self._scores[key] = score
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"key": key, "score": score}) + "\n")
