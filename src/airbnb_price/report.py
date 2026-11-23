"""Informe legible de una ejecución del pipeline (``reports/report.md``).

Función pura: recibe las predicciones y las métricas y devuelve el texto en
Markdown; no toca disco ni red. Lo escribe ``pipeline.run_pipeline``.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

MAX_EXPLANATION_CHARS = 220
TOP_BARRIOS = 5


def _cell(value: object) -> str:
    """Valor apto para una celda de tabla Markdown (sin saltos de línea ni barras)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        # Importes con 2 decimales; proporciones (R², sentimiento) con hasta 4 cifras
        return f"{value:.4g}" if abs(value) < 1 else f"{value:.2f}"
    text = " ".join(str(value).split()).replace("|", "\\|")
    if len(text) > MAX_EXPLANATION_CHARS:
        text = text[: MAX_EXPLANATION_CHARS - 1].rstrip() + "…"
    return text


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines


def render_report(
    frame: pd.DataFrame,
    outliers: pd.Index,
    metrics: Mapping[str, object],
    meta: Mapping[str, object],
    generated_at: str,
) -> str:
    """Devuelve el informe en Markdown.

    ``frame`` son las predicciones (columnas de ``pipeline.PREDICTION_COLUMNS``),
    ``outliers`` los índices enriquecidos con IA, ``metrics`` las de la ejecución
    y ``meta`` los metadatos del modelo.
    """
    lines = [
        "# Informe de ejecución del pipeline",
        "",
        f"Generado el {generated_at} · modelo `{meta.get('modelo_version', 'desconocida')}`"
        f" entrenado el {meta.get('entrenado_en', 'desconocido')}",
        "",
        "## Resumen",
        "",
    ]
    test = meta.get("metricas_test") or {}
    deviation = frame["desviacion"].abs() if len(frame) else pd.Series(dtype=float)
    summary = [
        ["Alojamientos de entrada", metrics.get("filas_entrada")],
        ["Alojamientos con precio estimado", metrics.get("filas_predichas")],
        ["Desviación absoluta media (€)", metrics.get("desviacion_abs_media")],
        [
            "Desviación absoluta mediana (€)",
            round(float(deviation.median()), 2) if len(deviation) else None,
        ],
        ["MAE del modelo en test (€)", test.get("mae")],
        ["R² del modelo en test", test.get("r2")],
        [
            "Proveedor de IA",
            f"{metrics.get('ai_provider')} ({metrics.get('ai_sentiment_model')}, "
            f"{metrics.get('ai_chat_model')})",
        ],
        [
            "Llamadas / errores de IA",
            f"{metrics.get('ai_llamadas')} / {metrics.get('ai_errores')}",
        ],
        ["Filas enriquecidas con IA", f"{metrics.get('filas_enriquecidas')} de {len(outliers)}"],
    ]
    lines += _table(["Indicador", "Valor"], summary)

    if metrics.get("ai_desactivada"):
        lines += [
            "",
            "> **Aviso:** la IA se desactivó tras varios errores seguidos. Los precios"
            " estimados son válidos; faltan el sentimiento y la explicación.",
        ]

    lines += [
        "",
        f"## Los {len(outliers)} alojamientos más desviados",
        "",
        "Desviación = precio anunciado − precio estimado. Positiva: anunciado por encima"
        " de lo esperable.",
        "",
    ]
    top = frame.loc[outliers]
    lines += _table(
        [
            "id",
            "Barrio",
            "Precio (€)",
            "Estimado (€)",
            "Desviación (€)",
            "Sentimiento",
            "Explicación",
        ],
        [
            [
                row["id"],
                row["barrio"],
                row["precio"],
                row["precio_estimado"],
                row["desviacion"],
                row["sentimiento"],
                row["explicacion"],
            ]
            for _, row in top.iterrows()
        ],
    )

    if len(frame):
        by_barrio = (
            frame.assign(_abs=deviation)
            .groupby("barrio")
            .agg(n=("_abs", "size"), media=("_abs", "mean"))
            .query("n >= 5")
            .sort_values("media", ascending=False)
            .head(TOP_BARRIOS)
        )
        lines += [
            "",
            f"## Barrios con mayor desviación media (mín. 5 alojamientos, top {TOP_BARRIOS})",
            "",
        ]
        lines += _table(
            ["Barrio", "Alojamientos", "Desviación abs. media (€)"],
            [
                [barrio, int(row["n"]), round(float(row["media"]), 2)]
                for barrio, row in by_barrio.iterrows()
            ],
        )

    durations = metrics.get("duracion") or {}
    lines += ["", "## Duración por etapa (s)", ""]
    lines += _table(["Etapa", "Segundos"], [[k, v] for k, v in durations.items()])
    lines.append("")
    return "\n".join(lines)
