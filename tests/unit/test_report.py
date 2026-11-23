import pandas as pd
import pytest

from airbnb_price.report import MAX_EXPLANATION_CHARS, render_report


@pytest.fixture
def frame():
    data = pd.DataFrame(
        {
            "id": [str(i) for i in range(6)],
            "barrio": ["Sol"] * 5 + ["Goya"],
            "precio": [500.0, 100.0, 90.0, 80.0, 120.0, 60.0],
            "precio_estimado": [150.0, 110.0, 95.0, 85.0, 100.0, 70.0],
            "sentimiento": [0.9, None, None, None, None, None],
            "explicacion": ["Piso | de lujo\ncon vistas", None, None, None, None, None],
        }
    )
    data["desviacion"] = data["precio"] - data["precio_estimado"]
    return data


METRICS = {
    "filas_entrada": 7,
    "filas_predichas": 6,
    "desviacion_abs_media": 66.67,
    "ai_provider": "mock",
    "ai_sentiment_model": "mock-sentiment",
    "ai_chat_model": "mock-chat",
    "ai_llamadas": 2,
    "ai_errores": 0,
    "filas_enriquecidas": 1,
    "ai_desactivada": False,
    "duracion": {"ingesta_s": 0.1, "total_s": 0.5},
}
META = {"modelo_version": "0.3.0", "entrenado_en": "2026-11-23", "metricas_test": {"mae": 48.4}}


def test_report_has_sections_and_values(frame):
    text = render_report(frame, frame.index[:2], METRICS, META, "2026-11-23T10:00:00+00:00")

    assert text.startswith("# Informe de ejecución del pipeline")
    for section in ("## Resumen", "## Los 2 alojamientos más desviados", "## Duración"):
        assert section in text
    assert "`0.3.0`" in text
    assert "| MAE del modelo en test (€) | 48.40 |" in text
    assert "| 1 / 0 |" not in text  # las llamadas van en su fila
    assert "| Llamadas / errores de IA | 2 / 0 |" in text
    assert "Aviso" not in text


def test_report_escapes_table_cells_and_marks_missing(frame):
    text = render_report(frame, frame.index[:2], METRICS, META, "ahora")

    assert r"Piso \| de lujo con vistas" in text
    row_without_ai = next(line for line in text.splitlines() if line.startswith("| 1 |"))
    assert row_without_ai.count("—") == 2


def test_report_truncates_long_explanations(frame):
    frame.loc[0, "explicacion"] = "x" * 1000

    text = render_report(frame, frame.index[:1], METRICS, META, "ahora")

    longest = max(len(cell) for line in text.splitlines() for cell in line.split("|"))
    assert longest <= MAX_EXPLANATION_CHARS + 2


def test_report_warns_when_ai_was_disabled(frame):
    text = render_report(frame, frame.index[:1], {**METRICS, "ai_desactivada": True}, META, "x")

    assert "la IA se desactivó" in text


def test_report_ranks_barrios_with_enough_listings(frame):
    text = render_report(frame, frame.index[:1], METRICS, META, "x")

    barrios = text.split("## Barrios")[1].split("## Duración")[0]
    assert "| Sol | 5 |" in barrios
    assert "Goya" not in barrios  # solo 1 alojamiento


def test_report_with_empty_frame(frame):
    text = render_report(frame.iloc[0:0], pd.Index([]), METRICS, {}, "x")

    assert "desconocida" in text
    assert "## Barrios" not in text


def test_report_number_formatting(frame):
    text = render_report(
        frame, frame.index[:1], METRICS, {**META, "metricas_test": {"r2": 0.6224}}, "x"
    )

    assert "| R² del modelo en test | 0.6224 |" in text
    assert "| 0 | Sol | 500.00 | 150.00 | 350.00 | 0.9 |" in text
