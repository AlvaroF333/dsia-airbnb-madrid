import json

import pandas as pd
import pytest

from airbnb_price.cli import main
from airbnb_price.exceptions import EXIT_INVALID_INPUT, EXIT_OK
from tests.conftest import make_listing

pytestmark = pytest.mark.integration


@pytest.fixture
def raw_csv(tmp_path, listings_frame):
    """CSV comprimido con 3 filas válidas y 1 sin precio."""
    frame = pd.concat(
        [listings_frame, pd.DataFrame([make_listing(id="4", price=None)])], ignore_index=True
    )
    path = tmp_path / "raw" / "listings.csv.gz"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


def run_ingest_cli(raw_csv, tmp_path, *extra):
    output = tmp_path / "processed" / "listings_clean.csv"
    metrics = tmp_path / "reports" / "ingest_metrics.json"
    code = main(
        ["ingest", "--input", str(raw_csv), "--output", str(output), "--metrics", str(metrics)]
        + list(extra)
    )
    return code, output, metrics


def test_ingest_writes_clean_data_and_metrics(raw_csv, tmp_path):
    code, output, metrics_path = run_ingest_cli(raw_csv, tmp_path)

    assert code == EXIT_OK
    assert len(pd.read_csv(output)) == 3
    metrics = json.loads(metrics_path.read_text())
    assert metrics["rows_in"] == 4
    assert metrics["rows_out"] == 3
    assert metrics["error_rate"] == 0.25
    assert metrics["rejects"]["price_missing"] == 1


def test_ingest_is_idempotent(raw_csv, tmp_path):
    _, output, metrics_path = run_ingest_cli(raw_csv, tmp_path)
    first = (output.read_bytes(), metrics_path.read_bytes())

    run_ingest_cli(raw_csv, tmp_path)

    assert (output.read_bytes(), metrics_path.read_bytes()) == first


def test_error_rate_gate_fails_with_exit_2_and_keeps_metrics(raw_csv, tmp_path):
    code, output, metrics_path = run_ingest_cli(raw_csv, tmp_path, "--max-error-rate", "0.1")

    assert code == EXIT_INVALID_INPUT
    assert not output.exists()
    assert json.loads(metrics_path.read_text())["error_rate"] == 0.25


def test_missing_input_returns_exit_2(tmp_path):
    code, _, _ = run_ingest_cli(tmp_path / "no_existe.csv.gz", tmp_path)

    assert code == EXIT_INVALID_INPUT


def test_missing_column_returns_exit_2(tmp_path, listings_frame):
    path = tmp_path / "listings.csv"
    listings_frame.drop(columns=["price"]).to_csv(path, index=False)

    code, _, _ = run_ingest_cli(path, tmp_path)

    assert code == EXIT_INVALID_INPUT


def test_default_paths_come_from_settings(monkeypatch, tmp_path, raw_csv):
    monkeypatch.setenv("AIRBNB_PRICE_HOME", str(tmp_path))
    (tmp_path / "data").mkdir()
    raw_csv.parent.rename(tmp_path / "data" / "raw")

    assert main(["ingest"]) == EXIT_OK
    assert (tmp_path / "data" / "processed" / "listings_clean.csv").is_file()
    assert (tmp_path / "reports" / "ingest_metrics.json").is_file()
