import json

import pytest

from airbnb_price.data.loader import load_json_records
from airbnb_price.data.validator import validate_prediction_input
from airbnb_price.exceptions import DataLoadError, DataValidationError

MINIMAL = {"latitude": 40.42, "longitude": -3.70, "room_type": "Private room", "accommodates": 2}


def test_minimal_record_gets_optional_columns_and_id():
    frame = validate_prediction_input([MINIMAL])

    assert frame.loc[0, "id"] == "0"
    assert "bathrooms_text" in frame.columns
    assert frame["amenities"].isna().all()


@pytest.mark.parametrize(
    "records",
    [
        [],
        ["no es un objeto"],
        [{"latitude": 40.42, "longitude": -3.70, "room_type": "Private room"}],
        [{**MINIMAL, "accommodates": None}],
        [{**MINIMAL, "accommodates": 40}],
        [{**MINIMAL, "latitude": 41.5}],
    ],
)
def test_invalid_records_raise_domain_error(records):
    with pytest.raises(DataValidationError):
        validate_prediction_input(records)


def test_load_json_records_accepts_object_or_list(tmp_path):
    single, many = tmp_path / "uno.json", tmp_path / "varios.json"
    single.write_text(json.dumps(MINIMAL), encoding="utf-8")
    many.write_text(json.dumps([MINIMAL, MINIMAL]), encoding="utf-8")

    assert len(load_json_records(single)) == 1
    assert len(load_json_records(many)) == 2


def test_load_json_records_errors(tmp_path):
    broken = tmp_path / "roto.json"
    broken.write_text("{no json", encoding="utf-8")

    with pytest.raises(DataLoadError):
        load_json_records(broken)
    with pytest.raises(DataLoadError):
        load_json_records(tmp_path / "no_existe.json")
