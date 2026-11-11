import pytest

from airbnb_price.data.loader import load_listings
from airbnb_price.exceptions import EXIT_INVALID_INPUT, DataLoadError


@pytest.mark.parametrize("name", ["listings.csv", "listings.csv.gz"])
def test_loads_plain_and_gzip_csv(tmp_path, listings_frame, name):
    path = tmp_path / name
    listings_frame.to_csv(path, index=False)

    loaded = load_listings(path)

    assert len(loaded) == 3
    assert loaded["id"].tolist() == ["1", "2", "3"]


def test_missing_file_raises_domain_error(tmp_path):
    with pytest.raises(DataLoadError) as exc_info:
        load_listings(tmp_path / "no_existe.csv")

    assert exc_info.value.exit_code == EXIT_INVALID_INPUT


def test_empty_file_raises_domain_error(tmp_path):
    path = tmp_path / "vacio.csv"
    path.write_text("")

    with pytest.raises(DataLoadError):
        load_listings(path)
