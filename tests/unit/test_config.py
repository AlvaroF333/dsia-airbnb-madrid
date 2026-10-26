from pathlib import Path

import pytest

from airbnb_price.config import Settings
from airbnb_price.exceptions import EXIT_INVALID_INPUT, ConfigError


def test_defaults_use_mock_provider():
    settings = Settings.from_env({})

    assert settings.ai_provider == "mock"
    assert settings.hf_token is None
    assert settings.log_level == "INFO"


def test_hf_provider_with_token():
    settings = Settings.from_env({"AI_PROVIDER": "HF", "HF_TOKEN": "hf_test"})

    assert settings.ai_provider == "hf"
    assert settings.hf_token == "hf_test"


@pytest.mark.parametrize(
    "env",
    [
        {"AI_PROVIDER": "openai"},
        {"AI_PROVIDER": "hf"},
        {"AI_PROVIDER": "hf", "HF_TOKEN": "   "},
        {"LOG_LEVEL": "verbose"},
    ],
)
def test_invalid_config_raises_domain_error(env):
    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env(env)

    assert exc_info.value.exit_code == EXIT_INVALID_INPUT


def test_paths_derive_from_home(tmp_path):
    settings = Settings.from_env({"AIRBNB_PRICE_HOME": str(tmp_path)})

    assert settings.home == tmp_path
    assert settings.data_dir == tmp_path / "data"
    assert settings.models_dir == tmp_path / "models"
    assert settings.reports_dir == tmp_path / "reports"


def test_default_home_is_project_root():
    settings = Settings.from_env({})

    assert (settings.home / "pyproject.toml").is_file()
    assert isinstance(settings.home, Path)
