import pytest

from airbnb_price import __version__
from airbnb_price.cli import main
from airbnb_price.exceptions import EXIT_INVALID_INPUT, EXIT_OK


def test_info_prints_config_without_secrets(monkeypatch, capsys):
    monkeypatch.setenv("AI_PROVIDER", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_secreto")

    code = main(["info"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "ai_provider: hf" in out
    assert "configurado" in out
    assert "hf_secreto" not in out


def test_invalid_config_returns_exit_code_2(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "desconocido")

    assert main(["info"]) == EXIT_INVALID_INPUT


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_is_required():
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
