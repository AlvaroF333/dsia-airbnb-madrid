import subprocess
import sys

import pytest


@pytest.mark.integration
def test_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "airbnb_price", "info"],
        capture_output=True,
        text=True,
        check=False,
        env={"AI_PROVIDER": "mock", "PATH": ""},
    )

    assert result.returncode == 0, result.stderr
    assert "ai_provider: mock" in result.stdout
