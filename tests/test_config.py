"""环境配置加载测试。"""

import os
import subprocess
import sys
from pathlib import Path

from pytest import MonkeyPatch

from stockg.infrastructure.config import load_environment


def test_load_environment_reads_explicit_env_file(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("STOCKG_TEST_VALUE=from-env-file\n", encoding="utf-8")
    monkeypatch.delenv("STOCKG_TEST_VALUE", raising=False)

    load_environment(env_file)

    assert os.environ["STOCKG_TEST_VALUE"] == "from-env-file"


def test_load_environment_preserves_process_environment(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("STOCKG_TEST_VALUE=from-env-file\n", encoding="utf-8")
    monkeypatch.setenv("STOCKG_TEST_VALUE", "from-process")

    load_environment(env_file)

    assert os.environ["STOCKG_TEST_VALUE"] == "from-process"


def test_secret_settings_have_no_source_fallback() -> None:
    process_env = os.environ.copy()
    process_env["DEEPSEEK_API_KEY"] = ""
    process_env["FINNHUB_API_KEY"] = ""
    command = (
        "from stockg.infrastructure.config import "
        "DEEPSEEK_API_KEY, FINNHUB_API_KEY; "
        "assert DEEPSEEK_API_KEY == ''; "
        "assert FINNHUB_API_KEY == ''"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
