from __future__ import annotations

import json
import shutil
import stat
import tempfile
from pathlib import Path

from webui.env_file import (
    masked_env,
    read_managed_values,
    subprocess_environment,
    update_env_file,
)


def test_env_update_preserves_unmanaged_lines_and_never_returns_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# local settings\n"
        "UNRELATED=value\n"
        "OPENAI_API_KEY=old-key\n"
        "OPENAI_API_KEY=duplicate-old-key\n"
        "SERPER_API_KEY=remove-me\n",
        encoding="utf-8",
    )

    update_env_file(
        path,
        {
            "OPENAI_API_KEY": "new-secret-123456",
            "OPENAI_BASE_URL": "https://api.example.test/v1",
            "SERPER_API_KEY": None,
        },
    )

    rendered = path.read_text(encoding="utf-8")
    values = read_managed_values(path)
    masked = masked_env(path)
    assert "# local settings" in rendered
    assert "UNRELATED=value" in rendered
    assert rendered.count("OPENAI_API_KEY=") == 1
    assert "SERPER_API_KEY" not in rendered
    assert values["OPENAI_API_KEY"] == "new-secret-123456"
    assert values["OPENAI_BASE_URL"] == "https://api.example.test/v1"
    assert "new-secret-123456" not in json.dumps(masked)
    assert masked["OPENAI_API_KEY"] == {
        "configured": True,
        "masked": "********",
        "secret": True,
    }
    assert masked["OPENAI_BASE_URL"] == {
        "configured": True,
        "masked": "https://api.example.test/v1",
        "value": "https://api.example.test/v1",
        "secret": False,
    }


def test_model_settings_are_editable_and_loaded_into_subprocess(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "BIOMAKER_MAIN_MODEL=main-from-file\n"
        "BIOMAKER_HELPER_MODEL=helper-from-file\n"
        "OPENAI_MODEL=fallback-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BIOMAKER_MAIN_MODEL", "inherited-main")
    monkeypatch.setenv("OPENAI_MODEL", "inherited-fallback")

    update_env_file(
        path,
        {
            "BIOMAKER_MAIN_MODEL": "selected-model",
            "BIOMAKER_HELPER_MODEL": "selected-model",
        },
    )

    environment = subprocess_environment(path)
    assert environment["BIOMAKER_MAIN_MODEL"] == "selected-model"
    assert environment["BIOMAKER_HELPER_MODEL"] == "selected-model"
    assert environment["OPENAI_MODEL"] == "fallback-from-file"

    empty_path = tmp_path / "missing.env"
    inherited_environment = subprocess_environment(empty_path)
    assert inherited_environment["BIOMAKER_MAIN_MODEL"] == "inherited-main"
    assert inherited_environment["OPENAI_MODEL"] == "inherited-fallback"


def test_new_env_file_uses_private_permissions():
    # The workspace's configured temp root may be a Windows mount that cannot
    # represent POSIX modes, while BioMaker/.env lives on the Linux filesystem.
    temporary_dir = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        path = temporary_dir / ".env"
        update_env_file(path, {"JINA_API_KEY": "secret"})
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        shutil.rmtree(temporary_dir)
