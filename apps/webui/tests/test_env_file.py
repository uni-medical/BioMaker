from __future__ import annotations

import json
import shutil
import stat
import tempfile
from pathlib import Path

from webui.env_file import (
    mask_value,
    masked_env,
    read_env_values,
    resolve_env_file,
    subprocess_environment,
    update_env_file,
)


def test_env_file_resolution_is_independent_of_cwd(tmp_path, monkeypatch):
    method_root = tmp_path / "BioMaker"
    elsewhere = tmp_path / "elsewhere"
    method_root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("BIOMAKER_ENV_FILE", raising=False)

    assert resolve_env_file(method_root) == (method_root / ".env").resolve()

    monkeypatch.setenv("BIOMAKER_ENV_FILE", "secrets/provider.env")
    assert (
        resolve_env_file(method_root)
        == (method_root / "secrets" / "provider.env").resolve()
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
    values = read_env_values(path)
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
        "masked": "••••••••",
        "secret": True,
    }
    assert masked["OPENAI_BASE_URL"] == {
        "configured": True,
        "masked": "https://api.example.test/v1",
        "value": "https://api.example.test/v1",
        "secret": False,
    }


def test_secret_mask_preserves_only_known_prefixes():
    black_dots = "•" * 8
    assert mask_value("") == ""
    assert mask_value("sk-secret-value") == f"sk-{black_dots}"
    assert mask_value("sk_secret-value") == f"sk_{black_dots}"
    assert mask_value("jina_secret-value") == f"jina_{black_dots}"
    assert mask_value("jina-secret-value") == f"jina-{black_dots}"


def test_secret_mask_hides_unknown_prefix_suffix_and_total_length():
    black_dots = "•" * 8
    assert mask_value("secret") == black_dots
    assert mask_value("other-prefix-with-a-very-long-secret-value") == black_dots
    assert mask_value("SK-uppercase-is-not-a-known-prefix") == black_dots


def test_model_settings_are_editable_and_loaded_into_subprocess(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "BIOMAKER_MAIN_MODEL=main-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BIOMAKER_MAIN_MODEL", "inherited-main")

    update_env_file(
        path,
        {"BIOMAKER_MAIN_MODEL": "selected-model"},
    )

    environment = subprocess_environment(path)
    assert environment["BIOMAKER_MAIN_MODEL"] == "selected-model"

    empty_path = tmp_path / "missing.env"
    inherited_environment = subprocess_environment(empty_path)
    assert inherited_environment["BIOMAKER_MAIN_MODEL"] == "inherited-main"


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
