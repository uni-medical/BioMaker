from __future__ import annotations

from webui.settings import Settings


def test_default_env_file_is_method_level_with_shared_and_legacy_overrides(
    tmp_path, monkeypatch
):
    workspace_root = tmp_path / "workspace"
    method_root = workspace_root / "BioMaker"
    method_root.mkdir(parents=True)
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("BIOMAKER_ENV_FILE", raising=False)
    monkeypatch.delenv("BIOMAKER_WEBUI_ENV_FILE", raising=False)

    assert Settings.default().env_file == (method_root / ".env").resolve()

    (workspace_root / ".env").write_text(
        "OPENAI_API_KEY=workspace\n", encoding="utf-8"
    )
    assert Settings.default().env_file == (method_root / ".env").resolve()

    monkeypatch.setenv("BIOMAKER_ENV_FILE", "config/provider.env")
    assert Settings.default().env_file == (
        method_root / "config" / "provider.env"
    ).resolve()

    monkeypatch.delenv("BIOMAKER_ENV_FILE")
    monkeypatch.setenv("BIOMAKER_WEBUI_ENV_FILE", "legacy/provider.env")
    assert Settings.default().env_file == (
        method_root / "legacy" / "provider.env"
    ).resolve()
