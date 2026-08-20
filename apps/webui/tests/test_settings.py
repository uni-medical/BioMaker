from __future__ import annotations

from webui.settings import Settings


def test_default_env_file_is_workspace_level_with_explicit_override(
    tmp_path, monkeypatch
):
    workspace_root = tmp_path / "workspace"
    method_root = workspace_root / "BioMaker"
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("BIOMAKER_WEBUI_ENV_FILE", raising=False)

    assert Settings.default().env_file == (workspace_root / ".env").resolve()

    monkeypatch.setenv("BIOMAKER_WEBUI_ENV_FILE", "config/provider.env")
    assert Settings.default().env_file == (
        workspace_root / "config" / "provider.env"
    ).resolve()
