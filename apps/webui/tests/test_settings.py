from __future__ import annotations

from webui.settings import Settings


def test_default_queue_domain_uses_parent_of_inner_method_repo(tmp_path, monkeypatch):
    shared_workspace = tmp_path / "BioMaker-workspace"
    method_root = shared_workspace / "BioMaker"
    method_root.mkdir(parents=True)
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.delenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("BIOMAKER_WEBUI_STATE_DIR", raising=False)

    settings = Settings.default()

    assert settings.workspace_root == shared_workspace.resolve()
    assert settings.state_dir == (shared_workspace / ".biomaker").resolve()
    assert settings.logs_root == (shared_workspace / "logs").resolve()
    assert settings.workspaces_root == (shared_workspace / "workspaces").resolve()
    assert settings.experiments_root == (shared_workspace / "experiments").resolve()
    assert settings.datasets_root == (shared_workspace / "datasets").resolve()


def test_queue_domain_environment_overrides_are_deterministic(tmp_path, monkeypatch):
    method_root = tmp_path / "checkout" / "BioMaker"
    method_root.mkdir(parents=True)
    workspace_override = tmp_path / "shared"
    state_override = tmp_path / "queue-state"
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", str(workspace_override))
    monkeypatch.setenv("BIOMAKER_WEBUI_STATE_DIR", str(state_override))

    settings = Settings.default()

    assert settings.workspace_root == workspace_override.resolve()
    assert settings.state_dir == state_override.resolve()
    assert settings.logs_root == (workspace_override / "logs").resolve()
    assert settings.workspaces_root == (workspace_override / "workspaces").resolve()


def test_default_env_file_is_method_level_with_explicit_override(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    method_root = workspace_root / "BioMaker"
    method_root.mkdir(parents=True)
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("BIOMAKER_WEBUI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("BIOMAKER_ENV_FILE", raising=False)

    assert Settings.default().env_file == (method_root / ".env").resolve()

    (workspace_root / ".env").write_text("OPENAI_API_KEY=workspace\n", encoding="utf-8")
    assert Settings.default().env_file == (method_root / ".env").resolve()

    monkeypatch.setenv("BIOMAKER_ENV_FILE", "config/provider.env")
    assert (
        Settings.default().env_file
        == (method_root / "config" / "provider.env").resolve()
    )
