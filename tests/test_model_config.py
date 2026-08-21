from pathlib import Path

from biomaker.utils import config as config_module


def _config_for_test(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = config_module._load_cfg(use_cli_args=False)
    cfg.data_dir = str(data_dir)
    cfg.desc_file = None
    cfg.goal = "Test model configuration"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "model-config"
    cfg.agent.enable_web_search = False
    cfg.agent.enable_monai_knowledge_base = False
    return cfg


def _configured_models(cfg) -> set[str]:
    return {
        cfg.agent.code.model,
        cfg.agent.feedback.model,
        cfg.agent.knowledge_base.model,
        cfg.report.model,
    }


def test_legacy_model_environment_variables_are_ignored(tmp_path, monkeypatch):
    for key in (
        "BIOMAKER_MAIN_MODEL",
        "BIOMAKER_HELPER_MODEL",
        "OPENAI_MODEL",
        "OPENAI_HELPER_MODEL",
    ):
        monkeypatch.setenv(key, "legacy-model")
    monkeypatch.setattr(config_module, "load_local_env", lambda: tmp_path / ".env")

    cfg = _config_for_test(tmp_path)

    prepared = config_module.prep_cfg(cfg)

    assert _configured_models(prepared) == {"deepseek-v4-flash"}


def test_model_for_all_agents_overrides_every_role(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "load_local_env", lambda: tmp_path / ".env")
    cfg = _config_for_test(tmp_path)
    cfg.model_for_all_agents = "task-selected-model"

    prepared = config_module.prep_cfg(cfg)

    assert _configured_models(prepared) == {"task-selected-model"}
