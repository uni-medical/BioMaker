import json
from types import SimpleNamespace

import pytest

from biomaker.utils import run_manifest


@pytest.mark.parametrize("private_per_node", [None, True, False])
def test_run_manifest_preserves_identity_across_atomic_updates(
    tmp_path, monkeypatch, private_per_node
):
    workspace = tmp_path / "workspaces" / "1-test"
    workspace.mkdir(parents=True)
    log_dir = tmp_path / "logs" / "1-test"
    log_dir.mkdir(parents=True)
    knowledge = SimpleNamespace(
        model="helper-model",
        retrieval_mode="keyword",
        embedding_model="embedding-model",
        topk=5,
    )
    agent = SimpleNamespace(
        steps=35,
        total_time_limit=86400,
        code=SimpleNamespace(model="main-model"),
        feedback=SimpleNamespace(model="feedback-model"),
        knowledge_base=knowledge,
        enable_node_tools=True,
        enable_monai_knowledge_base=False,
        enable_web_search=False,
    )
    cfg = SimpleNamespace(
        exp_name="1-test",
        data_dir=tmp_path / "data",
        workspace_dir=workspace,
        log_dir=log_dir,
        model_for_all_agents=None,
        agent=agent,
        exec=SimpleNamespace(
            timeout=600,
            parallel_workers=1,
            include_medical_packages=False,
        ),
        final_retrain=SimpleNamespace(enabled=True),
        report=SimpleNamespace(model="report-model"),
        copy_data=False,
        preprocess_data=False,
    )
    if private_per_node is not None:
        cfg.private_evaluation = SimpleNamespace(per_node=private_per_node)
    monkeypatch.setenv("BIOMAKER_LAUNCH_TOKEN", "webui-run-token")
    monkeypatch.setattr(run_manifest, "environment_manifest_text", lambda **_: "facts")

    path = run_manifest.write_run_manifest(cfg, None)
    initial = json.loads(path.read_text(encoding="utf-8"))
    updated_path = run_manifest.write_run_manifest(cfg, "Train a classifier")
    updated = json.loads(updated_path.read_text(encoding="utf-8"))

    assert path == updated_path
    assert updated["created_at"] == initial["created_at"]
    assert updated["launch_token"] == "webui-run-token"
    assert updated["task_description"] == "Train a classifier"
    assert updated["config"]["model_for_all_agents"] is None
    assert updated["config"]["main_model"] == "main-model"
    assert updated["config"]["knowledge_base_model"] == "helper-model"
    assert updated["config"]["report_model"] == "report-model"
    assert updated["config"]["private_evaluation_protocol"] == "post_selection_once"
    assert "private_evaluation_per_node" not in updated["config"]
    assert len(updated["engine"]["package_source_sha256"]) == 64
    assert updated["engine"]["version"]
    assert len(updated["engine"]["lock_sha256"]) == 64
    assert not list(log_dir.glob("*.tmp"))
