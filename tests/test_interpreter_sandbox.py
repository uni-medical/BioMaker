from __future__ import annotations

import os
import json
import platform
import shutil
import socket
from types import SimpleNamespace
from pathlib import Path

import pytest

from biomaker import interpreter as interpreter_module
from biomaker.interpreter import (
    Interpreter,
    SandboxConfigurationError,
    _sandbox_launch_plan,
    execution_sandbox_receipt,
)


def _sandbox_environment(data_root: Path) -> dict[str, str]:
    return {
        "BIOMAKER_EXECUTION_SANDBOX": "bwrap",
        "BIOMAKER_SANDBOX_READONLY_PATHS": str(data_root),
        "BIOMAKER_SANDBOX_NETWORK": "none",
        "CUDA_VISIBLE_DEVICES": "GPU-exact-selector",
        "OPENAI_API_KEY": "must-not-reach-candidate",
        "SERPER_API_KEY": "must-not-reach-candidate",
        "JINA_API_KEY": "must-not-reach-candidate",
        "CUSTOM_AUTH_TOKEN": "must-not-reach-candidate",
    }


def test_bwrap_plan_is_allowlisted_and_scrubs_candidate_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "formal" / "workspaces" / "node"
    input_dir = tmp_path / "formal" / "workspaces" / "run" / "input"
    data_root = tmp_path / "datasets" / "task"
    private_dir = tmp_path / "formal" / "logs" / "run" / "private_gt"
    workspace.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    data_root.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (workspace / "input").symlink_to(input_dir)
    (input_dir / "images").symlink_to(data_root)
    evaluator_spec = input_dir.parent / "evaluator_prediction_spec.json"
    evaluator_spec.write_text('{"primary_metric":"accuracy"}', encoding="utf-8")
    (workspace / "evaluator_prediction_spec.json").symlink_to(evaluator_spec)
    (private_dir / "test.csv").write_text("hidden\n", encoding="utf-8")

    fake_venv = tmp_path / "runtime" / ".venv"
    fake_uv = tmp_path / "home" / ".local" / "share" / "uv" / "python"
    fake_venv.mkdir(parents=True)
    fake_uv.mkdir(parents=True)
    monkeypatch.setattr(interpreter_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interpreter_module,
        "_bwrap_identity",
        lambda: (Path("/usr/bin/bwrap"), "bubblewrap 0.test"),
    )
    monkeypatch.setattr(
        interpreter_module,
        "_runtime_readonly_roots",
        lambda: [fake_venv, fake_uv],
    )
    monkeypatch.setattr(interpreter_module, "_system_readonly_binds", lambda: [])

    environment = _sandbox_environment(data_root)
    receipt = execution_sandbox_receipt(environment)
    plan = _sandbox_launch_plan(
        workspace,
        ["/runtime/python", "runfile.py"],
        {**os.environ, **environment},
    )

    assert plan is not None
    assert receipt["bwrap_path"] == "/usr/bin/bwrap"
    assert set(receipt["runtime_readonly_roots"]) == {str(fake_venv), str(fake_uv)}
    assert "--unshare-pid" in plan.command_prefix
    assert "--unshare-net" in plan.command_prefix
    assert receipt["network"] == plan.receipt["network"] == "none"
    assert "--dev-bind" in plan.command_prefix
    assert "--tmpfs" in plan.command_prefix
    assert str(workspace) in plan.command_prefix
    assert str(input_dir) in plan.command_prefix
    assert str(data_root) in plan.command_prefix
    assert str(evaluator_spec) in plan.command_prefix
    assert str(private_dir) not in plan.command_prefix
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "GPU-exact-selector"
    for name in (
        "OPENAI_API_KEY",
        "SERPER_API_KEY",
        "JINA_API_KEY",
        "CUSTOM_AUTH_TOKEN",
        "BIOMAKER_EXECUTION_SANDBOX",
        "BIOMAKER_SANDBOX_READONLY_PATHS",
        "BIOMAKER_SANDBOX_NETWORK",
    ):
        assert name not in plan.environment
    assert plan.environment["HOME"].startswith(str(workspace))
    assert plan.receipt["framework_readonly_binds"] == [
        {
            "source": str(evaluator_spec),
            "destination": str(evaluator_spec),
            "kind": "framework-contract",
        }
    ]
    assert set(plan.receipt["effective_scrubbed_environment_names"]) >= {
        "OPENAI_API_KEY",
        "SERPER_API_KEY",
        "JINA_API_KEY",
        "CUSTOM_AUTH_TOKEN",
    }


def test_bwrap_plan_rejects_private_gt_symlink_even_if_parent_is_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    private_dir = tmp_path / "authorized" / "private_gt"
    input_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (private_dir / "score_val.csv").write_text("label\n1\n", encoding="utf-8")
    (input_dir / "leak.csv").symlink_to(private_dir / "score_val.csv")

    monkeypatch.setattr(interpreter_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interpreter_module,
        "_bwrap_identity",
        lambda: (Path("/usr/bin/bwrap"), "bubblewrap 0.test"),
    )
    monkeypatch.setattr(interpreter_module, "_runtime_readonly_roots", lambda: [])
    monkeypatch.setattr(interpreter_module, "_system_readonly_binds", lambda: [])

    with pytest.raises(
        SandboxConfigurationError,
        match="evaluator-private data",
    ):
        _sandbox_launch_plan(
            workspace,
            ["python", "runfile.py"],
            {
                "BIOMAKER_EXECUTION_SANDBOX": "bwrap",
                "BIOMAKER_SANDBOX_READONLY_PATHS": str(private_dir.parent),
            },
        )


def test_requested_bwrap_fails_closed_when_platform_is_not_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(interpreter_module.platform, "system", lambda: "Darwin")

    result = Interpreter(
        tmp_path,
        enforce_environment=False,
        environment={"BIOMAKER_EXECUTION_SANDBOX": "bwrap"},
    ).run("print('must not execute')")

    assert result.exc_type == "SandboxConfigurationError"
    assert result.failure_phase == "environment_preflight"
    assert "supported only on Linux" in result.stderr
    assert not (tmp_path / "runfile.py").exists()


@pytest.mark.parametrize("network", ["", "unrestricted", "None!", "false"])
def test_bwrap_rejects_unknown_network_policy(network, monkeypatch):
    monkeypatch.setattr(interpreter_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interpreter_module,
        "_bwrap_identity",
        lambda: (Path("/usr/bin/bwrap"), "bubblewrap 0.test"),
    )
    with pytest.raises(SandboxConfigurationError, match="BIOMAKER_SANDBOX_NETWORK"):
        execution_sandbox_receipt(
            {
                "BIOMAKER_EXECUTION_SANDBOX": "bwrap",
                "BIOMAKER_SANDBOX_NETWORK": network,
            }
        )


def test_bwrap_host_network_is_explicitly_available(monkeypatch):
    monkeypatch.setattr(interpreter_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interpreter_module,
        "_bwrap_identity",
        lambda: (Path("/usr/bin/bwrap"), "bubblewrap 0.test"),
    )
    receipt = execution_sandbox_receipt(
        {
            "BIOMAKER_EXECUTION_SANDBOX": "bwrap",
            "BIOMAKER_SANDBOX_NETWORK": "host",
        }
    )
    assert receipt["network"] == "host"


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="real bubblewrap isolation test requires Linux and bwrap",
)
def test_real_bwrap_hides_private_ground_truth_and_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    private_dir = tmp_path / "private_gt"
    authorized_root = tmp_path / "unused-authorized-root"
    (workspace / "input").mkdir(parents=True)
    (workspace / "working").mkdir()
    private_dir.mkdir()
    authorized_root.mkdir()
    (workspace / "input" / "public.txt").write_text("public", encoding="utf-8")
    (private_dir / "test.csv").write_text("secret-label", encoding="utf-8")
    code = f"""
import os
from pathlib import Path

assert Path('input/public.txt').read_text() == 'public'
assert not Path({str(private_dir)!r}).exists()
assert 'OPENAI_API_KEY' not in os.environ
assert 'SERPER_API_KEY' not in os.environ
assert 'JINA_API_KEY' not in os.environ
assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-exact-selector'
try:
    Path('input/public.txt').write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('agent-visible input was writable')
Path('working/result.txt').write_text('ok')
"""

    result = Interpreter(
        workspace,
        timeout=10,
        enforce_environment=False,
        environment=_sandbox_environment(authorized_root),
    ).run(code)

    assert result.exc_type is None, result.candidate_output
    assert result.sandbox_receipt is not None
    assert (workspace / "working" / "result.txt").read_text() == "ok"
    assert (workspace / "input" / "public.txt").read_text() == "public"


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="real bubblewrap network test requires Linux and bwrap",
)
def test_real_offline_bwrap_cannot_reach_host_listener(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        # Verify the endpoint exists in the parent network namespace first.
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        code = f"""
import socket
try:
    conn = socket.create_connection(('127.0.0.1', {port}), timeout=1)
except OSError:
    print('host network inaccessible')
else:
    conn.close()
    raise AssertionError('offline candidate reached host network')
"""
        result = Interpreter(
            workspace,
            timeout=10,
            enforce_environment=False,
            environment={
                "BIOMAKER_EXECUTION_SANDBOX": "bwrap",
                "BIOMAKER_SANDBOX_NETWORK": "none",
            },
        ).run(code)
    assert result.exc_type is None, result.candidate_output
    assert result.sandbox_receipt["network"] == "none"


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="real task input isolation requires Linux and bwrap",
)
@pytest.mark.parametrize("final_fit", [False, True])
def test_real_subset_input_views_hide_omitted_frames_and_test_targets(
    tmp_path, final_fit
):
    from biomaker.utils.final_retrain import prepare_final_retrain_workspace

    workspace = tmp_path / "run"
    split_dir = workspace / "input" / "biomaker_splits"
    split_dir.mkdir(parents=True)
    public_data = tmp_path / "task_data"
    images = public_data / "images"
    targets = public_data / "train_targets"
    images.mkdir(parents=True)
    targets.mkdir()
    for name in ("train", "val", "test"):
        (images / f"{name}.png").write_bytes(b"synthetic-public-frame")
    (targets / "train.png").write_bytes(b"synthetic-train-target")
    (workspace / "input" / "images").symlink_to(images)
    (workspace / "input" / "train_targets").symlink_to(targets)
    omitted = tmp_path / "raw_source" / "omitted_sibling.png"
    omitted.parent.mkdir()
    omitted.write_bytes(b"synthetic-omitted-frame")
    private_dir = tmp_path / "private_gt"
    (private_dir / "targets").mkdir(parents=True)
    (private_dir / "targets" / "val.png").write_bytes(b"synthetic-val-target")
    (private_dir / "targets" / "test.png").write_bytes(b"synthetic-test-target")
    (split_dir / "agent_train.csv").write_text(
        "image_id,noisy_image_path,clean_image_path\n"
        "train,input/images/train.png,input/train_targets/train.png\n"
    )
    (split_dir / "score_val.csv").write_text(
        "image_id,noisy_image_path\nval,input/images/val.png\n"
    )
    (split_dir / "agent_test.csv").write_text(
        "image_id,noisy_image_path\ntest,input/images/test.png\n"
    )
    (private_dir / "score_val.csv").write_text(
        "image_id,clean_image_path\nval,private_gt/targets/val.png\n"
    )
    (split_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "key_columns": ["image_id"],
                "hidden_answers": {
                    "score_val": {"answer_columns": ["clean_image_path"]}
                },
            }
        )
    )
    if final_fit:
        workspace, manifest = prepare_final_retrain_workspace(
            SimpleNamespace(workspace_dir=workspace),
            private_dir,
        )
        assert manifest["score_rows_added"] == 1
    code = f"""
import csv
from pathlib import Path
assert sorted(p.name for p in Path('input/images').iterdir()) == ['test.png', 'train.png', 'val.png']
assert not Path({str(omitted)!r}).exists()
assert not Path({str(private_dir)!r}).exists()
rows = list(csv.DictReader(Path('input/biomaker_splits/agent_train.csv').open()))
assert len(rows) == {2 if final_fit else 1}
for row in rows:
    target = Path(row['clean_image_path'])
    assert target.is_file()
    assert target.read_bytes() != b'synthetic-test-target'
assert not any(p.name == 'omitted_sibling.png' for p in Path('input').rglob('*'))
print('subset input view verified')
"""
    result = Interpreter(
        workspace,
        timeout=15,
        enforce_environment=False,
        environment=_sandbox_environment(public_data),
    ).run(code)
    assert result.exc_type is None, result.candidate_output
    assert result.sandbox_receipt["network"] == "none"
