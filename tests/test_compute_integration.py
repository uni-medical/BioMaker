from __future__ import annotations

import json
import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event
from types import SimpleNamespace

import pytest

from biomaker import interpreter as interpreter_module
from biomaker.interpreter import ExecutionResult, Interpreter
from biomaker.utils import config as config_module
from biomaker.utils import evaluator_setup as evaluator_setup_module
from biomaker.utils import final_retrain as final_retrain_module
from biomaker.utils import run_manifest
from biomaker.utils.compute import ComputePool, ComputeSlot


def _minimal_exec_config(parallel_workers):
    return SimpleNamespace(timeout=600, parallel_workers=parallel_workers)


def test_default_config_uses_auto_workers_and_explicit_integer_is_valid():
    default_cfg = config_module._load_cfg(use_cli_args=False)

    assert default_cfg.exec.parallel_workers == "auto"
    config_module._validate_exec_config(default_cfg)
    config_module._validate_exec_config(SimpleNamespace(exec=_minimal_exec_config(3)))


@pytest.mark.parametrize("invalid", ["1", "AUTO", "cuda", "", 0, -1, True])
def test_worker_config_rejects_invalid_strings_and_non_positive_values(invalid):
    cfg = SimpleNamespace(exec=_minimal_exec_config(invalid))

    with pytest.raises(
        ValueError,
        match="exec.parallel_workers must be 'auto' or an integer >= 1",
    ):
        config_module._validate_exec_config(cfg)


class _ImmediateProcess:
    pid = 43210

    def __init__(self):
        self.returncode = None

    def wait(self):
        self.returncode = 0
        return self.returncode


class _FakeProcessTree:
    def __init__(self, *, exited=True):
        self.refresh_calls = 0
        self.terminate_calls = []
        self.exited = exited
        self.drained = False

    def refresh_best_effort(self):
        self.refresh_calls += 1
        return True

    def root_has_exited(self, root_process):
        assert root_process.returncode is None
        return self.exited

    def terminate_and_wait(self, *, root_process):
        if self.drained:
            return
        self.terminate_calls.append(root_process)
        root_process.wait()
        self.drained = True


def test_interpreter_merges_cuda_env_and_waits_for_process_tree(tmp_path, monkeypatch):
    popen_calls = []
    process = _ImmediateProcess()
    process_tree = _FakeProcessTree()

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        kwargs["stdout"].write("candidate output\n")
        return process

    monkeypatch.setenv("BIOMAKER_TEST_PARENT_ENV", "preserved")
    monkeypatch.setattr(interpreter_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        interpreter_module, "_sample_cuda_total_gpu_memory_mb", lambda: None
    )
    monkeypatch.setattr(
        interpreter_module,
        "_sample_process_usage",
        lambda _pid, *, sample_gpu: None,
    )
    monkeypatch.setattr(
        interpreter_module.ProcessTreeTracker,
        "capture",
        lambda _pid: process_tree,
    )

    result = Interpreter(
        tmp_path,
        timeout=10,
        enforce_environment=False,
        environment={"CUDA_VISIBLE_DEVICES": "GPU-exact-selector"},
    ).run("print('ok')")

    assert result.exc_type is None
    assert len(popen_calls) == 1
    _, kwargs = popen_calls[0]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-exact-selector"
    assert kwargs["env"]["BIOMAKER_TEST_PARENT_ENV"] == "preserved"
    assert "start_new_session" not in kwargs
    assert kwargs["process_group"] == 0
    assert kwargs["stdout"] is not subprocess.PIPE
    assert kwargs["stderr"] is not subprocess.PIPE
    assert process_tree.refresh_calls == 1
    assert process_tree.terminate_calls == [process]
    assert not (tmp_path / "runfile.py").exists()


def test_single_slot_stays_leased_until_process_tree_drain_finishes(
    tmp_path, monkeypatch
):
    pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))
    process = _ImmediateProcess()
    drain_started = Event()
    allow_drain = Event()

    class BlockingProcessTree(_FakeProcessTree):
        def terminate_and_wait(self, *, root_process):
            if self.drained:
                return
            drain_started.set()
            assert allow_drain.wait(timeout=2)
            super().terminate_and_wait(root_process=root_process)

    process_tree = BlockingProcessTree()

    def popen(_command, **kwargs):
        kwargs["stdout"].write("candidate output\n")
        return process

    monkeypatch.setattr(interpreter_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        interpreter_module, "_sample_cuda_total_gpu_memory_mb", lambda: None
    )
    monkeypatch.setattr(
        interpreter_module,
        "_sample_process_usage",
        lambda _pid, *, sample_gpu: None,
    )
    monkeypatch.setattr(
        interpreter_module.ProcessTreeTracker,
        "capture",
        lambda _pid: process_tree,
    )

    def run_candidate():
        with pool.lease():
            return Interpreter(
                tmp_path,
                timeout=10,
                enforce_environment=False,
            ).run("print('ok')")

    with ThreadPoolExecutor(max_workers=2) as executor:
        candidate = executor.submit(run_candidate)
        assert drain_started.wait(timeout=1)
        next_lease = executor.submit(pool.acquire, 1)
        with pytest.raises(FutureTimeout):
            next_lease.result(timeout=0.05)

        allow_drain.set()
        assert candidate.result(timeout=1).exc_type is None
        acquired = next_lease.result(timeout=1)
        pool.release(acquired)

    assert pool.leased_count == 0


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.setsid()",
        "import os\nos.setpgid(0, 0)",
        "import os\nos.setpgrp()",
        "from os import setsid as detach\ndetach()",
        "import os\ndetach = os.setsid\ndetach()",
        "import os\ndetach = getattr(os, 'setsid')\ndetach()",
        "import os\ngetattr(os, 'setpgid')(0, 0)",
        ("import subprocess\n" "subprocess.Popen(['python'], start_new_session=True)"),
        "import subprocess\nsubprocess.Popen(['python'], process_group=0)",
        ("import subprocess\n" "subprocess.Popen(['python'], preexec_fn=lambda: None)"),
        "import subprocess\nsubprocess.run('setsid sleep 1', shell=True)",
        (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', "
            "'import os, time; os.setsid(); time.sleep(30)'])"
        ),
        (
            "import subprocess, sys\n"
            "child_code = 'from os import setsid as detach; detach()'\n"
            "command = [sys.executable, '-c', child_code]\n"
            "subprocess.Popen(command)"
        ),
        (
            "import subprocess as sp, sys\n"
            "sp.Popen([sys.executable, '-c', 'import os; os.setsid()'])"
        ),
        (
            "from subprocess import Popen as launch\n"
            "import sys\n"
            "launch([sys.executable, '-c', 'import os; os.setpgid(0, 0)'])"
        ),
    ],
)
def test_interpreter_rejects_process_group_escape_even_without_full_policy(
    tmp_path, code
):
    result = Interpreter(
        tmp_path,
        timeout=5,
        enforce_environment=False,
    ).run(code)

    assert result.exc_type == "ProcessContainmentError"
    assert "managed process group" in result.stderr
    assert result.failure_phase == "environment_preflight"
    assert not (tmp_path / "runfile.py").exists()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _descendant_spawning_code(child_pid_path, grandchild_pid_path) -> str:
    grandchild_code = "import time; time.sleep(30)"
    child_code = (
        "import pathlib, subprocess, sys, time\n"
        f"grandchild = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(grandchild.pid))\n"
        "time.sleep(30)\n"
    )
    return (
        "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        f"grandchild_path = pathlib.Path({str(grandchild_pid_path)!r})\n"
        "deadline = time.monotonic() + 10\n"
        "while not grandchild_path.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(30)\n"
    )


def _read_pid(path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_interpreter_timeout_drains_child_and_grandchild_before_return(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"

    result = Interpreter(
        tmp_path,
        timeout=2,
        enforce_environment=False,
    ).run(_descendant_spawning_code(child_pid_path, grandchild_pid_path))

    assert result.exc_type == "TimeoutError"
    child_pid = _read_pid(child_pid_path)
    grandchild_pid = _read_pid(grandchild_pid_path)
    assert not _pid_exists(child_pid)
    assert not _pid_exists(grandchild_pid)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_interpreter_success_drains_an_orphan_before_return(tmp_path):
    orphan_pid_path = tmp_path / "orphan.pid"
    code = (
        "import pathlib, subprocess, sys\n"
        "orphan = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(orphan_pid_path)!r}).write_text(str(orphan.pid))\n"
    )

    result = Interpreter(
        tmp_path,
        timeout=5,
        enforce_environment=False,
    ).run(code)

    assert result.exc_type is None
    assert not _pid_exists(_read_pid(orphan_pid_path))


def _pipe_inheriting_orphan_code(orphan_pid_path) -> str:
    return (
        "import pathlib, subprocess, sys\n"
        "orphan = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(orphan_pid_path)!r}).write_text(str(orphan.pid))\n"
        "print('leader completed', flush=True)\n"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_interpreter_success_drains_orphan_that_inherits_output_pipes(tmp_path):
    orphan_pid_path = tmp_path / "pipe-orphan.pid"

    result = Interpreter(
        tmp_path,
        timeout=4,
        enforce_environment=False,
    ).run(_pipe_inheriting_orphan_code(orphan_pid_path))

    assert result.exc_type is None
    assert "leader completed" in result.stdout
    assert not _pid_exists(_read_pid(orphan_pid_path))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_evaluator_timeout_drains_child_and_grandchild_before_return(tmp_path):
    child_pid_path = tmp_path / "evaluator-child.pid"
    grandchild_pid_path = tmp_path / "evaluator-grandchild.pid"

    with pytest.raises(subprocess.TimeoutExpired):
        evaluator_setup_module._run_python(
            tmp_path,
            _descendant_spawning_code(child_pid_path, grandchild_pid_path),
            timeout=2,
        )

    child_pid = _read_pid(child_pid_path)
    grandchild_pid = _read_pid(grandchild_pid_path)
    assert not _pid_exists(child_pid)
    assert not _pid_exists(grandchild_pid)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_evaluator_success_drains_orphan_that_inherits_output_pipes(tmp_path):
    orphan_pid_path = tmp_path / "evaluator-pipe-orphan.pid"

    completed = evaluator_setup_module._run_python(
        tmp_path,
        _pipe_inheriting_orphan_code(orphan_pid_path),
        timeout=4,
    )

    assert completed.returncode == 0
    assert "leader completed" in completed.stdout
    assert not _pid_exists(_read_pid(orphan_pid_path))


class _TimeoutProcess:
    pid = 54320

    def __init__(self):
        self.returncode = None

    def wait(self):
        self.returncode = -signal.SIGKILL
        return self.returncode


def test_interpreter_timeout_signals_known_descendant_tree(tmp_path, monkeypatch):
    process = _TimeoutProcess()
    process_tree = _FakeProcessTree(exited=False)
    clock = iter([0.0, 0.1, 2.0, 2.0, 2.0])

    monkeypatch.setattr(
        interpreter_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(interpreter_module.time, "time", lambda: next(clock))
    monkeypatch.setattr(
        interpreter_module, "_sample_cuda_total_gpu_memory_mb", lambda: None
    )
    monkeypatch.setattr(
        interpreter_module,
        "_sample_process_usage",
        lambda _pid, *, sample_gpu: None,
    )
    monkeypatch.setattr(
        interpreter_module.ProcessTreeTracker,
        "capture",
        lambda _pid: process_tree,
    )

    result = Interpreter(
        tmp_path,
        timeout=1,
        enforce_environment=False,
    ).run("while True: pass")

    assert result.exc_type == "TimeoutError"
    assert process_tree.terminate_calls == [process]
    assert not (tmp_path / "runfile.py").exists()


def _manifest_cfg(tmp_path):
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
    return SimpleNamespace(
        exp_name="1-compute-test",
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
        log_dir=tmp_path / "logs",
        model_for_all_agents=None,
        agent=agent,
        exec=SimpleNamespace(
            timeout=600,
            parallel_workers="auto",
            requested_parallel_workers="auto",
            effective_parallel_workers=2,
            compute_slots=[
                {"index": 0, "backend": "cuda", "selector": "GPU-a"},
                {"index": 1, "backend": "cuda", "selector": "GPU-b"},
            ],
            include_medical_packages=False,
        ),
        final_retrain=SimpleNamespace(enabled=True),
        report=SimpleNamespace(model="report-model"),
        copy_data=False,
        preprocess_data=False,
    )


def test_run_manifest_records_requested_effective_workers_and_slots(
    tmp_path, monkeypatch
):
    cfg = _manifest_cfg(tmp_path)
    cfg.log_dir.mkdir()
    cfg.workspace_dir.mkdir()
    monkeypatch.setattr(run_manifest, "environment_manifest_text", lambda **_: "facts")

    path = run_manifest.write_run_manifest(cfg, "Train a model")
    selected = json.loads(path.read_text(encoding="utf-8"))["config"]

    assert selected["parallel_workers"] == 2
    assert selected["requested_parallel_workers"] == "auto"
    assert selected["effective_parallel_workers"] == 2
    assert selected["compute_slots"] == [
        {"index": 0, "backend": "cuda", "selector": "GPU-a"},
        {"index": 1, "backend": "cuda", "selector": "GPU-b"},
    ]


def test_final_retrain_forwards_slot_env_to_all_subprocess_paths(tmp_path, monkeypatch):
    retrain_workspace = tmp_path / "workspace" / "final_retrain"
    retrain_workspace.mkdir(parents=True)
    (retrain_workspace / "working").mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    private_dir = log_dir / "private_gt"
    private_dir.mkdir()
    cfg = SimpleNamespace(
        workspace_dir=tmp_path / "workspace",
        log_dir=log_dir,
        exec=SimpleNamespace(
            format_tb_ipython=False,
            agent_file_name="runfile.py",
            enforce_environment=False,
            include_medical_packages=False,
        ),
    )
    best_node = SimpleNamespace(code="print('retrain')")
    expected_env = {"CUDA_VISIBLE_DEVICES": "GPU-final"}
    calls = {}

    class FakeInterpreter:
        def __init__(self, workspace, **kwargs):
            calls["interpreter"] = (workspace, kwargs)

        def run(self, code, reset_session):
            calls["run"] = (code, reset_session)
            return ExecutionResult(term_out=["ok"], exec_time=0.1, exc_type=None)

    monkeypatch.setattr(
        final_retrain_module,
        "prepare_final_retrain_workspace",
        lambda _cfg, _private: (
            retrain_workspace,
            {
                "train_rows_before": 10,
                "score_rows_added": 2,
                "train_rows_after": 12,
            },
        ),
    )
    monkeypatch.setattr(
        final_retrain_module, "clean_candidate_outputs", lambda *_: None
    )
    monkeypatch.setattr(final_retrain_module, "Interpreter", FakeInterpreter)
    monkeypatch.setattr(
        final_retrain_module, "normalize_working_outputs", lambda *_: []
    )
    monkeypatch.setattr(
        final_retrain_module,
        "validate_working_outputs",
        lambda *_: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        final_retrain_module,
        "attach_artifact_validation_result",
        lambda *_a, **_k: None,
    )

    def validate(workspace, *, timeout, env_overrides):
        calls["evaluator"] = (workspace, timeout, env_overrides)
        return SimpleNamespace(ok=True, error=None)

    monkeypatch.setattr(
        final_retrain_module, "validate_submission_with_evaluator", validate
    )
    monkeypatch.setattr(
        final_retrain_module, "attach_evaluator_run_result", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        final_retrain_module, "_copy_retrain_outputs_to_log", lambda *_: None
    )

    def private(workspace, private_path, *, env_overrides):
        calls["private"] = (workspace, private_path, env_overrides)
        return {"metric": 0.9}

    monkeypatch.setattr(final_retrain_module, "evaluate_private_outputs", private)
    monkeypatch.setattr(final_retrain_module, "_write_result", lambda *_: None)

    result = final_retrain_module.run_final_retrain(
        cfg,
        best_node,
        private_dir,
        timeout=300,
        env_overrides=expected_env,
    )

    assert result.ok
    assert calls["interpreter"] == (
        retrain_workspace,
        {
            "timeout": 300,
            "format_tb_ipython": False,
            "agent_file_name": "runfile.py",
            "enforce_environment": False,
            "include_medical_packages": False,
            "environment": expected_env,
        },
    )
    assert calls["evaluator"] == (retrain_workspace, 180, expected_env)
    assert calls["private"] == (retrain_workspace, private_dir, expected_env)

    monkeypatch.setattr(
        final_retrain_module,
        "_copy_retrain_outputs_to_log",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("copy failed")),
    )
    copy_failure = final_retrain_module.run_final_retrain(
        cfg,
        best_node,
        private_dir,
        timeout=300,
        env_overrides=expected_env,
    )
    assert copy_failure.ok is False
    assert copy_failure.error == "copy failed"

    monkeypatch.setattr(
        final_retrain_module, "_copy_retrain_outputs_to_log", lambda *_: None
    )
    monkeypatch.setattr(
        final_retrain_module,
        "evaluate_private_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private evaluation failed")
        ),
    )
    private_failure = final_retrain_module.run_final_retrain(
        cfg,
        best_node,
        private_dir,
        timeout=300,
        env_overrides=expected_env,
    )
    assert private_failure.ok is False
    assert private_failure.error == (
        "Private evaluation failed after final retrain: private evaluation failed"
    )
