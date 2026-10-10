"""Exited leaders retain their real result only after verified group cleanup."""

import os
import time
from types import SimpleNamespace

import pytest

from biomaker import interpreter
from biomaker.utils import evaluator_setup
from biomaker.utils.process_tree import (
    ProcessTreeCleanupError,
    read_process_tree_state,
)


def invoke(entrypoint, tmp_path, code):
    if entrypoint == "evaluator":
        return evaluator_setup._run_python(tmp_path, code, timeout=10)
    return interpreter.Interpreter(
        tmp_path, timeout=10, enforce_environment=False
    ).run(code)


@pytest.mark.skipif(os.name != "posix", reason="POSIX unreaped child ownership")
@pytest.mark.parametrize("entrypoint", ["evaluator", "interpreter"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_exit_before_identity_capture_preserves_output_and_exit_code(
    tmp_path, monkeypatch, entrypoint, exit_code
):
    module = evaluator_setup if entrypoint == "evaluator" else interpreter
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    observations = []

    def exited_before_capture(pid):
        deadline = time.monotonic() + 5
        while read_process_tree_state(pid).lower() not in {"z", "x", "zombie", "dead"}:
            assert time.monotonic() < deadline, "child did not exit"
            time.sleep(.01)
        # macOS cannot read the session of this unreaped zombie. Inject the
        # same boundary on every POSIX platform without polling/reaping it.
        raise ProcessLookupError(pid)

    drain = module.drain_uncaptured_process_group

    def verified_drain(process):
        assert process.returncode is None
        drain(process)
        observations.append(process.returncode)

    monkeypatch.setattr(module.ProcessTreeTracker, "capture", exited_before_capture)
    monkeypatch.setattr(module, "drain_uncaptured_process_group", verified_drain)
    result = invoke(entrypoint, tmp_path,
        "import sys; print('stdout preserved'); "
        "print('stderr preserved', file=sys.stderr); "
        f"sys.exit({exit_code})")
    assert observations == [exit_code]
    assert result.stdout == "stdout preserved\n"
    assert result.stderr == "stderr preserved\n"
    if entrypoint == "evaluator":
        assert result.returncode == exit_code
    else:
        assert result.candidate_returncode == exit_code
        assert (result.exc_type is None) == (exit_code == 0)


@pytest.mark.parametrize("entrypoint", ["evaluator", "interpreter"])
def test_missing_capture_identity_never_bypasses_fatal_cleanup(
    tmp_path, monkeypatch, entrypoint
):
    module = evaluator_setup if entrypoint == "evaluator" else interpreter
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    process = SimpleNamespace(pid=50, returncode=None)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    failure = ProcessTreeCleanupError(50, (50,), "inspection unavailable")
    calls = []

    def missing(_pid):
        raise ProcessLookupError(50)

    def fail_drain(child):
        calls.append(child)
        raise failure

    monkeypatch.setattr(module.ProcessTreeTracker, "capture", missing)
    monkeypatch.setattr(module, "drain_uncaptured_process_group", fail_drain)
    with pytest.raises(ProcessTreeCleanupError) as caught:
        invoke(entrypoint, tmp_path, "print('not executed')")
    assert caught.value is failure
    assert calls == [process]
