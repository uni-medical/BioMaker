"""Ownership remains held when process cleanup cannot be verified."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from biomaker.utils.compute import ComputePool, ComputeSlot
from biomaker.utils.process_tree import (
    ProcessTreeCleanupError, ProcessTreeIdentity, ProcessTreeInspectionError,
    ProcessTreeTracker, drain_uncaptured_process_group,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


@pytest.mark.parametrize("failure_at", ["group", "identity"])
def test_repeated_inspection_failure_is_bounded_fatal_and_cached(failure_at):
    clock = Clock()
    reads = []
    def unavailable(_pgid):
        reads.append(True)
        raise ProcessTreeInspectionError("synthetic inspection unavailable")
    tracker = ProcessTreeTracker(
        ProcessTreeIdentity(50, 50, 40, "original"),
        group_reader=unavailable if failure_at == "group" else lambda _: [50],
        identity_reader=unavailable, sleeper=clock.sleep,
        signal_sender=lambda *_: pytest.fail("unknown identity must not be signalled"),
    )
    with pytest.raises(ProcessTreeCleanupError, match="synthetic inspection unavailable") as caught:
        tracker.terminate_and_wait(timeout=.1, monotonic=clock)
    assert not isinstance(caught.value, Exception)
    count = len(reads)
    with pytest.raises(ProcessTreeCleanupError) as again:
        tracker.terminate_and_wait(timeout=20, monotonic=clock)
    assert again.value is caught.value and len(reads) == count
    assert caught.value.as_dict()["cleanup_unverified"] is True


def test_empty_group_is_proven_without_reading_vanished_identity():
    tracker = ProcessTreeTracker(
        ProcessTreeIdentity(50, 50, 40, "original"),
        group_reader=lambda _: [],
        identity_reader=lambda _: pytest.fail("no live member requires an identity read"),
    )
    tracker.terminate_and_wait()
    # A later group allocation must not reactivate a completed tracker.
    tracker._group_reader = lambda _: pytest.fail("completed cleanup must not inspect reused PGID")
    tracker.terminate_and_wait()


def test_reused_leader_group_is_never_adopted_or_signalled():
    root = ProcessTreeIdentity(50, 50, 40, "original")
    clock = Clock()
    def identity(pid):
        return ProcessTreeIdentity(pid, 50, 40, "new")
    tracker = ProcessTreeTracker(
        root, identity_reader=identity,
        group_reader=lambda _: [50, 51],
        signal_sender=lambda *_: pytest.fail("unowned group must not be signalled"),
        sleeper=clock.sleep,
    )
    with pytest.raises(ProcessTreeCleanupError, match="reused") as caught:
        tracker.terminate_and_wait(timeout=.1, monotonic=clock)
    assert tracker.tracked_pids == (50,)
    # The newly observed leader can vanish while its unrelated children stay.
    # That must never reset the rejected group's ownership on a later refresh.
    tracker._identity_reader = lambda _: pytest.fail("rejected ownership is final")
    tracker._group_reader = lambda _: [51]
    with pytest.raises(ProcessTreeCleanupError) as again:
        tracker.refresh_best_effort()
    assert again.value is caught.value


def test_uncaptured_group_wait_uses_budget_and_never_signals_after_reap():
    clock = Clock()
    signals, waits = [], []
    process = SimpleNamespace(pid=50, returncode=None)
    def wait(timeout):
        waits.append(timeout)
        process.returncode = 0
    process.wait = wait
    def unavailable(_):
        raise ProcessTreeInspectionError("enumeration unavailable after reap")
    with pytest.raises(ProcessTreeCleanupError, match="enumeration unavailable"):
        drain_uncaptured_process_group(
            process, group_reader=unavailable,
            group_signal_sender=lambda *args: signals.append(args),
            sleeper=clock.sleep, monotonic=clock, timeout=.1,
        )
    assert signals and all(item == (50, signal.SIGKILL) for item in signals)
    assert waits == [], "inspection failure must not reap the identity-owning leader"


def test_uncaptured_wait_timeout_is_fatal_not_a_clean_result():
    clock = Clock()
    waits = []
    def wait(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired("synthetic child", timeout)
    process = SimpleNamespace(pid=50, returncode=None, wait=wait)
    with pytest.raises(ProcessTreeCleanupError, match="did not exit after group kill"):
        drain_uncaptured_process_group(
            process, group_reader=lambda _: [], group_signal_sender=lambda *_: None, timeout=.1,
            # Exercise the wait failure, not scheduler delay between empty-group reads.
            sleeper=clock.sleep, monotonic=clock,
        )
    assert waits == [pytest.approx(.05)]


@pytest.mark.parametrize("observation", ["empty", "live", "unavailable"])
def test_permission_denied_group_signal_still_requires_verified_empty_group(observation):
    clock = Clock()
    reads, waits = [], []
    process = SimpleNamespace(pid=50, returncode=None)

    def wait(timeout):
        assert reads[-2:] == [[], []]
        waits.append(timeout)
        process.returncode = 0

    def denied(_pgid, _signal):
        raise PermissionError("synthetic group signal denied")

    def inspect(_pgid):
        if observation == "unavailable":
            raise ProcessTreeInspectionError("synthetic group inspection denied")
        members = [] if observation == "empty" else [50, 51]
        reads.append(members)
        return members

    process.wait = wait
    kwargs = dict(group_reader=inspect, group_signal_sender=denied,
                  sleeper=clock.sleep, monotonic=clock, timeout=.1)
    if observation == "empty":
        drain_uncaptured_process_group(process, **kwargs)
        assert reads == [[], []]
        assert waits == [pytest.approx(.05)]
        assert process.returncode == 0
    else:
        reason = "signal denied" if observation == "live" else "inspection denied"
        with pytest.raises(ProcessTreeCleanupError, match=reason):
            drain_uncaptured_process_group(process, **kwargs)
        assert waits == []
        assert process.returncode is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS zombie session and signal semantics")
def test_uncaptured_real_macos_zombie_group_is_verified_before_reaping():
    import psutil
    from biomaker.utils.process_tree import read_process_tree_identity

    process = subprocess.Popen([sys.executable, "-I", "-B", "-c", "pass"], process_group=0)
    try:
        deadline = time.monotonic() + 5
        while psutil.Process(process.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "synthetic child did not exit"
            time.sleep(.01)
        assert process.returncode is None
        with pytest.raises(ProcessLookupError):
            read_process_tree_identity(process.pid)
        drain_uncaptured_process_group(process)
        assert process.returncode == 0
    finally:
        # This known child never forks; the test owns its final reap even if a
        # regression prevents the production cleanup from reaching wait().
        process.wait(timeout=5)


def test_fatal_cleanup_quarantines_slot_and_wakes_all_acquisition_kinds():
    pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))
    failure = ProcessTreeCleanupError(50, (50, 51), "inspection unavailable")
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ProcessTreeCleanupError), pool.lease() as slot:
            candidate = executor.submit(pool.acquire)
            final = executor.submit(pool.acquire_final)
            raise failure
        for future in (candidate, final):
            with pytest.raises(ProcessTreeCleanupError):
                future.result(timeout=1)
    assert pool.leased_count == 1
    with pytest.raises(RuntimeError, match="cannot release"):
        pool.release(slot)
    with pytest.raises(ProcessTreeCleanupError):
        pool.acquire(timeout=0)
    with pytest.raises(ProcessTreeCleanupError):
        pool.acquire_final(timeout=0)


def test_interpreter_does_not_repeat_uncaptured_fatal_cleanup(tmp_path, monkeypatch):
    from biomaker import interpreter
    failure = ProcessTreeCleanupError(50, (50,), "inspection unavailable")
    process = SimpleNamespace(pid=50)
    process.poll = lambda: pytest.fail("fatal drain must not fall through to poll/wait")
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    monkeypatch.setattr(interpreter.subprocess, "Popen", lambda *a, **k: process)
    def fail_capture(_):
        raise ProcessTreeInspectionError("capture failed")
    def fail_drain(_):
        raise failure
    monkeypatch.setattr(interpreter.ProcessTreeTracker, "capture", fail_capture)
    monkeypatch.setattr(interpreter, "drain_uncaptured_process_group", fail_drain)
    with pytest.raises(ProcessTreeCleanupError) as caught:
        interpreter.Interpreter(tmp_path, timeout=1).run("print('synthetic')")
    assert caught.value is failure
    assert (tmp_path / "runfile.py").exists()


def test_evaluator_does_not_repeat_uncaptured_fatal_cleanup(tmp_path, monkeypatch):
    from biomaker.utils import evaluator_setup
    failure = ProcessTreeCleanupError(50, (50,), "inspection unavailable")
    process = SimpleNamespace(pid=50)
    calls = []
    monkeypatch.setattr(evaluator_setup.subprocess, "Popen", lambda *a, **k: process)
    def fail_capture(_):
        raise ProcessTreeInspectionError("capture failed")
    def fail_drain(_):
        calls.append(True)
        raise failure
    monkeypatch.setattr(evaluator_setup.ProcessTreeTracker, "capture", fail_capture)
    monkeypatch.setattr(evaluator_setup, "drain_uncaptured_process_group", fail_drain)
    with pytest.raises(ProcessTreeCleanupError) as caught:
        evaluator_setup._run_python(tmp_path, "print('synthetic')", timeout=1)
    assert caught.value is failure
    assert calls == [True]


def test_fatal_cleanup_emits_failed_terminal_event(tmp_path, monkeypatch):
    import biomaker.run as run
    failure = ProcessTreeCleanupError(50, (50,), "inspection unavailable")
    events = []
    cfg = SimpleNamespace(log_dir=tmp_path)
    monkeypatch.setattr(run, "load_cfg", lambda: cfg)
    monkeypatch.setattr(run, "write_run_manifest", lambda *_: tmp_path / "manifest.json")
    monkeypatch.setattr(run, "append_event", lambda _, name, **payload: events.append((name, payload)))
    monkeypatch.setattr(run, "append_terminal_event", lambda _, name, **payload: events.append((name, payload)))
    def fail(_):
        raise failure
    monkeypatch.setattr(run, "_run", fail)
    with pytest.raises(ProcessTreeCleanupError):
        run._run_direct_entry()
    assert [name for name, _ in events] == ["run_manifest_written", "run_failed"]
    assert events[-1][1]["cleanup_unverified"] is True
    assert events[-1][1]["pgid"] == 50


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group/session ownership")
@pytest.mark.parametrize("inspection_failure", [False, True])
def test_real_tiny_metadata_subprocess_boundary_is_bounded(tmp_path, inspection_failure):
    # A separate bounded worker makes this a real subprocess boundary test and
    # preserves the admission unit tests' deliberately isolated scope.
    (tmp_path / "evaluator.py").write_text(
        "import os\ndef prediction_spec():\n"
        "    return {'synthetic': True, 'sid': os.getsid(0), 'pgid': os.getpgrp()}\n"
    )
    source = f'''
import faulthandler, json, os, sys, time
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from biomaker.utils import evaluator_setup as e
from biomaker.utils.process_tree import ProcessTreeCleanupError, ProcessTreeInspectionError, list_process_group_members
faulthandler.dump_traceback_later(10, exit=True)
started = time.monotonic()
seen = []
if {inspection_failure!r}:
    original = e.ProcessTreeTracker.capture
    original_drain = e.drain_uncaptured_process_group
    def unavailable(_):
        raise ProcessTreeInspectionError('injected process inspection failure')
    # The real child may become a zombie before capture finishes on macOS.
    # Inject the same inspection failure into both cleanup paths in advance.
    e.drain_uncaptured_process_group = lambda process: original_drain(
        process, group_reader=unavailable, timeout=.05)
    def capture(pid):
        seen.append(pid)
        tracker = original(pid)
        tracker._group_reader = unavailable
        terminate = tracker.terminate_and_wait
        tracker.terminate_and_wait = lambda **kw: terminate(timeout=.05, **kw)
        return tracker
    e.ProcessTreeTracker.capture = capture
    try:
        e._load_prediction_spec(Path({str(tmp_path)!r}))
    except ProcessTreeCleanupError as error:
        assert 'injected process inspection failure' in str(error)
        assert all(not list_process_group_members(pid) for pid in seen)
    else:
        raise AssertionError('cleanup failure was silently accepted')
else:
    for _ in range(5):
        spec = json.loads(e._load_prediction_spec(Path({str(tmp_path)!r})))
        assert spec['synthetic'] and spec['sid'] == os.getsid(0)
        assert spec['pgid'] != os.getpgrp()
faulthandler.cancel_dump_traceback_later()
print(json.dumps({{'passed': True, 'inspection_failure': {inspection_failure!r}, 'seconds': time.monotonic()-started}}))
'''
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", source],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["passed"] is True
