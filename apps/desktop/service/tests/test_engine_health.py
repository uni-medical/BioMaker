from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from test_desktop_service import config, fake_uv, ready_installer, SERVICE, REPOSITORY
from desktop_service import DesktopConfig, EngineInstaller
import desktop_service


def interpreter(engine, body):
    engine.interpreter.parent.mkdir(parents=True, exist_ok=True)
    engine.interpreter.write_text(f"#!{sys.executable}\n" + body)
    engine.interpreter.chmod(0o755)


def finished(engine, seconds=8):
    deadline = time.monotonic() + seconds
    while engine.snapshot()["operation"] and time.monotonic() < deadline:
        time.sleep(0.01)
    state = engine.snapshot()
    assert state["operation"] is None, state
    return state


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_legacy_ready_receipt_does_not_claim_a_completed_health_check(config):
    engine = ready_installer(config)
    assert engine.snapshot() | {} == {
        "ready": True, "installing": False, "checking": False, "operation": None,
        "health": "unknown", "checked_at": None, "busy_external": False, "maintenance_unavailable": False, "error": None, "log": "",
    }


def test_check_uses_isolated_bounded_verifier_without_install_or_receipt_rewrite(config, monkeypatch):
    engine = ready_installer(config)
    interpreter(engine, "import json,pathlib,sys\npathlib.Path('verify-args.json').write_text(json.dumps(sys.argv[1:]))\nprint('verified')\n")
    before = (engine.receipt.read_bytes(), engine.receipt.stat().st_mtime_ns)
    monkeypatch.setattr(desktop_service, "provision_native_runtime", lambda *_args, **_kwargs: pytest.fail("check must not provision"))
    fake_uv(config, "raise AssertionError('check must not install')\n")
    try:
        state = engine.check()
        assert state["checking"] and not state["installing"] and not state["ready"]
        state = finished(engine)
        assert state["ready"] and state["health"] == "healthy" and state["checked_at"].endswith("+00:00")
        assert json.loads((config.engine_root / "verify-args.json").read_text()) == ["-I", "-B", str(SERVICE / "verify_engine.py")]
        assert (engine.receipt.read_bytes(), engine.receipt.stat().st_mtime_ns) == before
    finally:
        engine.close()


@pytest.mark.parametrize("missing", ["interpreter", "receipt", "invalid_receipt"])
def test_check_rejects_missing_or_invalid_environment_and_invalidates_readiness(config, missing):
    engine = ready_installer(config)
    if missing == "interpreter":
        engine.interpreter.unlink()
    elif missing == "receipt":
        engine.receipt.unlink()
    else:
        engine.receipt.write_text("[]")
    try:
        engine.check()
        state = finished(engine)
        assert state["health"] == "failed" and not state["ready"] and state["checked_at"]
        assert not engine.receipt.exists()
        assert not EngineInstaller(config).snapshot()["ready"]
    finally:
        engine.close()


def test_real_check_failure_is_redacted_and_cannot_be_trusted_after_restart(config):
    engine = ready_installer(config)
    config.env_file.write_text('OPENAI_API_KEY="private-health-test-token"\n')
    interpreter(engine, "import sys\nprint('bad private-health-test-token sk-secret-secret-secret')\nsys.exit(7)\n")
    try:
        engine.check()
        state = finished(engine)
        assert "7" in state["error"] and state["health"] == "failed"
        assert "private-health-test-token" not in state["log"] and "sk-secret-secret-secret" not in state["log"]
        assert not engine.receipt.exists()
        assert not EngineInstaller(config).snapshot()["ready"]
    finally:
        engine.close()


def test_verification_timeout_terminates_real_process_and_invalidates_receipt(config):
    engine = ready_installer(config)
    engine.VERIFY_TIMEOUT_SECONDS = 0.15
    engine.TERMINATE_GRACE_SECONDS = 0.15
    interpreter(engine, "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint('checking',flush=True)\ntime.sleep(30)\n")
    try:
        engine.check()
        wait_for(lambda: engine._process is not None)
        child = engine._process
        state = finished(engine)
        assert state["health"] == "failed" and "timed out" in state["error"]
        assert child.poll() is not None and not engine.receipt.exists()
        assert not EngineInstaller(config).snapshot()["ready"]
    finally:
        engine.close()


def test_check_cancel_and_all_concurrent_operations_share_one_worker(config):
    engine = ready_installer(config)
    interpreter(engine, "import time\nprint('checking',flush=True)\ntime.sleep(30)\n")
    try:
        engine.check()
        wait_for(lambda: "checking" in engine.snapshot()["log"])
        child, worker = engine._process, engine._thread
        for action in (engine.check, engine.start, engine.repair):
            assert action()["operation"] == "check"
            assert engine._process is child and engine._thread is worker
        other = EngineInstaller(config)
        for action in (other.snapshot, other.check, other.start, other.repair):
            state = action()
            assert state["busy_external"] and not state["ready"] and state["operation"] is None
        engine.cancel()
        state = finished(engine)
        assert state["error"] == "Engine check cancelled."
        assert state["health"] == "unknown" and state["checked_at"] is None and not state["ready"]
        assert child.poll() is not None and engine.receipt.exists()
        assert not other.snapshot()["busy_external"]
    finally:
        engine.close()


def test_repair_forces_frozen_reinstall_then_provisions_and_verifies(config, monkeypatch):
    engine = ready_installer(config)
    interpreter(engine, "import pathlib,sys,json\nassert pathlib.Path('provisioned').exists()\npathlib.Path('verify-args.json').write_text(json.dumps(sys.argv[1:]))\n")
    fake_uv(config, "import pathlib,sys,json\nassert not pathlib.Path('.desktop-engine.json').exists()\npathlib.Path('uv-args.json').write_text(json.dumps(sys.argv[1:]))\n")
    def provision(root, run):
        assert (root / "uv-args.json").exists()
        (root / "provisioned").write_text("synthetic")
        return {"test_native_verified": True}
    monkeypatch.setattr(desktop_service, "provision_native_runtime", provision)
    try:
        assert engine.repair()["installing"]
        state = finished(engine)
        assert state["ready"] and state["health"] == "healthy"
        command = json.loads((config.engine_root / "uv-args.json").read_text())
        assert command[:3] == ["sync", "--frozen", "--reinstall"]
        assert json.loads((config.engine_root / "verify-args.json").read_text())[:2] == ["-I", "-B"]
        assert EngineInstaller(config).snapshot()["ready"]
    finally:
        engine.close()


def test_cancelled_repair_invalidates_receipt_and_other_instances_cached_readiness(config):
    engine = ready_installer(config)
    other = EngineInstaller(config)
    assert other.snapshot()["ready"]
    fake_uv(config, "import time\nprint('repair started',flush=True)\ntime.sleep(30)\n")
    try:
        engine.repair()
        wait_for(lambda: "repair started" in engine.snapshot()["log"])
        assert not engine.receipt.exists()
        assert other.snapshot()["busy_external"]
        engine.cancel()
        state = finished(engine)
        assert state["error"] == "Engine repair cancelled." and state["health"] == "unknown"
        assert not other.snapshot()["ready"] and not other.snapshot()["busy_external"]
        assert not EngineInstaller(config).snapshot()["ready"]
    finally:
        engine.close()


def test_output_without_newlines_and_many_lines_is_bounded(config):
    engine = ready_installer(config)
    interpreter(engine, "import sys\nsys.stdout.write('x'*1048576+'sk-secret-secret-secret'+'\\n')\nprint(('bounded line\\n')*1000)\n")
    try:
        engine.check()
        state = finished(engine)
        assert state["ready"]
        assert len(state["log"]) <= 120 * 2001
        assert "sk-secret-secret-secret" not in state["log"]
    finally:
        engine.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lease and process group acceptance")
def test_cancel_terminates_descendant_even_after_leader_exits(config):
    engine = ready_installer(config)
    engine.TERMINATE_GRACE_SECONDS = 0.3
    child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    interpreter(engine, "import pathlib,subprocess,sys,time\n" + f"child=subprocess.Popen([sys.executable,'-I','-B','-c',{child_code!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n" + "pathlib.Path('descendant.pid').write_text(str(child.pid))\nprint('checking',flush=True)\ntime.sleep(30)\n")
    try:
        engine.check()
        wait_for(lambda: (config.engine_root / "descendant.pid").exists())
        pid = int((config.engine_root / "descendant.pid").read_text())
        engine.cancel()
        state = finished(engine)
        assert state["health"] == "unknown"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        engine.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited flock acceptance")
def test_worker_inherits_lease_after_owner_abrupt_exit(config):
    engine = ready_installer(config)
    interpreter(engine, "import os,pathlib,time\npathlib.Path('orphan.pid').write_text(str(os.getpid()))\nwhile not pathlib.Path('release-worker').exists(): time.sleep(.02)\n")
    code = f"""
import os,sys,time
from pathlib import Path
sys.path[:0]=[{str(REPOSITORY)!r},{str(SERVICE)!r},{str(REPOSITORY / 'apps/webui')!r}]
import desktop_service as module
module.native_receipt_is_current=lambda root,value:value=={{'test_native_verified':True}}
config=module.DesktopConfig(Path({str(config.home)!r}),Path({str(config.engine_root)!r}),{config.token!r},{config.uv!r},{config.python!r})
engine=module.EngineInstaller(config)
engine.check()
deadline=time.monotonic()+5
while not (config.engine_root/'orphan.pid').exists() and time.monotonic()<deadline:time.sleep(.01)
os._exit(0 if (config.engine_root/'orphan.pid').exists() else 9)
"""
    owner = subprocess.Popen([sys.executable, "-I", "-B", "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker_pid = None
    try:
        assert owner.wait(timeout=8) == 0
        worker_pid = int((config.engine_root / "orphan.pid").read_text())
        other = EngineInstaller(config)
        assert other.snapshot()["busy_external"]
        assert other.repair()["busy_external"]
        assert other._thread is None
        (config.engine_root / "release-worker").touch()
        wait_for(lambda: not other.snapshot()["busy_external"])
        assert other.snapshot()["ready"] and other.snapshot()["health"] == "unknown"
    finally:
        (config.engine_root / "release-worker").touch()
        if owner.poll() is None: owner.kill(); owner.wait(timeout=3)
        if worker_pid:
            try: os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError: pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX nofollow lease acceptance")
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_lease_rejects_symlinks_and_non_regular_files(config, kind):
    engine = ready_installer(config)
    lock = config.engine_root / ".desktop-engine-operation.lock"
    secret = config.home / "private-value"
    secret.write_text("unchanged-private-sentinel")
    if kind == "symlink": lock.symlink_to(secret)
    else: os.mkfifo(lock)
    state = engine.snapshot()
    assert state["maintenance_unavailable"] and not state["busy_external"] and not state["ready"]
    assert state["health"] == "failed" and "disk access" in state["error"]
    with pytest.raises(ValueError, match="disk access"):
        engine.repair()
    assert engine._thread is None
    assert secret.read_text() == "unchanged-private-sentinel"


def test_signal_permission_failure_retains_claim_without_uncaught_thread_exception(config, monkeypatch):
    from types import SimpleNamespace

    engine = ready_installer(config)
    engine._claim_fd = engine._open_claim()
    engine._operation = "check"
    child = SimpleNamespace(pid=123456789, poll=lambda: None)
    engine._process = child
    errors = []
    def denied(*_args):
        raise PermissionError("synthetic signal denial")
    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(threading, "excepthook", errors.append)
    worker = threading.Thread(target=engine._cancel_process, args=(child,))
    try:
        worker.start(); worker.join(timeout=1)
        assert not worker.is_alive() and not errors
        assert engine._claim_fd is not None and engine._process is child
        assert engine.snapshot()["checking"] and not engine.snapshot()["ready"]
    finally:
        engine._process = None
        engine._finish_operation()


def test_unconfirmed_exit_keeps_operation_and_lease_until_later_confirmation(config, monkeypatch):
    engine = ready_installer(config)
    actual_terminate = engine._terminate
    monkeypatch.setattr(engine, "_terminate", lambda process: False)
    engine.check()
    engine._thread.join(timeout=3)
    try:
        state = engine.snapshot()
        assert state["operation"] == "check" and state["health"] == "failed" and not state["ready"]
        assert engine._claim_fd is not None and engine._process is not None
        assert EngineInstaller(config).snapshot()["busy_external"]
        monkeypatch.setattr(engine, "_terminate", actual_terminate)
        engine.cancel()
        finished(engine)
        assert not EngineInstaller(config).snapshot()["busy_external"]
    finally:
        monkeypatch.setattr(engine, "_terminate", actual_terminate)
        engine.close()


def test_failed_check_is_durable_even_when_old_receipt_cannot_be_deleted(config, monkeypatch):
    engine = ready_installer(config)
    interpreter(engine, "import sys\nsys.exit(7)\n")
    original = Path.unlink
    def deny_receipt(path, *args, **kwargs):
        if path == engine.receipt:
            raise PermissionError("synthetic receipt deletion denial")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", deny_receipt)
    try:
        engine.check()
        state = finished(engine)
        assert state["health"] == "failed" and not state["ready"]
        assert engine.receipt.exists(), "This regression specifically retains the formerly valid receipt"
        assert (config.engine_root / ".desktop-engine-operation.lock").read_bytes() == engine.FAILURE_MARKER
        restarted = EngineInstaller(config).snapshot()
        assert not restarted["ready"] and restarted["health"] == "failed"
        assert "Repair" in restarted["error"] and not restarted["busy_external"]
    finally:
        monkeypatch.setattr(Path, "unlink", original)
        engine.close()


def test_successful_repair_clears_failure_marker_only_after_verified_receipt_commit(config, monkeypatch):
    engine = ready_installer(config)
    marker = config.engine_root / ".desktop-engine-operation.lock"
    marker.write_bytes(engine.FAILURE_MARKER)
    inode = marker.stat().st_ino
    engine = EngineInstaller(config)
    assert engine.snapshot()["health"] == "failed"
    interpreter(engine, "import pathlib\nassert pathlib.Path('.desktop-engine-operation.lock').read_bytes()\nassert not pathlib.Path('.desktop-engine.json').exists()\n")
    fake_uv(config, "pass\n")
    save_receipt = engine._save_receipt
    def checked_save(native):
        assert marker.read_bytes() == engine.FAILURE_MARKER
        save_receipt(native)
        assert marker.read_bytes() == engine.FAILURE_MARKER
    monkeypatch.setattr(engine, "_save_receipt", checked_save)
    try:
        engine.repair()
        state = finished(engine)
        assert state["ready"] and state["health"] == "healthy"
        assert marker.read_bytes() == b"" and marker.stat().st_ino == inode
        restarted = EngineInstaller(config).snapshot()
        assert restarted["ready"] and restarted["health"] == "unknown"
    finally:
        engine.close()


@pytest.mark.parametrize("data", [b"broken", b"x" * 129, b"\x00\xff"])
def test_malformed_failure_marker_blocks_legacy_ready_receipt(config, data):
    ready_installer(config)
    marker = config.engine_root / ".desktop-engine-operation.lock"
    marker.write_bytes(data)
    state = EngineInstaller(config).snapshot()
    assert not state["ready"] and state["health"] == "failed"
    assert "invalid" in state["error"] and not state["busy_external"]


def test_healthy_and_cancelled_check_do_not_rewrite_failure_marker(config):
    engine = ready_installer(config)
    marker = config.engine_root / ".desktop-engine-operation.lock"
    assert engine.snapshot()["ready"]
    before = (marker.read_bytes(), marker.stat().st_mtime_ns)
    try:
        engine.check()
        assert finished(engine)["health"] == "healthy"
        assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
        checked_at = engine.snapshot()["checked_at"]
        interpreter(engine, "import time\nprint('checking',flush=True)\ntime.sleep(30)\n")
        engine.check()
        wait_for(lambda: "checking" in engine.snapshot()["log"])
        engine.cancel()
        state = finished(engine)
        assert state["health"] == "unknown" and state["checked_at"] == checked_at
        assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
    finally:
        engine.close()
