from __future__ import annotations

import builtins
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from test_desktop_service import AUTH, SERVICE, client_for, config, create_desktop_app
from test_engine_admission import configured_app
import dataset_inspection
import dataset_probe
from dataset_inspection import DatasetInspector, decode_output
from dataset_probe import MAX_ENTRIES, MAX_NUMBER, WARNINGS, report, scan

ROUTE = "/api/desktop/dataset/inspect"


@pytest.fixture
def directory(tmp_path):
    path = tmp_path / "dataset"
    path.mkdir()
    return path.resolve()


def request(path, roots=None):
    return {"path": str(path), "authorized_roots": [str(path if roots is None else root) for root in ([path] if roots is None else roots)]}


def assert_report(value, code="ok"):
    assert value["code"] == code
    assert value["status"] == ("ok" if code == "ok" else "error")
    assert isinstance(value["complete"], bool)
    assert set(value["warnings"]) <= WARNINGS
    for key in ("files", "directories", "symlinks", "other", "bytes"):
        assert type(value[key]) is int and 0 <= value[key] <= MAX_NUMBER
    assert len(value["entries"]) <= 10
    assert len(value["extensions"]) <= 20
    assert all("/" not in entry["name"] and len(entry["name"]) <= 160 for entry in value["entries"])
    assert len(json.dumps(value, ensure_ascii=False).encode()) < 16_384


def test_real_worker_reports_metadata_without_engine_or_configuration_changes(config, directory):
    (directory / "images").mkdir()
    (directory / "images" / "scan.nii.gz").write_bytes(b"private-content-not-returned")
    (directory / "rows.csv.gz").write_bytes(b"a,b\nsecret-label,private-sample\n")
    (directory / "README").write_bytes(b"")
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in directory.rglob("*")}
    app = create_desktop_app(config)
    with client_for(app) as client:
        assert not app.state.desktop_engine.snapshot()["ready"]
        before_config = config.env_file.read_bytes() if config.env_file.exists() else None
        before_runs = app.state.run_store.list_records("runs")
        response = client.post(ROUTE, headers=AUTH, json=request(directory))
        assert response.status_code == 200
        value = response.json()
        assert_report(value)
        assert value["complete"] and value["files"] == 3 and value["directories"] == 1
        assert {item["extension"] for item in value["extensions"]} == {".nii.gz", ".csv.gz", ""}
        assert "private-content" not in response.text and "secret-label" not in response.text
        assert str(directory) not in response.text
        assert before_config == (config.env_file.read_bytes() if config.env_file.exists() else None)
        assert before_runs == app.state.run_store.list_records("runs")
    assert before == {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in directory.rglob("*")}


@pytest.mark.parametrize("body", [None, [], "path", {"path": 3}, {"path": ""}, {"path": "relative"}, {"path": "/x\n"}, {"path": "/" + "a" * 4096}, {"path": "/x", "authorized_roots": "bad"}, {"path": "/x", "authorized_roots": [3]}, {"path": "/x", "authorized_roots": ["/x"] * 129}, {"path": "/x", "mode": "admission"}])
def test_bad_payload_is_sanitized(config, body):
    with client_for(create_desktop_app(config)) as client:
        response = client.post(ROUTE, headers=AUTH, json=body)
        assert response.status_code == 200
        assert_report(response.json(), "invalid_request")
        assert "authorized_roots" not in response.text


def test_missing_roots_and_ungranted_or_prefix_sibling(config, directory, tmp_path):
    (directory / "data.bin").touch()
    sibling = tmp_path / "dataset-more"
    sibling.mkdir()
    (sibling / "secret.bin").touch()
    inspector = DatasetInspector(config.workspace)
    assert_report(inspector.inspect({"path": str(directory)}), "unauthorized")
    assert_report(inspector.inspect(request(directory, [tmp_path / "ungranted"])), "unauthorized")
    assert_report(inspector.inspect(request(sibling, [directory])), "unauthorized")
    inspector.close()


def test_service_owned_workspace_grant_handles_noncanonical_native_alias(config):
    data = config.workspace / "child"
    data.mkdir()
    (data / "data.bin").touch()
    with client_for(create_desktop_app(config)) as client:
        response = client.post(ROUTE, headers=AUTH, json=request(data, ["/unrelated/no-such-native-grant"]))
        assert_report(response.json())
        assert_report(client.post(ROUTE, headers=AUTH, json={"path": str(data)}).json(), "unauthorized")


def test_root_alias_allowed_and_saved_grant_retarget_denied(config, directory, tmp_path):
    (directory / "zero.unknown").touch()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    inspector = DatasetInspector(config.workspace)
    value = inspector.inspect(request(alias, [directory]))
    assert_report(value)
    assert value["complete"] and "root_symlink_resolved" in value["warnings"]
    assert_report(inspector.inspect(request(alias, [alias])), "unauthorized")
    original = tmp_path / "original-grant"
    original.mkdir()
    original.rmdir()
    original.symlink_to(directory, target_is_directory=True)
    assert_report(inspector.inspect(request(original, [original])), "unauthorized")
    inspector.close()


def test_irrelevant_old_network_grants_are_not_resolved(directory, monkeypatch):
    (directory / "x").touch()
    original = Path.resolve
    visited = []
    def checked(self, *args, **kwargs):
        visited.append(str(self))
        assert not str(self).startswith("/Volumes/Disconnected")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", checked)
    value = scan(request(directory, ["/Volumes/Disconnected/old", directory]))
    assert_report(value)
    assert all("Disconnected" not in item for item in visited)


@pytest.mark.parametrize("kind,code", [("missing", "not_found"), ("file", "not_directory"), ("broken", "not_found"), ("loop", "not_directory"), ("empty", "empty")])
def test_real_root_failure_codes(config, directory, tmp_path, kind, code):
    candidate = tmp_path / "candidate"
    if kind == "file": candidate.write_bytes(b"not-a-directory")
    if kind == "broken": candidate.symlink_to(tmp_path / "missing")
    if kind == "loop": candidate.symlink_to(candidate)
    if kind == "empty": candidate.mkdir()
    inspector = DatasetInspector(config.workspace)
    value = inspector.inspect(request(candidate, [tmp_path.resolve()]))
    assert_report(value, code)
    assert value["complete"] == (kind == "empty")
    inspector.close()


def test_complete_nested_empty_is_empty_but_symlinks_are_nonempty(config, directory, tmp_path):
    (directory / "empty-child").mkdir()
    inspector = DatasetInspector(config.workspace)
    assert_report(inspector.inspect(request(directory)), "empty")
    (directory / "dangling").symlink_to(tmp_path / "does-not-exist")
    value = inspector.inspect(request(directory))
    assert_report(value)
    assert value["complete"] and value["symlinks"] == 1 and value["files"] == 0
    assert value["warnings"] == ["symlinks_not_followed"]
    inspector.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses permission bits")
def test_real_root_permission_blocks_but_unreadable_child_is_partial(config, directory):
    inspector = DatasetInspector(config.workspace)
    directory.chmod(0)
    try:
        assert_report(inspector.inspect(request(directory)), "permission")
    finally:
        directory.chmod(0o700)
    child = directory / "restricted"
    child.mkdir()
    child.chmod(0)
    try:
        value = inspector.inspect(request(directory))
        assert_report(value)
        assert not value["complete"] and value["files"] == 0
        assert "descendant_unreadable" in value["warnings"]
    finally:
        child.chmod(0o700)
        inspector.close()


def test_three_descendant_directory_levels_then_partial_fourth(directory):
    third = directory / "one" / "two" / "three"
    third.mkdir(parents=True)
    (third / "visible.bin").touch()
    value = scan(request(directory))
    assert_report(value)
    assert value["complete"] and value["files"] == 1 and value["directories"] == 3
    fourth = third / "four"
    fourth.mkdir()
    (fourth / "beyond-limit.bin").touch()
    value = scan(request(directory))
    assert_report(value)
    assert not value["complete"] and value["files"] == 1 and value["directories"] == 4
    assert "depth_limit" in value["warnings"]
    (third / "visible.bin").unlink()
    value = scan(request(directory))
    assert_report(value)
    assert value["files"] == 0  # A capped search is never proof of emptiness.


def test_large_directory_has_bounded_counts_and_display(directory):
    for i in range(MAX_ENTRIES + 10):
        (directory / f"item-{i}.bin").touch()
    value = scan(request(directory))
    assert_report(value)
    assert not value["complete"] and value["files"] == MAX_ENTRIES
    assert len(value["entries"]) == 10 and "entry_limit" in value["warnings"]


def test_summary_limits_unicode_control_names_and_compound_extensions(directory):
    for i in range(23):
        (directory / f"file.x{i}").touch()
    (directory / ("a" * 200 + ".txt")).touch()
    (directory / "newline\n.csv").touch()
    value = scan(request(directory))
    assert_report(value)
    assert value["complete"] and "extension_limit" in value["warnings"]
    assert len(value["extensions"]) == 20
    single = directory / "unicode"
    single.mkdir()
    for i in range(10):
        (single / ("😀" * 60 + str(i))).touch()
    value = scan(request(single))
    assert_report(value)
    assert value["files"] == 10 and len(value["entries"]) == 10


def test_maximum_unicode_packet_stays_bounded(monkeypatch):
    import io
    value = report()
    value["entries"] = [{"name": "😀" * 160, "kind": "file", "bytes": 0} for _ in range(10)]
    value["extensions"] = [{"extension": "😀" * 32, "count": 1} for _ in range(20)]
    buffer = io.BytesIO()
    class Output:
        def __init__(self): self.buffer = buffer
    monkeypatch.setattr(dataset_probe.sys, "stdout", Output())
    dataset_probe.emit("final", value)
    encoded = buffer.getvalue()
    assert len(encoded) < 16_384
    assert decode_output(encoded)["code"] == "ok"


def test_no_dataset_content_open_and_source_metadata_unchanged(directory, monkeypatch):
    sample = directory / "secret-labels.csv"
    sample.write_text("label,sample\nprivate,private\n")
    before = sample.stat()
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset file content must never be opened")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    value = scan(request(directory))
    assert_report(value)
    assert value["files"] == 1
    assert sample.stat().st_mtime_ns == before.st_mtime_ns


def test_admission_accepts_file_special_or_link_without_recursive_walk(directory, tmp_path):
    file = directory / "zero.unknown"
    file.touch()
    value = scan({"path": str(directory)}, mode="admission")
    assert_report(value)
    assert "admission_short_circuit" in value["warnings"]
    file.unlink()
    os.mkfifo(directory / "pipe")
    value = scan({"path": str(directory)}, mode="admission")
    assert_report(value)
    assert value["other"] == 1


def test_subdirectory_link_replacement_never_follows_external_target(directory, tmp_path, monkeypatch):
    child = directory / "branch"
    child.mkdir()
    (child / "inside.bin").touch()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "DO-NOT-ENUMERATE.bin").touch()
    original = os.open
    replaced = False
    def switching(path, flags, *args, **kwargs):
        nonlocal replaced
        if path == "branch" and not replaced:
            replaced = True
            child.rename(directory / "old")
            child.symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", switching)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {switching})
    value = scan(request(directory))
    assert_report(value)
    assert "changed_during_scan" in value["warnings"]
    assert "DO-NOT-ENUMERATE" not in json.dumps(value)


def test_root_component_replacement_is_not_followed(directory, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "DO-NOT-ENUMERATE.bin").touch()
    original = dataset_probe.open_directory_chain
    def switching(path):
        directory.rename(tmp_path / "old-root")
        directory.symlink_to(outside, target_is_directory=True)
        return original(path)
    monkeypatch.setattr(dataset_probe, "open_directory_chain", switching)
    value = scan(request(directory))
    assert_report(value, "not_directory")
    assert "DO-NOT-ENUMERATE" not in json.dumps(value)


def slow_worker(tmp_path, monkeypatch, *, after_root):
    worker = tmp_path / "slow_worker.py"
    worker.write_text(f'''import sys,time
sys.path.insert(0,{str(SERVICE)!r})
import dataset_probe as probe
original = probe.os.scandir
calls = 0
def slow(fd):
    global calls
    calls += 1
    if calls == {2 if after_root else 1}:
        time.sleep(90)
    return original(fd)
probe.os.scandir = slow
probe.os.supports_fd = probe.os.supports_fd | {{slow}}
probe.main()
''')
    monkeypatch.setattr(dataset_inspection, "PROBE", worker)
    return worker


@pytest.mark.parametrize("after_root", [False, True])
def test_real_hard_deadline_before_and_after_root_confirmation(config, directory, tmp_path, monkeypatch, after_root):
    (directory / "branch").mkdir()
    (directory / "branch" / "data.bin").touch()
    slow_worker(tmp_path, monkeypatch, after_root=after_root)
    inspector = DatasetInspector(config.workspace, timeout=0.3)
    spawned = []
    original = subprocess.Popen
    def capture(*args, **kwargs):
        assert str(directory) not in repr(args)
        assert kwargs["env"] == {}
        child = original(*args, **kwargs)
        spawned.append(child)
        return child
    monkeypatch.setattr(dataset_inspection.subprocess, "Popen", capture)
    started = time.monotonic()
    value = inspector.inspect(request(directory))
    assert_report(value, "ok" if after_root else "timeout")
    assert not value["complete"]
    assert ("hard_timeout" in value["warnings"]) == after_root
    assert time.monotonic() - started < 1.5
    assert spawned[0].poll() is not None and spawned[0].stdout.closed
    inspector.close()


def test_worker_watchdog_and_owner_loss(tmp_path, directory, monkeypatch):
    worker = slow_worker(tmp_path, monkeypatch, after_root=False)
    packet = json.dumps({"payload": request(directory), "mode": "overview"}).encode()
    child = subprocess.Popen([sys.executable, "-I", "-B", str(worker), str(os.getpid()), "0.25"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = child.communicate(packet, timeout=2)
    assert child.returncode == 71 and not output and not errors
    owner_source = '''import os,subprocess,sys
child=subprocess.Popen([sys.executable,'-I','-B',sys.argv[1],str(os.getpid()),'5'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
child.stdin.write(sys.stdin.buffer.readline())
child.stdin.close()
print(child.pid,flush=True)
sys.stdin.readline()
os._exit(0)
'''
    owner = subprocess.Popen([sys.executable, "-I", "-B", "-c", owner_source, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    owner.stdin.write(packet + b"\n")
    owner.stdin.flush()
    pid = int(owner.stdout.readline())
    owner.stdin.write(b"exit\n")
    owner.stdin.flush()
    owner.wait(timeout=2)
    deadline = time.monotonic() + 2
    while psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
        time.sleep(0.025)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    for pipe in (owner.stdin, owner.stdout, owner.stderr): pipe.close()


def test_busy_close_and_cleanup_failure_ownership(config, directory, tmp_path, monkeypatch):
    slow_worker(tmp_path, monkeypatch, after_root=False)
    inspector = DatasetInspector(config.workspace, timeout=0.7)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(inspector.inspect, request(directory))
        deadline = time.monotonic() + 1
        while inspector._process is None and time.monotonic() < deadline: time.sleep(0.005)
        child = inspector._process
        assert_report(inspector.inspect(request(directory)), "busy")
        def denied(): raise PermissionError("synthetic private-path cleanup error")
        monkeypatch.setattr(child, "kill", denied)
        inspector.close()
        assert inspector._process is child and inspector._busy
        value = future.result(timeout=2)
        assert_report(value, "timeout")
        assert "private-path" not in json.dumps(value)
        deadline = time.monotonic() + 1
        while child.poll() is None and time.monotonic() < deadline: time.sleep(0.01)
        inspector.close()
        assert child.poll() is not None and inspector._process is None


def test_service_lifespan_closes_inspector_even_when_provider_close_raises(config, monkeypatch):
    app = create_desktop_app(config)
    def fail(): raise PermissionError("synthetic close failure")
    monkeypatch.setattr(app.state.provider_checker, "close", fail)
    with client_for(app): pass
    assert app.state.dataset_inspector._closed


def test_empty_create_and_changed_requeue_make_no_new_records(config, monkeypatch):
    app, engine, body = configured_app(config, monkeypatch)
    data = Path(body["experiment"]["dataset_path"])
    with client_for(app) as client:
        accepted = client.post("/api/runs", headers=AUTH, json=body)
        assert accepted.status_code == 202
        record = accepted.json()
        assert client.post(f"/api/runs/{record['id']}/cancel", headers=AUTH).status_code == 200
        (data / "fixture.bin").unlink()
        before = app.state.run_store.list_records("runs")
        tasks = list((app.state.manager.settings.state_dir / "tasks").iterdir())
        for path in ("/api/runs", f"/api/runs/{record['id']}/requeue"):
            response = client.post(path, headers=AUTH, json=body if path == "/api/runs" else None)
            assert response.status_code == 422
            assert response.json()["code"] == "empty"
            assert "[dataset:empty]" in response.json()["detail"]
            assert app.state.run_store.list_records("runs") == before
            assert list((app.state.manager.settings.state_dir / "tasks").iterdir()) == tasks


def test_invalid_run_schema_skips_scanner_and_resume_unchanged(config, monkeypatch):
    app, engine, body = configured_app(config, monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError("Malformed schemas/resume must not scan")
    monkeypatch.setattr(app.state.dataset_inspector, "inspect", forbidden)
    with client_for(app) as client:
        body["experiment"]["task_body"] = ""
        response = client.post("/api/runs", headers=AUTH, json=body)
        assert response.status_code == 422 and isinstance(response.json()["detail"], list)
        assert client.post("/api/queue/resume", headers=AUTH, json={"pause_generation": 0}).status_code == 202


def test_scan_soft_deadline_after_root_is_partial_not_empty(directory, monkeypatch):
    (directory / "file.bin").touch()
    real_scandir = os.scandir
    def slow_open(path):
        time.sleep(0.03)
        return real_scandir(path)
    monkeypatch.setattr(os, "scandir", slow_open)
    monkeypatch.setattr(os, "supports_fd", os.supports_fd | {slow_open})
    value = scan(request(directory), soft_seconds=0.02)
    assert_report(value)
    assert not value["complete"] and value["files"] == 0 and "soft_timeout" in value["warnings"]


def test_busy_admission_does_not_create_records(config, tmp_path, monkeypatch):
    app, engine, body = configured_app(config, monkeypatch)
    slow_worker(tmp_path, monkeypatch, after_root=False)
    inspector = app.state.dataset_inspector
    with client_for(app) as client, concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(inspector.inspect, request(Path(body["experiment"]["dataset_path"])))
        deadline = time.monotonic() + 1
        while inspector._process is None and time.monotonic() < deadline: time.sleep(0.005)
        response = client.post("/api/runs", headers=AUTH, json=body)
        assert response.status_code == 409 and response.json()["code"] == "busy"
        assert "retry" in response.json()["detail"]
        assert app.state.run_store.list_records("runs") == []
        inspector.close()
        assert_report(pending.result(timeout=2), "internal")


def test_overview_requires_auth_and_body_errors_do_not_echo(config, directory):
    with client_for(create_desktop_app(config)) as client:
        assert client.post(ROUTE, json=request(directory)).status_code == 401
        assert client.post(ROUTE, headers={**AUTH, "Origin": "https://evil.example"}, json=request(directory)).status_code == 403
        response = client.post(ROUTE, headers=AUTH, content=b'{"path":"private-secret-name",')
        assert_report(response.json(), "invalid_request")
        assert "private-secret-name" not in response.text
        assert_report(client.post(ROUTE, headers=AUTH, content=b" " * 524_289).json(), "invalid_request")
