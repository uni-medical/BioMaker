from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPOSITORY = SERVICE.parents[2]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(REPOSITORY / "apps/webui"))
sys.path.insert(0, str(REPOSITORY))

import history_import
from history_import import HistoryImporter, HistoryImportError
from webui.run_store import SQLiteRunStore
from webui.runner import RunManager
from webui.settings import Settings
from test_desktop_service import AUTH, client_for, config, create_desktop_app


@pytest.fixture
def importer(tmp_path):
    root = tmp_path / "desktop"
    settings = Settings(
        method_root=REPOSITORY, workspace_root=root, state_dir=root / "state",
        logs_root=root / "logs", workspaces_root=root / "workspaces",
        experiments_root=root / "experiments", datasets_root=root / "datasets",
        env_file=root / ".env",
    )
    settings.ensure_directories()
    store = SQLiteRunStore(settings.state_dir / "runs.sqlite3")
    manager = RunManager(settings, store)
    value = HistoryImporter(settings, manager)
    yield value
    value.close()


def make_run(path: Path, *, terminal=True):
    path.mkdir(parents=True)
    manifest = {
        "created_at": "2026-10-08T00:00:00Z", "data_dir": "/original/datasets/example",
        "exp_name": path.name, "task_description": "真实记录 fixture",
        "workspace_dir": "/original/workspace/active", "launch_token": "original-owner",
        "pid": os.getpid(), "pgid": os.getpgrp(), "process_started_at": "original",
        "config": {"agent_steps": 1, "main_model": "historical-model"},
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    events = [{"event": "run_started", "ts": "2026-10-08T00:00:00Z"}]
    if terminal:
        events.append({"event": "run_finished", "ts": "2026-10-08T00:01:00Z", "completed_steps": 1})
    (path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    (path / "journal.json").write_text(json.dumps({"nodes": [{"id": "candidate-1", "step": 0, "code": "print('historical')", "metric": {"value": 0.75}}], "node2parent": {}}))
    (path / "results.tsv").write_text("step\tstatus\tmetric\tmaximize\n1\tsucceeded\t0.75\ttrue\n")
    (path / "report.md").write_text("# Historical report\n真实报告\n")
    (path / "best_solution.py").write_text("print('historical')\n")
    (path / f"{path.name}.log").write_text("Recorded training log\n")
    (path / "working_best").mkdir()
    (path / "working_best/model.bin").write_bytes(b"model weights must not migrate")
    (path / "private_gt").mkdir()
    (path / "private_gt/labels.csv").write_text("private labels must not migrate")
    (path / "runs.sqlite3").write_bytes(b"original ledger must not be opened")
    (path / ".env").write_text("SECRET=must-not-be-copied\n")
    return path


def snapshot(path):
    return {str(item.relative_to(path)): (item.stat().st_mtime_ns, hashlib.sha256(item.read_bytes()).hexdigest()) for item in path.rglob("*") if item.is_file() and not item.is_symlink()}


def finish(importer, preview, ids=None):
    ids = ids or [item["source_id"] for item in preview["runs"] if item["eligible"]]
    result = importer.submit(preview["preview_id"], ids)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = importer.get_job(result["job_id"])
        if result["status"] in {"completed", "failed"}:
            # status may be saved just before worker cleanup finishes.
            importer._thread.join(timeout=2)
            return result
        time.sleep(0.01)
    pytest.fail(f"Import did not finish: {result}")


def test_real_records_copy_preserves_source_and_clears_process_ownership(importer, tmp_path):
    source = make_run(tmp_path / "research/中文实验")
    before = snapshot(source)
    original_manifest = (source / "run_manifest.json").read_bytes()
    preview = importer.preview(str(source))
    row = preview["runs"][0]
    assert preview["source_kind"] == "run"
    assert row["name"] == "中文实验" and row["eligible"]
    assert row["status"] == "succeeded"
    assert all(row["capabilities"].values())
    assert "working_best" in " ".join(row["warnings"])
    result = finish(importer, preview)
    assert result["status"] == "completed" and not result["errors"]
    identifier = result["imported"][0]["run_id"]
    destination = importer.settings.logs_root / identifier
    for name in ("journal.json", "report.md", "best_solution.py", "events.jsonl", "results.tsv"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert (destination / "source_run_manifest.json").read_bytes() == original_manifest
    assert not (destination / "private_gt").exists()
    assert not (destination / "working_best").exists()
    assert not (destination / ".env").exists()
    assert not (destination / "runs.sqlite3").exists()
    copied_manifest = json.loads((destination / "run_manifest.json").read_text())
    record = importer.manager.store.get_record("runs", identifier)
    for key in ("pid", "pgid", "process_started_at", "launch_token", "workspace_dir"):
        assert copied_manifest[key] is None
        assert record[key] is None
    assert record["managed"] is False and record["read_only"] is True
    assert record["allowed_actions"] == [] and not record["controllable"]
    assert record["history_import"]["name"] == "中文实验"
    assert record["history_import"]["source_path"] == str(source)
    assert snapshot(source) == before
    assert not list(importer.staging_root.iterdir())


def test_same_source_is_idempotent_and_same_names_from_other_sources_are_distinct(importer, tmp_path):
    first = make_run(tmp_path / "one/same-name")
    second = make_run(tmp_path / "two/same-name")
    preview = importer.preview(str(first))
    original = finish(importer, preview)
    repeated = finish(importer, preview)
    assert repeated["status"] == "completed" and len(repeated["skipped"]) == 1
    newer = importer.preview(str(first))["runs"][0]
    assert newer["already_imported"] and not newer["eligible"]
    assert newer["existing_run_id"] == original["imported"][0]["run_id"]
    other = finish(importer, importer.preview(str(second)))
    assert other["imported"][0]["run_id"] != original["imported"][0]["run_id"]


def test_running_source_is_only_a_snapshot_and_never_reconciled(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/live", terminal=False)
    before = snapshot(source)
    preview = importer.preview(str(source))
    assert preview["runs"][0]["status"] == "snapshot"
    result = finish(importer, preview)
    identifier = result["imported"][0]["run_id"]
    record = importer.manager.store.get_record("runs", identifier)
    assert record["detailed_status"] == "orphaned"
    assert "may still be running" in record["status_reason"]
    monkeypatch.setattr(importer.manager, "_record_identity_state", lambda *_: pytest.fail("Source process must not be inspected"))
    importer.manager.recover_active_runs()
    assert snapshot(source) == before


@pytest.mark.parametrize("mutation", ["file", "symlink", "directory"])
def test_source_changes_after_preview_fail_without_partial_publication(importer, tmp_path, mutation):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    if mutation == "file":
        (source / "report.md").write_text("new report after preview")
    elif mutation == "symlink":
        (source / "report.md").unlink()
        (source / "report.md").symlink_to(source / "journal.json")
    else:
        moved = source.with_name("moved")
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    result = finish(importer, preview)
    assert result["status"] == "failed" and result["errors"]
    assert not result["imported"]
    assert not list(importer.settings.logs_root.glob("desktop-import-*"))
    assert not list(importer.staging_root.iterdir())


def test_source_change_during_copy_rolls_back_only_own_staging(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    original = importer._copy_file
    def changing_copy(item, name, destination):
        value = original(item, name, destination)
        if name == "report.md":
            (source / name).write_text("concurrent researcher update")
        return value
    monkeypatch.setattr(importer, "_copy_file", changing_copy)
    result = finish(importer, preview)
    assert result["status"] == "failed"
    assert (source / "report.md").read_text() == "concurrent researcher update"
    assert not list(importer.settings.logs_root.glob("desktop-import-*"))


def test_bad_manifest_empty_shell_and_symlink_sources_are_ineligible(importer, tmp_path):
    parent = tmp_path / "research"
    bad = make_run(parent / "bad")
    (bad / "run_manifest.json").write_text("not json")
    empty = parent / "empty"
    empty.mkdir()
    (empty / "run_manifest.json").write_text('{"data_dir":"/dataset"}')
    link = parent / "link"
    link.symlink_to(bad, target_is_directory=True)
    rows = importer.preview(str(parent))["runs"]
    assert len(rows) == 3 and all(not row["eligible"] for row in rows)
    assert all(row["warnings"] for row in rows)


def test_symlink_artifacts_and_nested_weights_are_not_copied(importer, tmp_path):
    source = make_run(tmp_path / "research/run")
    secret = tmp_path / "secret"
    secret.write_text("not an experiment artifact")
    (source / "best_solution.py").unlink()
    (source / "best_solution.py").symlink_to(secret)
    preview = importer.preview(str(source))
    assert preview["runs"][0]["eligible"]
    assert "best_solution.py" not in {item["name"] for item in preview["runs"][0]["files"]}
    result = finish(importer, preview)
    destination = importer.settings.logs_root / result["imported"][0]["run_id"]
    assert not (destination / "best_solution.py").exists()
    assert not any(path.is_symlink() for path in destination.iterdir())
    assert secret.read_text() == "not an experiment artifact"


def test_large_journal_is_export_only_and_hard_limits_are_explicit(importer, tmp_path):
    source = make_run(tmp_path / "research/run")
    with (source / "journal.json").open("wb") as stream:
        stream.truncate(history_import.MAX_JOURNAL_VIEW_BYTES + 1)
    row = importer.preview(str(source))["runs"][0]
    assert row["eligible"] and not row["capabilities"]["candidates"]
    assert "export only" in " ".join(row["warnings"])
    with (source / "journal.json").open("wb") as stream:
        stream.truncate(history_import.MAX_FILE_BYTES + 1)
    row = importer.preview(str(source))["runs"][0]
    assert not row["eligible"] and "64 MiB" in " ".join(row["warnings"])


def test_scan_limits_do_not_silently_truncate(importer, tmp_path):
    parent = tmp_path / "many"
    parent.mkdir()
    for number in range(101):
        (parent / str(number)).mkdir()
    with pytest.raises(HistoryImportError, match="100 child"):
        importer.preview(str(parent))
    source = make_run(tmp_path / "research/run")
    for number in range(129):
        (source / f"extra-{number}").touch()
    row = importer.preview(str(source))["runs"][0]
    assert not row["eligible"] and "128" in " ".join(row["warnings"])


def test_manifest_byte_limit_and_invalid_data_dir(importer, tmp_path):
    source = make_run(tmp_path / "research/run")
    (source / "run_manifest.json").write_text('{"data_dir": {"unexpected": true}}')
    assert not importer.preview(str(source))["runs"][0]["eligible"]
    with (source / "run_manifest.json").open("wb") as stream:
        stream.truncate(history_import.MAX_MANIFEST_BYTES + 1)
    row = importer.preview(str(source))["runs"][0]
    assert not row["eligible"] and "size limit" in " ".join(row["warnings"])


def test_disk_space_and_copy_error_preserve_sources(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    before = snapshot(source)
    preview = importer.preview(str(source))
    with monkeypatch.context() as patch:
        patch.setattr(history_import.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(HistoryImportError, match="free disk"):
            importer.submit(preview["preview_id"], [preview["runs"][0]["source_id"]])
    def no_space(*_):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(importer, "_copy_file", no_space)
    result = finish(importer, preview)
    assert result["status"] == "failed" and "space" in result["errors"][0]["message"]
    assert snapshot(source) == before
    assert not list(importer.staging_root.iterdir())


def test_preview_expiry_foreign_ids_and_batch_limits(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    identifier = preview["runs"][0]["source_id"]
    with pytest.raises(HistoryImportError, match="belong"):
        importer.submit(preview["preview_id"], ["f" * 64])
    with monkeypatch.context() as patch:
        patch.setattr(history_import, "MAX_BATCH_BYTES", 1)
        with pytest.raises(HistoryImportError, match="256 MiB"):
            importer.submit(preview["preview_id"], [identifier])
    importer._previews[preview["preview_id"]] = (0, importer._previews[preview["preview_id"]][1])
    with pytest.raises(HistoryImportError, match="expired"):
        importer.submit(preview["preview_id"], [identifier])


def test_import_is_background_and_rejects_overlapping_jobs(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    gate = threading.Event()
    original = importer._copy_file
    def slow_copy(*args):
        assert gate.wait(3)
        return original(*args)
    monkeypatch.setattr(importer, "_copy_file", slow_copy)
    started = time.monotonic()
    job = importer.submit(preview["preview_id"], [preview["runs"][0]["source_id"]])
    assert time.monotonic() - started < 1
    try:
        with pytest.raises(HistoryImportError, match="already in progress"):
            importer.submit(preview["preview_id"], [preview["runs"][0]["source_id"]])
    finally:
        gate.set()
        importer._thread.join(5)
    assert importer.get_job(job["job_id"])["status"] == "completed"


def test_interrupted_job_recovery_and_published_copy_registration(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    register = importer._register
    monkeypatch.setattr(importer, "_register", lambda *_: (_ for _ in ()).throw(OSError("ledger temporarily unavailable")))
    result = finish(importer, preview)
    assert result["status"] == "failed" and "recover" in result["errors"][0]["message"]
    assert len(list(importer.settings.logs_root.glob("desktop-import-*"))) == 1
    monkeypatch.setattr(importer, "_register", register)
    incomplete = dict(result, job_id="a" * 32, status="running", errors=[])
    importer._save_job(incomplete)
    staging = importer.staging_root / incomplete["job_id"]
    staging.mkdir()
    (staging / "incomplete-copy").write_text("partial")
    importer.recover()
    assert importer.get_job(incomplete["job_id"])["status"] == "failed"
    assert "interrupted" in importer.get_job(incomplete["job_id"])["errors"][0]["message"]
    assert not staging.exists()
    row = importer.preview(str(source))["runs"][0]
    assert row["already_imported"] and row["existing_run_id"]


def test_atomic_publication_never_replaces_concurrent_empty_directory(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    publish = history_import._publish_directory
    created = []
    def conflicting_publish(staging, destination):
        destination.mkdir()
        created.append((destination, destination.stat().st_ino))
        publish(staging, destination)
    monkeypatch.setattr(history_import, "_publish_directory", conflicting_publish)
    job = finish(importer, preview)
    assert job["status"] == "failed" and not job["imported"]
    destination, inode = created[0]
    assert destination.stat().st_ino == inode and not list(destination.iterdir())
    assert not list(importer.staging_root.iterdir())


def test_disk_full_while_saving_job_still_returns_failed_status(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    preview = importer.preview(str(source))
    atomic = history_import._atomic_json
    def full_receipts(path, value):
        if path.parent == importer.jobs_dir and value.get("status") != "queued":
            raise OSError(errno.ENOSPC, "No space left for job receipt")
        return atomic(path, value)
    monkeypatch.setattr(history_import, "_atomic_json", full_receipts)
    job = finish(importer, preview)
    assert job["status"] == "failed"
    assert "space" in job["errors"][0]["message"]
    assert not list(importer.settings.logs_root.glob("desktop-import-*"))


def test_partial_batch_reports_errors_without_discarding_success(importer, tmp_path):
    parent = tmp_path / "research"
    make_run(parent / "a-good")
    bad = make_run(parent / "b-changing")
    preview = importer.preview(str(parent))
    (bad / "report.md").write_text("new source report")
    job = finish(importer, preview)
    assert job["status"] == "completed" and job["completed_runs"] == 2
    assert len(job["imported"]) == 1 and len(job["errors"]) == 1
    assert job["imported"][0]["name"] == "a-good"
    assert job["errors"][0]["name"] == "b-changing"


def test_service_close_interrupts_unpublished_copy_without_source_changes(importer, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/run")
    before = snapshot(source)
    preview = importer.preview(str(source))
    original = importer._copy_file
    def closing_copy(*args):
        value = original(*args)
        importer._closed.set()
        return value
    monkeypatch.setattr(importer, "_copy_file", closing_copy)
    job = finish(importer, preview)
    assert job["status"] == "failed"
    assert snapshot(source) == before
    assert not list(importer.settings.logs_root.glob("desktop-import-*"))
    assert not list(importer.staging_root.iterdir())


def test_malformed_import_provenance_does_not_break_startup(importer):
    destination = importer.settings.logs_root / ("desktop-import-" + "a" * 32)
    destination.mkdir()
    (destination / "desktop_import.json").write_text(json.dumps({"run_id": destination.name}))
    importer.recover()
    assert not importer.manager.store.list_records("runs")


@pytest.mark.parametrize("event", [
    '{"event":"run_finished","completed_steps":NaN}',
    '{"event":"run_finished","unused":{"metric":Infinity}}',
    '{"event":"run_finished","completed_steps":1e999}',
    '{"event":[]}',
    '{"event":"run_finished","deep":' + '[' * 2000 + '0' + ']' * 2000 + '}',
], ids=["nan", "infinity", "overflow", "invalid-event-name", "deep-nesting"])
def test_invalid_event_data_is_rejected_before_publication(importer, tmp_path, event):
    source = make_run(tmp_path / "research/invalid")
    (source / "events.jsonl").write_text(event + "\n")
    before = snapshot(source)
    preview = importer.preview(str(source))
    job = finish(importer, preview)
    assert job["status"] == "failed" and not job["imported"]
    assert not list(importer.settings.logs_root.glob("desktop-import-*"))
    assert not importer.manager.store.list_records("runs")
    assert snapshot(source) == before


def test_nonfinite_events_cannot_poison_original_webui_startup(config, tmp_path):
    parent = tmp_path / "research"
    bad = make_run(parent / "bad")
    make_run(parent / "good")
    (bad / "events.jsonl").write_text('{"event":"run_finished","completed_steps":NaN}\n')
    with client_for(create_desktop_app(config)) as client:
        preview = client.post("/api/desktop/history/preview", json={"path": str(parent)}, headers=AUTH).json()
        response = client.post("/api/desktop/history/import", json={"preview_id": preview["preview_id"], "source_ids": [row["source_id"] for row in preview["runs"]]}, headers=AUTH)
        identifier = response.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/desktop/history/imports/{identifier}", headers=AUTH).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed" and len(job["imported"]) == 1
        assert job["errors"][0]["name"] == "bad"
        assert "events.jsonl" in job["errors"][0]["message"]
    # Run the real lifespan: baseline WebUI imports history BEFORE the desktop
    # recovery hook. The rejected bad event must never reach that startup path.
    with client_for(create_desktop_app(config)) as client:
        assert client.get("/api/health", headers=AUTH).status_code == 200
        runs = client.get("/api/runs", headers=AUTH).json()
        assert len(runs) == 1 and runs[0]["history_import"]["name"] == "good"
        assert client.get(f"/api/runs/{runs[0]['id']}/events", headers=AUTH).status_code == 200


def test_routes_show_real_artifacts_and_reject_all_controls_even_if_status_tampered(config, tmp_path, monkeypatch):
    source = make_run(tmp_path / "research/中文实验")
    before = snapshot(source)
    app = create_desktop_app(config)
    for method in ("stop", "cancel", "requeue"):
        monkeypatch.setattr(app.state.manager, method, lambda *_: pytest.fail("Read-only records must never reach process controls"))
    with client_for(app) as client:
        assert client.post("/api/desktop/history/preview", json={"path": str(source)}).status_code == 401
        response = client.post("/api/desktop/history/preview", json={"path": str(source)}, headers=AUTH)
        assert response.status_code == 200, response.text
        preview = response.json()
        response = client.post("/api/desktop/history/import", json={"preview_id": preview["preview_id"], "source_ids": [preview["runs"][0]["source_id"]]}, headers=AUTH)
        assert response.status_code == 202
        identifier = response.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/desktop/history/imports/{identifier}", headers=AUTH).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed", job
        imported = job["imported"][0]["run_id"]
        detail = client.get(f"/api/runs/{imported}", headers=AUTH).json()
        assert detail["observed"]["results"]["best_metric"] == 0.75
        assert detail["history_import"]["name"] == "中文实验"
        assert detail["workspace_dir"] is None
        assert detail["read_only"] and detail["allowed_actions"] == []
        for name in ("journal.json", "report.md", "best_solution.py"):
            artifact = client.get(f"/artifacts/runs/{imported}/{name}", headers=AUTH)
            assert artifact.status_code == 200
            assert artifact.content == (source / name).read_bytes()
        assert "Recorded training log" in client.get(f"/api/runs/{imported}/log", headers=AUTH).text
        for status in ("running", "queued", "succeeded"):
            app.state.run_store.update_record("runs", imported, {"status": status})
            for action in ("stop", "cancel", "requeue"):
                response = client.post(f"/api/runs/{imported}/{action}", headers=AUTH)
                assert response.status_code == 409 and "view-only" in response.text
        assert client.get("/api/desktop/history/imports/unknown", headers=AUTH).status_code == 404
    # Restart preserves the real copy and the original title/provenance.
    with client_for(create_desktop_app(config)) as client:
        detail = client.get(f"/api/runs/{imported}", headers=AUTH).json()
        assert detail["history_import"]["name"] == "中文实验"
        assert detail["allowed_actions"] == [] and detail["workspace_dir"] is None
        assert client.get(f"/api/desktop/history/imports/{identifier}", headers=AUTH).json()["status"] == "completed"
    assert snapshot(source) == before
