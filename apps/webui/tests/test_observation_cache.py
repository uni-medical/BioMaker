from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

import pytest

import webui.observation_cache as cache_module
from webui.artifacts import read_events
from webui.observation_cache import ObservationCache


def history(settings, name="cached-run", *, results=True):
    directory = settings.logs_root / name
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({
        "data_dir": str(settings.datasets_root / "synthetic"),
        "created_at": "manifest-time", "task_description": "original task",
        "config": {"agent_steps": 3},
    }))
    (directory / "events.jsonl").write_text(
        '{"event":"run_started","ts":"started-time"}\n'
        '{"event":"round_finished","completed_steps":1}\n')
    if results:
        (directory / "results.tsv").write_text("step\tmetric\tmaximize\n0\t0.5\ttrue\n")
    return directory, {"id": name, "log_dir": str(directory), "status": "running",
                       "allowed_actions": ["stop"], "cleanup_pending": False}


def content_counter(monkeypatch):
    calls = []
    original = Path.open
    def opened(path, *args, **kwargs):
        if path.name in {"events.jsonl", "run_manifest.json", "results.tsv"}:
            calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    return calls


def test_unchanged_catalog_and_detail_read_no_content_after_warmup(settings, monkeypatch):
    directory, run = history(settings)
    cache = ObservationCache(settings)
    calls = content_counter(monkeypatch)
    first_catalog, first_detail = cache.catalog(directory), cache.detail(run)
    assert calls
    calls.clear()
    for _ in range(3):
        assert cache.catalog(directory) == first_catalog
        assert cache.detail(run) == first_detail
    assert calls == []


def test_detail_never_caches_ledger_actions_or_artifact_metadata(settings, monkeypatch):
    directory, run = history(settings)
    output = directory / "report.md"
    output.write_text("first")
    cache = ObservationCache(settings)
    before = cache.detail(run)
    output.write_text("changed artifact")
    calls = content_counter(monkeypatch)
    latest = cache.detail({**run, "status": "stopping", "allowed_actions": [], "cleanup_pending": True})
    assert latest["status"] == "stopping" and latest["allowed_actions"] == []
    assert latest["cleanup_pending"] is True
    assert latest["artifacts"] != before["artifacts"]
    assert calls == []


def test_mutating_returned_nested_objects_cannot_poison_cache(settings):
    directory, run = history(settings)
    cache = ObservationCache(settings)
    result = cache.catalog(directory)
    result["progress"]["completed_steps"] = 999
    result["project_snapshot"]["task_body"] = "changed by caller"
    assert cache.catalog(directory)["progress"]["completed_steps"] == 1
    detail = cache.detail(run)
    detail["observed"]["last_event"]["event"] = "poisoned"
    assert cache.detail(run)["observed"]["last_event"]["event"] == "round_finished"


def test_last_5000_semantics_exact_nonempty_count_and_nonfinite_diagnostics(settings):
    directory, run = history(settings)
    events = [{"event": "run_finished", "completed_steps": 77},
              *({"event": "observation", "step": i} for i in range(5000))]
    text = "\n \r\n\u00a0\r" + "\r\n".join(json.dumps(event) for event in events)
    text += '\r\nnot-json\r\n[]\r\n{"event":"failure","metric":NaN,"error":"keep me"}'
    (directory / "events.jsonl").write_bytes(text.encode())
    cache = ObservationCache(settings)
    observed = cache.detail(run)["observed"]
    assert observed["event_count"] == 5004  # Nonempty lines, including invalid/nonobject JSON.
    assert observed["last_event"] == {"event": "failure", "metric": "NaN", "error": "keep me"}
    # The terminal event is outside the last 5000 valid dicts: caching must not
    # retain an all-time terminal event or silently change historical inference.
    assert cache.catalog(directory)["terminal_event"] is None
    assert cache.catalog(directory)["progress"]["completed_steps"] is None
    assert read_events(settings, run, tail=0)["events"] == []
    assert len(read_events(settings, run, tail=5000)["events"]) == 5000


@pytest.mark.parametrize("mutation", ["append", "same_size", "replacement", "truncate", "delete"])
def test_event_dependency_changes_refresh_count_and_catalog(settings, mutation):
    directory, run = history(settings)
    path = directory / "events.jsonl"
    cache = ObservationCache(settings)
    cache.catalog(directory)
    cache.detail(run)
    before = path.stat()
    if mutation == "append":
        with path.open("a") as handle:
            handle.write('{"event":"run_finished","completed_steps":2}\n')
    elif mutation == "same_size":
        path.write_text(path.read_text().replace('"completed_steps":1', '"completed_steps":2'))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif mutation == "replacement":
        replacement = directory / "replacement"
        replacement.write_text(path.read_text().replace('"completed_steps":1', '"completed_steps":2'))
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)
    elif mutation == "truncate":
        path.write_text('{"event":"round_finished","completed_steps":2}\n')
    else:
        path.unlink()
    detail, catalog = cache.detail(run), cache.catalog(directory)
    assert detail["observed"]["event_count"] == {"append": 3, "same_size": 2, "replacement": 2, "truncate": 1, "delete": 0}[mutation]
    assert catalog["progress"]["completed_steps"] == (None if mutation == "delete" else 2)


def test_manifest_results_and_selected_console_dependencies_refresh(settings):
    directory, run = history(settings)
    console = directory / f"{directory.name}.log"
    console.write_text("console")
    cache = ObservationCache(settings)
    initial_catalog, initial_detail = cache.catalog(directory), cache.detail(run)
    manifest = directory / "run_manifest.json"
    before = manifest.stat()
    manifest.write_text(manifest.read_text().replace("original task", "modified task"))
    os.utime(manifest, ns=(before.st_atime_ns, before.st_mtime_ns))
    (directory / "results.tsv").write_text("step\tmetric\tmaximize\n0\t0.7\ttrue\n")
    assert cache.catalog(directory)["project_snapshot"]["task_body"] == "modified task"
    assert cache.detail(run)["observed"]["results"]["best_metric"] == 0.7
    console.unlink()
    assert cache.catalog(directory)["console_log"] is None
    assert initial_catalog["console_log"] and initial_detail["observed"]["results"]["best_metric"] == 0.5


@pytest.mark.parametrize("missing", ["results.tsv", "events.jsonl", "run_manifest.json"])
def test_confirmed_missing_dependencies_cache_and_invalidate_on_creation(settings, monkeypatch, missing):
    directory, run = history(settings)
    path = directory / missing
    raw = path.read_bytes()
    path.unlink()
    cache = ObservationCache(settings)
    first = cache.detail(run)
    calls = content_counter(monkeypatch)
    assert cache.detail(run) == first
    assert calls == []
    path.write_bytes(raw)
    assert cache.detail(run)["observed"] != first["observed"]


def test_io_fallback_is_never_cached_and_recovers_without_metadata_change(settings, monkeypatch):
    directory, run = history(settings)
    cache = ObservationCache(settings)
    original = Path.open
    fail = True
    def opened(path, *args, **kwargs):
        if fail and path.name == "events.jsonl":
            raise PermissionError("synthetic transient read failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    assert cache.detail(run)["observed"]["event_count"] == 0
    assert cache.catalog(directory)["progress"]["completed_steps"] is None
    assert not cache._entries
    fail = False
    assert cache.detail(run)["observed"]["event_count"] == 2
    assert cache.catalog(directory)["progress"]["completed_steps"] == 1


@pytest.mark.parametrize("outside", [False, True])
def test_file_symlink_replacement_discards_cache_and_uses_legacy_fallback(settings, tmp_path, outside):
    directory, run = history(settings)
    cache = ObservationCache(settings)
    cache.detail(run)
    target = (tmp_path if outside else directory) / "linked-events.jsonl"
    target.write_text('{"event":"replacement"}\n')
    path = directory / "events.jsonl"
    path.unlink()
    path.symlink_to(target)
    result = cache.detail(run)
    assert result["observed"]["event_count"] == (0 if outside else 1)
    assert not cache._entries


def test_directory_deletion_and_symlink_replacement_do_not_return_cached_summary(settings):
    directory, _ = history(settings)
    cache = ObservationCache(settings)
    cache.catalog(directory)
    previous = directory.with_name("old-run")
    directory.rename(previous)
    assert cache.catalog(directory) is None
    directory.symlink_to(previous, target_is_directory=True)
    assert cache.catalog(directory) is None
    assert not cache._entries


def test_unavailable_metadata_bypasses_cache(settings, monkeypatch):
    directory, run = history(settings)
    cache = ObservationCache(settings)
    cache.detail(run)
    original = Path.lstat
    def metadata(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise PermissionError("synthetic metadata failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", metadata)
    cache.detail(run)
    assert not cache._entries


def test_mutation_during_loading_is_not_published(settings, monkeypatch):
    directory, _ = history(settings)
    cache = ObservationCache(settings)
    original = cache_module.run_from_log_dir
    changed = False
    def load(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            with (directory / "events.jsonl").open("a") as handle:
                handle.write('{"event":"run_finished","completed_steps":2}\n')
        return result
    monkeypatch.setattr(cache_module, "run_from_log_dir", load)
    cache.catalog(directory)
    assert not cache._entries
    assert cache.catalog(directory)["progress"]["completed_steps"] == 2


def test_lru_bounds_and_oversized_summary_preserve_uncached_response(settings, monkeypatch):
    cache = ObservationCache(settings, max_entries=2, max_bytes=12000, max_item_bytes=4000)
    directories = [history(settings, str(i))[0] for i in range(4)]
    for directory in directories:
        cache.catalog(directory)
        assert len(cache._entries) <= 2 and cache._bytes <= 12000
    assert ("catalog", str(directories[-1])) in cache._entries
    assert ("catalog", str(directories[0])) not in cache._entries
    manifest = directories[0] / "run_manifest.json"
    data = json.loads(manifest.read_text())
    data["task_description"] = "x" * 10000
    manifest.write_text(json.dumps(data))
    assert cache.catalog(directories[0])["project_snapshot"]["task_body"] == "x" * 10000
    assert ("catalog", str(directories[0])) not in cache._entries


def test_same_key_singleflight_and_clear_cannot_publish_old_inflight_work(settings, monkeypatch):
    directory, _ = history(settings)
    cache = ObservationCache(settings)
    original = cache_module.run_from_log_dir
    started, release = threading.Event(), threading.Event()
    calls = 0
    def load(*args, **kwargs):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(cache_module, "run_from_log_dir", load)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(cache.catalog, directory) for _ in range(6)]
        assert started.wait(2)
        release.set()
        values = [future.result(timeout=3) for future in futures]
    assert calls == 1 and all(value == values[0] for value in values)
    cache.clear()
    started.clear()
    release.clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.catalog, directory)
        assert started.wait(2)
        cache.clear()
        release.set()
        assert future.result(timeout=3)
    assert not cache._entries and cache._bytes == 0


def test_unrelated_key_does_not_wait_on_global_io_lock(settings, monkeypatch):
    first, _ = history(settings, "blocked")
    cache = ObservationCache(settings)
    for index in range(100):
        second = settings.logs_root / f"independent-{index}"
        if hash(("catalog", str(second))) % 64 != hash(("catalog", str(first))) % 64:
            second, _ = history(settings, second.name)
            break
    original = cache_module.run_from_log_dir
    started, release = threading.Event(), threading.Event()
    def load(settings, directory, **kwargs):
        if directory == first:
            started.set()
            assert release.wait(3)
        return original(settings, directory, **kwargs)
    monkeypatch.setattr(cache_module, "run_from_log_dir", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(cache.catalog, first)
        assert started.wait(2)
        try:
            assert executor.submit(cache.catalog, second).result(timeout=2)
        finally:
            release.set()
        assert pending.result(timeout=2)


def test_new_process_owned_cache_has_no_persisted_entries(settings, monkeypatch):
    directory, _ = history(settings)
    first = ObservationCache(settings)
    expected = first.catalog(directory)
    second = ObservationCache(settings)
    calls = content_counter(monkeypatch)
    assert second.catalog(directory) == expected
    assert calls
