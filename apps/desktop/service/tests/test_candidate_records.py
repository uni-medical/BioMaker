from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_records as records
from test_desktop_service import AUTH, client_for, config, create_desktop_app


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "logs"
    directory = root / "live"
    directory.mkdir(parents=True)
    return root, directory, {"id": "live", "status": "running", "log_dir": str(directory)}


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value))


def event(directory, name, **fields):
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": name, **fields}) + "\n")


def read(experiment):
    root, _, run = experiment
    return records.build_candidate_records(run, logs_root=root)


def test_generation_queue_code_and_evaluation_exist_before_first_journal(experiment):
    _, directory, _ = experiment
    event(directory, "node_generation_started", placeholder_id="first", scheduled_step=0, selection_reason="Use actual images")
    value = read(experiment)
    assert value["nodes"][0]["phase"] == "generating"
    assert value["nodes"][0]["selection_reason"] == "Use actual images"
    assert not (directory / "journal.json").exists()
    write(directory, "candidates_live.json", {"nodes": [{"id": "first", "code": "# generated code", "phase": "queued", "is_buggy": False, "metric": {"value": 999}, "private_metrics": "private-sentinel"}]})
    event(directory, "node_generation_finished", node_id="first")
    event(directory, "node_queued_for_execution", node_id="first")
    node = read(experiment)["nodes"][0]
    assert node["phase"] == "queued" and node["code"] == "# generated code"
    assert "metric" not in node and "is_buggy" not in node
    assert "private-sentinel" not in json.dumps(node)
    event(directory, "execution_started", node_id="first")
    # Worker starts can race ahead of the scheduler's submitted event.
    event(directory, "node_submitted", node_id="first")
    assert read(experiment)["nodes"][0]["phase"] == "running"
    event(directory, "candidate_evaluation_started", node_id="first")
    assert read(experiment)["nodes"][0]["phase"] == "evaluating"
    event(directory, "node_evaluated", node_id="first", is_buggy=False, metric=0.8)
    node = read(experiment)["nodes"][0]
    assert node["phase"] == "evaluating" and "metric" not in node
    write(directory, "journal.json", {"nodes": [{"id": "first", "step": 0, "is_buggy": False, "metric": {"value": 0.8, "maximize": True}, "code": "# committed code"}]})
    node = read(experiment)["nodes"][0]
    assert node["phase"] == "valid" and node["recorded"]
    assert node["metric"]["value"] == 0.8 and node["code"] == "# committed code"


@pytest.mark.parametrize("status,terminal,expected", [("stopped", None, "interrupted"), ("failed", None, "interrupted"), ("running", "run_interrupted", "interrupted"), ("running", "run_finished", "unconfirmed"), ("complete", None, "unconfirmed")])
def test_terminal_run_never_promotes_unfinished_candidate(experiment, status, terminal, expected):
    _, directory, run = experiment
    event(directory, "node_generation_started", placeholder_id="pending")
    if terminal:
        event(directory, terminal)
    run["status"] = status
    assert read(experiment)["nodes"][0]["phase"] == expected
    assert "metric" not in read(experiment)["nodes"][0]


def test_generation_failure_survives_missing_live_placeholder_and_stale_journal(experiment):
    _, directory, _ = experiment
    event(directory, "node_generation_started", placeholder_id="failed")
    event(directory, "node_generation_failed", placeholder_id="failed", exc_type="ProviderError")
    write(directory, "candidates_live.json", {"nodes": []})
    write(directory, "journal.json", {"nodes": []})
    node = read(experiment)["nodes"][0]
    assert node["phase"] == "failed" and node["exc_type"] == "ProviderError"


def test_old_completed_journal_retains_failures_nonfinite_metrics_and_lineage(experiment):
    _, directory, run = experiment
    run["status"] = "complete"
    write(directory, "journal.json", {"nodes": [{"id": "old", "step": 0, "is_buggy": False, "metric": {"value": 0, "maximize": False}, "code": "old code"}, {"id": "bad", "step": 1, "is_buggy": True, "metric": {"value": float("nan")}, "_term_out": ["actual failure"], "exc_type": "ArtifactValidationError"}], "node2parent": {"bad": "old"}})
    before = (directory / "journal.json").stat().st_mtime_ns
    value = read(experiment)
    assert [n["phase"] for n in value["nodes"]] == ["valid", "failed"]
    assert value["nodes"][0]["metric"] == {"value": 0, "maximize": False}
    assert "metric" not in value["nodes"][1]
    assert value["nodes"][1]["_term_out"] == ["actual failure"]
    assert value["node2parent"] == {"bad": "old"}
    assert (directory / "journal.json").stat().st_mtime_ns == before


def test_bounded_sources_symlinks_and_corrupt_records_are_not_hidden_successes(experiment, tmp_path, monkeypatch):
    root, directory, run = experiment
    private = tmp_path / "private"
    private.write_text("private-sentinel")
    (directory / "journal.json").symlink_to(private)
    result = read(experiment)
    assert result["state"] == "updating" and result["nodes"] == []
    assert "private-sentinel" not in json.dumps(result)
    (directory / "journal.json").unlink()
    monkeypatch.setitem(records.LIMITS, "journal.json", 5)
    (directory / "journal.json").write_text("123456")
    assert "journal.json:too_large" in read(experiment)["warnings"]
    run["log_dir"] = str(tmp_path)
    assert records.build_candidate_records(run, logs_root=root)["state"] == "unavailable"


def test_incomplete_append_is_ignored_and_malformed_complete_event_is_reported(experiment):
    _, directory, _ = experiment
    event(directory, "node_generation_started", placeholder_id="one")
    with (directory / "events.jsonl").open("a") as stream:
        stream.write('{"event": []}\ninvalid\n{"event":"execution_sta')
    result = read(experiment)
    assert result["nodes"][0]["phase"] == "generating"
    assert result["warnings"] == ["event_invalid"]


def test_combined_response_is_bounded_without_losing_candidate_states(experiment, monkeypatch):
    _, directory, _ = experiment
    monkeypatch.setattr(records, "MAX_RESPONSE_BYTES", 2500)
    write(directory, "candidates_live.json", {"nodes": [{"id": name, "phase": "queued", "code": name * 1800} for name in ("one", "two", "three")]})
    result = read(experiment)
    assert len(json.dumps(result).encode()) <= 2500
    assert [n["phase"] for n in result["nodes"]] == ["queued"] * 3
    assert len(result["nodes"]) == 3
    assert any(n.get("content_truncated") for n in result["nodes"])


def test_new_run_with_not_yet_created_log_directory_is_empty_not_disconnected(experiment):
    root, _, run = experiment
    run["log_dir"] = str(root / "preparing")
    result = read(experiment)
    assert result["state"] == "ready" and result["nodes"] == []


def test_unchanged_poll_uses_stat_cache_and_atomic_replacement_invalidates_it(experiment, monkeypatch):
    _, directory, run = experiment
    write(directory, "candidates_live.json", {"nodes": [{"id": "one", "phase": "running", "code": "first"}]})
    original_read = records._read
    reads = []

    def counted_read(*args):
        reads.append(args[1])
        return original_read(*args)

    monkeypatch.setattr(records, "_read", counted_read)
    first = read(experiment)
    assert len(reads) == 3
    assert read(experiment) == first
    assert len(reads) == 3
    write(directory, "replacement.json", {"nodes": [{"id": "one", "phase": "running", "code": "later"}]})
    (directory / "replacement.json").replace(directory / "candidates_live.json")
    second = read(experiment)
    assert second["nodes"][0]["code"] == "later"
    assert second["revision"] != first["revision"] and len(reads) == 6
    run["status"] = "stopped"
    assert read(experiment)["nodes"][0]["phase"] == "interrupted"
    assert len(reads) == 9


def test_candidate_cache_is_bounded_by_entries_and_serialized_size(experiment, monkeypatch):
    root, directory, run = experiment
    with records._cache_lock:
        records._cache.clear()
    monkeypatch.setattr(records, "MAX_CACHE_ENTRIES", 2)
    monkeypatch.setattr(records, "MAX_CACHE_BYTES", 2000)
    write(directory, "journal.json", {"nodes": [{"id": "one", "code": "bounded", "is_buggy": False}]})
    for requested in ("one", "two", "three"):
        records.build_candidate_records(run, logs_root=root, requested_id=requested)
    assert len(records._cache) <= 2
    assert sum(value[2] for value in records._cache.values()) <= 2000
    assert all(key[-1] != "one" for key in records._cache)


@pytest.mark.parametrize("failure", ["too_large", "invalid"])
def test_events_failure_does_not_hide_independent_saved_journal(experiment, monkeypatch, failure):
    _, directory, _ = experiment
    write(directory, "journal.json", {"nodes": [{"id": "saved", "is_buggy": False, "metric": {"value": 0.7}, "code": "saved code"}]})
    if failure == "too_large":
        monkeypatch.setitem(records.LIMITS, "events.jsonl", 1)
        (directory / "events.jsonl").write_text("oversized\n")
    else:
        (directory / "events.jsonl").write_bytes(b"\xff\n")
    value = read(experiment)
    assert value["state"] == "ready"
    assert value["warnings"]
    assert value["nodes"][0]["phase"] == "valid"
    assert value["nodes"][0]["metric"]["value"] == 0.7


@pytest.mark.parametrize("saved_journal", [True, False])
@pytest.mark.parametrize("failure", ["too_large", "invalid", "duplicate"])
def test_bad_live_snapshot_only_pauses_when_no_independent_journal_exists(experiment, monkeypatch, saved_journal, failure):
    _, directory, run = experiment
    if saved_journal:
        run["status"] = "complete"
        write(directory, "journal.json", {"nodes": [{"id": "saved", "is_buggy": False, "metric": {"value": 0.7}, "code": "saved code"}]})
    if failure == "too_large":
        monkeypatch.setitem(records.LIMITS, "candidates_live.json", 1)
        (directory / "candidates_live.json").write_text("oversized")
    elif failure == "duplicate":
        write(directory, "candidates_live.json", {"nodes": [{"id": "untrusted", "phase": "queued", "code": "must not survive invalid source"}, {"id": "untrusted"}]})
    else:
        (directory / "candidates_live.json").write_text('{"nodes":')
    value = read(experiment)
    assert value["warnings"]
    assert all(node["id"] != "untrusted" for node in value["nodes"])
    if saved_journal:
        assert value["state"] == "ready"
        assert [node["id"] for node in value["nodes"]] == ["saved"]
        assert value["nodes"][0]["phase"] == "valid"
        assert value["nodes"][0]["metric"]["value"] == 0.7
    else:
        assert value["state"] == "updating"
        assert not value["nodes"]


def test_oversized_event_parent_ids_never_expand_the_response(experiment):
    _, directory, _ = experiment
    oversized_parent = "x" * 32000
    for index in range(500):
        event(directory, "node_generation_started", placeholder_id=f"node-{index}", parent_id=oversized_parent)
    assert (directory / "events.jsonl").stat().st_size < records.LIMITS["events.jsonl"]
    value = read(experiment)
    assert len(value["nodes"]) == 500 and value["state"] == "ready"
    assert not value["node2parent"]
    assert all("parent_id" not in node for node in value["nodes"])
    assert len(json.dumps(value, ensure_ascii=True).encode()) <= records.MAX_RESPONSE_BYTES


def test_final_response_bound_includes_parent_structure_and_preserves_requested(experiment, monkeypatch):
    root, directory, run = experiment
    monkeypatch.setattr(records, "MAX_RESPONSE_BYTES", 2500)
    write(directory, "candidates_live.json", {"nodes": [{"id": f"node-{i}", "parent_id": "p" * 256, "phase": "queued"} for i in range(20)]})
    value = records.build_candidate_records(run, logs_root=root, requested_id="node-19")
    assert len(json.dumps(value, ensure_ascii=True).encode()) <= 2500
    assert value["total"] == 20
    assert any(node["id"] == "node-19" for node in value["nodes"])
    assert "response_truncated" in value["warnings"]


def test_candidate_route_is_authenticated_registered_and_read_only(config):
    app = create_desktop_app(config)
    directory = config.workspace / "logs" / "live"
    directory.mkdir(parents=True)
    event(directory, "node_generation_started", placeholder_id="live-id")
    app.state.run_store.add_record("runs", {"id": "live", "status": "running", "log_dir": str(directory)})
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with client_for(app) as client:
        route = "/api/desktop/runs/live/candidates"
        assert client.get(route).status_code == 401
        assert client.get(route, headers={**AUTH, "Origin": "https://example.org"}).status_code == 403
        assert client.get("/api/desktop/runs/unregistered/candidates", headers=AUTH).status_code == 404
        result = client.get(route, headers=AUTH)
        assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
        assert result.json()["nodes"][0]["id"] == "live-id"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
