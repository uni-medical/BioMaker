from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
import result_overview as overview


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "logs"
    log = root / "run-one"
    log.mkdir(parents=True)
    run = {"id": "run-one", "status": "succeeded", "managed": True, "log_dir": str(log)}
    return root, log, run


def write_json(log, name, value):
    (log / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def candidate(identifier="chosen", *, code="print('candidate')\n", value=0.8, maximize=True, step=0, good=True):
    return {"id": identifier, "code": code, "step": step, "is_buggy": not good,
            "metric": {"value": value, "maximize": maximize}}


def seed(experiment, *, nodes=None, best="chosen", events=True, code=None):
    root, log, run = experiment
    nodes = [candidate()] if nodes is None else nodes
    write_json(log, "journal.json", {"nodes": nodes, "node2parent": {}})
    write_json(log, "run_manifest.json", {"config": {}})
    (log / "best_solution.py").write_text(nodes[0]["code"] if code is None and nodes else code or "", encoding="utf-8")
    (log / "report.md").write_text("# Generated report\nThis claim is not parsed.\n")
    (log / "results.tsv").write_text("step\tmetric\tmaximize\n0\t999\ttrue\n")
    if events:
        write_events(log, [{"event": "step_saved", "node_id": nodes[-1]["id"] if nodes else None, "best_node_id": best}, {"event": "run_finished"}])
    return root, log, run


def write_events(log, events):
    (log / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def build(experiment):
    root, _, run = experiment
    return overview.build_result_overview(run, logs_root=root)


def artifact(result, name):
    return next(item for item in result["artifacts"] if item["name"] == name)


@pytest.mark.parametrize("value,maximize", [(0, True), (-0.25, True), (0.12, False), (0.8, True)])
def test_recorded_selection_uses_event_identity_and_journal_metric_not_table_or_report(experiment, value, maximize):
    nodes = [candidate(value=value, maximize=maximize), candidate("later", code="different", step=1, value=1000)]
    seed(experiment, nodes=nodes)
    result = build(experiment)
    assert result["selection"] == {"status": "recorded", "source": "step_saved", "node_id": "chosen", "candidate_step": 0,
                                   "metric": {"value": value, "maximize": maximize, "source": "journal"}, "code_match": True}
    assert result["snapshot"]["status"] == "stable"
    assert len(result["snapshot"]["revision"]) == 64
    assert set(item["name"] for item in result["artifacts"]) == set(overview.ARTIFACT_NAMES)
    assert "Generated report" not in json.dumps(result)


def test_equal_scores_and_mixed_directions_never_trigger_numeric_inference(experiment):
    nodes = [candidate("earlier", code="a", value=0.5, maximize=False), candidate("chosen", code="b", value=0.5, step=1)]
    seed(experiment, nodes=nodes, code="b")
    assert build(experiment)["selection"]["node_id"] == "chosen"
    (experiment[1] / "events.jsonl").unlink()
    (experiment[1] / "best_solution.py").unlink()
    result = build(experiment)
    assert result["selection"]["status"] == "unavailable"
    assert result["selection"]["node_id"] is None


def test_unique_exact_code_match_is_labelled_as_code_match(experiment):
    nodes = [candidate("first", code="x", value=999), candidate("chosen", code="y", step=1, value=-3)]
    seed(experiment, nodes=nodes, code="y", events=False)
    result = build(experiment)
    assert result["selection"]["status"] == "code_match"
    assert result["selection"]["source"] == "best_solution.py"
    assert result["selection"]["node_id"] == "chosen"
    assert result["selection"]["metric"]["value"] == -3


@pytest.mark.parametrize("code,expected", [("print('candidate')", "best_code_no_match"), ("different", "best_code_no_match"), ("", "best_code_unavailable")])
def test_code_match_does_not_trim_or_guess(experiment, code, expected):
    seed(experiment, events=False, code=code)
    assert build(experiment)["selection"]["reason"] == expected


def test_duplicate_code_is_ambiguous_but_recorded_id_resolves_it(experiment):
    nodes = [candidate(), candidate("other", step=1)]
    seed(experiment, nodes=nodes, events=False)
    assert build(experiment)["selection"]["reason"] == "best_code_ambiguous"
    write_events(experiment[1], [{"event": "step_saved", "node_id": "other", "best_node_id": "chosen"}])
    assert build(experiment)["selection"]["status"] == "recorded"


def test_failed_high_scoring_candidate_never_matches_code(experiment):
    seed(experiment, nodes=[candidate(good=False, value=999)], events=False)
    result = build(experiment)
    assert result["selection"]["status"] == "none"
    assert result["selection"]["reason"] == "no_valid_candidates"


def test_explicit_null_selection_is_preserved_without_promoting_failed_code(experiment):
    seed(experiment, nodes=[candidate(good=False)], best=None)
    (experiment[1] / "best_solution.py").unlink()
    result = build(experiment)
    assert result["selection"]["status"] == "none"
    assert result["selection"]["source"] == "step_saved"
    assert result["selection"]["reason"] == "recorded_no_selection"


@pytest.mark.parametrize("mutation,reason", [
    ("missing-id", "selected_candidate_missing"),
    ("buggy", "selected_candidate_invalid"),
    ("stale-step", "selection_snapshot_mismatch"),
    ("missing-best-field", "selection_event_invalid"),
    ("numeric-best", "selection_event_invalid"),
    ("duplicate-id", "duplicate_candidate_ids"),
    ("code-mismatch", "selected_code_mismatch"),
    ("null-with-good", "selection_snapshot_mismatch"),
])
def test_conflicting_evidence_is_explicit(experiment, mutation, reason):
    seed(experiment)
    log = experiment[1]
    event = {"event": "step_saved", "node_id": "chosen", "best_node_id": "chosen"}
    if mutation == "missing-id": event["best_node_id"] = "absent"
    elif mutation == "buggy": write_json(log, "journal.json", {"nodes": [candidate(good=False)]})
    elif mutation == "stale-step": event["node_id"] = "older"
    elif mutation == "missing-best-field": event.pop("best_node_id")
    elif mutation == "numeric-best": event["best_node_id"] = 12
    elif mutation == "duplicate-id": write_json(log, "journal.json", {"nodes": [candidate(), candidate()]})
    elif mutation == "code-mismatch": (log / "best_solution.py").write_text("changed")
    elif mutation == "null-with-good": event["best_node_id"] = None
    write_events(log, [event])
    result = build(experiment)
    assert result["selection"]["status"] == "conflict"
    assert result["selection"]["reason"] == reason


def test_missing_code_does_not_erase_recorded_choice(experiment):
    seed(experiment)
    (experiment[1] / "best_solution.py").unlink()
    result = build(experiment)
    assert result["selection"]["status"] == "recorded"
    assert result["selection"]["code_match"] is None
    assert "selection_code_unverified" in result["warnings"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.8", None, 10**400])
def test_nonfinite_or_invalid_metric_never_escapes_json(experiment, value):
    seed(experiment, nodes=[candidate(value=value)])
    result = build(experiment)
    assert result["selection"]["status"] == "recorded"
    assert result["selection"]["metric"] is None
    assert "metric_unavailable" in result["warnings"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("payload", [b"not json", b"[]", b"\xff", b'{"nodes":12}', b'{"nodes":[null]}', b'{"nodes":[{"id":3}]}', b"[" * 2000])
def test_bad_journal_remains_an_explicit_unavailable_artifact(experiment, payload):
    seed(experiment)
    (experiment[1] / "journal.json").write_bytes(payload)
    result = build(experiment)
    assert result["selection"]["status"] == "unavailable"
    assert artifact(result, "journal.json")["state"] == "invalid"


def test_corrupt_new_event_never_reuses_old_recorded_selection(experiment):
    seed(experiment)
    with (experiment[1] / "events.jsonl").open("ab") as stream:
        stream.write(b'{"event":"step_saved","best_node_id":')
    result = build(experiment)
    assert result["selection"]["status"] == "code_match"
    assert "events_invalid" in result["warnings"]


@pytest.mark.parametrize("name", list(overview.LIMITS))
def test_file_size_limits_never_read_oversize_bodies(experiment, monkeypatch, name):
    seed(experiment)
    monkeypatch.setitem(overview.LIMITS, name, 32)
    (experiment[1] / name).write_bytes(b"x" * 33)
    original = overview.os.open
    def guard(path, flags, *args, **kwargs):
        if path == name:
            pytest.fail("Oversize file must be rejected before open/read")
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(overview.os, "open", guard)
    monkeypatch.setattr(overview.os, "supports_dir_fd", {*os.supports_dir_fd, guard})
    result = build(experiment)
    assert name.split(".")[0] + "_too_large" in result["warnings"]
    if name in overview.ARTIFACT_NAMES:
        assert artifact(result, name)["state"] == "too_large"


def test_event_and_candidate_count_limits_are_bounded(experiment, monkeypatch):
    seed(experiment)
    monkeypatch.setattr(overview, "MAX_EVENTS", 1)
    result = build(experiment)
    assert "events_too_large" in result["warnings"]
    monkeypatch.setattr(overview, "MAX_NODES", 0)
    assert artifact(build(experiment), "journal.json")["state"] == "too_large"


@pytest.mark.parametrize("ok", [True, False])
def test_retrain_uses_receipt_and_start_event_without_exposing_private_payload(experiment, ok):
    seed(experiment)
    write_events(experiment[1], [
        {"event": "step_saved", "node_id": "chosen", "best_node_id": "chosen"},
        {"event": "final_retrain_started", "node_id": "chosen"},
        {"event": "final_retrain_finished", "ok": ok},
    ])
    write_json(experiment[1], "final_retrain_result.json", {"ok": ok, "train_rows_before": 10, "score_rows_added": 3, "train_rows_after": 13,
                                                            "private_metrics": {"SENSITIVE": 1}, "error": None if ok else "SECRET", "execution": {"stdout": "PRIVATE"}})
    result = build(experiment)
    assert result["final_retrain"] == {"state": "succeeded" if ok else "failed", "source": "final_retrain_result.json", "node_id": "chosen",
                                       "train_rows_before": 10, "score_rows_added": 3, "train_rows_after": 13}
    assert not any(value in json.dumps(result) for value in ("SENSITIVE", "SECRET", "PRIVATE"))


def test_retrain_receipt_alone_does_not_invent_candidate_identity(experiment):
    seed(experiment)
    write_json(experiment[1], "final_retrain_result.json", {"ok": True})
    result = build(experiment)
    assert result["final_retrain"]["state"] == "succeeded"
    assert result["final_retrain"]["node_id"] is None


@pytest.mark.parametrize("case", ["disabled", "opposite-outcome", "different-candidate", "skipped"])
def test_retrain_conflicts_are_not_success(experiment, case):
    seed(experiment)
    events = [{"event": "step_saved", "node_id": "chosen", "best_node_id": "chosen"}]
    if case == "disabled": write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    elif case == "opposite-outcome": events.append({"event": "final_retrain_finished", "ok": False})
    elif case == "different-candidate": events.append({"event": "final_retrain_started", "node_id": "other"})
    elif case == "skipped": events.append({"event": "final_retrain_skipped", "reason": "no_best_node"})
    write_events(experiment[1], events)
    write_json(experiment[1], "final_retrain_result.json", {"ok": True})
    result = build(experiment)["final_retrain"]
    assert result["state"] == "unknown"
    assert result["reason"] == "conflicting_retrain_records"


@pytest.mark.parametrize("source", ["options", "manifest"])
def test_disabled_requires_explicit_original_manifest_not_catalog_inference(experiment, source):
    seed(experiment)
    if source == "manifest":
        write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    else:
        experiment[2]["options"] = {"final_retrain_enabled": False}
    overview_result = build(experiment)
    assert overview_result["final_retrain"]["state"] == ("disabled" if source == "manifest" else "unknown")
    assert ("final_retrain_result_missing" in overview_result["warnings"]) is (source != "manifest")


@pytest.mark.parametrize("missing", [None, "report.md", "journal.json"])
def test_disabled_retrain_suppresses_only_the_optional_missing_receipt(experiment, missing):
    seed(experiment)
    write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    if missing:
        (experiment[1] / missing).unlink()
    result = build(experiment)
    assert result["final_retrain"]["state"] == "disabled"
    expected = [] if missing is None else [missing.rsplit(".", 1)[0] + "_missing"]
    assert result["warnings"] == expected


def test_disabled_manifest_with_retrain_activity_keeps_missing_receipt_warning(experiment):
    seed(experiment)
    write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    write_events(experiment[1], [{"event": "final_retrain_started", "node_id": "chosen"}])
    result = build(experiment)
    assert result["final_retrain"]["state"] == "unknown"
    assert "final_retrain_result_missing" in result["warnings"]


@pytest.mark.parametrize("origin", ["managed", "imported_snapshot"])
@pytest.mark.parametrize("ok", [True, False])
def test_completion_event_without_receipt_never_claims_receipt_verification(experiment, origin, ok):
    seed(experiment)
    if origin == "imported_snapshot": experiment[2]["source"] = "desktop_import"
    write_events(experiment[1], [{"event": "final_retrain_started", "node_id": "chosen"}, {"event": "final_retrain_finished", "ok": ok}])
    result = build(experiment)
    assert result["origin"] == origin
    assert result["final_retrain"]["state"] == "recorded_without_receipt"
    assert result["final_retrain"]["reason"] == ("imported_receipt_not_included" if origin == "imported_snapshot" else "receipt_unavailable")


@pytest.mark.parametrize("status,reason", [("running", "final_retrain_in_progress"), ("failed", "final_retrain_incomplete")])
def test_started_retrain_is_not_reported_as_finished(experiment, status, reason):
    seed(experiment)
    experiment[2]["status"] = status
    write_events(experiment[1], [{"event": "final_retrain_started", "node_id": "chosen"}])
    result = build(experiment)["final_retrain"]
    assert result["state"] == "unknown" and result["reason"] == reason


@pytest.mark.parametrize("payload", [{"ok": 1}, {"ok": "true"}, {}, None])
def test_invalid_retrain_receipt_is_not_success(experiment, payload):
    seed(experiment)
    write_json(experiment[1], "final_retrain_result.json", payload)
    result = build(experiment)
    assert result["final_retrain"]["state"] == "unknown"
    assert result["final_retrain"]["reason"] == "receipt_invalid"
    assert artifact(result, "final_retrain_result.json")["state"] == "invalid"


@pytest.mark.parametrize("fields,origin", [({"source": "desktop_import"}, "imported_snapshot"), ({"history_import": {"mode": "snapshot"}}, "imported_snapshot"), ({"managed": False}, "historical"), ({"read_only": True}, "historical")])
def test_historical_origin_remains_explicit(experiment, fields, origin):
    seed(experiment)
    experiment[2].update(fields)
    result = build(experiment)
    assert result["origin"] == origin
    assert "history_snapshot" in result["warnings"]


def test_no_log_directory_is_unavailable_without_io(experiment):
    experiment[2]["log_dir"] = None
    assert build(experiment)["snapshot"] == {"status": "unavailable", "revision": None}


@pytest.mark.parametrize("target", ["outside", "root", "dotdot", "relative"])
def test_log_directory_must_be_inside_authorized_root(experiment, target):
    root, log, run = experiment
    run["log_dir"] = {"outside": str(root.parent), "root": str(root), "dotdot": str(root / "run-one" / ".." / "run-one"), "relative": "logs/run-one"}[target]
    result = build(experiment)
    assert result["snapshot"]["status"] == "unavailable"
    assert "log_directory_unsafe" in result["warnings"]


@pytest.mark.parametrize("kind", ["artifact", "directory", "ancestor"])
def test_symbolic_links_never_read_target(experiment, kind):
    seed(experiment)
    root, log, run = experiment
    if kind == "artifact":
        target = root.parent / "outside.json"
        target.write_text('{"nodes":[]}')
        (log / "journal.json").unlink()
        (log / "journal.json").symlink_to(target)
    elif kind == "directory":
        log.rename(root / "moved")
        log.symlink_to(root / "moved", target_is_directory=True)
    else:
        root.rename(root.parent / "moved-root")
        root.symlink_to(root.parent / "moved-root", target_is_directory=True)
    result = build(experiment)
    assert artifact(result, "journal.json")["state"] == "unsafe"
    assert result["selection"]["status"] == "unavailable"


def test_fifo_is_rejected_without_blocking(experiment):
    seed(experiment)
    path = experiment[1] / "journal.json"
    path.unlink()
    os.mkfifo(path)
    assert artifact(build(experiment), "journal.json")["state"] == "unsafe"


def test_permission_failure_is_partial_availability_not_raw_error(experiment, monkeypatch):
    seed(experiment)
    original = overview.os.open
    def denied(path, flags, *args, **kwargs):
        if path == "journal.json":
            raise PermissionError(errno.EACCES, "PRIVATE ERROR SENTINEL")
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(overview.os, "open", denied)
    monkeypatch.setattr(overview.os, "supports_dir_fd", {*os.supports_dir_fd, denied})
    result = build(experiment)
    assert artifact(result, "journal.json")["state"] == "unreadable"
    assert "PRIVATE ERROR SENTINEL" not in json.dumps(result)


@pytest.mark.parametrize("change", ["file", "new-file", "directory", "ancestor", "same-size-same-mtime"])
def test_concurrent_change_suppresses_cross_file_claims(experiment, monkeypatch, change):
    seed(experiment)
    root, log, run = experiment
    original = overview._Snapshot.read
    def changing(self, name, **kwargs):
        result = original(self, name, **kwargs)
        if name == "run_manifest.json":
            if change == "file": (log / "best_solution.py").write_text("new contents")
            elif change == "new-file": write_json(log, "final_retrain_result.json", {"ok": True})
            elif change == "directory":
                log.rename(root / "old-run")
                log.mkdir()
            elif change == "ancestor":
                root.rename(root.parent / "old-root")
                root.mkdir()
                log.mkdir()
            else:
                path = log / "best_solution.py"
                before = path.stat()
                path.write_bytes(b"x" * before.st_size)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result
    monkeypatch.setattr(overview._Snapshot, "read", changing)
    result = build(experiment)
    assert result["snapshot"] == {"status": "changing", "revision": None}
    assert result["selection"]["status"] == "unavailable"
    assert result["selection"]["reason"] == "snapshot_changed"
    assert result["final_retrain"]["state"] == "unknown"


def test_artifact_swap_between_stat_and_open_does_not_follow_link(experiment, monkeypatch):
    seed(experiment)
    root, log, _ = experiment
    secret = root.parent / "secret.json"
    secret.write_text('{"nodes":[]}')
    original = overview.os.open
    def swapping(path, flags, *args, **kwargs):
        if path == "journal.json":
            (log / path).unlink()
            (log / path).symlink_to(secret)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(overview.os, "open", swapping)
    monkeypatch.setattr(overview.os, "supports_dir_fd", {*os.supports_dir_fd, swapping})
    result = build(experiment)
    assert result["snapshot"]["status"] == "changing"
    assert artifact(result, "journal.json")["state"] == "unsafe"


def test_repeated_reads_leave_file_bytes_and_mtimes_unchanged(experiment):
    seed(experiment)
    log = experiment[1]
    def snapshot():
        return {path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in log.iterdir()}
    before = snapshot()
    first = build(experiment)
    assert first == build(experiment)
    assert snapshot() == before


def test_report_or_tsv_alone_does_not_generate_selection_or_retrain_claim(experiment):
    seed(experiment)
    log = experiment[1]
    for name in ("journal.json", "best_solution.py", "events.jsonl"):
        (log / name).unlink()
    result = build(experiment)
    assert artifact(result, "report.md")["state"] == "available"
    assert artifact(result, "results.tsv")["state"] == "available"
    assert result["selection"]["status"] == "unavailable"
    assert result["final_retrain"]["state"] == "unknown"


@pytest.mark.parametrize("name,payload", [
    ("journal.json", b'{"nodes":[],"nodes":[{"id":"overwritten"}]}'),
    ("final_retrain_result.json", b'{"ok":false,"ok":true}'),
    ("events.jsonl", b'{"event":"step_saved","best_node_id":NaN}\n'),
    ("events.jsonl", b'{"event":"step_saved","best_node_id":"x","best_node_id":"y"}\n'),
])
def test_duplicate_keys_and_nonfinite_events_are_invalid(experiment, name, payload):
    seed(experiment)
    (experiment[1] / name).write_bytes(payload)
    result = build(experiment)
    assert name.split(".")[0] + "_invalid" in result["warnings"]
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("field", ["candidate_id", "metric_name", "event_id"])
def test_lone_surrogates_cannot_break_http_utf8_encoding(experiment, field):
    seed(experiment)
    if field == "candidate_id":
        write_json(experiment[1], "journal.json", {"nodes": []})
        (experiment[1] / "journal.json").write_bytes(b'{"nodes":[{"id":"\\ud800","is_buggy":false}]}')
    elif field == "metric_name":
        raw = candidate()
        raw["metric"]["name"] = "\ud800"
        (experiment[1] / "journal.json").write_text(json.dumps({"nodes": [raw]}))
    else:
        write_events(experiment[1], [{"event": "step_saved", "node_id": "chosen", "best_node_id": "\ud800"}])
    result = build(experiment)
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if field == "metric_name":
        assert "name" not in result["selection"]["metric"]
    else:
        assert result["selection"]["status"] in {"unavailable", "conflict"}


@pytest.mark.parametrize("status", ["running", "succeeded"])
def test_new_retrain_start_cannot_be_paired_with_old_receipt(experiment, status):
    seed(experiment)
    experiment[2]["status"] = status
    write_events(experiment[1], [
        {"event": "final_retrain_started", "node_id": "older"},
        {"event": "final_retrain_finished", "ok": True},
        {"event": "final_retrain_started", "node_id": "chosen"},
    ])
    write_json(experiment[1], "final_retrain_result.json", {"ok": True})
    result = build(experiment)["final_retrain"]
    assert result["state"] == "unknown"
    assert result["reason"] == ("final_retrain_in_progress" if status == "running" else "final_retrain_incomplete")


def test_ok_receipt_with_nonempty_error_is_conflicting(experiment):
    seed(experiment)
    write_json(experiment[1], "final_retrain_result.json", {"ok": True, "error": "Private evaluation failed"})
    result = build(experiment)["final_retrain"]
    assert result["state"] == "unknown"
    assert result["reason"] == "conflicting_retrain_records"


def test_unreadable_present_receipt_prevents_disabled_claim(experiment, monkeypatch):
    seed(experiment)
    write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    monkeypatch.setitem(overview.LIMITS, "final_retrain_result.json", 1)
    write_json(experiment[1], "final_retrain_result.json", {"ok": True})
    result = build(experiment)
    assert result["final_retrain"]["state"] == "unknown"
    assert result["final_retrain"]["reason"] == "receipt_unavailable"
    assert "final_retrain_result_too_large" in result["warnings"]


def test_snapshot_revision_captures_changed_run_state(experiment):
    seed(experiment)
    first = build(experiment)["snapshot"]["revision"]
    experiment[2]["status"] = "failed"
    assert build(experiment)["snapshot"]["revision"] != first


@pytest.mark.parametrize("kind", ["malformed", "incomplete_new_start", "oversized", "too_many_events", "unreadable", "symlink"])
def test_unusable_existing_events_cannot_make_an_old_receipt_current(experiment, monkeypatch, kind):
    seed(experiment)
    root, log, run = experiment
    run["status"] = "running"
    write_json(log, "final_retrain_result.json", {"ok": True})
    path = log / "events.jsonl"
    if kind == "malformed":
        path.write_bytes(b"malformed event\n")
    elif kind == "incomplete_new_start":
        write_events(log, [{"event": "final_retrain_finished", "ok": True}])
        with path.open("ab") as stream:
            stream.write(b'{"event":"final_retrain_started","node_id":')
    elif kind == "oversized":
        monkeypatch.setitem(overview.LIMITS, "events.jsonl", 1)
    elif kind == "too_many_events":
        monkeypatch.setattr(overview, "MAX_EVENTS", 1)
    elif kind == "unreadable":
        original = overview.os.open
        def denied(name, flags, *args, **kwargs):
            if name == "events.jsonl":
                raise PermissionError(errno.EACCES, "PRIVATE SENTINEL")
            return original(name, flags, *args, **kwargs)
        monkeypatch.setattr(overview.os, "open", denied)
        monkeypatch.setattr(overview.os, "supports_dir_fd", {*os.supports_dir_fd, denied})
    else:
        outside = root.parent / "outside-events.jsonl"
        outside.write_text('{"event":"final_retrain_finished","ok":true}\n')
        path.unlink()
        path.symlink_to(outside)
    result = build(experiment)
    assert result["final_retrain"] == {"state": "unknown", "source": "events.jsonl", "node_id": None, "reason": "events_unavailable"}
    assert artifact(result, "final_retrain_result.json")["state"] == "available"
    # Independent byte equality remains correctly labelled without asserting
    # that an explicit selection record never existed in the unusable stream.
    assert result["selection"]["status"] == "code_match"
    assert result["selection"]["source"] == "best_solution.py"
    assert "PRIVATE SENTINEL" not in json.dumps(result)


def test_missing_events_keep_receipt_only_result_without_candidate_identity(experiment):
    seed(experiment)
    (experiment[1] / "events.jsonl").unlink()
    write_json(experiment[1], "final_retrain_result.json", {"ok": True})
    result = build(experiment)
    assert result["final_retrain"] == {"state": "succeeded", "source": "final_retrain_result.json", "node_id": None}


def test_unusable_events_cannot_prove_explicit_disabled_when_new_activity_is_hidden(experiment):
    seed(experiment)
    write_json(experiment[1], "run_manifest.json", {"config": {"final_retrain_enabled": False}})
    (experiment[1] / "events.jsonl").write_bytes(b'{"event":"final_retrain_started"')
    result = build(experiment)
    assert result["final_retrain"]["state"] == "unknown"
    assert result["final_retrain"]["reason"] == "events_unavailable"
    assert "final_retrain_result_missing" in result["warnings"]


@pytest.mark.parametrize("failure", [
    {"exc_type": "RuntimeError"},
    {"exc_type": "TimeoutError"},
    {"exc_type": ""},
    {"execution": {"final_failure": {"phase": "candidate_run", "type": "RuntimeError", "info": {"private": "PRIVATE FAILURE SENTINEL"}}}},
])
@pytest.mark.parametrize("ok", [True, False])
def test_formal_failure_fields_cannot_coexist_with_success(experiment, failure, ok):
    seed(experiment)
    write_json(experiment[1], "final_retrain_result.json", {"ok": ok, "error": None, **failure})
    result = build(experiment)
    if ok:
        assert result["final_retrain"]["state"] == "unknown"
        assert result["final_retrain"]["reason"] == "conflicting_retrain_records"
    else:
        assert result["final_retrain"]["state"] == "failed"
        assert "reason" not in result["final_retrain"]
    assert "PRIVATE FAILURE SENTINEL" not in json.dumps(result)
    assert "RuntimeError" not in json.dumps(result)


def test_success_receipt_with_explicit_null_failure_fields_remains_success(experiment):
    seed(experiment)
    write_json(experiment[1], "final_retrain_result.json", {
        "ok": True, "error": None, "exc_type": None,
        "execution": {"final_failure": None},
    })
    result = build(experiment)["final_retrain"]
    assert result["state"] == "succeeded"
    assert result["source"] == "final_retrain_result.json"
