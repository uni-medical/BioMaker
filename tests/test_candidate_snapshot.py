import json
from types import SimpleNamespace

from biomaker.utils import candidate_snapshot


def node(identifier="candidate", **changes):
    values = dict(id=identifier, step=None, scheduled_step=0, parent=None, tag="generating", agent_name="proposal", node_status="draft", plan="plan", selection_reason="reason", code="", private_metrics={"private": "sentinel"}, prompt_messages=["private prompt"])
    values.update(changes)
    return SimpleNamespace(**values)


def test_atomic_public_snapshot_shows_generated_code_without_validity(tmp_path):
    candidate_snapshot.write_candidate_snapshot(tmp_path, [node()])
    path = tmp_path / "candidates_live.json"
    first = json.loads(path.read_text())
    assert first["nodes"][0]["phase"] == "generating"
    candidate_snapshot.write_candidate_snapshot(tmp_path, [node(tag="generated", code="print('real generated code')")])
    second = json.loads(path.read_text())
    assert second["nodes"][0]["phase"] == "queued"
    assert second["nodes"][0]["code"] == "print('real generated code')"
    assert "private" not in path.read_text() and "prompt_messages" not in path.read_text()
    assert "metric" not in second["nodes"][0] and "is_buggy" not in second["nodes"][0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates_live.json"]


def test_snapshot_keeps_lifecycle_rows_when_code_exceeds_size_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate_snapshot, "MAX_BYTES", 2000)
    candidate_snapshot.write_candidate_snapshot(tmp_path, [node("one", code="x" * 4000), node("two", tag="running", code="y" * 4000)])
    path = tmp_path / "candidates_live.json"
    assert path.stat().st_size <= 2000
    result = json.loads(path.read_text())
    assert [n["id"] for n in result["nodes"]] == ["one", "two"]
    assert result["nodes"][1]["phase"] == "running"
    assert all(n["content_truncated"] for n in result["nodes"])


def test_failed_publish_preserves_previous_snapshot_and_cleans_temporary_file(tmp_path, monkeypatch):
    import pytest
    candidate_snapshot.write_candidate_snapshot(tmp_path, [node()])
    original = (tmp_path / "candidates_live.json").read_bytes()
    def fail(*_args):
        raise OSError("synthetic publish failure")
    monkeypatch.setattr(candidate_snapshot.os, "replace", fail)
    with pytest.raises(OSError):
        candidate_snapshot.write_candidate_snapshot(tmp_path, [node(tag="generated", code="new")])
    assert (tmp_path / "candidates_live.json").read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates_live.json"]


def test_real_workflow_publishes_before_any_candidate_result_and_after_generation(tmp_path, monkeypatch):
    import biomaker.run as run_module
    from test_presentation_failure_lifecycle import _install_minimal_run
    cfg, events, _order, _live = _install_minimal_run(monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    snapshots = []
    original = candidate_snapshot.write_candidate_snapshot
    def observe(log_dir, nodes):
        original(log_dir, nodes)
        snapshots.append(json.loads((log_dir / "candidates_live.json").read_text()))
    monkeypatch.setattr(run_module, "write_candidate_snapshot", observe)
    run_module.run()
    assert any(n["phase"] == "generating" and not n["code"] for snapshot in snapshots for n in snapshot["nodes"])
    assert any(n["code"] == "print('candidate')" for snapshot in snapshots for n in snapshot["nodes"])
    names = [name for name, _ in events]
    assert names.index("candidate_validation_started") < names.index("candidate_evaluation_started") < names.index("execution_finished")
