from __future__ import annotations

import json
import math
import pytest
from webui.artifacts import (
    ArtifactError,
    artifact_collection_revision,
    list_artifacts,
    read_events,
    read_results_summary,
    resolve_artifact,
)
from starlette.responses import JSONResponse
from fastapi.testclient import TestClient
from webui.server import create_app


def test_artifact_resolution_is_confined_to_logs_root(settings, tmp_path):
    log_dir = settings.logs_root / "1-safe-run"
    log_dir.mkdir()
    tree = log_dir / "tree_plot.html"
    tree.write_text("<html>tree</html>", encoding="utf-8")
    run = {"id": "run", "log_dir": str(log_dir)}

    assert resolve_artifact(settings, run, "tree_plot.html") == tree.resolve()
    with pytest.raises(ArtifactError, match="Invalid artifact name"):
        resolve_artifact(settings, run, "../tree_plot.html")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tree_plot.html").write_text("outside", encoding="utf-8")
    with pytest.raises(ArtifactError, match="outside"):
        resolve_artifact(
            settings,
            {"id": "bad", "log_dir": str(outside)},
            "tree_plot.html",
        )


def test_artifact_listing_excludes_direct_symlinks(settings, tmp_path):
    log_dir = settings.logs_root / "2-safe-run"
    log_dir.mkdir()
    (log_dir / "results.tsv").write_text("metric\n1\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (log_dir / "linked-secret.txt").symlink_to(outside)

    run = {"id": "r", "log_dir": str(log_dir)}
    artifacts = list_artifacts(settings, run)
    names = [item["name"] for item in artifacts]
    assert names == ["results.tsv"]
    assert isinstance(artifacts[0]["revision"], str)

    first_revision = artifacts[0]["revision"]
    first_collection_revision = artifact_collection_revision(artifacts)
    (log_dir / "results.tsv").write_text("metric\n10\n", encoding="utf-8")
    changed = list_artifacts(settings, run)

    assert changed[0]["revision"] != first_revision
    assert artifact_collection_revision(changed) != first_collection_revision
    assert artifact_collection_revision([]) is None


@pytest.mark.parametrize("maximize, expected", [("True", 0.8), ("False", 0.2)])
def test_metric_summary_ignores_undefined_scores_and_preserves_direction(
    settings, maximize, expected
):
    log_dir = settings.logs_root / "undefined-evaluator-score"
    log_dir.mkdir()
    values = ["nan", "0.2", "inf", "not-a-score", "0.8", "-inf"]
    (log_dir / "results.tsv").write_text(
        "step\tmetric\tmaximize\n"
        + "".join(
            f"{step}\t{metric}\t{maximize}\n" for step, metric in enumerate(values)
        ),
        encoding="utf-8",
    )
    summary = read_results_summary(settings, {"log_dir": str(log_dir)})
    assert summary["count"] == len(values)
    assert summary["best_metric"] == expected
    assert summary["latest"]["metric"] == "-inf"
    assert JSONResponse(summary).status_code == 200


def test_metric_summary_with_no_finite_score_has_no_best_metric(settings):
    log_dir = settings.logs_root / "no-comparable-score"
    log_dir.mkdir()
    (log_dir / "results.tsv").write_text(
        "step\tmetric\tmaximize\n0\tnan\tTrue\n1\t-inf\tTrue\n",
        encoding="utf-8",
    )
    summary = read_results_summary(settings, {"log_dir": str(log_dir)})
    assert summary["count"] == 2
    assert summary["best_metric"] is None
    assert JSONResponse(summary).status_code == 200


def test_event_nonfinite_numbers_become_explicit_strings_without_losing_diagnostics(settings):
    directory = settings.logs_root / "nonfinite-event"
    directory.mkdir()
    raw = (b'{"event":"execution_finished","exc_type":"ValueError","error":"undefined score",'
           b'"is_buggy":true,"ok":false,"metric":NaN,"memory":Infinity,'
           b'"nested":[-Infinity,1e999,-1e999,{"finite":1.25,"zero":-0.0}],'
           b'"step":4,"label":"NaN","optional":null}\n')
    path = directory / "events.jsonl"
    path.write_bytes(raw)
    before = path.stat().st_mtime_ns
    result = read_events(settings, {"log_dir": str(directory)})
    assert result["count"] == 1
    event = result["events"][0]
    assert event["event"] == "execution_finished"
    assert event["exc_type"] == "ValueError" and event["error"] == "undefined score"
    assert event["is_buggy"] is True and event["ok"] is False
    assert event["metric"] == "NaN" and event["memory"] == "Infinity"
    assert event["nested"][:3] == ["-Infinity", "Infinity", "-Infinity"]
    assert event["nested"][3]["finite"] == 1.25
    assert math.copysign(1, event["nested"][3]["zero"]) == -1
    assert event["step"] == 4 and event["label"] == "NaN" and event["optional"] is None
    assert JSONResponse(result).status_code == 200
    assert path.read_bytes() == raw and path.stat().st_mtime_ns == before


@pytest.mark.parametrize("route", ["events", "detail"])
def test_event_and_detail_http_responses_preserve_nonfinite_failure_record(settings, route):
    directory = settings.logs_root / "nonfinite-api"
    directory.mkdir()
    raw = (b'{"event":"run_failed","error":"failure with undefined metric",'
           b'"exc_type":"EvaluatorError","metric":NaN,"ok":false,"is_buggy":true,'
           b'"resource_usage":{"memory":Infinity,"score":-1e999}}\n')
    path = directory / "events.jsonl"
    path.write_bytes(raw)
    app = create_app(settings)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        app.state.run_store.add_record("runs", {"id": "nonfinite-api", "status": "failed",
                                              "managed": False, "read_only": True,
                                              "log_dir": str(directory)})
        suffix = "/events" if route == "events" else ""
        response = client.get(f"/api/runs/nonfinite-api{suffix}")
        assert response.status_code == 200
        payload = response.json()
        # Strict JSON serialization remains possible; nonfinite floats cannot
        # hide in nested diagnostics or prevent opening the run details.
        json.dumps(payload, allow_nan=False)
        event = payload["events"][0] if route == "events" else payload["observed"]["last_event"]
        assert event["event"] == "run_failed" and event["exc_type"] == "EvaluatorError"
        assert event["error"] == "failure with undefined metric"
        assert event["ok"] is False and event["is_buggy"] is True
        assert event["metric"] == "NaN"
        assert event["resource_usage"] == {"memory": "Infinity", "score": "-Infinity"}
    assert path.read_bytes() == raw
