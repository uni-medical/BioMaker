"""The desktop result adapters use registered evidence, without execution."""
import hashlib
import json

from test_desktop_service import AUTH, client_for, config, create_desktop_app


def fixture_run(app, config):
    directory = config.workspace / "logs" / "route-fixture"
    directory.mkdir(parents=True)
    code = "# Synthetic route fixture; not executable training\r\n"
    files = {
        "journal.json": json.dumps({"nodes": [{"id": "chosen", "step": 0, "is_buggy": False, "code": code, "metric": {"value": 0, "maximize": False}}]}),
        "events.jsonl": json.dumps({"event": "step_saved", "node_id": "chosen", "best_node_id": "chosen"}) + "\n",
        "best_solution.py": code,
        "run_manifest.json": json.dumps({"config": {"final_retrain_enabled": False}}),
        "report.md": "\ufeff# 原始报告\r\nSynthetic report, not evaluation evidence.\r\n",
    }
    for name, value in files.items():
        (directory / name).write_bytes(value.encode("utf-8"))
    record = {"id": "route-fixture", "status": "complete", "managed": False, "read_only": True, "log_dir": str(directory)}
    app.state.run_store.add_record("runs", record)
    return directory, record


def test_result_and_preview_require_desktop_auth_and_origin(config):
    app = create_desktop_app(config)
    fixture_run(app, config)
    with client_for(app) as client:
        for route in ["/api/desktop/runs/route-fixture/result", "/api/desktop/runs/route-fixture/artifacts/report.md/preview"]:
            assert client.get(route).status_code == 401
            assert client.get(route, headers={**AUTH, "Origin": "https://example.org"}).status_code == 403


def test_unregistered_run_is_not_discovered_by_reading_result(config):
    app = create_desktop_app(config)
    with client_for(app) as client:
        for route in ["/api/desktop/runs/unknown/result", "/api/desktop/runs/unknown/artifacts/report.md/preview"]:
            assert client.get(route, headers=AUTH).status_code == 404
        assert app.state.run_store.list_records("runs") == []


def test_routes_return_same_bytes_and_recorded_selection_without_writes(config):
    app = create_desktop_app(config)
    directory, record = fixture_run(app, config)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    with client_for(app) as client:
        stored = app.state.run_store.get_record("runs", record["id"])
        result = client.get("/api/desktop/runs/route-fixture/result", headers=AUTH)
        assert result.status_code == 200
        assert result.headers["cache-control"] == "no-store"
        overview = result.json()
        assert overview["selection"]["status"] == "recorded"
        assert overview["selection"]["node_id"] == "chosen"
        assert overview["selection"]["metric"]["value"] == 0
        assert overview["final_retrain"]["state"] == "disabled"
        preview = client.get("/api/desktop/runs/route-fixture/artifacts/report.md/preview", headers=AUTH)
        assert preview.status_code == 200
        assert preview.headers["cache-control"] == "no-store"
        payload = preview.json()
        original = before["report.md"][0]
        assert payload["content"].encode("utf-8") == original
        assert payload["sha256"] == hashlib.sha256(original).hexdigest()
        assert payload["size"] == len(original)
        assert app.state.run_store.get_record("runs", record["id"]) == stored
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == before


def test_preview_refusal_has_fixed_message_and_no_private_path(config):
    app = create_desktop_app(config)
    directory, _ = fixture_run(app, config)
    (directory / "report.md").write_bytes(b"\xffinvalid UTF8")
    with client_for(app) as client:
        response = client.get("/api/desktop/runs/route-fixture/artifacts/report.md/preview", headers=AUTH)
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_text"
        assert "[artifact:invalid_text]" in response.json()["detail"]
        assert str(directory) not in response.text
