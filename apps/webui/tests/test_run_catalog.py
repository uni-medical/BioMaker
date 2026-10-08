import json

from webui.run_catalog import run_from_log_dir


def test_unstructured_event_names_do_not_hide_historical_experiments(settings):
    directory = settings.logs_root / "mixed-event-schema"
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({
        "data_dir": str(settings.datasets_root / "example"),
        "config": {"agent_steps": 3},
    }))
    events = [{"event": "run_started"}, {"event": "run_finished", "completed_steps": 2}]
    events.extend({"event": name} for name in [{"malformed": True}, [], None, 12, "constructor"])
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    run = run_from_log_dir(settings, directory)
    assert run["status"] == "complete"
    assert run["terminal_event"] == "run_finished"
    assert run["progress"]["completed_steps"] == 2
