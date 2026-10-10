import json

import pytest

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
    assert run["event_tail"] == {
        "valid_event_limit": 5000, "valid_event_count": len(events),
        "terminal_event": "run_finished", "reason_code": None,
    }


@pytest.mark.parametrize("events", [[], [{"event": "run_started"}], [{"event": "run_failed"}]])
def test_initial_history_lifecycle_inference_stays_conservative(settings, events):
    directory = settings.logs_root / "initial-history"
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({"data_dir": "/synthetic"}))
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    run = run_from_log_dir(settings, directory)
    terminal = "run_failed" if events and events[-1]["event"] == "run_failed" else None
    assert run["status"] == "incomplete"
    assert run["detailed_status"] == ("failed" if terminal else "orphaned")
    assert run["status_source"] == "events"
    assert run["read_only"] is True
    assert run["allowed_actions"] == []
    assert run["event_tail"] == {
        "valid_event_limit": 5000, "valid_event_count": len(events),
        "terminal_event": terminal,
        "reason_code": None if terminal else "no_terminal_in_tail",
    }
    if terminal is None:
        assert "last up to 5,000 valid event records" in run["status_reason"]
        assert "does not determine whether the original process is running" in run["status_reason"]
