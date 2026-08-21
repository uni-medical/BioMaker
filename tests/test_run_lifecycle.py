import json
from types import SimpleNamespace

import pytest

import biomaker.run as run_module
from biomaker.utils import events as events_module


def capture_run_events(monkeypatch, events):
    def record(_log_dir, event, **payload):
        events.append((event, payload))

    monkeypatch.setattr(run_module, "append_event", record)
    monkeypatch.setattr(run_module, "append_terminal_event", record)


def test_run_emits_finished_only_after_inner_run_returns(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", lambda _cfg: 35)
    capture_run_events(monkeypatch, events)

    run_module.run()

    assert events == [
        ("run_manifest_written", {"path": tmp_path / "run_manifest.json"}),
        ("run_finished", {"completed_steps": 35}),
    ]


def test_run_emits_failed_as_the_last_lifecycle_fact(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []

    def fail(_cfg):
        raise RuntimeError("report failed")

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", fail)
    capture_run_events(monkeypatch, events)

    with pytest.raises(RuntimeError, match="report failed"):
        run_module.run()

    assert events == [
        ("run_manifest_written", {"path": tmp_path / "run_manifest.json"}),
        (
            "run_failed",
            {"exc_type": "RuntimeError", "error": "report failed"},
        ),
    ]


def test_manifest_failure_is_recorded_as_run_failed(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("manifest failed")),
    )
    monkeypatch.setattr(
        run_module,
        "_run",
        lambda _cfg: pytest.fail("the workflow must not start without a manifest"),
    )
    capture_run_events(monkeypatch, events)

    with pytest.raises(RuntimeError, match="manifest failed"):
        run_module.run()

    assert events == [
        (
            "run_failed",
            {"exc_type": "RuntimeError", "error": "manifest failed"},
        )
    ]


def test_run_emits_interrupted_for_keyboard_interrupt(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []

    def interrupt(_cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", interrupt)
    capture_run_events(monkeypatch, events)

    with pytest.raises(KeyboardInterrupt):
        run_module.run()

    assert events == [
        ("run_manifest_written", {"path": tmp_path / "run_manifest.json"}),
        ("run_interrupted", {"reason": "keyboard_interrupt"}),
    ]


def test_run_records_sigterm_before_terminating(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []
    installed_handlers = []

    def install(_signal_number, handler):
        installed_handlers.append(handler)
        return "previous-handler"

    def terminate(_cfg):
        installed_handlers[0](run_module.signal.SIGTERM, None)

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", terminate)
    monkeypatch.setattr(run_module.signal, "signal", install)
    capture_run_events(monkeypatch, events)

    with pytest.raises(SystemExit):
        run_module.run()

    assert events == [
        ("run_manifest_written", {"path": tmp_path / "run_manifest.json"}),
        (
            "run_interrupted",
            {"reason": "sigterm", "signal": run_module.signal.SIGTERM},
        ),
    ]
    assert installed_handlers[-1] == "previous-handler"


def test_nonterminal_event_write_failure_remains_fail_soft(tmp_path, monkeypatch):
    monkeypatch.setattr(
        events_module,
        "_write_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    events_module.append_event(tmp_path, "round_finished", completed_steps=35)


def test_terminal_event_is_flushed_and_fsynced(tmp_path, monkeypatch):
    fsync_calls = []
    monkeypatch.setattr(events_module.os, "fsync", fsync_calls.append)

    events_module.append_terminal_event(
        tmp_path,
        "run_finished",
        completed_steps=35,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == ["run_finished"]
    assert records[0]["completed_steps"] == 35
    assert len(fsync_calls) == 1


def test_terminal_write_failure_prevents_success(tmp_path, monkeypatch):
    cfg = SimpleNamespace(log_dir=tmp_path)
    events = []
    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", lambda _cfg: 35)
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )
    monkeypatch.setattr(
        run_module,
        "append_terminal_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("terminal event fsync failed")
        ),
    )

    with pytest.raises(OSError, match="terminal event fsync failed"):
        run_module.run()

    assert events == [
        ("run_manifest_written", {"path": tmp_path / "run_manifest.json"})
    ]


def test_sigterm_reentry_during_terminal_write_does_not_reserve_second_event(
    tmp_path, monkeypatch
):
    cfg = SimpleNamespace(log_dir=tmp_path)
    installed_handlers = []
    attempts = []

    def install(_signal_number, handler):
        installed_handlers.append(handler)
        return "previous-handler"

    def interrupt_writer(_log_dir, event, **_payload):
        attempts.append(event)
        installed_handlers[0](run_module.signal.SIGTERM, None)

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", lambda _cfg: 35)
    monkeypatch.setattr(run_module.signal, "signal", install)
    monkeypatch.setattr(run_module, "append_terminal_event", interrupt_writer)

    with pytest.raises(SystemExit, match=str(128 + run_module.signal.SIGTERM)):
        run_module.run()

    assert attempts == ["run_finished"]
    assert installed_handlers[-1] == "previous-handler"


def test_fsync_failure_after_terminal_bytes_does_not_append_second_terminal(
    tmp_path, monkeypatch
):
    cfg = SimpleNamespace(log_dir=tmp_path)
    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda _cfg, _task: tmp_path / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "_run", lambda _cfg: 35)
    monkeypatch.setattr(
        events_module.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("terminal fsync failed")),
    )

    with pytest.raises(OSError, match="terminal fsync failed"):
        run_module.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = [
        record
        for record in records
        if record["event"] in {"run_finished", "run_failed", "run_interrupted"}
    ]
    assert [record["event"] for record in terminal] == ["run_finished"]
