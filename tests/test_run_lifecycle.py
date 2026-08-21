from types import SimpleNamespace

import pytest

import biomaker.run as run_module


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
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )

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
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )

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
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )

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
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )

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
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )

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
