import signal
from types import SimpleNamespace

import psutil
import pytest

from biomaker.utils.process_identity import (
    ProcessIdentity,
    ProcessIdentityError,
    ProcessSnapshot,
    capture_process_identity,
    is_process_alive,
    read_process_snapshot,
    read_session_process_groups,
    terminate_process_group,
    terminate_process_session,
    verify_process_identity,
)


def _proc_stat(pid: int, pgid: int, start_ticks: int) -> str:
    suffix = ["S", "1", str(pgid), *("0" for _ in range(16)), str(start_ticks)]
    return f"{pid} (worker ) with spaces) {' '.join(suffix)}\n"


def test_linux_proc_snapshot_handles_parentheses_and_spaces_in_process_name():
    snapshot = read_process_snapshot(
        123,
        platform_name="Linux",
        proc_stat_reader=lambda pid: _proc_stat(pid, 456, 987654),
    )

    assert snapshot == ProcessSnapshot(
        pid=123,
        pgid=456,
        start_time="proc:987654",
    )


def test_ps_fallback_brackets_ps_with_matching_precise_creation_times():
    calls = []

    def run(command, **kwargs):
        calls.append(("ps", command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=" 123  456\n",
        )

    def create_time(_pid):
        calls.append(("create_time",))
        return 1787287883.123456

    snapshot = read_process_snapshot(
        123,
        platform_name="Darwin",
        ps_runner=run,
        create_time_reader=create_time,
    )

    assert snapshot == ProcessSnapshot(
        pid=123,
        pgid=456,
        start_time="ps:precise:1787287883.123456",
    )
    assert calls == [
        ("create_time",),
        (
            "ps",
            [
                "ps",
                "-o",
                "pid=",
                "-o",
                "pgid=",
                "-p",
                "123",
            ],
            {
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 2,
            },
        ),
        ("create_time",),
    ]


def test_ps_fallback_prefers_precise_process_creation_time():
    snapshot = read_process_snapshot(
        123,
        platform_name="Darwin",
        ps_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="123 123\n",
        ),
        create_time_reader=lambda _pid: 1787287883.123456,
    )

    assert snapshot == ProcessSnapshot(
        pid=123,
        pgid=123,
        start_time="ps:precise:1787287883.123456",
    )


def test_ps_fallback_rejects_pid_reuse_between_precise_time_reads():
    create_times = iter([1000.125, 1000.25])

    with pytest.raises(ProcessIdentityError, match="changed identity"):
        read_process_snapshot(
            123,
            platform_name="Darwin",
            ps_runner=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="123 123\n",
            ),
            create_time_reader=lambda _pid: next(create_times),
        )


@pytest.mark.parametrize("create_time", [0.0, -1.0, float("inf"), float("nan")])
def test_ps_fallback_rejects_non_positive_or_non_finite_creation_time(create_time):
    with pytest.raises(ProcessIdentityError, match="invalid precise creation time"):
        read_process_snapshot(
            123,
            platform_name="Darwin",
            ps_runner=lambda *_args, **_kwargs: pytest.fail("must not run ps"),
            create_time_reader=lambda _pid: create_time,
        )


def test_psutil_no_such_process_maps_to_missing(monkeypatch):
    identity = ProcessIdentity(123, 123, "ps:precise:1.0", "launch-abc")

    def missing(_pid):
        raise psutil.NoSuchProcess(123)

    monkeypatch.setattr(psutil, "Process", missing)
    verification = verify_process_identity(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda pid: read_process_snapshot(
            pid,
            platform_name="Darwin",
            ps_runner=lambda *_args, **_kwargs: pytest.fail("must not run ps"),
        ),
    )

    assert verification.reason == "missing"


def test_psutil_access_denied_maps_to_unreadable(monkeypatch):
    identity = ProcessIdentity(123, 123, "ps:precise:1.0", "launch-abc")

    def unreadable(_pid):
        raise psutil.AccessDenied(123)

    monkeypatch.setattr(psutil, "Process", unreadable)
    verification = verify_process_identity(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda pid: read_process_snapshot(
            pid,
            platform_name="Darwin",
            ps_runner=lambda *_args, **_kwargs: pytest.fail("must not run ps"),
        ),
    )

    assert verification.reason == "unreadable"


def test_capture_round_trips_and_exact_identity_matches():
    observed = ProcessSnapshot(100, 100, "proc:55")
    identity = capture_process_identity(
        100,
        "launch-abc",
        snapshot_reader=lambda _pid: observed,
    )

    assert identity.as_dict() == {
        "pid": 100,
        "pgid": 100,
        "start_time": "proc:55",
        "launch_token": "launch-abc",
    }
    assert ProcessIdentity.from_mapping(identity.as_dict()) == identity
    verification = verify_process_identity(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: observed,
    )
    assert verification.matched
    assert verification.reason == "match"
    assert is_process_alive(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: observed,
    )


@pytest.mark.parametrize(
    ("observed", "reason"),
    [
        (ProcessSnapshot(999, 100, "proc:55"), "pid_mismatch"),
        (ProcessSnapshot(100, 999, "proc:55"), "process_group_mismatch"),
        (ProcessSnapshot(100, 100, "proc:99"), "start_time_mismatch"),
    ],
)
def test_reused_or_changed_pid_is_not_alive(observed, reason):
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")

    verification = verify_process_identity(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: observed,
    )

    assert not verification.matched
    assert verification.reason == reason
    assert not is_process_alive(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: observed,
    )


def test_missing_process_is_not_alive():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")

    def missing(_pid):
        raise ProcessLookupError

    verification = verify_process_identity(
        identity,
        launch_token="launch-abc",
        snapshot_reader=missing,
    )

    assert verification.reason == "missing"
    assert not is_process_alive(
        identity,
        launch_token="launch-abc",
        snapshot_reader=missing,
    )


def test_launch_token_mismatch_is_rejected_without_reading_or_signaling():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []

    result = terminate_process_group(
        identity,
        launch_token="wrong-run",
        snapshot_reader=lambda _pid: pytest.fail("must not inspect a different run"),
        group_probe=lambda _pgid: pytest.fail("must not probe a rejected group"),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
    )

    assert result.outcome == "rejected"
    assert result.signals_sent == ()
    assert calls == []


def test_stale_reused_pid_is_rejected_without_signaling():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:999"),
        group_probe=lambda _pgid: pytest.fail("must not probe a stale identity"),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
    )

    assert result.outcome == "rejected"
    assert result.verification.reason == "start_time_mismatch"
    assert calls == []


def test_missing_process_returns_already_exited_without_signaling():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []

    def missing(_pid):
        raise ProcessLookupError

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        snapshot_reader=missing,
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
    )

    assert result.outcome == "already_exited"
    assert calls == []


def test_matching_group_gets_term_only_when_it_exits_during_grace_period():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:55"),
        group_probe=lambda _pgid: False,
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
    )

    assert result.outcome == "terminated"
    assert result.signals_sent == (signal.SIGTERM,)
    assert calls == [(100, signal.SIGTERM)]


def test_matching_group_escalates_from_term_to_kill_after_bounded_wait():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    clock = [0.0]

    def sleep(duration):
        clock[0] += duration

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        grace_seconds=0.2,
        poll_interval=0.1,
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:55"),
        group_probe=lambda _pgid: True,
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )

    assert result.outcome == "killed"
    assert result.signals_sent == (signal.SIGTERM, signal.SIGKILL)
    assert calls == [(100, signal.SIGTERM), (100, signal.SIGKILL)]
    assert clock[0] == pytest.approx(0.2)


def test_identity_change_during_grace_period_prevents_kill_escalation():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    clock = [0.0]
    snapshots = iter(
        [
            ProcessSnapshot(100, 100, "proc:55"),
            ProcessSnapshot(100, 100, "proc:999"),
        ]
    )

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        grace_seconds=0,
        snapshot_reader=lambda _pid: next(snapshots),
        group_probe=lambda _pgid: True,
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=777,
        monotonic=lambda: clock[0],
        sleeper=lambda duration: clock.__setitem__(0, clock[0] + duration),
    )

    assert result.outcome == "rejected"
    assert result.verification.reason == "start_time_mismatch"
    assert result.signals_sent == (signal.SIGTERM,)
    assert calls == [(100, signal.SIGTERM)]


def test_current_process_group_is_never_signaled():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []

    result = terminate_process_group(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:55"),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_pgid=100,
    )

    assert result.outcome == "rejected"
    assert result.reason == "refusing to signal an unsafe process group"
    assert calls == []


def test_session_group_reader_enumerates_only_the_requested_session():
    result = read_session_process_groups(
        100,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="100\n101\n200\ninvalid\n",
        ),
        sid_reader=lambda pid: {100: 100, 101: 100, 200: 200}[pid],
        pgid_reader=lambda pid: {100: 100, 101: 101, 200: 200}[pid],
    )

    assert result == {100, 101}


def test_session_group_reader_ignores_zombies_but_keeps_live_group_members():
    result = read_session_process_groups(
        100,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="100 Z\n101 S\n102 Z+\n",
        ),
        sid_reader=lambda _pid: 100,
        pgid_reader=lambda pid: {100: 100, 101: 100, 102: 102}[pid],
    )

    assert result == {100}


def test_matching_run_session_terminates_every_process_group_before_returning():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    observed_groups = iter([{100, 101}, set()])

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:55"),
        session_groups_reader=lambda _sid: next(observed_groups),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        sid_reader=lambda _pid: 100,
        current_sid=777,
    )

    assert result.outcome == "terminated"
    assert result.signals_sent == (signal.SIGTERM,)
    assert calls == [
        (101, signal.SIGTERM),
        (100, signal.SIGTERM),
    ]


def test_missing_session_leader_still_drains_its_remaining_process_groups():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    observed_groups = iter([{101, 102}, set()])

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
        session_groups_reader=lambda _sid: next(observed_groups),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_sid=777,
    )

    assert result.outcome == "terminated"
    assert result.signals_sent == (signal.SIGTERM,)
    assert calls == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
    ]


def test_missing_session_leader_with_no_descendants_is_already_exited():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
        session_groups_reader=lambda _sid: set(),
        signal_sender=lambda *_args: pytest.fail("must not signal"),
        current_sid=777,
    )

    assert result.outcome == "already_exited"
    assert result.reason == "managed session has no remaining process groups"


@pytest.mark.parametrize(
    ("replacement", "expected_reason"),
    [
        (ProcessSnapshot(100, 100, "proc:55"), "match"),
        (ProcessSnapshot(100, 100, "proc:999"), "start_time_mismatch"),
        (ProcessIdentityError("denied"), "unreadable"),
    ],
)
def test_missing_leader_change_during_session_enumeration_is_rejected_without_signal(
    replacement, expected_reason
):
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    reads = 0

    def snapshot(_pid):
        nonlocal reads
        reads += 1
        if reads <= 2:
            raise ProcessLookupError
        if isinstance(replacement, BaseException):
            raise replacement
        return replacement

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        snapshot_reader=snapshot,
        session_groups_reader=lambda _sid: {101},
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_sid=777,
    )

    assert result.outcome == "rejected"
    assert result.verification is not None
    assert result.verification.reason == expected_reason
    assert calls == []


def test_session_identity_change_stops_cleanup_before_any_new_signal():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    snapshots = iter(
        [
            ProcessSnapshot(100, 100, "proc:55"),
            ProcessSnapshot(100, 100, "proc:999"),
        ]
    )

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        snapshot_reader=lambda _pid: next(snapshots),
        session_groups_reader=lambda _sid: {100},
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        sid_reader=lambda _pid: 100,
        current_sid=777,
    )

    assert result.outcome == "rejected"
    assert "identity changed" in result.reason
    assert calls == []


def test_missing_leader_reappearance_before_kill_escalation_is_rejected():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    reads = 0
    observed_groups = iter([{101}, {101}])

    def snapshot(_pid):
        nonlocal reads
        reads += 1
        if reads < 6:
            raise ProcessLookupError
        return ProcessSnapshot(100, 100, "proc:55")

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        grace_seconds=0,
        snapshot_reader=snapshot,
        session_groups_reader=lambda _sid: next(observed_groups),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        current_sid=777,
    )

    assert result.outcome == "rejected"
    assert result.verification is not None
    assert result.verification.reason == "match"
    assert result.signals_sent == (signal.SIGTERM,)
    assert calls == [(101, signal.SIGTERM)]


def test_session_kill_reenumerates_groups_created_during_escalation():
    identity = ProcessIdentity(100, 100, "proc:55", "launch-abc")
    calls = []
    observed_groups = iter([{100}, {100}, {100}, {101}, set()])
    clock = [0.0]

    result = terminate_process_session(
        identity,
        launch_token="launch-abc",
        grace_seconds=0,
        poll_interval=0.1,
        snapshot_reader=lambda _pid: ProcessSnapshot(100, 100, "proc:55"),
        session_groups_reader=lambda _sid: next(observed_groups),
        signal_sender=lambda pgid, sig: calls.append((pgid, sig)),
        sid_reader=lambda _pid: 100,
        current_sid=777,
        monotonic=lambda: clock[0],
        sleeper=lambda duration: clock.__setitem__(0, clock[0] + duration),
    )

    assert result.outcome == "killed"
    assert result.signals_sent == (signal.SIGTERM, signal.SIGKILL)
    assert calls == [
        (100, signal.SIGTERM),
        (100, signal.SIGKILL),
        (101, signal.SIGKILL),
    ]
