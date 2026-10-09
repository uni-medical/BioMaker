from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest

from biomaker.utils.process_tree import (
    ProcessTreeIdentity,
    ProcessTreeInspectionError,
    ProcessTreeTracker,
    _read_ps_identity,
    drain_uncaptured_process_group,
    list_process_group_members,
)


def _identity(
    pid: int,
    *,
    pgid: int,
    sid: int = 50,
    started: str | None = None,
) -> ProcessTreeIdentity:
    return ProcessTreeIdentity(
        pid=pid,
        pgid=pgid,
        sid=sid,
        start_time=started or f"start-{pid}",
    )


def test_tracker_captures_descendants_and_never_signals_a_reused_pid():
    processes = {
        10: _identity(10, pgid=10),
        11: _identity(11, pgid=10),
        12: _identity(12, pgid=10),
    }
    signals: list[tuple[int, int]] = []

    def read(pid: int) -> ProcessTreeIdentity:
        try:
            return processes[pid]
        except KeyError:
            raise ProcessLookupError(pid) from None

    def send(pid: int, sent_signal: int) -> None:
        signals.append((pid, sent_signal))
        if sent_signal == signal.SIGKILL:
            processes.pop(pid, None)

    tracker = ProcessTreeTracker.capture(
        10,
        identity_reader=read,
        group_reader=lambda pgid: [
            pid for pid, identity in processes.items() if identity.pgid == pgid
        ],
        signal_sender=send,
        sleeper=lambda _seconds: None,
    )
    assert tracker.refresh() == 2
    assert tracker.tracked_pids == (10, 11, 12)

    # PID 11 now denotes an unrelated incarnation.  Every durable identity fact
    # differs, so it must not be signalled even though the PID was once tracked.
    processes[11] = _identity(11, pgid=99, sid=77, started="reused-11")
    tracker.terminate_and_wait(poll_interval=0.001)

    assert 11 in processes
    assert all(pid != 11 for pid, _sent_signal in signals)
    assert (12, signal.SIGKILL) in signals
    assert (10, signal.SIGKILL) in signals


def test_group_reader_ignores_a_member_from_an_unrelated_session():
    processes = {
        20: _identity(20, pgid=20, sid=50),
        21: _identity(21, pgid=20, sid=99),
    }
    signals: list[tuple[int, int]] = []

    def read(pid: int) -> ProcessTreeIdentity:
        try:
            return processes[pid]
        except KeyError:
            raise ProcessLookupError(pid) from None

    tracker = ProcessTreeTracker.capture(
        20,
        identity_reader=read,
        group_reader=lambda pgid: [
            pid for pid, identity in processes.items() if identity.pgid == pgid
        ],
        signal_sender=lambda pid, sent_signal: (
            signals.append((pid, sent_signal)),
            processes.pop(pid, None) if sent_signal == signal.SIGKILL else None,
        )[-1],
        sleeper=lambda _seconds: None,
    )

    assert tracker.refresh() == 0
    assert tracker.tracked_pids == (20,)
    processes.pop(21)

    tracker.terminate_and_wait(poll_interval=0.001)

    assert (20, signal.SIGKILL) in signals
    assert all(pid != 21 for pid, _sent_signal in signals)
    assert 20 not in processes


def test_system_group_reader_ignores_zombies_but_keeps_live_group_members(
    monkeypatch,
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="10 50 Z\n11 50 S\n12 50 Z+\n13 50 R+\n14 99 S\n",
        )

    monkeypatch.setattr("biomaker.utils.process_tree.subprocess.run", run)

    assert list_process_group_members(50) == [11, 13]
    assert calls[0][0] == ["ps", "-axo", "pid=,pgid=,stat="]


def test_cached_zombie_identity_is_not_signalled_after_live_group_reader_drops_it():
    zombie_identity = _identity(25, pgid=25)
    signals = []
    tracker = ProcessTreeTracker.capture(
        25,
        # A zombie remains readable through /proc with the same durable identity.
        identity_reader=lambda _pid: zombie_identity,
        # The system group reader filters it because it is no longer executable.
        group_reader=lambda _pgid: [],
        signal_sender=lambda pid, sent_signal: signals.append((pid, sent_signal)),
        sleeper=lambda _seconds: None,
    )

    tracker.terminate_and_wait(poll_interval=0.001)

    assert signals == []


def test_terminate_waits_until_a_known_process_is_observed_gone():
    processes = {30: _identity(30, pgid=30)}
    kill_attempts = 0
    sleeps = 0

    def read(pid: int) -> ProcessTreeIdentity:
        try:
            return processes[pid]
        except KeyError:
            raise ProcessLookupError(pid) from None

    def send(pid: int, sent_signal: int) -> None:
        nonlocal kill_attempts
        if sent_signal != signal.SIGKILL:
            return
        kill_attempts += 1
        if kill_attempts == 3:
            processes.pop(pid, None)

    def sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1

    tracker = ProcessTreeTracker.capture(
        30,
        identity_reader=read,
        group_reader=lambda pgid: [
            pid for pid, identity in processes.items() if identity.pgid == pgid
        ],
        signal_sender=send,
        sleeper=sleep,
    )

    tracker.terminate_and_wait(poll_interval=0.001)

    assert kill_attempts == 3
    assert sleeps >= 3
    assert processes == {}


def test_portable_identity_uses_precise_create_time_not_second_resolution():
    calls = []
    created_at = 1724241000.123456

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="42 42\n")

    identity = _read_ps_identity(
        42,
        command_runner=run,
        create_time_reader=lambda _pid: created_at,
        session_reader=lambda _pid: 7,
    )

    assert identity == ProcessTreeIdentity(
        pid=42,
        pgid=42,
        sid=7,
        start_time=f"create:{created_at.hex()}",
    )
    assert calls[0][0] == [
        "ps",
        "-o",
        "pid=",
        "-o",
        "pgid=",
        "-p",
        "42",
    ]


def test_portable_identity_fails_closed_if_pid_is_reused_during_inspection():
    create_times = iter([1000.1, 1000.2])

    with pytest.raises(ProcessTreeInspectionError, match="changed identity"):
        _read_ps_identity(
            42,
            command_runner=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="42 42\n",
            ),
            create_time_reader=lambda _pid: next(create_times),
            session_reader=lambda _pid: 7,
        )


def test_current_process_can_be_captured_as_a_real_identity():
    # This keeps both the Linux /proc and macOS ps + psutil readers exercised in CI.
    from biomaker.utils.process_tree import read_process_tree_identity

    identity = read_process_tree_identity(os.getpid())

    assert identity.pid == os.getpid()
    assert identity.pgid == os.getpgid(os.getpid())
    assert identity.sid == os.getsid(os.getpid())
    assert identity.start_time


def test_uncaptured_group_is_not_released_until_every_member_is_gone(monkeypatch):
    processes = {50, 51}
    signals = []
    waits = []

    class Process:
        pid = 50

        def wait(self, timeout):
            assert timeout > 0
            waits.append(True)
            processes.discard(50)

    def signal_group(pgid, sent_signal):
        signals.append((pgid, sent_signal))

    def sleep(_seconds):
        processes.discard(51)

    monkeypatch.setattr("biomaker.utils.process_tree.os.name", "posix")
    drain_uncaptured_process_group(
        Process(),
        group_reader=lambda _pgid: sorted(processes),
        group_signal_sender=signal_group,
        sleeper=sleep,
        poll_interval=0.001,
    )

    assert signals == [(50, signal.SIGKILL)]
    assert waits == [True]
    assert processes == set()
