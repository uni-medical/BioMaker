"""POSIX process identity checks and guarded process-group termination."""

from __future__ import annotations

import math
import os
import platform
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

SnapshotReader: TypeAlias = Callable[[int], "ProcessSnapshot"]
ProcStatReader: TypeAlias = Callable[[int], str]
CommandRunner: TypeAlias = Callable[..., Any]
SignalSender: TypeAlias = Callable[[int, int], None]
GroupProbe: TypeAlias = Callable[[int], bool]
SessionGroupsReader: TypeAlias = Callable[[int], set[int]]


class ProcessIdentityError(RuntimeError):
    """Raised when a process exists but its identity cannot be read safely."""


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    """OS-observed facts that distinguish one PID incarnation from another."""

    pid: int
    pgid: int
    start_time: str

    def __post_init__(self) -> None:
        _validate_pid(self.pid)
        if (
            isinstance(self.pgid, bool)
            or not isinstance(self.pgid, int)
            or self.pgid <= 0
        ):
            raise ValueError("process group id must be positive")
        if not isinstance(self.start_time, str) or not self.start_time:
            raise ValueError("process start time must not be empty")


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Durable launch identity stored alongside a managed run."""

    pid: int
    pgid: int
    start_time: str
    launch_token: str

    def __post_init__(self) -> None:
        _validate_pid(self.pid)
        if (
            isinstance(self.pgid, bool)
            or not isinstance(self.pgid, int)
            or self.pgid <= 0
        ):
            raise ValueError("process group id must be positive")
        if not isinstance(self.start_time, str) or not self.start_time:
            raise ValueError("process start time must not be empty")
        if not isinstance(self.launch_token, str) or not self.launch_token:
            raise ValueError("launch token must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "start_time": self.start_time,
            "launch_token": self.launch_token,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProcessIdentity:
        try:
            pid = value["pid"]
            pgid = value["pgid"]
            start_time = value["start_time"]
            launch_token = value["launch_token"]
        except KeyError as exc:
            raise ValueError(f"missing process identity field: {exc.args[0]}") from exc
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise TypeError("process id must be an integer")
        if isinstance(pgid, bool) or not isinstance(pgid, int):
            raise TypeError("process group id must be an integer")
        if not isinstance(start_time, str):
            raise TypeError("process start time must be a string")
        if not isinstance(launch_token, str):
            raise TypeError("launch token must be a string")
        return cls(pid, pgid, start_time, launch_token)


VerificationReason: TypeAlias = Literal[
    "match",
    "missing",
    "launch_token_mismatch",
    "pid_mismatch",
    "process_group_mismatch",
    "start_time_mismatch",
    "unreadable",
]


@dataclass(frozen=True, slots=True)
class IdentityVerification:
    matched: bool
    reason: VerificationReason
    observed: ProcessSnapshot | None = None


TerminationOutcome: TypeAlias = Literal[
    "already_exited",
    "terminated",
    "killed",
    "rejected",
    "signal_error",
]


@dataclass(frozen=True, slots=True)
class TerminationResult:
    outcome: TerminationOutcome
    reason: str
    signals_sent: tuple[int, ...] = ()
    verification: IdentityVerification | None = None

    @property
    def accepted(self) -> bool:
        return self.outcome not in {"rejected", "signal_error"}


def _validate_pid(pid: int) -> None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process id must be a positive integer")


def _default_proc_stat_reader(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")


def _snapshot_from_proc_stat(pid: int, raw_stat: str) -> ProcessSnapshot:
    """Parse pgrp (field 5) and starttime ticks (field 22) from proc stat."""

    closing_parenthesis = raw_stat.rfind(")")
    opening_parenthesis = raw_stat.find("(")
    if opening_parenthesis <= 0 or closing_parenthesis <= opening_parenthesis:
        raise ProcessIdentityError(f"malformed /proc/{pid}/stat")
    try:
        observed_pid = int(raw_stat[:opening_parenthesis].strip())
    except ValueError as exc:
        raise ProcessIdentityError(f"malformed /proc/{pid}/stat pid") from exc
    if observed_pid != pid:
        raise ProcessIdentityError(
            f"/proc stat pid mismatch: requested {pid}, observed {observed_pid}"
        )

    # The suffix starts at field 3 (state); pgrp is suffix[2] and starttime is
    # suffix[19]. Parsing after the final ')' handles spaces and ')' in comm.
    suffix = raw_stat[closing_parenthesis + 1 :].split()
    if len(suffix) <= 19:
        raise ProcessIdentityError(f"malformed /proc/{pid}/stat fields")
    try:
        pgid = int(suffix[2])
        start_ticks = int(suffix[19])
    except ValueError as exc:
        raise ProcessIdentityError(f"malformed /proc/{pid}/stat values") from exc
    return ProcessSnapshot(pid=pid, pgid=pgid, start_time=f"proc:{start_ticks}")


def _snapshot_from_ps(
    pid: int,
    *,
    command_runner: CommandRunner,
    create_time_reader: Callable[[int], float] | None = None,
) -> ProcessSnapshot:
    if create_time_reader is None:
        raise ProcessIdentityError(
            f"precise process creation time is unavailable for process {pid}"
        )

    def read_create_time() -> float:
        try:
            value = float(create_time_reader(pid))
        except ProcessLookupError:
            raise
        except Exception as exc:
            raise ProcessIdentityError(
                f"could not read precise creation time for process {pid}"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ProcessIdentityError(
                f"invalid precise creation time for process {pid}"
            )
        return value

    # ``ps`` does not provide a sub-second creation timestamp on Darwin/BSD.
    # Bracket it with exact psutil create-time reads so PID reuse during the
    # snapshot cannot produce an executable identity for the wrong process.
    start_before = read_create_time()
    try:
        result = command_runner(
            [
                "ps",
                "-o",
                "pid=",
                "-o",
                "pgid=",
                "-p",
                str(pid),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise ProcessIdentityError(f"could not inspect process {pid} with ps") from exc

    stdout = getattr(result, "stdout", "")
    if getattr(result, "returncode", 1) != 0 or not isinstance(stdout, str):
        raise ProcessLookupError(pid)
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ProcessLookupError(pid)
    fields = lines[0].split()
    if len(fields) != 2:
        raise ProcessIdentityError(f"malformed ps identity output for process {pid}")
    try:
        observed_pid = int(fields[0])
        pgid = int(fields[1])
    except ValueError as exc:
        raise ProcessIdentityError(
            f"malformed ps identity values for process {pid}"
        ) from exc
    if observed_pid != pid:
        raise ProcessIdentityError(
            f"ps pid mismatch: requested {pid}, observed {observed_pid}"
        )
    start_after = read_create_time()
    if start_before != start_after:
        raise ProcessIdentityError(
            f"process {pid} changed identity while reading its snapshot"
        )
    return ProcessSnapshot(
        pid=pid,
        pgid=pgid,
        start_time=f"ps:precise:{start_before!r}",
    )


def _psutil_create_time(pid: int) -> float:
    import psutil  # type: ignore[import-not-found]

    try:
        return float(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess as exc:
        raise ProcessLookupError(pid) from exc
    except psutil.AccessDenied as exc:
        raise ProcessIdentityError(
            f"permission denied while reading creation time for process {pid}"
        ) from exc
    except psutil.Error as exc:
        raise ProcessIdentityError(
            f"could not read creation time for process {pid}"
        ) from exc


def read_process_snapshot(
    pid: int,
    *,
    platform_name: str | None = None,
    proc_stat_reader: ProcStatReader | None = None,
    ps_runner: CommandRunner | None = None,
    create_time_reader: Callable[[int], float] | None = None,
) -> ProcessSnapshot:
    """Read one process identity from Linux procfs or the portable ``ps`` fallback."""

    _validate_pid(pid)
    system = platform.system() if platform_name is None else platform_name
    if system == "Linux":
        reader = (
            _default_proc_stat_reader if proc_stat_reader is None else proc_stat_reader
        )
        try:
            raw_stat = reader(pid)
        except (FileNotFoundError, ProcessLookupError):
            raise ProcessLookupError(pid) from None
        except OSError as exc:
            raise ProcessIdentityError(f"could not read /proc/{pid}/stat") from exc
        if not isinstance(raw_stat, str):
            raise ProcessIdentityError(
                f"/proc/{pid}/stat reader returned non-text data"
            )
        return _snapshot_from_proc_stat(pid, raw_stat)

    return _snapshot_from_ps(
        pid,
        command_runner=subprocess.run if ps_runner is None else ps_runner,
        create_time_reader=(
            _psutil_create_time if create_time_reader is None else create_time_reader
        ),
    )


def capture_process_identity(
    pid: int,
    launch_token: str,
    *,
    snapshot_reader: SnapshotReader | None = None,
) -> ProcessIdentity:
    """Capture a durable identity immediately after launching a managed process."""

    if not isinstance(launch_token, str) or not launch_token:
        raise ValueError("launch token must not be empty")
    reader = read_process_snapshot if snapshot_reader is None else snapshot_reader
    observed = reader(pid)
    if observed.pid != pid:
        raise ProcessIdentityError(
            f"snapshot pid mismatch: requested {pid}, observed {observed.pid}"
        )
    return ProcessIdentity(
        pid=observed.pid,
        pgid=observed.pgid,
        start_time=observed.start_time,
        launch_token=launch_token,
    )


def verify_process_identity(
    identity: ProcessIdentity,
    *,
    launch_token: str,
    snapshot_reader: SnapshotReader | None = None,
) -> IdentityVerification:
    """Compare a durable identity with both its caller token and current OS facts."""

    if launch_token != identity.launch_token:
        return IdentityVerification(False, "launch_token_mismatch")
    reader = read_process_snapshot if snapshot_reader is None else snapshot_reader
    try:
        observed = reader(identity.pid)
    except ProcessLookupError:
        return IdentityVerification(False, "missing")
    except (OSError, ProcessIdentityError, ValueError):
        return IdentityVerification(False, "unreadable")
    if observed.pid != identity.pid:
        return IdentityVerification(False, "pid_mismatch", observed)
    if observed.pgid != identity.pgid:
        return IdentityVerification(False, "process_group_mismatch", observed)
    if observed.start_time != identity.start_time:
        return IdentityVerification(False, "start_time_mismatch", observed)
    return IdentityVerification(True, "match", observed)


def is_process_alive(
    identity: ProcessIdentity,
    *,
    launch_token: str,
    snapshot_reader: SnapshotReader | None = None,
) -> bool:
    """Return true only while the exact captured PID incarnation is still alive."""

    return verify_process_identity(
        identity,
        launch_token=launch_token,
        snapshot_reader=snapshot_reader,
    ).matched


def _default_group_probe(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    identity: ProcessIdentity,
    *,
    launch_token: str,
    grace_seconds: float = 5.0,
    poll_interval: float = 0.1,
    snapshot_reader: SnapshotReader | None = None,
    group_probe: GroupProbe | None = None,
    signal_sender: SignalSender | None = None,
    current_pgid: int | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> TerminationResult:
    """Terminate an exactly verified process group, escalating after a bounded wait."""

    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")

    verification = verify_process_identity(
        identity,
        launch_token=launch_token,
        snapshot_reader=snapshot_reader,
    )
    if verification.reason == "missing":
        return TerminationResult(
            "already_exited", "process is no longer present", verification=verification
        )
    if not verification.matched:
        return TerminationResult(
            "rejected",
            f"process identity rejected: {verification.reason}",
            verification=verification,
        )

    own_group = os.getpgrp() if current_pgid is None else current_pgid
    if identity.pgid <= 1 or identity.pgid == own_group:
        return TerminationResult(
            "rejected",
            "refusing to signal an unsafe process group",
            verification=verification,
        )

    probe = _default_group_probe if group_probe is None else group_probe
    sender = os.killpg if signal_sender is None else signal_sender
    sent: list[int] = []
    try:
        sender(identity.pgid, signal.SIGTERM)
        sent.append(signal.SIGTERM)
    except ProcessLookupError:
        return TerminationResult(
            "already_exited",
            "process group exited before SIGTERM",
            tuple(sent),
            verification,
        )
    except OSError as exc:
        return TerminationResult(
            "signal_error", f"SIGTERM failed: {exc}", tuple(sent), verification
        )

    deadline = monotonic() + grace_seconds
    while probe(identity.pgid):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleeper(min(poll_interval, remaining))
    else:
        return TerminationResult(
            "terminated",
            "process group exited after SIGTERM",
            tuple(sent),
            verification,
        )

    # Recheck a still-present leader before escalation. A missing leader is expected
    # when descendants remain in its old group; a reused leader PID is never killed.
    escalation_check = verify_process_identity(
        identity,
        launch_token=launch_token,
        snapshot_reader=snapshot_reader,
    )
    if escalation_check.reason not in {"match", "missing"}:
        return TerminationResult(
            "rejected",
            f"process identity changed before SIGKILL: {escalation_check.reason}",
            tuple(sent),
            escalation_check,
        )

    try:
        sender(identity.pgid, signal.SIGKILL)
        sent.append(signal.SIGKILL)
    except ProcessLookupError:
        return TerminationResult(
            "terminated",
            "process group exited before SIGKILL",
            tuple(sent),
            escalation_check,
        )
    except OSError as exc:
        return TerminationResult(
            "signal_error", f"SIGKILL failed: {exc}", tuple(sent), escalation_check
        )
    return TerminationResult(
        "killed", "process group required SIGKILL", tuple(sent), escalation_check
    )


def read_session_process_groups(
    session_id: int,
    *,
    command_runner: CommandRunner | None = None,
    sid_reader: Callable[[int], int] = os.getsid,
    pgid_reader: Callable[[int], int] = os.getpgid,
) -> set[int]:
    """Return every live process group in one POSIX session."""

    _validate_pid(session_id)
    runner = subprocess.run if command_runner is None else command_runner
    try:
        result = runner(
            ["ps", "-axo", "pid=,stat="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise ProcessIdentityError("could not enumerate session processes") from exc
    stdout = getattr(result, "stdout", "")
    if getattr(result, "returncode", 1) != 0 or not isinstance(stdout, str):
        raise ProcessIdentityError("could not enumerate session processes")

    groups: set[int] = set()
    for raw_process in stdout.splitlines():
        try:
            fields = raw_process.split(maxsplit=1)
            pid = int(fields[0])
            # A killed orphan can remain visible as a zombie until its new
            # parent reaps it. Zombies hold no executable resources and cannot
            # be removed by sending another signal, so they must not keep Stop
            # stuck in its TERM/KILL loop. A live member of the same group still
            # contributes that PGID on its own row.
            if len(fields) > 1 and fields[1].lstrip().startswith("Z"):
                continue
            if pid <= 0 or sid_reader(pid) != session_id:
                continue
            pgid = pgid_reader(pid)
        except (ValueError, OSError, ProcessLookupError):
            continue
        if pgid > 1:
            groups.add(pgid)
    return groups


def terminate_process_session(
    identity: ProcessIdentity,
    *,
    launch_token: str,
    grace_seconds: float = 5.0,
    poll_interval: float = 0.1,
    snapshot_reader: SnapshotReader | None = None,
    session_groups_reader: SessionGroupsReader | None = None,
    signal_sender: SignalSender | None = None,
    sid_reader: Callable[[int], int] = os.getsid,
    current_sid: int | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> TerminationResult:
    """Terminate all process groups in an exactly verified managed run session."""

    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")

    verification = verify_process_identity(
        identity,
        launch_token=launch_token,
        snapshot_reader=snapshot_reader,
    )
    if verification.reason not in {"match", "missing"}:
        return TerminationResult(
            "rejected",
            f"process identity rejected: {verification.reason}",
            verification=verification,
        )

    own_sid = sid_reader(0) if current_sid is None else current_sid
    if identity.pid <= 1 or identity.pgid != identity.pid or identity.pid == own_sid:
        return TerminationResult(
            "rejected",
            "refusing to signal an unsafe or non-leader process session",
            verification=verification,
        )
    leader_missing = verification.reason == "missing"
    if verification.matched:
        try:
            observed_sid = sid_reader(identity.pid)
        except (OSError, ProcessLookupError):
            verification = verify_process_identity(
                identity,
                launch_token=launch_token,
                snapshot_reader=snapshot_reader,
            )
            if verification.reason != "missing":
                return TerminationResult(
                    "rejected",
                    "session leader identity changed before cleanup",
                    verification=verification,
                )
            leader_missing = True
        else:
            if observed_sid != identity.pid:
                return TerminationResult(
                    "rejected",
                    "refusing to signal an unsafe or non-leader process session",
                    verification=verification,
                )

    groups_reader = (
        read_session_process_groups
        if session_groups_reader is None
        else session_groups_reader
    )
    sender = os.killpg if signal_sender is None else signal_sender

    class LeaderIdentityChanged(ProcessIdentityError):
        def __init__(self, observed: IdentityVerification) -> None:
            super().__init__("session leader identity changed while stopping the run")
            self.observed = observed

    def verify_leader() -> IdentityVerification:
        nonlocal leader_missing
        observed = verify_process_identity(
            identity,
            launch_token=launch_token,
            snapshot_reader=snapshot_reader,
        )
        if observed.reason not in {"match", "missing"}:
            raise LeaderIdentityChanged(observed)
        # Once the original leader is gone, a process appearing at the same PID
        # is a reuse candidate even if its observable fields happen to match.
        if leader_missing and observed.reason != "missing":
            raise LeaderIdentityChanged(observed)
        if observed.reason == "missing":
            leader_missing = True
        return observed

    def groups() -> set[int]:
        verify_leader()
        observed = set(groups_reader(identity.pid))
        # Enumeration is not atomic with PID/SID allocation. Verify again before
        # any signal so a reused session ID cannot redirect cleanup.
        verify_leader()
        if any(group <= 1 or group == os.getpgrp() for group in observed):
            raise ProcessIdentityError(
                "managed session overlaps an unsafe process group"
            )
        return observed

    def send_all(observed: set[int], sent_signal: int) -> None:
        for pgid in sorted(observed, key=lambda value: value == identity.pgid):
            try:
                sender(pgid, sent_signal)
            except ProcessLookupError:
                continue

    sent: list[int] = []
    try:
        active_groups = groups()
        if not active_groups:
            return TerminationResult(
                "already_exited",
                "managed session has no remaining process groups",
                verification=verification,
            )
        send_all(active_groups, signal.SIGTERM)
        sent.append(signal.SIGTERM)
    except LeaderIdentityChanged as exc:
        return TerminationResult(
            "rejected",
            str(exc),
            tuple(sent),
            exc.observed,
        )
    except (OSError, ProcessIdentityError) as exc:
        return TerminationResult(
            "signal_error", f"session SIGTERM failed: {exc}", tuple(sent), verification
        )

    deadline = monotonic() + grace_seconds
    while True:
        try:
            active_groups = groups()
        except LeaderIdentityChanged as exc:
            return TerminationResult(
                "rejected",
                str(exc),
                tuple(sent),
                exc.observed,
            )
        except (OSError, ProcessIdentityError) as exc:
            return TerminationResult(
                "signal_error",
                f"session inspection failed: {exc}",
                tuple(sent),
                verification,
            )
        if not active_groups:
            return TerminationResult(
                "terminated",
                "managed session exited after SIGTERM",
                tuple(sent),
                verification,
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleeper(min(poll_interval, remaining))

    try:
        escalation_check = verify_leader()
    except LeaderIdentityChanged as exc:
        return TerminationResult(
            "rejected",
            f"process identity changed before SIGKILL: {exc.observed.reason}",
            tuple(sent),
            exc.observed,
        )
    deadline = monotonic() + max(grace_seconds, 1.0)
    sent_kill = False
    while True:
        try:
            active_groups = groups()
        except LeaderIdentityChanged as exc:
            return TerminationResult(
                "rejected",
                str(exc),
                tuple(sent),
                exc.observed,
            )
        except (OSError, ProcessIdentityError) as exc:
            return TerminationResult(
                "signal_error",
                f"session inspection failed: {exc}",
                tuple(sent),
                escalation_check,
            )
        if not active_groups:
            return TerminationResult(
                "killed",
                "managed session required SIGKILL",
                tuple(sent),
                escalation_check,
            )
        try:
            # A process in the managed session can create a new process group
            # while TERM is draining. Re-enumerate and KILL every observed group
            # until the session is empty instead of signalling one stale snapshot.
            send_all(active_groups, signal.SIGKILL)
            if not sent_kill:
                sent.append(signal.SIGKILL)
                sent_kill = True
        except OSError as exc:
            return TerminationResult(
                "signal_error",
                f"session SIGKILL failed: {exc}",
                tuple(sent),
                escalation_check,
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            return TerminationResult(
                "signal_error",
                "managed session still has processes after SIGKILL",
                tuple(sent),
                escalation_check,
            )
        sleeper(min(poll_interval, remaining))
