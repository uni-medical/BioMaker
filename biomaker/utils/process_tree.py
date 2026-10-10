"""Identity-checked cleanup for one isolated candidate process group.

Candidates stay in the top-level BioMaker session but receive their own process group.
That group remains enumerable after its leader exits, unlike a parent/child PID chain.
"""

from __future__ import annotations

import math
import os
import platform
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


class ProcessTreeInspectionError(RuntimeError):
    """Raised when a process group cannot be inspected safely."""


class ProcessTreeCleanupError(BaseException):
    """Fatal ownership failure; ordinary candidate retries must not swallow it.

    The caller must retain/quarantine its compute lease. A managed desktop run
    then relies on its outer session owner to drain or persist cleanup_pending.
    This is deliberately outside Exception, like a cancellation of the run.
    """

    def __init__(self, pgid: int, pids: tuple[int, ...], reason: str):
        self.pgid, self.pids, self.reason = pgid, pids, reason
        super().__init__(
            f"Process cleanup remains unverified for PGID {pgid}, "
            f"known PIDs {list(pids)}: {reason}. The compute slot must not be reused."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cleanup_unverified": True,
            "pgid": self.pgid,
            "known_pids": list(self.pids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProcessTreeIdentity:
    """Facts that distinguish one PID incarnation inside a run session."""

    pid: int
    pgid: int
    sid: int
    start_time: str

    def __post_init__(self) -> None:
        for name, value in (("pid", self.pid), ("pgid", self.pgid), ("sid", self.sid)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.start_time, str) or not self.start_time:
            raise ValueError("start_time must not be empty")


IdentityReader: TypeAlias = Callable[[int], ProcessTreeIdentity]
GroupReader: TypeAlias = Callable[[int], list[int]]
SignalSender: TypeAlias = Callable[[int, int], None]
CommandRunner: TypeAlias = Callable[..., Any]
CreateTimeReader: TypeAlias = Callable[[int], float]
SessionReader: TypeAlias = Callable[[int], int]
StateReader: TypeAlias = Callable[[int], str]


def _read_proc_stat_fields(pid: int) -> list[str]:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        raise ProcessLookupError(pid) from None
    except OSError as exc:
        raise ProcessTreeInspectionError(f"could not read /proc/{pid}/stat") from exc

    opening_parenthesis = raw_stat.find("(")
    closing_parenthesis = raw_stat.rfind(")")
    if opening_parenthesis <= 0 or closing_parenthesis <= opening_parenthesis:
        raise ProcessTreeInspectionError(f"malformed /proc/{pid}/stat")
    suffix = raw_stat[closing_parenthesis + 1 :].split()
    try:
        observed_pid = int(raw_stat[:opening_parenthesis].strip())
    except ValueError as exc:
        raise ProcessTreeInspectionError(f"malformed /proc/{pid}/stat") from exc
    if observed_pid != pid or not suffix:
        raise ProcessTreeInspectionError(f"malformed /proc/{pid}/stat")
    return suffix


def _read_proc_identity(pid: int) -> ProcessTreeIdentity:
    suffix = _read_proc_stat_fields(pid)
    try:
        pgid = int(suffix[2])
        sid = int(suffix[3])
        start_ticks = int(suffix[19])
    except (IndexError, ValueError) as exc:
        raise ProcessTreeInspectionError(f"malformed /proc/{pid}/stat") from exc
    return ProcessTreeIdentity(pid, pgid, sid, f"proc:{start_ticks}")


def _read_proc_state(pid: int) -> str:
    return _read_proc_stat_fields(pid)[0]


def _psutil_create_time(pid: int) -> float:
    import psutil

    try:
        return float(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        raise ProcessLookupError(pid) from None
    except (psutil.AccessDenied, psutil.Error, OSError, ValueError) as exc:
        raise ProcessTreeInspectionError(
            f"could not read precise start time for process {pid}"
        ) from exc


def _psutil_process_state(pid: int) -> str:
    import psutil

    try:
        state = str(psutil.Process(pid).status())
    except psutil.NoSuchProcess:
        raise ProcessLookupError(pid) from None
    except (psutil.AccessDenied, psutil.Error, OSError, ValueError) as exc:
        raise ProcessTreeInspectionError(
            f"could not read state for process {pid}"
        ) from exc
    if not state:
        raise ProcessTreeInspectionError(f"invalid state for process {pid}")
    return state


def _read_ps_identity(
    pid: int,
    *,
    command_runner: CommandRunner = subprocess.run,
    create_time_reader: CreateTimeReader = _psutil_create_time,
    session_reader: SessionReader | None = None,
) -> ProcessTreeIdentity:
    """Read a portable identity while detecting PID reuse across ``ps``.

    BSD ``ps lstart`` is only precise to one second.  The precise creation time is
    therefore read on both sides of the ``ps`` query; a changed value makes the
    compound observation unsafe rather than silently accepting a mixed identity.
    """

    try:
        start_before = float(create_time_reader(pid))
    except ProcessLookupError:
        raise
    except ProcessTreeInspectionError:
        raise
    except (OSError, ValueError) as exc:
        raise ProcessTreeInspectionError(
            f"could not read precise start time for process {pid}"
        ) from exc
    if not math.isfinite(start_before) or start_before <= 0:
        raise ProcessTreeInspectionError(
            f"invalid precise start time for process {pid}"
        )
    try:
        completed = command_runner(
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
        raise ProcessTreeInspectionError(f"could not inspect process {pid}") from exc
    stdout = getattr(completed, "stdout", "")
    if not isinstance(stdout, str):
        raise ProcessTreeInspectionError(f"non-text ps output for process {pid}")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        raise ProcessLookupError(pid)
    fields = lines[0].split()
    if len(fields) != 2:
        raise ProcessTreeInspectionError(f"malformed ps output for process {pid}")
    try:
        observed_pid, pgid = map(int, fields)
    except ValueError as exc:
        raise ProcessTreeInspectionError(
            f"malformed ps output for process {pid}"
        ) from exc
    if observed_pid != pid:
        raise ProcessTreeInspectionError(
            f"process identity mismatch: requested {pid}, observed {observed_pid}"
        )
    read_session = (
        getattr(os, "getsid", None) if session_reader is None else session_reader
    )
    if read_session is None:
        raise ProcessTreeInspectionError("process sessions are unavailable")
    try:
        sid = int(read_session(pid))
    except ProcessLookupError:
        raise
    except (OSError, ValueError) as exc:
        raise ProcessTreeInspectionError(
            f"could not read session for process {pid}"
        ) from exc
    try:
        start_after = float(create_time_reader(pid))
    except ProcessLookupError:
        raise
    except ProcessTreeInspectionError:
        raise
    except (OSError, ValueError) as exc:
        raise ProcessTreeInspectionError(
            f"could not verify precise start time for process {pid}"
        ) from exc
    if start_after != start_before:
        raise ProcessTreeInspectionError(
            f"process {pid} changed identity while it was inspected"
        )
    return ProcessTreeIdentity(pid, pgid, sid, f"create:{start_before.hex()}")


def read_process_tree_identity(pid: int) -> ProcessTreeIdentity:
    """Read process group, session, and stable start time for one PID."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process id must be a positive integer")
    if platform.system() == "Linux":
        return _read_proc_identity(pid)
    return _read_ps_identity(pid)


def read_process_tree_state(pid: int) -> str:
    """Read process state without waiting for or reaping the process."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process id must be a positive integer")
    if platform.system() == "Linux":
        return _read_proc_state(pid)
    return _psutil_process_state(pid)


def list_process_group_members(pgid: int) -> list[int]:
    """Return the current members of one POSIX process group."""

    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,pgid=,stat="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise ProcessTreeInspectionError(
            f"could not list process group {pgid}"
        ) from exc
    if completed.returncode != 0:
        raise ProcessTreeInspectionError(f"could not list process group {pgid}")
    members: list[int] = []
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid, observed_pgid = map(int, fields[:2])
        except ValueError:
            continue
        # A zombie holds no executable resources and cannot be removed with
        # another signal. Do not let an unreaped zombie keep a compute slot
        # leased forever; a live member contributes the same PGID on its own row.
        if observed_pgid == pgid and not fields[2].lstrip().startswith("Z"):
            members.append(pid)
    return members


def drain_uncaptured_process_group(
    process: Any,
    *,
    group_reader: GroupReader = list_process_group_members,
    group_signal_sender: SignalSender | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.05,
    timeout: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Drain a newly launched PGID before reaping its identity-owning leader.

    A bounded inspection failure is fatal: the caller retains its compute lease.
    Never poll/wait first, because that would allow the PGID to be reused.
    """
    if poll_interval <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("poll_interval and timeout must be positive")
    pid = getattr(process, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessTreeInspectionError("uncaptured process has no valid pid")
    if getattr(process, "returncode", None) is not None:
        raise ProcessTreeCleanupError(pid, (pid,), "uncaptured leader already reaped")
    deadline = monotonic() + timeout
    reason = "process group has not been confirmed empty"

    def remaining() -> float:
        value = deadline - monotonic()
        if value <= 0:
            raise ProcessTreeCleanupError(pid, (pid,), reason)
        return value

    if os.name == "posix":
        sender = os.killpg if group_signal_sender is None else group_signal_sender
        empty_rounds = 0
        while empty_rounds < 2:
            remaining()
            try:
                sender(pid, signal.SIGKILL)
                empty = not group_reader(pid)
            except (ProcessLookupError, PermissionError) as exc:
                # macOS can return EPERM for a group containing only the
                # unreaped zombie leader. A failed signal is never evidence of
                # emptiness: require the same independent live-member checks.
                if isinstance(exc, PermissionError):
                    reason = f"{type(exc).__name__}: {exc}"
                try:
                    empty = not group_reader(pid)
                except (ProcessTreeInspectionError, OSError, ValueError) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    empty = False
            except (ProcessTreeInspectionError, OSError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                empty = False
            empty_rounds = empty_rounds + 1 if empty else 0
            if empty_rounds < 2:
                sleeper(min(poll_interval, remaining()))
    else:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=remaining())
    except (subprocess.TimeoutExpired, OSError, ChildProcessError) as exc:
        raise ProcessTreeCleanupError(
            pid, (pid,), f"uncaptured leader did not exit after group kill: {type(exc).__name__}"
        ) from None


class ProcessTreeTracker:
    """Drain one candidate-owned PGID without signalling reused PIDs."""

    def __init__(
        self,
        root: ProcessTreeIdentity,
        *,
        identity_reader: IdentityReader = read_process_tree_identity,
        state_reader: StateReader = read_process_tree_state,
        create_time_reader: CreateTimeReader = _psutil_create_time,
        group_reader: GroupReader = list_process_group_members,
        signal_sender: SignalSender = os.kill,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if root.pgid != root.pid:
            raise ValueError("managed process must lead its own process group")
        self.root_pid = root.pid
        self.pgid = root.pgid
        self.sid = root.sid
        self._identity_reader = identity_reader
        self._state_reader = state_reader
        self._create_time_reader = create_time_reader
        self._group_reader = group_reader
        self._signal_sender = signal_sender
        self._sleeper = sleeper
        self._identities = {root.pid: root}
        self._drained = False
        self._cleanup_complete = False
        self._cleanup_failure: ProcessTreeCleanupError | None = None
        self._cleanup_budget_check: Callable[[], None] | None = None

    @classmethod
    def capture(
        cls,
        root_pid: int,
        *,
        identity_reader: IdentityReader = read_process_tree_identity,
        state_reader: StateReader = read_process_tree_state,
        create_time_reader: CreateTimeReader = _psutil_create_time,
        group_reader: GroupReader = list_process_group_members,
        signal_sender: SignalSender = os.kill,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> ProcessTreeTracker:
        return cls(
            identity_reader(root_pid),
            identity_reader=identity_reader,
            state_reader=state_reader,
            create_time_reader=create_time_reader,
            group_reader=group_reader,
            signal_sender=signal_sender,
            sleeper=sleeper,
        )

    @property
    def tracked_pids(self) -> tuple[int, ...]:
        return tuple(sorted(self._identities))

    def _matching_identity(
        self, expected: ProcessTreeIdentity
    ) -> ProcessTreeIdentity | None:
        try:
            observed = self._read_identity(expected.pid)
        except ProcessLookupError:
            return None
        except ProcessTreeInspectionError:
            raise
        except (OSError, ValueError) as exc:
            raise ProcessTreeInspectionError(
                f"could not verify process {expected.pid}"
            ) from exc
        if (
            observed.pid != expected.pid
            or observed.pgid != expected.pgid
            or observed.sid != expected.sid
            or observed.start_time != expected.start_time
        ):
            return None
        return observed

    def _read_identity(self, pid: int) -> ProcessTreeIdentity:
        if self._cleanup_budget_check is not None:
            self._cleanup_budget_check()
        return self._identity_reader(pid)

    def _read_group(self) -> list[int]:
        if self._cleanup_budget_check is not None:
            self._cleanup_budget_check()
        return self._group_reader(self.pgid)

    def refresh(self) -> int:
        """Capture every current PGID member after verifying its run session."""

        if self._cleanup_failure is not None:
            raise self._cleanup_failure
        if self._cleanup_complete:
            return 0
        members = self._read_group()
        if not members:
            return 0
        # A newly allocated group in the same session is not our descendant.
        root = self._identities[self.root_pid]
        try:
            observed_root = self._read_identity(self.root_pid)
        except ProcessLookupError:
            observed_root = None
        if observed_root is not None and observed_root != root:
            # Ownership cannot become trustworthy again if this new leader
            # subsequently exits and leaves children in its recycled group.
            self._cleanup_failure = ProcessTreeCleanupError(
                self.pgid, self.tracked_pids,
                "original process-group leader PID was reused",
            )
            raise self._cleanup_failure
        added = 0
        for pid in members:
            if pid in self._identities:
                continue
            try:
                observed = self._read_identity(pid)
            except ProcessLookupError:
                continue
            except ProcessTreeInspectionError:
                raise
            except (OSError, ValueError) as exc:
                raise ProcessTreeInspectionError(
                    f"could not capture process-group member {pid}"
                ) from exc
            if observed.pgid != self.pgid or observed.sid != self.sid:
                continue
            self._identities[pid] = observed
            added += 1
        return added

    def refresh_best_effort(self) -> bool:
        try:
            self.refresh()
        except ProcessTreeInspectionError:
            return False
        return True

    def _live_identities(self) -> list[ProcessTreeIdentity]:
        current_members = set(self._read_group())
        return [
            identity
            for identity in self._identities.values()
            if identity.pid in current_members
            and self._matching_identity(identity) is not None
        ]

    def _signal_if_matching(self, identity: ProcessTreeIdentity, sig: int) -> bool:
        if self._matching_identity(identity) is None:
            return False
        try:
            self._signal_sender(identity.pid, sig)
        except ProcessLookupError:
            return False
        except OSError as exc:
            raise ProcessTreeInspectionError(
                f"could not signal process {identity.pid}"
            ) from exc
        return True

    def root_has_exited(self, root_process: Any) -> bool:
        """Observe child exit without releasing its PID or process-group ID.

        ``Popen.poll()`` calls ``waitpid(WNOHANG)`` and therefore reaps an exited
        leader.  Its PID can then be reused as a process-group ID before the
        descendants that owned the old group have been drained.  Reading the
        kernel/psutil process state reports the same zombie fact without waiting.
        """

        pid = getattr(root_process, "pid", None)
        if pid != self.root_pid:
            raise ProcessTreeInspectionError("root process identity mismatch")
        if self._drained:
            return True
        if getattr(root_process, "returncode", None) is not None:
            raise ProcessTreeInspectionError(
                "root process was reaped before its process group was drained"
            )
        if os.name != "posix":
            return root_process.poll() is not None
        try:
            expected = self._identities[self.root_pid]
            if expected.start_time.startswith("create:"):
                start_before = self._create_time_reader(self.root_pid)
                state = self._state_reader(self.root_pid)
                start_after = self._create_time_reader(self.root_pid)
                expected_start = f"create:{start_before.hex()}"
                if expected_start != expected.start_time or start_after != start_before:
                    raise ProcessTreeInspectionError(
                        f"root process {self.root_pid} changed identity before reaping"
                    )
            else:
                if self._matching_identity(expected) is None:
                    raise ProcessTreeInspectionError(
                        f"root process {self.root_pid} changed identity before reaping"
                    )
                state = self._state_reader(self.root_pid)
        except ProcessTreeInspectionError:
            raise
        except ProcessLookupError as exc:
            raise ProcessTreeInspectionError(
                f"root process {self.root_pid} vanished before it could be reaped"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ProcessTreeInspectionError(
                f"could not inspect child process {self.root_pid} without reaping it"
            ) from exc
        return state.lower() in {"z", "x", "zombie", "dead"}

    def _group_is_empty(self) -> bool:
        return not self._read_group()

    def terminate_and_wait(
        self,
        *,
        root_process: Any | None = None,
        poll_interval: float = 0.05,
        timeout: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Freeze the PGID, kill exact members, and wait for it to disappear.

        A deadline aborts the run with unverified ownership; it never declares
        cleanup successful or permits the compute slot to be reused. Terminal
        results are cached so a caller's finally cannot restart cleanup.
        """

        if self._cleanup_failure is not None:
            raise self._cleanup_failure
        if self._cleanup_complete:
            return
        if poll_interval <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("poll_interval and timeout must be positive")
        if root_process is not None and getattr(root_process, "pid", None) != self.root_pid:
            raise ProcessTreeInspectionError("root process identity mismatch")
        deadline = monotonic() + timeout
        last_error = "process group has not been confirmed empty"

        def check_deadline() -> None:
            if monotonic() >= deadline:
                self._cleanup_failure = ProcessTreeCleanupError(
                    self.pgid, self.tracked_pids, last_error
                )
                raise self._cleanup_failure

        self._cleanup_budget_check = check_deadline

        stable_rounds = 0
        while stable_rounds < 2:
            check_deadline()
            try:
                added_before = self.refresh()
                live = self._live_identities()
                for identity in live:
                    check_deadline()
                    self._signal_if_matching(identity, signal.SIGSTOP)
                added_after = self.refresh()
            except (ProcessTreeInspectionError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleeper(poll_interval)
                continue
            stable_rounds = (
                stable_rounds + 1 if added_before == 0 and added_after == 0 else 0
            )

        empty_rounds = 0
        while empty_rounds < 2:
            check_deadline()
            try:
                self.refresh()
                live = self._live_identities()
                for identity in live:
                    check_deadline()
                    self._signal_if_matching(identity, signal.SIGKILL)
                empty = not live and self._group_is_empty()
            except (ProcessTreeInspectionError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                empty_rounds = 0
                self._sleeper(poll_interval)
                continue
            empty_rounds = empty_rounds + 1 if empty else 0
            if empty_rounds < 2:
                self._sleeper(poll_interval)
        if root_process is not None:
            try:
                root_process.wait(timeout=max(0.001, deadline - monotonic()))
            except (subprocess.TimeoutExpired, OSError, ChildProcessError) as exc:
                self._cleanup_failure = ProcessTreeCleanupError(
                    self.pgid, self.tracked_pids, f"could not reap drained leader: {type(exc).__name__}"
                )
                raise self._cleanup_failure from None
        self._drained = True
        self._cleanup_complete = True
