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


def _read_proc_identity(pid: int) -> ProcessTreeIdentity:
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
    try:
        observed_pid = int(raw_stat[:opening_parenthesis].strip())
        suffix = raw_stat[closing_parenthesis + 1 :].split()
        pgid = int(suffix[2])
        sid = int(suffix[3])
        start_ticks = int(suffix[19])
    except (IndexError, ValueError) as exc:
        raise ProcessTreeInspectionError(f"malformed /proc/{pid}/stat") from exc
    if observed_pid != pid:
        raise ProcessTreeInspectionError(
            f"process identity mismatch: requested {pid}, observed {observed_pid}"
        )
    return ProcessTreeIdentity(pid, pgid, sid, f"proc:{start_ticks}")


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
) -> None:
    """Fail closed when identity capture fails immediately after ``Popen``.

    The unreaped ``Popen`` object still reserves its PID, and ``process_group=0``
    made that exact PID the new group's ID before exec.  It is therefore safe to
    kill that group once.  The function does not return until the group is empty,
    so callers cannot release a compute slot while an untracked child survives.
    """

    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    pid = getattr(process, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessTreeInspectionError("uncaptured process has no valid pid")

    if os.name != "posix":
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()
        return

    sender = os.killpg if group_signal_sender is None else group_signal_sender
    while True:
        try:
            sender(pid, signal.SIGKILL)
            break
        except ProcessLookupError:
            break
        except OSError:
            sleeper(poll_interval)
    process.wait()
    while True:
        try:
            if not group_reader(pid):
                return
        except ProcessTreeInspectionError:
            pass
        except (OSError, ValueError):
            pass
        sleeper(poll_interval)


class ProcessTreeTracker:
    """Drain one candidate-owned PGID without signalling reused PIDs."""

    def __init__(
        self,
        root: ProcessTreeIdentity,
        *,
        identity_reader: IdentityReader = read_process_tree_identity,
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
        self._group_reader = group_reader
        self._signal_sender = signal_sender
        self._sleeper = sleeper
        self._identities = {root.pid: root}

    @classmethod
    def capture(
        cls,
        root_pid: int,
        *,
        identity_reader: IdentityReader = read_process_tree_identity,
        group_reader: GroupReader = list_process_group_members,
        signal_sender: SignalSender = os.kill,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> ProcessTreeTracker:
        return cls(
            identity_reader(root_pid),
            identity_reader=identity_reader,
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
            observed = self._identity_reader(expected.pid)
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

    def refresh(self) -> int:
        """Capture every current PGID member after verifying its run session."""

        added = 0
        for pid in self._group_reader(self.pgid):
            if pid in self._identities:
                continue
            try:
                observed = self._identity_reader(pid)
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
        current_members = set(self._group_reader(self.pgid))
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

    @staticmethod
    def _poll_root(root_process: Any | None) -> None:
        if root_process is None:
            return
        try:
            root_process.poll()
        except (AttributeError, OSError):
            return

    def _group_is_empty(self) -> bool:
        return not self._group_reader(self.pgid)

    def terminate_and_wait(
        self,
        *,
        root_process: Any | None = None,
        poll_interval: float = 0.05,
    ) -> None:
        """Freeze the PGID, kill exact members, and wait for it to disappear.

        The method has no "give up and release" branch. A known live member must not
        overlap the next owner of the same compute slot.
        """

        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        stable_rounds = 0
        while stable_rounds < 2:
            self._poll_root(root_process)
            try:
                added_before = self.refresh()
                live = self._live_identities()
                if not live and self._group_is_empty():
                    return
                for identity in live:
                    self._signal_if_matching(identity, signal.SIGSTOP)
                added_after = self.refresh()
            except ProcessTreeInspectionError:
                self._sleeper(poll_interval)
                continue
            stable_rounds = (
                stable_rounds + 1 if added_before == 0 and added_after == 0 else 0
            )

        empty_rounds = 0
        while empty_rounds < 2:
            self._poll_root(root_process)
            try:
                self.refresh()
                live = self._live_identities()
                for identity in live:
                    self._signal_if_matching(identity, signal.SIGKILL)
                self._poll_root(root_process)
                empty = not live and self._group_is_empty()
            except ProcessTreeInspectionError:
                empty_rounds = 0
                self._sleeper(poll_interval)
                continue
            empty_rounds = empty_rounds + 1 if empty else 0
            if empty_rounds < 2:
                self._sleeper(poll_interval)
