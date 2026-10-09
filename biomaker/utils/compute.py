"""Small, process-local compute slot discovery and allocation primitives.

The module deliberately does not import an accelerator framework at import time.  CUDA
discovery uses the process environment first and ``nvidia-smi`` second; the optional
MPS probe is only evaluated when no CUDA slot is visible.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .process_tree import ProcessTreeCleanupError

ComputeBackend: TypeAlias = Literal["cuda", "mps", "cpu"]
WorkerRequest: TypeAlias = Literal["auto"] | int
CommandRunner: TypeAlias = Callable[..., Any]

_CUDA_HIDDEN_VALUES = {"", "-1", "none", "nodevfiles"}
_CUDA_SELECTOR_RE = re.compile(
    r"(?:\d+|GPU-[0-9A-Za-z-]+|MIG-(?:GPU-)?[0-9A-Za-z-]+(?:/\d+/\d+)?)"
)


@dataclass(frozen=True, slots=True)
class ComputeSlot:
    """One independently assignable accelerator (or the single host fallback)."""

    index: int
    backend: ComputeBackend
    selector: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"unsupported compute backend: {self.backend}")
        if self.index < 0:
            raise ValueError("compute slot index must be non-negative")
        if self.backend == "cuda" and not self.selector:
            raise ValueError("CUDA compute slots require a selector")
        if self.backend != "cuda" and self.selector is not None:
            raise ValueError("only CUDA compute slots may have a selector")

    @property
    def env_overrides(self) -> dict[str, str]:
        """Return only the environment values needed to isolate this slot."""

        if self.backend == "cuda":
            assert self.selector is not None
            return {"CUDA_VISIBLE_DEVICES": self.selector}
        return {}

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "backend": self.backend,
            "selector": self.selector,
        }


def _default_mps_probe() -> bool:
    """Best-effort MPS detection, kept lazy to avoid a routine torch import."""

    if platform.system() != "Darwin":
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001 - optional framework probing is best effort
        return False

    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def _default_cuda_count_probe() -> int:
    """Return the visible CUDA device count when ``nvidia-smi`` is unavailable."""

    try:
        import torch
    except Exception:  # noqa: BLE001 - optional framework probing is best effort
        return 0

    try:
        return int(torch.cuda.device_count())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def _selectors_from_environment(environ: Mapping[str, str]) -> tuple[str, ...] | None:
    """Return an explicit CUDA mask, or ``None`` when no mask was supplied."""

    if "CUDA_VISIBLE_DEVICES" not in environ:
        return None

    value = environ.get("CUDA_VISIBLE_DEVICES", "")
    selectors: list[str] = []
    seen: set[str] = set()
    for raw_entry in value.split(","):
        selector = raw_entry.strip()
        # CUDA ignores entries after an invalid selector.  Treat empty and known
        # hidden markers as terminators so slot discovery never broadens the mask.
        if (
            selector.lower() in _CUDA_HIDDEN_VALUES
            or _CUDA_SELECTOR_RE.fullmatch(selector) is None
        ):
            break
        # Leasing an exact duplicate twice would run concurrent candidates on one
        # physical selector, so preserve order while collapsing duplicates.
        if selector in seen:
            continue
        seen.add(selector)
        selectors.append(selector)
    return tuple(selectors)


def _query_nvidia_uuids(
    *,
    command_runner: CommandRunner,
    which: Callable[[str], str | None],
) -> tuple[str, ...]:
    executable = which("nvidia-smi")
    if executable is None:
        return ()

    try:
        result = command_runner(
            [
                executable,
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return ()

    if getattr(result, "returncode", 1) != 0:
        return ()
    stdout = getattr(result, "stdout", "")
    if not isinstance(stdout, str):
        return ()
    selectors: list[str] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        selector = line.strip()
        if not selector:
            continue
        # This query promises UUIDs, never ordinals or diagnostic text. Treat
        # any malformed rc=0 response as a failed query so discovery falls back
        # to the CUDA runtime count instead of creating a bogus or broad slot.
        if not selector.startswith(("GPU-", "MIG-")) or (
            _CUDA_SELECTOR_RE.fullmatch(selector) is None
        ):
            return ()
        if selector in seen:
            continue
        seen.add(selector)
        selectors.append(selector)
    return tuple(selectors)


def detect_compute_slots(
    *,
    environ: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
    which: Callable[[str], str | None] | None = None,
    cuda_count_probe: Callable[[], int] | None = None,
    mps_probe: Callable[[], bool] | None = None,
) -> tuple[ComputeSlot, ...]:
    """Detect candidate slots in CUDA -> MPS -> CPU priority order.

    An explicit ``CUDA_VISIBLE_DEVICES`` value is authoritative, including an empty
    value that deliberately hides CUDA.  Its selectors are passed through unchanged
    rather than being translated to local ordinal indices.
    """

    source_environ = os.environ if environ is None else environ
    selectors = _selectors_from_environment(source_environ)
    if selectors is None:
        selectors = _query_nvidia_uuids(
            command_runner=subprocess.run if command_runner is None else command_runner,
            which=shutil.which if which is None else which,
        )
        if not selectors:
            count_probe = (
                _default_cuda_count_probe
                if cuda_count_probe is None
                else cuda_count_probe
            )
            try:
                cuda_count = count_probe()
            except Exception:  # noqa: BLE001 - injected probes are best effort
                cuda_count = 0
            if (
                isinstance(cuda_count, int)
                and not isinstance(cuda_count, bool)
                and cuda_count > 0
            ):
                selectors = tuple(str(index) for index in range(cuda_count))

    if selectors:
        return tuple(
            ComputeSlot(index=index, backend="cuda", selector=selector)
            for index, selector in enumerate(selectors)
        )

    probe = _default_mps_probe if mps_probe is None else mps_probe
    try:
        has_mps = bool(probe())
    except Exception:  # noqa: BLE001 - injected platform probes are best effort
        has_mps = False
    if has_mps:
        return (ComputeSlot(index=0, backend="mps"),)
    return (ComputeSlot(index=0, backend="cpu"),)


class ComputePool:
    """Thread-safe allocation over the effective set of local compute slots."""

    def __init__(
        self,
        slots: Sequence[ComputeSlot],
        requested_workers: WorkerRequest = "auto",
    ) -> None:
        detected = tuple(slots)
        if not detected:
            raise ValueError("at least one compute slot is required")
        indices = [slot.index for slot in detected]
        if len(indices) != len(set(indices)):
            raise ValueError("compute slot indices must be unique")

        self.slots = detected
        self.requested_workers = requested_workers
        self.effective_workers = self._resolve_workers(requested_workers, len(detected))
        self.active_slots = detected[: self.effective_workers]
        self.warning_metadata = self._worker_warning(
            requested_workers, self.effective_workers, len(detected)
        )

        self._condition = threading.Condition()
        self._available = set(self.active_slots)
        self._leases: dict[ComputeSlot, Literal["candidate", "final"]] = {}
        self._final_pending = False
        self._final_active = False
        self._cleanup_failure: ProcessTreeCleanupError | None = None
        self._quarantined: set[ComputeSlot] = set()

    @classmethod
    def detect(
        cls,
        requested_workers: WorkerRequest = "auto",
        *,
        environ: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        which: Callable[[str], str | None] | None = None,
        cuda_count_probe: Callable[[], int] | None = None,
        mps_probe: Callable[[], bool] | None = None,
    ) -> ComputePool:
        return cls(
            detect_compute_slots(
                environ=environ,
                command_runner=command_runner,
                which=which,
                cuda_count_probe=cuda_count_probe,
                mps_probe=mps_probe,
            ),
            requested_workers=requested_workers,
        )

    @staticmethod
    def _resolve_workers(requested: WorkerRequest, slot_count: int) -> int:
        if requested == "auto":
            return slot_count
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested <= 0
        ):
            raise ValueError("requested_workers must be 'auto' or a positive integer")
        return min(requested, slot_count)

    @staticmethod
    def _worker_warning(
        requested: WorkerRequest, effective: int, slot_count: int
    ) -> dict[str, object] | None:
        if requested == "auto" or requested <= slot_count:
            return None
        return {
            "code": "parallel_workers_capped",
            "requested_workers": requested,
            "effective_workers": effective,
            "available_slots": slot_count,
            "message": (
                f"parallel_workers={requested} exceeds the {slot_count} available "
                f"compute slot(s); using {effective}"
            ),
        }

    @property
    def final_slot(self) -> ComputeSlot:
        """The stable slot selected for exclusive final retraining."""

        return self.active_slots[0]

    @property
    def leased_count(self) -> int:
        with self._condition:
            return len(self._leases)

    @property
    def has_cleanup_failure(self) -> bool:
        """Whether a quarantined lease still requires the outer owner's cleanup."""

        with self._condition:
            return self._cleanup_failure is not None

    def _wait_locked(
        self, predicate: Callable[[], bool], timeout: float | None
    ) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._cleanup_failure is not None:
                raise self._cleanup_failure
            if predicate():
                return
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("timed out waiting for a compute slot")
            self._condition.wait(remaining)

    def acquire(self, timeout: float | None = None) -> ComputeSlot:
        """Acquire one candidate slot, blocking until one is available."""

        with self._condition:
            self._wait_locked(
                lambda: bool(self._available)
                and not self._final_pending
                and not self._final_active,
                timeout,
            )
            slot = min(self._available, key=lambda item: item.index)
            self._available.remove(slot)
            self._leases[slot] = "candidate"
            return slot

    def acquire_final(self, timeout: float | None = None) -> ComputeSlot:
        """Wait for all candidate leases, then acquire the deterministic final slot."""

        with self._condition:
            if self._final_pending or self._final_active:
                raise RuntimeError("a final compute slot acquisition is already active")
            self._final_pending = True
            try:
                self._wait_locked(lambda: not self._leases, timeout)
                slot = self.final_slot
                self._available.remove(slot)
                self._leases[slot] = "final"
                self._final_active = True
                return slot
            finally:
                self._final_pending = False
                self._condition.notify_all()

    def release(self, slot: ComputeSlot) -> None:
        """Release a previously acquired slot; double releases are rejected."""

        with self._condition:
            if slot in self._quarantined:
                raise RuntimeError("cannot release a slot with unverified process cleanup")
            lease_kind = self._leases.pop(slot, None)
            if lease_kind is None:
                raise ValueError("compute slot is not leased by this pool")
            self._available.add(slot)
            if lease_kind == "final":
                self._final_active = False
            self._condition.notify_all()

    @contextmanager
    def lease(
        self, *, final: bool = False, timeout: float | None = None
    ) -> Iterator[ComputeSlot]:
        """Release clean leases; quarantine fatal cleanup failures and abort waiters."""

        slot = self.acquire_final(timeout) if final else self.acquire(timeout)
        try:
            yield slot
        except ProcessTreeCleanupError as exc:
            with self._condition:
                self._quarantined.add(slot)
                self._cleanup_failure = exc
                self._condition.notify_all()
            raise
        finally:
            if slot not in self._quarantined:
                self.release(slot)

    def metadata(self) -> dict[str, object]:
        """Return JSON-friendly detection and worker-resolution facts."""

        return {
            "requested_workers": self.requested_workers,
            "effective_workers": self.effective_workers,
            "slots": [slot.as_dict() for slot in self.slots],
            "warning": self.warning_metadata,
        }
