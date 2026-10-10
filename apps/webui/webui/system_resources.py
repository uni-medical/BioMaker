"""Best-effort, system-wide resource telemetry for the WebUI."""

from __future__ import annotations

import copy
import csv
import math
import os
import platform
import plistlib
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

CACHE_TTL_SECONDS = 1.0
GPU_PROBE_TIMEOUT_SECONDS = 1.0
MEBIBYTE = 1024 * 1024

_AUTO = object()
_UNAVAILABLE_VALUES = {
    "",
    "n/a",
    "[n/a]",
    "not supported",
    "[not supported]",
}
_NVIDIA_QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
)


def _load_psutil() -> Any | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil


def _probe_mps_with_python(
    python_executable: Path,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> bool:
    result = command_runner(
        [
            str(python_executable),
            "-c",
            "import torch; print(int(torch.backends.mps.is_available()))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = str(getattr(result, "stdout", "")).strip()
    if getattr(result, "returncode", 1) != 0 or output not in {"0", "1"}:
        raise RuntimeError("BioMaker's Python runtime could not report MPS support")
    return output == "1"


@lru_cache(maxsize=1)
def _default_mps_probe() -> bool:
    """Probe the BioMaker runtime, not the intentionally small WebUI environment."""

    try:
        import torch
    except Exception:
        method_root = Path(os.environ.get("BIOMAKER_WEBUI_METHOD_ROOT", Path(__file__).resolve().parents[3]))
        method_python = method_root / ".venv" / "bin" / "python"
        if not method_python.is_file():
            raise RuntimeError("BioMaker's Python runtime is unavailable") from None
        return _probe_mps_with_python(method_python)
    else:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(backend is not None and backend.is_available())


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} is outside its expected range")
    return result


def _percent(value: Any, *, label: str) -> float:
    result = _number(value, label=label)
    if result > 100.0:
        raise ValueError(f"{label} is outside its expected range")
    return result


def _optional_number(value: str, *, label: str) -> float | None:
    if value.strip().casefold() in _UNAVAILABLE_VALUES:
        return None
    return _number(value, label=label)


def _parse_nvidia_csv(output: str) -> list[dict[str, Any]]:
    if not output.strip():
        return []

    gpus: list[dict[str, Any]] = []
    rows = csv.reader(output.splitlines(), skipinitialspace=True)
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(_NVIDIA_QUERY_FIELDS):
            raise ValueError(f"NVIDIA row {row_number} has an unexpected field count")
        values = [value.strip() for value in row]
        try:
            index = int(values[0])
        except ValueError as exc:
            raise ValueError(f"NVIDIA row {row_number} has an invalid index") from exc
        if index < 0 or not values[1]:
            raise ValueError(f"NVIDIA row {row_number} is missing its identity")

        utilization = _optional_number(
            values[2], label=f"NVIDIA row {row_number} utilization"
        )
        if utilization is not None and utilization > 100.0:
            raise ValueError(
                f"NVIDIA row {row_number} utilization is outside its expected range"
            )
        memory_used = _optional_number(
            values[3], label=f"NVIDIA row {row_number} used memory"
        )
        memory_total = _optional_number(
            values[4], label=f"NVIDIA row {row_number} total memory"
        )
        temperature = _optional_number(
            values[5], label=f"NVIDIA row {row_number} temperature"
        )

        gpus.append(
            {
                "index": index,
                "name": values[1],
                "utilization_percent": utilization,
                "memory_used_bytes": (
                    round(memory_used * MEBIBYTE) if memory_used is not None else None
                ),
                "memory_total_bytes": (
                    round(memory_total * MEBIBYTE) if memory_total is not None else None
                ),
                "temperature_celsius": temperature,
            }
        )
    return gpus


def _parse_apple_gpu_plist(output: bytes | str) -> dict[str, Any]:
    raw = output.encode("utf-8") if isinstance(output, str) else output
    try:
        payload = plistlib.loads(raw)
    except (plistlib.InvalidFileException, TypeError, ValueError) as exc:
        raise ValueError("Apple GPU metrics are not a valid property list") from exc
    if not isinstance(payload, list):
        raise ValueError("Apple GPU metrics have an unexpected shape")

    statistics = [
        item.get("PerformanceStatistics")
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("PerformanceStatistics"), dict)
    ]
    if len(statistics) != 1:
        raise ValueError("Apple GPU metrics have an unexpected accelerator count")
    values = statistics[0]
    utilization = (
        _percent(
            values["Device Utilization %"],
            label="Apple GPU utilization",
        )
        if "Device Utilization %" in values
        else None
    )
    shared_memory = (
        round(
            _number(
                values["In use system memory"],
                label="Apple GPU shared memory",
            )
        )
        if "In use system memory" in values
        else None
    )
    if utilization is None and shared_memory is None:
        raise ValueError("Apple GPU metrics have no supported fields")
    return {
        "source": "ioreg",
        "utilization_percent": utilization,
        "shared_memory_used_bytes": shared_memory,
    }


def _error_text(errors: list[str]) -> str | None:
    return " ".join(errors) if errors else None


class SystemResourceSampler:
    """Collect a small, cached snapshot without making telemetry a dependency."""

    def __init__(
        self,
        *,
        psutil_provider: Any = _AUTO,
        command_runner: Callable[..., Any] = subprocess.run,
        command_resolver: Callable[[str], str | None] = shutil.which,
        system_name: str | None = None,
        mps_probe: Callable[[], bool] = _default_mps_probe,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        cache_ttl_seconds: float = CACHE_TTL_SECONDS,
        gpu_probe_timeout_seconds: float = GPU_PROBE_TIMEOUT_SECONDS,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        if gpu_probe_timeout_seconds <= 0:
            raise ValueError("gpu_probe_timeout_seconds must be positive")

        self._psutil = _load_psutil() if psutil_provider is _AUTO else psutil_provider
        self._cpu_has_public_sample = False
        if self._psutil is not None:
            try:
                # psutil documents the first non-blocking reading as meaningless.
                self._psutil.cpu_percent(interval=None)
            except Exception:
                # Collection remains fail-soft and will report a public error later.
                pass
        self._command_runner = command_runner
        self._command_resolver = command_resolver
        self._system_name = system_name or platform.system()
        self._mps_probe = mps_probe
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock
        self._cache_ttl_seconds = cache_ttl_seconds
        self._gpu_probe_timeout_seconds = gpu_probe_timeout_seconds
        self._cache: dict[str, Any] | None = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def sample(self) -> dict[str, Any]:
        """Return a fresh snapshot or a defensive copy of the one-second cache."""

        with self._lock:
            now = self._monotonic_clock()
            if (
                self._cache is not None
                and now - self._cached_at < self._cache_ttl_seconds
            ):
                return copy.deepcopy(self._cache)

            try:
                snapshot = self._collect()
            except Exception:
                if self._cache is not None:
                    snapshot = copy.deepcopy(self._cache)
                    snapshot["stale"] = True
                    snapshot["error"] = "System resource refresh failed."
                else:
                    snapshot = self._empty_snapshot(
                        sampled_at=datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        error="System resource sampling failed.",
                    )

            self._cache = snapshot
            self._cached_at = self._monotonic_clock()
            return copy.deepcopy(snapshot)

    def _collect(self) -> dict[str, Any]:
        errors: list[str] = []
        cpu = {"utilization_percent": None}
        memory = {
            "used_bytes": None,
            "total_bytes": None,
            "utilization_percent": None,
        }

        if self._psutil is None:
            errors.append("CPU and RAM metrics require psutil.")
        else:
            try:
                cpu["utilization_percent"] = _percent(
                    self._psutil.cpu_percent(
                        interval=None if self._cpu_has_public_sample else 0.1
                    ),
                    label="CPU utilization",
                )
                self._cpu_has_public_sample = True
            except Exception:
                errors.append("CPU utilization is unavailable.")

            try:
                virtual_memory = self._psutil.virtual_memory()
                total = round(
                    _number(virtual_memory.total, label="total system memory")
                )
                available = round(
                    _number(
                        virtual_memory.available,
                        label="available system memory",
                    )
                )
                if total <= 0 or available > total:
                    raise ValueError("system memory values are inconsistent")
                used = total - available
                memory.update(
                    {
                        "used_bytes": used,
                        "total_bytes": total,
                        "utilization_percent": 100.0 * used / total,
                    }
                )
            except Exception:
                errors.append("RAM utilization is unavailable.")

        nvidia_gpus, nvidia_error = self._nvidia_gpus()
        if nvidia_error:
            errors.append(nvidia_error)

        apple_mps = {"available": False}
        apple_gpu = None
        if self._system_name == "Darwin":
            try:
                apple_mps["available"] = bool(self._mps_probe())
            except Exception:
                apple_mps["available"] = None
                errors.append("Apple MPS availability could not be determined.")
            apple_gpu, apple_gpu_error = self._apple_gpu()
            if apple_gpu_error:
                errors.append(apple_gpu_error)

        return {
            "sampled_at": _timestamp(self._wall_clock),
            "stale": False,
            "error": _error_text(errors),
            "cpu": cpu,
            "memory": memory,
            "nvidia_gpus": nvidia_gpus,
            "apple_mps": apple_mps,
            "apple_gpu": apple_gpu,
        }

    def _nvidia_gpus(self) -> tuple[list[dict[str, Any]], str | None]:
        try:
            executable = self._command_resolver("nvidia-smi")
        except Exception:
            return [], "NVIDIA metrics are unavailable."
        if executable is None:
            return [], None

        command = [
            executable,
            f"--query-gpu={','.join(_NVIDIA_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = self._command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=self._gpu_probe_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return [], "NVIDIA metrics are unavailable."

        if result.returncode != 0:
            return [], "NVIDIA metrics are unavailable."
        try:
            return _parse_nvidia_csv(result.stdout), None
        except (AttributeError, TypeError, ValueError):
            return [], "NVIDIA metrics returned an invalid response."

    def _apple_gpu(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            executable = self._command_resolver("ioreg")
        except Exception:
            return None, "Apple GPU metrics are unavailable."
        if executable is None:
            return None, "Apple GPU metrics are unavailable."
        try:
            result = self._command_runner(
                [executable, "-a", "-r", "-c", "AGXAccelerator", "-d", "1"],
                capture_output=True,
                timeout=self._gpu_probe_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None, "Apple GPU metrics are unavailable."
        if result.returncode != 0:
            return None, "Apple GPU metrics are unavailable."
        try:
            return _parse_apple_gpu_plist(result.stdout), None
        except (AttributeError, TypeError, ValueError):
            return None, "Apple GPU metrics returned an invalid response."

    @staticmethod
    def _empty_snapshot(*, sampled_at: str, error: str) -> dict[str, Any]:
        return {
            "sampled_at": sampled_at,
            "stale": False,
            "error": error,
            "cpu": {"utilization_percent": None},
            "memory": {
                "used_bytes": None,
                "total_bytes": None,
                "utilization_percent": None,
            },
            "nvidia_gpus": [],
            "apple_mps": {"available": None},
            "apple_gpu": None,
        }


_DEFAULT_SAMPLER = SystemResourceSampler()


def get_system_resources() -> dict[str, Any]:
    """Return the process-wide cached system resource snapshot."""

    return _DEFAULT_SAMPLER.sample()
