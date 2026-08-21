from __future__ import annotations

import plistlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from webui.system_resources import (
    MEBIBYTE,
    SystemResourceSampler,
    _parse_apple_gpu_plist,
    _probe_mps_with_python,
)


class FakePsutil:
    def __init__(self, *, cpu_percent=12.5, memory_percent=25.0):
        self.cpu_value = cpu_percent
        self.memory_percent = memory_percent
        self.cpu_calls = 0
        self.memory_calls = 0

    def cpu_percent(self, *, interval):
        assert interval is None
        self.cpu_calls += 1
        if isinstance(self.cpu_value, Exception):
            raise self.cpu_value
        return self.cpu_value

    def virtual_memory(self):
        self.memory_calls += 1
        if isinstance(self.memory_percent, Exception):
            raise self.memory_percent
        return SimpleNamespace(
            used=2 * MEBIBYTE,
            available=6 * MEBIBYTE,
            total=8 * MEBIBYTE,
            percent=self.memory_percent,
        )


def test_mps_probe_uses_biomaker_python_and_accepts_only_boolean_output(tmp_path):
    python_executable = tmp_path / "python"
    python_executable.touch()
    calls = []

    def available(command, **kwargs):
        calls.append((command, kwargs))
        return _completed("1\n")

    assert (
        _probe_mps_with_python(
            python_executable,
            command_runner=available,
        )
        is True
    )
    assert calls[0][0][0] == str(python_executable)
    assert calls[0][1]["timeout"] == 5

    with pytest.raises(RuntimeError, match="could not report MPS"):
        _probe_mps_with_python(
            python_executable,
            command_runner=lambda *_args, **_kwargs: _completed("maybe\n"),
        )


def _completed(stdout: Any, *, returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def _apple_gpu_plist(
    *,
    utilization: Any = 97,
    shared_memory: Any = 1_204_289_536,
) -> bytes:
    return plistlib.dumps(
        [
            {
                "IOClass": "AGXAccelerator",
                "PerformanceStatistics": {
                    "Device Utilization %": utilization,
                    "In use system memory": shared_memory,
                    "Alloc system memory": 2_147_483_648,
                },
            }
        ]
    )


def test_samples_system_cpu_ram_nvidia_and_honest_mps_fields():
    psutil = FakePsutil()
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if command[0] == "/usr/sbin/ioreg":
            return _completed(_apple_gpu_plist())
        return _completed(
            '0, "NVIDIA RTX, Test", 87, 1024, 24576, 63\n'
            "1, NVIDIA RTX B, N/A, 0, 24576, [Not Supported]\n"
        )

    def resolve(name):
        return {
            "nvidia-smi": "/usr/bin/nvidia-smi",
            "ioreg": "/usr/sbin/ioreg",
        }.get(name)

    sampler = SystemResourceSampler(
        psutil_provider=psutil,
        command_runner=run,
        command_resolver=resolve,
        system_name="Darwin",
        mps_probe=lambda: True,
        wall_clock=lambda: datetime(2026, 8, 21, 8, 9, 10, tzinfo=timezone.utc),
    )

    result = sampler.sample()

    assert result == {
        "sampled_at": "2026-08-21T08:09:10Z",
        "stale": False,
        "error": None,
        "cpu": {"utilization_percent": 12.5},
        "memory": {
            "used_bytes": 2 * MEBIBYTE,
            "total_bytes": 8 * MEBIBYTE,
            "utilization_percent": 25.0,
        },
        "nvidia_gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX, Test",
                "utilization_percent": 87.0,
                "memory_used_bytes": 1024 * MEBIBYTE,
                "memory_total_bytes": 24576 * MEBIBYTE,
                "temperature_celsius": 63.0,
            },
            {
                "index": 1,
                "name": "NVIDIA RTX B",
                "utilization_percent": None,
                "memory_used_bytes": 0,
                "memory_total_bytes": 24576 * MEBIBYTE,
                "temperature_celsius": None,
            },
        ],
        "apple_mps": {"available": True},
        "apple_gpu": {
            "source": "ioreg",
            "utilization_percent": 97.0,
            "shared_memory_used_bytes": 1_204_289_536,
        },
    }
    nvidia_command, nvidia_kwargs = commands[0]
    assert nvidia_command == [
        "/usr/bin/nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert nvidia_kwargs == {
        "capture_output": True,
        "text": True,
        "timeout": 1.0,
        "check": False,
    }
    apple_command, apple_kwargs = commands[1]
    assert apple_command == [
        "/usr/sbin/ioreg",
        "-a",
        "-r",
        "-c",
        "AGXAccelerator",
        "-d",
        "1",
    ]
    assert apple_kwargs == {
        "capture_output": True,
        "timeout": 1.0,
        "check": False,
    }


def test_apple_gpu_plist_parses_utilization_and_shared_memory():
    assert _parse_apple_gpu_plist(_apple_gpu_plist()) == {
        "source": "ioreg",
        "utilization_percent": 97.0,
        "shared_memory_used_bytes": 1_204_289_536,
    }


def test_apple_gpu_plist_rejects_malformed_payload():
    with pytest.raises(ValueError, match="not a valid property list"):
        _parse_apple_gpu_plist(b"not a plist")


def test_apple_gpu_plist_rejects_multiple_accelerators():
    accelerator = {
        "PerformanceStatistics": {
            "Device Utilization %": 50,
            "In use system memory": 1024,
        }
    }
    payload = plistlib.dumps([accelerator, accelerator])

    with pytest.raises(ValueError, match="unexpected accelerator count"):
        _parse_apple_gpu_plist(payload)


@pytest.mark.parametrize(
    "missing_field",
    ["Device Utilization %", "In use system memory"],
)
def test_apple_gpu_plist_rejects_missing_required_fields(missing_field):
    statistics = {
        "Device Utilization %": 50,
        "In use system memory": 1024,
    }
    statistics.pop(missing_field)
    payload = plistlib.dumps([{"PerformanceStatistics": statistics}])

    with pytest.raises(ValueError, match="is not numeric"):
        _parse_apple_gpu_plist(payload)


def test_apple_gpu_probe_timeout_is_non_fatal_and_reported():
    commands = []

    def time_out(command, **kwargs):
        commands.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_runner=time_out,
        command_resolver=lambda name: ("/usr/sbin/ioreg" if name == "ioreg" else None),
        system_name="Darwin",
        mps_probe=lambda: True,
    )

    result = sampler.sample()

    assert result["apple_mps"] == {"available": True}
    assert result["apple_gpu"] is None
    assert result["error"] == "Apple GPU metrics are unavailable."
    assert commands == [
        (
            [
                "/usr/sbin/ioreg",
                "-a",
                "-r",
                "-c",
                "AGXAccelerator",
                "-d",
                "1",
            ],
            {
                "capture_output": True,
                "timeout": 1.0,
                "check": False,
            },
        )
    ]


def test_apple_gpu_probe_is_independent_of_pytorch_mps_support():
    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_runner=lambda *_args, **_kwargs: _completed(_apple_gpu_plist()),
        command_resolver=lambda name: ("/usr/sbin/ioreg" if name == "ioreg" else None),
        system_name="Darwin",
        mps_probe=lambda: False,
    )

    result = sampler.sample()

    assert result["apple_mps"] == {"available": False}
    assert result["apple_gpu"] == {
        "source": "ioreg",
        "utilization_percent": 97.0,
        "shared_memory_used_bytes": 1_204_289_536,
    }


def test_cache_is_thread_safe_expires_after_one_second_and_returns_copies():
    psutil = FakePsutil()
    monotonic = [10.0]
    sampler = SystemResourceSampler(
        psutil_provider=psutil,
        command_resolver=lambda _name: None,
        system_name="Linux",
        monotonic_clock=lambda: monotonic[0],
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: sampler.sample(), range(24)))

    assert psutil.cpu_calls == 1
    assert psutil.memory_calls == 1
    results[0]["cpu"]["utilization_percent"] = 99.0
    assert results[1]["cpu"]["utilization_percent"] == 12.5

    monotonic[0] = 10.999
    sampler.sample()
    assert psutil.cpu_calls == 1

    monotonic[0] = 11.0
    sampler.sample()
    assert psutil.cpu_calls == 2


def test_metric_and_command_failures_are_non_fatal_and_reported():
    psutil = FakePsutil(
        cpu_percent=RuntimeError("cpu exploded"),
        memory_percent=RuntimeError("memory exploded"),
    )

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", timeout=1)

    sampler = SystemResourceSampler(
        psutil_provider=psutil,
        command_runner=time_out,
        command_resolver=lambda _name: "nvidia-smi",
        system_name="Darwin",
        mps_probe=lambda: (_ for _ in ()).throw(RuntimeError("mps exploded")),
    )

    result = sampler.sample()

    assert result["stale"] is False
    assert result["cpu"] == {"utilization_percent": None}
    assert result["memory"] == {
        "used_bytes": None,
        "total_bytes": None,
        "utilization_percent": None,
    }
    assert result["nvidia_gpus"] == []
    assert result["apple_mps"]["available"] is None
    assert result["apple_gpu"] is None
    assert "CPU utilization is unavailable" in result["error"]
    assert "RAM utilization is unavailable" in result["error"]
    assert "NVIDIA metrics are unavailable" in result["error"]
    assert "Apple MPS availability" in result["error"]
    assert "exploded" not in result["error"]


def test_missing_optional_collectors_are_safe_and_malformed_gpu_csv_is_rejected():
    without_collectors = SystemResourceSampler(
        psutil_provider=None,
        command_resolver=lambda _name: None,
        system_name="Linux",
    ).sample()

    assert without_collectors["cpu"]["utilization_percent"] is None
    assert without_collectors["memory"]["total_bytes"] is None
    assert without_collectors["nvidia_gpus"] == []
    assert without_collectors["apple_mps"]["available"] is False
    assert without_collectors["apple_gpu"] is None
    assert without_collectors["error"] == "CPU and RAM metrics require psutil."

    malformed_gpu = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_runner=lambda *_args, **_kwargs: _completed("0, too, short\n"),
        command_resolver=lambda _name: "nvidia-smi",
        system_name="Linux",
    ).sample()

    assert malformed_gpu["nvidia_gpus"] == []
    assert "invalid response" in malformed_gpu["error"]


def test_unexpected_refresh_failure_serves_previous_snapshot_as_stale():
    wall_clock_calls = [0]

    def wall_clock():
        wall_clock_calls[0] += 1
        if wall_clock_calls[0] == 2:
            raise RuntimeError("clock failed")
        return datetime(2026, 8, 21, tzinfo=timezone.utc)

    monotonic = [1.0]
    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_resolver=lambda _name: None,
        system_name="Linux",
        wall_clock=wall_clock,
        monotonic_clock=lambda: monotonic[0],
    )
    first = sampler.sample()

    monotonic[0] = 2.0
    stale = sampler.sample()

    assert stale["sampled_at"] == first["sampled_at"]
    assert stale["stale"] is True
    assert stale["error"] == "System resource refresh failed."
