from __future__ import annotations

import math
import plistlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from webui.system_resources import (
    MEBIBYTE,
    SystemResourceSampler,
    _parse_apple_gpu_plist,
)


class FakePsutil:
    def __init__(self, *, cpu_percent=12.5, memory_percent=25.0):
        self.cpu_value = cpu_percent
        self.memory_percent = memory_percent
        self.cpu_calls = 0
        self.cpu_intervals = []
        self.memory_calls = 0

    def cpu_percent(self, *, interval):
        assert interval is None or interval == 0.1
        self.cpu_calls += 1
        self.cpu_intervals.append(interval)
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


class SequencedCpuPsutil(FakePsutil):
    def __init__(self, values):
        super().__init__()
        self._values = iter(values)

    def cpu_percent(self, *, interval):
        assert interval is None or interval == 0.1
        self.cpu_calls += 1
        self.cpu_intervals.append(interval)
        return next(self._values)


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


def test_samples_system_cpu_ram_nvidia_and_apple_gpu_fields():
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
    "payload, message",
    [
        (plistlib.dumps({}), "unexpected shape"),
        (plistlib.dumps([]), "unexpected accelerator count"),
    ],
)
def test_apple_gpu_plist_rejects_non_list_and_missing_accelerator(payload, message):
    with pytest.raises(ValueError, match=message):
        _parse_apple_gpu_plist(payload)


@pytest.mark.parametrize(
    "missing_field, expected_utilization, expected_memory",
    [
        ("Device Utilization %", None, 1024),
        ("In use system memory", 50.0, None),
    ],
)
def test_apple_gpu_plist_keeps_each_supported_metric_independently(
    missing_field,
    expected_utilization,
    expected_memory,
):
    statistics = {
        "Device Utilization %": 50,
        "In use system memory": 1024,
    }
    statistics.pop(missing_field)
    payload = plistlib.dumps([{"PerformanceStatistics": statistics}])

    assert _parse_apple_gpu_plist(payload) == {
        "source": "ioreg",
        "utilization_percent": expected_utilization,
        "shared_memory_used_bytes": expected_memory,
    }


def test_apple_gpu_plist_rejects_payload_without_supported_fields():
    payload = plistlib.dumps([{"PerformanceStatistics": {"Other": 1}}])

    with pytest.raises(ValueError, match="no supported fields"):
        _parse_apple_gpu_plist(payload)


@pytest.mark.parametrize(
    "utilization, shared_memory",
    [
        (-0.1, 1024),
        (100.1, 1024),
        (math.nan, 1024),
        (math.inf, 1024),
        (True, 1024),
        (50, -1),
        (50, math.nan),
        (50, math.inf),
        (50, True),
    ],
)
def test_apple_gpu_plist_rejects_invalid_metric_ranges(
    utilization,
    shared_memory,
):
    with pytest.raises(ValueError):
        _parse_apple_gpu_plist(
            _apple_gpu_plist(
                utilization=utilization,
                shared_memory=shared_memory,
            )
        )


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
    )

    result = sampler.sample()

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


def test_apple_gpu_probe_missing_executable_is_non_fatal():
    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_runner=lambda *_args, **_kwargs: pytest.fail(
            "missing commands must not run"
        ),
        command_resolver=lambda _name: None,
        system_name="Darwin",
    )

    result = sampler.sample()

    assert result["apple_gpu"] is None
    assert result["error"] == "Apple GPU metrics are unavailable."


def test_apple_gpu_probe_nonzero_exit_is_non_fatal():
    commands = []

    def fail(command, **kwargs):
        commands.append((command, kwargs))
        return _completed(b"permission denied", returncode=1)

    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_runner=fail,
        command_resolver=lambda name: ("/usr/sbin/ioreg" if name == "ioreg" else None),
        system_name="Darwin",
    )

    result = sampler.sample()

    assert result["apple_gpu"] is None
    assert result["error"] == "Apple GPU metrics are unavailable."
    assert commands[0][0][0] == "/usr/sbin/ioreg"


@pytest.mark.parametrize(
    "system_name, failing_command, expected_error",
    [
        ("Linux", "nvidia-smi", "NVIDIA metrics are unavailable."),
        ("Darwin", "ioreg", "Apple GPU metrics are unavailable."),
    ],
)
def test_gpu_command_resolution_failure_preserves_cpu_and_ram(
    system_name,
    failing_command,
    expected_error,
):
    def resolve(name):
        if name == failing_command:
            raise OSError("resolver failed")
        return None

    sampler = SystemResourceSampler(
        psutil_provider=FakePsutil(),
        command_resolver=resolve,
        system_name=system_name,
    )

    result = sampler.sample()

    assert result["cpu"] == {"utilization_percent": 12.5}
    assert result["memory"] == {
        "used_bytes": 2 * MEBIBYTE,
        "total_bytes": 8 * MEBIBYTE,
        "utilization_percent": 25.0,
    }
    assert result["error"] == expected_error


def test_primes_psutil_before_returning_the_first_public_cpu_sample():
    psutil = SequencedCpuPsutil([0.0, 37.5])
    sampler = SystemResourceSampler(
        psutil_provider=psutil,
        command_resolver=lambda _name: None,
        system_name="Linux",
    )

    result = sampler.sample()

    assert result["cpu"] == {"utilization_percent": 37.5}
    assert psutil.cpu_calls == 2
    assert psutil.cpu_intervals == [None, 0.1]


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

    assert psutil.cpu_calls == 2
    assert psutil.memory_calls == 1
    results[0]["cpu"]["utilization_percent"] = 99.0
    assert results[1]["cpu"]["utilization_percent"] == 12.5

    monotonic[0] = 10.999
    sampler.sample()
    assert psutil.cpu_calls == 2

    monotonic[0] = 11.0
    sampler.sample()
    assert psutil.cpu_calls == 3


def test_cache_ttl_starts_after_slow_collection_and_waiters_reuse_result():
    monotonic = [0.0]
    collection_started = Event()
    release_collection = Event()

    class SlowPsutil(FakePsutil):
        def cpu_percent(self, *, interval):
            assert interval is None or interval == 0.1
            self.cpu_calls += 1
            self.cpu_intervals.append(interval)
            if self.cpu_calls == 1:
                return 0.0
            collection_started.set()
            assert release_collection.wait(timeout=2)
            monotonic[0] = 2.0
            return 12.5

    psutil = SlowPsutil()
    sampler = SystemResourceSampler(
        psutil_provider=psutil,
        command_resolver=lambda _name: None,
        system_name="Linux",
        monotonic_clock=lambda: monotonic[0],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(sampler.sample)
        assert collection_started.wait(timeout=2)
        second = executor.submit(sampler.sample)
        release_collection.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert first_result == second_result
    assert psutil.cpu_calls == 2
    assert psutil.memory_calls == 1


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
        system_name="Linux",
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
    assert result["apple_gpu"] is None
    assert "CPU utilization is unavailable" in result["error"]
    assert "RAM utilization is unavailable" in result["error"]
    assert "NVIDIA metrics are unavailable" in result["error"]
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
