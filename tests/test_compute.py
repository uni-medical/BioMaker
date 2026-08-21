from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

import pytest

from biomaker.utils.compute import ComputePool, ComputeSlot, detect_compute_slots


def test_explicit_cuda_mask_preserves_raw_selectors_and_skips_other_probes():
    def unexpected(*_args, **_kwargs):
        raise AssertionError("explicit CUDA visibility must be authoritative")

    slots = detect_compute_slots(
        environ={"CUDA_VISIBLE_DEVICES": "2,GPU-raw,MIG-GPU-raw/3/7"},
        command_runner=unexpected,
        which=unexpected,
        mps_probe=unexpected,
    )

    assert [slot.selector for slot in slots] == [
        "2",
        "GPU-raw",
        "MIG-GPU-raw/3/7",
    ]
    assert [slot.env_overrides for slot in slots] == [
        {"CUDA_VISIBLE_DEVICES": "2"},
        {"CUDA_VISIBLE_DEVICES": "GPU-raw"},
        {"CUDA_VISIBLE_DEVICES": "MIG-GPU-raw/3/7"},
    ]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ("0,0,GPU-a,GPU-a", ["0", "GPU-a"]),
        ("0,-1,1", ["0"]),
        ("0,,1", ["0"]),
        ("0,not-a-selector,1", ["0"]),
    ],
)
def test_explicit_cuda_mask_never_creates_duplicate_or_broadened_slots(mask, expected):
    slots = detect_compute_slots(
        environ={"CUDA_VISIBLE_DEVICES": mask},
        command_runner=lambda *_args, **_kwargs: pytest.fail("must not query"),
        which=lambda _name: pytest.fail("must not query"),
        mps_probe=lambda: False,
    )

    assert [slot.selector for slot in slots if slot.backend == "cuda"] == expected


@pytest.mark.parametrize("hidden_value", ["", "-1", "NoDevFiles"])
def test_explicit_hidden_cuda_mask_falls_back_without_querying_nvidia(hidden_value):
    slots = detect_compute_slots(
        environ={"CUDA_VISIBLE_DEVICES": hidden_value},
        command_runner=lambda *_args, **_kwargs: pytest.fail("must not query"),
        which=lambda _name: pytest.fail("must not query"),
        mps_probe=lambda: True,
    )

    assert slots == (ComputeSlot(index=0, backend="mps"),)
    assert slots[0].env_overrides == {}


def test_unmasked_cuda_uses_nvidia_uuids_in_reported_order():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="GPU-bbb\nGPU-aaa\n")

    slots = detect_compute_slots(
        environ={},
        command_runner=run,
        which=lambda name: f"/usr/bin/{name}",
        mps_probe=lambda: pytest.fail("CUDA should win"),
    )

    assert [slot.selector for slot in slots] == ["GPU-bbb", "GPU-aaa"]
    assert calls == [
        (
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            {
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 2,
            },
        )
    ]


def test_unmasked_cuda_uuid_query_deduplicates_in_reported_order():
    slots = detect_compute_slots(
        environ={},
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="GPU-bbb\nGPU-bbb\nGPU-aaa\n",
        ),
        which=lambda _name: "/usr/bin/nvidia-smi",
        cuda_count_probe=lambda: pytest.fail("valid UUIDs must not use fallback"),
        mps_probe=lambda: pytest.fail("valid UUIDs must not use host fallback"),
    )

    assert [slot.selector for slot in slots] == ["GPU-bbb", "GPU-aaa"]


def test_malformed_successful_uuid_query_uses_cuda_count_fallback():
    slots = detect_compute_slots(
        environ={},
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="GPU-valid\nwarning: partial nvidia-smi output\nGPU-also-valid\n",
        ),
        which=lambda _name: "/usr/bin/nvidia-smi",
        cuda_count_probe=lambda: 2,
        mps_probe=lambda: pytest.fail("CUDA count fallback should win"),
    )

    assert [slot.selector for slot in slots] == ["0", "1"]


@pytest.mark.parametrize(
    ("result", "mps_available", "expected_backend"),
    [
        (SimpleNamespace(returncode=1, stdout="error"), True, "mps"),
        (SimpleNamespace(returncode=0, stdout=""), False, "cpu"),
    ],
)
def test_failed_or_empty_cuda_query_uses_single_host_slot(
    result, mps_available, expected_backend
):
    slots = detect_compute_slots(
        environ={},
        command_runner=lambda *_args, **_kwargs: result,
        which=lambda _name: "/usr/bin/nvidia-smi",
        cuda_count_probe=lambda: 0,
        mps_probe=lambda: mps_available,
    )

    assert slots == (ComputeSlot(index=0, backend=expected_backend),)
    assert slots[0].env_overrides == {}


@pytest.mark.parametrize("query_mode", ["missing", "timeout"])
def test_cuda_count_fallback_isolates_each_visible_device_when_uuid_query_fails(
    query_mode,
):
    def run(*_args, **_kwargs):
        raise TimeoutError("nvidia-smi timed out")

    slots = detect_compute_slots(
        environ={},
        command_runner=run,
        which=(
            (lambda _name: None)
            if query_mode == "missing"
            else (lambda _name: "/usr/bin/nvidia-smi")
        ),
        cuda_count_probe=lambda: 2,
        mps_probe=lambda: pytest.fail("CUDA count fallback should win"),
    )

    assert [slot.backend for slot in slots] == ["cuda", "cuda"]
    assert [slot.selector for slot in slots] == ["0", "1"]
    assert [slot.env_overrides for slot in slots] == [
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "1"},
    ]


def test_mps_probe_failure_falls_back_to_cpu():
    def broken_probe():
        raise RuntimeError("framework unavailable")

    slots = detect_compute_slots(
        environ={},
        which=lambda _name: None,
        cuda_count_probe=lambda: 0,
        mps_probe=broken_probe,
    )

    assert slots == (ComputeSlot(index=0, backend="cpu"),)


def test_auto_uses_every_slot_and_explicit_workers_are_capped_with_metadata():
    slots = tuple(
        ComputeSlot(index=index, backend="cuda", selector=f"GPU-{index}")
        for index in range(3)
    )

    automatic = ComputePool(slots, requested_workers="auto")
    capped = ComputePool(slots, requested_workers=8)
    limited = ComputePool(slots, requested_workers=2)

    assert automatic.effective_workers == 3
    assert automatic.warning_metadata is None
    assert capped.effective_workers == 3
    assert capped.warning_metadata == {
        "code": "parallel_workers_capped",
        "requested_workers": 8,
        "effective_workers": 3,
        "available_slots": 3,
        "message": (
            "parallel_workers=8 exceeds the 3 available compute slot(s); using 3"
        ),
    }
    assert limited.effective_workers == 2
    assert limited.active_slots == slots[:2]
    assert limited.warning_metadata is None


@pytest.mark.parametrize("requested", [0, -1, True, "2", "AUTO"])
def test_worker_request_must_be_auto_or_a_positive_integer(requested):
    with pytest.raises(ValueError, match="'auto' or a positive integer"):
        ComputePool((ComputeSlot(index=0, backend="cpu"),), requested)


def test_candidate_acquisition_is_exclusive_and_release_unblocks_a_waiter():
    pool = ComputePool(
        (
            ComputeSlot(index=0, backend="cuda", selector="GPU-0"),
            ComputeSlot(index=1, backend="cuda", selector="GPU-1"),
        )
    )
    first = pool.acquire()
    second = pool.acquire()

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(pool.acquire, 1)
        with pytest.raises(FutureTimeout):
            waiting.result(timeout=0.05)
        pool.release(first)
        assert waiting.result(timeout=1) == first
        pool.release(first)

    pool.release(second)
    assert pool.leased_count == 0
    with pytest.raises(ValueError, match="not leased"):
        pool.release(second)


def test_final_slot_waits_for_all_candidates_and_is_exclusive_and_deterministic():
    slots = (
        ComputeSlot(index=0, backend="cuda", selector="GPU-z"),
        ComputeSlot(index=1, backend="cuda", selector="GPU-a"),
    )
    pool = ComputePool(slots)
    first = pool.acquire()
    second = pool.acquire()

    with ThreadPoolExecutor(max_workers=2) as executor:
        final_future = executor.submit(pool.acquire_final, 1)
        pool.release(first)
        with pytest.raises(FutureTimeout):
            final_future.result(timeout=0.05)
        pool.release(second)
        final = final_future.result(timeout=1)
        assert final == slots[0]

        candidate_future = executor.submit(pool.acquire, 0.1)
        with pytest.raises(TimeoutError, match="timed out"):
            candidate_future.result(timeout=1)

        pool.release(final)

    candidate = pool.acquire()
    assert candidate == slots[0]
    pool.release(candidate)


def test_lease_context_releases_after_an_exception():
    pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))

    with pytest.raises(RuntimeError, match="candidate failed"), pool.lease() as slot:
        assert slot.backend == "cpu"
        raise RuntimeError("candidate failed")

    assert pool.leased_count == 0
    with pool.lease(final=True) as final:
        assert final == pool.final_slot


def test_detect_constructor_exposes_json_friendly_metadata():
    pool = ComputePool.detect(
        requested_workers="auto",
        environ={"CUDA_VISIBLE_DEVICES": "GPU-one,GPU-two"},
    )

    assert pool.metadata() == {
        "requested_workers": "auto",
        "effective_workers": 2,
        "slots": [
            {"index": 0, "backend": "cuda", "selector": "GPU-one"},
            {"index": 1, "backend": "cuda", "selector": "GPU-two"},
        ],
        "warning": None,
    }
