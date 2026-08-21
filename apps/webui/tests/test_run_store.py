from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from webui.run_store import MemoryRunStore


def test_run_store_crud_and_callback_failure_is_atomic():
    store = MemoryRunStore()
    run = {"id": "r1", "status": "queued"}
    store.add_record("runs", run)

    assert store.get_record("runs", "r1") == run

    def fail_after_mutation(data):
        data["runs"]["r1"]["status"] = "not persisted"
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        store.mutate(fail_after_mutation)

    assert store.get_record("runs", "r1")["status"] == "queued"


def test_run_store_serializes_concurrent_writers():
    store = MemoryRunStore()

    def add(index):
        store.add_record("runs", {"id": f"r{index}", "value": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(40)))

    assert len(store.list_records("runs")) == 40
