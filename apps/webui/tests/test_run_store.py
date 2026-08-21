from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from webui.run_store import DEFAULT_QUEUE_CONTROL, MemoryRunStore, SQLiteRunStore


def test_run_store_crud_and_callback_failure_is_atomic():
    store = MemoryRunStore()
    run = {"id": "r1", "status": "queued"}
    added = store.add_record("runs", run)

    assert added == {**run, "queue_sequence": 1}
    assert store.get_record("runs", "r1") == added

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


def test_sqlite_store_persists_records_across_instances(tmp_path):
    database = tmp_path / "state" / "runs.sqlite3"
    first = SQLiteRunStore(database)
    record = {
        "id": "persisted",
        "status": "queued",
        "nested": {"name": "实验", "values": [1, True, None]},
    }
    first.add_record("runs", record)
    first.close()

    second = SQLiteRunStore(database)
    try:
        expected = {**record, "queue_sequence": 1}
        assert second.get_record("runs", "persisted") == expected
        assert second.list_records("runs") == [expected]
        with sqlite3.connect(database) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"
    finally:
        second.close()


def test_sqlite_store_serializes_cross_instance_mutations(tmp_path):
    database = tmp_path / "runs.sqlite3"
    stores = [SQLiteRunStore(database) for _ in range(6)]
    stores[0].add_record("runs", {"id": "counter", "value": 0})

    def increment(index):
        def apply(record):
            record["value"] += 1

        stores[index % len(stores)].update_record("runs", "counter", apply)

    try:
        with ThreadPoolExecutor(max_workers=len(stores)) as executor:
            list(executor.map(increment, range(120)))

        assert stores[-1].get_record("runs", "counter")["value"] == 120
    finally:
        for store in stores:
            store.close()


def test_queue_control_is_atomic_and_persistent(tmp_path):
    database = tmp_path / "runs.sqlite3"
    first = SQLiteRunStore(database)
    second = SQLiteRunStore(database)
    try:
        assert first.get_queue_control() == DEFAULT_QUEUE_CONTROL
        updated = first.update_queue_control(
            {
                "paused": True,
                "reason": "run failed",
                "runner_pid": 1234,
                "runner_process_started_at": "2026-08-21T10:00:00Z",
                "runner_token": "runner-token",
            }
        )
        assert second.get_queue_control() == updated

        def fail_after_mutation(control):
            control["paused"] = False
            raise RuntimeError("do not commit")

        with pytest.raises(RuntimeError, match="do not commit"):
            second.update_queue_control(fail_after_mutation)
        assert first.get_queue_control() == updated
    finally:
        first.close()
        second.close()


def test_runs_and_queue_control_share_one_atomic_transaction(tmp_path):
    database = tmp_path / "runs.sqlite3"
    first = SQLiteRunStore(database)
    second = SQLiteRunStore(database)
    first.add_record("runs", {"id": "queued", "status": "queued"})

    def cancel_and_fail(state):
        state["runs"]["queued"]["status"] = "cancelled"
        state["queue_control"].update({"paused": True, "reason": "cancelled"})
        raise RuntimeError("rollback both")

    try:
        with pytest.raises(RuntimeError, match="rollback both"):
            first.mutate_state(cancel_and_fail)

        assert second.get_record("runs", "queued")["status"] == "queued"
        assert second.get_queue_control()["paused"] is False

        def cancel(state):
            state["runs"]["queued"]["status"] = "cancelled"
            state["queue_control"].update({"paused": True, "reason": "cancelled"})

        first.mutate_state(cancel)
        assert second.get_record("runs", "queued")["status"] == "cancelled"
        assert second.get_queue_control() == {
            **DEFAULT_QUEUE_CONTROL,
            "paused": True,
            "reason": "cancelled",
        }
    finally:
        first.close()
        second.close()


def test_mutate_can_add_and_remove_records_atomically(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    try:
        store.add_record("runs", {"id": "remove", "status": "queued"})

        def replace(data):
            del data["runs"]["remove"]
            data["runs"]["added"] = {"id": "added", "status": "running"}
            return "changed"

        snapshot, result = store.mutate(replace)

        assert result == "changed"
        assert snapshot == {
            "runs": {
                "added": {
                    "id": "added",
                    "status": "running",
                    "queue_sequence": 2,
                }
            }
        }
        assert store.list_records("runs") == [
            {"id": "added", "status": "running", "queue_sequence": 2}
        ]
    finally:
        store.close()


def test_queue_sequence_is_exposed_and_never_reused(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    try:
        first = store.add_record("runs", {"id": "first", "status": "queued"})
        second = store.add_record("runs", {"id": "second", "status": "queued"})

        def replace_first(data):
            del data["runs"]["first"]
            data["runs"]["third"] = {"id": "third", "status": "queued"}
            return data["runs"]["third"]

        state, third = store.mutate_state(replace_first)

        assert first["queue_sequence"] == 1
        assert second["queue_sequence"] == 2
        assert third["queue_sequence"] == 3
        assert [record["queue_sequence"] for record in state["runs"].values()] == [2, 3]
        assert [record["id"] for record in store.list_records("runs")] == [
            "second",
            "third",
        ]
    finally:
        store.close()


def test_non_json_update_rolls_back(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    try:
        original = {"id": "r1", "status": "queued"}
        store.add_record("runs", original)

        with pytest.raises(TypeError):
            store.update_record("runs", "r1", {"bad": object()})

        assert store.get_record("runs", "r1") == {
            **original,
            "queue_sequence": 1,
        }
    finally:
        store.close()
