from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from webui.registry import JsonRegistry, RegistryError


def test_registry_crud_and_atomic_callback_failure(tmp_path):
    path = tmp_path / "state" / "registry.json"
    registry = JsonRegistry(path)
    project = {"id": "p1", "name": "first"}
    registry.add_record("projects", project)

    assert registry.get_record("projects", "p1") == project
    assert json.loads(path.read_text(encoding="utf-8"))["projects"]["p1"] == project

    def fail_after_mutation(data):
        data["projects"]["p1"]["name"] = "not persisted"
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        registry.mutate(fail_after_mutation)

    assert registry.get_record("projects", "p1")["name"] == "first"
    assert not list(path.parent.glob("*.tmp"))


def test_registry_serializes_concurrent_writers(tmp_path):
    registry = JsonRegistry(tmp_path / "registry.json")

    def add(index):
        registry.add_record("runs", {"id": f"r{index}", "value": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(40)))

    assert len(registry.list_records("runs")) == 40


def test_registry_refuses_corrupt_state(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(RegistryError, match="Cannot read"):
        JsonRegistry(path).read()
