from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact_preview
from artifact_preview import ArtifactPreviewError, MAX_PREVIEW_BYTES, preview_artifact
from test_desktop_service import AUTH, client_for, config, create_desktop_app


class Store:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def get_record(self, table, identifier):
        self.calls.append((table, identifier))
        return self.records.get(identifier)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "logs"
    directory = root / "registered"
    directory.mkdir(parents=True)
    path = directory / "report.md"
    path.write_bytes(b"original")
    store = Store({"registered": {"id": "registered", "log_dir": str(directory)}})
    return root, directory, path, store


def read(source, name="report.md", run_id="registered"):
    root, _, _, store = source
    return preview_artifact(root, store, run_id, name)


def error(source, code, **kwargs):
    with pytest.raises(ArtifactPreviewError) as raised:
        read(source, **kwargs)
    assert raised.value.code == code
    assert str(raised.value).startswith(f"[artifact:{code}]")
    assert "private-sentinel" not in str(raised.value)
    assert str(source[0]) not in str(raised.value)
    return raised.value


def test_exact_snapshot_preserves_utf8_bom_crlf_nonfinite_json_and_zero_bytes(source):
    raw = '\ufeff{ "message": "实验 😀", "metric": NaN }\r\n\0'.encode("utf-8")
    source[2].write_bytes(raw)
    before = source[2].stat()
    result = read(source)
    assert result == {"name": "report.md", "content": raw.decode("utf-8"), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert result["content"].encode("utf-8") == raw
    assert source[2].read_bytes() == raw
    assert source[2].stat().st_mtime_ns == before.st_mtime_ns
    assert source[3].calls == [("runs", "registered")]
    source[2].write_bytes(b"")
    assert read(source)["sha256"] == hashlib.sha256(b"").hexdigest()


def test_unregistered_directory_is_never_discovered_on_disk(source):
    root, _, _, store = source
    other = root / "unregistered"
    other.mkdir()
    (other / "report.md").write_text("private-sentinel")
    assert error(source, "unavailable", run_id="unregistered").status_code == 404
    assert store.calls == [("runs", "unregistered")]


@pytest.mark.parametrize("field,value", [
    ("name", "../private-sentinel"), ("name", ".env"), ("name", "a/b"),
    ("name", "a\\b"), ("name", "x\0y"), ("name", "x\ny"), ("name", "x" * 241),
    ("run_id", ".."), ("run_id", "../private-sentinel"), ("run_id", "x" * 181),
])
def test_invalid_identifiers_fail_before_store_or_filesystem_access(source, field, value):
    assert error(source, "invalid_request", **{field: value}).status_code == 422
    assert not source[3].calls


def test_snapshot_size_limit_is_byte_based_and_inclusive(source, monkeypatch):
    source[2].write_bytes(b"x" * MAX_PREVIEW_BYTES)
    assert read(source)["size"] == MAX_PREVIEW_BYTES
    source[2].write_bytes(b"x" * (MAX_PREVIEW_BYTES + 1))
    monkeypatch.setattr(artifact_preview, "_read_bytes", lambda _fd: pytest.fail("oversized file must not be read"))
    assert error(source, "too_large").status_code == 413


@pytest.mark.parametrize("raw", [b"\xffprivate-sentinel", b"\xe4\xb8", b"\xed\xa0\x80"])
def test_invalid_utf8_has_a_fixed_nonleaking_error(source, raw):
    source[2].write_bytes(raw)
    assert error(source, "invalid_text").status_code == 422


def test_new_snapshot_digest_detects_equal_length_content_with_restored_mtime(source):
    first = read(source)
    before = source[2].stat()
    source[2].write_bytes(b"changed!")
    os.utime(source[2], ns=(before.st_atime_ns, before.st_mtime_ns))
    second = read(source)
    assert first["size"] == second["size"]
    assert source[2].stat().st_mtime_ns == before.st_mtime_ns
    assert first["sha256"] != second["sha256"]


@pytest.mark.parametrize("mutation", ["in_place", "replace", "delete", "directory_replace"])
def test_changes_during_read_are_rejected_without_retries(source, monkeypatch, mutation):
    original_read = artifact_preview._read_bytes
    reads = 0

    def changing(descriptor):
        nonlocal reads
        reads += 1
        raw = original_read(descriptor)
        _, directory, path, _ = source
        before = path.stat()
        if mutation == "in_place":
            path.write_bytes(b"changed!")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif mutation == "replace":
            replacement = directory / "replacement"
            replacement.write_bytes(b"original")
            os.replace(replacement, path)
        elif mutation == "delete":
            path.unlink()
        else:
            directory.rename(directory.with_name("previous"))
            directory.mkdir()
            path.write_bytes(b"original")
        return raw

    monkeypatch.setattr(artifact_preview, "_read_bytes", changing)
    assert error(source, "changed").status_code == 409
    assert reads == 1


def test_source_identity_is_checked_between_lstat_and_open(source, monkeypatch):
    original_open = artifact_preview._open_source

    def replaced(path, identities):
        replacement = path.with_name("replacement")
        replacement.write_bytes(b"original")
        os.replace(replacement, path)
        return original_open(path, identities)

    monkeypatch.setattr(artifact_preview, "_open_source", replaced)
    error(source, "changed")


@pytest.mark.parametrize("outside", [False, True])
def test_direct_file_symlinks_are_unavailable_even_if_target_is_inside_run(source, tmp_path, outside):
    target = (tmp_path if outside else source[1]) / "private-sentinel.txt"
    target.write_text("private-sentinel")
    source[2].unlink()
    source[2].symlink_to(target)
    error(source, "unavailable")


@pytest.mark.parametrize("outside", [False, True])
def test_registered_run_directory_symlinks_are_unavailable(source, tmp_path, outside):
    directory = (tmp_path if outside else source[0]) / "private-sentinel"
    directory.mkdir()
    (directory / "report.md").write_text("private-sentinel")
    linked = source[0] / "linked"
    linked.symlink_to(directory, target_is_directory=True)
    source[3].records["registered"]["log_dir"] = str(linked)
    error(source, "unavailable")


def test_authorized_logs_root_cannot_be_replaced_with_a_symlink(source, tmp_path):
    outside = tmp_path / "private-sentinel"
    (outside / "registered").mkdir(parents=True)
    (outside / "registered" / "report.md").write_text("private-sentinel")
    source[0].rename(source[0].with_name("previous-logs"))
    source[0].symlink_to(outside, target_is_directory=True)
    error(source, "unavailable")


@pytest.mark.parametrize("stage", ["before_request", "before_open", "during_read"])
def test_replaced_workspace_ancestor_never_authorizes_outside_files(tmp_path, monkeypatch, stage):
    workspace = tmp_path / "authorized-workspace"
    root = workspace / "logs"
    directory = root / "registered"
    directory.mkdir(parents=True)
    path = directory / "report.md"
    path.write_bytes(b"original")
    store = Store({"registered": {"id": "registered", "log_dir": str(directory)}})
    fixture = root, directory, path, store
    outside = tmp_path / "private-sentinel"
    (outside / "logs" / "registered").mkdir(parents=True)
    outside_file = outside / "logs" / "registered" / "report.md"
    outside_file.write_text("outside-private-sentinel")

    def swap():
        workspace.rename(tmp_path / "old-workspace")
        workspace.symlink_to(outside, target_is_directory=True)

    if stage == "before_request":
        swap()
        monkeypatch.setattr(artifact_preview, "_read_bytes", lambda _fd: pytest.fail("ancestor link must not be read"))
    elif stage == "before_open":
        original_open = artifact_preview._open_source
        def swapped_open(path, identities):
            swap()
            return original_open(path, identities)
        monkeypatch.setattr(artifact_preview, "_open_source", swapped_open)
        monkeypatch.setattr(artifact_preview, "_read_bytes", lambda _fd: pytest.fail("ancestor swap must not be read"))
    else:
        original_read = artifact_preview._read_bytes
        def swapped_read(descriptor):
            raw = original_read(descriptor)
            assert raw == b"original"
            swap()
            return raw
        monkeypatch.setattr(artifact_preview, "_read_bytes", swapped_read)
    error(fixture, "changed" if stage == "during_read" else "unavailable")
    assert outside_file.read_text() == "outside-private-sentinel"


def test_symlink_swap_before_open_never_reads_the_outside_target(source, monkeypatch, tmp_path):
    target = tmp_path / "private-sentinel"
    target.mkdir()
    (target / "report.md").write_text("private-sentinel")
    original_open = artifact_preview._open_source

    def swapped(path, identities):
        source[1].rename(source[1].with_name("before"))
        source[1].symlink_to(target, target_is_directory=True)
        return original_open(path, identities)

    monkeypatch.setattr(artifact_preview, "_open_source", swapped)
    monkeypatch.setattr(artifact_preview, "_read_bytes", lambda _fd: pytest.fail("symlink swap must not read outside bytes"))
    error(source, "unavailable")


@pytest.mark.parametrize("location", ["root", "outside", "relative", "missing"])
def test_registered_log_location_must_be_a_strict_child_of_the_authorized_root(source, tmp_path, location):
    locations = {"root": str(source[0]), "outside": str(tmp_path), "relative": "relative/private-sentinel", "missing": None}
    source[3].records["registered"]["log_dir"] = locations[location]
    error(source, "unavailable")


def test_nested_regular_run_directory_remains_supported(source):
    nested = source[1] / "nested"
    nested.mkdir()
    source[2].rename(nested / "report.md")
    source[3].records["registered"]["log_dir"] = str(nested)
    assert read(source)["content"] == "original"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files are a POSIX feature")
def test_fifo_is_rejected_without_opening_or_blocking(source, monkeypatch):
    source[2].unlink()
    os.mkfifo(source[2])
    monkeypatch.setattr(artifact_preview, "_open_source", lambda *_: pytest.fail("FIFO must not be opened"))
    error(source, "unavailable")


def test_filesystem_and_store_errors_do_not_expose_private_paths(source, monkeypatch):
    def denied(*_):
        raise PermissionError("private-sentinel")
    monkeypatch.setattr(artifact_preview, "_open_source", denied)
    error(source, "unavailable")
    monkeypatch.setattr(source[3], "get_record", denied)
    error(source, "unavailable")


def test_run_database_failures_use_the_same_fixed_public_error(source, monkeypatch):
    def broken_store(*_):
        raise sqlite3.OperationalError("private-sentinel")
    monkeypatch.setattr(source[3], "get_record", broken_store)
    error(source, "unavailable")


@pytest.mark.parametrize("missing", ["dir_fd", "O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"])
def test_platform_without_required_safe_open_primitives_fails_closed(source, monkeypatch, missing):
    if missing == "dir_fd":
        monkeypatch.setattr(os, "supports_dir_fd", set())
    else:
        monkeypatch.delattr(os, missing, raising=False)
    monkeypatch.setattr(artifact_preview, "_read_bytes", lambda _fd: pytest.fail("unsafe platform must not read"))
    error(source, "unavailable")


def test_authenticated_route_reads_a_registered_artifact_without_modifying_it(config):
    app = create_desktop_app(config)
    with client_for(app) as client:
        root = app.state.settings.logs_root
        directory = root / "preview-fixture"
        directory.mkdir()
        name = "报告.md"
        raw = '\ufeff# Synthetic preview\r\n实验 😀'.encode("utf-8")
        path = directory / name
        path.write_bytes(raw)
        app.state.run_store.add_record("runs", {"id": "preview-fixture", "log_dir": str(directory), "status": "complete", "managed": False, "read_only": True})
        endpoint = f"/api/desktop/runs/preview-fixture/artifacts/{name}/preview"
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers={**AUTH, "Origin": "https://private-sentinel.example"}).status_code == 403
        response = client.get(endpoint, headers=AUTH)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == {"name": name, "content": raw.decode("utf-8"), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        assert path.read_bytes() == raw
        assert set(directory.iterdir()) == {path}


def test_preview_route_returns_fixed_errors_and_never_discovers_unregistered_files(config):
    app = create_desktop_app(config)
    with client_for(app) as client:
        directory = app.state.settings.logs_root / "private-sentinel"
        directory.mkdir()
        path = directory / "report.md"
        path.write_text("private-sentinel")
        endpoint = "/api/desktop/runs/private-sentinel/artifacts/report.md/preview"
        response = client.get(endpoint, headers=AUTH)
        assert response.status_code == 404
        assert response.json()["code"] == "unavailable"
        assert "private-sentinel" not in response.text
        app.state.run_store.add_record("runs", {"id": "private-sentinel", "log_dir": str(directory), "status": "complete", "managed": False, "read_only": True})
        for raw, status, code in [(b"\xffprivate-sentinel", 422, "invalid_text"), (b"x" * (MAX_PREVIEW_BYTES + 1), 413, "too_large")]:
            path.write_bytes(raw)
            response = client.get(endpoint, headers=AUTH)
            assert response.status_code == status
            assert response.headers["Cache-Control"] == "no-store"
            assert response.json()["code"] == code
            assert "private-sentinel" not in response.text
            assert path.read_bytes() == raw
