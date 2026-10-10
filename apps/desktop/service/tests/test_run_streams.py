from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_streams
from run_streams import RunStreams, StreamError
from test_desktop_service import AUTH, client_for, config, create_desktop_app


class Store:
    def __init__(self, records):
        self.records, self.calls = records, []

    def get_record(self, table, run_id):
        self.calls.append((table, run_id))
        return self.records.get(run_id)

    def list_records(self, *_args):
        pytest.fail("stream reads must never scan the catalog")


@pytest.fixture
def source(tmp_path):
    root, state = tmp_path / "workspace" / "logs", tmp_path / "state"
    directory, console_dir = root / "registered", state / "console"
    directory.mkdir(parents=True)
    console_dir.mkdir(parents=True)
    console, events = console_dir / "registered.log", directory / "events.jsonl"
    console.write_bytes(b"original\r\n")
    events.write_bytes(b'{"event":"run_started"}\n')
    store = Store({"registered": {"id": "registered", "log_dir": str(directory), "console_log": str(console)}})
    streams = RunStreams(root, state, store)
    return streams, console, events, store


def read(source, kind="console", before=None, run_id="registered"):
    return source[0].window(run_id, kind, before)


def fail(source, code, **kwargs):
    with pytest.raises(StreamError) as raised:
        read(source, **kwargs)
    assert raised.value.code == code
    assert str(raised.value).startswith(f"[stream:{code}]")
    assert str(source[1].parent) not in str(raised.value)
    assert "private-sentinel" not in str(raised.value)
    return raised.value


def test_exact_text_empty_and_missing_are_distinct_without_writes(source):
    raw = '\ufeff原始 😀\r\nline 2\r\n\0'.encode()
    source[1].write_bytes(raw)
    before = source[1].stat()
    result = read(source)
    assert result["text"].encode() == raw
    assert result["start"] == 0 and result["end"] == result["source_size"] == len(raw)
    assert result["status"] == "ready" and not result["has_older"]
    assert not result["invalid_utf8"] and not result["partial_first_line"]
    assert result["older_cursor"] is None
    assert source[1].read_bytes() == raw and source[1].stat().st_mtime_ns == before.st_mtime_ns
    assert source[3].calls == [("runs", "registered")]
    second = read(source)
    assert second["revision"] == result["revision"]
    source[1].write_bytes(b"")
    empty = read(source)
    assert empty["status"] == "ready" and empty["source_id"] is not None
    source[1].unlink()
    missing = read(source)
    assert missing["status"] == "missing" and missing["source_id"] is None
    assert missing["text"] == "" and missing["source_size"] == 0


@pytest.mark.parametrize("kind", ["console", "activity"])
def test_new_registered_run_without_paths_is_missing_not_error(source, kind):
    source[3].records["registered"] = {"id": "registered"}
    assert read(source, kind)["status"] == "missing"
    fail(source, "unavailable", kind=kind, run_id="not-registered")


def test_console_pages_replace_without_overlap_and_allow_append(source):
    raw = b"".join(f"line {i:04d}\n".encode() for i in range(1000))
    source[1].write_bytes(raw)
    latest = read(source)
    assert len(latest["text"].splitlines()) == 400
    assert latest["text"].startswith("line 0600\n")
    with source[1].open("ab") as handle:
        handle.write(b"line 1000\n")
    earlier = read(source, before=latest["older_cursor"])
    oldest = read(source, before=earlier["older_cursor"])
    assert earlier["end"] == latest["start"] and oldest["end"] == earlier["start"]
    assert oldest["text"] + earlier["text"] + latest["text"] == raw.decode()
    assert not oldest["has_older"] and not earlier["partial_first_line"]
    assert read(source)["source_id"] == latest["source_id"]
    assert read(source)["revision"] != latest["revision"]


def test_console_byte_cap_marks_first_fragment_and_limits_reads(source, monkeypatch):
    raw = b"prefix\n" + b"z" * (run_streams.CONSOLE_BYTES * 4)
    source[1].write_bytes(raw)
    original = run_streams._pread
    reads = []
    def bounded(fd, start, length):
        reads.append(length)
        return original(fd, start, length)
    monkeypatch.setattr(run_streams, "_pread", bounded)
    result = read(source)
    assert result["text"] == "z" * run_streams.CONSOLE_BYTES
    assert result["partial_first_line"] and result["has_older"]
    assert max(reads) <= run_streams.CONSOLE_BYTES
    assert sum(reads) <= 2 * run_streams.CONSOLE_BYTES + 1024


@pytest.mark.parametrize("separator", [b"\r", b"\r\n", b"\n"])
def test_console_line_pages_preserve_cr_crlf_lf_bytes_without_overlap(source, separator):
    raw = b"".join(f"line {i:04d}".encode() + separator for i in range(1000))
    source[1].write_bytes(raw)
    latest = read(source)
    earlier = read(source, before=latest["older_cursor"])
    oldest = read(source, before=earlier["older_cursor"])
    assert oldest["text"].encode() + earlier["text"].encode() + latest["text"].encode() == raw
    assert latest["start"] == earlier["end"] and earlier["start"] == oldest["end"]
    assert all(not page["partial_first_line"] for page in [latest, earlier, oldest])


@pytest.mark.parametrize("prefix,tail,partial", [
    (b"earlier\r", b"complete\r", False),
    (b"earlier\r\n", b"complete\r\n", False),
    (b"earlier\r", b"\ncomplete\r\n", True),
    (b"earlier", b"\r\ncomplete\r\n", True),
])
def test_console_byte_boundary_distinguishes_standalone_cr_from_split_crlf(source, monkeypatch, prefix, tail, partial):
    monkeypatch.setattr(run_streams, "CONSOLE_BYTES", len(tail))
    source[1].write_bytes(prefix + tail)
    page = read(source)
    assert page["text"].encode() == tail
    assert page["partial_first_line"] is partial
    assert page["start"] == len(prefix)


def test_activity_valid_records_have_stable_ids_across_append_and_pages(source):
    source[2].write_bytes(b"".join(json.dumps({"event": "step_saved", "step": i}).encode() + b"\n" for i in range(501)))
    latest = read(source, "activity")
    assert [item["data"]["step"] for item in latest["events"]] == list(range(301, 501))
    with source[2].open("ab") as handle:
        handle.write(b'{"event":"step_saved","step":501}\n')
    refreshed = read(source, "activity")
    assert refreshed["events"][:-1] == latest["events"][1:]
    earlier = read(source, "activity", latest["older_cursor"])
    oldest = read(source, "activity", earlier["older_cursor"])
    combined = oldest["events"] + earlier["events"] + latest["events"]
    assert [item["data"]["step"] for item in combined] == list(range(501))
    assert len({item["id"] for item in combined}) == 501
    assert not oldest["has_older"]


def test_activity_skips_nonobjects_nonfinite_malformed_utf8_and_unfinished_record(source):
    source[2].write_bytes(b'\n{"event":"valid"}\nnot-json\n[]\nnull\n{"x":NaN}\n{"x":Infinity}\n{"x":-Infinity}\n{"x":1e999}\n{"x":"\xff"}\n{"event":"pending"}')
    result = read(source, "activity")
    assert [item["data"] for item in result["events"]] == [{"event": "valid"}]
    assert result["skipped_records"] == 9 and result["invalid_utf8"]
    json.dumps(result, allow_nan=False)
    with source[2].open("ab") as handle:
        handle.write(b"\n")
    assert read(source, "activity")["events"][-1]["data"]["event"] == "pending"


def test_activity_byte_limit_skips_large_record_fragment_but_keeps_latest(source, monkeypatch):
    source[2].write_bytes(b'{"large":"' + b"x" * (run_streams.ACTIVITY_BYTES * 3) + b'"}\n{"event":"latest"}\n')
    original = run_streams._pread
    lengths = []
    def bounded(fd, start, length):
        lengths.append(length)
        return original(fd, start, length)
    monkeypatch.setattr(run_streams, "_pread", bounded)
    result = read(source, "activity")
    assert result["partial_first_line"] and result["has_older"]
    assert result["skipped_records"] == 1
    assert result["events"][0]["data"] == {"event": "latest"}
    assert sum(lengths) <= 2 * run_streams.ACTIVITY_BYTES + 1024


def test_invalid_console_utf8_is_explicit_and_visible(source):
    source[1].write_bytes(b"good\n\xffbad\n")
    result = read(source)
    assert result["invalid_utf8"] and result["text"] == "good\n\ufffdbad\n"


@pytest.mark.parametrize("mutation", ["replace", "truncate", "rewrite_same_size", "remove"])
def test_cursor_refuses_rotation_truncation_and_continuity_changes(source, mutation):
    raw = b"".join(f"row {i:04d}\n".encode() for i in range(900))
    source[1].write_bytes(raw)
    cursor = read(source)["older_cursor"]
    if mutation == "replace":
        replacement = source[1].with_name("replacement")
        replacement.write_bytes(raw)
        os.replace(replacement, source[1])
    elif mutation == "truncate":
        source[1].write_bytes(raw[:5000])
    elif mutation == "rewrite_same_size":
        before = source[1].stat()
        source[1].write_bytes(b"CHANGED!" + raw[8:])
        os.utime(source[1], ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        source[1].unlink()
    assert fail(source, "changed", before=cursor).status_code == 409


def test_cursor_is_bound_to_run_kind_and_service_instance(source):
    source[1].write_bytes(b"line\n" * 500)
    cursor = read(source)["older_cursor"]
    source[3].records["other"] = source[3].records["registered"]
    fail(source, "changed", run_id="other", before=cursor)
    fail(source, "changed", kind="activity", before=cursor)
    reader = RunStreams(source[0].logs_root, source[0].state_dir, source[3])
    with pytest.raises(StreamError, match="stream:changed"):
        reader.window("registered", "console", cursor)
    assert str(source[1]).encode() not in __import__("base64").urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))


@pytest.mark.parametrize("restore_mtime", [False, True])
def test_cursor_rejects_same_size_inplace_rewrite_outside_all_anchors(source, restore_mtime):
    raw = b"".join(f"row {i:04d}\n".encode() for i in range(900))
    source[1].write_bytes(raw)
    cursor = read(source)["older_cursor"]
    decoded = source[0]._parse_cursor(cursor)
    changed_offset = 2000
    assert all(not anchor["start"] <= changed_offset < anchor["start"] + anchor["length"]
               for anchor in decoded["anchors"])
    before = source[1].stat()
    with source[1].open("r+b") as handle:
        handle.seek(changed_offset)
        handle.write(b"changed")
    if restore_mtime:
        os.utime(source[1], ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source[1].stat().st_size == len(raw)
    fail(source, "changed", before=cursor)


@pytest.mark.parametrize("run_id,before", [("../private-sentinel", None), ("x" * 181, None), ("registered", "!"), ("registered", "x" * 2049)])
def test_invalid_requests_fail_before_store_access(source, run_id, before):
    fail(source, "invalid_request", run_id=run_id, before=before)
    assert not source[3].calls


@pytest.mark.parametrize("kind", ["console", "activity"])
@pytest.mark.parametrize("mutation", ["append", "rewrite", "replace", "delete", "parent_replace"])
def test_midread_append_is_allowed_but_rewrite_or_path_change_is_rejected(source, monkeypatch, kind, mutation):
    path = source[1] if kind == "console" else source[2]
    original = run_streams._pread
    changed = False
    def changing(fd, start, length):
        nonlocal changed
        raw = original(fd, start, length)
        if not changed:
            changed = True
            if mutation == "append":
                with path.open("ab") as handle:
                    handle.write(b'{}\n')
            elif mutation == "rewrite":
                path.write_bytes(b"x" * len(path.read_bytes()))
            elif mutation == "replace":
                replacement = path.with_name("replacement")
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
            elif mutation == "delete":
                path.unlink()
            else:
                path.parent.rename(path.parent.with_name("old-parent"))
                path.parent.mkdir()
                path.write_bytes(raw)
        return raw
    monkeypatch.setattr(run_streams, "_pread", changing)
    if mutation == "append":
        assert read(source, kind)["status"] == "ready"
    else:
        fail(source, "changed", kind=kind)


@pytest.mark.parametrize("kind", ["console", "activity"])
@pytest.mark.parametrize("target_type", ["symlink_inside", "symlink_outside", "fifo", "directory"])
def test_nonregular_or_link_source_never_read(source, tmp_path, monkeypatch, kind, target_type):
    path = source[1] if kind == "console" else source[2]
    path.unlink()
    if target_type.startswith("symlink"):
        target = (path.parent if target_type == "symlink_inside" else tmp_path) / "private-sentinel"
        target.write_bytes(b"private-sentinel")
        path.symlink_to(target)
    elif target_type == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    monkeypatch.setattr(run_streams, "_pread", lambda *_: pytest.fail("unsafe file must not be read"))
    fail(source, "unavailable", kind=kind)


@pytest.mark.parametrize("kind", ["console", "activity"])
@pytest.mark.parametrize("stage", ["before_request", "before_open", "during_read"])
def test_ancestor_symlink_substitution_cannot_read_outside(source, tmp_path, monkeypatch, kind, stage):
    path = source[1] if kind == "console" else source[2]
    ancestor = source[0].state_dir if kind == "console" else source[0].logs_root.parent
    outside = tmp_path / "private-sentinel"
    relative = path.relative_to(ancestor)
    (outside / relative.parent).mkdir(parents=True)
    (outside / relative).write_bytes(b"private-sentinel")
    def swap():
        ancestor.rename(ancestor.with_name("old-ancestor"))
        ancestor.symlink_to(outside, target_is_directory=True)
    if stage == "before_request":
        swap()
        monkeypatch.setattr(run_streams, "_pread", lambda *_: pytest.fail("outside file must not be read"))
    elif stage == "before_open":
        original = run_streams._open_source
        def swapped(path, identities):
            swap()
            return original(path, identities)
        monkeypatch.setattr(run_streams, "_open_source", swapped)
        monkeypatch.setattr(run_streams, "_pread", lambda *_: pytest.fail("outside file must not be read"))
    else:
        original = run_streams._pread
        changed = False
        def swapped(fd, start, length):
            nonlocal changed
            raw = original(fd, start, length)
            assert b"private-sentinel" not in raw
            if not changed:
                changed = True
                swap()
            return raw
        monkeypatch.setattr(run_streams, "_pread", swapped)
    with pytest.raises(StreamError) as raised:
        read(source, kind)
    assert raised.value.code in {"changed", "unavailable"}


def test_console_accepts_exact_registered_log_parent_only(source, tmp_path):
    path = source[2].with_name("run.log")
    path.write_bytes(b"registered log")
    source[3].records["registered"]["console_log"] = str(path)
    assert read(source)["text"] == "registered log"
    outside = tmp_path / "private-sentinel"
    outside.write_bytes(b"private-sentinel")
    source[3].records["registered"]["console_log"] = str(outside)
    fail(source, "unavailable")
    child = source[1].parent / "nested" / "output.log"
    child.parent.mkdir()
    child.write_bytes(b"nested not registered")
    source[3].records["registered"]["console_log"] = str(child)
    fail(source, "unavailable")


def test_store_and_read_failures_use_fixed_nonleaking_errors(source, monkeypatch):
    def broken(*_):
        raise OSError(f"private-sentinel {source[1]}")
    monkeypatch.setattr(run_streams, "_pread", broken)
    fail(source, "unavailable")
    monkeypatch.setattr(source[3], "get_record", broken)
    fail(source, "unavailable", kind="activity")


def test_download_uses_exact_snapshot_hash_and_closes_descriptor(source):
    raw = "报告 😀\r\n".encode() * 20000
    source[1].write_bytes(raw)
    result = source[0].download("registered")
    with source[1].open("ab") as handle:
        handle.write(b"later append\n")
    assert result.size == len(raw)
    assert result.sha256 == hashlib.sha256(raw).hexdigest()
    assert b"".join(result.chunks()) == raw
    assert result.source.descriptor == -1
    result.close()


def test_download_refuses_large_sparse_file_without_content_read(source, monkeypatch):
    with source[1].open("wb") as handle:
        handle.truncate(run_streams.MAX_CONSOLE_DOWNLOAD_BYTES + 1)
    monkeypatch.setattr(run_streams, "_pread", lambda *_: pytest.fail("oversized download must not be read"))
    with pytest.raises(StreamError) as raised:
        source[0].download("registered")
    assert raised.value.code == "too_large" and raised.value.status_code == 413


@pytest.mark.parametrize("mutation", ["replace", "truncate", "rewrite"])
def test_download_refuses_changes_after_headers_and_closes_on_failure(source, mutation):
    result = source[0].download("registered")
    if mutation == "replace":
        replacement = source[1].with_name("replacement")
        replacement.write_bytes(source[1].read_bytes())
        os.replace(replacement, source[1])
    elif mutation == "truncate":
        source[1].write_bytes(b"")
    else:
        source[1].write_bytes(b"x" * result.size)
    with pytest.raises(StreamError, match="stream:changed"):
        b"".join(result.chunks())
    assert result.source.descriptor == -1


def test_cancelled_download_closes_descriptor(source):
    source[1].write_bytes(b"x" * 100000)
    download = source[0].download("registered")
    iterator = download.chunks()
    assert next(iterator)
    iterator.close()
    assert download.source.descriptor == -1


def test_unconsumed_download_finalizer_closes_descriptor(source):
    import gc
    download = source[0].download("registered")
    descriptor = download.source.descriptor
    del download
    gc.collect()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_desktop_routes_auth_headers_registered_reads_and_shared_compatibility(config, monkeypatch):
    app = create_desktop_app(config)
    directory = config.workspace / "logs" / "stream-route"
    console = config.home / "state" / "console" / "stream-route.log"
    directory.mkdir(parents=True)
    console.parent.mkdir(parents=True, exist_ok=True)
    raw = "控制台\r\n".encode()
    console.write_bytes(raw)
    (directory / "events.jsonl").write_bytes(b'{"event":"run_started"}\n')
    app.state.run_store.add_record("runs", {"id": "stream-route", "status": "complete", "managed": False,
                                          "read_only": True, "log_dir": str(directory), "console_log": str(console)})
    # Only terminal read-only synthetic records exist; no experiment is launched.
    with client_for(app) as client:
        for suffix in ["console", "activity", "console/download"]:
            route = f"/api/desktop/runs/stream-route/{suffix}"
            assert client.get(route).status_code == 401
            assert client.get(route, headers={**AUTH, "Origin": "https://example.org"}).status_code == 403
        original_store = app.state.run_streams.store
        guarded = Store({"stream-route": original_store.get_record("runs", "stream-route")})
        app.state.run_streams.store = guarded
        output = client.get("/api/desktop/runs/stream-route/console", headers=AUTH)
        assert output.status_code == 200 and output.headers["cache-control"] == "no-store"
        assert output.json()["text"].encode() == raw
        activity = client.get("/api/desktop/runs/stream-route/activity", headers=AUTH)
        assert activity.json()["events"][0]["data"]["event"] == "run_started"
        download = client.get("/api/desktop/runs/stream-route/console/download", headers=AUTH)
        assert download.content == raw and int(download.headers["content-length"]) == len(raw)
        assert download.headers["x-biomaker-console-sha256"] == hashlib.sha256(raw).hexdigest()
        assert download.headers["content-type"] == "application/octet-stream"
        unknown = client.get("/api/desktop/runs/private-sentinel/console", headers=AUTH)
        assert unknown.status_code == 404 and unknown.json()["code"] == "unavailable"
        assert "private-sentinel" not in unknown.text
        invalid = client.get("/api/desktop/runs/stream-route/console", params={"before": "!"}, headers=AUTH)
        assert invalid.status_code == 422 and invalid.json()["code"] == "invalid_request"
        assert any(route.path == "/api/runs/{run_id}/log" for route in app.routes)
        assert any(route.path == "/api/runs/{run_id}/events" for route in app.routes)
