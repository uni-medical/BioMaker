from collections import deque
import io
import json
import math
from pathlib import Path
import random

import pytest

import webui.event_reader as reader


def legacy_oracle(raw: bytes, tail: int):
    """The pre-optimization text iterator, independent of the reverse reader."""
    def event_float(value):
        number = float(value)
        return number if math.isfinite(number) else ("Infinity" if number > 0 else "-Infinity")

    parsed = deque(maxlen=max(0, tail))
    count = 0
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            try:
                event = json.loads(line, parse_constant=str, parse_float=event_float)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                parsed.append(event)
    return count, list(parsed)


def assert_same(left, right):
    # Serialization also preserves negative zero and escaped lone surrogates.
    assert json.dumps(left, ensure_ascii=True, allow_nan=False) == json.dumps(
        right, ensure_ascii=True, allow_nan=False
    )


@pytest.mark.parametrize("block_size", [1, 2, 3, 7, 64, 65536])
@pytest.mark.parametrize("tail", [-1, 0, 1, 3, 5000])
def test_text_iterator_oracle_for_newlines_utf8_and_json(tmp_path, monkeypatch, block_size, tail):
    monkeypatch.setattr(reader, "_BLOCK_BYTES", block_size)
    raw = (
        b'\r\n \t\r' + '\u00a0\u0085\u2028\u2029'.encode() + b'\n'
        b'\xef\xbb\xbf{}\r\nnot-json\r[]\nnull\rfalse\n42\r"text"\n'
        b'{}\r\n{"event":"first","text":"' + '\u4e2d\U0001f9ec\u2028\u2029'.encode() + b'"}\r'
        b'{"event":"replacement","text":"\xe2\x82\xff"}\n'
        b'{"event":"diagnostic","a":NaN,"b":Infinity,"c":-Infinity,'
        b'"nested":[1e999,-1e999,-0.0,1.25,"NaN",null,true]}\r\n'
        b'{"event":"last","surrogate":"\\ud800"}\rbroken-tail'
    )
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)
    expected = legacy_oracle(raw, tail)
    assert_same(reader.read_event_window(path, tail=tail, count_all=True), expected)
    assert_same(reader.read_event_window(path, tail=tail, count_all=False), (None, expected[1]))
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw", [b"", b"\r", b"\n", b"\r\n", b"{}", b"{}\r", b"{}\n", b"{}\r\n"])
def test_empty_and_unterminated_files(tmp_path, raw):
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)
    assert_same(reader.read_event_window(path, tail=2, count_all=True), legacy_oracle(raw, 2))


def test_seeded_mixed_histories_match_successful_legacy_oracle(tmp_path, monkeypatch):
    rng = random.Random(87219)
    path = tmp_path / "events.jsonl"
    tokens = [b"", b" \t", '\u00a0'.encode(), b"[]", b"0", b"null", b"bad-json", b"\xff", b"{}",
              b'{"event":"done","metric":NaN}', b'{"raw":"\xff"}', b'{"nested":{"x":[1,2,true]}}']
    breaks = [b"\r", b"\n", b"\r\n"]
    for _ in range(60):
        raw = b"".join(rng.choice(tokens) + rng.choice(breaks) for _ in range(rng.randrange(50)))
        raw += rng.choice(tokens)
        path.write_bytes(raw)
        monkeypatch.setattr(reader, "_BLOCK_BYTES", rng.choice([1, 2, 7, 31, 65536]))
        for tail in [0, 1, 7, 5000]:
            expected = legacy_oracle(raw, tail)
            assert_same(reader.read_event_window(path, tail=tail, count_all=True), expected)
            assert_same(reader.read_event_window(path, tail=tail, count_all=False), (None, expected[1]))


def test_last_5000_valid_objects_not_last_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    raw = b'{"event":"run_finished"}\n' + b"".join(
        json.dumps({"event": "observation", "i": i}).encode() + b"\r\n[]\rbad\n \t\r"
        for i in range(5000)
    )
    path.write_bytes(raw)
    count, events = reader.read_event_window(path, tail=5000, count_all=True)
    assert count == 15001
    assert [x["i"] for x in events] == list(range(5000))
    assert_same(reader.read_event_window(path, tail=5000, count_all=False), (None, events))


def test_large_multiblock_record_and_invalid_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "_BLOCK_BYTES", 257)
    path = tmp_path / "events.jsonl"
    event = {"event": "large", "message": '\u4e2d\U0001f9ec' * 40000}
    raw = b'{}\r\n' + json.dumps(event, ensure_ascii=False).encode() + b'\r\nbroken' * 1000
    path.write_bytes(raw)
    assert_same(reader.read_event_window(path, tail=1, count_all=True), legacy_oracle(raw, 1))


def test_resource_pathological_json_is_counted_and_skipped(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    giant_integer = b'{"number":' + b'9' * 10000 + b'}'
    deep = b'{"deep":' + b'[' * 5000 + b'0' + b']' * 5000 + b'}'
    raw = b'{"event":"old"}\n' + giant_integer + b'\r' + deep + b'\r\n{"event":"new"}'
    path.write_bytes(raw)
    original = reader.json.loads
    def parsed(line, **kwargs):
        # Recent Python runtimes can decode deeply nested input iteratively;
        # model the RecursionError still raised by supported older runtimes.
        if line == deep.decode():
            raise RecursionError("synthetic decoder nesting limit")
        return original(line, **kwargs)
    monkeypatch.setattr(reader.json, "loads", parsed)
    assert reader.read_event_window(path, tail=3, count_all=True) == (4, [{"event": "old"}, {"event": "new"}])
    assert reader.read_event_window(path, tail=3, count_all=False) == (None, [{"event": "old"}, {"event": "new"}])


class TrackingFile:
    def __init__(self, handle, reads, on_first_read=None):
        self.handle, self.reads, self.on_first_read = handle, reads, on_first_read

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def fileno(self):
        return self.handle.fileno()

    def seek(self, *args):
        return self.handle.seek(*args)

    def read(self, size):
        if self.on_first_read:
            callback, self.on_first_read = self.on_first_read, None
            callback()
        value = self.handle.read(size)
        self.reads.append(len(value))
        return value


def tracked_opens(monkeypatch, path, reads, on_first_read=None):
    original = Path.open
    def opened(candidate, *args, **kwargs):
        handle = original(candidate, *args, **kwargs)
        return TrackingFile(handle, reads, on_first_read) if candidate == path and args == ("rb",) else handle
    monkeypatch.setattr(Path, "open", opened)


def test_catalog_stops_early_legacy_counts_all_both_parse_only_tail(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    raw = b'{"event":"history"}\n' * 12000
    path.write_bytes(raw)
    reads, parses = [], []
    tracked_opens(monkeypatch, path, reads)
    original = reader.json.loads
    def parsed(line, **kwargs):
        parses.append(line)
        return original(line, **kwargs)
    monkeypatch.setattr(reader.json, "loads", parsed)
    assert len(reader.read_event_window(path, tail=5000, count_all=False)[1]) == 5000
    assert sum(reads) < len(raw) and len(parses) == 5000
    reads.clear(); parses.clear()
    assert reader.read_event_window(path, tail=1, count_all=True) == (12000, [{"event": "history"}])
    assert sum(reads) == len(raw) and len(parses) == 1
    reads.clear(); parses.clear()
    assert reader.read_event_window(path, tail=0, count_all=True) == (12000, [])
    assert sum(reads) == len(raw) and parses == []


def test_append_does_not_extend_initial_descriptor_extent(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    raw = b'{"event":"original"}\n' * 100
    path.write_bytes(raw)
    reads = []
    def append():
        with path.open("ab") as handle:
            handle.write(b'{"event":"appended"}\n')
    tracked_opens(monkeypatch, path, reads, append)
    assert reader.read_event_window(path, tail=1, count_all=True) == (100, [{"event": "original"}])
    assert sum(reads) == len(raw)


def test_replacement_does_not_switch_open_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"old"}\n')
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b'{"event":"new"}\n')
    tracked_opens(monkeypatch, path, [], lambda: replacement.replace(path))
    assert reader.read_event_window(path, tail=1, count_all=True) == (1, [{"event": "old"}])


def test_truncation_short_read_raises_io_error(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"old"}\n')
    tracked_opens(monkeypatch, path, [], lambda: path.write_bytes(b""))
    with pytest.raises(OSError, match="changed while reading"):
        reader.read_event_window(path, tail=1, count_all=True)


@pytest.mark.parametrize("strict", [False, True])
def test_artifact_wrappers_preserve_missing_and_io_policy(settings, monkeypatch, strict):
    from webui.artifacts import read_events, read_event_tail
    path = settings.logs_root / "event-policy"
    path.mkdir()
    run = {"log_dir": str(path)}
    assert read_events(settings, run, strict=strict) == {"count": 0, "events": []}
    assert read_event_tail(settings, run, strict=strict) == []
    event_file = path / "events.jsonl"
    event_file.write_text('{"event":"present"}\n')
    original = Path.open
    def opened(candidate, *args, **kwargs):
        if candidate == event_file:
            raise PermissionError("synthetic read failure")
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    for operation in [read_events, read_event_tail]:
        if strict:
            with pytest.raises(PermissionError, match="synthetic read failure"):
                operation(settings, run, strict=True)
        else:
            assert operation(settings, run) == ({"count": 0, "events": []} if operation is read_events else [])
