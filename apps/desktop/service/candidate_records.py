"""Bounded public candidate display: lifecycle events + live code + journal facts."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import stat
from threading import Lock

from result_overview import _open_directory, _ReadProblem, _signature, _unique_object

LIMITS = {"candidates_live.json": 12 * 1024 * 1024, "journal.json": 12 * 1024 * 1024, "events.jsonl": 16 * 1024 * 1024}
MAX_NODES = 1500
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
MAX_CACHE_ENTRIES = 16
MAX_CACHE_BYTES = 32 * 1024 * 1024
_cache = OrderedDict()
_cache_lock = Lock()
ACTIVE = {"queued", "starting", "running", "stopping"}
PHASES = {"generating", "queued", "running", "evaluating", "valid", "failed", "interrupted", "unconfirmed"}
PHASE_ORDER = {"unconfirmed": -1, "generating": 0, "queued": 1, "running": 2, "evaluating": 3, "failed": 4}
EVENT_PHASES = {
    "node_generation_started": "generating", "node_generation_finished": "queued",
    "node_queued_for_execution": "queued", "node_submitted": "queued",
    "execution_started": "running", "candidate_validation_started": "evaluating",
    "candidate_evaluation_started": "evaluating", "execution_finished": "evaluating",
    "node_evaluated": "evaluating", "node_generation_failed": "failed",
}


def _source_signatures(directory):
    signatures = []
    for name in LIMITS:
        try:
            value = _signature(os.stat(name, dir_fd=directory, follow_symlinks=False))
        except FileNotFoundError:
            value = None
        signatures.append((name, value))
    return tuple(signatures)


def _remember(key, signature, value, size):
    # HTTP serialization treats these cached records as immutable. Bound both
    # entries and their serialized size; never retain source file contents.
    if value["state"] != "ready" or size > MAX_CACHE_BYTES:
        return
    with _cache_lock:
        _cache[key] = (signature, value, size)
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHE_ENTRIES or sum(item[2] for item in _cache.values()) > MAX_CACHE_BYTES:
            _cache.popitem(last=False)


def _read(directory, name):
    try:
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            return None, "unsafe"
        if before.st_size > LIMITS[name]:
            return None, "too_large"
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if _signature(os.fstat(stream.fileno())) != _signature(before):
                return None, "changing"
            data = stream.read(LIMITS[name] + 1)
            after = os.fstat(stream.fileno())
        if len(data) > LIMITS[name] or _signature(after) != _signature(before) or _signature(os.stat(name, dir_fd=directory, follow_symlinks=False)) != _signature(before):
            return None, "changing"
        return data, "ready"
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unreadable"


def _id(value):
    return value if isinstance(value, str) and 0 < len(value) <= 256 and not any(ord(c) < 32 for c in value) else None


def _integer(value):
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _node(raw, recorded=False):
    if not isinstance(raw, dict) or not _id(raw.get("id")):
        return None
    value = {"id": raw["id"], "step": _integer(raw.get("step")), "scheduled_step": _integer(raw.get("scheduled_step")), "recorded": recorded}
    for key in ("plan", "analysis", "code", "agent_name", "outcome", "exc_type", "selection_reason"):
        text = raw.get(key)
        if isinstance(text, str):
            limit = 256000 if key == "code" else 32000 if key in {"plan", "analysis", "selection_reason"} else 256
            value[key] = text[:limit]
            if len(text) > limit:
                value["content_truncated"] = True
    value["parent_id"] = _id(raw.get("parent_id"))
    value["phase"] = raw.get("phase") if raw.get("phase") in PHASES else "unconfirmed"
    if raw.get("content_truncated") is True:
        value["content_truncated"] = True
    if recorded:
        buggy = raw.get("is_buggy")
        value["is_buggy"] = buggy if type(buggy) is bool else None
        value["phase"] = "failed" if buggy is True else "valid" if buggy is False else "unconfirmed"
        metric = raw.get("metric")
        if isinstance(metric, dict):
            score = metric.get("value")
            if _finite(score):
                value["metric"] = {"value": score, "maximize": metric.get("maximize") if type(metric.get("maximize")) is bool else None}
        duration = raw.get("exec_time")
        if _finite(duration) and duration >= 0:
            value["exec_time"] = duration
        lines = raw.get("_term_out")
        if isinstance(lines, list):
            text = "\n".join(line for line in lines if isinstance(line, str))
            value["_term_out"] = [text[-64000:]]
            if len(text) > 64000:
                value["content_truncated"] = True
    return value


def build_candidate_records(run, *, logs_root, requested_id=None):
    result = {"nodes": [], "node2parent": {}, "total": 0, "state": "ready", "warnings": [], "revision": "", "source": "public_candidate_records"}
    raw = run.get("log_dir")
    if not raw:
        return result
    if not isinstance(raw, str):
        result.update(state="unavailable", warnings=["records_unavailable"])
        return result
    try:
        path, root = Path(raw), Path(logs_root)
        if not path.is_absolute() or ".." in path.parts or not path.relative_to(root).parts:
            raise ValueError()
        files = {}
        with _open_directory(path) as directory:
            identity = _signature(os.fstat(directory))[:3]
            signature = _source_signatures(directory)
            requested_id = _id(requested_id)
            cache_key = (str(path), str(root), identity, str(run.get("status")), requested_id)
            with _cache_lock:
                cached = _cache.get(cache_key)
                if cached is not None and cached[0] == signature:
                    with _open_directory(path) as current:
                        if _signature(os.fstat(current))[:3] == identity and _source_signatures(current) == signature:
                            _cache.move_to_end(cache_key)
                            return cached[1]
            for name in LIMITS:
                data, state = _read(directory, name)
                if data is not None:
                    files[name] = data
                elif state != "missing":
                    result["warnings"].append(name + ":" + state)
            with _open_directory(path) as current:
                if _signature(os.fstat(current))[:3] != identity or _source_signatures(current) != signature:
                    raise _ReadProblem("changing")
    except _ReadProblem as exc:
        if exc.state == "missing":
            return result
        result.update(state="unavailable", warnings=["records_unavailable"])
        return result
    except (ValueError, OSError):
        result.update(state="unavailable", warnings=["records_unavailable"])
        return result
    result["revision"] = hashlib.sha256(repr((signature, run.get("status"))).encode()).hexdigest()
    by_id = {}
    terminal_event = None
    journal_ready = False
    for name in ("candidates_live.json", "events.jsonl", "journal.json"):
        data = files.get(name)
        if data is None:
            continue
        try:
            if name == "events.jsonl":
                # Ignore only the incomplete append at EOF. A corrupt complete
                # line remains a visible warning, never a fabricated success.
                lines = data.splitlines(keepends=True)
                if lines and not lines[-1].endswith((b"\n", b"\r")):
                    lines.pop()
                for line in lines:
                    if len(line) > 256000:
                        result["warnings"].append("event_too_large")
                        continue
                    try:
                        event = json.loads(line, object_pairs_hook=_unique_object)
                    except (ValueError, UnicodeError, RecursionError):
                        result["warnings"].append("event_invalid")
                        continue
                    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                        continue
                    if event["event"] in {"run_finished", "run_failed", "run_interrupted"}:
                        terminal_event = event["event"]
                    if event["event"] not in EVENT_PHASES:
                        continue
                    node_id = _id(event.get("node_id")) or _id(event.get("placeholder_id"))
                    if not node_id:
                        continue
                    value = by_id.setdefault(node_id, {"id": node_id, "step": None, "recorded": False})
                    next_phase = EVENT_PHASES[event["event"]]
                    if event.get("exc_type") or event.get("is_buggy") is True:
                        next_phase = "failed"
                    if PHASE_ORDER.get(next_phase, -1) >= PHASE_ORDER.get(value.get("phase"), -1):
                        value["phase"] = next_phase
                    parent_id = _id(event.get("parent_id"))
                    if parent_id:
                        value["parent_id"] = parent_id
                    for key in ("selection_reason", "exc_type"):
                        if isinstance(event.get(key), str):
                            value[key] = event[key][:32000 if key == "selection_reason" else 256]
                    if _integer(event.get("scheduled_step")) is not None:
                        value["scheduled_step"] = event["scheduled_step"]
                continue
            payload = json.loads(data, parse_constant=lambda _value: None, object_pairs_hook=_unique_object)
            if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
                raise ValueError()
            seen = set()
            parsed = {}
            for raw_node in payload["nodes"][:10000]:
                value = _node(raw_node, recorded=name == "journal.json")
                if value is None:
                    continue
                if value["id"] in seen:
                    raise ValueError()
                seen.add(value["id"])
                if name == "journal.json":
                    parent = payload.get("node2parent", {})
                    if isinstance(parent, dict):
                        value["parent_id"] = _id(parent.get(value["id"]))
                previous = by_id.get(value["id"], {})
                if value.get("scheduled_step") is None:
                    value["scheduled_step"] = previous.get("scheduled_step")
                parsed[value["id"]] = {**previous, **value}
            # Reject a malformed source as a whole. A valid journal can be
            # displayed independently without leaking partially parsed live
            # rows from a snapshot that later failed duplicate/type checks.
            by_id.update(parsed)
            if name == "journal.json":
                journal_ready = True
        except (ValueError, UnicodeError, RecursionError, TypeError):
            result["warnings"].append(name + ":invalid")
    terminal = run.get("status") not in ACTIVE or terminal_event is not None
    for value in by_id.values():
        if terminal and not value.get("recorded") and value.get("phase") not in {"failed", "interrupted"}:
            value["phase"] = "interrupted" if run.get("status") in {"stopped", "cancelled", "interrupted", "failed"} or terminal_event in {"run_failed", "run_interrupted"} else "unconfirmed"
    ordered = sorted(by_id.values(), key=lambda n: (n.get("scheduled_step") if n.get("scheduled_step") is not None else n.get("step") if n.get("step") is not None else 2**53, n["id"]))
    result["total"] = len(ordered)
    result["nodes"] = ordered[:MAX_NODES]
    if requested_id in by_id and by_id[requested_id] not in result["nodes"]:
        result["nodes"].append(by_id[requested_id])
    result["node2parent"] = {n["id"]: n["parent_id"] for n in result["nodes"] if n.get("parent_id")}
    # Stay below the desktop bridge's 16 MiB response limit even when journal
    # and live snapshots each contribute different generated programs.
    estimated = len(json.dumps(result, ensure_ascii=True).encode())
    for value in sorted(result["nodes"], key=lambda n: n["id"] == requested_id):
        for field in ("_term_out", "code", "analysis", "plan", "selection_reason"):
            if estimated <= MAX_RESPONSE_BYTES:
                break
            if field in value:
                estimated -= len(json.dumps({field: value.pop(field)}, ensure_ascii=True).encode()) - 2
                if not value.get("content_truncated"):
                    value["content_truncated"] = True
                    estimated += 30
    result["warnings"] = sorted(set(result["warnings"]))
    source_failures = [w for w in result["warnings"] if w.endswith((":changing", ":invalid", ":unreadable", ":unsafe", ":too_large"))]
    journal_failed = any(w.startswith("journal.json:") for w in source_failures)
    live_failed = any(w.startswith("candidates_live.json:") for w in source_failures)
    if journal_failed or not journal_ready and (live_failed or source_failures and not result["nodes"]):
        result["state"] = "updating"
    encoded = json.dumps(result, ensure_ascii=True).encode()
    if len(encoded) > MAX_RESPONSE_BYTES:
        # A hard guard covers structural overhead as well as text. Keep the
        # explicitly requested result if a future schema expands row metadata.
        result["warnings"].append("response_truncated")
        nodes = result["nodes"]
        requested = next((n for n in nodes if n["id"] == requested_id), None)
        others = [n for n in nodes if n is not requested]
        lower, upper = 0, len(others)
        while lower < upper:
            count = (lower + upper + 1) // 2
            result["nodes"] = others[:count] + ([requested] if requested is not None else [])
            result["node2parent"] = {n["id"]: n["parent_id"] for n in result["nodes"] if n.get("parent_id")}
            if len(json.dumps(result, ensure_ascii=True).encode()) <= MAX_RESPONSE_BYTES:
                lower = count
            else:
                upper = count - 1
        result["nodes"] = others[:lower] + ([requested] if requested is not None else [])
        result["node2parent"] = {n["id"]: n["parent_id"] for n in result["nodes"] if n.get("parent_id")}
        encoded = json.dumps(result, ensure_ascii=True).encode()
        if len(encoded) > MAX_RESPONSE_BYTES:
            result.update(nodes=[], node2parent={}, state="unavailable")
            encoded = json.dumps(result, ensure_ascii=True).encode()
    _remember(cache_key, signature, result, len(encoded))
    return result
