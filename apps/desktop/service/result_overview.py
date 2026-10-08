"""Small, read-only summaries of recorded desktop experiment evidence.

No engine objects are imported or instantiated. Selection is an explicit saved
event, or a unique exact code match; this module never ranks metric values.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


LIMITS = {
    "journal.json": 12 * 1024 * 1024,
    "best_solution.py": 2 * 1024 * 1024,
    "results.tsv": 8 * 1024 * 1024,
    "report.md": 16 * 1024 * 1024,
    "final_retrain_result.json": 1024 * 1024,
    "events.jsonl": 4 * 1024 * 1024,
    "run_manifest.json": 1024 * 1024,
}
ARTIFACT_NAMES = tuple(LIMITS)[:5]
MAX_NODES = 10_000
MAX_EVENTS = 30_000
MAX_EVENT_BYTES = 256 * 1024
ACTIVE_STATUSES = frozenset({"queued", "starting", "running", "stopping"})


class _ReadProblem(Exception):
    def __init__(self, state: str):
        self.state = state


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _revision(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _io_state(exc: OSError) -> str:
    if exc.errno == errno.ENOENT:
        return "missing"
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return "unsafe"
    return "unreadable"


@contextmanager
def _open_directory(path: Path):
    """Anchor every component without following links, including ancestor links.

On a platform without descriptor-relative opens, fail closed. A native platform
adapter can add equivalent semantics when that platform is released.
"""
    if not path.is_absolute() or ".." in path.parts:
        raise _ReadProblem("unsafe")
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise _ReadProblem("unreadable")
    descriptor = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except OSError as exc:
        raise _ReadProblem(_io_state(exc)) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


class _Snapshot:
    def __init__(self, directory: int, path: Path):
        self.directory = directory
        self.path = path
        self.identity = _signature(os.fstat(directory))
        self.before: dict[str, tuple[int, ...] | None] = {}
        self.files: dict[str, dict[str, Any]] = {}

    def _stat(self, name: str) -> tuple[int, ...] | None:
        try:
            return _signature(os.stat(name, dir_fd=self.directory, follow_symlinks=False))
        except FileNotFoundError:
            return None

    def read(self, name: str, *, content: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"name": name, "state": "missing", "revision": None}
        self.files[name] = result
        try:
            expected = self._stat(name)
            self.before[name] = expected
            if expected is None:
                return result
            if not stat.S_ISREG(expected[2]):
                result["state"] = "unsafe"
                return result
            result["revision"] = _revision(expected)
            if expected[3] > LIMITS[name]:
                result["state"] = "too_large"
                return result
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            descriptor = os.open(name, flags, dir_fd=self.directory)
            with os.fdopen(descriptor, "rb") as stream:
                if _signature(os.fstat(stream.fileno())) != expected:
                    result["state"] = "changing"
                    return result
                data = stream.read(LIMITS[name] + 1) if content else None
                if (data is not None and len(data) > LIMITS[name]) or _signature(os.fstat(stream.fileno())) != expected or self._stat(name) != expected:
                    result["state"] = "changing"
                    return result
            result["state"] = "available"
            if content:
                result["data"] = data
            return result
        except OSError as exc:
            result["state"] = _io_state(exc)
            return result

    def stable(self) -> bool:
        try:
            if _signature(os.fstat(self.directory)) != self.identity:
                return False
            # Reopen the original pathname to detect a renamed/replaced directory
            # or ancestor; all file access above stayed anchored to the old fd.
            with _open_directory(self.path) as current:
                if _signature(os.fstat(current)) != self.identity:
                    return False
            return all(self._stat(name) == expected for name, expected in self.before.items())
        except (OSError, _ReadProblem):
            return False


def _json_object(file: dict[str, Any], *, allow_nonfinite: bool = False) -> dict[str, Any] | None:
    if file["state"] != "available":
        return None
    try:
        # Old Python journals can contain NaN/Infinity for failed candidates.
        # They stay inert data and are never emitted as HTTP metric values.
        def constant(token):
            if allow_nonfinite:
                return None
            raise ValueError(token)
        value = json.loads(file["data"].decode("utf-8"), parse_constant=constant, object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value
    except (ValueError, UnicodeError, RecursionError):
        file["state"] = "invalid"
        return None


def _text(file: dict[str, Any]) -> str | None:
    if file["state"] != "available":
        return None
    try:
        return file["data"].decode("utf-8")
    except UnicodeError:
        file["state"] = "invalid"
        return None


def _events(file: dict[str, Any]) -> list[dict[str, Any]] | None:
    if file["state"] != "available":
        return None
    try:
        if file["data"].count(b"\n") > MAX_EVENTS:
            file["state"] = "too_large"
            return None
        lines = file["data"].splitlines()
        if len(lines) > MAX_EVENTS:
            file["state"] = "too_large"
            return None
        result = []
        for line in lines:
            if not line.strip():
                continue
            if len(line) > MAX_EVENT_BYTES:
                file["state"] = "too_large"
                return None
            value = json.loads(line.decode("utf-8"), parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
            if not isinstance(value, dict) or not isinstance(value.get("event"), str):
                raise ValueError("invalid event")
            result.append(value)
        return result
    except (ValueError, UnicodeError, RecursionError):
        # Do not skip a corrupt newer selection and silently trust an older one.
        file["state"] = "invalid"
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(token):
    raise ValueError(token)


def _safe_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not 0 < len(value) <= limit or any(ord(ch) < 32 for ch in value):
        return None
    try:
        value.encode("utf-8")
    except UnicodeError:
        return None
    return value


def _identifier(value: Any) -> str | None:
    return _safe_text(value, 256)


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


def _metric(node: dict[str, Any]) -> dict[str, Any] | None:
    raw = node.get("metric")
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if type(value) not in {int, float}:
        return None
    try:
        if not math.isfinite(value):
            return None
    except OverflowError:
        return None
    result = {"value": value, "maximize": raw.get("maximize") if type(raw.get("maximize")) is bool else None, "source": "journal"}
    name = raw.get("name")
    if _safe_text(name, 120) is not None:
        result["name"] = name
    return result


def _empty_selection(reason: str, status: str = "unavailable") -> dict[str, Any]:
    return {"status": status, "source": None, "node_id": None, "candidate_step": None,
            "metric": None, "code_match": None, "reason": reason}


def _selection(journal: dict[str, Any] | None, events: list[dict[str, Any]] | None,
               code: str | None, files: dict[str, dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    selected = next((event for event in reversed(events or []) if event["event"] == "step_saved"), None)
    if journal is None:
        return _empty_selection("journal_invalid" if files["journal.json"]["state"] == "invalid" else "journal_unavailable")
    nodes = journal.get("nodes")
    if not isinstance(nodes, list) or len(nodes) > MAX_NODES or any(not isinstance(node, dict) or _identifier(node.get("id")) is None for node in nodes):
        files["journal.json"]["state"] = "invalid" if not isinstance(nodes, list) or len(nodes) <= MAX_NODES else "too_large"
        return _empty_selection("journal_invalid")
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        return _empty_selection("duplicate_candidate_ids", "conflict")
    good = [node for node in nodes if node.get("is_buggy") is False]
    if selected is not None:
        if "best_node_id" not in selected or (selected["best_node_id"] is not None and _identifier(selected["best_node_id"]) is None):
            return _empty_selection("selection_event_invalid", "conflict")
        identifier = selected["best_node_id"]
        result = _empty_selection("recorded_no_selection", "none")
        result.update(source="step_saved", node_id=identifier)
        if not nodes or selected.get("node_id") != nodes[-1]["id"]:
            result.update(status="conflict", reason="selection_snapshot_mismatch")
            return result
        if identifier is None:
            if good or code:
                result.update(status="conflict", reason="selection_snapshot_mismatch")
            return result
        node = by_id.get(identifier)
        if node is None:
            result.update(status="conflict", reason="selected_candidate_missing")
            return result
        if node.get("is_buggy") is not False:
            result.update(status="conflict", reason="selected_candidate_invalid")
            return result
        result.update(status="recorded", candidate_step=_nonnegative_int(node.get("step")), metric=_metric(node))
        result.pop("reason", None)
        if not isinstance(node.get("code"), str) or code is None:
            warnings.append("selection_code_unverified")
        else:
            result["code_match"] = node["code"] == code
            if not result["code_match"]:
                result.update(status="conflict", reason="selected_code_mismatch")
        if result["metric"] is None:
            warnings.append("metric_unavailable")
        return result
    if not good:
        return _empty_selection("no_valid_candidates", "none")
    if code is None or not code:
        return _empty_selection("best_code_unavailable")
    matches = [node for node in good if node.get("code") == code]
    if len(matches) != 1:
        return _empty_selection("best_code_ambiguous" if matches else "best_code_no_match")
    node = matches[0]
    result = {"status": "code_match", "source": "best_solution.py", "node_id": node["id"],
              "candidate_step": _nonnegative_int(node.get("step")), "metric": _metric(node), "code_match": True}
    if result["metric"] is None:
        warnings.append("metric_unavailable")
    return result


def _retrain(receipt: dict[str, Any] | None, manifest: dict[str, Any] | None,
             events: list[dict[str, Any]] | None, selection: dict[str, Any],
             files: dict[str, dict[str, Any]], *, origin: str, run_status: str) -> dict[str, Any]:
    result: dict[str, Any] = {"state": "unknown", "source": None, "node_id": None, "reason": "no_retrain_record"}
    # A present but unreadable/incomplete event stream may contain a newer
    # attempt start. The identity-free receipt cannot disprove that possibility.
    # An actually missing event file is different: receipt-only history remains
    # displayable, without inventing a candidate or attempt association.
    if events is None and files["events.jsonl"]["state"] != "missing":
        result.update(source="events.jsonl", reason="events_unavailable")
        return result
    relevant = [event for event in events or [] if event["event"] in {"final_retrain_started", "final_retrain_finished", "final_retrain_skipped"}]
    started = next((event for event in reversed(relevant) if event["event"] == "final_retrain_started"), None)
    last = relevant[-1] if relevant else None
    result["node_id"] = _identifier(started.get("node_id")) if started else None
    config = manifest.get("config") if manifest and isinstance(manifest.get("config"), dict) else {}
    disabled = config.get("final_retrain_enabled") is False
    if receipt is not None:
        if type(receipt.get("ok")) is not bool:
            files["final_retrain_result.json"]["state"] = "invalid"
            result.update(source="final_retrain_result.json", reason="receipt_invalid")
            return result
        conflict = disabled or (last is not None and last["event"] == "final_retrain_skipped")
        if receipt["ok"] and isinstance(receipt.get("error"), str) and receipt["error"].strip():
            conflict = True
        # These are formal FinalRetrainResult/ExecutionResult fields, not free
        # form report prose. The engine cannot succeed with either failure set.
        execution = receipt.get("execution")
        if receipt["ok"] and (receipt.get("exc_type") is not None or
                              isinstance(execution, dict) and execution.get("final_failure") is not None):
            conflict = True
        if last and last["event"] == "final_retrain_finished":
            conflict = conflict or type(last.get("ok")) is not bool or last["ok"] != receipt["ok"]
        if result["node_id"] is not None and selection["status"] == "recorded":
            conflict = conflict or result["node_id"] != selection["node_id"]
        if conflict:
            result.update(source="final_retrain_result.json", reason="conflicting_retrain_records")
            return result
        # The receipt has no candidate ID or attempt ID. A newer start can refer
        # to an attempt whose receipt has not replaced the old one yet.
        if last is not None and last["event"] == "final_retrain_started":
            result.update(source="events.jsonl", reason="final_retrain_in_progress" if run_status in ACTIVE_STATUSES else "final_retrain_incomplete")
            return result
        result.update(state="succeeded" if receipt["ok"] else "failed", source="final_retrain_result.json")
        result.pop("reason", None)
        for key in ("train_rows_before", "train_rows_after", "score_rows_added"):
            value = _nonnegative_int(receipt.get(key))
            if value is not None:
                result[key] = value
        return result
    if files["final_retrain_result.json"]["state"] == "invalid":
        result.update(source="final_retrain_result.json", reason="receipt_invalid")
    elif last is not None:
        result["source"] = "events.jsonl"
        if disabled:
            result["reason"] = "conflicting_retrain_records"
        elif last["event"] == "final_retrain_finished" and type(last.get("ok")) is bool:
            result.update(state="recorded_without_receipt", reason="imported_receipt_not_included" if origin == "imported_snapshot" else "receipt_unavailable")
        elif last["event"] == "final_retrain_started":
            result["reason"] = "final_retrain_in_progress" if run_status in ACTIVE_STATUSES else "final_retrain_incomplete"
        else:
            result["reason"] = "final_retrain_incomplete"
    elif files["final_retrain_result.json"]["state"] != "missing":
        result["reason"] = "receipt_unavailable"
    elif disabled:
        result.update(state="disabled", source="run_manifest.json", reason="explicit_disabled")
    elif origin == "imported_snapshot":
        result["reason"] = "imported_receipt_not_included"
    return result


def _base(run: dict[str, Any]) -> dict[str, Any]:
    origin = "imported_snapshot" if run.get("source") == "desktop_import" or run.get("history_import") else "historical" if run.get("managed") is False or run.get("read_only") is True else "managed"
    return {"run_id": str(run.get("id", "")), "origin": origin, "run_status": str(run.get("status") or "unknown"),
            "snapshot": {"status": "unavailable", "revision": None},
            "selection": _empty_selection("journal_unavailable"),
            "final_retrain": {"state": "unknown", "source": None, "node_id": None, "reason": "receipt_unavailable"},
            "artifacts": [{"name": name, "state": "missing", "revision": None} for name in ARTIFACT_NAMES],
            "warnings": ["history_snapshot"] if origin != "managed" else []}


def build_result_overview(run: dict[str, Any], *, logs_root: Path) -> dict[str, Any]:
    """Read fixed evidence files from a run already authorized by the service.

``run['log_dir']`` must come from the trusted run catalog, never a client path.
Missing/malformed/oversized evidence yields explicit partial availability. A
changed snapshot suppresses every cross-file interpretation, without retries.
"""
    result = _base(run)
    raw_path = run.get("log_dir")
    if not isinstance(raw_path, str) or not raw_path:
        result["warnings"].append("log_directory_unavailable")
        return result
    path = Path(raw_path)
    root = Path(logs_root)
    try:
        relative = path.relative_to(root)
        allowed = root.is_absolute() and path != root and ".." not in path.parts and ".." not in root.parts and bool(relative.parts)
    except ValueError:
        allowed = False
    if not allowed:
        for artifact in result["artifacts"]:
            artifact["state"] = "unsafe"
        result["warnings"].append("log_directory_unsafe")
        return result
    try:
        with _open_directory(path) as directory:
            snapshot = _Snapshot(directory, path)
            files = {name: snapshot.read(name, content=name not in {"report.md", "results.tsv"}) for name in LIMITS}
            journal = _json_object(files["journal.json"], allow_nonfinite=True)
            receipt = _json_object(files["final_retrain_result.json"])
            manifest = _json_object(files["run_manifest.json"])
            events = _events(files["events.jsonl"])
            code = _text(files["best_solution.py"])
            result["selection"] = _selection(journal, events, code, files, result["warnings"])
            result["final_retrain"] = _retrain(receipt, manifest, events, result["selection"], files,
                                                 origin=result["origin"], run_status=result["run_status"])
            stable = snapshot.stable() and not any(file["state"] == "changing" for file in files.values())
            revision_input = {"run_status": result["run_status"], "origin": result["origin"],
                              "files": {name: {"state": file["state"], "revision": file["revision"]} for name, file in files.items()}}
            result["snapshot"] = {"status": "stable" if stable else "changing", "revision": _revision(revision_input) if stable else None}
            if not stable:
                result["selection"] = _empty_selection("snapshot_changed")
                result["final_retrain"] = {"state": "unknown", "source": None, "node_id": None, "reason": "snapshot_changed"}
                result["warnings"].append("snapshot_changed")
            result["artifacts"] = [{key: file[key] for key in ("name", "state", "revision")} for name in ARTIFACT_NAMES for file in [files[name]]]
            for name, file in files.items():
                if file["state"] != "available":
                    result["warnings"].append(name.split(".")[0] + "_" + file["state"])
            return result
    except (_ReadProblem, OSError) as exc:
        state = exc.state if isinstance(exc, _ReadProblem) else _io_state(exc)
        for artifact in result["artifacts"]:
            artifact["state"] = state
        result["warnings"].append("log_directory_" + state)
        return result
