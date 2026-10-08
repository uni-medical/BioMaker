"""Bounded directory metadata scanner. Never opens dataset file contents."""
from __future__ import annotations

import errno
import json
import os
import stat
import sys
import threading
import time
import unicodedata
from pathlib import Path

MAX_INPUT_BYTES = 524_288
MAX_OUTPUT_BYTES = 32_768
MAX_ENTRIES = 2000
MAX_DEPTH = 3
MAX_DISPLAY_ENTRIES = 10
MAX_EXTENSIONS = 20
MAX_NAME = 160
MAX_EXTENSION = 32
MAX_NUMBER = 2**53 - 1
SOFT_SECONDS = 2.0
HARD_SECONDS = 5.0
CODES = frozenset({"ok", "empty", "not_found", "not_directory", "permission", "unauthorized", "invalid_request", "timeout", "busy", "internal"})
WARNINGS = frozenset({"entry_limit", "depth_limit", "soft_timeout", "hard_timeout", "descendant_unreadable", "symlinks_not_followed", "changed_during_scan", "admission_short_circuit", "root_symlink_resolved", "name_truncated", "extension_limit", "size_limit"})


def report(code="ok"):
    return {"status": "ok" if code == "ok" else "error", "code": code,
            "complete": False, "files": 0, "directories": 0, "symlinks": 0,
            "other": 0, "bytes": 0, "extensions": [], "entries": [], "warnings": []}


def valid_path(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 4096
            and not any(ord(c) < 32 or ord(c) == 127 for c in value)
            and len(value.encode("utf-8", errors="replace")) <= 4096)


def validate_payload(payload, mode):
    if not isinstance(payload, dict) or set(payload) - {"path", "authorized_roots"}:
        return "invalid_request"
    if not valid_path(payload.get("path")):
        return "invalid_request"
    if mode == "overview" and not os.path.isabs(payload["path"]):
        return "invalid_request"
    roots = payload.get("authorized_roots", [])
    if not isinstance(roots, list) or len(roots) > 128 or any(not valid_path(root) or not os.path.isabs(root) for root in roots):
        return "invalid_request"
    if mode == "overview" and not roots:
        return "unauthorized"
    return None


def watch_owner(parent_pid, timeout):
    deadline = time.monotonic() + min(max(timeout, 0.05), HARD_SECONDS)
    while True:
        if os.name == "posix" and os.getppid() != parent_pid:
            os._exit(70)
        if time.monotonic() >= deadline:
            os._exit(71)
        time.sleep(0.025)


def emit(kind, value):
    encoded = json.dumps({"kind": kind, "report": value}, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(encoded) > MAX_OUTPUT_BYTES // 2:
        encoded = json.dumps({"kind": "final", "report": report("internal")}).encode("ascii") + b"\n"
    # Exactly one root checkpoint and one final packet; never provider/file data.
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


class ScanFailure(Exception):
    pass


def error_code(error):
    if isinstance(error, PermissionError) or getattr(error, "errno", None) in {errno.EACCES, errno.EPERM}:
        return "permission"
    if isinstance(error, FileNotFoundError):
        return "not_found"
    if isinstance(error, (NotADirectoryError, RuntimeError)) or getattr(error, "errno", None) == errno.ELOOP:
        return "not_directory"
    return "internal"


def open_directory_chain(path):
    """Pin every canonical path component without following replacement links."""
    if os.name != "posix" or os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd:
        raise ScanFailure("internal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    current = os.open("/", flags)
    try:
        for component in Path(path).parts[1:]:
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def scan(payload, *, mode="overview", workspace=None, checkpoint=None, soft_seconds=SOFT_SECONDS):
    invalid = validate_payload(payload, mode)
    if invalid:
        return report(invalid)
    started = time.monotonic()
    result = report()
    warnings = set()
    histogram = {}
    seen = 0
    complete = True
    stop = False

    def warn(code, *, partial=False):
        nonlocal complete
        warnings.add(code)
        if partial:
            complete = False

    def expired():
        return time.monotonic() - started >= min(max(soft_seconds, 0), SOFT_SECONDS)

    def finish():
        result["complete"] = complete
        result["warnings"] = sorted(warnings)
        result["extensions"] = [{"extension": extension, "count": count} for extension, count in sorted(histogram.items(), key=lambda item: (-item[1], item[0]))]
        if complete and not any(result[key] for key in ("files", "symlinks", "other")):
            result.update(status="error", code="empty")
        return result

    def safe_name(name):
        cleaned = "".join("�" if unicodedata.category(char) in {"Cc", "Cf", "Cs"} else char for char in name)
        if cleaned != name or len(cleaned) > MAX_NAME:
            warn("name_truncated")
        return cleaned[:MAX_NAME]

    def add_size(size):
        amount = max(0, min(int(size), MAX_NUMBER))
        total = result["bytes"] + amount
        if size > MAX_NUMBER or total > MAX_NUMBER:
            warn("size_limit")
        result["bytes"] = min(total, MAX_NUMBER)
        return amount

    def walk(descriptor, depth, iterator, first=None):
        nonlocal seen, stop
        pending = first
        while not stop:
            if expired():
                warn("soft_timeout", partial=True)
                stop = True
                break
            if seen >= MAX_ENTRIES:
                warn("entry_limit", partial=True)
                stop = True
                break
            try:
                entry = pending if pending is not None else next(iterator, None)
                pending = None
            except OSError:
                warn("descendant_unreadable", partial=True)
                break
            if entry is None:
                break
            seen += 1
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                warn("changed_during_scan" if isinstance(error, FileNotFoundError) else "descendant_unreadable", partial=True)
                if depth == 0 and len(result["entries"]) < MAX_DISPLAY_ENTRIES:
                    result["entries"].append({"name": safe_name(entry.name), "kind": "other", "bytes": None})
                continue
            kind, size = "other", None
            if stat.S_ISLNK(metadata.st_mode):
                kind = "symlink"
                result["symlinks"] += 1
                warn("symlinks_not_followed")
            elif stat.S_ISDIR(metadata.st_mode):
                kind = "directory"
                result["directories"] += 1
            elif stat.S_ISREG(metadata.st_mode):
                kind = "file"
                result["files"] += 1
                size = add_size(metadata.st_size)
                suffixes = Path(entry.name).suffixes
                extension = ("".join(suffixes[-2:]) if suffixes and suffixes[-1].lower() in {".gz", ".bz2", ".xz", ".zst", ".zip", ".z"} else Path(entry.name).suffix).lower()
                if len(extension) > MAX_EXTENSION or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in extension):
                    warn("extension_limit")
                elif extension in histogram or len(histogram) < MAX_EXTENSIONS:
                    histogram[extension] = histogram.get(extension, 0) + 1
                else:
                    warn("extension_limit")
            else:
                result["other"] += 1
            if depth == 0 and len(result["entries"]) < MAX_DISPLAY_ENTRIES:
                result["entries"].append({"name": safe_name(entry.name), "kind": kind, "bytes": size})
            if mode == "admission" and kind != "directory":
                warn("admission_short_circuit", partial=True)
                stop = True
                break
            if kind != "directory":
                continue
            if depth >= MAX_DEPTH:
                warn("depth_limit", partial=True)
                continue
            child = None
            try:
                child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=descriptor)
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    warn("changed_during_scan", partial=True)
                    continue
                with os.scandir(child) as children:
                    walk(child, depth + 1, children)
            except OSError as error:
                warn("changed_during_scan" if getattr(error, "errno", None) in {errno.ENOENT, errno.ELOOP, errno.ENOTDIR} else "descendant_unreadable", partial=True)
            finally:
                if child is not None:
                    os.close(child)

    descriptor = None
    try:
        raw = os.path.expanduser(payload["path"])
        if not os.path.isabs(raw):
            raw = os.path.join(workspace or os.getcwd(), raw)
        original = os.path.abspath(raw)
        # All resolution and directory I/O runs in this isolated worker.
        canonical = str(Path(original).resolve(strict=True))
        if mode == "overview":
            authorized = False
            # workspace is supplied only by the owning desktop service, never
            # from the renderer's body. It already belongs to DesktopConfig.
            grants = [*payload.get("authorized_roots", []), *([workspace] if workspace else [])]
            for root in grants:
                grant = os.path.abspath(root)
                # Do not touch disconnected network grants unrelated to this
                # canonical candidate. Picker grants were canonical when saved.
                if os.path.commonpath((canonical, grant)) != grant:
                    continue
                if expired():
                    raise ScanFailure("timeout")
                try:
                    resolved_grant = str(Path(grant).resolve(strict=True))
                    # Picker grants are canonical when saved. Retargeting a
                    # saved grant to a new symlink cannot authorize that target.
                    if resolved_grant == grant and os.path.commonpath((canonical, grant)) == grant:
                        authorized = True
                        break
                except (OSError, ValueError, RuntimeError):
                    continue
            if not authorized:
                raise ScanFailure("unauthorized")
        if expired():
            raise ScanFailure("timeout")
        descriptor = open_directory_chain(canonical)
        if original != canonical:
            warn("root_symlink_resolved")
        with os.scandir(descriptor) as entries:
            # A real enumeration step establishes root accessibility. Opening a
            # directory alone does not establish that a remote readdir works.
            first = next(entries, None)
            if checkpoint:
                current = report()
                current["warnings"] = sorted(warnings)
                checkpoint(current)
            if first is not None:
                # Root directory depth is zero. Enumerate descendants through
                # directory depth three; never enter a depth-four directory.
                walk(descriptor, 0, entries, first)
        return finish()
    except ScanFailure as error:
        return report(str(error) if str(error) in CODES else "internal")
    except (OSError, RuntimeError, ValueError) as error:
        return report(error_code(error))
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main():
    try:
        owner, lifetime = int(sys.argv[1]), min(float(sys.argv[2]), HARD_SECONDS)
    except (IndexError, ValueError):
        os._exit(72)
    threading.Thread(target=watch_owner, args=(owner, lifetime), daemon=True).start()
    output = report("internal")
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) <= MAX_INPUT_BYTES:
            packet = json.loads(raw)
            output = scan(packet["payload"], mode=packet["mode"], workspace=packet.get("workspace"), checkpoint=lambda value: emit("checkpoint", value))
    except BaseException:
        pass
    emit("final", output)


if __name__ == "__main__":
    main()
