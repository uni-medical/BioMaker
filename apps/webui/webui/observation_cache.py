"""Process-owned, byte-bounded caches of file observations, never live ledger state.

Every cache hit validates metadata again. Any changed file is read in full;
this module makes no append-only assumption. Cold reads and API responses keep
the existing parser's semantics, including its potentially large record size.
The limits below bound retained cache data, not all transient response memory.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import threading
from typing import Any, Callable

from .artifacts import list_artifacts, read_observed_summary
from .run_catalog import run_from_log_dir
from .settings import Settings


def _signature(value) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class _Entry:
    stamp: tuple
    payload: bytes
    extra: str | None
    weight: int


class ObservationCache:
    def __init__(self, settings: Settings, *, max_entries: int = 512,
                 max_bytes: int = 8 * 1024 * 1024,
                 max_item_bytes: int = 512 * 1024):
        self.settings = settings
        self.max_entries = max(0, max_entries)
        self.max_bytes = max(0, max_bytes)
        self.max_item_bytes = max(0, max_item_bytes)
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._bytes = 0
        self._epoch = 0
        self._lock = threading.RLock()
        # Fixed stripes avoid an unbounded per-path lock table. Only histories
        # colliding on a stripe serialize; no global lock covers filesystem I/O.
        self._stripes = [threading.RLock() for _ in range(64)]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._epoch += 1

    def _discard(self, key: tuple[str, str]) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._bytes -= entry.weight

    def _stamp(self, directory: Path, kind: str, extra: str | None = None) -> tuple | None:
        """Never cache through links or unavailable metadata.

        Legacy readers retain their existing path behavior on a cache bypass.
        Pinning file descriptors is outside this optimization's API scope.
        """
        root = self.settings.logs_root
        if (not directory.is_absolute() or not root.is_absolute()
                or ".." in directory.parts or ".." in root.parts):
            return None
        try:
            relative = directory.relative_to(root)
            if not relative.parts:
                return None
        except ValueError:
            return None
        try:
            ancestors = [*reversed(directory.parents), directory]
            identities = []
            for ancestor in ancestors:
                value = ancestor.lstat()
                if not stat.S_ISDIR(value.st_mode):
                    return None
                identities.append((value.st_dev, value.st_ino, value.st_mode))
            directory_stamp = _signature(directory.lstat())

            def file_stamp(name: str) -> tuple | None:
                try:
                    value = (directory / name).lstat()
                except FileNotFoundError:
                    return ("missing",)
                return _signature(value) if stat.S_ISREG(value.st_mode) else None

            names = ["run_manifest.json", "events.jsonl"]
            if kind == "detail":
                names.append("results.tsv")
            files = tuple(file_stamp(name) for name in names)
            if any(value is None for value in files):
                return None
            selected = file_stamp(extra) if extra is not None else None
            if extra is not None and selected is None:
                return None
            if _signature(directory.lstat()) != directory_stamp:
                return None
            for ancestor, identity in zip(ancestors, identities):
                current = ancestor.lstat()
                if (current.st_dev, current.st_ino, current.st_mode) != identity:
                    return None
            return (tuple(identities), directory_stamp, files, selected)
        except OSError:
            return None

    def _get(self, directory: Path, kind: str, load: Callable[[], Any],
             fallback: Callable[[], Any]) -> Any:
        key = kind, str(directory)
        with self._stripes[hash(key) % len(self._stripes)]:
            with self._lock:
                entry = self._entries.get(key)
                epoch = self._epoch
            if entry is not None:
                stamp = self._stamp(directory, kind, entry.extra)
                if stamp is not None and stamp == entry.stamp:
                    with self._lock:
                        if self._entries.get(key) is entry:
                            self._entries.move_to_end(key)
                    # Cache users receive independent nested values. They cannot
                    # mutate a later request by modifying an earlier response.
                    return json.loads(entry.payload)
                self._discard(key)
            before = self._stamp(directory, kind)
            if before is None:
                return fallback()
            try:
                result = load()
            except Exception:
                # Strict readers reveal I/O failures that compatibility readers
                # turn into an empty observation. Such fallbacks are never cached.
                return fallback()
            after = self._stamp(directory, kind)
            if after is None or before != after:
                return result
            extra = None
            if kind == "catalog" and isinstance(result, dict) and result.get("console_log"):
                console = Path(result["console_log"])
                if console.parent != directory:
                    return result
                extra = console.name
            final_stamp = self._stamp(directory, kind, extra)
            if final_stamp is None or final_stamp[:-1] != after[:-1]:
                return result
            try:
                payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, UnicodeError, RecursionError):
                return result
            # Include the key and metadata representation in retained-byte
            # accounting; entry and lock counts are independently bounded.
            weight = len(payload) + len(key[1].encode("utf-8")) + len(repr(final_stamp).encode()) + 256
            if (len(payload) > self.max_item_bytes or weight > self.max_bytes
                    or self.max_entries == 0):
                return result
            with self._lock:
                if self._epoch != epoch:
                    return result
                while self._entries and (len(self._entries) >= self.max_entries
                                         or self._bytes + weight > self.max_bytes):
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= evicted.weight
                self._entries[key] = _Entry(final_stamp, payload, extra, weight)
                self._bytes += weight
            return result

    def catalog(self, directory: Path) -> dict[str, Any] | None:
        directory = Path(directory)
        return self._get(directory, "catalog",
                         lambda: run_from_log_dir(self.settings, directory, strict=True),
                         lambda: run_from_log_dir(self.settings, directory))

    def detail(self, run: dict[str, Any]) -> dict[str, Any]:
        result = dict(run)
        raw = run.get("log_dir")
        if isinstance(raw, str) and raw:
            result["observed"] = self._get(Path(raw), "detail",
                                           lambda: read_observed_summary(self.settings, run, strict=True),
                                           lambda: read_observed_summary(self.settings, run))
        else:
            result["observed"] = read_observed_summary(self.settings, run)
        # Neither the live ledger nor the artifact listing enters the cache.
        result["artifacts"] = list_artifacts(self.settings, run)
        return result
