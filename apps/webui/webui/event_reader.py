"""Read the needed JSON-object tail without parsing discarded event history.

Each read pins one descriptor and its initial byte length. Appends do not extend
the work and path replacements do not switch sources midway through a read.
This is not a transactional snapshot of a file being rewritten in place; the
observation cache separately checks source metadata before publishing a result.
Memory is proportional to one block, the longest record, and retained objects,
not a fixed byte cap: preserving arbitrarily large records requires that space.
"""
from __future__ import annotations

import errno
import json
import math
import os
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterator


_BLOCK_BYTES = 64 * 1024
_LINE_BREAK = re.compile(rb"[\r\n]")


def _event_float(value: str) -> float | str:
    number = float(value)
    return number if math.isfinite(number) else ("Infinity" if number > 0 else "-Infinity")


def _reverse_lines(handle: BinaryIO, end: int, *, block_size: int) -> Iterator[bytes]:
    """Yield complete raw records in reverse order, splitting only CR and LF.

    Treating CRLF as two separators adds an empty record, which callers ignore.
    Unlike str.splitlines, this does not split Unicode whitespace inside JSON.
    Accumulate long-line fragments without repeatedly copying a growing prefix.
    """
    fragments: list[bytes] = []
    position = end
    while position:
        size = min(block_size, position)
        position -= size
        handle.seek(position)
        block = handle.read(size)
        if len(block) != size:
            raise OSError(errno.EIO, "Event history changed while reading")
        parts = _LINE_BREAK.split(block)
        if len(parts) == 1:
            fragments.append(parts[0])
            continue
        fragments.append(parts[-1])
        yield b"".join(reversed(fragments))
        fragments.clear()
        yield from reversed(parts[1:-1])
        fragments.append(parts[0])
    if fragments:
        yield b"".join(reversed(fragments))


def read_event_window(
    path: Path, *, tail: int, count_all: bool
) -> tuple[int | None, list[dict[str, Any]]]:
    """Return the last ``tail`` JSON objects in original chronological order.

    With count_all, scan every initial byte and count decoded nonempty lines,
    including malformed JSON and nonobject JSON, but parse only the needed tail.
    Without count_all, stop once the object tail is full and return no count.
    Resource-pathological JSON (integer limits or nesting) is treated as an
    invalid event, deliberately improving on the legacy parser's exceptions.
    I/O errors propagate so callers can preserve their strict/fallback policy.
    """
    limit = max(0, tail)
    if not count_all and not limit:
        return None, []
    events: list[dict[str, Any]] = []
    count = 0
    with path.open("rb") as handle:
        end = os.fstat(handle.fileno()).st_size
        for raw in _reverse_lines(handle, end, block_size=_BLOCK_BYTES):
            # Decode complete records, never separate blocks: otherwise a
            # multibyte codepoint crossing a block boundary becomes replacement.
            line = raw.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            count += 1
            if len(events) >= limit:
                continue
            try:
                event = json.loads(line, parse_constant=str, parse_float=_event_float)
            except (ValueError, RecursionError):
                # JSONDecodeError is a ValueError; integer-conversion limits and
                # deeply nested malformed histories must not hide other events.
                continue
            if isinstance(event, dict):
                events.append(event)
                if not count_all and len(events) == limit:
                    break
    events.reverse()
    return count if count_all else None, events
