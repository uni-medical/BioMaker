"""Relocate LightGBM's macOS OpenMP lookup inside the installed engine.

Upstream macOS wheels search Homebrew/MacPorts for libomp. PyTorch already ships
the compatible runtime. We only replace existing RPATH strings within their
allocated Mach-O command space, then ad-hoc sign an atomic private copy. Neither
the package manager cache nor global libraries are modified.
"""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

NATIVE_RUNTIME_REVISION = "lightgbm-local-openmp-v1"
LOCAL_RPATH = "@loader_path/../../torch/lib"
EXTERNAL_RPATHS = {
    "/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib", "/opt/local/lib/libomp"
}


def _slices(data: bytes) -> list[tuple[int, int]]:
    if len(data) < 8:
        raise ValueError("Truncated Mach-O header")
    if data[:4] == b"\xcf\xfa\xed\xfe":
        return [(0, len(data))]
    if data[:4] not in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        raise ValueError("Unsupported Mach-O format")
    count = struct.unpack_from(">I", data, 4)[0]
    width = 32 if data[:4] == b"\xca\xfe\xba\xbf" else 20
    if not 1 <= count <= 16 or 8 + count * width > len(data):
        raise ValueError("Invalid universal Mach-O architecture table")
    result = []
    for index in range(count):
        cursor = 8 + index * width
        offset, size = struct.unpack_from(">QQ" if width == 32 else ">II", data, cursor + 8)
        if offset < 8 + count * width or size < 32 or offset + size > len(data):
            raise ValueError("Invalid universal Mach-O slice bounds")
        if any(offset < old_offset + old_size and old_offset < offset + size for old_offset, old_size in result):
            raise ValueError("Overlapping universal Mach-O slices")
        result.append((offset, size))
    return result


def relocate_lightgbm(data: bytes) -> tuple[bytes, int]:
    """Return a bounded binary edit, rejecting formats we cannot safely rewrite."""
    output = bytearray(data)
    replacements = 0
    for base, size in _slices(data):
        if size < 32 or data[base:base + 4] != b"\xcf\xfa\xed\xfe":
            raise ValueError("Only little-endian 64-bit Mach-O slices are supported")
        _, cpu, _, _, count, command_bytes, _, _ = struct.unpack_from("<8I", data, base)
        if cpu not in (0x01000007, 0x0100000C):
            raise ValueError("Unsupported Mach-O CPU architecture")
        if not 1 <= count <= 4096 or command_bytes > size - 32:
            raise ValueError("Invalid Mach-O load command table")
        cursor, end = base + 32, base + 32 + command_bytes
        found_dependency, found_local = False, False
        rpaths = []
        for _ in range(count):
            if cursor + 8 > end:
                raise ValueError("Truncated Mach-O load command")
            command, length = struct.unpack_from("<II", data, cursor)
            if length < 8 or length % 8 or cursor + length > end:
                raise ValueError("Invalid Mach-O load command size")
            if command in (0x8000001C, 0xC, 0x80000018):
                minimum = 12 if command == 0x8000001C else 24
                if length < minimum:
                    raise ValueError("Truncated Mach-O path command")
                start = struct.unpack_from("<I", data, cursor + 8)[0]
                if start < minimum or start >= length:
                    raise ValueError("Invalid Mach-O path offset")
                raw = data[cursor + start:cursor + length]
                if b"\0" not in raw:
                    raise ValueError("Unterminated Mach-O path")
                value = raw.split(b"\0", 1)[0].decode("utf-8")
                if command == 0x8000001C:
                    rpaths.append((cursor + start, length - start, value))
                    found_local |= value == LOCAL_RPATH
                else:
                    found_dependency |= value == "@rpath/libomp.dylib"
            cursor += length
        if cursor != end or not found_dependency:
            raise ValueError("Unexpected LightGBM OpenMP load commands")
        local = LOCAL_RPATH.encode() + b"\0"
        if not found_local and any(value in EXTERNAL_RPATHS for _, _, value in rpaths) and not any(value in EXTERNAL_RPATHS and capacity >= len(local) for _, capacity, value in rpaths):
            raise ValueError("Insufficient Mach-O RPATH space for a safe local replacement")
        for start, capacity, value in rpaths:
            if value not in EXTERNAL_RPATHS:
                continue
            # Some wheels allocate a shorter secondary MacPorts RPATH. Remove
            # that external fallback too, without resizing any load command.
            replacement = local if len(local) <= capacity else b"@loader_path\0"
            if len(replacement) > capacity:
                raise ValueError("Insufficient Mach-O RPATH space for a safe local replacement")
            output[start:start + capacity] = replacement.ljust(capacity, b"\0")
            replacements += 1
            found_local |= replacement == local
        if not found_local:
            raise ValueError("LightGBM has no supported OpenMP RPATH to relocate")
    return bytes(output), replacements


def _libraries(engine_root: Path) -> tuple[Path, Path]:
    venv = (engine_root / ".venv").resolve()
    sites = list((venv / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise RuntimeError("Cannot locate the engine's Python site-packages directory")
    lightgbm = sites[0] / "lightgbm/lib/lib_lightgbm.dylib"
    openmp = sites[0] / "torch/lib/libomp.dylib"
    for library in (lightgbm, openmp):
        if not library.is_file() or not library.resolve().is_relative_to(venv):
            raise RuntimeError(f"The engine's bundled native library is missing: {library.name}")
    return lightgbm, openmp


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provision_native_runtime(engine_root: Path, run: Callable[[list[str]], None] | None = None) -> dict:
    if sys.platform != "darwin":
        return {"revision": NATIVE_RUNTIME_REVISION, "platform": sys.platform, "status": "not-required"}
    library, openmp = _libraries(engine_root)
    original = library.read_bytes()
    patched, replacements = relocate_lightgbm(original)
    if replacements:
        fd, temporary = tempfile.mkstemp(prefix=".biomaker-lightgbm-", suffix=".dylib", dir=library.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(patched)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, library.stat().st_mode & 0o777)
            command = ["/usr/bin/codesign", "--force", "--sign", "-", temporary]
            if run is not None:
                run(command)
            else:
                subprocess.run(command, check=True, capture_output=True, timeout=30)
            os.replace(temporary, library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"revision": NATIVE_RUNTIME_REVISION, "platform": sys.platform, "status": "relocated", "lightgbm_sha256": _sha(library), "openmp_sha256": _sha(openmp), "rpath": LOCAL_RPATH}


def native_receipt_is_current(engine_root: Path, receipt: object) -> bool:
    if not isinstance(receipt, dict) or receipt.get("revision") != NATIVE_RUNTIME_REVISION or receipt.get("platform") != sys.platform:
        return False
    if sys.platform != "darwin":
        return receipt.get("status") == "not-required"
    try:
        library, openmp = _libraries(engine_root)
        return receipt.get("status") == "relocated" and receipt.get("lightgbm_sha256") == _sha(library) and receipt.get("openmp_sha256") == _sha(openmp)
    except (OSError, ValueError, RuntimeError):
        return False
