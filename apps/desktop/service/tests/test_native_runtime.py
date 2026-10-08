from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_runtime
from native_runtime import LOCAL_RPATH, native_receipt_is_current, provision_native_runtime, relocate_lightgbm


def rpath(value, capacity=None):
    raw = value.encode() + b"\0"
    size = capacity or max(48 if "homebrew" in value else 40, ((12 + len(raw) + 7) // 8 * 8))
    return struct.pack("<III", 0x8000001C, size, 12) + raw.ljust(size - 12, b"\0")


def macho(paths=("/opt/homebrew/opt/libomp/lib", "/opt/local/lib/libomp"), cpu=0x0100000C):
    dependency = b"@rpath/libomp.dylib\0"
    load = struct.pack("<6I", 0xC, 48, 24, 0, 0, 0) + dependency.ljust(24, b"\0")
    commands = load + b"".join(rpath(item) for item in paths)
    header = struct.pack("<8I", 0xFEEDFACF, cpu, 0, 6, len(paths) + 1, len(commands), 0, 0)
    return header + commands + b"EXECUTABLE-CONTENT" * 8


def test_relocates_only_rpath_bytes_and_is_idempotent():
    original = macho()
    modified, count = relocate_lightgbm(original)
    assert count == 2 and len(modified) == len(original)
    assert b"/opt/homebrew" not in modified and b"/opt/local" not in modified
    assert modified.endswith(b"EXECUTABLE-CONTENT" * 8)
    assert relocate_lightgbm(modified) == (modified, 0)
    assert LOCAL_RPATH.encode() in modified


def test_universal_slices_each_get_local_lookup():
    arm, intel = macho(), macho(cpu=0x01000007)
    first, second = 4096, 8192
    header = struct.pack(">II", 0xCAFEBABE, 2)
    table = struct.pack(">5I", 0x0100000C, 0, first, len(arm), 12) + struct.pack(">5I", 0x01000007, 0, second, len(intel), 12)
    data = (header + table).ljust(first, b"\0") + arm
    data = data.ljust(second, b"\0") + intel
    modified, count = relocate_lightgbm(data)
    assert count == 4 and len(modified) == len(data)
    assert b"/opt/homebrew" not in modified


@pytest.mark.parametrize("data", [b"", b"not-a-mach-o", macho(cpu=0xC), macho()[:60]])
def test_rejects_invalid_format_architecture_and_truncation(data):
    with pytest.raises(ValueError):
        relocate_lightgbm(data)


def test_rejects_malformed_command_bounds_and_unrecognized_lookup():
    data = bytearray(macho())
    struct.pack_into("<I", data, 32 + 4, 0xFFFFFFF8)
    with pytest.raises(ValueError, match="command size"):
        relocate_lightgbm(bytes(data))
    with pytest.raises(ValueError, match="supported OpenMP RPATH"):
        relocate_lightgbm(macho(paths=("/unrecognized/path",)))


def test_rejects_insufficient_space_without_expanding_binary(monkeypatch):
    monkeypatch.setattr(native_runtime, "LOCAL_RPATH", "@loader_path/" + "too-long/" * 20)
    with pytest.raises(ValueError, match="Insufficient"):
        relocate_lightgbm(macho())


def test_atomic_relocation_preserves_cache_hardlink_and_checks_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(native_runtime.sys, "platform", "darwin")
    sites = tmp_path / ".venv/lib/python3.12/site-packages"
    library = sites / "lightgbm/lib/lib_lightgbm.dylib"
    openmp = sites / "torch/lib/libomp.dylib"
    library.parent.mkdir(parents=True)
    openmp.parent.mkdir(parents=True)
    cached = tmp_path / "cached-wheel-library"
    cached.write_bytes(macho())
    os.link(cached, library)
    openmp.write_bytes(b"local-openmp")
    commands = []
    receipt = provision_native_runtime(tmp_path, run=commands.append)
    assert cached.read_bytes() == macho()
    assert library.stat().st_ino != cached.stat().st_ino
    assert commands[0][:4] == ["/usr/bin/codesign", "--force", "--sign", "-"]
    assert native_receipt_is_current(tmp_path, receipt)
    assert provision_native_runtime(tmp_path, run=commands.append) == receipt
    assert len(commands) == 1
    library.write_bytes(macho())
    assert not native_receipt_is_current(tmp_path, receipt)


def test_failed_signing_leaves_original_and_cleans_temporary(tmp_path, monkeypatch):
    monkeypatch.setattr(native_runtime.sys, "platform", "darwin")
    sites = tmp_path / ".venv/lib/python3.12/site-packages"
    library = sites / "lightgbm/lib/lib_lightgbm.dylib"
    openmp = sites / "torch/lib/libomp.dylib"
    library.parent.mkdir(parents=True)
    openmp.parent.mkdir(parents=True)
    library.write_bytes(macho())
    openmp.write_bytes(b"local-openmp")
    def fail(_command):
        raise RuntimeError("Signing failed")
    with pytest.raises(RuntimeError, match="Signing failed"):
        provision_native_runtime(tmp_path, run=fail)
    assert library.read_bytes() == macho()
    assert not list(library.parent.glob(".biomaker-*.dylib"))
