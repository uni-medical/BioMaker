"""Real offline dependency damage/check/repair on APFS clones, never the user engine.

Run with a bundled Python using -I -B. The caller must wrap the process in
sandbox-exec with network-outbound denied. Source-engine and cache trees are
fingerprinted before and after; all mutations stay in isolated APFS clones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import errno
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(root: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *names]):
            item = Path(parent) / name
            info = item.lstat()
            facts = [str(item.relative_to(root)), info.st_size, info.st_mtime_ns]
            if item.is_symlink():
                facts.append(os.readlink(item))
            elif item.is_file():
                # Include every byte, not just package metadata or selected files.
                with item.open("rb") as stream:
                    hashed = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hashed.update(block)
                facts.append(hashed.hexdigest())
                count += 1
            digest.update(json.dumps(facts, sort_keys=True).encode())
    return {"files": count, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--service", type=Path, default=ROOT / "service")
    parser.add_argument("--report", type=Path, default=ROOT / "output/engine-health-real.json")
    args = parser.parse_args()
    assert sys.platform == "darwin"
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("192.0.2.1", 443)) == errno.EPERM, "Requires network-denying OS sandbox"
    source = args.engine.resolve()
    source_cache = source.parent.parent / "cache/uv"
    assert source_cache.is_dir(), "Offline acceptance needs the existing populated package cache"
    before = {"engine": fingerprint(source), "cache": fingerprint(source_cache)}
    home = Path(tempfile.mkdtemp(prefix="biomaker-engine-health-qa-")).resolve()
    engine_root = home / "engine"
    cache = home / "profile/cache/uv"
    cache.parent.mkdir(parents=True)
    subprocess.run(["/bin/cp", "-cR", str(source), str(engine_root)], check=True)
    subprocess.run(["/bin/cp", "-cR", str(source_cache), str(cache)], check=True)
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime / "site-packages"))
    sys.path.insert(0, str(args.service.resolve()))
    from desktop_service import DesktopConfig, EngineInstaller
    config = DesktopConfig(home / "profile", engine_root, "synthetic-private-health-token-" + "x" * 32,
                           str(runtime / "bin/uv"), str(runtime / "python/bin/python3"))
    config.prepare()
    os.environ["UV_OFFLINE"] = "1"
    os.environ["UV_LINK_MODE"] = "copy"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    engine = EngineInstaller(config)
    commands = []
    original_run = engine._run
    def recorded_run(command, environment, *args, **kwargs):
        commands.append(command)
        return original_run(command, environment, *args, **kwargs)
    engine._run = recorded_run
    def settled(timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = engine.snapshot()
            if not state.get("operation"):
                return state
            time.sleep(.05)
        engine.cancel()
        raise AssertionError("Runtime operation exceeded acceptance deadline")
    report = {"passed": False, "home": str(home), "source_engine": str(source),
              "service": str(args.service.resolve()), "network_outbound_denied": True,
              "offline_repair": True, "before": before, "checks": []}
    started = time.monotonic()
    try:
        assert engine.snapshot()["ready"], "Cloned installed receipt should be reusable"
        engine.check()
        healthy = settled()
        assert healthy["ready"] and healthy["health"] == "healthy", healthy
        assert commands and all(command[0] != config.uv for command in commands)
        report["checks"].append("healthy-check-runs-real-imports-and-fixed-cpu-self-test-without-uv")
        monai, = (engine_root / ".venv/lib").glob("python*/site-packages/monai")
        monai.rename(home / "removed-monai")
        commands.clear()
        engine.check()
        damaged = settled()
        assert not damaged["ready"] and damaged["health"] == "failed", damaged
        assert "monai" in damaged["log"], damaged
        assert not engine.receipt.exists()
        assert all(command[0] != config.uv for command in commands)
        restarted = EngineInstaller(config)
        assert not restarted.snapshot()["ready"]
        restarted.close()
        report["checks"].append("missing-real-monai-detected-and-restart-does-not-claim-ready")
        commands.clear()
        engine.repair()
        repaired = settled()
        assert repaired["ready"] and repaired["health"] == "healthy", repaired
        assert monai.is_dir() and engine.receipt.is_file()
        assert any(command[0] == config.uv and "--frozen" in command and "--reinstall" in command for command in commands)
        report["checks"].append("real-offline-forced-reinstall-native-provision-and-full-verification")
        commands.clear()
        engine.check()
        final = settled()
        assert final["ready"] and final["health"] == "healthy", final
        assert all(command[0] != config.uv for command in commands)
        report["checks"].append("repaired-runtime-passes-second-download-free-health-check")
        report.update(passed=True, initial_check=healthy, damaged_check=damaged,
                      repaired=repaired, final_check=final)
    finally:
        engine.close()
        after = {"engine": fingerprint(source), "cache": fingerprint(source_cache)}
        report.update(after=after, source_engine_and_cache_unchanged=before == after,
                      elapsed_seconds=round(time.monotonic() - started, 3))
        report["passed"] = report["passed"] and before == after
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ["passed", "home", "checks", "elapsed_seconds", "source_engine_and_cache_unchanged"]}))
    assert report["passed"]


if __name__ == "__main__":
    main()
