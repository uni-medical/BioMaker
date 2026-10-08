"""macOS native-runtime acceptance without changing the supplied engine.

Usage: python3 tests/native-portability-smoke.py /path/to/installed/engine
Clones its virtualenv using APFS copy-on-write, blocks package-manager library
roots, provisions that clone, and runs an actual synthetic CPU LightGBM model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
from native_runtime import native_receipt_is_current, provision_native_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/native-portability-proof.json")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This acceptance test requires macOS with APFS and sandbox-exec.")
    source = args.engine.resolve()
    source_libraries = list((source / ".venv/lib").glob("python*/site-packages/lightgbm/lib/lib_lightgbm.dylib"))
    if len(source_libraries) != 1:
        parser.error("The input engine must have LightGBM installed in its .venv.")
    source_library = source_libraries[0]
    before = hashlib.sha256(source_library.read_bytes()).hexdigest()
    temporary = Path(tempfile.mkdtemp(prefix="biomaker-no-homebrew-"))
    profile = '(version 1)(allow default)(deny file-read* (subpath "/opt/homebrew") (subpath "/usr/local") (subpath "/opt/local"))'
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        subprocess.run(["/bin/cp", "-cR", str(source / ".venv"), str(temporary / ".venv")], check=True)
        command = ["/usr/bin/sandbox-exec", "-p", profile, str(temporary / ".venv/bin/python"), "-I", str(ROOT / "service/verify_engine.py")]
        baseline = subprocess.run(command + ["--lightgbm-only"], env=env, capture_output=True, text=True, timeout=90)
        receipt = provision_native_runtime(temporary)
        assert native_receipt_is_current(temporary, receipt)
        provisioned_library = temporary / source_library.relative_to(source)
        subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(provisioned_library)], check=True, capture_output=True)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        unchanged = hashlib.sha256(source_library.read_bytes()).hexdigest() == before
        report = {
            "passed": result.returncode == 0 and unchanged,
            "source_engine": str(source), "source_unchanged": unchanged,
            "sandbox_denied_roots": ["/opt/homebrew", "/usr/local", "/opt/local"],
            "baseline_exit_code": baseline.returncode, "baseline_stderr": baseline.stderr,
            "receipt": receipt, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "verified": "LightGBM imported before Torch; synthetic CPU LightGBM calculation; no biomedical/model-provider workflow.",
            "limitation": "macOS sandbox denial on this machine; not a separate clean-machine or Intel-hardware test.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "report": str(args.output)}))
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
