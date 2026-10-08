"""UI -> durable queue -> installed engine -> result UI, using local test data.

macOS only. Reuses the engine named in an installation receipt without relocating its
venv, installing packages, or overriding engine source. A separate QA profile
owns all settings, queue state and outputs. Playwright wraps only the QA
sidecar's spawn and exercises its normal recovery once, applying an inherited
localhost-only OS network sandbox to the service and engine process tree.
Electron retains its production Chromium sandbox and normal UI code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

DESKTOP = Path(__file__).resolve().parents[1]
TASK = "Synthetic software acceptance only. Train a binary classifier using x0 and x1 to predict label. Id is a sample key. Use the explicit train.csv, validation.csv, and test.csv splits; label is private in validation and test. Primary validation metric: accuracy. Export class probabilities prob_0 and prob_1 with Id. Use CPU only. This is not a biomedical benchmark."
# macOS ps requires an unsandboxed exec for read-only process identity checks.
# The exact executable exception does not release Python, uv, or candidates
# from the inherited localhost-only network restriction.
SANDBOX = (
    '(version 1)\n'
    '(allow default)\n'
    '(deny network-outbound)\n'
    '(allow network-outbound (remote ip "localhost:*"))\n'
    '(allow process-exec (literal "/bin/ps") (with no-sandbox))\n'
)
PROBE = r"""
import errno, json, os, socket
from pathlib import Path
with socket.socket() as local:
    local.settimeout(2)
    local.connect(('127.0.0.1', int(os.environ['BIOMAKER_QA_FIXTURE_PORT'])))
with socket.socket() as forbidden:
    forbidden.settimeout(2)
    result = forbidden.connect_ex(('192.0.2.1', 443))
assert result == errno.EPERM, result
Path(os.environ['BIOMAKER_QA_SANDBOX_PROOF']).write_text(json.dumps({
    'localhost_allowed': True, 'nonlocal_connect_errno': result,
    'nonlocal_connect_blocked': True, 'external_endpoint': 'TEST-NET-1:443',
    'note': 'Socket denied before transmission; no HTTP/provider request sent.'
}))
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_snapshot(roots: list[Path]) -> dict:
    """Fingerprint file/link metadata plus engine sources, receipts and locks."""
    entries = []
    for root in roots:
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in sorted([*dirs, *names]):
                item = Path(directory) / name
                info = item.lstat()
                if item.is_symlink():
                    entries.append((str(item), "link", os.readlink(item)))
                elif item.is_file():
                    entries.append((str(item), info.st_size, info.st_mtime_ns))
    entries.sort()
    return {
        "entries": len(entries),
        "metadata_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed-receipt",
        type=Path,
        default=DESKTOP / "output/installed-engine-acceptance-0.1.2.json",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("/Applications/BioMaker.app/Contents/MacOS/BioMaker"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DESKTOP / "output/desktop-workflow-acceptance-0.1.2.json",
    )
    parser.add_argument(
        "--expected-version",
        help="Application version; defaults to the installed receipt version",
    )
    args = parser.parse_args()
    assert sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()
    installed = json.loads(args.installed_receipt.read_text())
    assert installed["ready"]
    expected_version = args.expected_version or installed["version"]
    engine = Path(installed["engine"]).resolve()
    data_root = engine.parent.parent
    runtime = (engine / ".venv/bin/python").resolve().parents[2]
    assert runtime.parent == data_root / "runtimes"
    assert not (
        engine / "biomaker/monai_rag/monai_rag_db/chunks.jsonl"
    ).exists(), "Fixture requires disabled optional RAG"
    before = installed_snapshot([engine, data_root / "runtimes"])
    immutable = {
        str(path): sha(path)
        for path in [
            engine / "uv.lock",
            engine / "pyproject.toml",
            engine / ".desktop-engine.json",
            engine / ".desktop-engine-identity.json",
            engine / "biomaker/run.py",
            engine / "biomaker/utils/config.py",
            args.executable,
        ]
    }
    home = Path(tempfile.mkdtemp(prefix="biomaker-desktop-workflow-qa-")).resolve()
    profile = home / "profile"
    profile.mkdir()
    # Only the QA profile is changed. The referenced install is verified before
    # and after, and uv is explicitly forbidden from syncing/installing anything.
    (profile / "engines").symlink_to(engine.parent, target_is_directory=True)
    (profile / "runtimes").symlink_to(data_root / "runtimes", target_is_directory=True)
    spec = importlib.util.spec_from_file_location(
        "workflow_fixture", DESKTOP / "tests/engine-workflow-smoke.py"
    )
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    fixture_module.CANDIDATE = fixture_module.CANDIDATE.replace(
        "assert os.environ.get('BIOMAKER_QA_GUARD_ACTIVE') == str(os.getpid())",
        "import socket, errno\n"
        "assert os.environ.get('BIOMAKER_QA_OS_SANDBOX_ACTIVE') == '1'\n"
        "with socket.socket() as qa_socket:\n"
        "    qa_socket.settimeout(2)\n"
        "    qa_network_errno = qa_socket.connect_ex(('192.0.2.1', 443))\n"
        "assert qa_network_errno == errno.EPERM, qa_network_errno",
    ).replace(
        "'guard_active': True",
        "'guard_active': True, 'sandbox_external_errno': qa_network_errno",
    )
    dataset = fixture_module.write_dataset(home)
    fixture = fixture_module.FixtureServer(home)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    (home / "network.sb").write_text(SANDBOX)
    (home / "sandbox-probe.py").write_text(PROBE)
    config = {
        "home": str(home),
        "profile": str(profile),
        "dataset": str(dataset),
        "task": TASK,
        "executable": str(args.executable),
        "engine": str(engine),
        "model": fixture_module.MODEL,
        "fakeKey": fixture_module.SYNTHETIC_KEY,
        "expectedVersion": expected_version,
        "baseURL": f"http://127.0.0.1:{fixture.server_port}/v1",
    }
    (home / "qa-config.json").write_text(json.dumps(config))
    # Allowlist, never inherit provider/search/cloud credentials from the caller.
    env = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR")
        if key in os.environ
    }
    env.update(
        PATH="/usr/bin:/bin:/usr/sbin:/sbin",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        BIOMAKER_DESKTOP_USER_DATA=str(profile),
        BIOMAKER_DESKTOP_HOME=str(profile),
        BIOMAKER_QA_FIXTURE_PORT=str(fixture.server_port),
        BIOMAKER_QA_OS_SANDBOX_ACTIVE="1",
        BIOMAKER_QA_SANDBOX_PROOF=str(home / "sandbox-proof.json"),
        OPENAI_REQUEST_TIMEOUT_SECONDS="5",
        NO_PROXY="127.0.0.1,localhost,::1",
        UV_NO_SYNC="1",
        UV_FROZEN="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        MPLBACKEND="Agg",
    )
    proof = {
        "passed": False,
        "synthetic": True,
        "home": str(home),
        "engine": str(engine),
        "profile": str(profile),
        "application": str(args.executable),
        "version": expected_version,
        "installed_source_override": False,
        "cold_install": False,
        "installed_engine_reused": True,
        "external_provider_calls": 0,
        "scientific_performance_claim": False,
        "network_enforcement": "Test-only spawn wrapper sandboxes Python sidecar and all queue/uv/engine descendants to localhost; Electron keeps production Chromium sandbox",
        "qa_sandbox_exception": {
            "rule": '(allow process-exec (literal "/bin/ps") (with no-sandbox))',
            "scope": "Exact /bin/ps executable only, for read-only process identity inspection.",
            "production_source_modified": False,
            "chromium_sandbox_modified": False,
            "candidate_external_connect_assertion": "connect_ex(TEST-NET-1:443) == errno.EPERM",
            "python_uv_training_remain_network_restricted": True,
        },
        "sandbox_policy_sha256": sha(home / "network.sb"),
        "installed_before": before,
        "immutable_sha256": immutable,
        "final_retraining": False,
    }
    started = time.monotonic()
    process = None
    try:
        subprocess.run(
            [
                "/usr/bin/sandbox-exec",
                "-f",
                str(home / "network.sb"),
                str(runtime / "python/bin/python3"),
                "-I",
                "-B",
                str(home / "sandbox-probe.py"),
            ],
            env=env,
            check=True,
            timeout=10,
        )
        node = shutil.which("node")
        assert node
        with (home / "electron-workflow.log").open("w") as output:
            process = subprocess.Popen(
                [
                    node,
                    str(DESKTOP / "tests/electron-workflow-smoke.mjs"),
                    str(home / "qa-config.json"),
                ],
                cwd=DESKTOP,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            proof["returncode"] = process.wait(timeout=180)
        assert (
            proof["returncode"] == 0
        ), f"Electron QA failed; inspect {home / 'electron-workflow.log'}"
        proof["sandbox_probe"] = json.loads((home / "sandbox-proof.json").read_text())
        assert proof["sandbox_probe"]["nonlocal_connect_blocked"]
        proof["ui"] = json.loads((home / "ui-proof.json").read_text())
        proof["sandbox_spawns"] = [
            json.loads(line)
            for line in (home / "sandbox-spawns.jsonl").read_text().splitlines()
        ]
        assert proof[
            "sandbox_spawns"
        ], "Sidecar must be launched through the OS sandbox"
        assert proof["ui"]["passed"]
        fixture_module.verify(
            home,
            fixture,
            proof,
            logs_root=profile / "workspace/logs",
            python_network_guard=False,
        )
        # The inherited OS restriction covers all engine subprocesses; the
        # original Python-hook counter does not apply to this integration test.
        proof.pop("guarded_python_processes", None)
    except Exception as exc:
        proof["passed"] = False
        proof["error"] = str(exc)
        proof["traceback"] = traceback.format_exc()
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=2)
        after = installed_snapshot([engine, data_root / "runtimes"])
        proof["installed_after"] = after
        proof["installed_files_unchanged"] = before == after and all(
            sha(Path(name)) == value for name, value in immutable.items()
        )
        proof["elapsed_seconds"] = round(time.monotonic() - started, 3)
        proof["local_api_calls"] = fixture.requests_seen
        proof["fixture_errors"] = fixture.errors
        if not proof["installed_files_unchanged"]:
            proof["passed"] = False
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(proof, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "passed": proof["passed"],
                    "home": str(home),
                    "report": str(args.report),
                    "elapsed_seconds": proof["elapsed_seconds"],
                    "installed_files_unchanged": proof["installed_files_unchanged"],
                },
                indent=2,
            )
        )
    assert proof["passed"]


if __name__ == "__main__":
    main()
