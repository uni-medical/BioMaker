"""Run the real BioMaker workflow against a strictly local model fixture.

This is an operational acceptance test, not a model or biomedical benchmark.
The fixture supplies deterministic responses through the actual OpenAI adapter;
BioMaker still builds/verifies the splits, compiles its frozen evaluator, runs a
real CPU classifier, evaluates its predictions, and exports the normal artifacts.
No production engine code is patched. All run data stays in a fresh QA directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
MODEL = "biomaker-local-fixture"
SYNTHETIC_KEY = "biomaker-offline-fixture-not-a-real-key"

BUILDER = r"""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
public = args.output_dir / 'biomaker_splits'
private = args.output_dir / 'private_gt'
public.mkdir(parents=True, exist_ok=True)
private.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(args.input_dir / 'train.csv')
train.to_csv(public / 'agent_train.csv', index=False)
for source, public_name, private_name in [
    ('validation.csv', 'score_val.csv', 'score_val.csv'),
    ('test.csv', 'agent_test.csv', 'test.csv'),
]:
    frame = pd.read_csv(args.input_dir / source)
    frame.drop(columns=['label']).to_csv(public / public_name, index=False)
    frame.to_csv(private / private_name, index=False)
hidden = {'score_val': {'answer_columns': ['label']},
          'test': {'answer_columns': ['label']}}
(private / 'private_gt.json').write_text(json.dumps(hidden))
(args.output_dir / 'split_manifest.json').write_text(json.dumps({
    'manifest_version': 1,
    'sample_unit': 'row',
    'key_columns': ['Id'],
    'task_heads': [{'type': 'classification', 'class_labels': [0, 1]}],
    'visibility': {'agent_visible': ['biomaker_splits/**'],
                   'evaluator_only': ['private_gt/**'],
                   'forbidden': ['validation.csv', 'test.csv']},
    'prediction_contract': {
        'key_columns': ['Id'],
        'prediction_columns': ['prob_0', 'prob_1'],
        'prediction_type': 'class_probabilities',
        'validation_artifact': 'working/validation_predictions.csv',
        'test_submission': 'working/submission.csv',
    },
    'metric_contract': {'primary_metric': 'accuracy', 'maximize': True},
    'hidden_answers': hidden,
    'sample_universe_summary': {'train': 12, 'score_val': 4, 'test': 4},
}))
print('Prepared the three explicit synthetic QA splits.')
"""

CANDIDATE = r"""
import json
import os
import hashlib
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

assert os.environ.get('BIOMAKER_QA_GUARD_ACTIVE') == str(os.getpid())
split = Path('input/biomaker_splits')
output = Path('working')
output.mkdir(exist_ok=True)
train = pd.read_csv(split / 'agent_train.csv')
features = ['x0', 'x1']
model = LogisticRegression(random_state=42, max_iter=50, n_jobs=1)
model.fit(train[features], train['label'])
for source, target in [('score_val.csv', 'validation_predictions.csv'),
                       ('agent_test.csv', 'submission.csv')]:
    frame = pd.read_csv(split / source)
    assert 'label' not in frame.columns
    probabilities = model.predict_proba(frame[features])
    result = frame[['Id']].copy()
    result['prob_0'] = probabilities[:, 0]
    result['prob_1'] = probabilities[:, 1]
    result.to_csv(output / target, index=False)
proof = {'synthetic': True, 'implementation': 'sklearn.linear_model.LogisticRegression',
         'device': 'cpu', 'training_rows': len(train), 'candidate_pid': os.getpid(),
         'guard_active': True, 'classes': model.classes_.tolist(),
         'coefficients': model.coef_.tolist(), 'intercept': model.intercept_.tolist()}
(output / 'synthetic_model.json').write_text(json.dumps(proof, indent=2))
joblib.dump(model, output / 'model.joblib')
probe_inputs = train[features].iloc[:2].values.tolist()
(output / 'smoke_input.json').write_text(json.dumps(probe_inputs))
(output / 'smoke_expected.json').write_text(json.dumps(model.predict_proba(pd.DataFrame(probe_inputs, columns=features)).tolist()))
(output / 'inference.py').write_text(
    'import joblib\nimport pandas as pd\n'
    'def predict(inputs, artifact_dir):\n'
    '    model = joblib.load(artifact_dir / "model.joblib")\n'
    '    return model.predict_proba(pd.DataFrame(inputs, columns=["x0", "x1"]))\n'
)
artifact_files = []
for name in ['model.joblib', 'inference.py', 'smoke_input.json', 'smoke_expected.json']:
    artifact_files.append({'path': name, 'role': 'checkpoint' if name == 'model.joblib' else 'asset',
        'format': 'joblib' if name == 'model.joblib' else None,
        'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
(output / 'model_manifest.json').write_text(json.dumps({
    'schema_version': 1, 'framework': 'sklearn', 'files': artifact_files,
    'inference': {'entrypoint': 'inference.py', 'input': 'smoke_input.json', 'expected_output': 'smoke_expected.json'},
    'metadata': {'input_schema': {'features': features}, 'output_schema': {'columns': ['prob_0', 'prob_1']},
        'preprocessing': 'numeric x0 and x1, no scaling', 'class_labels': [0, 1],
        'dependencies': {'scikit-learn': sklearn.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
}, indent=2))
print('BIO_MAKER_QA_CPU_FIT ' + json.dumps(proof))
"""

REPORT = """# Synthetic workflow acceptance

## Introduction
This run exercises BioMaker with deterministic local fixture responses and twenty synthetic rows.

## Preprocessing
The explicit training, validation, and test files were converted to the public split contract. Validation and test labels were kept in evaluator-only files.

## Modeling Methods
One scikit-learn logistic regression candidate was fitted on twelve training rows on CPU.

## Results Discussion
The frozen BioMaker evaluator checked validation predictions. The authoritative test result is the engine journal and artifact receipt, not this fixture-authored report.

## Limitations
This is a software integration test. It does not assess model-provider intelligence, biomedical performance, or generalization. Final retraining is disabled.
"""

NETWORK_GUARD = r"""
import json
import os
import sys

_port = int(os.environ['BIOMAKER_QA_FIXTURE_PORT'])
_log = os.environ['BIOMAKER_QA_NETWORK_LOG']
def _record(event, **details):
    with open(_log, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'event': event, 'pid': os.getpid(), **details}) + '\n')

def _guard(event, args):
    if event == 'socket.getaddrinfo':
        host = args[0]
        if host not in ('127.0.0.1', '::1', 'localhost', None):
            _record('blocked_resolution', host=str(host))
            raise RuntimeError('QA forbids non-loopback name resolution')
    if event in ('socket.connect', 'socket.sendto'):
        address = args[-1]
        if not isinstance(address, tuple):
            return  # Local Unix sockets are not network-provider connections.
        allowed = address[0] in ('127.0.0.1', '::1') and address[1] == _port
        _record('connect' if allowed else 'blocked_connect', address=list(address))
        if not allowed:
            raise RuntimeError('QA allows connections only to its own loopback fixture')

sys.addaudithook(_guard)
os.environ['BIOMAKER_QA_GUARD_ACTIVE'] = str(os.getpid())
_record('guard_started')
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_dataset(home: Path) -> Path:
    dataset = home / "dataset"
    dataset.mkdir()
    index = 0
    for name, count in [("train.csv", 12), ("validation.csv", 4), ("test.csv", 4)]:
        with (dataset / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Id", "x0", "x1", "label"])
            writer.writeheader()
            for local_index in range(count):
                label = local_index % 2
                writer.writerow(
                    {
                        "Id": index,
                        "x0": (2 * label - 1) * (1 + local_index / 100),
                        "x1": (local_index % 3) / 10,
                        "label": label,
                    }
                )
                index += 1
    return dataset


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, home: Path):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.home = home
        self.requests_seen: list[dict] = []
        self.errors: list[str] = []
        self.lock = threading.Lock()


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *_args):
        pass

    def do_POST(self):
        try:
            assert self.path == "/v1/chat/completions", self.path
            assert self.headers.get("Authorization") == f"Bearer {SYNTHETIC_KEY}"
            length = int(self.headers.get("Content-Length", "0"))
            assert 0 < length < 2_000_000, length
            payload = json.loads(self.rfile.read(length))
            assert payload["model"] == MODEL
            messages = payload["messages"]
            prompt = "\n".join(str(message.get("content", "")) for message in messages)
            function = (
                (payload.get("tool_choice") or {}).get("function", {}).get("name")
            )
            if function == "create_draft_plan_batch":
                stage, data = "draft_plan", {
                    "plans": [
                        {
                            "name": "Synthetic CPU classification",
                            "plan": "Fit logistic regression on the public training features x0 and x1. Export class probabilities for validation and test; leave scoring to the frozen evaluator.",
                        }
                    ]
                }
            elif function == "submit_tool_requests":
                stage, data = "tool_selection", {"requests": []}
            elif function:
                raise AssertionError(f"Unexpected tool contract: {function}")
            elif "You are a data-preparation coding agent" in prompt:
                stage, data = (
                    "split_builder",
                    "Use the explicit synthetic splits.\n```python\n"
                    + BUILDER
                    + "\n```",
                )
            elif "factual technical report summarising" in prompt:
                stage, data = "report", REPORT
            elif "Response format" in prompt and "Task Description" in prompt:
                stage, data = (
                    "candidate",
                    "Fit a deterministic, small CPU classifier using only the public training split.\n```python\n"
                    + CANDIDATE
                    + "\n```",
                )
            else:
                raise AssertionError("Unknown fixture request: " + prompt[:200])
            message = {"role": "assistant", "content": data if not function else None}
            if function:
                message["tool_calls"] = [
                    {
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": function, "arguments": json.dumps(data)},
                    }
                ]
            response = {
                "id": "chatcmpl-local-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "system_fingerprint": "deterministic-local-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if function else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            record = {
                "stage": stage,
                "path": self.path,
                "model": MODEL,
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            with self.server.lock:
                self.server.requests_seen.append(record)
                with (self.server.home / "fixture-requests.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps({**record, "request": payload, "response": response})
                        + "\n"
                    )
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            with self.server.lock:
                self.server.errors.append(str(exc))
            self.send_error(400, "Fixture contract mismatch")


def environment(home: Path, port: int, source: Path) -> dict[str, str]:
    # An allowlist prevents ambient model/search/cloud credentials being inherited.
    result = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
        if key in os.environ
    }
    guard = home / "python-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(NETWORK_GUARD)
    (home / "empty.env").write_text("# QA fixture only; never read a user's .env\n")
    result.update(
        {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": os.pathsep.join([str(guard), str(source)]),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            # Retain this explicitly: the credential allowlist above drops
            # ambient Python settings, and child tools must preserve QA sources.
            "PYTHONDONTWRITEBYTECODE": "1",
            "BIOMAKER_ENV_FILE": str(home / "empty.env"),
            "BIOMAKER_QUEUE_DIRECT": "1",
            "OPENAI_API_KEY": SYNTHETIC_KEY,
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "OPENAI_REQUEST_TIMEOUT_SECONDS": "5",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "BIOMAKER_QA_FIXTURE_PORT": str(port),
            "BIOMAKER_QA_NETWORK_LOG": str(home / "network.jsonl"),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "MPLBACKEND": "Agg",
        }
    )
    return result


def verify(
    home: Path,
    fixture: FixtureServer,
    proof: dict,
    *,
    logs_root: Path | None = None,
    python_network_guard: bool = True,
) -> None:
    manifests = list((logs_root or home / "logs").glob("*/run_manifest.json"))
    assert len(manifests) == 1, manifests
    log = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    workspace = Path(manifest["workspace_dir"])
    events = [
        json.loads(line) for line in (log / "events.jsonl").read_text().splitlines()
    ]
    names = [event["event"] for event in events]
    for required in (
        "run_started",
        "workspace_prepared",
        "split_protocol_prepared",
        "fixed_evaluator_prepared",
        "evaluator_prepared",
        "execution_started",
        "execution_finished",
        "report_finished",
        "run_finished",
    ):
        assert required in names, (required, names)
    assert "run_failed" not in names
    assert events[-1]["event"] == "run_finished" and events[-1]["completed_steps"] == 1
    journal = json.loads((log / "journal.json").read_text())
    assert len(journal["nodes"]) == 1
    node = journal["nodes"][0]
    assert node["is_buggy"] is False, node
    assert node["metric"]["value"] == 1.0, node["metric"]
    public = workspace / "input" / "biomaker_splits"
    assert len(rows(public / "agent_train.csv")) == 12
    for name in ("score_val.csv", "agent_test.csv"):
        frame = rows(public / name)
        assert len(frame) == 4 and "label" not in frame[0]
    assert not (workspace / "input" / "validation.csv").exists()
    assert not (workspace / "input" / "test.csv").exists()
    assert "label" in rows(log / "private_gt" / "score_val.csv")[0]
    candidate_workspace = Path(node["execution_workspace"])
    model_path = candidate_workspace / "working" / "synthetic_model.json"
    fit = json.loads(model_path.read_text())
    assert fit["device"] == "cpu" and fit["training_rows"] == 12 and fit["guard_active"]
    for name in ("validation_predictions.csv", "submission.csv"):
        predictions = rows(candidate_workspace / "working" / name)
        assert len(predictions) == 4 and set(predictions[0]) == {
            "Id",
            "prob_0",
            "prob_1",
        }
    for name in (
        "best_solution.py",
        "report.md",
        "tree_plot.html",
        "evaluator.py",
        "results.tsv",
        "evaluator_prediction_spec.json",
    ):
        assert (log / name).is_file() and (log / name).stat().st_size > 0, name
    # BioMaker formats generated code; compare syntax trees instead of spelling.
    assert ast.dump(ast.parse((log / "best_solution.py").read_text())) == ast.dump(
        ast.parse(CANDIDATE)
    )
    network = []
    if python_network_guard:
        network = [
            json.loads(line)
            for line in (home / "network.jsonl").read_text().splitlines()
        ]
        assert not [
            item for item in network if item["event"].startswith("blocked")
        ], network
        assert fit["candidate_pid"] in {
            item["pid"] for item in network if item["event"] == "guard_started"
        }
        assert all(
            item["address"][0] in ("127.0.0.1", "::1")
            and item["address"][1] == fixture.server_port
            for item in network
            if item["event"] == "connect"
        )
    stages = [request["stage"] for request in fixture.requests_seen]
    for stage in ("split_builder", "draft_plan", "candidate"):
        assert stages.count(stage) == 1, stages
    assert stages.count("report") == 0, stages
    model_check = node["execution"]["artifact_validation"]["model_artifact"]
    assert model_check["ok"] and model_check["checkpoint_count"] == 1
    assert model_check["relocated"] and model_check["probe_examples"] == 2
    assert model_check["artifact_sha256"]["model.joblib"] == digest(
        log / "working_best/model.joblib"
    )
    assert not fixture.errors, fixture.errors
    proof.update(
        {
            "passed": True,
            "log_dir": str(log),
            "workspace": str(workspace),
            "candidate_workspace": str(candidate_workspace),
            "node_id": node["id"],
            "completed_steps": 1,
            "synthetic_fixture_accuracy": node["metric"]["value"],
            "candidate_fit": fit,
            "saved_model_check": model_check,
            "events": names,
            "local_api_calls": fixture.requests_seen,
            "guarded_python_processes": len(
                {item["pid"] for item in network if item["event"] == "guard_started"}
            ),
            "artifact_sha256": {
                name: digest(log / name)
                for name in (
                    "journal.json",
                    "evaluator.py",
                    "best_solution.py",
                    "report.md",
                    "tree_plot.html",
                )
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="Installed desktop scientific engine directory",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional checkout under test; default is the installed engine itself",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DESKTOP / "output" / "engine-workflow-acceptance.json",
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    engine = args.engine.resolve()
    source = args.source.resolve() if args.source else engine
    python = (
        engine / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    assert python.is_file(), python
    home = Path(tempfile.mkdtemp(prefix="biomaker-workflow-qa-")).resolve()
    dataset = write_dataset(home)
    task = home / "task.md"
    task.write_text(
        "Synthetic software acceptance only. Train a binary classifier using x0 and x1 to predict label. Id is a sample key. Use the explicit train.csv, validation.csv, and test.csv splits; label is private in validation and test. Primary validation metric: accuracy. Export class probabilities prob_0 and prob_1 with Id. Use CPU only. This is not a biomedical benchmark.\n"
    )
    fixture = FixtureServer(home)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    proof = {
        "passed": False,
        "synthetic": True,
        "external_provider_calls": 0,
        "scientific_performance_claim": False,
        "home": str(home),
        "engine": str(engine),
        "source": str(source),
        "installed_source_override": args.source is not None,
        "engine_source_sha256": {
            name: digest(source / name)
            for name in (
                "biomaker/run.py",
                "biomaker/backend/backend_openai.py",
                "biomaker/utils/config.py",
                "biomaker/utils/split_protocol.py",
                "biomaker/utils/evaluator_setup.py",
            )
        },
        "fixture_scope": "Deterministic local model responses; real engine split preparation, frozen evaluation, sklearn CPU execution, journal and report export. No production code patches.",
        "final_retraining": False,
    }
    command = [
        str(python),
        "-B",
        "-m",
        "biomaker.run",
        f"data_dir={dataset}",
        f"desc_file={task}",
        f"log_dir={home / 'logs'}",
        f"workspace_dir={home / 'workspaces'}",
        "exp_name=synthetic-desktop-workflow",
        f"model_for_all_agents={MODEL}",
        "agent.steps=1",
        "agent.total_time_limit=60",
        "exec.timeout=30",
        "exec.parallel_workers=1",
        "agent.enable_monai_knowledge_base=false",
        "agent.enable_query2doc=false",
        "agent.enable_web_search=false",
        "agent.enable_node_tools=true",
        "agent.max_node_tool_calls=1",
        "split_builder.attempts=1",
        "evaluator_cache.enabled=false",
        "final_retrain.enabled=false",
        "generate_report=true",
    ]
    process = None
    started = time.monotonic()
    try:
        with (home / "engine-output.log").open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment(home, fixture.server_port, source),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            proof["engine_pid"] = process.pid
            proof["returncode"] = process.wait(timeout=args.timeout)
        assert (
            proof["returncode"] == 0
        ), f"Engine exited {proof['returncode']}; inspect {home / 'engine-output.log'}"
        verify(home, fixture, proof)
    except Exception as exc:
        proof["error"] = str(exc)
        proof["traceback"] = traceback.format_exc()
        raise
    finally:
        if process is not None and process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=2)
        proof["elapsed_seconds"] = round(time.monotonic() - started, 3)
        proof["local_api_calls"] = fixture.requests_seen
        proof["fixture_errors"] = fixture.errors
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(proof, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "passed": proof["passed"],
                    "home": str(home),
                    "report": str(args.report),
                    "elapsed_seconds": proof["elapsed_seconds"],
                    "local_api_calls": len(fixture.requests_seen),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
