"""Portable fitted-model artifacts and a supervised reload contract.

Training-call detection is deliberately bounded to common Python APIs. It is not
a proof that arbitrary generated programs did (or did not) train a model. An
explicit ``model_artifact_required`` evaluator contract covers other trainers.
No generated code is imported or deserialized in the BioMaker parent process.
"""

from __future__ import annotations

import ast
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MANIFEST_NAME = "model_manifest.json"
PROBE_REPORT_NAME = "model_probe_report.json"
MODEL_ARTIFACT_GUIDELINE = """If your solution fits/trains any model or preprocessing pipeline, save its fitted state and every inference dependency under ./working; code and predictions alone are not a saved model. This includes every executed fallback branch. Do not rely on an external cache, an absolute dataset path, a later download, or retraining for reuse.
Write ./working/model_manifest.json with schema_version=1, framework (nonempty string), files (list of {path, role, format, sha256}), metadata, and inference. Paths are relative to ./working, portable regular files (no symlinks); role is checkpoint or asset. At least one checkpoint is required. Supported checkpoint formats: joblib (fitted sklearn estimators/pipelines, including LightGBM sklearn wrappers), lightgbm (native Booster.save_model), torch_state_dict (torch.save of a state_dict or a dict containing state_dict/model_state_dict; include architecture code in inference.py), xgboost (native save_model), npz (nonempty finite learned parameter arrays). Use the matching real serialization, not placeholder files. Save the actual selected fitted model, plus scalers/encoders and backbone weights when used.
metadata must document input_schema, output_schema, preprocessing, class_labels (ordered list; [] for non-classification), and dependencies ({package: version}). inference must contain entrypoint (e.g. inference.py), input (e.g. smoke_input.json), expected_output (e.g. smoke_expected.json). Include these three files and all helper/sample/weight files in files as assets with SHA256. input is a JSON list of 1–8 fixed small public/synthetic examples (image examples may reference copies under ./working); expected_output is the finite numeric JSON prediction array produced by the in-memory fitted model on these same examples before serialization. Never include private labels. inference.py must expose predict(inputs, artifact_dir) returning the same numeric array, reload all saved state using artifact_dir, use CPU for this tiny probe, and work after the working directory is relocated. Do not train, download, spawn subprocesses, access input/ or depend on the original training script at import or inference time. BioMaker will independently reload the checkpoints and execute this function in a supervised offline probe, and compare its predictions (rtol=1e-4, atol=1e-6). Include all necessary architecture and preprocessing in the saved inference module. Hash files only after finishing writes.
For a genuinely stateless algorithm, no model manifest is required. For custom training APIs outside the common fit/partial_fit/backward/train API families, the same saved-model contract still applies; do not treat static training detection as permission to omit state.
For classification, keep class weights (one value per output class, in the fixed class order) separate from sample weights (one value per training row). CrossEntropyLoss(weight=...) needs exactly num_classes weights; assert this before training. Keep every required training row: default DataLoader drop_last=False, including a final short batch; choose normalization/batch handling that works for a batch of size one. If a different sample policy is explicitly required, document the omitted count and reason. Never silently substitute an unvalidated fallback branch."""


@dataclass(frozen=True)
class ModelArtifactCheck:
    ok: bool
    messages: list[str]
    evidence: dict[str, Any] | None = None


def training_calls(code: str | None) -> list[str]:
    """Return supported training-call evidence, excluding strings/comments."""
    if not code:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Execution, not this heuristic, owns syntax failures.
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name: item.name for item in node.names})
        elif isinstance(node, ast.ImportFrom):
            aliases.update(
                {
                    item.asname or item.name: f"{node.module}.{item.name}"
                    for item in node.names
                }
            )
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in {
                "fit",
                "partial_fit",
                "fit_transform",
                "backward",
                "fit_generator",
                "train_on_batch",
            }:
                found.add(function.attr)
            elif function.attr == "train" and isinstance(function.value, ast.Name):
                if aliases.get(function.value.id, "").split(".")[0] in {
                    "lightgbm",
                    "xgboost",
                }:
                    found.add(aliases[function.value.id] + ".train")
        elif isinstance(function, ast.Name):
            name = aliases.get(function.id, "")
            if name in {"lightgbm.train", "xgboost.train"}:
                found.add(name)
    return sorted(found)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact paths must be nonempty strings")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path must stay relative to working: {value}")
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"artifact must not use symlinks: {value}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"artifact is missing or empty: {value}")
    return path


def _read_small_json(path: Path) -> Any:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"JSON artifact exceeds 1 MiB: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_artifacts(
    workspace_dir: Path | str,
    *,
    candidate_code: str | None = None,
    required: bool = False,
    model_probe: Callable[[str], Any] | None = None,
) -> ModelArtifactCheck:
    working = Path(workspace_dir).resolve() / "working"
    manifest_path = working / MANIFEST_NAME
    calls = training_calls(candidate_code)
    if not manifest_path.exists() and not (required or calls):
        return ModelArtifactCheck(True, [])
    try:
        local_artifact(working, MANIFEST_NAME)
        manifest = _read_small_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("model_manifest.json requires schema_version=1")
        if (
            not isinstance(manifest.get("framework"), str)
            or not manifest["framework"].strip()
        ):
            raise ValueError("model manifest requires a framework")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("model manifest requires inference metadata")
        for name in ("input_schema", "output_schema", "preprocessing", "dependencies"):
            if not metadata.get(name):
                raise ValueError(f"model metadata is missing {name}")
        if not isinstance(metadata.get("class_labels"), list):
            raise ValueError(
                "model metadata requires ordered class_labels ([] for non-classification)"
            )
        if not isinstance(metadata["dependencies"], dict):
            raise ValueError("metadata.dependencies must map package names to versions")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("model manifest requires files")
        hashes = {}
        checkpoint_count = 0
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("model files must be objects")
            path = local_artifact(working, entry.get("path"))
            name = str(path.relative_to(working))
            if name in hashes:
                raise ValueError(f"duplicate model artifact: {name}")
            if entry.get("role") not in {"checkpoint", "asset"}:
                raise ValueError(f"unknown model artifact role: {name}")
            if entry["role"] == "checkpoint":
                checkpoint_count += 1
                if entry.get("format") not in {
                    "joblib",
                    "lightgbm",
                    "torch_state_dict",
                    "xgboost",
                    "npz",
                }:
                    raise ValueError(
                        f"unsupported checkpoint format: {entry.get('format')}"
                    )
            hashes[name] = sha256_file(path)
            if hashes[name] != entry.get("sha256"):
                raise ValueError(f"model artifact SHA256 mismatch: {name}")
        if checkpoint_count == 0:
            raise ValueError(
                "model manifest must include at least one fitted checkpoint"
            )
        inference = manifest.get("inference")
        if not isinstance(inference, dict):
            raise ValueError("model manifest requires an inference contract")
        for key in ("entrypoint", "input", "expected_output"):
            path = local_artifact(working, inference.get(key))
            if str(path.relative_to(working)) not in hashes:
                raise ValueError(f"inference {key} must be included in hashed files")
        inputs = _read_small_json(working / inference["input"])
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 8:
            raise ValueError("model smoke input must contain 1–8 examples")
        _read_small_json(working / inference["expected_output"])
        if model_probe is None:
            raise ValueError("supervised model reload probe is unavailable")
        report_path = working / PROBE_REPORT_NAME
        if report_path.exists() or report_path.is_symlink():
            report_path.unlink()
        nonce = uuid.uuid4().hex
        runner = Path(__file__).with_name("model_probe.py")
        code = (
            "import runpy\n"
            f"runpy.run_path({str(runner)!r})['run_probe']({str(working)!r}, {nonce!r})\n"
        )
        # Reuse Interpreter's existing environment, preflight, process-group
        # cleanup and deadline machinery, with a distinct temporary script/log
        # label. Do not mutate the candidate's interpreter or ExecutionResult.
        from ..interpreter import Interpreter

        bound_interpreter = getattr(model_probe, "__self__", None)
        if isinstance(bound_interpreter, Interpreter):
            probe_interpreter = Interpreter(
                bound_interpreter.working_dir,
                timeout=min(bound_interpreter.timeout, 60),
                format_tb_ipython=bound_interpreter.format_tb_ipython,
                agent_file_name=f".biomaker-model-probe-{nonce}.py",
                enforce_environment=bound_interpreter.enforce_environment,
                include_medical_packages=bound_interpreter.include_medical_packages,
                environment=bound_interpreter.environment,
                output_label=f"probe-{nonce[:6]}",
            )
            result = probe_interpreter.run(code)
        else:
            result = model_probe(code)
        output = getattr(result, "term_out", "")
        output = "\n".join(output) if isinstance(output, list) else str(output)
        if getattr(result, "exc_type", None) is not None:
            raise ValueError(
                f"saved model reload/inference failed ({result.exc_type}): {output[-1500:]}"
            )
        receipt_lines = [
            line.removeprefix("BIOMAKER_MODEL_PROBE ")
            for line in output.splitlines()
            if line.startswith("BIOMAKER_MODEL_PROBE ")
        ]
        if len(receipt_lines) != 1:
            raise ValueError("saved model probe did not produce exactly one receipt")
        report = json.loads(receipt_lines[0])
        if report.get("nonce") != nonce or report.get("ok") is not True:
            raise ValueError(
                "saved model probe did not produce a matching successful receipt"
            )
        if report.get("artifact_sha256") != hashes:
            raise ValueError("saved model probe verified different artifacts")
        # Verify against mutation during generated inference, including the manifest.
        for name, expected in hashes.items():
            if sha256_file(local_artifact(working, name)) != expected:
                raise ValueError(f"inference mutated model artifact: {name}")
        if _read_small_json(manifest_path) != manifest:
            raise ValueError("inference mutated model manifest")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        evidence = {
            **report,
            "detected_training_calls": calls,
            "detection_scope": "common_python_training_apis",
            "framework": manifest["framework"],
            "probe_exec_time": getattr(result, "exec_time", None),
            "checkpoint_sha256": {
                str(Path(item["path"])): item["sha256"]
                for item in files
                if item["role"] == "checkpoint"
            },
        }
        evidence.pop("nonce", None)
        return ModelArtifactCheck(
            True,
            [
                f"ModelArtifactValidation: reloaded {checkpoint_count} checkpoint(s), offline inference matched {len(inputs)} saved probe examples."
            ],
            evidence,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return ModelArtifactCheck(
            False,
            [f"ArtifactValidationError: saved model contract: {exc}"],
            {"detected_training_calls": calls, "required": bool(required or calls)},
        )
