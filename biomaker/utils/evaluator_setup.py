"""Set up and run the frozen evaluator used by generated solutions."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .. import backend
from .environment import environment_manifest_text, validate_code_environment
from .evaluator_adapter import ADAPTER_API, compile_adapter
from .metric_source import frozen_metric_source, METRIC_SOURCE_FILES
from .restoration_contract import fixed_restoration_spec
from .restoration_evaluator_template import _RESTORATION_DATA_ADAPTER
from .detection_contract import fixed_detection_spec
from .standard_data_adapters import (
    _REGRESSION_DATA_ADAPTER,
    _CLASSIFICATION_DATA_ADAPTER,
    _SEGMENTATION_DATA_ADAPTER,
    _DETECTION_DATA_ADAPTER,
    data_adapter_capabilities,
)
from .evaluator_protocol import (
    EVALUATOR_API_VERSION,
    metric_name,
    metric_definitions,
    requested_metric_definitions,
    requested_metric_contract,
    validate_task_metric_contract,
    requested_metrics,
    validate_score,
    validate_scoring_rows,
)
from .events import append_event
from .exp_guide import (
    build_exp_guide,
    evaluator_exp_guide_view,
    load_exp_guide,
)
from .experiment_contract import load_experiment_contract
from .metric_declarations import (
    PROTOCOL_DOCUMENTATION_FIELDS as _PROTOCOL_DOCUMENTATION_FIELDS,
    canonical_metric_contract,
    declared_metric_names,
    declared_metric_definitions,
    manifest_head_types as _manifest_head_types,
    manifest_primary_metric as _manifest_primary_metric,
    validate_declared_capabilities,
)
from .metric_registry import (
    is_classification_metric_supported,
    is_segmentation_metric_supported,
    metric_info,
    metric_maximize,
)
from .process_tree import (
    ProcessTreeInspectionError,
    ProcessTreeTracker,
    drain_uncaptured_process_group,
)
from .response import extract_code, truncate_text

logger = logging.getLogger("biomaker")

EXTERNAL_VALIDATION_PREFIX = "BioMakerExternalValidation:"
_RESULT_PREFIX = "EVALUATOR_RESULT_JSON="
_SPEC_PREFIX = "EVALUATOR_SPEC_JSON="
_ARTIFACT_REPORT_PREFIX = "EVALUATOR_ARTIFACT_REPORT_JSON="
_SCORE_ENV_VAR = "BIOMAKER_SCORE_VAL_PATH"
_EVALUATOR_CACHE_VERSION = 6


class EvaluatorSetupError(RuntimeError):
    """Raised when BioMaker cannot create a usable evaluator before search."""


@dataclass(frozen=True)
class PreparedEvaluator:
    path: Path
    prediction_spec: str


@dataclass(frozen=True)
class EvaluatorRunResult:
    ok: bool
    metric: float | None
    maximize: bool
    details: dict[str, Any]
    output: str
    error: str | None = None

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.details.get('score', {}).get('metrics', {}))

    @property
    def metric_specs(self) -> list[dict[str, Any]]:
        return list(self.details.get('score', {}).get('metric_specs', []))

    def log_line(self) -> str:
        details = dict(self.details)
        details.pop("artifact_report", None)
        payload = {
            "metric": self.metric,
            "maximize": self.maximize,
            "metrics": self.metrics,
            "metric_specs": self.metric_specs,
            "details": details,
        }
        return EXTERNAL_VALIDATION_PREFIX + " " + json.dumps(payload, sort_keys=True)


def clean_candidate_outputs(workspace_dir: Path | str) -> None:
    """Remove candidate-owned artifacts before running a candidate."""
    working_dir = Path(workspace_dir) / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    for path in working_dir.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def parse_external_validation(term_out: list[str] | str) -> tuple[float, bool] | None:
    text = "\n".join(term_out) if isinstance(term_out, list) else term_out
    for line in reversed(text.splitlines()):
        if not line.startswith(EXTERNAL_VALIDATION_PREFIX):
            continue
        payload = line[len(EXTERNAL_VALIDATION_PREFIX) :].strip()
        try:
            data = json.loads(payload)
            metric = data.get("metric")
            maximize = bool(data.get("maximize", True))
            if isinstance(metric, bool) or metric is None:
                return None
            return float(metric), maximize
        except Exception:
            return None
    return None


def parse_external_validation_details(
    term_out: list[str] | str,
) -> dict[str, Any] | None:
    text = "\n".join(term_out) if isinstance(term_out, list) else term_out
    for line in reversed(text.splitlines()):
        if not line.startswith(EXTERNAL_VALIDATION_PREFIX):
            continue
        payload = line[len(EXTERNAL_VALIDATION_PREFIX) :].strip()
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def parse_artifact_report(term_out: list[str] | str) -> dict[str, Any] | None:
    text = "\n".join(term_out) if isinstance(term_out, list) else term_out
    for line in reversed(text.splitlines()):
        if not line.startswith(_ARTIFACT_REPORT_PREFIX):
            continue
        payload = line[len(_ARTIFACT_REPORT_PREFIX) :].strip()
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _task_payload(task_desc: Any) -> str:
    if isinstance(task_desc, str):
        return task_desc
    return json.dumps(task_desc, indent=2, ensure_ascii=False, default=str)


def _exp_guide_prompt_view(
    workspace_dir: Path, hidden_score_path: Path | None = None
) -> dict[str, Any]:
    guide = load_exp_guide(workspace_dir)
    if guide is None:
        guide = build_exp_guide(workspace_dir)
    return evaluator_exp_guide_view(guide)


def _preparation_context(workspace_dir: Path, hidden_score_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Complete public input contract, independent of earlier evaluator outputs."""
    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    declared = {key: manifest[key] for key in (
        "sample_unit", "key_columns", "task_heads", "metric_contract", "prediction_contract",
        "data_semantics", "row_representation", "semantic_assumptions", "sample_universe_summary",
        "input_file_reference_columns",
    ) if key in manifest}
    declared = json.loads(_sanitize_private_paths(json.dumps(declared, default=str), hidden_score_path))
    guide = _exp_guide_prompt_view(workspace_dir, hidden_score_path)
    guide["prediction"] = declared.get("prediction_contract", {})
    guide["metric"] = declared.get("metric_contract", {})
    observations = build_exp_guide(workspace_dir).get("samples", {})
    return guide, declared, observations


def _evaluator_prompt(
    task_desc: Any, workspace_dir: Path, hidden_score_path: Path | None = None
) -> dict[str, Any]:
    guide, declared, observations = _preparation_context(workspace_dir, hidden_score_path)
    return {
        "Introduction": (
            "You are writing a custom-metric evaluator because a requested metric "
            "has no registered trusted Harness implementation. This route is never "
            "used solely for a data-schema mismatch. "
            "This evaluator is created once before solution search. Candidate solutions "
            "will train models and write prediction artifacts; BioMaker will call this evaluator "
            "after each candidate finishes, so all candidates are scored with the same code."
        ),
        "Task description": _task_payload(task_desc),
        "Experiment guide": guide,
        "Prepared data and metric declarations": declared,
        "Public sample observations": observations,
        "Environment contract": environment_manifest_text(include_medical=True),
        "Evaluator requirements": [
            "Write exactly one Python file.",
            "Use only Python standard library and allowed installed packages.",
            "Do not train a model. Do not install packages. Do not use network access.",
            "Do not use Python eval() or exec(). Parse serialized values with json.loads or ast.literal_eval.",
            "Use the Experiment guide as the source of truth for public splits, hidden answers, visibility boundaries, and output paths.",
            "Define prediction_spec() returning a concise JSON-serializable dict with these top-level keys: validation_artifact, test_submission, key_columns, prediction_columns, metrics. key_columns may be [] only when row order is the contract.",
            "Evaluator API v3: prediction_spec() must include evaluator_api_version=3 and a nonempty metrics list of {name, maximize} objects. Every metric has its own boolean direction; custom metrics require an explicit direction. Preserve every requested objective; there is no primary or secondary role. Declare metric parameters (thresholds, averaging, class mapping, coordinate conventions) explicitly in the public spec; use exactly those settings on every split.",
            "Define score(prediction_path, truth_path, expected_path). This is the ONLY scoring implementation for both validation and private test. Read predictions, hidden truth, and public expected rows from these explicit paths. Never hardcode validation filenames, row counts, or the score-validation environment variable inside score(). Never infer the split from its size. Referenced relative image/mask/annotation paths remain workspace-relative. Return {'metrics': {metric_name: finite_number, ...}} and optional diagnostics. Include every declared objective. Do not return a scalar metric or any rank: the Journal computes average ranks across candidates. For detection, explicitly freeze the confidence threshold or operating-point rule for Recall as well as IoU and box format; do not tune the rule on private test answers.",
            "score_val() must only delegate to score() with the validation prediction path, os.environ['BIOMAKER_SCORE_VAL_PATH'], and the public validation CSV. Do not duplicate metric computation for different splits. validate_submission() checks format only and must not read private test answers.",
            "test_submission must be exactly './working/submission.csv'.",
            "prediction_columns means the CSV columns containing predictions or referenced prediction files. Do not include key columns in prediction_columns.",
            "The candidate validation artifact must contain predictions for rows in the Experiment guide's score validation split, usually ./input/biomaker_splits/score_val.csv and sometimes legacy ./input/biomaker_splits/agent_val.csv.",
            "When resolving file paths from split CSVs or hidden truth, values starting with `input/` or `./input/` are already workspace-relative paths; do not prepend `./input` a second time.",
            "The candidate validation artifact must follow the same public prediction contract as the test submission: key_columns plus prediction_columns only. Do not require candidates to output validation ground-truth columns such as label, class, target, or true_class. score_val() must load hidden score labels from os.environ['BIOMAKER_SCORE_VAL_PATH'] and align them to the validation artifact by key columns.",
            "When reading candidate artifacts, validate the required columns and then restrict the dataframe to key_columns plus prediction_columns before merging with ground truth. Extra candidate columns such as class, label, target, or true_class must not collide with ground-truth columns or change scoring.",
            "For tabular score_val(), use this safe merge pattern: copy the hidden ground-truth label column to a private column named `y_true_private`, build `truth_view = hidden_answers[key_columns + ['y_true_private']]`, build `pred_view = predictions[key_columns + prediction_columns]`, convert every key column in both views to string before merging, merge those restricted views only, and read labels with dataframe/series indexing such as `merged['y_true_private']` or `row['y_true_private']`. Never rely on tuple attribute access for helper columns, and never read `merged['class']`, `merged['label']`, `merged['target']`, or candidate-supplied label-like columns after merge.",
            "For tabular validate_submission(), apply the same key-column string normalization and restrict submission data to key_columns plus prediction_columns before checking duplicates, missing keys, or shapes.",
            "For probability outputs, include class_labels when prediction_columns do not directly encode class names; for single-column binary probability outputs, include positive_class.",
            "The submission contract must preserve enough information for the requested user-only test metrics. For classification with AUC, require probabilities rather than only hard labels.",
            f"Define score_val() with no required arguments. It must read hidden score truth from os.environ['{_SCORE_ENV_VAR}'], read the candidate validation artifact from ./working, validate the contract, and return a dict containing the complete raw metrics mapping and optional diagnostics.",
            "Define validate_submission() with no required arguments. It must validate ./working/submission.csv and any referenced files against the same contract, returning a dict or raising a clear exception.",
            "For tabular prediction artifacts, align rows by the declared key columns after normalizing keys to strings; reject missing keys, duplicate keys, extra keys, bad shapes, and invalid numeric values.",
            "For probability outputs, validate with numeric tolerance and return a clear error when rows are not valid probabilities.",
            "If any requested metric is mathematically undefined, raise a clear error naming that metric. Never substitute accuracy for AUC or silently omit a requested metric.",
            f"Define self_test() with no required arguments. It must derive dummy prediction artifacts from the real public score validation and agent_test split CSVs, exactly follow prediction_spec(), call score_val() with os.environ['{_SCORE_ENV_VAR}'] available, call validate_submission(), assert every declared metric is finite numeric, then print EVALUATOR_OK.",
            "self_test() MUST leave its validation/submission CSVs and ALL referenced files available after returning. Harness reads them again for a relocated-path scoring probe and owns cleanup after every preparation check finishes. Do not delete them or place referenced files in a TemporaryDirectory context that closes before returning.",
            "Keep the evaluator simple and deterministic. Implement and verify only unavailable metric definitions; reuse supplied trusted_score_samples(samples, spec) for registered metrics. Its canonical sample/parameter interface is documented below. Do not reimplement PSNR, SSIM, Dice, AJI, HD95, accuracy, AUC, AP50, or Recall.",
            "For segmentation masks, support mask files referenced from CSV. The prediction spec should make this explicit with prediction_type='mask_file_path' or file_reference_columns for the mask path column. Binary Dice should treat nonzero pixels as foreground. AJI/instance metrics should use integer connected/instance labels if the task asks for them.",
        ],
        "Trusted metric API": _trusted_metric_api(),
        "Output format": (
            "Return a brief note followed by exactly one markdown Python code block. "
            "No extra code blocks."
        ),
    }


def _trusted_metric_api() -> str:
    from .trusted_metrics import METRIC_API_DESCRIPTION

    return (
        "trusted_score_samples(samples, spec) is supplied by Harness in this standalone file. "
        "samples is an iterable of mappings with prediction, target and optional group, "
        "input_shape, spacing, prediction_spacing. It returns raw metrics/n and diagnostics (never ranks). "
        "Use a spec naming only the registered metrics for that call, then combine your "
        "independently verified new metric in the final metrics mapping.\n" + METRIC_API_DESCRIPTION
    )


def _adapter_prompt(task_desc: Any, workspace_dir: Path, hidden_score_path: Path) -> dict[str, Any]:
    from .trusted_metrics import METRIC_API_DESCRIPTION, metric_capabilities

    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    guide, declared, observations = _preparation_context(workspace_dir, hidden_score_path)
    return {
        "Introduction": (
            "Write only a thin data adapter for the task's trusted metrics. "
            "The task and experiment protocol determine metric semantics. "
            "Harness supplies tested metric functions and owns scoring, alignment, "
            "aggregation, private answers, preparation probes, and frozen entry points. "
            "A dataset schema mismatch is an adapter issue, never a reason to rewrite metrics."
        ),
        "Task description": _task_payload(task_desc),
        "Experiment guide": guide,
        "Prepared data and metric declarations": declared,
        "Public sample observations": observations,
        "Environment contract": environment_manifest_text(include_medical=True),
        "Data adapter API": ADAPTER_API,
        "Trusted metric API and parameters": METRIC_API_DESCRIPTION,
        "Metric capability catalog": metric_capabilities(),
        "Built-in data reader catalog": data_adapter_capabilities(),
        "Complete reference adapter (illustrative schema only)": _adapter_reference_example(task_desc, manifest),
        "Protocol requirements": (
            "Preserve every task-specified metric, parameter, crop, group, and aggregation. "
            "Do not choose alternative semantics merely to pass a probe. Unsupported "
            "metric semantics must be reported explicitly. Implement only data decoding, "
            "path/column mapping and representation conversion. No hidden truth values "
            "or private test answers are provided to you."
        ),
        "Output format": "Return a brief note and exactly one markdown Python code block.",
    }


def _adapter_reference_example(task_desc: Any, manifest: dict[str, Any]) -> str:
    primary, _ = requested_metrics(_task_payload(task_desc))
    info = metric_info(primary or _manifest_primary_metric(manifest))
    family = info.task_family if info else "classification"
    spec = dict(evaluator_api_version=3, key_columns=["sample_id"],
                validation_artifact="./working/validation_predictions.csv",
                test_submission="./working/submission.csv", maximize=True)
    if family == "regression":
        spec.update(task_type=family, primary_metric="mae", secondary_metrics=["rmse"],
                    maximize=False, prediction_columns=["estimated_value"], answer_columns=["reference_value"],
                    target_shape=[], aggregation="mean_samples", multioutput="uniform_average", rounding="none")
        template = _REGRESSION_DATA_ADAPTER
    elif family == "image_restoration":
        spec.update(task_type=family, primary_metric="psnr", secondary_metrics=["ssim"],
                    prediction_columns=["estimate_file"], answer_columns=["reference_file"],
                    restoration_evaluation=dict(input_column="observed_file", target_column="reference_file",
                        data_range=255, crop_size=None, group_column=None, groups=None, aggregation="mean_images"))
        template = _RESTORATION_DATA_ADAPTER
    elif family == "segmentation":
        spec.update(task_type=family, primary_metric="binary_dice", secondary_metrics=["aji"],
                    prediction_columns=["estimated_mask"], answer_columns=["reference_mask"],
                    image_column="observed_image")
        contract = requested_metric_contract(_task_payload(task_desc)) or {}
        if contract:
            from .trusted_metrics import metric_capabilities
            allowed = set(metric_capabilities()['families']['segmentation']['allowed_contract_fields'])
            for key in ('primary_metric', 'secondary_metrics', 'maximize'):
                spec.pop(key, None)
            spec.update({key: value for key, value in contract.items() if key in allowed})
            names = {item['name'] for item in metric_definitions(spec)}
            auxiliary = {}
            if names & {'cldice', 'multiclass_cldice', 'tree_length_detected', 'branch_detected'}:
                auxiliary['skeleton'] = 'reference_skeleton'
            if names & {'branch_detected', 'branch_accuracy', 'branch_f1', 'subtree_consistency', 'topological_distance'}:
                auxiliary['parsing'] = 'reference_branches'
            if names & {'subtree_consistency', 'topological_distance'}:
                auxiliary['graph'] = 'reference_tree'
            spec.update(target_column='reference_mask', auxiliary_columns=auxiliary,
                        answer_columns=['reference_mask', *auxiliary.values()])
        template = _SEGMENTATION_DATA_ADAPTER
    elif family == "detection":
        schema = dict(class_key="category", coordinate_keys=["bbox"], box_format="xyxy")
        spec.update(task_type=family, primary_metric="ap50", secondary_metrics=["recall"],
                    prediction_columns=["detections"], answer_columns=["reference_boxes"],
                    prediction_schema=schema, truth_schema=schema)
        template = _DETECTION_DATA_ADAPTER
    else:
        spec.update(task_type="binary", primary_metric="auc", secondary_metrics=["accuracy"],
                    prediction_columns=["probability_no", "probability_yes"], answer_columns=["class_name"],
                    class_labels=["no", "yes"], positive_class="yes")
        template = _CLASSIFICATION_DATA_ADAPTER
    spec['metrics'] = metric_definitions(spec)
    for key in ('primary_metric', 'secondary_metrics', 'maximize'):
        spec.pop(key, None)
    return template.replace("__BIOMAKER_SPEC__", repr(spec)).strip()


def _run_python(
    workspace_dir: Path,
    code: str,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    run_env = None
    if env is not None:
        run_env = os.environ.copy()
        run_env.update(env)
    command = [sys.executable, "-c", code]
    process: subprocess.Popen[str] | None = None
    process_tree: ProcessTreeTracker | None = None
    stdout_capture: Any | None = None
    stderr_capture: Any | None = None
    uncaptured_drained = False
    capture_stack = ExitStack()
    try:
        stdout_capture = capture_stack.enter_context(
            tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        )
        stderr_capture = capture_stack.enter_context(
            tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        )
        popen_kwargs: dict[str, Any] = {
            "cwd": workspace_dir,
            "stdout": stdout_capture,
            "stderr": stderr_capture,
            "text": True,
            "env": run_env,
        }
        if os.name == "posix":
            popen_kwargs["process_group"] = 0
        process = subprocess.Popen(command, **popen_kwargs)
        try:
            process_tree = ProcessTreeTracker.capture(process.pid)
        except (OSError, ProcessLookupError, ProcessTreeInspectionError, ValueError):
            drain_uncaptured_process_group(process)
            uncaptured_drained = True
            raise RuntimeError("Could not capture evaluator process identity") from None

        started_at = time.monotonic()
        while True:
            process_tree.refresh_best_effort()
            if process_tree.root_has_exited(process):
                process_tree.terminate_and_wait(root_process=process)
                stdout_capture.seek(0)
                stderr_capture.seek(0)
                stdout = stdout_capture.read()
                stderr = stderr_capture.read()
                return subprocess.CompletedProcess(
                    command,
                    process.returncode,
                    stdout,
                    stderr,
                )
            remaining = timeout - (time.monotonic() - started_at)
            if remaining <= 0:
                process_tree.terminate_and_wait(root_process=process)
                stdout_capture.seek(0)
                stderr_capture.seek(0)
                stdout = stdout_capture.read()
                stderr = stderr_capture.read()
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=stdout,
                    stderr=stderr,
                )
            time.sleep(min(0.25, max(0.05, remaining)))
    finally:
        try:
            if process_tree is not None:
                process_tree.terminate_and_wait(root_process=process)
            elif process is not None and not uncaptured_drained:
                drain_uncaptured_process_group(process)
        finally:
            capture_stack.close()


def _uses_eval_or_exec(code: str) -> bool:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec"}:
                return True
    return False


def _hidden_score_env(
    workspace_dir: Path, hidden_score_path: Path | str | None = None
) -> dict[str, str]:
    truth_path = Path(hidden_score_path) if hidden_score_path is not None else None
    if truth_path is None:
        truth_path = workspace_dir / "private_gt" / "score_val.csv"
    if not truth_path.is_file():
        input_path = workspace_dir / "input"
        try:
            truth_path = input_path.resolve().parent / "private_gt" / "score_val.csv"
        except Exception:
            pass
    return {_SCORE_ENV_VAR: str(truth_path)}


def _sanitize_private_paths(
    text: str, hidden_score_path: Path | str | None = None
) -> str:
    replacements = {"<hidden_score_path>": set(), "<private_gt_dir>": set()}
    if hidden_score_path is not None:
        path = Path(hidden_score_path)
        replacements["<hidden_score_path>"].update({str(path), str(path.resolve())})
        replacements["<private_gt_dir>"].update(
            {str(path.parent), str(path.parent.resolve())}
        )
    sanitized = text
    for token, values in replacements.items():
        for value in sorted(values, key=len, reverse=True):
            if value:
                sanitized = sanitized.replace(value, token)
    return sanitized


def _run_self_test(
    workspace_dir: Path,
    timeout: int,
    hidden_score_path: Path | str | None = None,
) -> str:
    code = """
import evaluator
import json
import os
import shutil
import tempfile
from pathlib import Path
required = ['prediction_spec', 'score', 'score_val', 'validate_submission', 'self_test']
missing = [name for name in required if not hasattr(evaluator, name)]
if missing:
    raise RuntimeError('evaluator missing required functions: ' + ', '.join(missing))
spec = evaluator.prediction_spec()
if isinstance(spec, str):
    spec = json.loads(spec)
if not isinstance(spec, dict):
    raise RuntimeError('prediction_spec() must return a dict')
for key in ['validation_artifact', 'test_submission', 'key_columns']:
    if key not in spec:
        raise RuntimeError('prediction_spec() missing key: ' + key)
if 'prediction_columns' not in spec and 'file_reference_columns' not in spec:
    raise RuntimeError('prediction_spec() must declare prediction_columns or file_reference_columns')
try:
    probe = evaluator.self_test()
    if spec.get('adapter_api_version') == 1:
        sources = [probe[k] for k in ('prediction_path', 'truth_path', 'expected_path')]
    else:
        sources = [spec['validation_artifact'], os.environ['BIOMAKER_SCORE_VAL_PATH'],
                   spec.get('score_split', './input/biomaker_splits/score_val.csv')]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, source in enumerate(sources):
            if not Path(source).is_file():
                raise RuntimeError('self_test() must retain CSVs and referenced files until Harness finishes all probes; cleanup belongs to Harness.')
            target = Path(tmp) / (str(i) + '.csv')
            shutil.copyfile(source, target)
            paths.append(str(target))
        os.environ.pop('BIOMAKER_SCORE_VAL_PATH', None)
        expected_error = probe.get('score_error') if spec.get('adapter_api_version') == 1 else None
        score_error = None
        try:
            score = evaluator.score(*paths)
        except Exception as exc:
            if (not expected_error or not isinstance(exc, evaluator.UndefinedMetricError)
                    or {'metric': exc.metric, 'reason': exc.reason} != expected_error):
                raise
            score = None
            score_error = expected_error
        if spec.get('adapter_api_version') == 1 and (score != probe['score'] or score_error != expected_error):
            raise RuntimeError('Adapter probe changed when paths were relocated or test fixtures were written; preserve all referenced files and split-independent interpretation.')
        print('EVALUATOR_PROBE_JSON=' + json.dumps({'spec': spec, 'score': score,
            'score_error': score_error, 'structural_probe': probe.get('structural_probe') if isinstance(probe, dict) else None}))
finally:
    if spec.get('adapter_api_version') == 1:
        evaluator.cleanup_self_test()
print('EVALUATOR_OK')
"""
    completed = _run_python(
        workspace_dir,
        code,
        timeout=timeout,
        env=_hidden_score_env(workspace_dir, hidden_score_path),
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 or "EVALUATOR_OK" not in output:
        output = _sanitize_private_paths(output, hidden_score_path)
        raise EvaluatorSetupError("Evaluator self-test failed.\n" + output[-4000:])
    probes = [
        line.removeprefix("EVALUATOR_PROBE_JSON=")
        for line in output.splitlines()
        if line.startswith("EVALUATOR_PROBE_JSON=")
    ]
    if not probes:
        raise EvaluatorSetupError(
            "Evaluator did not pass the split-independent scoring probe."
        )
    probe = json.loads(probes[-1])
    _validate_scoring_contract(probe["spec"])
    if probe.get("score_error"):
        if probe["spec"].get("adapter_api_version") != 1 or not (probe.get("structural_probe") or {}).get("validated_samples"):
            raise EvaluatorSetupError("Undefined dummy metric requires a completed Harness structural probe.")
    else:
        validate_score(probe["score"], probe["spec"])
    return output


def _validate_scoring_contract(spec: dict[str, Any], task_desc: Any = None, *, allow_legacy: bool = False) -> None:
    version = spec.get('evaluator_api_version')
    if version != EVALUATOR_API_VERSION and not (allow_legacy and version == 2):
        raise EvaluatorSetupError('prediction_spec must declare evaluator_api_version=3.')
    if version == 3 and 'metrics' not in spec:
        raise EvaluatorSetupError('Evaluator API v3 requires a nonempty metrics list.')
    try:
        definitions = metric_definitions(spec, allow_legacy_unknown=allow_legacy and version == 2)
    except ValueError as exc:
        raise EvaluatorSetupError(str(exc)) from exc
    try:
        validate_task_metric_contract(spec, _task_payload(task_desc))
    except ValueError as exc:
        raise EvaluatorSetupError(str(exc)) from exc
    primary, requested_secondary = requested_metrics(_task_payload(task_desc))
    if primary and set([primary, *requested_secondary]) != {item['name'] for item in definitions}:
        raise EvaluatorSetupError(
            'Evaluator metrics must match every requested metric; objectives cannot be added or omitted.'
        )


def _load_prediction_spec(workspace_dir: Path) -> str:
    code = (
        "import json, evaluator\n"
        "spec = evaluator.prediction_spec()\n"
        f"print('{_SPEC_PREFIX}' + json.dumps(spec, ensure_ascii=False, default=str))\n"
    )
    completed = _run_python(workspace_dir, code, timeout=30)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise EvaluatorSetupError(
            "Could not read evaluator prediction_spec.\n" + output
        )
    for line in output.splitlines():
        if line.startswith(_SPEC_PREFIX):
            return line[len(_SPEC_PREFIX) :]
    raise EvaluatorSetupError("Evaluator prediction_spec did not emit JSON.")


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _path_from_spec_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ["path", "output_path", "file"]:
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw
    return None


def _nested_dict(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key)
    return value if isinstance(value, dict) else {}


def _prediction_columns_from_raw_spec(
    spec: dict[str, Any], key_columns: list[str]
) -> list[str]:
    submission_spec = _nested_dict(spec, "test_submission")
    for container in [spec, submission_spec]:
        for key in [
            "prediction_columns",
            "file_reference_columns",
            "columns",
        ]:
            columns = _list_of_strings(container.get(key))
            if columns:
                return [column for column in columns if column not in set(key_columns)]

    format_spec = _nested_dict(spec, "format")
    raw_columns = format_spec.get("columns")
    if isinstance(raw_columns, list):
        return [
            str(column) for column in raw_columns if str(column) not in set(key_columns)
        ]
    if isinstance(raw_columns, dict):
        return [
            str(column) for column in raw_columns if str(column) not in set(key_columns)
        ]
    return []


def _primary_metric_from_raw_spec(spec: dict[str, Any]) -> str | None:
    submission_spec = _nested_dict(spec, "test_submission")
    for container in [spec, submission_spec]:
        value = container.get("primary_metric")
        if isinstance(value, str) and value.strip():
            return value.strip()
    metrics = spec.get("metrics")
    if isinstance(metrics, list) and metrics:
        return metric_definitions(spec)[0]['name']
    return None


def _normalize_prediction_spec_json(prediction_spec: str) -> str:
    """Validate and normalize the evaluator's public artifact contract."""
    try:
        spec = json.loads(prediction_spec)
    except Exception as exc:
        raise EvaluatorSetupError(f"prediction_spec() did not return valid JSON: {exc}")
    if not isinstance(spec, dict):
        raise EvaluatorSetupError("prediction_spec() must return a JSON object.")

    validation_artifact = _path_from_spec_value(
        spec.get("validation_artifact")
        or spec.get("validation_predictions")
        or spec.get("val_predictions")
    )
    test_submission = _path_from_spec_value(spec.get("test_submission"))
    key_columns = _list_of_strings(spec.get("key_columns"))
    if not key_columns:
        key_columns = _list_of_strings(
            _nested_dict(spec, "test_submission").get("key_columns")
        )
    file_reference_columns = _list_of_strings(spec.get("file_reference_columns"))
    if not file_reference_columns:
        file_reference_columns = _list_of_strings(
            _nested_dict(spec, "test_submission").get("file_reference_columns")
        )
    prediction_columns = _prediction_columns_from_raw_spec(spec, key_columns)
    primary_metric = _primary_metric_from_raw_spec(spec)

    if not test_submission:
        raise EvaluatorSetupError(
            "prediction_spec() must declare test_submission, usually './working/submission.csv'."
        )
    if test_submission.removeprefix("./") != "working/submission.csv":
        raise EvaluatorSetupError(
            "prediction_spec() test_submission must be exactly './working/submission.csv'."
        )
    if "key_columns" not in spec and "key_columns" not in _nested_dict(
        spec, "test_submission"
    ):
        raise EvaluatorSetupError(
            "prediction_spec() must declare key_columns. Use [] only when row order is the contract."
        )
    if not prediction_columns:
        raise EvaluatorSetupError(
            "prediction_spec() must declare non-empty prediction_columns or file_reference_columns."
        )
    if not primary_metric:
        raise EvaluatorSetupError(
            "prediction_spec() must declare a nonempty metrics list with metric names and directions."
        )

    normalized: dict[str, Any] = {
        **spec,
        "validation_artifact": validation_artifact
        or "./working/validation_predictions.csv",
        "test_submission": test_submission,
        "key_columns": key_columns,
        "prediction_columns": prediction_columns,
    }
    if spec.get('evaluator_api_version') == 3 or 'metrics' in spec:
        normalized['metrics'] = metric_definitions(spec)
        for key in ('primary_metric', 'secondary_metrics', 'maximize'):
            normalized.pop(key, None)
    else:
        normalized['primary_metric'] = primary_metric
    if file_reference_columns:
        normalized["file_reference_columns"] = [
            column
            for column in file_reference_columns
            if column not in set(key_columns)
        ]
    metric_lower = primary_metric.lower()
    if any(token in metric_lower for token in ["aji", "dice", "mask", "segmentation"]):
        normalized.setdefault("prediction_type", "mask_file_path")
    for optional_key in [
        "constraints",
        "notes",
        "class_labels",
        "positive_class",
        "prediction_type",
        "task_type",
        "score_split",
    ]:
        if optional_key in spec:
            normalized[optional_key] = spec[optional_key]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _should_run_tabular_contract_test(spec: dict[str, Any]) -> bool:
    metric = _primary_metric_from_raw_spec(spec) or ''
    has_class_labels = isinstance(spec.get("class_labels"), list)
    tabular_metric = any(
        token in metric
        for token in ["auc", "accuracy", "classification", "logloss", "f1"]
    )
    return bool(tabular_metric or has_class_labels)


def _write_dummy_tabular_artifact(
    split_csv: Path,
    output_path: Path,
    key_columns: list[str],
    prediction_columns: list[str],
    class_labels: list[Any] | None,
    include_extra_label_like_columns: bool = False,
) -> None:
    split_df = pd.read_csv(split_csv)
    output: dict[str, Any] = {}
    for column in key_columns:
        if column not in split_df.columns:
            raise EvaluatorSetupError(
                f"prediction_spec key column `{column}` is not present in {split_csv}."
            )
        output[column] = split_df[column].tolist()

    if class_labels and len(class_labels) == len(prediction_columns):
        value = 1.0 / len(prediction_columns)
        for column in prediction_columns:
            output[column] = [value] * len(split_df)
    elif len(prediction_columns) == 1:
        output[prediction_columns[0]] = [0.5] * len(split_df)
    else:
        value = 1.0 / len(prediction_columns)
        for column in prediction_columns:
            output[column] = [value] * len(split_df)

    if include_extra_label_like_columns:
        for column in ["class", "label", "target", "true_class"]:
            output[column] = [0] * len(split_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(output_path, index=False)


def _run_public_artifact_contract_test(
    workspace_dir: Path,
    prediction_spec: str,
    timeout: int,
    hidden_score_path: Path | str | None = None,
) -> None:
    """Reject evaluators whose score_val requires hidden columns in predictions."""
    try:
        spec = json.loads(prediction_spec)
    except Exception as exc:
        raise EvaluatorSetupError(f"Could not parse normalized prediction_spec: {exc}")
    if not isinstance(spec, dict) or spec.get("adapter_api_version") == 1 or not _should_run_tabular_contract_test(spec):
        return

    key_columns = _list_of_strings(spec.get("key_columns"))
    prediction_columns = _list_of_strings(spec.get("prediction_columns"))
    class_labels = spec.get("class_labels")
    if not isinstance(class_labels, list):
        class_labels = None

    for artifact_key in ["validation_artifact", "test_submission"]:
        artifact_value = str(spec.get(artifact_key, "")).removeprefix("./")
        if not artifact_value.startswith("working/"):
            raise EvaluatorSetupError(
                f"prediction_spec {artifact_key} must point under ./working for "
                "public artifact contract tests."
            )

    working_dir = workspace_dir / "working"
    original_working_existed = working_dir.exists()
    backup_root = Path(
        tempfile.mkdtemp(prefix=".evaluator_contract_test_", dir=workspace_dir)
    )
    backup_working = backup_root / "working"
    try:
        if original_working_existed:
            shutil.move(str(working_dir), str(backup_working))
        working_dir.mkdir(parents=True, exist_ok=True)

        score_split = workspace_dir / str(
            spec.get("score_split", "./input/biomaker_splits/score_val.csv")
        ).removeprefix("./")
        _write_dummy_tabular_artifact(
            score_split,
            workspace_dir / str(spec["validation_artifact"]).removeprefix("./"),
            key_columns,
            prediction_columns,
            class_labels,
            include_extra_label_like_columns=True,
        )
        _write_dummy_tabular_artifact(
            workspace_dir / "input" / "biomaker_splits" / "agent_test.csv",
            workspace_dir / str(spec["test_submission"]).removeprefix("./"),
            key_columns,
            prediction_columns,
            class_labels,
            include_extra_label_like_columns=True,
        )

        code = (
            "import evaluator\n"
            "score = evaluator.score_val()\n"
            "if not isinstance(score.get('metric'), (int, float)):\n"
            "    raise RuntimeError('score_val() metric must be numeric: ' + repr(score))\n"
            "evaluator.validate_submission()\n"
        )
        completed = _run_python(
            workspace_dir,
            code,
            timeout=timeout,
            env=_hidden_score_env(workspace_dir, hidden_score_path),
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            output = _sanitize_private_paths(output, hidden_score_path)
            raise EvaluatorSetupError(
                "Evaluator public artifact contract test failed. "
                "score_val() must accept validation predictions containing only "
                "key_columns plus prediction_columns from prediction_spec(); it should "
                f"load hidden score labels from ${_SCORE_ENV_VAR}.\n" + output[-4000:]
            )
    finally:
        if working_dir.exists():
            shutil.rmtree(working_dir)
        if original_working_existed:
            shutil.move(str(backup_working), str(working_dir))
        shutil.rmtree(backup_root, ignore_errors=True)


def _resolve_candidate_path(
    workspace_dir: Path, raw_value: Any, base_dir: Path | None = None
) -> Path:
    path = Path(str(raw_value).strip())
    if path.is_absolute():
        return path
    normalized = Path(str(path).removeprefix("./"))
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / normalized)
    candidates.extend(
        [
            workspace_dir / normalized,
            workspace_dir / "working" / normalized,
            workspace_dir / "input" / normalized,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else workspace_dir / normalized


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _summarize_series(series: pd.Series) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
    }
    non_null = series.dropna()
    if non_null.empty:
        summary["sample_values"] = []
        return summary
    if pd.api.types.is_numeric_dtype(non_null):
        values = pd.to_numeric(non_null, errors="coerce").dropna()
        if not values.empty:
            summary.update(
                {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                }
            )
    summary["sample_values"] = [
        _safe_scalar(value) for value in non_null.head(3).tolist()
    ]
    return summary


def _summarize_array_file(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return summary | {"error": "file not found"}
    summary["size_bytes"] = path.stat().st_size
    try:
        with Image.open(path) as image:
            arr = np.asarray(image)
            unique_preview = np.unique(
                arr[: min(arr.shape[0], 256), : min(arr.shape[1], 256)]
            )
            result: dict[str, Any] = {
                "kind": "image_array",
                "mode": image.mode,
                "size": list(image.size),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": _safe_scalar(arr.min()) if arr.size else None,
                "max": _safe_scalar(arr.max()) if arr.size else None,
                "sample_unique_count": int(len(unique_preview)),
                "sample_unique_values": [
                    _safe_scalar(v) for v in unique_preview[:8].tolist()
                ],
            }
            if arr.size:
                result["nonzero_fraction"] = float(np.count_nonzero(arr) / arr.size)
                if arr.ndim <= 2:
                    values, counts = np.unique(arr, return_counts=True)
                    if len(values):
                        idx = int(np.argmax(counts))
                        result["most_common_value"] = _safe_scalar(values[idx])
                        result["most_common_fraction"] = float(counts[idx] / arr.size)
            summary.update(result)
            return summary
    except Exception as image_exc:
        summary["image_read_error"] = str(image_exc)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        summary.update(
            {
                "kind": "text_file",
                "preview": truncate_text(text, max_chars=300),
            }
        )
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _align_truth_rows(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    key_columns: list[str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not key_columns:
        n = min(len(pred_df), len(truth_df))
        return truth_df.iloc[:n].reset_index(drop=True), {
            "method": "row_order",
            "truth_rows_used": n,
        }
    missing = [
        column
        for column in key_columns
        if column not in pred_df.columns or column not in truth_df.columns
    ]
    if missing:
        return None, {"method": "key_columns", "missing_key_columns": missing}
    pred_keys = pred_df[key_columns].astype(str)
    truth_lookup = truth_df.copy()
    for column in key_columns:
        truth_lookup[column] = truth_lookup[column].astype(str)
    merged = pred_keys.merge(
        truth_lookup.reset_index().rename(columns={"index": "__truth_index"}),
        on=key_columns,
        how="left",
    )
    missing_truth = int(merged["__truth_index"].isna().sum())
    if missing_truth:
        return None, {
            "method": "key_columns",
            "matched_rows": int(len(merged) - missing_truth),
            "missing_truth_rows": missing_truth,
        }
    indices = merged["__truth_index"].astype(int).tolist()
    return truth_df.iloc[indices].reset_index(drop=True), {
        "method": "key_columns",
        "matched_rows": len(indices),
    }


def inspect_candidate_artifacts(
    workspace_dir: Path | str,
    max_samples: int = 3,
) -> dict[str, Any]:
    """Best-effort deterministic summary of artifacts consumed by the evaluator."""
    workspace_dir = Path(workspace_dir)
    report: dict[str, Any] = {
        "available": True,
        "warnings": [],
    }
    try:
        spec_path = workspace_dir / "evaluator_prediction_spec.json"
        if not spec_path.is_file():
            return {
                "available": False,
                "error": "evaluator_prediction_spec.json missing",
            }
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            return {"available": False, "error": "prediction spec is not a JSON object"}
        report["prediction_spec"] = {
            key: spec.get(key)
            for key in [
                "validation_artifact",
                "key_columns",
                "prediction_columns",
                "primary_metric",
                "metrics",
                "prediction_type",
            ]
            if key in spec
        }

        validation_artifact = spec.get("validation_artifact")
        if not isinstance(validation_artifact, str) or not validation_artifact:
            return report | {
                "available": False,
                "error": "validation_artifact missing from prediction spec",
            }
        artifact_path = _resolve_candidate_path(workspace_dir, validation_artifact)
        report["validation_artifact"] = {
            "path": str(artifact_path),
            "exists": artifact_path.is_file(),
        }
        if not artifact_path.is_file():
            report["warnings"].append("validation artifact file is missing")
            return report

        pred_df = pd.read_csv(artifact_path)
        report["validation_artifact"].update(
            {
                "rows": int(len(pred_df)),
                "columns": list(pred_df.columns),
            }
        )
        key_columns = _list_of_strings(spec.get("key_columns"))
        prediction_columns = _list_of_strings(spec.get("prediction_columns"))
        missing_prediction_columns = [
            column for column in prediction_columns if column not in pred_df.columns
        ]
        if missing_prediction_columns:
            report["warnings"].append(
                "missing prediction columns: " + ", ".join(missing_prediction_columns)
            )

        score_split = str(
            spec.get("score_split", "./input/biomaker_splits/score_val.csv")
        )
        truth_path = workspace_dir / score_split.removeprefix("./")
        truth_df = pd.read_csv(truth_path) if truth_path.is_file() else pd.DataFrame()
        report["target_artifact"] = {
            "path": str(truth_path),
            "exists": truth_path.is_file(),
            "rows": int(len(truth_df)) if truth_path.is_file() else None,
            "columns": list(truth_df.columns) if truth_path.is_file() else [],
        }
        aligned_truth, alignment = (
            _align_truth_rows(pred_df, truth_df, key_columns)
            if truth_path.is_file()
            else (None, {"error": f"{score_split} missing"})
        )
        report["alignment"] = alignment

        column_summaries = {}
        for column in prediction_columns:
            if column in pred_df.columns:
                column_summaries[column] = _summarize_series(pred_df[column])
        report["prediction_column_summaries"] = column_summaries

        target_columns = [
            column
            for column in truth_df.columns
            if column not in set(key_columns) and column not in set(pred_df.columns)
        ]
        target_path_columns = [
            column
            for column in target_columns
            if "path" in column.lower() or "file" in column.lower()
        ]
        report["target_columns_not_in_prediction"] = target_columns[:12]

        sample_reports = []
        n_samples = min(max_samples, len(pred_df))
        for idx in range(n_samples):
            pred_row = pred_df.iloc[idx]
            sample: dict[str, Any] = {
                "row_index": idx,
                "keys": {
                    column: _safe_scalar(pred_row[column])
                    for column in key_columns
                    if column in pred_df.columns
                },
                "predictions": {},
            }
            for column in prediction_columns:
                if column not in pred_df.columns:
                    continue
                value = pred_row[column]
                if "path" in column.lower() or "file" in column.lower():
                    sample["predictions"][column] = _summarize_array_file(
                        _resolve_candidate_path(
                            workspace_dir, value, base_dir=artifact_path.parent
                        )
                    )
                else:
                    sample["predictions"][column] = {"value": _safe_scalar(value)}

            if aligned_truth is not None and idx < len(aligned_truth):
                truth_row = aligned_truth.iloc[idx]
                target_summary: dict[str, Any] = {}
                for column in target_path_columns[:3]:
                    value = truth_row[column]
                    target_summary[column] = _summarize_array_file(
                        _resolve_candidate_path(
                            workspace_dir, value, base_dir=truth_path.parent
                        )
                    )
                for column in target_columns:
                    if column in target_summary or column in target_path_columns:
                        continue
                    target_summary[column] = {"value": _safe_scalar(truth_row[column])}
                    if len(target_summary) >= 6:
                        break
                sample["targets"] = target_summary
            sample_reports.append(sample)
        report["sample_reports"] = sample_reports
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return report


def _json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _split_manifest(split_dir: Path) -> dict[str, Any]:
    manifest = _json_dict(split_dir / "split_manifest.json")
    if manifest:
        return manifest
    split_plan = _json_dict(split_dir / "split_plan.json")
    embedded = split_plan.get("split_manifest")
    return embedded if isinstance(embedded, dict) else {}


def _manifest_key_columns(manifest: dict[str, Any]) -> list[str]:
    columns = manifest.get("key_columns")
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns if str(column).strip()]


def _sort_label_values(values: list[Any]) -> list[Any]:
    cleaned = [
        value.item() if isinstance(value, np.generic) else value for value in values
    ]
    try:
        numeric = [float(value) for value in cleaned]
        order = sorted(range(len(cleaned)), key=lambda index: numeric[index])
        return [cleaned[index] for index in order]
    except Exception:
        return sorted(cleaned, key=lambda value: str(value))


def _prediction_column_name(value: Any, used: set[str]) -> str:
    raw = str(value).strip()
    match = re.fullmatch(r"(?:class|label|target)[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if match:
        text = match.group(1)
    else:
        text = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    text = text or "value"
    name = f"prob_{text}"
    if name not in used:
        used.add(name)
        return name
    index = 2
    while f"{name}_{index}" in used:
        index += 1
    unique = f"{name}_{index}"
    used.add(unique)
    return unique


def _truth_answer_columns(
    workspace_dir: Path,
    split_plan: dict[str, Any],
    hidden_score_path: Path,
    score_df: pd.DataFrame,
) -> list[str]:
    try:
        columns = load_experiment_contract(workspace_dir).hidden_answer_columns(
            "score_val"
        )
        if columns:
            return columns
    except Exception:
        pass
    hidden_answers = split_plan.get("hidden_answers") or split_plan.get("private_gt")
    if isinstance(hidden_answers, dict):
        score_meta = hidden_answers.get("score_val")
        if isinstance(score_meta, dict):
            columns = [str(column) for column in score_meta.get("answer_columns") or []]
            if columns:
                return columns
    columns_plan = split_plan.get("columns")
    if isinstance(columns_plan, dict):
        for key in ["label", "target", "mask", "answer"]:
            value = columns_plan.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
    if not hidden_score_path.is_file():
        return []
    hidden_df = pd.read_csv(hidden_score_path, nrows=1)
    legacy_label_columns = []
    for column in map(str, hidden_df.columns):
        lower = column.lower()
        if lower in {"class", "label", "target", "y"} or re.fullmatch(
            r"(?:class|label|target)[_-]?\d+", lower
        ):
            legacy_label_columns.append(column)
    if legacy_label_columns:
        return legacy_label_columns
    public_columns = set(map(str, score_df.columns))
    return [
        str(column) for column in hidden_df.columns if str(column) not in public_columns
    ]


def _infer_key_columns(
    score_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    answer_columns: list[str],
) -> list[str]:
    excluded = set(answer_columns)
    common_columns = [
        str(column)
        for column in score_df.columns
        if str(column) in set(map(str, test_df.columns))
        and str(column) in set(map(str, hidden_df.columns))
        and str(column) not in excluded
    ]
    common = [
        column
        for column in common_columns
        if "path" not in column.lower() and "file" not in column.lower()
    ]
    preferred = [
        "Id",
        "id",
        "ID",
        "patient",
        "patient_id",
        "case_id",
        "image_id",
        "uid",
    ]
    for column in preferred + common:
        if column not in common:
            continue
        if (
            score_df[column].astype(str).is_unique
            and test_df[column].astype(str).is_unique
            and hidden_df[column].astype(str).is_unique
        ):
            return [column]
    path_like = [
        column
        for column in common_columns
        if "path" in column.lower() or "file" in column.lower()
    ]
    for column in path_like:
        if (
            score_df[column].astype(str).is_unique
            and test_df[column].astype(str).is_unique
            and hidden_df[column].astype(str).is_unique
        ):
            return [column]
    return []


def _infer_segmentation_key_columns(
    score_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    answer_column: str,
) -> list[str]:
    key_columns = _infer_key_columns(score_df, test_df, hidden_df, [answer_column])
    common = set(map(str, test_df.columns)) & set(map(str, hidden_df.columns))
    metadata_columns = []
    blocked_tokens = {"path", "file", "mask", "image"}
    for column in map(str, score_df.columns):
        lower = column.lower()
        if column in key_columns or column == answer_column or column not in common:
            continue
        if any(token in lower for token in blocked_tokens):
            continue
        metadata_columns.append(column)
    return key_columns + metadata_columns


def _looks_like_file_values(values: pd.Series) -> bool:
    suffixes = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".mhd",
        ".npy",
        ".npz",
        ".dcm",
    )
    for value in values.dropna().astype(str).head(50):
        lower = value.lower()
        if "/" in value or "\\" in value or lower.endswith(suffixes):
            return True
    return False


def _looks_like_pil_image_file_values(values: pd.Series) -> bool:
    suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    seen = False
    for value in values.dropna().astype(str).head(50):
        lower = value.lower()
        if "/" not in value and "\\" not in value and "." not in lower:
            continue
        seen = True
        if not lower.endswith(suffixes):
            return False
    return seen


def _looks_like_volume_file_values(values: pd.Series) -> bool:
    suffixes = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".npy", ".npz")
    seen = False
    for value in values.dropna().astype(str).head(50):
        lower = value.lower()
        if "/" not in value and "\\" not in value and "." not in lower:
            continue
        seen = True
        if not lower.endswith(suffixes):
            return False
    return seen


def _scalar_classification_kind(
    train_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    answer_columns: list[str],
    split_plan: dict[str, Any],
) -> str | None:
    if not answer_columns:
        return None
    declared_task_type = str(split_plan.get("task_type", "")).lower()
    primary_metric = str(split_plan.get("primary_metric", "")).lower()
    if any(
        token in f"{declared_task_type} {primary_metric}"
        for token in ["aji", "dice", "mask", "segmentation"]
    ):
        return None
    for column in answer_columns:
        if column not in train_df.columns or column not in hidden_df.columns:
            return None
        if (
            "path" in column.lower()
            or "file" in column.lower()
            or "mask" in column.lower()
        ):
            return None
        values = pd.concat([train_df[column], hidden_df[column]], ignore_index=True)
        if _looks_like_file_values(values):
            return None
    if len(answer_columns) == 1:
        if declared_task_type in {"binary", "multiclass", "ordinal_multiclass"}:
            return "binary" if declared_task_type == "binary" else "multiclass"
        return "multiclass"
    for column in answer_columns:
        values = pd.concat([train_df[column], hidden_df[column]], ignore_index=True)
        non_null = values.dropna()
        if non_null.empty:
            return None
        if not set(non_null.astype(str).unique()).issubset({"0", "1", "False", "True"}):
            return None
    return "multilabel"


def _score_split_path(workspace_dir: Path) -> Path:
    contract_path = load_experiment_contract(workspace_dir).split_path("score_val")
    if contract_path is not None:
        return contract_path
    split_dir = workspace_dir / "input" / "biomaker_splits"
    return split_dir / "score_val.csv"


def _load_fixed_split_context(
    workspace_dir: Path, hidden_score_path: Path
) -> dict[str, Any] | None:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    train_path = split_dir / "agent_train.csv"
    score_path = _score_split_path(workspace_dir)
    lifecycle = _bind_public_split_lifecycle({}, workspace_dir)
    test_path = (workspace_dir / lifecycle["test_split"].removeprefix("./")
                 if lifecycle["test_split"] is not None else None)
    if not (
        train_path.is_file()
        and score_path.is_file()
        and (test_path is None or test_path.is_file())
        and hidden_score_path.is_file()
    ):
        return None
    train_df = pd.read_csv(train_path)
    score_df = pd.read_csv(score_path)
    # Empty schema only assists existing column inference. It is never written
    # as a split, used as prediction rows, or supplied to a test evaluator.
    test_df = (pd.read_csv(test_path) if test_path is not None
               else score_df.iloc[:0].copy())
    hidden_df = pd.read_csv(hidden_score_path)
    split_plan = _json_dict(split_dir / "split_plan.json")
    split_manifest = _split_manifest(split_dir)
    return {
        "score_path": score_path,
        "test_path": test_path,
        "split_lifecycle": lifecycle,
        "train_df": train_df,
        "score_df": score_df,
        "test_df": test_df,
        "hidden_df": hidden_df,
        "split_plan": split_plan,
        "split_manifest": split_manifest,
        "answer_columns": _truth_answer_columns(
            workspace_dir, split_plan, hidden_score_path, score_df
        ),
    }


def _infer_fixed_classification_spec(
    workspace_dir: Path,
    hidden_score_path: Path,
) -> dict[str, Any] | None:
    context = _load_fixed_split_context(workspace_dir, hidden_score_path)
    if context is None:
        return None
    train_df = context["train_df"]
    score_df = context["score_df"]
    test_df = context["test_df"]
    hidden_df = context["hidden_df"]
    split_plan = context["split_plan"]
    split_manifest = context["split_manifest"]
    answer_columns = context["answer_columns"]
    manifest_heads = _manifest_head_types(split_manifest)
    if manifest_heads and not (
        manifest_heads
        & {
            "classification",
            "binary",
            "multiclass",
            "multilabel",
            "single_label_classification",
            "multi_label_classification",
        }
    ):
        return None
    task_type = _scalar_classification_kind(
        train_df, hidden_df, answer_columns, split_plan
    )
    if task_type is None:
        return None

    key_columns = _manifest_key_columns(split_manifest) or _infer_key_columns(
        score_df, test_df, hidden_df, answer_columns
    )
    if not key_columns and not bool(split_plan.get("row_order_contract")):
        return None
    primary_metric = str(
        _manifest_primary_metric(split_manifest)
        or split_plan.get("primary_metric")
        or ("mean_binary_accuracy" if task_type == "multilabel" else "accuracy")
    )
    spec: dict[str, Any] = {
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "score_split": "./" + str(context["score_path"].relative_to(workspace_dir)),
        "key_columns": key_columns,
        "primary_metric": primary_metric,
        "maximize": True,
        "task_type": task_type,
        "answer_columns": answer_columns,
    }
    if task_type == "multilabel":
        used: set[str] = set()
        spec["prediction_columns"] = [
            _prediction_column_name(column, used) for column in answer_columns
        ]
        return spec

    label_column = answer_columns[0]
    labels = _sort_label_values(
        pd.concat([train_df[label_column], hidden_df[label_column]], ignore_index=True)
        .dropna()
        .unique()
        .tolist()
    )
    if not (2 <= len(labels) <= 200):
        return None
    spec["class_labels"] = labels
    if len(labels) == 2:
        spec["task_type"] = "binary"
        spec["positive_class"] = labels[-1]
    spec["prediction_columns"] = [f"prob_{index}" for index in range(len(labels))]
    return spec


def _primary_metric_from_task_desc(task_desc: Any, task_type: str) -> str | None:
    primary, _ = requested_metrics(_task_payload(task_desc))
    if task_type == "multilabel":
        return {"auc": "mean_roc_auc", "accuracy": "mean_binary_accuracy"}.get(primary, primary)
    return primary


def _fixed_classification_metric_supported(metric: str) -> bool:
    return is_classification_metric_supported(metric)


def _write_fixed_evaluator(
    workspace_dir: Path,
    log_dir: Path,
    hidden_score_path: Path,
    timeout: int,
    spec: dict[str, Any],
    template: str,
    *,
    run_public_contract: bool,
    log_message: str,
) -> PreparedEvaluator | None:
    evaluator_path = workspace_dir / "evaluator.py"
    evaluator_code = template.replace("__BIOMAKER_SPEC__", repr(spec)).lstrip()
    evaluator_path.write_text(evaluator_code, encoding="utf-8")
    try:
        _run_self_test(
            workspace_dir, timeout=timeout, hidden_score_path=hidden_score_path
        )
        prediction_spec = _normalize_prediction_spec_json(
            _load_prediction_spec(workspace_dir)
        )
        if run_public_contract:
            _run_public_artifact_contract_test(
                workspace_dir,
                prediction_spec,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
    except Exception:
        try:
            evaluator_path.unlink()
        except FileNotFoundError:
            pass
        clean_candidate_outputs(workspace_dir)
        raise

    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_spec_path = workspace_dir / "evaluator_prediction_spec.json"
    prediction_spec_path.write_text(prediction_spec, encoding="utf-8")
    shutil.copy2(evaluator_path, log_dir / "evaluator.py")
    (log_dir / "evaluator_prediction_spec.json").write_text(
        prediction_spec, encoding="utf-8"
    )
    append_event(
        log_dir,
        "fixed_evaluator_prepared",
        task_type=spec.get("task_type"),
        metric_specs=spec.get('metrics', []),
        prediction_columns=spec.get("prediction_columns"),
    )
    clean_candidate_outputs(workspace_dir)
    logger.info("%s at %s", log_message, evaluator_path)
    return PreparedEvaluator(path=evaluator_path, prediction_spec=prediction_spec)


def _write_fixed_classification_evaluator(
    workspace_dir: Path,
    log_dir: Path,
    hidden_score_path: Path,
    timeout: int,
    task_desc: Any | None = None,
) -> PreparedEvaluator | None:
    spec = _infer_fixed_classification_spec(workspace_dir, hidden_score_path)
    if spec is None:
        return None
    if task_desc is not None:
        metric = _primary_metric_from_task_desc(
            task_desc, str(spec.get("task_type", ""))
        )
        if metric is not None:
            spec["primary_metric"] = metric
    if not _fixed_classification_metric_supported(str(spec.get("primary_metric", ""))):
        return None
    _, secondary = requested_metrics(_task_payload(task_desc))
    if not secondary:
        secondary = _split_manifest(workspace_dir / "input/biomaker_splits").get("metric_contract", {}).get("secondary_metrics", [])
        secondary = [metric_name(m) for m in secondary]
    if any(name not in {"accuracy", "auc"} for name in secondary):
        return None
    spec.update(
        evaluator_api_version=EVALUATOR_API_VERSION, secondary_metrics=secondary
    )
    spec = _resolve_metric_contract(spec, workspace_dir, task_desc, require_match=False)
    return _write_fixed_evaluator(
        workspace_dir=workspace_dir,
        log_dir=log_dir,
        hidden_score_path=hidden_score_path,
        timeout=timeout,
        spec=spec,
        template=compile_adapter(_CLASSIFICATION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec)), spec),
        run_public_contract=True,
        log_message="Fixed evaluator prepared",
    )


def _write_fixed_regression_evaluator(
    workspace_dir: Path, log_dir: Path, hidden_score_path: Path,
    timeout: int, task_desc: Any = None,
) -> PreparedEvaluator | None:
    context = _load_fixed_split_context(workspace_dir, hidden_score_path)
    if context is None:
        return None
    manifest = context["split_manifest"]
    primary, secondary = _requested_metric_names(task_desc, workspace_dir)
    names = [primary, *secondary]
    if not primary or any(not metric_info(name) or metric_info(name).task_family != "regression" for name in names):
        return None
    answers = context["answer_columns"]
    if not answers or any(column not in context["train_df"] or column not in context["hidden_df"] for column in answers):
        return None
    # Numeric storage is the only built-in assumption. JSON, array files and
    # other representations go through the same generated data-adapter API.
    for frame in (context["train_df"], context["hidden_df"]):
        for column in answers:
            values = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(values).all():
                return None
    keys = _manifest_key_columns(manifest) or _infer_key_columns(
        context["score_df"], context["test_df"], context["hidden_df"], answers,
    )
    if not keys and not context["split_plan"].get("row_order_contract", manifest.get("row_order_contract", False)):
        return None
    columns = manifest.get("prediction_contract", {}).get("prediction_columns")
    if not columns:
        used = set(keys)
        columns = [_prediction_column_name("pred_" + column, used) for column in answers]
    if len(columns) != len(answers):
        return None
    spec = dict(
        evaluator_api_version=EVALUATOR_API_VERSION,
        validation_artifact="./working/validation_predictions.csv",
        test_submission="./working/submission.csv",
        score_split="./" + str(context["score_path"].relative_to(workspace_dir)),
        key_columns=keys, prediction_columns=columns, answer_columns=answers,
        primary_metric=primary, secondary_metrics=secondary, maximize=False,
        task_type="regression", target_shape=[] if len(answers) == 1 else [len(answers)],
    )
    spec = _resolve_metric_contract(spec, workspace_dir, task_desc, require_match=False)
    return _write_fixed_evaluator(
        workspace_dir, log_dir, hidden_score_path, timeout, spec,
        compile_adapter(_REGRESSION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec)), spec),
        run_public_contract=True, log_message="Fixed regression evaluator prepared",
    )


def _infer_fixed_segmentation_spec(
    workspace_dir: Path,
    hidden_score_path: Path,
) -> dict[str, Any] | None:
    context = _load_fixed_split_context(workspace_dir, hidden_score_path)
    if context is None:
        return None
    score_df = context["score_df"]
    test_df = context["test_df"]
    hidden_df = context["hidden_df"]
    split_plan = context["split_plan"]
    split_manifest = context["split_manifest"]
    answer_columns = context["answer_columns"]
    if not answer_columns:
        return None
    manifest_heads = _manifest_head_types(split_manifest)
    if manifest_heads and not (
        manifest_heads
        & {
            "segmentation",
            "segmentation_2d",
            "segmentation_3d",
            "binary_segmentation",
            "instance_segmentation",
            "volume_segmentation",
            "medical_volume_segmentation",
        }
    ):
        return None

    task_text = " ".join(
        [
            str(split_plan.get("task_type", "")),
            str(split_plan.get("primary_metric", "")),
            " ".join(sorted(manifest_heads)),
            json.dumps(split_plan.get("columns", {}), ensure_ascii=False),
        ]
    ).lower()
    segmentation_hint = any(
        token in task_text
        for token in ["segmentation", "mask", "dice", "aji", "hd95", "hausdorff"]
    )
    mask_candidates = []
    volume_mask = False
    for column in answer_columns:
        if column not in hidden_df.columns:
            continue
        values = hidden_df[column]
        name = column.lower()
        is_mask_named = "mask" in name or "label_path" in name
        is_file_column = _looks_like_file_values(values)
        if is_mask_named or is_file_column:
            if _looks_like_pil_image_file_values(values):
                mask_candidates.append(column)
            elif _looks_like_volume_file_values(values):
                mask_candidates.append(column)
                volume_mask = True
            elif is_mask_named:
                return None
    if not segmentation_hint and not mask_candidates:
        return None
    if not mask_candidates:
        return None
    declared = split_manifest.get('metric_contract', {})
    auxiliary = declared.get('auxiliary_columns', {})
    answer_column = declared.get('target_column')
    if answer_column is not None:
        if answer_column not in mask_candidates:
            return None
    elif len(answer_columns) == 1:
        answer_column = mask_candidates[0]
    else:
        # Multiple private paths need an explicit target; never score a
        # skeleton or branch parsing merely because its column came first.
        return None

    key_columns = _manifest_key_columns(
        split_manifest
    ) or _infer_segmentation_key_columns(score_df, test_df, hidden_df, answer_column)
    if not key_columns and not bool(split_plan.get("row_order_contract")):
        return None

    image_column = None
    columns_plan = split_plan.get("columns")
    if isinstance(columns_plan, dict):
        raw_image_column = columns_plan.get("path") or columns_plan.get("image")
        if isinstance(raw_image_column, str):
            image_column = raw_image_column
    public_columns = set(map(str, score_df.columns)) | set(map(str, test_df.columns))
    public_frames = [score_df, test_df]
    if image_column not in public_columns or not any(
        image_column in frame.columns and _looks_like_file_values(frame[image_column])
        for frame in public_frames
    ):
        image_column = None
        candidates: list[tuple[int, int, str]] = []
        seen_columns: set[str] = set()
        for order, column in enumerate(list(score_df.columns) + list(test_df.columns)):
            column = str(column)
            if column in seen_columns:
                continue
            seen_columns.add(column)
            if column == answer_column or column in key_columns:
                continue
            name = column.lower()
            if not any(token in name for token in ("image", "path", "file")):
                continue
            if not any(
                column in frame.columns and _looks_like_file_values(frame[column])
                for frame in public_frames
            ):
                continue
            priority = 0 if ("path" in name or "file" in name) else 1
            candidates.append((priority, order, column))
        if candidates:
            candidates.sort()
            image_column = candidates[0][2]

    primary_metric = str(
        _manifest_primary_metric(split_manifest)
        or split_plan.get("primary_metric")
        or ("binary_dice" if volume_mask else "aji")
    )
    if not is_segmentation_metric_supported(primary_metric):
        return None
    task_type = (
        "segmentation_3d"
        if volume_mask or "segmentation_3d" in manifest_heads
        else "segmentation"
    )
    output_column = (
        answer_column if answer_column not in key_columns else "pred_mask_path"
    )
    return {
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "score_split": "./" + str(context["score_path"].relative_to(workspace_dir)),
        "key_columns": key_columns,
        "prediction_columns": [output_column],
        "file_reference_columns": [output_column],
        "primary_metric": primary_metric,
        "maximize": metric_maximize(primary_metric),
        "task_type": task_type,
        "prediction_type": "mask_file_path",
        "answer_columns": list(answer_columns),
        "target_column": answer_column,
        "auxiliary_columns": auxiliary,
        "image_column": image_column,
    }


def _write_fixed_segmentation_evaluator(
    workspace_dir: Path,
    log_dir: Path,
    hidden_score_path: Path,
    timeout: int,
    task_desc: Any | None = None,
) -> PreparedEvaluator | None:
    spec = _infer_fixed_segmentation_spec(workspace_dir, hidden_score_path)
    if spec is None:
        return None
    if task_desc is not None:
        metric = _primary_metric_from_task_desc(
            task_desc, str(spec.get("task_type", ""))
        )
        if metric is not None:
            spec["primary_metric"] = metric
    if not is_segmentation_metric_supported(str(spec.get("primary_metric", ""))):
        return None
    _, secondary = requested_metrics(_task_payload(task_desc))
    if not secondary:
        secondary = _split_manifest(workspace_dir / "input/biomaker_splits").get("metric_contract", {}).get("secondary_metrics", [])
        secondary = [metric_name(m) for m in secondary]
    if any(not is_segmentation_metric_supported(name) for name in secondary):
        return None
    spec.update(
        evaluator_api_version=EVALUATOR_API_VERSION, secondary_metrics=secondary
    )
    spec["maximize"] = metric_maximize(str(spec.get("primary_metric", "")))
    spec = _resolve_metric_contract(spec, workspace_dir, task_desc, require_match=False)
    return _write_fixed_evaluator(
        workspace_dir=workspace_dir,
        log_dir=log_dir,
        hidden_score_path=hidden_score_path,
        timeout=timeout,
        spec=spec,
        template=compile_adapter(_SEGMENTATION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec)), spec),
        run_public_contract=False,
        log_message="Fixed segmentation evaluator prepared",
    )


def _write_fixed_detection_evaluator(
    workspace_dir, log_dir, hidden_score_path, timeout, task_desc=None
):
    context = _load_fixed_split_context(workspace_dir, hidden_score_path)
    if context is None:
        return None
    spec = fixed_detection_spec(context, _task_payload(task_desc))
    if spec is None:
        return None
    spec["score_split"] = "./" + str(context["score_path"].relative_to(workspace_dir))
    spec = _resolve_metric_contract(spec, workspace_dir, task_desc, require_match=False)
    return _write_fixed_evaluator(
        workspace_dir,
        log_dir,
        hidden_score_path,
        timeout,
        spec,
        compile_adapter(_DETECTION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec)), spec),
        run_public_contract=True,
        log_message="Fixed detection evaluator prepared",
    )


def _write_fixed_restoration_evaluator(
    workspace_dir, log_dir, hidden_score_path, timeout, task_desc=None
):
    context = _load_fixed_split_context(workspace_dir, hidden_score_path)
    if context is None:
        return None
    spec = fixed_restoration_spec(context, _task_payload(task_desc))
    if spec is None:
        return None
    spec["score_split"] = "./" + str(context["score_path"].relative_to(workspace_dir))
    spec = _resolve_metric_contract(spec, workspace_dir, task_desc, require_match=False)
    return _write_fixed_evaluator(
        workspace_dir, log_dir, hidden_score_path, timeout, spec,
        compile_adapter(_RESTORATION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec)), spec),
        run_public_contract=False,
        log_message="Fixed restoration evaluator prepared",
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requested_metric_names(task_desc: Any, workspace_dir: Path) -> tuple[str | None, list[str]]:
    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    fallback = _json_dict(workspace_dir / "input/biomaker_splits/split_plan.json").get("primary_metric")
    try:
        return declared_metric_names(manifest, _task_payload(task_desc), fallback_primary=fallback)
    except ValueError as exc:
        raise EvaluatorSetupError(str(exc)) from exc


def _standard_adapter_required(task_desc: Any, workspace_dir: Path) -> bool:
    """A schema miss for a known metric must never generate scoring formulas."""
    primary, secondary = _requested_metric_names(task_desc, workspace_dir)
    names = [m for m in [primary, *secondary] if m]
    return bool(names) and all(metric_info(m) is not None for m in names)


def _validate_declared_capabilities(task_desc: Any, workspace_dir: Path) -> None:
    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    fallback = _json_dict(workspace_dir / "input/biomaker_splits/split_plan.json").get("primary_metric")
    try:
        validate_declared_capabilities(manifest, _task_payload(task_desc), fallback_primary=fallback)
    except (ValueError, TypeError) as exc:
        raise EvaluatorSetupError(str(exc)) from exc


def _validate_custom_metric_contract(spec: dict[str, Any], workspace_dir: Path, task_desc: Any) -> None:
    """Custom formulas cannot redefine the prepared protocol or outputs."""
    primary, secondary = _requested_metric_names(task_desc, workspace_dir)
    actual = _canonical_declared_settings(spec)
    actual_names = [item['name'] for item in metric_definitions(actual)]
    if primary and set([primary, *secondary]) != set(actual_names):
        raise EvaluatorSetupError('Custom evaluator changed prepared primary/secondary metrics or added objectives.')
    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    declared = {**(requested_metric_contract(_task_payload(task_desc)) or {}), **_canonical_declared_settings(manifest.get("metric_contract", {}), fallback_primary=primary)}
    expected_definitions = declared_metric_definitions(manifest, _task_payload(task_desc), fallback_primary=primary)
    if {item['name']: item for item in metric_definitions(actual)} != {item['name']: item for item in expected_definitions}:
        raise EvaluatorSetupError('Custom evaluator changed prepared metrics, maximize directions or metric parameters.')
    blocks = re.findall(r"```restoration-evaluation\s*\n(.*?)```", _task_payload(task_desc), re.S)
    if len(blocks) > 1:
        raise EvaluatorSetupError("Declare exactly one restoration-evaluation contract.")
    if blocks:
        declared["restoration_evaluation"] = {**declared.get("restoration_evaluation", {}), **json.loads(blocks[0])}
        declared = _canonical_declared_settings(declared, fallback_primary=primary)
    def preserves(value, expected):
        if isinstance(expected, dict):
            return isinstance(value, dict) and all(key in value and preserves(value[key], item) for key, item in expected.items())
        return value == expected
    for key, value in declared.items():
        if key in {"primary_metric", "secondary_metrics", "maximize", "metrics"} | _PROTOCOL_DOCUMENTATION_FIELDS:
            continue
        if key not in actual or not preserves(actual[key], value):
            raise EvaluatorSetupError("Custom evaluator changed or omitted prepared metric protocol field: " + key)
    keys = _manifest_key_columns(manifest)
    if keys and spec.get("key_columns") != keys:
        raise EvaluatorSetupError("Custom evaluator changed the prepared sample keys.")
    for key in ("prediction_columns", "file_reference_columns", "prediction_type"):
        value = manifest.get("prediction_contract", {}).get(key)
        if value is not None and spec.get(key) != value:
            raise EvaluatorSetupError("Custom evaluator changed or omitted prepared prediction field: " + key)


def _canonical_declared_settings(values: dict[str, Any], *, fallback_primary: str | None = None) -> dict[str, Any]:
    try:
        return canonical_metric_contract(values, fallback_primary=fallback_primary)
    except ValueError as exc:
        raise EvaluatorSetupError(str(exc)) from exc


def _bind_public_split_lifecycle(spec: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    """Bind test presence from explicit prepared split declarations, never absence.

    A missing test CSV remains an error for legacy/test-enabled contracts. Only
    a manifest explicitly declaring train and score_val without test enables the
    no-test lifecycle. Adapter-provided split claims cannot override it.
    """
    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    declarations = []
    for field in ("public_splits", "required_splits"):
        if field not in manifest:
            continue
        roles = manifest[field]
        if (not isinstance(roles, list) or any(not isinstance(role, str) for role in roles)
                or len(set(roles)) != len(roles)
                or not {"train", "score_val"}.issubset(roles)
                or set(roles) - {"train", "score_val", "test"}):
            raise EvaluatorSetupError(
                "split_manifest." + field + " must explicitly list train, score_val and optionally test."
            )
        declarations.append(set(roles))
    if declarations and any(roles != declarations[0] for roles in declarations[1:]):
        raise EvaluatorSetupError("Conflicting public_splits and required_splits declarations.")
    result = dict(spec)
    result.pop("public_splits", None)
    result["test_split"] = "./input/biomaker_splits/agent_test.csv"
    if declarations:
        public = declarations[0]
        result["public_splits"] = [role for role in ("train", "score_val", "test") if role in public]
        if "test" not in public:
            for field in ("splits", "hidden_answers", "private_gt"):
                metadata = manifest.get(field)
                if isinstance(metadata, dict) and "test" in metadata:
                    raise EvaluatorSetupError("No-test split declaration conflicts with split_manifest." + field + ".test.")
            result["test_split"] = None
    return result


def _freeze_data_adapter(workspace_dir: Path, source: str, task_desc: Any, hidden_score_path: Path) -> None:
    """Validate declarations against the prepared protocol, then embed constants."""
    evaluator_path = workspace_dir / "evaluator.py"
    evaluator_path.write_text(compile_adapter(source), encoding="utf-8")
    completed = _run_python(
        workspace_dir,
        "import evaluator, json\nprint('ADAPTER_SPEC_JSON=' + json.dumps(evaluator._h_spec))\n",
        timeout=30,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    values = [line.removeprefix("ADAPTER_SPEC_JSON=") for line in output.splitlines() if line.startswith("ADAPTER_SPEC_JSON=")]
    if completed.returncode != 0 or not values:
        raise EvaluatorSetupError("Adapter declaration failed.\n" + output[-4000:])
    spec = json.loads(_normalize_prediction_spec_json(values[-1]))
    _validate_scoring_contract(spec, task_desc)
    split_dir = workspace_dir / "input/biomaker_splits"
    manifest = _split_manifest(split_dir)
    plan = _json_dict(split_dir / "split_plan.json")
    keys = _manifest_key_columns(manifest)
    if keys and spec["key_columns"] != keys:
        raise EvaluatorSetupError("Adapter sample keys must preserve the prepared split manifest.")
    if not spec["key_columns"] and not plan.get("row_order_contract", manifest.get("row_order_contract", False)):
        raise EvaluatorSetupError("Empty key_columns requires an explicit row_order_contract.")
    score_path = _score_split_path(workspace_dir)
    public = pd.read_csv(score_path, dtype=str)
    answers = _truth_answer_columns(workspace_dir, plan, hidden_score_path, public)
    if set(spec["answer_columns"]) != set(answers):
        raise EvaluatorSetupError("Adapter answer_columns must preserve the declared hidden answer schema.")
    spec["score_split"] = "./" + str(score_path.relative_to(workspace_dir))
    spec = _bind_public_split_lifecycle(spec, workspace_dir)

    actual = _resolve_metric_contract(spec, workspace_dir, task_desc)
    evaluator_path.write_text(compile_adapter(source, actual), encoding="utf-8")


def _resolve_metric_contract(spec: dict[str, Any], workspace_dir: Path, task_desc: Any, *, require_match: bool = True) -> dict[str, Any]:
    """One capability-backed protocol gate for built-in and generated readers."""
    from .trusted_metrics import metric_capabilities, normalize_metric_spec

    manifest = _split_manifest(workspace_dir / "input/biomaker_splits")
    if not require_match:
        # Declarative readers bind to the preparation agent's column choices.
        # More complex encodings reach the code-adapter route, with the metric
        # contract and trusted runtime unchanged.
        spec = dict(spec)
        output = manifest.get("prediction_contract", {})
        for key in ("prediction_columns", "file_reference_columns", "prediction_type", "class_labels", "positive_class", "image_column"):
            if key in output:
                spec[key] = output[key]
    # Every newly prepared evaluator is v3. Historical v2 files are read only by
    # the rescoring path and are never rewritten here.
    spec = _canonical_declared_settings(spec)
    if not require_match:
        definitions = declared_metric_definitions(manifest, _task_payload(task_desc), fallback_primary=_primary_metric_from_raw_spec(spec))
        spec['metrics'] = definitions
        for key in ('primary_metric', 'secondary_metrics', 'maximize'):
            spec.pop(key, None)
    else:
        spec['metrics'] = metric_definitions(spec)
        for key in ('primary_metric', 'secondary_metrics', 'maximize'):
            spec.pop(key, None)
    spec['evaluator_api_version'] = EVALUATOR_API_VERSION
    spec = normalize_metric_spec(spec)
    spec = _bind_public_split_lifecycle(spec, workspace_dir)
    if not require_match:
        reader = data_adapter_capabilities()["families"][spec["task_family"]]
        if spec.get("prediction_type") not in reader["prediction_types"]:
            raise EvaluatorSetupError("Declared prediction encoding needs a data adapter; metric implementation is unchanged.")
        for columns, count_key in (("prediction_columns", "prediction_column_count"), ("answer_columns", "answer_column_count")):
            if count_key in reader and len(spec[columns]) != reader[count_key]:
                raise EvaluatorSetupError("Declared " + columns + " require a custom data adapter for this reader.")
        if reader["file_reference_policy"] == "all_prediction_columns":
            spec["file_reference_columns"] = list(spec["prediction_columns"])
        elif spec.get("file_reference_columns"):
            raise EvaluatorSetupError("File-valued predictions require a data adapter for this reader.")
    # Structured protocol declarations are authoritative. The agent may map data
    # representations, but cannot weaken the evaluation to pass preparation.
    declared_contract = {**(requested_metric_contract(_task_payload(task_desc)) or {}), **_canonical_declared_settings(manifest.get("metric_contract", {}), fallback_primary=spec['metrics'][0]['name'])}
    contract = {key: value for key, value in declared_contract.items()
                if key not in _PROTOCOL_DOCUMENTATION_FIELDS}
    if 'primary_metric' in contract or 'metrics' in contract:
        contract['metrics'] = declared_metric_definitions(manifest, _task_payload(task_desc), fallback_primary=spec['metrics'][0]['name'])
        for key in ('primary_metric', 'secondary_metrics', 'maximize'):
            contract.pop(key, None)
    supported_fields = set(metric_capabilities()["families"][spec["task_family"]]["allowed_contract_fields"])
    unknown_fields = set(contract) - supported_fields
    if unknown_fields:
        raise EvaluatorSetupError("Unsupported declared metric protocol fields: " + ", ".join(sorted(unknown_fields)))
    restoration = contract.pop("restoration_evaluation", {})
    if spec.get("task_family") == "image_restoration":
        restoration_fields = {"input_column", "target_column", "data_range", "crop_size", "crop_mode", "group_column", "groups", "aggregation"}
        restoration = {**{k: contract.pop(k) for k in list(contract) if k in restoration_fields}, **restoration}
        blocks = re.findall(r"```restoration-evaluation\s*\n(.*?)```", _task_payload(task_desc), re.S)
        if len(blocks) > 1:
            raise EvaluatorSetupError("Declare exactly one restoration-evaluation contract.")
        if blocks:
            restoration.update(json.loads(blocks[0]))
    expected = json.loads(json.dumps(spec))
    for key, value in contract.items():
        if key not in expected and require_match:
            raise EvaluatorSetupError(
                "Adapter omitted declared metric protocol field: " + key
                + ". Declare supported semantics explicitly; unsupported semantics require a Harness implementation."
            )
        expected[key] = value
    if restoration:
        expected["restoration_evaluation"] = {**expected.get("restoration_evaluation", {}), **restoration}
        if "crop_size" in restoration and "crop_mode" not in restoration:
            expected["restoration_evaluation"].pop("crop_mode", None)
    try:
        expected = normalize_metric_spec(expected)
    except ValueError as exc:
        if require_match:
            raise EvaluatorSetupError("Adapter changed the frozen metric protocol or declared unsupported settings: " + str(exc)) from exc
        raise
    actual = normalize_metric_spec(spec)
    if not require_match:
        if expected['task_family'] == 'segmentation':
            auxiliary = expected.get('auxiliary_columns', {})
            target = expected.get('target_column', expected['answer_columns'][0])
            if (target in auxiliary.values()
                    or set(expected['answer_columns']) != {target, *auxiliary.values()}):
                raise EvaluatorSetupError('Mask reader requires exactly the target column and declared auxiliary columns as answers.')
        _validate_scoring_contract(expected, task_desc)
        return expected
    for key in set(contract) | ({"restoration_evaluation", "ssim_parameters"} if restoration else set()):
        if key == 'metrics':
            if {item['name']: item for item in actual[key]} != {item['name']: item for item in expected[key]}:
                raise EvaluatorSetupError('Adapter changed the frozen metrics or maximize directions.')
            actual[key] = expected[key]
            continue
        if key in actual and actual[key] != expected[key]:
            raise EvaluatorSetupError("Adapter changed the frozen metric protocol field: " + key)
    return actual


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _evaluator_input_fingerprints(
    workspace_dir: Path, hidden_score_path: Path | None
) -> dict[str, str | None]:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    fingerprints: dict[str, str | None] = {}
    for name in [
        "split_plan.json",
        "split_manifest.json",
        "agent_train.csv",
        "score_val.csv",
        "agent_val.csv",
        "agent_test.csv",
        "dataset_report.json",
    ]:
        fingerprints[f"input/biomaker_splits/{name}"] = _sha256_file(split_dir / name)
    if hidden_score_path is not None:
        fingerprints["hidden_score_truth"] = _sha256_file(hidden_score_path)
    return fingerprints


def _evaluator_cache_metadata(
    prompt: dict[str, Any],
    model: str,
    workspace_dir: Path,
    hidden_score_path: Path | None,
) -> dict[str, Any]:
    utils_dir = Path(__file__).resolve().parent
    code_fingerprints = {
        "evaluator_setup.py": _sha256_file(Path(__file__).resolve()),
        "evaluator_templates.py": _sha256_file(utils_dir / "evaluator_templates.py"),
        "detection_evaluator_template.py": _sha256_file(
            utils_dir / "detection_evaluator_template.py"
        ),
        "detection_contract.py": _sha256_file(utils_dir / "detection_contract.py"),
        "restoration_contract.py": _sha256_file(utils_dir / "restoration_contract.py"),
        "restoration_evaluator_template.py": _sha256_file(utils_dir / "restoration_evaluator_template.py"),
        "evaluator_protocol.py": _sha256_file(utils_dir / "evaluator_protocol.py"),
        "evaluator_adapter.py": _sha256_file(utils_dir / "evaluator_adapter.py"),
        **{name: _sha256_file(utils_dir / name) for name in METRIC_SOURCE_FILES},
        "metric_source.py": _sha256_file(utils_dir / "metric_source.py"),
        "metric_declarations.py": _sha256_file(utils_dir / "metric_declarations.py"),
        "standard_data_adapters.py": _sha256_file(utils_dir / "standard_data_adapters.py"),
        "exp_guide.py": _sha256_file(utils_dir / "exp_guide.py"),
        "data_summary.py": _sha256_file(utils_dir / "data_summary.py"),
    }
    runtime_versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "Pillow": _package_version("Pillow"),
        "scipy": _package_version("scipy"),
        "scikit-image": _package_version("scikit-image"),
        "SimpleITK": _package_version("SimpleITK"),
    }
    prompt_hash = hashlib.sha256(
        json.dumps({key: value for key, value in prompt.items() if key != "Preparation diagnostics"}, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "cache_version": _EVALUATOR_CACHE_VERSION,
        "model": model,
        "prompt_hash": prompt_hash,
        "code_fingerprints": code_fingerprints,
        "input_fingerprints": _evaluator_input_fingerprints(
            workspace_dir, hidden_score_path
        ),
        "runtime_versions": runtime_versions,
    }


def _stable_evaluator_cache_key(
    prompt: dict[str, Any],
    model: str,
    workspace_dir: Path,
    hidden_score_path: Path | None,
) -> str:
    payload = _evaluator_cache_metadata(
        prompt,
        model,
        workspace_dir=workspace_dir,
        hidden_score_path=hidden_score_path,
    )
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _try_reuse_cached_evaluator(
    cache_entry: Path,
    workspace_dir: Path,
    log_dir: Path,
    timeout: int,
    hidden_score_path: Path,
) -> PreparedEvaluator | None:
    cached_evaluator = cache_entry / "evaluator.py"
    cached_spec = cache_entry / "evaluator_prediction_spec.json"
    if not cached_evaluator.is_file() or not cached_spec.is_file():
        return None
    manifest = _json_dict(cache_entry / "manifest.json")
    if manifest.get("cache_version") != _EVALUATOR_CACHE_VERSION:
        append_event(
            log_dir,
            "evaluator_cache_reuse_failed",
            cache_entry=str(cache_entry),
            reason="cache manifest missing or version mismatch",
        )
        return None
    if (manifest.get("evaluator_sha256") != _sha256_file(cached_evaluator)
            or manifest.get("prediction_spec_sha256") != _sha256_file(cached_spec)):
        append_event(log_dir, "evaluator_cache_reuse_failed", cache_entry=str(cache_entry), reason="frozen code/spec integrity mismatch")
        return None

    evaluator_path = workspace_dir / "evaluator.py"
    prediction_spec_path = workspace_dir / "evaluator_prediction_spec.json"
    try:
        shutil.copy2(cached_evaluator, evaluator_path)
        prediction_spec = cached_spec.read_text(encoding="utf-8")
        prediction_spec_path.write_text(prediction_spec, encoding="utf-8")
        _run_self_test(
            workspace_dir,
            timeout=timeout,
            hidden_score_path=hidden_score_path,
        )
        if json.loads(_normalize_prediction_spec_json(_load_prediction_spec(workspace_dir))) != json.loads(prediction_spec):
            raise EvaluatorSetupError("Cached evaluator and public prediction spec disagree.")
        _run_public_artifact_contract_test(
            workspace_dir,
            prediction_spec,
            timeout=timeout,
            hidden_score_path=hidden_score_path,
        )
    except Exception as exc:
        reason = _sanitize_private_paths(str(exc), hidden_score_path)
        append_event(
            log_dir,
            "evaluator_cache_reuse_failed",
            cache_entry=str(cache_entry),
            reason=truncate_text(reason, max_chars=1200),
        )
        clean_candidate_outputs(workspace_dir)
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evaluator_path, log_dir / "evaluator.py")
    (log_dir / "evaluator_prediction_spec.json").write_text(
        prediction_spec, encoding="utf-8"
    )
    clean_candidate_outputs(workspace_dir)
    append_event(log_dir, "evaluator_cache_reused", cache_entry=str(cache_entry))
    logger.info("Reused cached evaluator from %s", cache_entry)
    return PreparedEvaluator(path=evaluator_path, prediction_spec=prediction_spec)


def _store_cached_evaluator(
    cache_entry: Path,
    evaluator_path: Path,
    prediction_spec: str,
    manifest: dict[str, Any] | None = None,
) -> None:
    if (cache_entry / "evaluator.py").is_file() and (
        cache_entry / "evaluator_prediction_spec.json"
    ).is_file():
        return
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_entry.parent / f".tmp-{cache_entry.name}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evaluator_path, tmp_dir / "evaluator.py")
    (tmp_dir / "evaluator_prediction_spec.json").write_text(
        prediction_spec, encoding="utf-8"
    )
    cache_manifest = dict(manifest or {})
    cache_manifest.update(
        {
            "cache_version": _EVALUATOR_CACHE_VERSION,
            "evaluator_sha256": _sha256_file(evaluator_path),
            "prediction_spec_sha256": hashlib.sha256(
                prediction_spec.encode("utf-8")
            ).hexdigest(),
            "tests_passed": [
                "self_test",
                "public_artifact_contract",
            ],
        }
    )
    (tmp_dir / "manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        tmp_dir.rename(cache_entry)
    except FileExistsError:
        shutil.rmtree(tmp_dir)


def prepare_evaluator(
    workspace_dir: Path | str,
    log_dir: Path | str,
    task_desc: Any,
    model: str,
    temperature: float = 0.0,
    attempts: int = 5,
    timeout: int = 120,
    cache_root: Path | str | None = None,
    hidden_score_path: Path | str | None = None,
) -> PreparedEvaluator:
    """Generate, smoke-test, and freeze evaluator.py in the workspace."""
    workspace_dir = Path(workspace_dir)
    log_dir = Path(log_dir)
    evaluator_path = workspace_dir / "evaluator.py"
    hidden_score_path = (
        Path(hidden_score_path)
        if hidden_score_path is not None
        else log_dir / "private_gt" / "score_val.csv"
    )
    _validate_declared_capabilities(task_desc, workspace_dir)
    preparation_diagnostics = []
    try:
        fixed = _write_fixed_regression_evaluator(
            workspace_dir, log_dir, hidden_score_path, timeout, task_desc,
        )
        if fixed is not None:
            return fixed
    except Exception as exc:
        reason = _sanitize_private_paths(str(exc), hidden_score_path)
        preparation_diagnostics.append({"reader": "regression", "reason": reason})
        append_event(log_dir, "fixed_regression_evaluator_failed", reason=truncate_text(reason, max_chars=1200))
    try:
        fixed = _write_fixed_classification_evaluator(
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            hidden_score_path=hidden_score_path,
            timeout=timeout,
            task_desc=task_desc,
        )
        if fixed is not None:
            return fixed
    except Exception as exc:
        reason = _sanitize_private_paths(str(exc), hidden_score_path)
        preparation_diagnostics.append({"reader": "classification", "reason": reason})
        append_event(
            log_dir,
            "fixed_evaluator_failed",
            reason=truncate_text(reason, max_chars=1200),
        )
    try:
        fixed = _write_fixed_segmentation_evaluator(
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            hidden_score_path=hidden_score_path,
            timeout=timeout,
            task_desc=task_desc,
        )
        if fixed is not None:
            return fixed
    except Exception as exc:
        reason = _sanitize_private_paths(str(exc), hidden_score_path)
        preparation_diagnostics.append({"reader": "segmentation", "reason": reason})
        append_event(
            log_dir,
            "fixed_segmentation_evaluator_failed",
            reason=truncate_text(reason, max_chars=1200),
        )

    try:
        fixed = _write_fixed_detection_evaluator(
            workspace_dir, log_dir, hidden_score_path, timeout, task_desc
        )
        if fixed is not None:
            return fixed
    except Exception as exc:
        preparation_diagnostics.append({"reader": "detection", "reason": _sanitize_private_paths(str(exc), hidden_score_path)})
        append_event(
            log_dir,
            "fixed_detection_evaluator_failed",
            reason=truncate_text(
                _sanitize_private_paths(str(exc), hidden_score_path), max_chars=1200
            ),
        )

    fixed_error = None
    try:
        fixed = _write_fixed_restoration_evaluator(
            workspace_dir, log_dir, hidden_score_path, timeout, task_desc
        )
        if fixed is not None:
            return fixed
    except Exception as exc:
        fixed_error = _sanitize_private_paths(str(exc), hidden_score_path)
        preparation_diagnostics.append({"reader": "image_restoration", "reason": fixed_error})
        append_event(log_dir, "fixed_restoration_evaluator_failed", reason=truncate_text(fixed_error, max_chars=1200))

    adapter_mode = _standard_adapter_required(task_desc, workspace_dir)
    prompt = (_adapter_prompt if adapter_mode else _evaluator_prompt)(task_desc, workspace_dir, hidden_score_path)
    if preparation_diagnostics:
        prompt["Preparation diagnostics"] = preparation_diagnostics
    cache_entry = None
    cache_metadata = None
    if cache_root is not None:
        cache_metadata = _evaluator_cache_metadata(
            prompt,
            model=model,
            workspace_dir=workspace_dir,
            hidden_score_path=hidden_score_path,
        )
        cache_key = _stable_evaluator_cache_key(
            prompt,
            model=model,
            workspace_dir=workspace_dir,
            hidden_score_path=hidden_score_path,
        )
        cache_entry = Path(cache_root) / cache_key
        append_event(
            log_dir,
            "evaluator_cache_lookup",
            cache_key=cache_key,
            cache_entry=str(cache_entry),
            exists=cache_entry.is_dir(),
        )
        reused = _try_reuse_cached_evaluator(
            cache_entry,
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            timeout=timeout,
            hidden_score_path=hidden_score_path,
        )
        if reused is not None:
            return reused
        if not cache_entry.is_dir():
            append_event(
                log_dir,
                "evaluator_cache_miss",
                cache_key=cache_key,
                cache_entry=str(cache_entry),
            )
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        append_event(log_dir, "evaluator_attempt_started", attempt=attempt)
        completion = backend.query(
            system_message=prompt
            | {
                "Previous evaluator errors": "\n\n".join(errors[-2:]),
                "Attempt": str(attempt),
            },
            user_message=None,
            model=model,
            temperature=temperature,
        )
        code = extract_code(str(completion))
        if not code:
            errors.append("No Python code block was returned.")
            append_event(
                log_dir,
                "evaluator_attempt_failed",
                attempt=attempt,
                reason=errors[-1],
            )
            continue

        preflight = validate_code_environment(code, include_medical=True)
        if not preflight.ok:
            errors.append(preflight.message)
            append_event(
                log_dir,
                "evaluator_attempt_failed",
                attempt=attempt,
                reason=preflight.message,
            )
            continue
        if _uses_eval_or_exec(code):
            errors.append(
                "Evaluator uses eval() or exec(). Use json.loads or ast.literal_eval instead."
            )
            append_event(
                log_dir,
                "evaluator_attempt_failed",
                attempt=attempt,
                reason=errors[-1],
            )
            continue

        clean_candidate_outputs(workspace_dir)
        try:
            if adapter_mode:
                _freeze_data_adapter(workspace_dir, code, task_desc, hidden_score_path)
            else:
                # Custom metrics can call the exact same trusted kernels without
                # depending on an installed/mutable biomaker package.
                tree = ast.parse(code)
                tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
                kernels = frozen_metric_source()
                evaluator_path.write_text(
                    "from __future__ import annotations\n" + kernels
                    + "\ntrusted_score_samples = score_samples\n"
                    + ast.unparse(tree) + "\n", encoding="utf-8"
                )
            _run_self_test(
                workspace_dir,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
            prediction_spec = _normalize_prediction_spec_json(
                _load_prediction_spec(workspace_dir)
            )
            _validate_scoring_contract(json.loads(prediction_spec), task_desc)
            if not adapter_mode:
                _validate_custom_metric_contract(json.loads(prediction_spec), workspace_dir, task_desc)
            _run_public_artifact_contract_test(
                workspace_dir,
                prediction_spec,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
        except Exception as exc:
            reason = _sanitize_private_paths(str(exc), hidden_score_path)
            errors.append(reason)
            append_event(
                log_dir,
                "evaluator_attempt_failed",
                attempt=attempt,
                reason=truncate_text(reason, max_chars=1200),
            )
            continue

        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evaluator_path, log_dir / "evaluator.py")
        (workspace_dir / "evaluator_prediction_spec.json").write_text(
            prediction_spec, encoding="utf-8"
        )
        (log_dir / "evaluator_prediction_spec.json").write_text(
            prediction_spec, encoding="utf-8"
        )
        if cache_entry is not None:
            _store_cached_evaluator(
                cache_entry,
                evaluator_path,
                prediction_spec,
                manifest=cache_metadata,
            )
        clean_candidate_outputs(workspace_dir)
        logger.info("Evaluator prepared at %s", evaluator_path)
        append_event(log_dir, "evaluator_attempt_succeeded", attempt=attempt)
        return PreparedEvaluator(path=evaluator_path, prediction_spec=prediction_spec)

    try:
        evaluator_path.unlink()
    except FileNotFoundError:
        pass
    try:
        (workspace_dir / "evaluator_prediction_spec.json").unlink()
    except FileNotFoundError:
        pass
    clean_candidate_outputs(workspace_dir)

    raise EvaluatorSetupError(
        "Failed to prepare evaluator after "
        f"{attempts} attempts:\n" + "\n\n".join(errors[-3:])
    )


def score_candidate_with_evaluator(
    workspace_dir: Path | str,
    timeout: int = 120,
    hidden_score_path: Path | str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> EvaluatorRunResult:
    """Run the frozen evaluator against the current ./working artifacts."""
    workspace_dir = Path(workspace_dir)
    contract = load_experiment_contract(workspace_dir)
    expected_path = contract.split_path("score_val")
    if expected_path is None:
        return EvaluatorRunResult(
            False,
            None,
            True,
            {},
            "",
            "Missing score-validation split in experiment contract.",
        )
    truth_path = _hidden_score_env(workspace_dir, hidden_score_path)[_SCORE_ENV_VAR]
    return score_artifacts_with_evaluator(
        workspace_dir,
        contract.prediction.validation_artifact,
        Path(truth_path),
        expected_path,
        timeout=timeout,
        env_overrides=env_overrides,
        include_artifact_report=True,
    )


def score_artifacts_with_evaluator(
    workspace_dir: Path | str,
    prediction_path: Path | str,
    truth_path: Path | str,
    expected_path: Path | str,
    *,
    timeout: int = 120,
    env_overrides: dict[str, str] | None = None,
    include_artifact_report: bool = False,
) -> EvaluatorRunResult:
    """Run one frozen scoring implementation on any explicitly supplied split."""
    workspace_dir = Path(workspace_dir)
    hidden_score_path = Path(truth_path).resolve()
    spec = load_experiment_contract(workspace_dir).prediction.raw
    if spec.get("evaluator_api_version") not in (2, EVALUATOR_API_VERSION):
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output="",
            error="This frozen evaluator predates split-independent scoring API v2. "
            "Start a new run to prepare a compatible evaluator; existing run artifacts were not changed.",
        )
    try:
        _validate_scoring_contract(spec, allow_legacy=True)
        sample_count = validate_scoring_rows(
            Path(prediction_path), hidden_score_path, Path(expected_path), spec
        )
    except Exception as exc:
        return EvaluatorRunResult(
            False,
            None,
            bool(spec.get("maximize", True)),
            {},
            "",
            _sanitize_private_paths(str(exc), hidden_score_path),
        )
    code = (
        "import json, evaluator\n"
        "if not callable(getattr(evaluator, 'score', None)):\n"
        "    raise RuntimeError('Frozen evaluator must implement score(prediction_path, truth_path, expected_path)')\n"
        "validation = None\n"
        "if hasattr(evaluator, 'validate_submission'):\n"
        "    validation = evaluator.validate_submission()\n"
        "    if validation is False or (isinstance(validation, dict) and validation.get('valid') is False):\n"
        "        raise ValueError('Submission validation rejected the predictions')\n"
        f"score = evaluator.score({str(Path(prediction_path).resolve())!r}, "
        f"{str(hidden_score_path)!r}, {str(Path(expected_path).resolve())!r})\n"
        "payload = {'score': score, 'submission_validation': validation}\n"
        f"print('{_RESULT_PREFIX}' + json.dumps(payload, ensure_ascii=False, default=str))\n"
    )
    try:
        # score() receives truth explicitly. Do not expose private test answers
        # through the legacy validation environment variable.
        evaluator_env = {_SCORE_ENV_VAR: ""}
        if env_overrides:
            evaluator_env.update(env_overrides)
        completed = _run_python(
            workspace_dir,
            code,
            timeout=timeout,
            env=evaluator_env,
        )
    except subprocess.TimeoutExpired as exc:
        output = _sanitize_private_paths(
            (exc.stdout or "") + (exc.stderr or ""),
            hidden_score_path,
        )
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output=output,
            error=f"Evaluator timed out after {timeout} seconds.",
        )

    output = _sanitize_private_paths(
        (completed.stdout or "") + (completed.stderr or ""),
        hidden_score_path,
    )
    if completed.returncode != 0:
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output=output,
            error=f"Evaluator exited with return code {completed.returncode}.",
        )

    for line in output.splitlines():
        if not line.startswith(_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(_RESULT_PREFIX) :])
            if include_artifact_report:
                payload["artifact_report"] = inspect_candidate_artifacts(workspace_dir)
            output_with_report = (
                output
                + "\n"
                + _ARTIFACT_REPORT_PREFIX
                + json.dumps(
                    payload.get("artifact_report", {}), ensure_ascii=False, default=str
                )
            )
            score = validate_score(payload.get("score"), spec)
            score["n"] = sample_count
            payload["score"] = score
            return EvaluatorRunResult(
                ok=True,
                metric=score.get('metric', score['metrics'][score['metric_specs'][0]['name']] if len(score['metric_specs']) == 1 else None),
                maximize=score.get('maximize', score['metric_specs'][0]['maximize'] if len(score['metric_specs']) == 1 else True),
                details=payload,
                output=output_with_report,
            )
        except Exception as exc:
            return EvaluatorRunResult(
                ok=False,
                metric=None,
                maximize=True,
                details={},
                output=output,
                error=f"Could not parse evaluator result: {exc}",
            )

    return EvaluatorRunResult(
        ok=False,
        metric=None,
        maximize=True,
        details={},
        output=output,
        error="Evaluator did not emit result JSON.",
    )


def validate_submission_with_evaluator(
    workspace_dir: Path | str,
    timeout: int = 120,
    env_overrides: dict[str, str] | None = None,
) -> EvaluatorRunResult:
    """Run only the frozen evaluator's public test-submission validation."""
    workspace_dir = Path(workspace_dir)
    code = (
        "import json, evaluator\n"
        "if not hasattr(evaluator, 'validate_submission'):\n"
        "    raise RuntimeError('evaluator.validate_submission() is missing')\n"
        "validation = evaluator.validate_submission()\n"
        "payload = {'submission_validation': validation}\n"
        f"print('{_RESULT_PREFIX}' + json.dumps(payload, ensure_ascii=False, default=str))\n"
    )
    try:
        completed = _run_python(
            workspace_dir,
            code,
            timeout=timeout,
            env=env_overrides,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output=output,
            error=f"Evaluator submission validation timed out after {timeout} seconds.",
        )

    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output=output,
            error=(
                "Evaluator submission validation exited with return code "
                f"{completed.returncode}."
            ),
        )
    for line in output.splitlines():
        if not line.startswith(_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(_RESULT_PREFIX) :])
        except Exception as exc:
            return EvaluatorRunResult(
                ok=False,
                metric=None,
                maximize=True,
                details={},
                output=output,
                error=f"Could not parse evaluator validation result: {exc}",
            )
        return EvaluatorRunResult(
            ok=True,
            metric=None,
            maximize=True,
            details=payload,
            output=output,
        )
    return EvaluatorRunResult(
        ok=False,
        metric=None,
        maximize=True,
        details={},
        output=output,
        error="Evaluator did not emit submission validation JSON.",
    )
