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
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .. import backend
from .environment import environment_manifest_text, validate_code_environment
from .score_validity import has_invalid_result
from .evaluator_templates import (
    _FIXED_CLASSIFICATION_EVALUATOR,
    _FIXED_SEGMENTATION_EVALUATOR,
    _STRICT_ARTIFACT_CONTRACT_CHECK,
)
from .events import append_event
from .exp_guide import (
    build_exp_guide,
    evaluator_exp_guide_view,
    load_exp_guide,
)
from .experiment_contract import load_experiment_contract
from .metric_registry import (
    METRIC_CONTRACT_REQUIREMENTS,
    canonical_metric_name,
    is_classification_metric_supported,
    is_segmentation_metric_supported,
    metric_maximize,
    require_supported_metric,
)
from .process_tree import (
    ProcessTreeCleanupError,
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
_EVALUATOR_CACHE_VERSION = 4


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

    def log_line(self) -> str:
        details = dict(self.details)
        details.pop("artifact_report", None)
        payload = {
            "metric": self.metric,
            "maximize": self.maximize,
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
            if not isinstance(data, dict) or has_invalid_result(data):
                return None
            metric = data.get("metric")
            maximize = data.get("maximize", True)
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                return None
            metric = float(metric)
            return (metric, maximize) if math.isfinite(metric) else None
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


def _evaluator_prompt(
    task_desc: Any, workspace_dir: Path, hidden_score_path: Path | None = None
) -> dict[str, Any]:
    return {
        "Introduction": (
            "You are writing the frozen evaluator for a machine learning task. "
            "This evaluator is created once before solution search. Candidate solutions "
            "will train models and write prediction artifacts; BioMaker will call this evaluator "
            "after each candidate finishes, so all candidates are scored with the same code."
        ),
        "Task description": _task_payload(task_desc),
        "Experiment guide": _exp_guide_prompt_view(workspace_dir, hidden_score_path),
        "Environment contract": environment_manifest_text(include_medical=True),
        "Evaluator requirements": [
            "Write exactly one Python file.",
            "Use only Python standard library and allowed installed packages.",
            "Do not train a model. Do not install packages. Do not use network access.",
            "Do not use Python eval() or exec(). Parse serialized values with json.loads or ast.literal_eval.",
            "Use the Experiment guide as the source of truth for public splits, hidden answers, visibility boundaries, and output paths.",
            "Define prediction_spec() returning a concise JSON-serializable dict with these top-level keys: validation_artifact, test_submission, key_columns, prediction_columns, primary_metric. key_columns may be [] only when row order is the contract.",
            METRIC_CONTRACT_REQUIREMENTS,
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
            f"Define score_val() with no required arguments. It must read hidden score truth from os.environ['{_SCORE_ENV_VAR}'], read the candidate validation artifact from ./working, validate the contract, and return a dict containing numeric key 'metric' and boolean key 'maximize'. Add useful extra metric keys when appropriate.",
            "Define validate_submission() with no required arguments. It must validate ./working/submission.csv and any referenced files against the same contract, returning {'valid': True} only when valid and raising a clear exception or returning {'valid': False} otherwise.",
            "For tabular prediction artifacts, align rows by the declared key columns after normalizing keys to strings; reject missing keys, duplicate keys, extra keys, bad shapes, and invalid numeric values.",
            "For probability outputs, validate with numeric tolerance and return a clear error when rows are not valid probabilities.",
            "Reject missing required columns (including split), missing or duplicate keys, incomplete rows, NaN/Inf, and malformed probabilities. Never infer or fill a missing required column, silently repair a candidate artifact, or modify an artifact during validation or scoring.",
            "If the requested primary metric is undefined (including missing classes for ROC AUC), raise a clear exception. Never substitute 0.5, accuracy, or any fallback score. Unsupported or invalid metrics are unavailable, not successful candidate results.",
            f"Define self_test() with no required arguments. It must derive dummy prediction artifacts from the real public score validation and agent_test split CSVs, exactly follow prediction_spec(), call score_val() with os.environ['{_SCORE_ENV_VAR}'] available, call validate_submission(), assert the returned metric is numeric, then print EVALUATOR_OK.",
            "The framework independently tests valid predictions and corrupted versions with each required column removed, missing/duplicate/unexpected keys and invalid numeric values. Both score_val and validate_submission must reject these; a self_test success marker alone is not accepted.",
            "Keep the evaluator simple and deterministic. Use standard metrics from sklearn when directly applicable and small direct numpy/PIL implementations for image masks.",
            "For segmentation masks, support mask files referenced from CSV. The prediction spec should make this explicit with prediction_type='mask_file_path' or file_reference_columns for the mask path column. Binary Dice should treat nonzero pixels as foreground. AJI/instance metrics should use integer connected/instance labels if the task asks for them.",
        ],
        "Output format": (
            "Return a brief note followed by exactly one markdown Python code block. "
            "No extra code blocks."
        ),
    }


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
    popen_kwargs: dict[str, Any] = {
        "cwd": workspace_dir,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
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
        raise RuntimeError("Could not capture evaluator process identity") from None

    started_at = time.monotonic()
    stdout = ""
    stderr = ""
    try:
        while True:
            process_tree.refresh_best_effort()
            if process.poll() is not None:
                # An evaluator descendant can keep the captured PIPEs open after
                # the evaluator leader succeeds. Drain the PGID before collecting
                # output so this cannot be misclassified as a timeout.
                process_tree.terminate_and_wait(root_process=process)
                stdout, stderr = process.communicate()
                return subprocess.CompletedProcess(
                    command,
                    process.returncode,
                    stdout,
                    stderr,
                )
            remaining = timeout - (time.monotonic() - started_at)
            if remaining <= 0:
                process_tree.terminate_and_wait(root_process=process)
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=stdout,
                    stderr=stderr,
                )
            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.25, max(0.05, remaining))
                )
                return subprocess.CompletedProcess(
                    command,
                    process.returncode,
                    stdout,
                    stderr,
                )
            except subprocess.TimeoutExpired:
                continue
    finally:
        process_tree.terminate_and_wait(root_process=process)


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
    code = (
        "import evaluator\n"
        "import json\n"
        "required = ['prediction_spec', 'score_val', 'validate_submission']\n"
        "missing = [name for name in required if not hasattr(evaluator, name)]\n"
        "if missing:\n"
        "    raise RuntimeError('evaluator missing required functions: ' + ', '.join(missing))\n"
        "spec = evaluator.prediction_spec()\n"
        "if isinstance(spec, str):\n"
        "    spec = json.loads(spec)\n"
        "if not isinstance(spec, dict):\n"
        "    raise RuntimeError('prediction_spec() must return a dict')\n"
        "for key in ['validation_artifact', 'test_submission', 'key_columns']:\n"
        "    if key not in spec:\n"
        "        raise RuntimeError('prediction_spec() missing key: ' + key)\n"
        "if 'prediction_columns' not in spec and 'file_reference_columns' not in spec:\n"
        "    raise RuntimeError('prediction_spec() must declare prediction_columns or file_reference_columns')\n"
        "if not hasattr(evaluator, 'self_test'):\n"
        "    raise RuntimeError('evaluator.self_test() is missing')\n"
        "evaluator.self_test()\n"
        "print('EVALUATOR_OK')\n"
    )
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
    return output


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
        return str(metrics[0])
    return None


def _normalize_prediction_spec_json(
    prediction_spec: str,
    *,
    workspace_dir: Path | None = None,
    requested_metric: str | None = None,
) -> str:
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
            "prediction_spec() must declare primary_metric, such as auc, accuracy, binary_dice, or aji."
        )

    if "maximize" in spec and not isinstance(spec["maximize"], bool):
        raise EvaluatorSetupError("prediction_spec() maximize must be a boolean when declared.")

    # Admission precedes freezing/training. A generated scorer supporting some
    # arbitrary name is insufficient: final user-only evaluation must support
    # the exact same metric and task, without substituting another objective.
    task_types = [str(spec["task_type"])] if spec.get("task_type") else []
    requested_metrics = [requested_metric] if requested_metric else []
    requested_directions = []
    if workspace_dir is not None:
        split_dir = workspace_dir / "input" / "biomaker_splits"
        manifest = _split_manifest(split_dir)
        plan = _json_dict(split_dir / "split_plan.json")
        task_types.extend(
            _manifest_head_types(manifest)
            or ([str(plan["task_type"])] if plan.get("task_type") else [])
        )
        declared_metric = _manifest_primary_metric(manifest) or plan.get("primary_metric")
        if declared_metric:
            requested_metrics.append(str(declared_metric))
        metric_contract = _nested_dict(manifest, "metric_contract")
        for container in [
            metric_contract, _nested_dict(metric_contract, "primary_metric"), plan
        ]:
            if "maximize" in container:
                requested_directions.append(container["maximize"])
    if file_reference_columns or spec.get("prediction_type") == "mask_file_path":
        task_types.append("segmentation")
    elif (not task_types or set(task_types) == {"classification"}) and isinstance(
        spec.get("class_labels"), list
    ):
        task_types.append("single_label_classification")
    try:
        info = require_supported_metric(primary_metric)
        if not task_types:
            raise ValueError(
                "prediction_spec() must declare task_type when the public split "
                "contract does not establish a supported task family."
            )
        for task_type in task_types:
            require_supported_metric(primary_metric, task_type)
        for requested in requested_metrics:
            expected = require_supported_metric(requested)
            if expected.name != info.name:
                raise ValueError(
                    f"prediction_spec() primary_metric {primary_metric!r} differs "
                    f"from requested metric {requested!r}; do not substitute objectives."
                )
        if "maximize" in spec and spec["maximize"] != info.maximize:
            raise ValueError(f"prediction_spec() maximize disagrees with primary_metric {primary_metric!r}.")
        if any(
            not isinstance(value, bool) or value != info.maximize
            for value in requested_directions
        ):
            raise ValueError(
                "Split metric contract maximize must be a boolean consistent "
                f"with primary_metric {primary_metric!r}."
            )
        primary_metric = info.name
    except ValueError as exc:
        raise EvaluatorSetupError(str(exc)) from exc

    normalized: dict[str, Any] = {
        "validation_artifact": validation_artifact
        or "./working/validation_predictions.csv",
        "test_submission": test_submission,
        "key_columns": key_columns,
        "prediction_columns": prediction_columns,
        "primary_metric": primary_metric,
        "maximize": info.maximize,
    }
    if file_reference_columns:
        normalized["file_reference_columns"] = [
            column
            for column in file_reference_columns
            if column not in set(key_columns)
        ]
    if info.task_family == "segmentation":
        normalized.setdefault("prediction_type", "mask_file_path")
    for optional_key in [
        "metrics",
        "constraints",
        "notes",
        "class_labels",
        "positive_class",
        "prediction_type",
        "task_type",
        "maximize",
        "score_split",
    ]:
        if optional_key in spec:
            normalized[optional_key] = spec[optional_key]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _should_run_tabular_contract_test(spec: dict[str, Any]) -> bool:
    metric = str(spec.get("primary_metric", "")).lower()
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
    if not isinstance(spec, dict):
        raise EvaluatorSetupError("prediction_spec must be a dictionary.")

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

    if not _should_run_tabular_contract_test(spec):
        # File-reference tasks use the valid artifacts produced by self_test;
        # synthesizing numeric CSV values would destroy their mask/volume paths.
        code = _STRICT_ARTIFACT_CONTRACT_CHECK.replace("__BIOMAKER_SPEC__", repr(spec)).replace("__BIOMAKER_ANSWER_COLUMNS__", "[]")
        completed = _run_python(workspace_dir, code, timeout=timeout,
                                env=_hidden_score_env(workspace_dir, hidden_score_path))
        if completed.returncode != 0:
            output = _sanitize_private_paths((completed.stdout or "") + (completed.stderr or ""), hidden_score_path)
            raise EvaluatorSetupError("Evaluator strict artifact contract test failed.\n" + output[-4000:])
        return

    working_dir = workspace_dir / "working"
    original_working_existed = working_dir.exists()
    backup_root = Path(
        tempfile.mkdtemp(prefix=".evaluator_contract_test_", dir=workspace_dir)
    )
    backup_working = backup_root / "working"
    cleanup_failed = False
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

        score_path = Path(_hidden_score_env(workspace_dir, hidden_score_path)[_SCORE_ENV_VAR])
        answer_columns = _truth_answer_columns(workspace_dir, {}, score_path, pd.read_csv(score_split))
        code = _STRICT_ARTIFACT_CONTRACT_CHECK.replace("__BIOMAKER_SPEC__", repr(spec)).replace("__BIOMAKER_ANSWER_COLUMNS__", repr(answer_columns))
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
                f"load hidden score labels from ${_SCORE_ENV_VAR}. Corrupted artifacts and undefined primary metrics must be rejected without modifying artifacts.\n" + output[-4000:]
            )
    except ProcessTreeCleanupError:
        cleanup_failed = True
        raise
    finally:
        # An unverified evaluator may still write here. Retain both the backup
        # and temporary artifacts for the outer session owner to drain first.
        if not cleanup_failed:
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


def _manifest_head_types(manifest: dict[str, Any]) -> set[str]:
    heads = manifest.get("task_heads")
    if isinstance(heads, str):
        return {heads.lower()}
    result: set[str] = set()
    if isinstance(heads, list):
        for head in heads:
            if isinstance(head, str):
                result.add(head.lower())
            elif isinstance(head, dict) and isinstance(head.get("type"), str):
                result.add(str(head["type"]).lower())
    return result


def _manifest_key_columns(manifest: dict[str, Any]) -> list[str]:
    columns = manifest.get("key_columns")
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns if str(column).strip()]


def _manifest_primary_metric(manifest: dict[str, Any]) -> str | None:
    metric_contract = manifest.get("metric_contract")
    if isinstance(metric_contract, dict) and metric_contract.get("primary_metric"):
        metric = metric_contract["primary_metric"]
        if isinstance(metric, dict):
            for key in ("name", "metric", "id"):
                value = metric.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        return str(metric)
    return None


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
    test_path = split_dir / "agent_test.csv"
    if not (
        train_path.is_file()
        and score_path.is_file()
        and test_path.is_file()
        and hidden_score_path.is_file()
    ):
        return None
    train_df = pd.read_csv(train_path)
    score_df = pd.read_csv(score_path)
    test_df = pd.read_csv(test_path)
    hidden_df = pd.read_csv(hidden_score_path)
    split_plan = _json_dict(split_dir / "split_plan.json")
    split_manifest = _split_manifest(split_dir)
    return {
        "score_path": score_path,
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
    text = _task_payload(task_desc)
    match = re.search(
        r"primary\s+validation\s+metric\s*:\s*([^\n\r.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    metric = match.group(1).strip().strip("`*_ ")
    return canonical_metric_name(metric)


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
        prediction_spec = _normalize_prediction_spec_json(
            _load_prediction_spec(workspace_dir), workspace_dir=workspace_dir
        )
        _run_self_test(
            workspace_dir, timeout=timeout, hidden_score_path=hidden_score_path
        )
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
        primary_metric=spec.get("primary_metric"),
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
    return _write_fixed_evaluator(
        workspace_dir=workspace_dir,
        log_dir=log_dir,
        hidden_score_path=hidden_score_path,
        timeout=timeout,
        spec=spec,
        template=_FIXED_CLASSIFICATION_EVALUATOR,
        run_public_contract=True,
        log_message="Fixed evaluator prepared",
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
    answer_column = mask_candidates[0]

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
        "answer_columns": [answer_column],
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
    spec["maximize"] = metric_maximize(str(spec.get("primary_metric", "")))
    return _write_fixed_evaluator(
        workspace_dir=workspace_dir,
        log_dir=log_dir,
        hidden_score_path=hidden_score_path,
        timeout=timeout,
        spec=spec,
        template=_FIXED_SEGMENTATION_EVALUATOR,
        run_public_contract=False,
        log_message="Fixed segmentation evaluator prepared",
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "exp_guide.py": _sha256_file(utils_dir / "exp_guide.py"),
        "data_summary.py": _sha256_file(utils_dir / "data_summary.py"),
        "metric_registry.py": _sha256_file(utils_dir / "metric_registry.py"),
        "prediction_contract.py": _sha256_file(utils_dir / "prediction_contract.py"),
        "experiment_contract.py": _sha256_file(utils_dir / "experiment_contract.py"),
    }
    runtime_versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "Pillow": _package_version("Pillow"),
        "scipy": _package_version("scipy"),
    }
    prompt_hash = hashlib.sha256(
        json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str).encode(
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
    requested_metric: str | None = None,
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

    evaluator_path = workspace_dir / "evaluator.py"
    prediction_spec_path = workspace_dir / "evaluator_prediction_spec.json"
    try:
        shutil.copy2(cached_evaluator, evaluator_path)
        prediction_spec = _normalize_prediction_spec_json(
            cached_spec.read_text(encoding="utf-8"),
            workspace_dir=workspace_dir,
            requested_metric=requested_metric,
        )
        actual_spec = _normalize_prediction_spec_json(
            _load_prediction_spec(workspace_dir),
            workspace_dir=workspace_dir,
            requested_metric=requested_metric,
        )
        if actual_spec != prediction_spec:
            raise EvaluatorSetupError("Cached contract differs from evaluator prediction_spec().")
        prediction_spec_path.write_text(prediction_spec, encoding="utf-8")
        _run_self_test(
            workspace_dir,
            timeout=timeout,
            hidden_score_path=hidden_score_path,
        )
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
        append_event(
            log_dir,
            "fixed_segmentation_evaluator_failed",
            reason=truncate_text(reason, max_chars=1200),
        )

    prompt = _evaluator_prompt(task_desc, workspace_dir, hidden_score_path)
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
            requested_metric=_primary_metric_from_task_desc(task_desc, ""),
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
        evaluator_path.write_text(code, encoding="utf-8")
        try:
            prediction_spec = _normalize_prediction_spec_json(
                _load_prediction_spec(workspace_dir),
                workspace_dir=workspace_dir,
                requested_metric=_primary_metric_from_task_desc(task_desc, ""),
            )
            _run_self_test(
                workspace_dir,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
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


def _artifact_fingerprints(workspace_dir: Path) -> dict[str, tuple | None]:
    spec = _json_dict(workspace_dir / "evaluator_prediction_spec.json")
    values = [spec.get("validation_artifact"), spec.get("test_submission", "./working/submission.csv")]
    fingerprints = {}
    for value in values:
        if not isinstance(value, str):
            continue
        relative = Path(value.removeprefix("./"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "working":
            continue
        path = workspace_dir / relative
        info = path.stat() if path.is_file() else None
        fingerprints[str(path)] = None if info is None else (info.st_size, info.st_mtime_ns, info.st_ctime_ns, _sha256_file(path))
    return fingerprints


def score_candidate_with_evaluator(
    workspace_dir: Path | str,
    timeout: int = 120,
    hidden_score_path: Path | str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> EvaluatorRunResult:
    """Run the frozen evaluator against the current ./working artifacts."""
    workspace_dir = Path(workspace_dir)
    artifacts_before = _artifact_fingerprints(workspace_dir)
    code = (
        "import json, evaluator\n"
        "validation = None\n"
        "if hasattr(evaluator, 'validate_submission'):\n"
        "    validation = evaluator.validate_submission()\n"
        "score = evaluator.score_val()\n"
        "payload = {'score': score, 'submission_validation': validation}\n"
        f"print('{_RESULT_PREFIX}' + json.dumps(payload, ensure_ascii=False, default=str))\n"
    )
    try:
        evaluator_env = _hidden_score_env(workspace_dir, hidden_score_path)
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
    if _artifact_fingerprints(workspace_dir) != artifacts_before:
        return EvaluatorRunResult(ok=False, metric=None, maximize=True, details={}, output=output,
                                  error="Evaluator modified a candidate artifact; validation must be read-only.")
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
            payload["artifact_report"] = inspect_candidate_artifacts(workspace_dir)
            output_with_report = (
                output
                + "\n"
                + _ARTIFACT_REPORT_PREFIX
                + json.dumps(
                    payload["artifact_report"], ensure_ascii=False, default=str
                )
            )
            score = payload.get("score") or {}
            validation = payload.get("submission_validation")
            if validation is not None or (workspace_dir / "evaluator_prediction_spec.json").is_file():
                _require_submission_success(validation)
            if not isinstance(score, dict):
                raise ValueError("score_val() did not return a metric dictionary.")
            if has_invalid_result(score):
                raise ValueError("score_val() returned a fallback or invalid metric result.")
            metric = score.get("metric")
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                raise ValueError("score_val() did not return numeric 'metric'.")
            metric = float(metric)
            if not math.isfinite(metric):
                raise ValueError("score_val() returned a non-finite metric.")
            maximize = score.get("maximize", True)
            spec_path = workspace_dir / "evaluator_prediction_spec.json"
            if spec_path.is_file():
                try:
                    spec = json.loads(spec_path.read_text(encoding="utf-8"))
                except Exception:
                    spec = {}
                if isinstance(spec, dict) and "maximize" in spec:
                    expected_maximize = spec["maximize"]
                    if not isinstance(expected_maximize, bool):
                        raise ValueError("prediction_spec maximize must be a boolean.")
                    if maximize != expected_maximize:
                        raise ValueError(
                            "score_val() returned maximize="
                            f"{maximize}, but prediction_spec declares "
                            f"maximize={expected_maximize}."
                        )
            return EvaluatorRunResult(
                ok=True,
                metric=metric,
                maximize=maximize,
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


def _require_submission_success(value: Any) -> None:
    if value is True:
        return
    if isinstance(value, dict):
        if has_invalid_result(value):
            raise ValueError("validate_submission() returned an invalid or fallback result.")
        if value.get("valid") is True or value.get("ok") is True:
            return
    raise ValueError("validate_submission() did not explicitly confirm a valid submission.")


def validate_submission_with_evaluator(
    workspace_dir: Path | str,
    timeout: int = 120,
    env_overrides: dict[str, str] | None = None,
) -> EvaluatorRunResult:
    """Run only the frozen evaluator's public test-submission validation."""
    workspace_dir = Path(workspace_dir)
    artifacts_before = _artifact_fingerprints(workspace_dir)
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
    if _artifact_fingerprints(workspace_dir) != artifacts_before:
        return EvaluatorRunResult(ok=False, metric=None, maximize=True, details={}, output=output,
                                  error="Evaluator modified a candidate artifact; validation must be read-only.")
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
            _require_submission_success(payload.get("submission_validation"))
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
