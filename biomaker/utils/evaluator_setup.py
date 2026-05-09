"""Set up and run the frozen evaluator used by generated solutions."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import numpy as np

from .. import backend
from . import data_summary
from .environment import environment_manifest_text, validate_code_environment
from .events import append_event
from .response import extract_code, truncate_text

logger = logging.getLogger("biomaker")

EXTERNAL_VALIDATION_PREFIX = "BioMakerExternalValidation:"
_RESULT_PREFIX = "EVALUATOR_RESULT_JSON="
_SPEC_PREFIX = "EVALUATOR_SPEC_JSON="
_ARTIFACT_REPORT_PREFIX = "EVALUATOR_ARTIFACT_REPORT_JSON="
_SCORE_ENV_VAR = "BIOMAKER_SCORE_VAL_PATH"
_EVALUATOR_CACHE_VERSION = 1


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
            metric = data.get("metric")
            maximize = bool(data.get("maximize", True))
            if isinstance(metric, bool) or metric is None:
                return None
            return float(metric), maximize
        except Exception:
            return None
    return None


def parse_external_validation_details(term_out: list[str] | str) -> dict[str, Any] | None:
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


def _artifact_contract_facts(
    workspace_dir: Path, hidden_score_path: Path | None = None
) -> dict[str, Any]:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    val_path = split_dir / "score_val.csv"
    dev_path = split_dir / "agent_dev.csv"
    hidden_score_path = hidden_score_path or (
        workspace_dir / "private_gt" / "score_val.csv"
    )
    test_path = split_dir / "agent_test.csv"
    if not val_path.is_file() or not test_path.is_file():
        return {}

    try:
        val_df = pd.read_csv(val_path, nrows=3)
        test_df = pd.read_csv(test_path, nrows=3)
        dev_df = pd.read_csv(dev_path, nrows=3) if dev_path.is_file() else None
        hidden_score_df = (
            pd.read_csv(hidden_score_path, nrows=3)
            if hidden_score_path.is_file()
            else None
        )
    except Exception as exc:
        return {"error": str(exc)}

    val_columns = list(val_df.columns)
    test_columns = list(test_df.columns)
    test_set = set(test_columns)
    label_tokens = {
        "annotation",
        "class",
        "diagnosis",
        "finding",
        "grade",
        "ground_truth",
        "gt",
        "instance",
        "label",
        "mask",
        "target",
        "y",
    }
    ground_truth_only = []
    for column in val_columns:
        normalized = column.lower()
        if column not in test_set or any(token in normalized for token in label_tokens):
            if column not in test_set or column not in {
                "image_path",
                "path",
                "file_path",
            }:
                ground_truth_only.append(column)

    common_columns = [column for column in val_columns if column in test_set]
    likely_key_columns = [
        column
        for column in common_columns
        if not any(token in column.lower() for token in ["path", "image", "mask"])
    ]

    private_gt_manifest = None
    for candidate in [
        hidden_score_path.parent / "private_gt.json",
        workspace_dir / "private_gt" / "private_gt.json",
    ]:
        if candidate.is_file():
            try:
                private_gt_manifest = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except Exception:
                private_gt_manifest = None

    return {
        "score_val_public_columns": val_columns,
        "agent_dev_columns": list(dev_df.columns) if dev_df is not None else None,
        "hidden_score_answer_columns": (
            list(hidden_score_df.columns) if hidden_score_df is not None else None
        ),
        "agent_test_columns": test_columns,
        "ground_truth_only_columns": ground_truth_only,
        "private_gt": private_gt_manifest,
        "likely_key_columns": likely_key_columns,
        "note": (
            "Candidate validation artifacts are predictions for score_val.csv and may use "
            "key columns plus new prediction columns only. Hidden score labels are not "
            "present in score_val.csv; score_val() must load them from the path in "
            f"environment variable {_SCORE_ENV_VAR}."
        ),
    }


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
        "Data and evaluation summary": data_summary.data_and_evaluation_summary(
            workspace_dir
        ),
        "Observed artifact contract facts": _artifact_contract_facts(
            workspace_dir, hidden_score_path
        ),
        "Environment contract": environment_manifest_text(include_medical=True),
        "Evaluator requirements": [
            "Write exactly one Python file.",
            "Use only Python standard library and allowed installed packages.",
            "Do not train a model. Do not install packages. Do not use network access.",
            "Do not use Python eval() or exec(). Parse serialized values with json.loads or ast.literal_eval.",
            "Use files under ./input/biomaker_splits as the source of truth for public train/development/score/test split.",
            "Define prediction_spec() returning a concise JSON-serializable dict with these top-level keys: validation_artifact, test_submission, key_columns, prediction_columns, primary_metric. key_columns may be [] only when row order is the contract.",
            "test_submission must be exactly './working/submission.csv'.",
            "prediction_columns means the CSV columns containing predictions or referenced prediction files. Do not include key columns in prediction_columns.",
            "The candidate validation artifact must contain predictions for rows in ./input/biomaker_splits/score_val.csv.",
            "The candidate validation artifact must follow the same public prediction contract as the test submission: key_columns plus prediction_columns only. Do not require candidates to output validation ground-truth columns such as label, class, target, or true_class. score_val() must load hidden score labels from os.environ['BIOMAKER_SCORE_VAL_PATH'] and align them to the validation artifact by key columns.",
            "When reading candidate artifacts, validate the required columns and then restrict the dataframe to key_columns plus prediction_columns before merging with ground truth. Extra candidate columns such as class, label, target, or true_class must not collide with ground-truth columns or change scoring.",
            "For tabular score_val(), use this safe merge pattern: copy the hidden ground-truth label column to a private column named `y_true_private`, build `truth_view = hidden_answers[key_columns + ['y_true_private']]`, build `pred_view = predictions[key_columns + prediction_columns]`, convert every key column in both views to string before merging, merge those restricted views only, and read labels with dataframe/series indexing such as `merged['y_true_private']` or `row['y_true_private']`. Never rely on tuple attribute access for helper columns, and never read `merged['class']`, `merged['label']`, `merged['target']`, or candidate-supplied label-like columns after merge.",
            "For tabular validate_submission(), apply the same key-column string normalization and restrict submission data to key_columns plus prediction_columns before checking duplicates, missing keys, or shapes.",
            "For probability outputs, include class_labels when prediction_columns do not directly encode class names; for single-column binary probability outputs, include positive_class.",
            "The submission contract must preserve enough information for the requested user-only test metrics. For classification with AUC, require probabilities rather than only hard labels.",
            f"Define score_val() with no required arguments. It must read hidden score truth from os.environ['{_SCORE_ENV_VAR}'], read the candidate validation artifact from ./working, validate the contract, and return a dict containing numeric key 'metric' and boolean key 'maximize'. Add useful extra metric keys when appropriate.",
            "Define validate_submission() with no required arguments. It must validate ./working/submission.csv and any referenced files against the same contract, returning a dict or raising a clear exception.",
            "For tabular prediction artifacts, align rows by the declared key columns after normalizing keys to strings; reject missing keys, duplicate keys, extra keys, bad shapes, and invalid numeric values.",
            "For probability outputs, validate with numeric tolerance and return a clear error when rows are not valid probabilities.",
            "If a primary metric is mathematically undefined on the current validation split, return a deterministic fallback metric with an explanatory field instead of crashing.",
            f"Define self_test() with no required arguments. It must derive dummy prediction artifacts from the real public score_val and agent_test split CSVs, exactly follow prediction_spec(), call score_val() with os.environ['{_SCORE_ENV_VAR}'] available, call validate_submission(), assert the returned metric is numeric, then print EVALUATOR_OK.",
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
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


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


def _run_self_test(
    workspace_dir: Path,
    timeout: int,
    hidden_score_path: Path | str | None = None,
) -> str:
    code = (
        "import evaluator\n"
        "import json\n"
        "assert hasattr(evaluator, 'prediction_spec')\n"
        "assert hasattr(evaluator, 'score_val')\n"
        "assert hasattr(evaluator, 'validate_submission')\n"
        "spec = evaluator.prediction_spec()\n"
        "if isinstance(spec, str):\n"
        "    spec = json.loads(spec)\n"
        "assert isinstance(spec, dict)\n"
        "assert 'validation_artifact' in spec\n"
        "assert 'test_submission' in spec\n"
        "assert 'key_columns' in spec\n"
        "assert ('prediction_columns' in spec) or ('file_reference_columns' in spec)\n"
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

    normalized: dict[str, Any] = {
        "validation_artifact": validation_artifact
        or "./working/validation_predictions.csv",
        "test_submission": test_submission,
        "key_columns": key_columns,
        "prediction_columns": prediction_columns,
        "primary_metric": primary_metric,
    }
    metric_lower = primary_metric.lower()
    if any(token in metric_lower for token in ["aji", "dice", "mask", "segmentation"]):
        normalized.setdefault("prediction_type", "mask_file_path")
    for optional_key in [
        "metrics",
        "constraints",
        "notes",
        "class_labels",
        "positive_class",
        "prediction_type",
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
    if not isinstance(spec, dict) or not _should_run_tabular_contract_test(spec):
        return

    key_columns = _list_of_strings(spec.get("key_columns"))
    prediction_columns = _list_of_strings(spec.get("prediction_columns"))
    class_labels = spec.get("class_labels")
    if not isinstance(class_labels, list):
        class_labels = None

    clean_candidate_outputs(workspace_dir)
    score_split = workspace_dir / "input" / "biomaker_splits" / "score_val.csv"
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
        "assert isinstance(score.get('metric'), (int, float)), score\n"
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
        raise EvaluatorSetupError(
            "Evaluator public artifact contract test failed. "
            "score_val() must accept validation predictions containing only "
            "key_columns plus prediction_columns from prediction_spec(); it should "
            f"load hidden score labels from ${_SCORE_ENV_VAR}.\n"
            + output[-4000:]
        )


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
    truth_keys = truth_df[key_columns].astype(str)
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

        truth_path = workspace_dir / "input" / "biomaker_splits" / "score_val.csv"
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
            else (None, {"error": "score_val.csv missing"})
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


def _stable_evaluator_cache_key(prompt: dict[str, Any], model: str) -> str:
    payload = {
        "cache_version": _EVALUATOR_CACHE_VERSION,
        "model": model,
        "prompt": prompt,
    }
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
        _run_public_artifact_contract_test(
            workspace_dir,
            prediction_spec,
            timeout=timeout,
            hidden_score_path=hidden_score_path,
        )
    except Exception as exc:
        append_event(
            log_dir,
            "evaluator_cache_reuse_failed",
            cache_entry=str(cache_entry),
            reason=truncate_text(str(exc), max_chars=1200),
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
    cache_entry: Path, evaluator_path: Path, prediction_spec: str
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
    (tmp_dir / "manifest.json").write_text(
        json.dumps(
            {"cache_version": _EVALUATOR_CACHE_VERSION},
            indent=2,
            ensure_ascii=False,
        ),
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
    prompt = _evaluator_prompt(task_desc, workspace_dir, hidden_score_path)
    cache_entry = None
    if cache_root is not None:
        cache_key = _stable_evaluator_cache_key(prompt, model=model)
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
        evaluator_path.write_text(code, encoding="utf-8")
        try:
            _run_self_test(
                workspace_dir,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
            prediction_spec = _normalize_prediction_spec_json(
                _load_prediction_spec(workspace_dir)
            )
            _run_public_artifact_contract_test(
                workspace_dir,
                prediction_spec,
                timeout=timeout,
                hidden_score_path=hidden_score_path,
            )
        except Exception as exc:
            errors.append(str(exc))
            append_event(
                log_dir,
                "evaluator_attempt_failed",
                attempt=attempt,
                reason=truncate_text(str(exc), max_chars=1200),
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
            _store_cached_evaluator(cache_entry, evaluator_path, prediction_spec)
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
) -> EvaluatorRunResult:
    """Run the frozen evaluator against the current ./working artifacts."""
    workspace_dir = Path(workspace_dir)
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
        completed = _run_python(
            workspace_dir,
            code,
            timeout=timeout,
            env=_hidden_score_env(workspace_dir, hidden_score_path),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return EvaluatorRunResult(
            ok=False,
            metric=None,
            maximize=True,
            details={},
            output=output,
            error=f"Evaluator timed out after {timeout} seconds.",
        )

    output = (completed.stdout or "") + (completed.stderr or "")
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
            metric = score.get("metric")
            if isinstance(metric, bool) or metric is None:
                raise ValueError("score_val() did not return numeric 'metric'.")
            metric = float(metric)
            if not math.isfinite(metric):
                raise ValueError("score_val() returned a non-finite metric.")
            maximize = bool(score.get("maximize", True))
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
