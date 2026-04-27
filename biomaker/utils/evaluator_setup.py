"""Set up and run the frozen evaluator used by generated solutions."""

from __future__ import annotations

import ast
import json
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import backend
from . import data_preview
from .environment import environment_manifest_text, validate_code_environment
from .response import extract_code

logger = logging.getLogger("biomaker")

EXTERNAL_VALIDATION_PREFIX = "BioMakerExternalValidation:"
_RESULT_PREFIX = "EVALUATOR_RESULT_JSON="
_SPEC_PREFIX = "EVALUATOR_SPEC_JSON="


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
        payload = {
            "metric": self.metric,
            "maximize": self.maximize,
            "details": self.details,
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


def _task_payload(task_desc: Any) -> str:
    if isinstance(task_desc, str):
        return task_desc
    return json.dumps(task_desc, indent=2, ensure_ascii=False, default=str)


def _artifact_contract_facts(workspace_dir: Path) -> dict[str, Any]:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    val_path = split_dir / "agent_val.csv"
    test_path = split_dir / "agent_test.csv"
    if not val_path.is_file() or not test_path.is_file():
        return {}

    try:
        val_df = pd.read_csv(val_path, nrows=3)
        test_df = pd.read_csv(test_path, nrows=3)
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
            if column not in test_set or column not in {"image_path", "path", "file_path"}:
                ground_truth_only.append(column)

    common_columns = [column for column in val_columns if column in test_set]
    likely_key_columns = [
        column
        for column in common_columns
        if not any(token in column.lower() for token in ["path", "image", "mask"])
    ]

    return {
        "agent_val_columns": val_columns,
        "agent_test_columns": test_columns,
        "ground_truth_only_columns": ground_truth_only,
        "likely_key_columns": likely_key_columns,
        "note": (
            "Candidate artifacts may use key columns and new prediction columns only. "
            "Columns listed as ground_truth_only_columns must be loaded from agent_val.csv "
            "inside score_val(), not required from candidate validation/test artifacts."
        ),
    }


def _evaluator_prompt(task_desc: Any, workspace_dir: Path) -> dict[str, Any]:
    return {
        "Introduction": (
            "You are writing the frozen evaluator for a machine learning task. "
            "This evaluator is created once before solution search. Candidate solutions "
            "will train models and write prediction artifacts; BioMaker will call this evaluator "
            "after each candidate finishes, so all candidates are scored with the same code."
        ),
        "Task description": _task_payload(task_desc),
        "Data overview": data_preview.generate(workspace_dir),
        "Observed artifact contract facts": _artifact_contract_facts(workspace_dir),
        "Environment contract": environment_manifest_text(include_medical=True),
        "Evaluator requirements": [
            "Write exactly one Python file.",
            "Use only Python standard library and allowed installed packages.",
            "Do not train a model. Do not install packages. Do not use network access.",
            "Do not use Python eval() or exec(). Parse serialized values with json.loads or ast.literal_eval.",
            "Use files under ./input/biomaker_splits as the source of truth for train/validation/test split.",
            "Define prediction_spec() returning a concise JSON-serializable dict with these top-level keys: validation_artifact, test_submission, key_columns, prediction_columns, primary_metric. key_columns may be [] only when row order is the contract.",
            "test_submission must be exactly './working/submission.csv'.",
            "prediction_columns means the CSV columns containing predictions or referenced prediction files. Do not include key columns in prediction_columns.",
            "The candidate validation artifact must follow the same public prediction contract as the test submission: key_columns plus prediction_columns only. Do not require candidates to output validation ground-truth columns such as label, class, target, or true_class. score_val() must load validation labels from ./input/biomaker_splits/agent_val.csv and align them to the validation artifact by key columns.",
            "When reading candidate artifacts, validate the required columns and then restrict the dataframe to key_columns plus prediction_columns before merging with ground truth. Extra candidate columns such as class, label, target, or true_class must not collide with ground-truth columns or change scoring.",
            "For tabular score_val(), use this safe merge pattern: copy the ground-truth label column to a private column named `__y_true`, build `truth_view = agent_val[key_columns + ['__y_true']]`, build `pred_view = predictions[key_columns + prediction_columns]`, convert every key column in both views to string before merging, merge those restricted views only, and read labels from `merged['__y_true']`. Never read `merged['class']`, `merged['label']`, `merged['target']`, or candidate-supplied label-like columns after merge.",
            "For tabular validate_submission(), apply the same key-column string normalization and restrict submission data to key_columns plus prediction_columns before checking duplicates, missing keys, or shapes.",
            "For probability outputs, include class_labels when prediction_columns do not directly encode class names; for single-column binary probability outputs, include positive_class.",
            "The submission contract must preserve enough information for the requested user-only test metrics. For classification with AUC, require probabilities rather than only hard labels.",
            "Define score_val() with no required arguments. It must read the candidate validation artifact from ./working, validate the contract, and return a dict containing numeric key 'metric' and boolean key 'maximize'. Add useful extra metric keys when appropriate.",
            "Define validate_submission() with no required arguments. It must validate ./working/submission.csv and any referenced files against the same contract, returning a dict or raising a clear exception.",
            "For tabular prediction artifacts, align rows by the declared key columns after normalizing keys to strings; reject missing keys, duplicate keys, extra keys, bad shapes, and invalid numeric values.",
            "For probability outputs, validate with numeric tolerance and return a clear error when rows are not valid probabilities.",
            "If a primary metric is mathematically undefined on the current validation split, return a deterministic fallback metric with an explanatory field instead of crashing.",
            "Define self_test() with no required arguments. It must derive dummy prediction artifacts from the real agent_val and agent_test split CSVs, exactly follow prediction_spec(), call score_val() and validate_submission(), assert the returned metric is numeric, then print EVALUATOR_OK.",
            "Keep the evaluator simple and deterministic. Use standard metrics from sklearn when directly applicable and small direct numpy/PIL implementations for image masks.",
            "For segmentation masks, support mask files referenced from CSV. The prediction spec should make this explicit with prediction_type='mask_file_path' or file_reference_columns for the mask path column. Binary Dice should treat nonzero pixels as foreground. AJI/instance metrics should use integer connected/instance labels if the task asks for them.",
        ],
        "Output format": (
            "Return a brief note followed by exactly one markdown Python code block. "
            "No extra code blocks."
        ),
    }


def _run_python(
    workspace_dir: Path, code: str, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _uses_eval_or_exec(code: str) -> bool:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec"}:
                return True
    return False


def _run_self_test(workspace_dir: Path, timeout: int) -> str:
    code = (
        "import evaluator\n"
        "assert hasattr(evaluator, 'prediction_spec')\n"
        "assert hasattr(evaluator, 'score_val')\n"
        "assert hasattr(evaluator, 'validate_submission')\n"
        "assert hasattr(evaluator, 'self_test')\n"
        "evaluator.self_test()\n"
    )
    completed = _run_python(workspace_dir, code, timeout=timeout)
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
    _write_dummy_tabular_artifact(
        workspace_dir / "input" / "biomaker_splits" / "agent_val.csv",
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
    completed = _run_python(workspace_dir, code, timeout=timeout)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise EvaluatorSetupError(
            "Evaluator public artifact contract test failed. "
            "score_val() must accept validation predictions containing only "
            "key_columns plus prediction_columns from prediction_spec(); it should "
            "load validation labels from ./input/biomaker_splits/agent_val.csv.\n"
            + output[-4000:]
        )


def prepare_evaluator(
    workspace_dir: Path | str,
    log_dir: Path | str,
    task_desc: Any,
    model: str,
    temperature: float = 0.0,
    attempts: int = 5,
    timeout: int = 120,
) -> PreparedEvaluator:
    """Generate, smoke-test, and freeze evaluator.py in the workspace."""
    workspace_dir = Path(workspace_dir)
    log_dir = Path(log_dir)
    evaluator_path = workspace_dir / "evaluator.py"
    prompt = _evaluator_prompt(task_desc, workspace_dir)
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
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
            continue

        preflight = validate_code_environment(code, include_medical=True)
        if not preflight.ok:
            errors.append(preflight.message)
            continue
        if _uses_eval_or_exec(code):
            errors.append(
                "Evaluator uses eval() or exec(). Use json.loads or ast.literal_eval instead."
            )
            continue

        clean_candidate_outputs(workspace_dir)
        evaluator_path.write_text(code, encoding="utf-8")
        try:
            _run_self_test(workspace_dir, timeout=timeout)
            prediction_spec = _normalize_prediction_spec_json(
                _load_prediction_spec(workspace_dir)
            )
            _run_public_artifact_contract_test(
                workspace_dir,
                prediction_spec,
                timeout=timeout,
            )
        except Exception as exc:
            errors.append(str(exc))
            continue

        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evaluator_path, log_dir / "evaluator.py")
        (workspace_dir / "evaluator_prediction_spec.json").write_text(
            prediction_spec, encoding="utf-8"
        )
        (log_dir / "evaluator_prediction_spec.json").write_text(
            prediction_spec, encoding="utf-8"
        )
        clean_candidate_outputs(workspace_dir)
        logger.info("Evaluator prepared at %s", evaluator_path)
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
    workspace_dir: Path | str, timeout: int = 120
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
        completed = _run_python(workspace_dir, code, timeout=timeout)
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
                output=output,
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
