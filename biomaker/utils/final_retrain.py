"""Final-only retrain of the selected best solution."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..interpreter import ExecutionResult, Interpreter
from ..journal import Node
from .artifact_validation import normalize_working_outputs, validate_working_outputs
from .config import Config, _working_copy_ignore, evaluator_timeout_seconds, evaluator_workers
from .evaluator_setup import clean_candidate_outputs, validate_submission_with_evaluator
from .execution_record import (
    attach_artifact_validation_result,
    attach_evaluator_run_result,
)
from .experiment_contract import load_experiment_contract

logger = logging.getLogger("biomaker")


@dataclass
class FinalRetrainResult:
    ok: bool
    workspace_dir: Path
    train_rows_before: int | None = None
    score_rows_added: int | None = None
    train_rows_after: int | None = None
    exec_result: ExecutionResult | None = None
    private_metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "workspace_dir": str(self.workspace_dir),
            "train_rows_before": self.train_rows_before,
            "score_rows_added": self.score_rows_added,
            "train_rows_after": self.train_rows_after,
            "exec_time": (
                None if self.exec_result is None else self.exec_result.exec_time
            ),
            "exc_type": None if self.exec_result is None else self.exec_result.exc_type,
            "exc_info": None if self.exec_result is None else self.exec_result.exc_info,
            "execution": (
                None
                if self.exec_result is None
                else self.exec_result.execution_record()
            ),
            "private_metrics": self.private_metrics,
            "error": self.error,
        }


def _replace_path_with_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def _copy_input_for_retrain(
    source_input: Path, retrain_input: Path, *, train_path: Path,
) -> None:
    retrain_input.mkdir(parents=True, exist_ok=True)
    for child in source_input.iterdir():
        if child.name == "biomaker_splits":
            shutil.copytree(child, retrain_input / child.name)
            continue
        if child == train_path:
            shutil.copy2(child, retrain_input / child.name)
        elif child in train_path.parents:
            _copy_input_for_retrain(child, retrain_input / child.name, train_path=train_path)
        else:
            _replace_path_with_symlink(retrain_input / child.name, child)


def _score_answer_columns(
    workspace_dir: Path,
    train_df: pd.DataFrame,
    score_visible_df: pd.DataFrame,
) -> list[str]:
    contract = load_experiment_contract(workspace_dir)
    answer_columns = contract.hidden_answer_columns("score_val")
    if answer_columns:
        return answer_columns
    if contract.split_manifest.get("manifest_version") == 2:
        raise ValueError("Canonical final retraining requires declared score_val answer columns.")

    # Fallback: generated code sees score_val without answers, while train has them.
    return [col for col in train_df.columns if col not in score_visible_df.columns]


def _score_key_columns(
    workspace_dir: Path,
    score_visible_df: pd.DataFrame,
    score_gt_df: pd.DataFrame,
) -> list[str]:
    contract = load_experiment_contract(workspace_dir)
    keys = contract.prediction.key_columns
    if keys and all(
        col in score_visible_df.columns and col in score_gt_df.columns for col in keys
    ):
        return keys
    if contract.split_manifest.get("manifest_version") == 2:
        if not keys:
            if contract.split_manifest.get("row_order_contract") is True:
                return []
            raise ValueError("Canonical final retraining requires declared sample key columns.")
        missing_public = [key for key in keys if key not in score_visible_df.columns]
        missing_hidden = [key for key in keys if key not in score_gt_df.columns]
        raise ValueError(
            "Declared sample key columns are missing from final retrain inputs: "
            f"score_val={missing_public}, hidden answers={missing_hidden}."
        )

    if "Id" in score_visible_df.columns and "Id" in score_gt_df.columns:
        return ["Id"]

    common = [
        col
        for col in score_visible_df.columns
        if col in score_gt_df.columns and col != "path"
    ]
    if common:
        return [common[0]]
    raise ValueError("Could not infer key columns for score-validation label merge.")


def _merge_visible_score_with_hidden_labels(
    workspace_dir: Path,
    train_df: pd.DataFrame,
    score_gt_df: pd.DataFrame,
) -> pd.DataFrame:
    contract = load_experiment_contract(workspace_dir)
    score_visible_path = contract.split_path("score_val")
    if score_visible_path is None:
        raise ValueError("Final retraining requires a declared score_val split.")
    if not score_visible_path.is_file():
        raise FileNotFoundError(f"Missing visible score split: {score_visible_path}")

    score_visible_df = pd.read_csv(score_visible_path)
    answer_columns = _score_answer_columns(workspace_dir, train_df, score_visible_df)
    key_columns = _score_key_columns(workspace_dir, score_visible_df, score_gt_df)

    missing_answers = [col for col in answer_columns if col not in score_gt_df.columns]
    if missing_answers:
        raise ValueError(
            f"Hidden score labels are missing answer columns: {missing_answers}"
        )

    label_source = score_gt_df[key_columns + answer_columns].copy()
    if not key_columns:
        if len(score_visible_df) != len(label_source):
            raise ValueError("Row-order score label merge requires equal public and private row counts.")
        leaked_answers = set(answer_columns) & set(score_visible_df.columns)
        if leaked_answers:
            raise ValueError(
                "Public score rows contain hidden answer columns: " + ", ".join(sorted(leaked_answers))
            )
        score_labeled = pd.concat(
            [score_visible_df.reset_index(drop=True), label_source.reset_index(drop=True)], axis=1,
        )
    else:
        if label_source.duplicated(key_columns).any():
            raise ValueError(f"Hidden score labels contain duplicate keys: {key_columns}")
        merge_indicator = "__biomaker_score_label_merge"
        score_labeled = score_visible_df.merge(
            label_source,
            on=key_columns,
            how="left",
            validate="one_to_one",
            indicator=merge_indicator,
        )
        unmatched_rows = score_labeled[merge_indicator] != "both"
        if unmatched_rows.any():
            raise ValueError(
                "Hidden score label merge produced unmatched score rows: "
                f"{int(unmatched_rows.sum())}"
            )
        score_labeled = score_labeled.drop(columns=[merge_indicator])

    missing_train_columns = [
        col for col in train_df.columns if col not in score_labeled.columns
    ]
    if missing_train_columns:
        raise ValueError(
            "Visible score split cannot be reshaped to training schema; "
            f"missing columns: {missing_train_columns}"
        )
    return score_labeled[list(train_df.columns)]


def _is_file_answer_column(column: str) -> bool:
    normalized = str(column).strip().casefold()
    if not normalized or "json" in normalized:
        return False
    return "path" in normalized or normalized.endswith(("_file", "_filename"))


def _resolve_private_answer_file(
    workspace_dir: Path,
    private_dir: Path,
    value: Any,
) -> Path:
    text = str(value).strip()
    if not text or text.startswith(("[", "{")):
        raise ValueError(f"Hidden file answer is not a path: {value!r}")
    raw = Path(text.removeprefix("./"))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if raw.parts and raw.parts[0].casefold() == "private_gt":
            tail = Path(*raw.parts[1:])
            candidates.extend([private_dir / tail, private_dir.parent / raw])
        candidates.extend(
            [
                private_dir / raw,
                workspace_dir / raw,
                workspace_dir / "input" / raw,
            ]
        )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Hidden score-validation file answer is unavailable: {value!r}"
    )


def _materialize_score_file_answers(
    workspace_dir: Path,
    private_dir: Path,
    score_labeled_df: pd.DataFrame,
    answer_columns: list[str],
    *,
    file_reference_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy file-valued score answers into the selected-node retrain workspace."""

    materialized = score_labeled_df.copy()
    root = workspace_dir / "input" / "final_retrain_score_answers"
    copied_by_source: dict[Path, Path] = {}
    rewritten_values = 0
    file_columns: list[str] = []

    for column in answer_columns:
        is_file = (
            column in file_reference_columns
            if file_reference_columns is not None
            else _is_file_answer_column(column)
        )
        if column not in materialized.columns or not is_file:
            continue
        values = materialized[column]
        non_null = values[values.notna()]
        if non_null.empty:
            continue
        file_columns.append(column)
        column_dir = root / "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(column)
        )
        rewritten: dict[Any, str] = {}
        for index, value in non_null.items():
            source = _resolve_private_answer_file(
                workspace_dir,
                private_dir,
                value,
            )
            destination = copied_by_source.get(source)
            if destination is None:
                digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
                destination = column_dir / f"{digest}-{source.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_by_source[source] = destination
            rewritten[index] = destination.relative_to(workspace_dir).as_posix()
        for index, value in rewritten.items():
            materialized.at[index, column] = value
            rewritten_values += 1

    return materialized, {
        "columns": file_columns,
        "unique_files_copied": len(copied_by_source),
        "values_rewritten": rewritten_values,
        "workspace_root": (
            root.relative_to(workspace_dir).as_posix() if file_columns else None
        ),
    }


def _write_retrain_training_table(
    workspace_dir: Path,
    private_dir: Path,
) -> tuple[int, int, int, dict[str, Any]]:
    contract = load_experiment_contract(workspace_dir, hidden_answers_dir=private_dir)
    train_path = contract.split_path("train")
    hidden = contract.hidden_answers.get("score_val")
    if train_path is None or hidden is None:
        raise ValueError("Final retraining requires declared train and score_val answers.")
    score_gt_path = hidden.path
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing training split: {train_path}")
    if not score_gt_path.is_file():
        raise FileNotFoundError(
            f"Missing hidden score-validation labels: {score_gt_path}"
        )

    train_df = pd.read_csv(train_path)
    score_gt_df = pd.read_csv(score_gt_path)
    score_labeled_df = _merge_visible_score_with_hidden_labels(
        workspace_dir, train_df, score_gt_df
    )
    declared_file_columns = hidden.file_reference_columns
    if declared_file_columns is None and contract.split_manifest.get("manifest_version") == 2:
        declared_file_columns = []
    score_labeled_df, file_materialization = _materialize_score_file_answers(
        workspace_dir,
        private_dir,
        score_labeled_df,
        _score_answer_columns(
            workspace_dir, train_df, pd.read_csv(contract.split_path("score_val"))
        ),
        file_reference_columns=declared_file_columns,
    )
    retrain_train = pd.concat(
        [train_df, score_labeled_df], ignore_index=True, sort=False
    )
    train_path.with_suffix(".search.csv").write_text(
        train_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    retrain_train.to_csv(train_path, index=False)
    return (
        int(len(train_df)), int(len(score_labeled_df)), int(len(retrain_train)),
        file_materialization,
    )


def _copy_retrain_outputs_to_log(cfg: Config, retrain_workspace: Path) -> None:
    log_dir = cfg.log_dir / "final_retrain"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    working_dir = retrain_workspace / "working"
    if working_dir.is_dir():
        shutil.copytree(
            working_dir, log_dir / "working", symlinks=True,
            ignore=_working_copy_ignore(retrain_workspace, working_dir),
        )

    runfile = retrain_workspace / cfg.exec.agent_file_name
    if runfile.is_file():
        shutil.copy2(runfile, log_dir / cfg.exec.agent_file_name)

    test_job = load_experiment_contract(retrain_workspace).prediction_jobs.get("test")
    if test_job is not None and test_job.artifact.is_file():
        shutil.copy2(test_job.artifact, cfg.log_dir / "final_submission.csv")

    for name in ["pred_masks", "best_ckpt.pth"]:
        src = working_dir / name
        dst = cfg.log_dir / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=True)
        elif src.is_file():
            shutil.copy2(src, dst)


def _write_result(cfg: Config, result: FinalRetrainResult) -> None:
    payload = result.to_json()
    path = cfg.log_dir / "final_retrain" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (cfg.log_dir / "final_retrain_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def prepare_final_retrain_workspace(
    cfg: Config, private_dir: Path
) -> tuple[Path, dict[str, Any]]:
    retrain_workspace = cfg.workspace_dir / "final_retrain"
    if retrain_workspace.exists():
        shutil.rmtree(retrain_workspace)
    retrain_workspace.mkdir(parents=True)
    (retrain_workspace / "working").mkdir()

    source_contract = load_experiment_contract(cfg.workspace_dir, hidden_answers_dir=private_dir)
    source_train = source_contract.split_path("train")
    if source_train is None:
        raise ValueError("Final retraining requires a declared train split.")
    source_train.relative_to(cfg.workspace_dir / "input")
    _copy_input_for_retrain(
        cfg.workspace_dir / "input", retrain_workspace / "input", train_path=source_train,
    )
    for filename in ["evaluator.py", "evaluator_prediction_spec.json"]:
        src = cfg.workspace_dir / filename
        if src.exists():
            _replace_path_with_symlink(retrain_workspace / filename, src)

    before, added, after, file_materialization = _write_retrain_training_table(
        retrain_workspace, private_dir
    )
    retrain_contract = load_experiment_contract(retrain_workspace, hidden_answers_dir=private_dir)
    manifest = {
        "mode": "final_retrain_after_node_selection",
        "source_train": str(source_train),
        "source_score_labels": str(source_contract.hidden_answers["score_val"].path),
        "retrain_train": str(retrain_contract.split_path("train")),
        "train_rows_before": before,
        "score_rows_added": added,
        "train_rows_after": after,
        "score_file_answers": file_materialization,
        "note": (
            "Hidden score-validation labels are merged into training only after "
            "the best node has already been selected."
        ),
    }
    (retrain_workspace / "final_retrain_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return retrain_workspace, manifest


def run_final_retrain(
    cfg: Config,
    best_node: Node,
    private_dir: Path,
    *,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
) -> FinalRetrainResult:
    """Rerun the selected best node after adding score-validation labels to train."""
    result = FinalRetrainResult(
        ok=False,
        workspace_dir=cfg.workspace_dir / "final_retrain",
    )

    try:
        retrain_workspace, manifest = prepare_final_retrain_workspace(cfg, private_dir)
        result.workspace_dir = retrain_workspace
        result.train_rows_before = manifest["train_rows_before"]
        result.score_rows_added = manifest["score_rows_added"]
        result.train_rows_after = manifest["train_rows_after"]

        clean_candidate_outputs(retrain_workspace)
        interpreter = Interpreter(
            retrain_workspace,
            timeout=timeout,
            format_tb_ipython=cfg.exec.format_tb_ipython,
            agent_file_name=cfg.exec.agent_file_name,
            enforce_environment=cfg.exec.enforce_environment,
            include_medical_packages=cfg.exec.include_medical_packages,
            environment=env_overrides,
        )
        exec_result = interpreter.run(best_node.code, reset_session=True)
        result.exec_result = exec_result
        if exec_result.exc_type is not None:
            result.error = f"Final retrain execution failed: {exec_result.exc_type}"
            return result

        normalization_messages = normalize_working_outputs(
            retrain_workspace, "final retrain"
        )
        if normalization_messages:
            exec_result.append_system_output(
                "artifact_normalization",
                "\n".join(normalization_messages),
                details={"messages": normalization_messages},
            )
        artifact_check = validate_working_outputs(
            retrain_workspace, "final retrain", candidate_code=best_node.code,
            model_probe=interpreter.run,
            model_artifact_required=getattr(getattr(cfg, "model_artifacts", None), "required", True),
        )
        attach_artifact_validation_result(
            exec_result,
            artifact_check,
            failure_reason="final retrain artifacts failed validation",
        )
        if not artifact_check.ok:
            result.error = "Final retrain artifacts failed validation."
            return result

        if "test" in load_experiment_contract(retrain_workspace).prediction_jobs:
            submission_check = validate_submission_with_evaluator(
                retrain_workspace,
                timeout=evaluator_timeout_seconds(cfg),
                env_overrides=env_overrides,
            )
            attach_evaluator_run_result(
                exec_result,
                submission_check,
                failure_phase="final_retrain_evaluator_validation",
                failure_type="EvaluatorValidationError",
                failure_message_prefix="FinalRetrainEvaluatorValidationError",
                extra_details={"mode": "final_retrain_submission_validation"},
            )
            if not submission_check.ok:
                result.error = (
                    "Final retrain submission failed frozen evaluator validation: "
                    + (submission_check.error or "unknown error")
                )
                return result

        _copy_retrain_outputs_to_log(cfg, retrain_workspace)
        result.ok = True
        return result
    except Exception as exc:
        logger.warning("Final retrain failed: %s", exc)
        result.error = str(exc)
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            (cfg.log_dir / "final_retrain").mkdir(parents=True, exist_ok=True)
            (cfg.log_dir / "final_retrain" / "completed_at.txt").write_text(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                encoding="utf-8",
            )
            _write_result(cfg, result)
        except Exception as exc:
            if not active_exception:
                raise
            try:
                logger.warning(
                    "Could not persist final retrain interruption details: %s", exc
                )
            except Exception:
                pass
