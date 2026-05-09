"""Final-only retrain of the selected best solution."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..interpreter import ExecutionResult, Interpreter
from ..journal import Node
from .artifact_validation import normalize_working_outputs, validate_working_outputs
from .config import Config
from .evaluator_setup import clean_candidate_outputs
from .private_evaluation import evaluate_private_outputs

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


def _copy_input_for_retrain(source_input: Path, retrain_input: Path) -> None:
    retrain_input.mkdir(parents=True, exist_ok=True)
    for child in source_input.iterdir():
        if child.name == "biomaker_splits":
            continue
        _replace_path_with_symlink(retrain_input / child.name, child)
    shutil.copytree(source_input / "biomaker_splits", retrain_input / "biomaker_splits")


def _write_retrain_training_table(
    workspace_dir: Path,
    private_dir: Path,
) -> tuple[int, int, int]:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    train_path = split_dir / "agent_train.csv"
    score_gt_path = private_dir / "score_val.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing training split: {train_path}")
    if not score_gt_path.is_file():
        raise FileNotFoundError(
            f"Missing hidden score-validation labels: {score_gt_path}"
        )

    train_df = pd.read_csv(train_path)
    score_gt_df = pd.read_csv(score_gt_path)
    retrain_train = pd.concat([train_df, score_gt_df], ignore_index=True, sort=False)
    train_path.with_suffix(".search.csv").write_text(
        train_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    retrain_train.to_csv(train_path, index=False)
    return int(len(train_df)), int(len(score_gt_df)), int(len(retrain_train))


def _copy_retrain_outputs_to_log(cfg: Config, retrain_workspace: Path) -> None:
    log_dir = cfg.log_dir / "final_retrain"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    working_dir = retrain_workspace / "working"
    if working_dir.is_dir():
        shutil.copytree(working_dir, log_dir / "working", symlinks=True)

    runfile = retrain_workspace / cfg.exec.agent_file_name
    if runfile.is_file():
        shutil.copy2(runfile, log_dir / cfg.exec.agent_file_name)

    submission = working_dir / "submission.csv"
    if submission.is_file():
        shutil.copy2(submission, cfg.log_dir / "final_submission.csv")
        shutil.copy2(submission, cfg.log_dir / "submission.csv")

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

    _copy_input_for_retrain(cfg.workspace_dir / "input", retrain_workspace / "input")
    for filename in ["evaluator.py", "evaluator_prediction_spec.json"]:
        src = cfg.workspace_dir / filename
        if src.exists():
            _replace_path_with_symlink(retrain_workspace / filename, src)

    before, added, after = _write_retrain_training_table(retrain_workspace, private_dir)
    manifest = {
        "mode": "final_retrain_after_node_selection",
        "source_train": str(
            cfg.workspace_dir / "input" / "biomaker_splits" / "agent_train.csv"
        ),
        "source_score_labels": str(private_dir / "score_val.csv"),
        "retrain_train": str(
            retrain_workspace / "input" / "biomaker_splits" / "agent_train.csv"
        ),
        "train_rows_before": before,
        "score_rows_added": added,
        "train_rows_after": after,
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
            exec_result.term_out.append("\n" + "\n".join(normalization_messages))
        artifact_check = validate_working_outputs(retrain_workspace, "final retrain")
        if artifact_check.messages:
            exec_result.term_out.append("\n" + "\n".join(artifact_check.messages))
        if not artifact_check.ok:
            result.error = "Final retrain artifacts failed validation."
            return result

        result.ok = True
        _copy_retrain_outputs_to_log(cfg, retrain_workspace)
        try:
            result.private_metrics = evaluate_private_outputs(
                retrain_workspace, private_dir
            )
        except Exception as exc:
            result.error = f"Private evaluation failed after final retrain: {exc}"
        return result
    except Exception as exc:
        logger.warning("Final retrain failed: %s", exc)
        result.error = str(exc)
        return result
    finally:
        (cfg.log_dir / "final_retrain").mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "final_retrain" / "completed_at.txt").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            encoding="utf-8",
        )
        _write_result(cfg, result)
