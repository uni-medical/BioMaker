"""Agentic dataset observation and split preparation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agentic_data_planner import (
    build_dataset_facts,
    run_agentic_split_builder_probe,
    split_builder_cache_key,
)


@dataclass(frozen=True)
class SplitProtocol:
    dataset_report: dict[str, Any]
    split_plan: dict[str, Any]
    prompt_text: str


def _render_prompt(split_plan: dict[str, Any]) -> str:
    agent_visible = split_plan.get("agent_visible", {})
    private_gt = split_plan.get("private_gt", {})
    lines = ["Use the BioMaker data split below as the source of truth for this run."]
    train_path = agent_visible.get("train")
    score_path = agent_visible.get("score_val")
    test_path = agent_visible.get("test")
    if train_path:
        lines.append(
            f"- Use `./input/{train_path}` as the only agent-visible training table."
        )
    if score_path:
        lines.append(
            f"- Generate validation predictions for every row in `./input/{score_path}`. "
            "This score split is used by BioMaker to score and select nodes. Its answers are hidden from generated code and must not be used for training, calibration, model selection, debugging, or metric reporting."
        )
    if test_path:
        lines.append(
            f"- Use `./input/{test_path}` only to generate `./working/submission.csv`; do not use it for training, model selection, debugging, or metric reporting."
        )
    score_gt = private_gt.get("score_val") if isinstance(private_gt, dict) else None
    if isinstance(score_gt, dict):
        answers = score_gt.get("answer_columns") or []
        if answers:
            lines.append(
                "- Hidden score-validation answer columns are held outside the agent workspace: "
                + ", ".join(f"`{column}`" for column in answers)
                + "."
            )
    test_gt = private_gt.get("test") if isinstance(private_gt, dict) else None
    if isinstance(test_gt, dict):
        answers = test_gt.get("answer_columns") or []
        if answers:
            lines.append(
                "- Private test answers are held outside the agent workspace for user-only evaluation: "
                + ", ".join(f"`{column}`" for column in answers)
                + "."
            )
    for warning in split_plan.get("warnings", []) or []:
        lines.append(f"- Warning: {warning}")
    return "\n".join(lines)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _copytree_replace(src: Path, dst: Path) -> None:
    _remove_path(dst)
    shutil.copytree(src, dst)


def _looks_like_path(value: str) -> bool:
    suffixes = {
        ".csv",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".nii",
        ".nii.gz",
        ".dcm",
        ".xml",
        ".txt",
    }
    lowered = value.lower()
    return "/" in value or "\\" in value or any(lowered.endswith(suffix) for suffix in suffixes)


def _workspace_relative_input_path(value: Any, input_dir: Path) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not _looks_like_path(text):
        return None

    raw = Path(text)
    if not raw.is_absolute():
        parts = raw.parts
        if parts and parts[0] == "input":
            return str(raw)
        if (input_dir / raw).exists() or raw.parts:
            return str(Path("input") / raw)
        return None

    try:
        return str(Path("input") / raw.relative_to(input_dir))
    except ValueError:
        pass

    try:
        resolved = raw.resolve()
    except Exception:
        resolved = raw
    for child in input_dir.iterdir():
        if not child.exists():
            continue
        try:
            rel = resolved.relative_to(child.resolve())
        except Exception:
            continue
        return str(Path("input") / child.name / rel)
    return None


def _normalize_split_paths_in_dir(csv_dir: Path, input_dir: Path) -> None:
    if not csv_dir.is_dir():
        return
    for csv_path in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        changed = False
        for column in df.columns:
            if df[column].dtype != object:
                continue
            normalized = []
            column_changed = False
            for value in df[column].tolist():
                replacement = _workspace_relative_input_path(value, input_dir)
                if replacement is not None:
                    normalized.append(replacement)
                    column_changed = column_changed or replacement != value
                else:
                    normalized.append(value)
            if column_changed:
                df[column] = normalized
                changed = True
        if changed:
            df.to_csv(csv_path, index=False)


def _evaluator_visible_path(value: Any, input_dir: Path) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not _looks_like_path(text):
        return None
    raw = Path(text)
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [input_dir / raw]
        if raw.parts and raw.parts[0] == "input":
            candidates.append(input_dir.parent / raw)
            if len(raw.parts) > 1:
                candidates.append(input_dir / Path(*raw.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.resolve())
            except Exception:
                return str(candidate)
    return None


def _make_private_paths_evaluator_visible(csv_dir: Path, input_dir: Path) -> None:
    if not csv_dir.is_dir():
        return
    for csv_path in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        changed = False
        for column in df.columns:
            if df[column].dtype != object:
                continue
            values = []
            column_changed = False
            for value in df[column].tolist():
                replacement = _evaluator_visible_path(value, input_dir)
                if replacement is not None:
                    values.append(replacement)
                    column_changed = column_changed or replacement != value
                else:
                    values.append(value)
            if column_changed:
                df[column] = values
                changed = True
        if changed:
            df.to_csv(csv_path, index=False)


def _collect_private_file_answers(input_dir: Path, private_dir: Path) -> set[Path]:
    manifest_path = private_dir / "private_gt.json"
    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(manifest, dict):
        return set()

    hidden_paths: set[Path] = set()
    for split_name, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        answer_columns = [str(column) for column in entry.get("answer_columns", [])]
        if not answer_columns:
            continue
        csv_name = "score_val.csv" if split_name == "score_val" else "test.csv"
        csv_path = private_dir / csv_name
        if not csv_path.is_file():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, usecols=lambda column: column in answer_columns)
        except Exception:
            continue
        for column in answer_columns:
            if column not in df.columns:
                continue
            for raw_value in df[column].dropna().astype(str):
                value = raw_value.strip()
                if not value or not _looks_like_path(value):
                    continue
                rel = Path(value)
                if rel.is_absolute():
                    try:
                        rel = rel.resolve().relative_to(input_dir.resolve())
                    except Exception:
                        mapped_rel = None
                        try:
                            resolved = rel.resolve()
                        except Exception:
                            resolved = rel
                        for child in input_dir.iterdir():
                            if not child.exists():
                                continue
                            try:
                                subrel = resolved.relative_to(child.resolve())
                            except Exception:
                                continue
                            mapped_rel = Path(child.name) / subrel
                            break
                        if mapped_rel is None:
                            continue
                        rel = mapped_rel
                if rel.parts and rel.parts[0] == "input":
                    rel = Path(*rel.parts[1:])
                candidate = input_dir / rel
                if candidate.exists() or candidate.is_symlink():
                    hidden_paths.add(rel)
    return hidden_paths


def _first_symlink_ancestor(input_dir: Path, rel: Path) -> Path | None:
    current = input_dir
    ancestor_parts = []
    for part in rel.parts:
        current = current / part
        ancestor_parts.append(part)
        if current.is_symlink():
            return Path(*ancestor_parts)
    return None


def _mirror_tree_excluding(src: Path, dst: Path, root_rel: Path, hidden_rels: set[Path]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        child_rel = root_rel / child.name
        if child_rel in hidden_rels:
            continue
        if any(rel.parts[: len(child_rel.parts)] == child_rel.parts for rel in hidden_rels):
            child_dst = dst / child.name
            if child.is_dir():
                _mirror_tree_excluding(child, child_dst, child_rel, hidden_rels)
            else:
                child_dst.symlink_to(child)
        else:
            (dst / child.name).symlink_to(child)


def _hide_private_file_answers(input_dir: Path, private_dir: Path) -> None:
    hidden_rels = _collect_private_file_answers(input_dir, private_dir)
    if not hidden_rels:
        return

    by_symlink_root: dict[Path, set[Path]] = {}
    direct_rels: set[Path] = set()
    for rel in hidden_rels:
        symlink_root = _first_symlink_ancestor(input_dir, rel)
        if symlink_root is None:
            direct_rels.add(rel)
        else:
            by_symlink_root.setdefault(symlink_root, set()).add(rel)

    for rel in direct_rels:
        target = input_dir / rel
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    for root_rel, root_hidden_rels in by_symlink_root.items():
        root_path = input_dir / root_rel
        if not root_path.is_symlink():
            continue
        source_root = root_path.resolve()
        if source_root.is_file():
            root_path.unlink()
            continue
        tmp_path = input_dir / f".{root_rel.name}.agent_visible"
        _remove_path(tmp_path)
        _mirror_tree_excluding(source_root, tmp_path, root_rel, root_hidden_rels)
        root_path.unlink()
        tmp_path.rename(root_path)


def _copy_split_package_from_cache(cache_entry: Path, split_dir: Path, private_dir: Path) -> dict[str, Any] | None:
    cached_split = cache_entry / "biomaker_splits"
    cached_private = cache_entry / "private_gt"
    cached_verifier = cache_entry / "verifier.json"
    if not cached_split.is_dir() or not cached_private.is_dir() or not cached_verifier.is_file():
        return None
    _copytree_replace(cached_split, split_dir)
    _copytree_replace(cached_private, private_dir)
    return json.loads(cached_verifier.read_text(encoding="utf-8"))


def _store_split_package_cache(cache_entry: Path, split_dir: Path, private_dir: Path, verifier: dict[str, Any]) -> None:
    if (cache_entry / "manifest.json").is_file():
        return
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_entry.parent / f".tmp-{cache_entry.name}"
    _remove_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(split_dir, tmp_dir / "biomaker_splits")
    shutil.copytree(private_dir, tmp_dir / "private_gt")
    (tmp_dir / "verifier.json").write_text(
        json.dumps(verifier, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_dir / "manifest.json").write_text(
        json.dumps({"cache": "split_builder", "verifier_ok": verifier.get("ok")}, indent=2),
        encoding="utf-8",
    )
    try:
        tmp_dir.rename(cache_entry)
    except FileExistsError:
        shutil.rmtree(tmp_dir)


def prepare_split_protocol(
    input_dir: Path | str,
    private_dir: Path | str | None = None,
    *,
    task_desc: Any | None = None,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
    attempts: int = 3,
    cache_root: Path | str | None = None,
) -> SplitProtocol:
    """Prepare BioMaker public splits and private ground truth using an LLM-built splitter."""
    input_dir = Path(input_dir).resolve()
    private_dir = Path(private_dir).resolve() if private_dir is not None else input_dir.parent / "private_gt"
    split_dir = input_dir / "biomaker_splits"
    _remove_path(split_dir)
    _remove_path(private_dir)

    if isinstance(task_desc, str):
        task_text = task_desc
    else:
        task_text = json.dumps(task_desc or {}, indent=2, ensure_ascii=False, default=str)

    cache_entry = None
    result_verifier: dict[str, Any] | None = None
    cache_reused = False
    if cache_root is not None:
        cache_key, _facts = split_builder_cache_key(
            input_dir=input_dir,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )
        cache_entry = Path(cache_root) / cache_key
        result_verifier = _copy_split_package_from_cache(cache_entry, split_dir, private_dir)
        cache_reused = result_verifier is not None

    run_root = private_dir.parent
    build_dir = run_root / "_split_builder"
    if result_verifier is None:
        _remove_path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        task_file = build_dir / "task.md"
        task_file.write_text(task_text, encoding="utf-8")

        result = run_agentic_split_builder_probe(
            input_dir,
            task_file,
            build_dir / "package",
            model=model,
            temperature=temperature,
            max_attempts=attempts,
            debug_artifacts=True,
        )
        _copytree_replace(result.split_dir, split_dir)
        _copytree_replace(result.private_gt_dir, private_dir)
        result_verifier = result.verifier
        if cache_entry is not None:
            _store_split_package_cache(cache_entry, split_dir, private_dir, result_verifier)

    _normalize_split_paths_in_dir(split_dir, input_dir)
    _make_private_paths_evaluator_visible(private_dir, input_dir)
    _hide_private_file_answers(input_dir, private_dir)

    private_gt_plan = result_verifier.get("private_gt", {}) or {}
    if isinstance(private_gt_plan, dict):
        if isinstance(private_gt_plan.get("score_val"), dict):
            private_gt_plan["score_val"] = {
                "file": "score_val.csv",
                **private_gt_plan["score_val"],
            }
        if isinstance(private_gt_plan.get("test"), dict):
            private_gt_plan["test"] = {
                "file": "test.csv",
                **private_gt_plan["test"],
            }

    split_plan = {
        "strategy": "agentic_builder",
        "agent_visible": {
            "train": "biomaker_splits/agent_train.csv",
            "score_val": "biomaker_splits/score_val.csv",
            "test": "biomaker_splits/agent_test.csv",
        },
        "private_gt": private_gt_plan,
        "metric_split": "score_val",
        "row_counts": result_verifier.get("row_counts", {}),
        "warnings": result_verifier.get("warnings", []),
        "cache_reused": cache_reused,
    }
    if not (split_dir / "agent_test.csv").is_file():
        split_plan["agent_visible"].pop("test", None)
    (split_dir / "split_plan.json").write_text(
        json.dumps(split_plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dataset_report = build_dataset_facts(input_dir)
    (split_dir / "dataset_report.json").write_text(
        json.dumps(dataset_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prompt_text = _render_prompt(split_plan)
    (split_dir / "split_protocol.txt").write_text(prompt_text, encoding="utf-8")
    return SplitProtocol(dataset_report=dataset_report, split_plan=split_plan, prompt_text=prompt_text)
