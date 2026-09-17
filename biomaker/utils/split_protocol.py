"""Agentic dataset observation and split preparation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import canonical_paths as cpaths
from .agentic_data_planner import (
    build_dataset_facts,
    parse_split_constraints,
    run_agentic_split_builder_probe,
    split_builder_cache_key,
    verify_split_package,
)
from .split_manifest import MANIFEST_NAME


@dataclass(frozen=True)
class SplitProtocol:
    dataset_report: dict[str, Any]
    split_plan: dict[str, Any]
    prompt_text: str


def _render_prompt(split_plan: dict[str, Any]) -> str:
    agent_visible = split_plan.get("agent_visible", {})
    hidden_answers = split_plan.get("hidden_answers", {})
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
    score_gt = (
        hidden_answers.get("score_val") if isinstance(hidden_answers, dict) else None
    )
    if isinstance(score_gt, dict):
        answers = score_gt.get("answer_columns") or []
        if answers:
            lines.append(
                "- Hidden score-validation answers are held outside the agent workspace."
            )
    test_gt = hidden_answers.get("test") if isinstance(hidden_answers, dict) else None
    if isinstance(test_gt, dict):
        answers = test_gt.get("answer_columns") or []
        if answers:
            lines.append(
                "- Private test answers are held outside the agent workspace for user-only evaluation."
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
    return cpaths.looks_like_path(value)


def _input_relative(candidate: Path, input_dir: Path) -> str | None:
    return cpaths.input_relative(candidate, input_dir)


def _generated_public_relative(
    candidate: Path, generated_output_dir: Path | None
) -> str | None:
    return cpaths.generated_public_relative(candidate, generated_output_dir)


def _workspace_relative_input_path(
    value: Any,
    input_dir: Path,
    *,
    base_dir: Path | None = None,
    generated_output_dir: Path | None = None,
) -> str | None:
    return cpaths.workspace_relative_input_path(
        value,
        input_dir,
        base_dir=base_dir,
        generated_output_dir=generated_output_dir,
    )


def _normalize_split_paths_in_dir(
    csv_dir: Path,
    input_dir: Path,
    *,
    base_dir: Path | None = None,
    generated_output_dir: Path | None = None,
) -> None:
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
                replacement = _workspace_relative_input_path(
                    value,
                    input_dir,
                    base_dir=base_dir,
                    generated_output_dir=generated_output_dir,
                )
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


def _evaluator_visible_path(value: Any, input_dir: Path, base_dir: Path) -> str | None:
    return cpaths.evaluator_visible_path(value, input_dir, base_dir)


def _make_private_paths_evaluator_visible(csv_dir: Path, input_dir: Path) -> None:
    if not csv_dir.is_dir():
        return
    answer_columns_by_file: dict[str, set[str]] = {}
    manifest_path = csv_dir / "private_gt.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            for split_name, entry in manifest.items():
                if not isinstance(entry, dict):
                    continue
                file_name = "score_val.csv" if split_name == "score_val" else "test.csv"
                answer_columns_by_file[file_name] = set(
                    map(str, entry.get("answer_columns") or [])
                )
    for csv_path in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        answer_columns = answer_columns_by_file.get(csv_path.name)
        changed = False
        for column in df.columns:
            if answer_columns is not None and str(column) not in answer_columns:
                continue
            if df[column].dtype != object:
                continue
            values = []
            column_changed = False
            for value in df[column].tolist():
                replacement = _evaluator_visible_path(value, input_dir, csv_dir)
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


def _mirror_tree_excluding(
    src: Path, dst: Path, root_rel: Path, hidden_rels: set[Path]
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        child_rel = root_rel / child.name
        if child_rel in hidden_rels:
            continue
        if any(
            rel.parts[: len(child_rel.parts)] == child_rel.parts for rel in hidden_rels
        ):
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


def _copy_generated_public_assets(package_dir: Path, input_dir: Path) -> None:
    public_dir = package_dir / "public"
    if not public_dir.is_dir():
        return
    target_dir = input_dir / "public"
    marker = target_dir / ".biomaker_generated_public"
    if target_dir.exists():
        if not marker.is_file():
            raise RuntimeError(
                "split builder generated public/ assets, but input/public already exists"
            )
        _remove_path(target_dir)
    shutil.copytree(public_dir, target_dir)
    (target_dir / ".biomaker_generated_public").write_text(
        "Generated by BioMaker split builder.\n", encoding="utf-8"
    )


_AGENT_VISIBLE_SPLIT_FILES = (
    "agent_train.csv",
    "score_val.csv",
    "agent_test.csv",
)

_SAFE_TOP_LEVEL_INFO_FILES = {
    "data_description.txt",
    "description.txt",
    "readme",
    "readme.md",
    "readme.txt",
    "license",
    "license.md",
    "license.txt",
}


def _existing_input_reference(value: Any, input_dir: Path) -> tuple[str, Path] | None:
    workspace_rel = _workspace_relative_input_path(value, input_dir)
    if workspace_rel is None:
        return None
    rel = Path(workspace_rel)
    if not rel.parts or rel.parts[0] != "input":
        return None
    source = input_dir / Path(*rel.parts[1:])
    if not source.exists() and not source.is_symlink():
        return None
    return workspace_rel, source


def _copy_or_link_visible_path(
    source: Path, destination: Path, input_dir: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    resolved_source = source.resolve()
    input_root = input_dir.resolve()
    source_inside_input = False
    try:
        resolved_source.relative_to(input_root)
        source_inside_input = True
    except ValueError:
        source_inside_input = False

    if source.is_dir():
        if source_inside_input:
            shutil.copytree(source, destination, symlinks=True)
        else:
            destination.symlink_to(resolved_source)
        return

    if source_inside_input:
        shutil.copy2(resolved_source, destination)
    else:
        destination.symlink_to(resolved_source)


def _can_expose_directory(source_dir: Path, referenced_names: set[str]) -> bool:
    if not source_dir.is_dir() or not referenced_names:
        return False
    child_names: set[str] = set()
    try:
        children = list(source_dir.iterdir())
    except OSError:
        return False
    for child in children:
        if child.is_dir():
            return False
        if child.is_file() or child.is_symlink():
            child_names.add(child.name)
            continue
        return False
    return child_names == referenced_names


def _copy_or_link_visible_references(
    visible_paths: dict[str, Path], temp_input: Path, input_dir: Path
) -> int:
    grouped: dict[Path, dict[str, Path]] = {}
    for workspace_rel, source in visible_paths.items():
        rel = Path(workspace_rel)
        grouped.setdefault(Path(*rel.parts[1:-1]), {})[rel.name] = source

    materialized = 0
    for rel_parent, entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        source_parents = {source.parent.resolve() for source in entries.values()}
        source_parent = next(iter(entries.values())).parent
        destination_parent = temp_input / rel_parent
        if len(source_parents) == 1 and _can_expose_directory(
            source_parent, set(entries)
        ):
            _copy_or_link_visible_path(source_parent, destination_parent, input_dir)
            materialized += len(entries)
            continue

        for filename, source in sorted(entries.items()):
            _copy_or_link_visible_path(source, destination_parent / filename, input_dir)
            materialized += 1
    return materialized


def sanitize_agent_visible_input(
    input_dir: Path | str, split_dir: Path | str
) -> dict[str, Any]:
    """Replace workspace input with only files the generated code is allowed to see.

    The split builder needs the original dataset view, but code nodes should only see
    the public split tables and files those tables explicitly reference. This prevents
    root dataset CSVs, metadata tables, and hidden score/test masks from remaining
    reachable through the initial dataset symlink.
    """
    input_dir = Path(input_dir).resolve()
    split_dir = Path(split_dir).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory does not exist: {split_dir}")

    visible_paths: dict[str, Path] = {}
    for filename in _AGENT_VISIBLE_SPLIT_FILES:
        csv_path = split_dir / filename
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for column in df.columns:
            if df[column].dtype != object:
                continue
            for value in df[column].dropna().tolist():
                reference = _existing_input_reference(value, input_dir)
                if reference is None:
                    continue
                workspace_rel, source = reference
                visible_paths.setdefault(workspace_rel, source)

    safe_info_files = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_file():
            continue
        if child.name.lower() in _SAFE_TOP_LEVEL_INFO_FILES:
            safe_info_files.append(child)

    temp_input = input_dir.parent / f".{input_dir.name}.agent_visible"
    _remove_path(temp_input)
    temp_input.mkdir(parents=True, exist_ok=True)

    shutil.copytree(split_dir, temp_input / "biomaker_splits", symlinks=True)
    for source in safe_info_files:
        _copy_or_link_visible_path(source, temp_input / source.name, input_dir)
    materialized_files = _copy_or_link_visible_references(
        visible_paths, temp_input, input_dir
    )

    _remove_path(input_dir)
    temp_input.rename(input_dir)
    return {
        "split_files": [
            name
            for name in _AGENT_VISIBLE_SPLIT_FILES
            if (input_dir / "biomaker_splits" / name).is_file()
        ],
        "referenced_files": len(visible_paths),
        "materialized_files": materialized_files,
        "info_files": [path.name for path in safe_info_files],
    }


def _copy_split_package_from_cache(
    cache_entry: Path, split_dir: Path, private_dir: Path, input_dir: Path,
    *, task_text: str = "",
) -> dict[str, Any] | None:
    cached_split = cache_entry / "biomaker_splits"
    cached_private = cache_entry / "private_gt"
    cached_verifier = cache_entry / "verifier.json"
    if (
        not cached_split.is_dir()
        or not cached_private.is_dir()
        or not cached_verifier.is_file()
    ):
        return None
    # A previously saved "ok" flag is not evidence that files still satisfy the
    # current task/source contract. Revalidate before admitting cached splits.
    verifier = verify_split_package(cache_entry, source_input_dir=input_dir, task_text=task_text)
    if not verifier.get("ok"):
        return None
    _copytree_replace(cached_split, split_dir)
    _copytree_replace(cached_private, private_dir)
    cached_manifest = cache_entry / MANIFEST_NAME
    if cached_manifest.is_file():
        shutil.copy2(cached_manifest, split_dir / MANIFEST_NAME)
    _copy_generated_public_assets(cache_entry, input_dir)
    return verifier


def _store_split_package_cache(
    cache_entry: Path,
    package_dir: Path,
    split_dir: Path,
    private_dir: Path,
    verifier: dict[str, Any],
) -> None:
    if (cache_entry / "manifest.json").is_file():
        return
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_entry.parent / f".tmp-{cache_entry.name}"
    _remove_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(split_dir, tmp_dir / "biomaker_splits")
    shutil.copytree(private_dir, tmp_dir / "private_gt")
    public_dir = package_dir / "public"
    if public_dir.is_dir():
        shutil.copytree(public_dir, tmp_dir / "public")
    manifest_path = package_dir / MANIFEST_NAME
    if manifest_path.is_file():
        shutil.copy2(manifest_path, tmp_dir / MANIFEST_NAME)
    (tmp_dir / "verifier.json").write_text(
        json.dumps(verifier, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_dir / "manifest.json").write_text(
        json.dumps(
            {"cache": "split_builder", "verifier_ok": verifier.get("ok")}, indent=2
        ),
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
    private_dir = (
        Path(private_dir).resolve()
        if private_dir is not None
        else input_dir.parent / "private_gt"
    )
    split_dir = input_dir / "biomaker_splits"
    _remove_path(split_dir)
    _remove_path(private_dir)

    if isinstance(task_desc, str):
        task_text = task_desc
    else:
        task_text = json.dumps(
            task_desc or {}, indent=2, ensure_ascii=False, default=str
        )
    parse_split_constraints(task_text)

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
        result_verifier = _copy_split_package_from_cache(
            cache_entry, split_dir, private_dir, input_dir, task_text=task_text
        )
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
        _normalize_split_paths_in_dir(
            result.split_dir,
            input_dir,
            base_dir=result.output_dir,
            generated_output_dir=result.output_dir,
        )
        _normalize_split_paths_in_dir(
            result.private_gt_dir,
            input_dir,
            base_dir=result.output_dir,
            generated_output_dir=result.output_dir,
        )
        _copytree_replace(result.split_dir, split_dir)
        _copytree_replace(result.private_gt_dir, private_dir)
        manifest_path = result.output_dir / MANIFEST_NAME
        if manifest_path.is_file():
            shutil.copy2(manifest_path, split_dir / MANIFEST_NAME)
        _copy_generated_public_assets(result.output_dir, input_dir)
        result_verifier = result.verifier
        if cache_entry is not None:
            _store_split_package_cache(
                cache_entry, result.output_dir, split_dir, private_dir, result_verifier
            )

    _normalize_split_paths_in_dir(split_dir, input_dir)
    _make_private_paths_evaluator_visible(private_dir, input_dir)
    _hide_private_file_answers(input_dir, private_dir)
    agent_visible_input = sanitize_agent_visible_input(input_dir, split_dir)

    hidden_answers_plan = (
        result_verifier.get("hidden_answers")
        or result_verifier.get("private_gt", {})
        or {}
    )
    if isinstance(hidden_answers_plan, dict):
        if isinstance(hidden_answers_plan.get("score_val"), dict):
            hidden_answers_plan["score_val"] = {
                "file": "score_val.csv",
                **hidden_answers_plan["score_val"],
            }
        if isinstance(hidden_answers_plan.get("test"), dict):
            hidden_answers_plan["test"] = {
                "file": "test.csv",
                **hidden_answers_plan["test"],
            }

    split_plan = {
        "strategy": "agentic_builder",
        "agent_visible": {
            "train": "biomaker_splits/agent_train.csv",
            "score_val": "biomaker_splits/score_val.csv",
            "test": "biomaker_splits/agent_test.csv",
        },
        "hidden_answers": hidden_answers_plan,
        "metric_split": "score_val",
        "row_counts": result_verifier.get("row_counts", {}),
        "warnings": result_verifier.get("warnings", []),
        "cache_reused": cache_reused,
        "agent_visible_input": agent_visible_input,
    }
    split_manifest = result_verifier.get("split_manifest")
    if isinstance(split_manifest, dict):
        split_plan["split_manifest"] = {
            "manifest_version": split_manifest.get("manifest_version"),
            "sample_unit": split_manifest.get("sample_unit"),
            "key_columns": split_manifest.get("key_columns", []),
            "task_heads": split_manifest.get("task_heads", []),
            "metric_contract": split_manifest.get("metric_contract", {}),
        }
        if split_manifest.get("key_columns") and not split_plan.get("key_columns"):
            split_plan["key_columns"] = split_manifest.get("key_columns")
        metric_contract = split_manifest.get("metric_contract")
        if isinstance(metric_contract, dict):
            if metric_contract.get("primary_metric"):
                split_plan["primary_metric"] = metric_contract["primary_metric"]
            if "maximize" in metric_contract:
                split_plan["maximize"] = bool(metric_contract["maximize"])
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
    return SplitProtocol(
        dataset_report=dataset_report, split_plan=split_plan, prompt_text=prompt_text
    )
