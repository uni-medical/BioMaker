"""Agentic dataset observation and split preparation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
from .split_contract import resolve_public_splits
from .split_manifest import (
    MANIFEST_NAME,
    load_split_manifest,
    split_file_reference_roles,
    file_columns_for_frame,
)


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
            f"- Use `./input/{test_path}` only to generate the test predictions declared by the frozen output contract; do not use it for training, model selection, debugging, or metric reporting."
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


def _declared_file_roles(csv_dir: Path, input_dir: Path) -> dict[str, Any]:
    """Load role declarations from either a builder package or a prepared run."""
    manifest: dict[str, Any] = {}
    for root in (csv_dir, csv_dir.parent, input_dir / "biomaker_splits"):
        manifest = load_split_manifest(root)
        if manifest:
            break
    hidden = manifest.get("hidden_answers", {})
    for path in (
        csv_dir / "private_gt.json",
        csv_dir.parent / "private_gt" / "private_gt.json",
    ):
        if path.is_file():
            hidden = json.loads(path.read_text(encoding="utf-8"))
            break
    if not isinstance(hidden, dict):
        raise ValueError("Private answer index must be an object.")
    errors: list[str] = []
    roles = split_file_reference_roles(manifest, hidden, errors)
    if errors:
        raise ValueError("Invalid split file roles: " + "; ".join(errors))
    return roles


def _role_for_csv(path: Path) -> str:
    return {
        "agent_train.csv": "train",
        "score_val.csv": "score_val",
        "agent_test.csv": "test",
        "test.csv": "test",
    }.get(path.name, "train")


def _normalize_split_paths_in_dir(
    csv_dir: Path,
    input_dir: Path,
    *,
    base_dir: Path | None = None,
    generated_output_dir: Path | None = None,
) -> None:
    if not csv_dir.is_dir():
        return
    roles = _declared_file_roles(csv_dir, input_dir)
    private = (csv_dir / "private_gt.json").is_file()
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        required, legacy = file_columns_for_frame(
            roles, df.columns, role=_role_for_csv(csv_path), private=private
        )
        changed = False
        for column in (required | legacy) & set(df.columns):
            normalized = []
            column_changed = False
            for value in df[column].tolist():
                replacement = _workspace_relative_input_path(
                    value,
                    input_dir,
                    base_dir=base_dir,
                    generated_output_dir=generated_output_dir,
                )
                if replacement is None and column in required and value:
                    raw = Path(value)
                    candidates = [raw] if raw.is_absolute() else [input_dir / raw]
                    if not raw.is_absolute():
                        candidates.extend(
                            root / raw
                            for root in (base_dir, generated_output_dir)
                            if root is not None
                        )
                    for candidate in candidates:
                        if candidate.is_file():
                            replacement = _input_relative(
                                candidate, input_dir
                            ) or _generated_public_relative(
                                candidate, generated_output_dir
                            )
                            if replacement is not None:
                                break
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retain_private_answer(source: Path, private_dir: Path) -> Path:
    """Copy an answer before the workspace input view is replaced or pruned."""
    source = source.resolve()
    if not source.is_file():
        raise ValueError(
            f"Private answer must reference an existing regular file: {source}"
        )
    digest = _sha256_file(source)
    destination = private_dir / "answer_files" / digest / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or _sha256_file(destination) != digest:
            raise ValueError("Retained private answer content changed.")
        return destination.resolve()
    fd, temporary = tempfile.mkstemp(prefix=".answer-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        if _sha256_file(Path(temporary)) != digest:
            raise ValueError("Private answer changed while being retained.")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination.resolve()


def _make_private_paths_evaluator_visible(csv_dir: Path, input_dir: Path) -> None:
    if not csv_dir.is_dir():
        return
    roles = _declared_file_roles(csv_dir, input_dir)
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        role = _role_for_csv(csv_path)
        answers = roles["answer_columns"].get(role, set())
        declared = roles["answer_files"].get(role)
        file_columns = answers if declared is None else declared
        changed = False
        for column in file_columns & set(df.columns):
            values = []
            for value in df[column].tolist():
                if declared is None and not _looks_like_path(value):
                    values.append(value)
                    continue
                replacement = _evaluator_visible_path(value, input_dir, csv_dir)
                # Explicit declarations are authoritative even for extensionless
                # files that the legacy path heuristic would not recognize.
                if replacement is None and declared is not None:
                    raw = Path(value)
                    candidates = (
                        [raw]
                        if raw.is_absolute()
                        else [input_dir / raw, csv_dir / raw, csv_dir.parent / raw]
                    )
                    if raw.parts and raw.parts[0] == "input":
                        candidates.append(input_dir / Path(*raw.parts[1:]))
                    replacement = next(
                        (str(path.resolve()) for path in candidates if path.is_file()),
                        None,
                    )
                if replacement is None:
                    if declared is not None:
                        raise ValueError(
                            f"Missing declared private answer file: {csv_path.name}:{column}"
                        )
                    values.append(value)
                    continue
                retained = _retain_private_answer(Path(replacement), csv_dir)
                values.append(str(retained))
                changed = changed or str(retained) != value
            df[column] = values
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

    roles = _declared_file_roles(private_dir, input_dir)
    hidden_paths: set[Path] = set()
    for split_name, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        declared = roles["answer_files"].get(split_name)
        answer_columns = list(
            roles["answer_columns"].get(split_name, set())
            if declared is None
            else declared
        )
        if not answer_columns:
            continue
        csv_name = "score_val.csv" if split_name == "score_val" else "test.csv"
        csv_path = private_dir / csv_name
        if not csv_path.is_file():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
                usecols=lambda column: column in answer_columns,
            )
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


def _existing_input_reference(
    value: Any, input_dir: Path, *, declared: bool = False
) -> tuple[str, Path] | None:
    workspace_rel = _workspace_relative_input_path(value, input_dir)
    if workspace_rel is None and declared and isinstance(value, str) and value:
        raw = Path(value)
        source = raw if raw.is_absolute() else input_dir / raw
        if source.is_file():
            workspace_rel = _input_relative(source, input_dir)
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

    roles = _declared_file_roles(split_dir, input_dir)
    visible_paths: dict[str, Path] = {}
    for filename in _AGENT_VISIBLE_SPLIT_FILES:
        csv_path = split_dir / filename
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception:
            continue
        required, legacy = file_columns_for_frame(
            roles, df.columns, role=_role_for_csv(csv_path)
        )
        for column in (required | legacy) & set(df.columns):
            for value in df[column].dropna().tolist():
                reference = _existing_input_reference(
                    value, input_dir, declared=column in required
                )
                if reference is None:
                    if column in required:
                        raise ValueError(
                            f"Missing declared public input file: {filename}:{column}"
                        )
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
    cache_entry: Path,
    split_dir: Path,
    private_dir: Path,
    input_dir: Path,
    *,
    task_text: str = "",
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
    # Current validation may normalize compatibility manifests. Run it on a
    # private copy so inspecting a cache never changes the cached evidence.
    with tempfile.TemporaryDirectory(prefix="biomaker-split-cache-") as temporary:
        staged = Path(temporary) / "package"
        try:
            shutil.copytree(cache_entry, staged)
            verifier = verify_split_package(
                staged, source_input_dir=input_dir, task_text=task_text
            )
        except (OSError, ValueError):
            return None
        if not verifier.get("ok"):
            return None
        _copytree_replace(staged / "biomaker_splits", split_dir)
        _copytree_replace(staged / "private_gt", private_dir)
        manifest_path = staged / MANIFEST_NAME
        if manifest_path.is_file():
            shutil.copy2(manifest_path, split_dir / MANIFEST_NAME)
        _copy_generated_public_assets(staged, input_dir)
        return verifier


def _require_visible_training_answers(split_dir: Path, private_dir: Path) -> None:
    """Check materialized training supervision before and after input pruning."""
    from .agentic_data_planner import validate_training_answers

    train_df = pd.read_csv(
        split_dir / "agent_train.csv", dtype=str, keep_default_na=False
    )
    hidden_answers = json.loads(
        (private_dir / "private_gt.json").read_text(encoding="utf-8")
    )
    if not isinstance(hidden_answers, dict):
        raise RuntimeError("Invalid hidden-answer index for training supervision.")
    errors: list[str] = []
    validate_training_answers(
        train_df, hidden_answers, errors, split_manifest=load_split_manifest(split_dir)
    )
    if errors:
        raise RuntimeError("Invalid supervised training split: " + "; ".join(errors))


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
    package_dir: Path | str | None = None,
) -> SplitProtocol:
    """Prepare public splits and hidden answers from generated or verified inputs."""
    input_dir = Path(input_dir).resolve()
    private_dir = (
        Path(private_dir).resolve()
        if private_dir is not None
        else input_dir.parent / "private_gt"
    )
    if (
        private_dir == input_dir
        or input_dir in private_dir.parents
        or private_dir in input_dir.parents
    ):
        raise ValueError(
            "Private answers must be outside and separate from the agent input directory."
        )
    split_dir = input_dir / "biomaker_splits"
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
    run_root = private_dir.parent
    build_dir = run_root / "_split_builder"
    if package_dir is not None:
        from .verified_split_package import verify_package_identity

        package = Path(package_dir).resolve()
        for output in (input_dir, private_dir, build_dir):
            if (
                package == output
                or output in package.parents
                or package in output.parents
            ):
                raise ValueError(
                    "Verified split package must be outside run input/output directories."
                )
        # Verify before touching existing run outputs. Re-run the current verifier
        # on an isolated copy: its compatibility normalization must not mutate
        # the supplied sealed package.
        verify_package_identity(package, input_dir, task_text=task_text)
        _remove_path(build_dir)
        staged = build_dir / "package"
        shutil.copytree(package, staged)
        result_verifier = verify_split_package(
            staged, source_input_dir=input_dir, task_text=task_text
        )
        if not result_verifier.get("ok"):
            raise ValueError(
                "Verified split package failed current validation: "
                + "; ".join(result_verifier.get("errors", []))
            )
        _copytree_replace(staged / "biomaker_splits", split_dir)
        _copytree_replace(staged / "private_gt", private_dir)
        shutil.copy2(staged / MANIFEST_NAME, split_dir / MANIFEST_NAME)
        _copy_generated_public_assets(staged, input_dir)
    else:
        _remove_path(split_dir)
        _remove_path(private_dir)
    if cache_root is not None and package_dir is None:
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
        result_verifier = verify_split_package(
            result.output_dir, source_input_dir=input_dir, task_text=task_text
        )
        if not result_verifier.get("ok"):
            raise RuntimeError(
                "Normalized split package failed verification: "
                + "; ".join(result_verifier.get("errors", []))
            )
        _copytree_replace(result.split_dir, split_dir)
        _copytree_replace(result.private_gt_dir, private_dir)
        manifest_path = result.output_dir / MANIFEST_NAME
        if manifest_path.is_file():
            shutil.copy2(manifest_path, split_dir / MANIFEST_NAME)
        _copy_generated_public_assets(result.output_dir, input_dir)
        if cache_entry is not None:
            _store_split_package_cache(
                cache_entry, result.output_dir, split_dir, private_dir, result_verifier
            )

    _require_visible_training_answers(split_dir, private_dir)
    _normalize_split_paths_in_dir(split_dir, input_dir)
    _make_private_paths_evaluator_visible(private_dir, input_dir)
    _hide_private_file_answers(input_dir, private_dir)
    agent_visible_input = sanitize_agent_visible_input(input_dir, split_dir)
    _require_visible_training_answers(split_dir, private_dir)

    prepared_manifest = load_split_manifest(split_dir)
    public_splits = resolve_public_splits(prepared_manifest)
    split_plan = {
        "strategy": (
            "verified_package" if package_dir is not None else "agentic_builder"
        ),
        "agent_visible": public_splits,
        "hidden_answers": prepared_manifest.get("hidden_answers", {}),
        "metric_split": "score_val",
        "row_counts": result_verifier.get("row_counts", {}),
        "warnings": result_verifier.get("warnings", []),
        "cache_reused": cache_reused,
        "agent_visible_input": agent_visible_input,
    }
    split_manifest = prepared_manifest
    if isinstance(split_manifest, dict):
        split_plan["split_manifest"] = {
            "manifest_version": split_manifest.get("manifest_version"),
            "splits": split_manifest.get("splits", {}),
            "sample_unit": split_manifest.get("sample_unit"),
            "key_columns": split_manifest.get("key_columns", []),
            "task_heads": split_manifest.get("task_heads", []),
            "metric_contract": split_manifest.get("metric_contract", {}),
        }
        if split_manifest.get("key_columns") and not split_plan.get("key_columns"):
            split_plan["key_columns"] = split_manifest.get("key_columns")
        metric_contract = split_manifest.get("metric_contract")
        if isinstance(metric_contract, dict):
            if "metrics" in metric_contract:
                split_plan["metrics"] = metric_contract["metrics"]
            if metric_contract.get("primary_metric"):
                split_plan["primary_metric"] = metric_contract["primary_metric"]
            if "maximize" in metric_contract:
                split_plan["maximize"] = bool(metric_contract["maximize"])
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
