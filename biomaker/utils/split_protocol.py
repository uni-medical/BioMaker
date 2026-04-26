"""Deterministic dataset inspection and split planning."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ID_CANDIDATES = ["patient", "patient_id", "case", "case_id", "study", "Id", "id"]
LABEL_CANDIDATES = [
    "label",
    "class",
    "target",
    "diagnosis",
    "category",
    "grade",
    "finding",
    "y",
    "mask_path",
]
PATH_CANDIDATES = ["path", "image_path", "filename", "file_path"]
MASK_CANDIDATES = ["mask_path", "mask", "label_path"]
GROUP_CANDIDATES = ["patient", "patient_id", "case", "case_id", "study"]
PRIVATE_TEST_COLUMN_TOKENS = [
    "annotation",
    "ground_truth",
    "gt",
    "instance",
    "label",
    "mask",
    "overlap",
    "target",
    "category",
    "grade",
    "finding",
]


@dataclass(frozen=True)
class SplitProtocol:
    dataset_report: dict[str, Any]
    split_plan: dict[str, Any]
    prompt_text: str


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _table_summary(path: Path, input_dir: Path) -> dict[str, Any]:
    df = _read_table(path)
    columns = list(df.columns)
    label_column = _first_existing(columns, LABEL_CANDIDATES)
    summary: dict[str, Any] = {
        "path": str(path.relative_to(input_dir)),
        "rows": len(df),
        "columns": columns,
        "id_column": _first_existing(columns, ID_CANDIDATES),
        "label_column": label_column,
        "path_column": _first_existing(columns, PATH_CANDIDATES),
        "mask_column": _first_existing(columns, MASK_CANDIDATES),
        "group_column": _first_existing(columns, GROUP_CANDIDATES),
    }
    if label_column and df[label_column].nunique(dropna=True) <= 30:
        counts = df[label_column].value_counts(dropna=False).head(30)
        summary["label_distribution"] = {
            str(label): int(count) for label, count in counts.items()
        }
    return summary


def inspect_dataset(input_dir: Path | str) -> dict[str, Any]:
    input_dir = Path(input_dir)
    table_paths = sorted(input_dir.rglob("*.csv"))
    tables = {}
    for path in table_paths:
        if "biomaker_splits" in path.parts or not path.exists():
            continue
        try:
            tables[str(path.relative_to(input_dir))] = _table_summary(path, input_dir)
        except Exception as exc:
            tables[str(path.relative_to(input_dir))] = {
                "path": str(path.relative_to(input_dir)),
                "error": str(exc),
            }
    return {"tables": tables}


def _candidate_table(input_dir: Path, *names: str) -> Path | None:
    for name in names:
        path = input_dir / name
        if path.is_file():
            return path
    for name in names:
        parts = Path(name).parts
        if len(parts) < 2:
            continue
        for path in sorted(input_dir.rglob(parts[-1])):
            if "biomaker_splits" in path.parts or not path.is_file():
                continue
            if path.parts[-len(parts) :] == parts:
                return path
    return None


def _split_dataframe(
    df: pd.DataFrame,
    label_column: str | None,
    group_column: str | None,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"seed": seed, "val_fraction": val_fraction}

    if group_column and group_column in df.columns and df[group_column].nunique() > 1:
        groups = pd.Series(df[group_column].dropna().unique()).sample(
            frac=1.0, random_state=seed
        )
        n_val_groups = max(1, int(round(len(groups) * val_fraction)))
        val_groups = set(groups.iloc[:n_val_groups])
        val_mask = df[group_column].isin(val_groups)
        diagnostics["grouped_by"] = group_column
        return df[~val_mask].copy(), df[val_mask].copy(), diagnostics

    if (
        label_column
        and label_column in df.columns
        and df[label_column].nunique(dropna=True) > 1
        and df[label_column].value_counts().min() >= 2
    ):
        try:
            from sklearn.model_selection import train_test_split

            train_df, val_df = train_test_split(
                df,
                test_size=val_fraction,
                random_state=seed,
                stratify=df[label_column],
            )
            diagnostics["stratified_by"] = label_column
            return train_df.copy(), val_df.copy(), diagnostics
        except Exception as exc:
            diagnostics["warnings"] = [f"Stratified split failed: {exc}"]

    val_df = df.sample(frac=val_fraction, random_state=seed)
    train_df = df.drop(val_df.index)
    diagnostics["random_split"] = True
    return train_df.copy(), val_df.copy(), diagnostics


def _sanitize_test_table(
    df: pd.DataFrame, label_column: str | None, mask_column: str | None
) -> pd.DataFrame:
    private_columns = {column for column in [label_column, mask_column] if column}
    for column in df.columns:
        normalized = column.lower()
        if any(token in normalized for token in PRIVATE_TEST_COLUMN_TOKENS):
            private_columns.add(column)
    keep_columns = [column for column in df.columns if column not in private_columns]
    return df[keep_columns].copy()


def _copy_private_truth(
    input_dir: Path,
    private_dir: Path,
    test_source: Path,
    test_df: pd.DataFrame,
    test_mask_column: str | None,
) -> None:
    source_rel = test_source.relative_to(input_dir)
    private_source = private_dir / source_rel
    private_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_source, private_source)

    if not test_mask_column or test_mask_column not in test_df.columns:
        return

    for value in test_df[test_mask_column].dropna():
        mask_rel = Path(str(value).removeprefix("./"))
        if mask_rel.is_absolute():
            mask_path = mask_rel
            private_mask = private_dir / mask_rel.name
        elif mask_rel.parts and mask_rel.parts[0] == "input":
            mask_path = input_dir.parent / mask_rel
            private_mask = private_dir / Path(*mask_rel.parts[1:])
        else:
            mask_path = test_source.parent / mask_rel
            if not mask_path.is_file():
                mask_path = input_dir / mask_rel
            private_mask = private_dir / mask_rel
        if not mask_path.is_file():
            continue
        private_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_path, private_mask)


def _is_private_test_child(path: Path, test_source: Path) -> bool:
    name = path.name.lower()
    if path.name == test_source.name:
        return True
    return any(token in name for token in PRIVATE_TEST_COLUMN_TOKENS)


def _replace_symlinked_test_dir_with_visible_view(
    test_dir: Path, test_source: Path, visible_test_df: pd.DataFrame
) -> None:
    original_dir = test_dir.resolve()
    metadata_name = test_source.name
    test_dir.unlink()
    test_dir.mkdir(parents=True, exist_ok=True)

    for child in original_dir.iterdir():
        if _is_private_test_child(child, test_source):
            continue
        target = test_dir / child.name
        target.symlink_to(child)

    visible_test_df.to_csv(test_dir / metadata_name, index=False)


def _sanitize_private_test_view(
    input_dir: Path, test_source: Path, visible_test_df: pd.DataFrame
) -> None:
    """Hide private test labels from the agent-visible input tree."""
    if test_source.parent != input_dir and test_source.parent.is_symlink():
        _replace_symlinked_test_dir_with_visible_view(
            test_source.parent, test_source, visible_test_df
        )
        return

    if test_source.is_symlink():
        test_source.unlink()
    visible_test_df.to_csv(test_source, index=False)

    test_dir = test_source.parent
    if test_dir == input_dir:
        return
    for child in list(test_dir.iterdir()):
        if child == test_source:
            continue
        if _is_private_test_child(child, test_source):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)


def _remove_private_workspace_dir(input_dir: Path) -> None:
    private_dir = input_dir.parent / "private"
    if private_dir.is_dir():
        shutil.rmtree(private_dir)


def _as_workspace_absolute(value: Any, input_dir: Path, base_dir: Path | None) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    normalized = value.removeprefix("./")
    if normalized.startswith("input/"):
        return str(input_dir.parent / normalized)
    anchor = base_dir or input_dir
    return str(anchor / normalized)


def _normalize_agent_paths(
    df: pd.DataFrame,
    input_dir: Path,
    path_column: str | None,
    mask_column: str | None,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    df = df.copy()
    for column in {path_column, mask_column}:
        if column and column in df.columns:
            df[column] = df[column].map(
                lambda value: _as_workspace_absolute(value, input_dir, base_dir)
            )
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _relative(path: Path, input_dir: Path) -> str:
    return str(path.relative_to(input_dir))


def _render_prompt(split_plan: dict[str, Any]) -> str:
    lines = [
        "Use the BioMaker split protocol below. Treat it as the source of truth for this run."
    ]
    agent_visible = split_plan.get("agent_visible", {})
    train_path = agent_visible.get("train")
    val_path = agent_visible.get("val")
    if train_path and val_path:
        lines.extend(
            [
                f"- Train only with `./input/{train_path}`.",
                f"- Compute the agent-visible validation metric only on `./input/{val_path}`.",
            ]
        )
    else:
        lines.append(
            "- No deterministic train/validation CSV split was inferred. Inspect the input folder and create any needed development split from training-eligible data only."
        )
    test_path = agent_visible.get("test")
    if test_path:
        lines.append(
            f"- Use `./input/{test_path}` only to generate `./working/submission.csv`; do not use it for training, model selection, debugging, or metric reporting."
        )
    for private_eval in split_plan.get("private_evaluation", []):
        lines.append(
            "- Private evaluation exists for user inspection only. Do not read private labels or report private metrics to the agent-visible validation output."
        )
        if private_eval.get("source"):
            lines.append(
                f"  Private source retained outside the agent metric: `{private_eval['source']}`."
            )
    for warning in split_plan.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    if split_plan.get("columns", {}).get("mask"):
        lines.append(
            "- Mask labels are present. If the task asks for AJI or instance segmentation, optimize AJI on the validation split and save instance masks; Binary Dice should treat every nonzero pixel as foreground."
        )
    return "\n".join(lines)


def prepare_split_protocol(
    input_dir: Path | str, private_dir: Path | str | None = None
) -> SplitProtocol:
    input_dir = Path(input_dir)
    split_dir = input_dir / "biomaker_splits"
    if split_dir.is_dir():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    _remove_private_workspace_dir(input_dir)

    dataset_report = inspect_dataset(input_dir)
    warnings: list[str] = []
    private_evaluation: list[dict[str, Any]] = []
    private_plan: dict[str, Any] = {"version": 1, "private_evaluation": []}
    private_dir = (
        Path(private_dir)
        if private_dir is not None
        else input_dir.parent.parent / "_private_truth" / input_dir.parent.name
    )
    if private_dir.is_dir():
        shutil.rmtree(private_dir)

    train_source = _candidate_table(input_dir, "train.csv", "train/metadata.csv")
    val_source = _candidate_table(
        input_dir, "val.csv", "validation.csv", "val/metadata.csv"
    )
    test_source = _candidate_table(input_dir, "test.csv", "test/metadata.csv")
    metadata_source = _candidate_table(input_dir, "metadata.csv")

    source_for_split = train_source or metadata_source
    if source_for_split is None:
        split_plan = {
            "version": 1,
            "strategy": "not_inferred",
            "agent_visible": {},
            "private_evaluation": [],
            "warnings": [
                "No train.csv, train/metadata.csv, or metadata.csv found. Agent must infer data loading from the raw files."
            ],
        }
    else:
        train_summary = _table_summary(source_for_split, input_dir)
        label_column = train_summary.get("label_column")
        mask_column = train_summary.get("mask_column")
        group_column = train_summary.get("group_column")

        if train_source and val_source:
            strategy = "official_train_val"
            train_df = _read_table(train_source)
            val_df = _read_table(val_source)
            train_base_dir = train_source.parent
            val_base_dir = val_source.parent
            split_diagnostics: dict[str, Any] = {}
        else:
            strategy = "split_from_train"
            source_df = _read_table(source_for_split)
            train_base_dir = source_for_split.parent
            val_base_dir = source_for_split.parent
            train_df, val_df, split_diagnostics = _split_dataframe(
                source_df,
                label_column=label_column,
                group_column=group_column,
            )
            warnings.append(
                "No official validation split found; created an agent-visible validation split from training data."
            )

        agent_train = split_dir / "agent_train.csv"
        agent_val = split_dir / "agent_val.csv"
        train_rel = _relative(agent_train, input_dir)
        val_rel = _relative(agent_val, input_dir)
        train_df = _normalize_agent_paths(
            train_df,
            input_dir,
            train_summary.get("path_column"),
            train_summary.get("mask_column"),
            train_base_dir,
        )
        val_df = _normalize_agent_paths(
            val_df,
            input_dir,
            train_summary.get("path_column"),
            train_summary.get("mask_column"),
            val_base_dir,
        )
        _write_csv(train_df, agent_train)
        _write_csv(val_df, agent_val)

        agent_visible: dict[str, str] = {"train": train_rel, "val": val_rel}

        if test_source:
            test_summary = _table_summary(test_source, input_dir)
            test_df = _read_table(test_source)
            test_label_column = test_summary.get("label_column")
            test_mask_column = test_summary.get("mask_column")
            _copy_private_truth(
                input_dir, private_dir, test_source, test_df, test_mask_column
            )
            agent_test = split_dir / "agent_test.csv"
            agent_test_df = _normalize_agent_paths(
                _sanitize_test_table(test_df, test_label_column, test_mask_column),
                input_dir,
                test_summary.get("path_column"),
                None,
                test_source.parent,
            )
            _write_csv(
                agent_test_df,
                agent_test,
            )
            agent_visible["test"] = _relative(agent_test, input_dir)
            private_evaluation.append(
                {
                    "name": "official_test",
                    "source": str(test_source.relative_to(input_dir)),
                    "visible_to_agent": False,
                    "report_to_user": True,
                }
            )
            private_plan["private_evaluation"].append(
                {
                    "name": "official_test",
                    "source": str(test_source.relative_to(input_dir)),
                    "visible_to_agent": False,
                    "report_to_user": True,
                    "label_column": test_label_column,
                    "mask_column": test_mask_column,
                    "id_column": test_summary.get("id_column"),
                }
            )
            _sanitize_private_test_view(input_dir, test_source, agent_test_df)

        split_plan = {
            "version": 1,
            "strategy": strategy,
            "agent_visible": agent_visible,
            "sources": {
                "train": str(train_source.relative_to(input_dir)) if train_source else str(metadata_source.relative_to(input_dir)),  # type: ignore[union-attr]
                "val": str(val_source.relative_to(input_dir)) if val_source else None,
                "test": (
                    str(test_source.relative_to(input_dir)) if test_source else None
                ),
            },
            "columns": {
                "id": train_summary.get("id_column"),
                "label": label_column,
                "path": train_summary.get("path_column"),
                "mask": mask_column,
                "group": group_column,
            },
            "metric_split": "agent_val",
            "private_evaluation": private_evaluation,
            "diagnostics": split_diagnostics,
            "warnings": warnings,
        }

    dataset_report = inspect_dataset(input_dir)
    (split_dir / "dataset_report.json").write_text(
        json.dumps(dataset_report, indent=2), encoding="utf-8"
    )
    (split_dir / "split_plan.json").write_text(
        json.dumps(split_plan, indent=2), encoding="utf-8"
    )
    if private_plan["private_evaluation"]:
        private_dir.mkdir(parents=True, exist_ok=True)
        (private_dir / "private_split_plan.json").write_text(
            json.dumps(private_plan, indent=2), encoding="utf-8"
        )
    prompt_text = _render_prompt(split_plan)
    (split_dir / "split_protocol.txt").write_text(prompt_text, encoding="utf-8")
    return SplitProtocol(dataset_report, split_plan, prompt_text)
