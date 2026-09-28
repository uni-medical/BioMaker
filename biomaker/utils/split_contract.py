"""The single import boundary for public split declarations.

Version 2 manifests declare the complete role set in ``splits``. Legacy
declarations are converted here, never inferred from missing files. A legacy
manifest without an exact declaration retains the historical three-split
protocol; checking its files is the caller's responsibility.
"""

from __future__ import annotations

from typing import Any


MANIFEST_VERSION = 2
SPLIT_TO_FILE = {
    "train": "biomaker_splits/agent_train.csv",
    "score_val": "biomaker_splits/score_val.csv",
    "test": "biomaker_splits/agent_test.csv",
}


class SplitContractError(ValueError):
    """A split declaration is ambiguous, inconsistent, or unsupported."""


def validate_split_roles(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(role, str) or role not in SPLIT_TO_FILE for role in value
    ):
        raise SplitContractError(f"{field} must list train, score_val and optionally test.")
    if len(value) != len(set(value)) or not {"train", "score_val"} <= set(value):
        raise SplitContractError(f"{field} must contain train and score_val exactly once, and optionally test.")
    return [role for role in SPLIT_TO_FILE if role in value]


def _public_path(value: Any, role: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitContractError(f"{field}.{role} must declare a nonempty public CSV path.")
    path = value.strip().removeprefix("./").removeprefix("input/")
    expected = SPLIT_TO_FILE[role]
    if path != expected:
        raise SplitContractError(
            f"Unsupported public split path {field}.{role}={value!r}; expected {expected!r}."
        )
    return expected


def _declaration(value: Any, field: str, *, canonical: bool = False) -> dict[str, str]:
    if isinstance(value, list) and not canonical:
        return {role: SPLIT_TO_FILE[role] for role in validate_split_roles(value, field)}
    if not isinstance(value, dict):
        raise SplitContractError(f"{field} must be an object mapping split roles to public CSV files.")
    roles = validate_split_roles(list(value), field)
    result = {}
    for role in roles:
        entry = value[role]
        if isinstance(entry, str) and not canonical:
            result[role] = _public_path(entry, role, field)
            continue
        if not isinstance(entry, dict):
            raise SplitContractError(f"{field}.{role} must be an object with a file path.")
        if canonical and any(key in entry for key in ("path", "public_rows")):
            raise SplitContractError(f"{field}.{role} must use file as its only public CSV path field.")
        keys = ("file",) if canonical else ("file", "path", "public_rows")
        paths = [_public_path(entry[key], role, field) for key in keys if key in entry]
        if not paths:
            raise SplitContractError(f"{field}.{role}.file is required.")
        if len(set(paths)) != 1:
            raise SplitContractError(f"Conflicting public CSV paths in {field}.{role}.")
        result[role] = paths[0]
    return result


def resolve_public_splits(
    manifest: dict[str, Any], *, task_splits: list[str] | None = None,
    require_explicit: bool = False,
) -> dict[str, str]:
    """Resolve an exact role/path map without consulting the filesystem.

    ``task_splits`` is the optional exact task protocol. Task-owned
    ``required_splits`` deliberately does not enter this API: it is a minimum,
    unlike the historical manifest field with the same name.
    """
    version = manifest.get("manifest_version", 1)
    if version not in (1, MANIFEST_VERSION):
        raise SplitContractError(f"Unsupported split manifest version: {version!r}.")
    if version == MANIFEST_VERSION and "splits" not in manifest:
        raise SplitContractError("Version 2 split_manifest.splits must declare the complete public split set.")
    declarations: list[tuple[str, dict[str, str]]] = []
    for field in ("splits", "public_splits", "required_splits"):
        if field in manifest:
            declarations.append((field, _declaration(
                manifest[field], "split_manifest." + field,
                canonical=version == MANIFEST_VERSION and field == "splits",
            )))
    if task_splits is not None:
        declarations.append(("task split-constraints.splits", {
            role: SPLIT_TO_FILE[role]
            for role in validate_split_roles(task_splits, "split-constraints.splits")
        }))
    if declarations:
        first_name, result = declarations[0]
        for name, declaration in declarations[1:]:
            if declaration != result:
                raise SplitContractError(f"Conflicting public split declarations: {first_name} and {name}.")
    elif require_explicit:
        raise SplitContractError("Declare the complete public split set in split_manifest.splits; missing test files are not a no-test declaration.")
    else:
        result = dict(SPLIT_TO_FILE)
    for field in ("hidden_answers", "private_gt"):
        answers = manifest.get(field)
        if isinstance(answers, dict):
            undeclared = (set(answers) & set(SPLIT_TO_FILE)) - set(result)
            if undeclared:
                raise SplitContractError(f"{field} declares answers for absent public splits: {sorted(undeclared)}.")
    return dict(result)


def normalize_split_manifest(
    manifest: dict[str, Any], *, task_splits: list[str] | None = None,
    require_explicit: bool = False,
) -> dict[str, Any]:
    """Import legacy spellings once while retaining split validation metadata."""
    paths = resolve_public_splits(manifest, task_splits=task_splits, require_explicit=require_explicit)
    splits = {}
    for role, path in paths.items():
        entry = {}
        for field in ("public_splits", "splits"):
            source = manifest.get(field)
            if isinstance(source, dict) and isinstance(source.get(role), dict):
                entry.update(source[role])
        entry.pop("path", None)
        entry.pop("public_rows", None)
        entry.pop("private_answers", None)
        entry.pop("private_gt", None)
        entry["file"] = path
        splits[role] = entry
    normalized = {**manifest, "manifest_version": MANIFEST_VERSION, "splits": splits}
    normalized.pop("public_splits", None)
    normalized.pop("required_splits", None)
    return normalized
