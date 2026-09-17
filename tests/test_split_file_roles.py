"""Input/answer encodings, rather than opaque identities, select file checks."""

import json

import pandas as pd
import pytest

from biomaker.utils.agentic_data_planner import verify_split_package
from biomaker.utils.split_manifest import legacy_manifest


@pytest.fixture
def package(tmp_path):
    source, output = tmp_path / "source", tmp_path / "package"
    source.mkdir()
    hidden = {role: {"answer_columns": ["reference"]} for role in ("score_val", "test")}
    for role, filename in (("train", "agent_train"), ("score_val", "score_val"), ("test", "agent_test")):
        scan, target = source / f"{role}.png", source / f"{role}_reference.png"
        scan.write_bytes(b"input fixture")
        target.write_bytes(b"answer fixture")
        row = {"case_uid": f"collection/{role}/frame.png", "scan": str(scan),
               "description": f"opaque/{role}/metadata.png"}
        truth = {"case_uid": row["case_uid"], "reference": str(target)}
        write_csv(output / "biomaker_splits" / f"{filename}.csv", [{**row, "reference": str(target)}] if role == "train" else [row])
        if role != "train":
            write_csv(output / "private_gt" / f"{role}.csv", [truth])
    write_json(output / "private_gt/private_gt.json", hidden)
    write_json(output / "split_manifest.json", {
        "manifest_version": 1, "sample_unit": "image", "key_columns": ["case_uid"],
        "task_heads": ["image_restoration"], "hidden_answers": hidden,
        "input_file_reference_columns": ["scan"],
        "prediction_contract": {"prediction_columns": ["estimate"], "file_reference_columns": ["estimate"]},
        "metric_contract": {"primary_metric": "psnr", "target_column": "reference"},
    })
    return source, output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def update_manifest(output, change):
    path = output / "split_manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    write_json(path, manifest)


def check(package, **kwargs):
    source, output = package
    return verify_split_package(output, source_input_dir=source, **kwargs)


def set_cell(output, csv, column, value):
    path = output / csv
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame[column] = value
    frame.to_csv(path, index=False)


def declare_answer_files(output, value):
    update_manifest(output, lambda manifest: [entry.update(file_reference_columns=value)
                                            for entry in manifest["hidden_answers"].values()])


@pytest.mark.parametrize("nested", [False, True])
def test_declared_inputs_do_not_turn_opaque_keys_or_metadata_into_files(package, nested):
    if nested:
        update_manifest(package[1], lambda manifest: manifest["prediction_contract"].update(
            input_file_reference_columns=manifest.pop("input_file_reference_columns")))
    result = check(package)
    assert result["ok"], result["errors"]


@pytest.mark.parametrize("missing", [False, True])
def test_key_column_can_also_be_an_explicit_input_file(package, missing):
    source, output = package
    update_manifest(output, lambda manifest: manifest.update(key_columns=["scan"]))
    for role in ("score_val", "test"):
        set_cell(output, f"private_gt/{role}.csv", "scan", str(source / f"{role}.png"))
    if missing:
        (source / "score_val.png").unlink()
    result = check(package)
    assert result["ok"] is not missing, result["errors"]
    if missing:
        assert any("references missing files" in error and "scan=" in error for error in result["errors"])


@pytest.mark.parametrize("bad", ["missing", "blank", "directory", "extensionless"])
def test_explicit_input_references_require_existing_files(package, bad):
    source, output = package
    value = {"missing": str(source / "gone.png"), "blank": " \t ",
             "directory": str(source), "extensionless": str(source / "gone")}[bad]
    set_cell(output, "biomaker_splits/score_val.csv", "scan", value)
    result = check(package)
    assert not result["ok"]
    assert any("references missing files" in error and "scan=" in error for error in result["errors"])


def test_declared_input_column_cannot_be_absent_from_public_rows(package):
    _, output = package
    path = output / "biomaker_splits/score_val.csv"
    pd.read_csv(path).drop(columns="scan").to_csv(path, index=False)
    result = check(package)
    assert not result["ok"]
    assert any("missing declared file reference columns" in error for error in result["errors"])


def test_file_validation_is_not_limited_to_first_200_rows(package):
    source, output = package
    rows = [{"case_uid": f"case/{i}.png", "scan": str(source / "score_val.png")} for i in range(202)]
    rows[-1]["scan"] = str(source / "last_missing.png")
    write_csv(output / "biomaker_splits/score_val.csv", rows)
    write_csv(output / "private_gt/score_val.csv", [{"case_uid": row["case_uid"], "reference": str(source / "score_val_reference.png")} for row in rows])
    result = check(package)
    assert not result["ok"]
    assert any("last_missing.png" in error for error in result["errors"])


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("csv", ["biomaker_splits/agent_train.csv", "private_gt/score_val.csv", "private_gt/test.csv"])
def test_answer_files_are_independent_of_input_file_declarations(package, explicit, csv):
    source, output = package
    if explicit:
        declare_answer_files(output, ["reference"])
    set_cell(output, csv, "reference", str(source / ("missing" if explicit else "missing.png")))
    result = check(package)
    assert not result["ok"]
    assert any("references missing files" in error and "reference=" in error for error in result["errors"])


@pytest.mark.parametrize("metric,encoding", [
    ("ap50", ""), ("ap50", "[]"), ("binary_dice", ""), ("binary_dice", "0"),
    ("mae", "0"), ("accuracy", "class/one.png"),
    ("psnr", "[[0, 1], [1, 0]]"), ("custom_metric", '{"id":"lesion/one.png"}'),
])
def test_explicit_inline_answers_are_not_treated_as_files(package, metric, encoding):
    _, output = package
    declare_answer_files(output, [])
    update_manifest(output, lambda manifest: manifest.update(metric_contract={"primary_metric": metric}))
    for csv in ("biomaker_splits/agent_train.csv", "private_gt/score_val.csv", "private_gt/test.csv"):
        set_cell(output, csv, "reference", encoding)
    result = check(package)
    assert result["ok"], result["errors"]


def test_explicit_inline_restoration_target_still_cannot_be_blank(package):
    _, output = package
    declare_answer_files(output, [])
    set_cell(output, "private_gt/score_val.csv", "reference", "")
    result = check(package)
    assert not result["ok"]
    assert any("declared restoration target" in error and "empty" in error for error in result["errors"])


@pytest.mark.parametrize("file_role", ["input", "answer"])
def test_legacy_roles_still_detect_true_missing_files(package, file_role):
    source, output = package
    update_manifest(output, lambda manifest: manifest.pop("input_file_reference_columns"))
    # Old manifests without encodings retain the old treatment of path-like data.
    for path in [*list((output / "biomaker_splits").glob("*.csv")), *list((output / "private_gt").glob("*.csv"))]:
        frame = pd.read_csv(path, dtype=str)
        frame["case_uid"] = frame["case_uid"].str.replace("/", "_").str.replace(".png", "", regex=False)
        if "description" in frame:
            frame["description"] = "plain metadata"
        frame.to_csv(path, index=False)
    assert check(package)["ok"]
    target = source / ("score_val.png" if file_role == "input" else "score_val_reference.png")
    target.unlink()
    result = check(package)
    assert not result["ok"]
    assert any("references missing files" in error for error in result["errors"])


@pytest.mark.parametrize("field,value", [
    ("input", "scan"), ("input", [None]), ("answer", "reference"), ("answer", ["unknown_answer"]),
])
def test_malformed_role_declarations_are_reported(package, field, value):
    _, output = package
    if field == "input":
        update_manifest(output, lambda manifest: manifest.update(input_file_reference_columns=value))
    else:
        declare_answer_files(output, value)
    result = check(package)
    assert not result["ok"]
    assert any("file_reference_columns" in error for error in result["errors"])


def test_private_answer_file_metadata_is_preserved_in_legacy_manifest(package):
    _, output = package
    hidden = {"score_val": {"answer_columns": ["reference"], "file_reference_columns": ["reference"]}}
    generated = legacy_manifest(output, private_gt=hidden, row_counts={})
    assert generated["hidden_answers"] == hidden


def test_source_coverage_cannot_count_opaque_metadata_as_a_file_reference(package):
    source, output = package
    extra = source / "required_original.bin"
    extra.write_bytes(b"must appear in an actual input file column")
    set_cell(output, "biomaker_splits/score_val.csv", "description", str(extra))
    task = '```split-constraints\n' + json.dumps({"source_rules": [{
        "root": extra.name, "allowed_splits": ["score_val"], "require_complete": True,
    }]}) + '\n```'
    result = check(package, task_text=task)
    assert not result["ok"]
    assert any("complete original-file coverage" in error for error in result["errors"])
