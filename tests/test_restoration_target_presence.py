"""Data preparation must catch absent declared restoration targets before scoring."""

import copy
import json

import pandas as pd
import pytest
from PIL import Image

from biomaker.utils import agentic_data_planner as planner


TARGET = "clean_reference"
PROTOCOL = {
    "data_range": 255, "crop_size": None, "groups": None,
    "aggregation": "mean_images", "input_column": "source_image", "target_column": TARGET,
}


@pytest.fixture
def package(tmp_path):
    source, output = tmp_path / "source", tmp_path / "template"
    source.mkdir()
    for name in ("train", "val", "test"):
        Image.new("L", (8, 8), 20).save(source / f"{name}.png")
    for role, name in (("train", "agent_train"), ("val", "score_val"), ("test", "agent_test")):
        row = {"sample": role, "source_image": str(source / f"{role}.png")}
        if role == "train":
            row[TARGET] = str(source / f"{role}.png")
        write_csv(output / "biomaker_splits" / f"{name}.csv", [row])
        if role != "train":
            private_name = "score_val" if role == "val" else "test"
            write_csv(output / "private_gt" / f"{private_name}.csv", [
                {"sample": role, TARGET: str(source / f"{role}.png")},
            ])
    hidden = {role: {"answer_columns": [TARGET]} for role in ("score_val", "test")}
    (output / "private_gt/private_gt.json").write_text(json.dumps(hidden))
    write_manifest(output, {
        "primary_metric": "psnr", "secondary_metrics": ["ssim"],
        "restoration_evaluation": PROTOCOL,
    })
    return source, output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_manifest(output, contract, task_head="image_restoration"):
    (output / "split_manifest.json").write_text(json.dumps({
        "manifest_version": 1, "sample_unit": "image", "key_columns": ["sample"],
        "task_heads": [task_head], "metric_contract": contract,
        "prediction_contract": {"prediction_columns": ["estimate"]},
    }))


@pytest.mark.parametrize("role", ["score_val", "test"])
@pytest.mark.parametrize("empty", ["", "  \t", None])
def test_missing_declared_restoration_target_rejects_entire_split(package, role, empty):
    source, output = package
    path = output / "private_gt" / f"{role}.csv"
    frame = pd.read_csv(path)
    frame[TARGET] = empty
    frame.to_csv(path, index=False)
    result = planner.verify_split_package(output, source_input_dir=source)
    assert not result["ok"]
    assert any(f"private_gt/{role}.csv declared restoration target '{TARGET}' is empty in 1 of 1 rows" in error
               for error in result["errors"]), result["errors"]
    assert pd.read_csv(output / "biomaker_splits/agent_train.csv")[TARGET].notna().all()


def test_checks_all_target_rows_beyond_path_observation_sample(package):
    source, output = package
    public = [{"sample": f"validation-{i}", "source_image": str(source / "val.png")} for i in range(202)]
    hidden = [{"sample": row["sample"], TARGET: str(source / "val.png")} for row in public]
    hidden[-1][TARGET] = ""
    write_csv(output / "biomaker_splits/score_val.csv", public)
    write_csv(output / "private_gt/score_val.csv", hidden)
    result = planner.verify_split_package(output, source_input_dir=source)
    assert not result["ok"]
    assert any("empty in 1 of 202 rows" in error for error in result["errors"])


@pytest.mark.parametrize("style", ["nested", "flat", "named", "task_block"])
def test_target_column_is_taken_from_declared_protocol(package, style):
    source, output = package
    contract = {"primary_metric": "psnr"}
    task = ""
    if style == "nested":
        contract["restoration_evaluation"] = PROTOCOL
    elif style == "flat":
        contract.update(PROTOCOL)
    elif style == "named":
        contract["primary_metric"] = {"name": "psnr", **PROTOCOL}
    else:
        task = "```restoration-evaluation\n" + json.dumps(PROTOCOL) + "\n```"
    write_manifest(output, contract)
    write_csv(output / "private_gt/score_val.csv", [{"sample": "val", TARGET: ""}])
    result = planner.verify_split_package(output, source_input_dir=source, task_text=task)
    assert not result["ok"]
    assert any("declared restoration target" in error and "empty" in error for error in result["errors"])


@pytest.mark.parametrize("inline", [False, True])
def test_complete_restoration_targets_allow_files_and_inline_encodings(package, inline):
    source, output = package
    if inline:
        for role in ("score_val", "test"):
            path = output / "private_gt" / f"{role}.csv"
            frame = pd.read_csv(path)
            frame[TARGET] = json.dumps([[0, 1], [1, 0]])
            frame.to_csv(path, index=False)
    result = planner.verify_split_package(output, source_input_dir=source)
    assert result["ok"], result["errors"]


def test_optional_empty_private_test_placeholder_is_not_a_missing_target(package):
    source, output = package
    (output / "biomaker_splits/agent_test.csv").unlink()
    (output / "private_gt/test.csv").write_text("sample\n")
    hidden = {"score_val": {"answer_columns": [TARGET]}}
    (output / "private_gt/private_gt.json").write_text(json.dumps(hidden))
    result = planner.verify_split_package(output, source_input_dir=source)
    assert result["ok"], result["errors"]


@pytest.mark.parametrize("metric,task_head,empty", [
    ("ap50", "detection", ""), ("ap50", "detection", "[]"),
    ("binary_dice", "segmentation_2d", ""), ("binary_dice", "segmentation_2d", "0"),
])
def test_legitimate_empty_box_and_mask_encodings_remain_valid(package, metric, task_head, empty):
    source, output = package
    write_manifest(output, {"primary_metric": metric}, task_head)
    for role in ("score_val", "test"):
        path = output / "private_gt" / f"{role}.csv"
        frame = pd.read_csv(path)
        frame[TARGET] = empty
        frame.to_csv(path, index=False)
    task = '```split-constraints\n{"required_answer_splits":["score_val","test"]}\n```'
    result = planner.verify_split_package(output, source_input_dir=source, task_text=task)
    assert result["ok"], result["errors"]


def test_missing_restoration_data_retries_builder_with_actionable_feedback(package, tmp_path, monkeypatch):
    source, template = package
    write_csv(template / "private_gt/score_val.csv", [{"sample": "val", TARGET: ""}])
    task_file = tmp_path / "task.md"
    task_file.write_text("Primary metric: PSNR.\nSecondary metric: SSIM.")
    failures = []

    def query(*args, previous_failure=None, **kwargs):
        failures.append(copy.deepcopy(previous_failure))
        script = (
            "import argparse, csv, shutil\nfrom pathlib import Path\n"
            "parser=argparse.ArgumentParser()\n"
            "parser.add_argument('--input-dir')\nparser.add_argument('--output-dir')\n"
            "args=parser.parse_args()\nout=Path(args.output_dir)\n"
            f"shutil.copytree({str(template)!r},out,dirs_exist_ok=True)\n"
        )
        if previous_failure:
            script += (
                "with (out/'private_gt/score_val.csv').open('w') as f:\n"
                f"    writer=csv.DictWriter(f,fieldnames=['sample',{TARGET!r}])\n"
                "    writer.writeheader()\n"
                f"    writer.writerow({{'sample':'val',{TARGET!r}:str(Path(args.input_dir)/'val.png')}})\n"
            )
        return script, "```python\n" + script + "\n```", {}

    monkeypatch.setattr(planner, "_query_builder", query)
    monkeypatch.setattr(planner, "split_builder_cache_key", lambda **kwargs: ("test", {}))
    result = planner.run_agentic_split_builder_probe(source, task_file, tmp_path / "generated", max_attempts=2)
    assert result.verifier["ok"]
    assert [attempt["ok"] for attempt in result.plan["attempts"]] == [False, True]
    assert len(failures) == 2 and failures[0] is None
    assert any("declared restoration target" in error for error in failures[1]["errors"])


@pytest.mark.parametrize("repair_style", ["nested", "named"])
def test_metric_repair_rechecks_new_target_declaration_before_accepting_data(
    package, tmp_path, monkeypatch, repair_style,
):
    source, template = package
    write_csv(template / "private_gt/score_val.csv", [{"sample": "val", TARGET: ""}])
    incomplete_protocol = {key: value for key, value in PROTOCOL.items() if key != "target_column"}
    write_manifest(template, {
        "primary_metric": "psnr", "secondary_metrics": ["ssim"],
        "restoration_evaluation": incomplete_protocol,
        "empty_policy": "A documentation field that needs metric-only repair.",
    })
    repaired_contract = {
        "primary_metric": "psnr", "secondary_metrics": ["ssim"],
        "restoration_evaluation": PROTOCOL,
    }
    if repair_style == "named":
        repaired_contract = {
            "primary_metric": {"name": "psnr", **PROTOCOL},
            "secondary_metrics": ["ssim"],
        }
    task_file = tmp_path / "task.md"
    task_file.write_text("Primary metric: PSNR.\nSecondary metric: SSIM.")
    output = tmp_path / "generated"
    builder_failures, repair_errors = [], []

    def query_builder(*args, previous_failure=None, **kwargs):
        builder_failures.append(copy.deepcopy(previous_failure))
        script = (
            "import argparse, csv, json, shutil\nfrom pathlib import Path\n"
            "parser=argparse.ArgumentParser()\n"
            "parser.add_argument('--input-dir')\nparser.add_argument('--output-dir')\n"
            "args=parser.parse_args()\nout=Path(args.output_dir)\n"
            f"shutil.copytree({str(template)!r},out,dirs_exist_ok=True)\n"
        )
        if previous_failure:
            # A metric-only repair cannot fill missing answers. It must return
            # to the data builder with the target-presence failure intact.
            assert not previous_failure["data_ok"]
            assert previous_failure["builder_returncode"] == 0
            assert previous_failure["metric_errors"] == []
            assert any("declared restoration target" in error and "empty" in error
                       for error in previous_failure["data_errors"])
            assert pd.read_csv(output / "private_gt/score_val.csv", keep_default_na=False)[TARGET].tolist() == [""]
            script += (
                "manifest_path=out/'split_manifest.json'\n"
                "manifest=json.loads(manifest_path.read_text())\n"
                f"manifest['metric_contract']=json.loads({json.dumps(repaired_contract)!r})\n"
                "manifest_path.write_text(json.dumps(manifest))\n"
                "with (out/'private_gt/score_val.csv').open('w') as f:\n"
                f"    writer=csv.DictWriter(f,fieldnames=['sample',{TARGET!r}])\n"
                "    writer.writeheader()\n"
                f"    writer.writerow({{'sample':'val',{TARGET!r}:str(Path(args.input_dir)/'val.png')}})\n"
            )
        return script, "```python\n" + script + "\n```", {}

    def query_repair(task_text, manifest, errors, **kwargs):
        repair_errors.append(errors)
        assert "target_column" not in manifest["metric_contract"]["restoration_evaluation"]
        assert any("empty_policy" in error for error in errors)
        return json.dumps(repaired_contract), {}

    monkeypatch.setattr(planner, "_query_builder", query_builder)
    monkeypatch.setattr(planner, "_query_metric_contract", query_repair)
    monkeypatch.setattr(planner, "split_builder_cache_key", lambda **kwargs: ("test", {}))
    result = planner.run_agentic_split_builder_probe(source, task_file, output, max_attempts=3)

    assert result.verifier["ok"]
    assert [attempt["stage"] for attempt in result.plan["attempts"]] == [
        "data_builder", "metric_contract", "data_builder",
    ]
    assert [attempt["ok"] for attempt in result.plan["attempts"]] == [False, False, True]
    assert len(builder_failures) == 2 and builder_failures[0] is None
    assert len(repair_errors) == 1
    assert pd.read_csv(output / "private_gt/score_val.csv")[TARGET].tolist() == [str(source / "val.png")]
    verified = planner.verify_split_package(output, source_input_dir=source, task_text=task_file.read_text())
    assert verified["ok"], verified["errors"]
