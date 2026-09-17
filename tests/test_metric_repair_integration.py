"""A repaired preparation declaration reaches the real frozen mask evaluator."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomaker.utils import agentic_data_planner as planner, evaluator_setup


def test_monuseg_declaration_repair_preserves_data_and_reaches_trusted_scoring(
    tmp_path, monkeypatch,
):
    task = "Primary evaluation metric: AJI.\nSecondary evaluation metric: Binary Dice."
    key, image_column = "slide token", "microscopy tensor"
    answer, prediction = "nuclear truth", "inferred objects"
    source, template = tmp_path / "source", tmp_path / "template"
    source.mkdir()
    (template / "biomaker_splits").mkdir(parents=True)
    (template / "private_gt/targets").mkdir(parents=True)

    empty = np.zeros((8, 8), dtype=np.uint16)
    nucleus = empty.copy()
    nucleus[1:3, 1:3] = 451
    instances = nucleus.copy()
    instances[5:7, 5:7] = 999
    relabeled = np.where(instances == 451, 42, np.where(instances == 999, 1023, 0)).astype(np.uint16)
    shifted = empty.copy()
    shifted[1:3, 2:4] = 83
    cases = {
        "both-empty": (empty, empty),
        "target-only": (nucleus, empty),
        "prediction-only": (empty, nucleus),
        "distinct-instances": (instances, relabeled),
        "partial-overlap": (nucleus, shifted),
    }
    rows = {}
    for token, target in [("train", instances), ("held-out", instances)] + [
        (token, pair[0]) for token, pair in cases.items()
    ]:
        input_path = source / (token + ".png")
        Image.new("RGB", (8, 8), color=(70, 30, 90)).save(input_path)
        target_path = template / "private_gt/targets" / (token + ".png")
        Image.fromarray(target).save(target_path)
        rows[token] = {key: token, image_column: str(input_path), answer: str(target_path)}

    def write_table(path, values):
        pd.DataFrame(values).to_csv(path, index=False)

    def public(token):
        return {column: value for column, value in rows[token].items() if column != answer}

    write_table(template / "biomaker_splits/agent_train.csv", [rows["train"]])
    write_table(template / "biomaker_splits/score_val.csv", [public(token) for token in cases])
    write_table(template / "biomaker_splits/agent_test.csv", [public("held-out")])
    write_table(template / "private_gt/score_val.csv", [rows[token] for token in reversed(cases)])
    write_table(template / "private_gt/test.csv", [rows["held-out"]])
    private_metadata = {split: {"answer_columns": [answer]} for split in ("score_val", "test")}
    (template / "private_gt/private_gt.json").write_text(json.dumps(private_metadata))
    empty_policy = "both empty: Dice/IoU/AJI 1; exactly one empty: 0"
    invalid_contract = {
        "primary_metric": "aji", "secondary_metrics": ["binary_dice"],
        "aji_matching": "per_gt_best_iou", "empty_policy": empty_policy,
    }
    repaired_contract = {**invalid_contract, "notes": invalid_contract["empty_policy"]}
    del repaired_contract["empty_policy"]
    manifest = {
        "manifest_version": 1, "sample_unit": "image", "key_columns": [key],
        "task_heads": ["instance_segmentation"],
        "prediction_contract": {
            "prediction_columns": [prediction], "file_reference_columns": [prediction],
            "image_column": image_column, "prediction_type": "mask_file_path",
        },
        "metric_contract": invalid_contract,
    }
    (template / "split_manifest.json").write_text(json.dumps(manifest))
    task_file = tmp_path / "task.md"
    task_file.write_text(task)
    output = tmp_path / "prepared"
    builder_calls, repair_calls = [], []

    def files(root):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("biomaker_splits", "private_gt")
            for path in (root / folder).rglob("*") if path.is_file()
        }

    original_files = files(template)

    def query_builder(*args, **kwargs):
        builder_calls.append(kwargs)
        return (
            "import argparse, shutil\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--input-dir')\nparser.add_argument('--output-dir')\n"
            f"shutil.copytree({str(template)!r}, parser.parse_args().output_dir, dirs_exist_ok=True)\n",
            "builder response", {},
        )

    def query_repair(task_text, prepared_manifest, errors, **kwargs):
        repair_calls.append(errors)
        assert task_text == task
        assert prepared_manifest["metric_contract"] == invalid_contract
        assert "empty_policy" in " ".join(errors)
        assert files(output) == original_files
        return json.dumps(repaired_contract), {}

    monkeypatch.setattr(planner, "_query_builder", query_builder)
    monkeypatch.setattr(planner, "_query_metric_contract", query_repair)
    monkeypatch.setattr(
        evaluator_setup.backend, "query",
        lambda **kwargs: pytest.fail("Known mask representation must use the trusted built-in adapter"),
    )
    result = planner.run_agentic_split_builder_probe(source, task_file, output, max_attempts=2)
    assert result.verifier["ok"]
    assert len(builder_calls) == len(repair_calls) == 1
    assert [attempt["stage"] for attempt in result.plan["attempts"]] == ["data_builder", "metric_contract"]
    assert files(output) == original_files
    assert json.loads((output / "split_manifest.json").read_text()) == {
        **manifest, "metric_contract": repaired_contract,
    }

    workspace = tmp_path / "workspace"
    split_dir = workspace / "input/biomaker_splits"
    shutil.copytree(result.split_dir, split_dir)
    shutil.copy2(output / "split_manifest.json", split_dir / "split_manifest.json")
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", task, model="unused", attempts=1,
        hidden_score_path=result.private_gt_dir / "score_val.csv", timeout=30,
    )
    spec = json.loads(prepared.prediction_spec)
    assert spec["scoring_backend"] == "trusted_metrics_v1"
    assert spec["key_columns"] == [key]
    assert spec["prediction_columns"] == [prediction]
    assert spec["image_column"] == image_column
    assert spec["aji_matching"] == "per_gt_best_iou"
    assert "answer_columns" not in spec
    assert answer not in pd.read_csv(split_dir / "score_val.csv").columns
    frozen_code = prepared.path.read_bytes()

    working = workspace / "working"
    working.mkdir(exist_ok=True)
    predictions = []
    for token, (_, mask) in reversed(cases.items()):
        filename = token + ".png"
        Image.fromarray(mask).save(working / filename)
        predictions.append({key: token, prediction: filename})
    write_table(working / "validation_predictions.csv", predictions)
    Image.fromarray(relabeled).save(working / "held-out.png")
    write_table(working / "submission.csv", [{key: "held-out", prediction: "held-out.png"}])
    scored = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=result.private_gt_dir / "score_val.csv", timeout=30,
    )
    assert scored.ok, scored.output + str(scored.error)
    # Per-image AJI: 1, 0, 0, 1, 2/6. Binary Dice: 1, 0, 0, 1, 4/8.
    assert scored.metric == pytest.approx(7 / 15)
    assert scored.details["score"]["metrics"] == pytest.approx({"aji": 7 / 15, "binary_dice": 0.5})
    assert scored.details["score"]["n"] == len(cases)
    assert prepared.path.read_bytes() == frozen_code
