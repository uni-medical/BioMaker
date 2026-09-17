"""Candidate prompts retain public encodings without adding target contracts."""

import json

import pandas as pd
import pytest

from biomaker.utils.data_summary import data_and_evaluation_summary
from biomaker.utils.exp_guide import build_exp_guide, write_exp_guide


@pytest.mark.parametrize("prediction_column", ["localizations", "answer_prediction", "target"])
def test_six_field_encoding_reaches_actual_candidate_summary(tmp_path, prediction_column):
    workspace = tmp_path / "workspace"
    split = workspace / "input/biomaker_splits"
    split.mkdir(parents=True)
    for name in ("agent_train", "score_val", "agent_test"):
        pd.DataFrame([{"record_key": name, "acquisition": "public-image.png"}]).to_csv(
            split / f"{name}.csv", index=False
        )
    output = {
        "prediction_columns": [prediction_column],
        "input_file_reference_columns": ["acquisition"],
        "record_key": {"encoding": "KEY_DESCRIPTOR_MUST_NOT_BECOME_PREDICTION"},
        prediction_column: {
            "encoding": "JSON list of tuples",
            "prediction_tuple": "[class_id, x1, y1, x2, y2, confidence]",
            "coordinates": "absolute pixels in the original input image",
            "confidence": "finite number in [0,1]",
            "empty": "[]",
            "truth_schema": {"secret": "NESTED_HIDDEN_SCHEMA"},
        },
        "answer_columns": ["PRIVATE_COLUMN"],
        "target_encoding": "TARGET_REPRESENTATION_MUST_NOT_BE_ADDED",
    }
    manifest = {
        "key_columns": ["record_key"],
        "prediction_contract": output,
        "metric_contract": {"primary_metric": "ap50", "maximize": True},
        "data_semantics": {
            "acquisition": {"axes": ["height", "width"], "encoding": "public RGB input"},
            "PRIVATE_COLUMN": "HIDDEN_INPUT_SEMANTICS",
            "boxes_encoding": "Mixed prediction and TARGET_TUPLE prose",
        },
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest))
    frozen = {
        "key_columns": ["record_key"], "prediction_columns": [prediction_column],
        "primary_metric": "ap50", "maximize": True,
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "answer_columns": ["PRIVATE_COLUMN"],
        "truth_schema": {"private": "FROZEN_TARGET_SCHEMA"},
        "metric_internal_detail": "FULL_METRIC_CONTRACT_MUST_NOT_BE_ADDED",
    }
    (workspace / "evaluator_prediction_spec.json").write_text(json.dumps(frozen))
    write_exp_guide(workspace)
    text = data_and_evaluation_summary(workspace)
    assert "[class_id, x1, y1, x2, y2, confidence]" in text
    assert prediction_column in text
    assert "absolute pixels in the original input image" in text
    assert "public RGB input" in text
    for marker in (
        "KEY_DESCRIPTOR_MUST_NOT_BECOME_PREDICTION", "NESTED_HIDDEN_SCHEMA",
        "PRIVATE_COLUMN", "TARGET_REPRESENTATION_MUST_NOT_BE_ADDED",
        "HIDDEN_INPUT_SEMANTICS", "TARGET_TUPLE", "FROZEN_TARGET_SCHEMA",
        "FULL_METRIC_CONTRACT_MUST_NOT_BE_ADDED",
    ):
        assert marker not in text


def test_frozen_named_box_schema_and_scalar_formats_remain_public(tmp_path):
    workspace = tmp_path / "workspace"
    split = workspace / "input/biomaker_splits"
    split.mkdir(parents=True)
    pd.DataFrame([{"sample": "validation"}]).to_csv(split / "score_val.csv", index=False)
    (split / "split_manifest.json").write_text(json.dumps({
        "key_columns": ["sample"],
        "prediction_contract": {"prediction_columns": ["estimate"]},
        "metric_contract": {"primary_metric": "ap50", "maximize": True},
    }))
    schema = {"class_key": "category", "coordinate_keys": ["bbox"],
              "box_format": "xyxy", "score_key": "probability"}
    frozen = {
        "key_columns": ["sample"], "prediction_columns": ["estimate"],
        "prediction_schema": schema, "coordinate_convention": "pixels",
        "truth_schema": {"private": "DO_NOT_RENDER"},
    }
    guide = build_exp_guide(workspace, prediction_spec=frozen)
    assert guide["output_encoding"]["frozen"]["prediction_schema"] == schema
    assert "DO_NOT_RENDER" not in json.dumps(guide["output_encoding"])
    # A scalar column needs no positional schema; publishing its documented
    # value encoding remains possible without adding metric or target details.
    frozen.pop("prediction_schema")
    frozen["value_encoding"] = "finite floating-point months, no rounding"
    guide = build_exp_guide(workspace, prediction_spec=frozen)
    assert guide["output_encoding"]["frozen"]["value_encoding"] == frozen["value_encoding"]
