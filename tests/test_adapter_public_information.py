"""Adapters and candidates receive public representation facts for arbitrary schemas."""

import json

import numpy as np
import pandas as pd
import pytest

from biomaker.utils.evaluator_setup import _preparation_context
from biomaker.utils.exp_guide import build_exp_guide, node_exp_guide_view


def public_workspace(tmp_path, extension="npy", *, declared=True):
    workspace, private = tmp_path / "workspace", tmp_path / "private_gt"
    split = workspace / "input/biomaker_splits"
    split.mkdir(parents=True)
    private.mkdir()
    captures = workspace / "input/captures"
    captures.mkdir()
    capture = captures / ("scan." + extension)
    if extension == "npy":
        np.save(capture, np.zeros((5, 7), dtype=np.uint16))
    else:
        np.savez(capture, signal=np.zeros((5, 7), dtype=np.uint16),
                 acquisition=np.ones((2,), dtype=np.float32))
    sensor = "input/captures/scan." + extension
    for name in ("agent_train", "score_val", "agent_test"):
        # The input reference deliberately lacks path/file/image in its column
        # name and follows more than eight other metadata columns.
        row = {"image_id": name, **{f"metadata_{index}": index for index in range(10)},
               "sensor_capture": sensor}
        if name == "agent_train":
            row["reference_grade"] = "PUBLIC_TRAIN_LABEL"
        pd.DataFrame([row]).to_csv(split / (name + ".csv"), index=False)
    for name, key in (("score_val", "score_val"), ("test", "agent_test")):
        pd.DataFrame([{"image_id": key, "reference_grade": "HIDDEN_ANSWER_" + name}]).to_csv(
            private / (name + ".csv"), index=False)
    hidden = {name: {"answer_columns": ["reference_grade"]} for name in ("score_val", "test")}
    (private / "private_gt.json").write_text(json.dumps(hidden))
    (split / "split_plan.json").write_text(json.dumps({"primary_metric": "accuracy", "private_gt": hidden}))
    semantics = {
        "data_semantics": {"sensor_capture": {"encoding": extension, "axes": ["height", "width"],
                                               "array_key": "signal" if extension == "npz" else None}},
        "row_representation": "One public scan per image_id; preserve its full native geometry.",
        "semantic_assumptions": ["Array axes are ordered height then width.", "No implicit color conversion."],
        "sample_unit": "scan",
    }
    output = {"prediction_columns": ["probability"], "class_labels": ["negative", "positive"]}
    if declared:
        output["input_file_reference_columns"] = ["sensor_capture"]
    manifest = {
        "manifest_version": 1, "key_columns": ["image_id"],
        "task_heads": [{"type": "binary"}],
        "metric_contract": {"primary_metric": "accuracy", "maximize": True},
        "prediction_contract": output, **semantics,
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest))
    return workspace, private, semantics


@pytest.mark.parametrize("extension", ["npy", "npz"])
@pytest.mark.parametrize("declared", [False, True])
def test_arbitrary_input_column_after_preview_limit_is_observed(tmp_path, extension, declared):
    workspace, private, _ = public_workspace(tmp_path, extension, declared=declared)
    guide = build_exp_guide(workspace)
    for name in ("train", "score_val", "test"):
        sample = guide["samples"][name]
        observation = sample["path_observations"]["sensor_capture"]
        assert observation["exists"] is True
        assert observation["value"] == "input/captures/scan." + extension
        if declared:
            assert sample["first_row"]["sensor_capture"] == observation["value"]
        if extension == "npy":
            assert observation["kind"] == "numpy_array"
            assert observation["array_shape"] == [5, 7]
            assert observation["array_dtype"] == "uint16"
        else:
            assert observation["kind"] == "numpy_archive"
            assert observation["arrays"] == {
                "signal": {"shape": [5, 7], "dtype": "uint16"},
                "acquisition": {"shape": [2], "dtype": "float32"},
            }
    _, declared_context, observations = _preparation_context(workspace, private / "score_val.csv")
    if declared:
        assert declared_context["prediction_contract"]["input_file_reference_columns"] == ["sensor_capture"]
    assert observations == guide["samples"]


def test_preparation_and_candidate_views_preserve_semantics_without_private_values(tmp_path):
    workspace, private, semantics = public_workspace(tmp_path)
    guide = build_exp_guide(workspace)
    node = node_exp_guide_view(guide)
    preparation, declared, observations = _preparation_context(workspace, private / "score_val.csv")
    for key, value in semantics.items():
        assert declared[key] == value
        assert node["data_semantics"][key] == value
        assert preparation["data_semantics"][key] == value
    payload = json.dumps([node, preparation, declared, observations])
    assert "PUBLIC_TRAIN_LABEL" in payload
    assert "reference_grade" in payload
    assert "HIDDEN_ANSWER_score_val" not in payload
    assert "HIDDEN_ANSWER_test" not in payload
    assert str(private) not in payload
    for name in ("score_val", "test"):
        assert "reference_grade" not in observations[name]["first_row"]
