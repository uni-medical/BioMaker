"""Exercise preparation through frozen, standalone data adapters, without an LLM."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from biomaker.utils import evaluator_setup
from biomaker.utils.private_evaluation import evaluate_private_outputs


TASK = "Primary evaluation metric: PSNR.\nSecondary evaluation metric: SSIM."
RESTORATION = {
    "data_range": 255,
    "crop_size": None,
    "group_column": None,
    "groups": None,
    "aggregation": "mean_images",
}
SPEC = {
    "evaluator_api_version": 2,
    "task_type": "image_restoration",
    "primary_metric": "psnr",
    "secondary_metrics": ["ssim"],
    "maximize": True,
    "key_columns": ["sample"],
    "prediction_columns": ["result_json"],
    "answer_columns": ["reference_json"],
    "validation_artifact": "./working/validation_predictions.csv",
    "test_submission": "./working/submission.csv",
    "restoration_evaluation": RESTORATION,
    "ssim_parameters": {"window_size": 7},
}


def adapter_source(spec=None):
    return "import json\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\nADAPTER_SPEC = " + repr(spec or SPEC) + "\n" + '''
def prediction_spec():
    return ADAPTER_SPEC

def load_sample(prediction_row, target_row, expected_row):
    sample = {
        'prediction': np.asarray(json.loads(prediction_row['result_json']), dtype=float),
        'input_shape': json.loads(expected_row['input_shape']),
    }
    if target_row is not None:
        sample['target'] = np.asarray(json.loads(target_row['reference_json']), dtype=float)
    return sample

def write_dummy_predictions(expected_rows, artifact_path):
    rows = []
    for row in expected_rows:
        values = np.full(json.loads(row['input_shape']), 0.5)
        rows.append({'sample': row['sample'], 'result_json': json.dumps(values.tolist())})
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(artifact_path, index=False)
'''


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def matrix(value):
    return json.dumps(np.full((8, 8), value).tolist())


@pytest.fixture
def run_inputs(tmp_path):
    workspace = tmp_path / "workspace"
    split_dir = workspace / "input/biomaker_splits"
    private = tmp_path / "private_gt"
    for name, ids in (("score_val", [f"val-{i}" for i in range(6)]),
                      ("agent_test", ["new-a", "new-b", "new-c"])):
        write_csv(split_dir / (name + ".csv"), [
            {"sample": key, "input_shape": "[8, 8]"} for key in ids
        ])
        hidden_name = "test" if name == "agent_test" else name
        write_csv(private / (hidden_name + ".csv"), [
            {"sample": key, "reference_json": matrix(20)} for key in reversed(ids)
        ])
    write_csv(split_dir / "agent_train.csv", [{
        "sample": "training", "input_shape": "[8, 8]", "reference_json": matrix(10),
    }])
    answers = {name: {"answer_columns": ["reference_json"]} for name in ("score_val", "test")}
    write_json(private / "private_gt.json", answers)
    write_json(split_dir / "split_plan.json", {"private_gt": answers, "primary_metric": "psnr"})
    write_json(split_dir / "split_manifest.json", {
        "manifest_version": 1,
        "key_columns": ["sample"],
        "task_heads": [{"type": "image_restoration"}],
        "prediction_contract": {"prediction_columns": ["result_json"]},
        "metric_contract": {
            "primary_metric": "psnr", "secondary_metrics": ["ssim"], "maximize": True,
            "restoration_evaluation": RESTORATION,
            "ssim_parameters": {"window_size": 7},
        },
    })
    return workspace, private


def backend_responses(monkeypatch, *sources):
    prompts = []
    responses = iter(sources)

    def query(**kwargs):
        prompts.append(copy.deepcopy(kwargs["system_message"]))
        try:
            source = next(responses)
        except StopIteration:
            pytest.fail("Preparation unexpectedly requested another generated adapter.")
        return "```python\n" + source + "\n```"

    monkeypatch.setattr(evaluator_setup.backend, "query", query)
    return prompts


def prepare(run_inputs, **kwargs):
    workspace, private = run_inputs
    return evaluator_setup.prepare_evaluator(
        workspace, workspace.parent / "logs", TASK, model="mock-adapter-model",
        hidden_score_path=private / "score_val.csv", timeout=30, **kwargs,
    )


def write_candidate_predictions(workspace, value=21):
    split_dir = workspace / "input/biomaker_splits"
    for name, artifact in (("score_val", "validation_predictions"), ("agent_test", "submission")):
        frame = pd.read_csv(split_dir / (name + ".csv"), dtype=str)
        write_csv(workspace / "working" / (artifact + ".csv"), [
            {"sample": key, "result_json": matrix(value)} for key in reversed(frame["sample"].tolist())
        ])


def test_json_restoration_adapter_is_frozen_and_shared_with_private_test(run_inputs, monkeypatch):
    prompts = backend_responses(monkeypatch, adapter_source())
    prepared = prepare(run_inputs, attempts=1)
    workspace, private = run_inputs
    spec = json.loads(prepared.prediction_spec)
    assert len(prompts) == 1
    assert "Data adapter API" in prompts[0]
    assert "Trusted metric API and parameters" in prompts[0]
    declarations = prompts[0]["Prepared data and metric declarations"]
    manifest = json.loads((workspace / "input/biomaker_splits/split_manifest.json").read_text())
    assert declarations["metric_contract"] == manifest["metric_contract"]
    assert declarations["prediction_contract"] == manifest["prediction_contract"]
    assert declarations["metric_contract"]["restoration_evaluation"] == RESTORATION
    assert str(private / "score_val.csv") not in json.dumps(prompts)
    assert matrix(20) not in json.dumps(prompts)
    assert spec["adapter_api_version"] == 1
    assert spec["scoring_backend"] == "trusted_metrics_v1"
    from biomaker.utils.exp_guide import build_exp_guide, node_exp_guide_view
    candidate_view = node_exp_guide_view(build_exp_guide(workspace))
    assert candidate_view["evaluation_protocol"]["restoration_evaluation"] == spec["restoration_evaluation"]
    assert candidate_view["evaluation_protocol"]["ssim_parameters"]["win_size"] == 7
    assert "answer_columns" not in candidate_view["evaluation_protocol"]
    assert spec["prediction_columns"] == ["result_json"]
    assert "answer_columns" not in spec
    assert spec["ssim_parameters"]["win_size"] == 7
    assert not (workspace / "working/.adapter_probe").exists()
    frozen = prepared.path.read_bytes()

    write_candidate_predictions(workspace)
    validation = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv", timeout=30,
    )
    assert validation.ok, validation.output + str(validation.error)
    test = evaluate_private_outputs(workspace, private)
    assert test is not None and "error" not in test, test
    assert validation.metric == pytest.approx(20 * np.log10(255))
    assert test["results"][0]["metrics"] == pytest.approx(validation.details["score"]["metrics"])
    assert validation.details["score"]["n"] == 6
    assert test["results"][0]["n"] == 3
    assert test["visible_to_agent"] is False
    assert prepared.path.read_bytes() == frozen

    # Different private answers affect only private scoring; the kernel stays frozen.
    truth = pd.read_csv(private / "test.csv", dtype=str)
    truth["reference_json"] = matrix(19)
    truth.to_csv(private / "test.csv", index=False)
    changed = evaluate_private_outputs(workspace, private)
    assert changed["results"][0]["value"] == pytest.approx(20 * np.log10(255 / 2))
    assert prepared.path.read_bytes() == frozen


def test_invalid_data_adapter_retries_only_adapter_generation(run_inputs, monkeypatch):
    invalid = adapter_source().replace("for row in expected_rows:", "for row in expected_rows[:-1]:")
    prompts = backend_responses(monkeypatch, invalid, adapter_source())
    prepare(run_inputs, attempts=2)
    assert len(prompts) == 2
    for prompt in prompts:
        assert "Data adapter API" in prompt
        assert "Write only a thin data adapter" in prompt["Introduction"]
        assert "Harness supplies tested metric functions" in prompt["Introduction"]
    assert "missing rows" in prompts[1]["Previous evaluator errors"]


def test_freeze_rejects_adapter_change_to_declared_scoring_protocol(run_inputs, monkeypatch):
    changed = copy.deepcopy(SPEC)
    changed["restoration_evaluation"]["crop_size"] = 8
    prompts = backend_responses(monkeypatch, adapter_source(changed), adapter_source())
    prepared = prepare(run_inputs, attempts=2)
    assert len(prompts) == 2
    assert "changed the frozen metric protocol" in prompts[1]["Previous evaluator errors"]
    assert json.loads(prepared.prediction_spec)["restoration_evaluation"]["crop_size"] is None


def test_cache_reuses_frozen_adapter_and_manifest_changes_invalidate(run_inputs, monkeypatch):
    workspace, _ = run_inputs
    cache = workspace.parent / "evaluator_cache"
    prompts = backend_responses(monkeypatch, adapter_source(), adapter_source())
    first = prepare(run_inputs, attempts=1, cache_root=cache)
    first_code = first.path.read_bytes()
    reused = prepare(run_inputs, attempts=1, cache_root=cache)
    assert len(prompts) == 1
    assert reused.prediction_spec == first.prediction_spec
    assert reused.path.read_bytes() == first_code

    manifest_path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["manifest_version"] = 2
    write_json(manifest_path, manifest)
    regenerated = prepare(run_inputs, attempts=1, cache_root=cache)
    assert len(prompts) == 2
    assert regenerated.path.read_bytes() == first_code
    assert len(list(cache.glob("*/manifest.json"))) == 2


def test_echoed_unknown_protocol_field_cannot_bypass_trusted_semantics(run_inputs, monkeypatch):
    workspace, _ = run_inputs
    path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metric_contract"]["aggregation_mode"] = "pooled_mse"
    write_json(path, manifest)
    echoed = {**SPEC, "aggregation_mode": "pooled_mse"}
    prompts = backend_responses(monkeypatch, adapter_source(echoed))
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="Unsupported declared metric protocol fields: aggregation_mode"):
        prepare(run_inputs, attempts=1)
    assert not prompts  # Semantic incompatibility is known before data-adapter generation.
    assert not (workspace / "evaluator.py").exists()


CUSTOM_EVALUATOR = '''
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

def prediction_spec():
    return {
        'evaluator_api_version': 2, 'task_type': 'image_restoration',
        'primary_metric': 'inverse_mae', 'secondary_metrics': [], 'maximize': True,
        'key_columns': ['sample'], 'prediction_columns': ['result_json'],
        'validation_artifact': './working/validation_predictions.csv',
        'test_submission': './working/submission.csv',
        'score_split': './input/biomaker_splits/score_val.csv',
    }

def score(prediction_path, truth_path, expected_path):
    expected = pd.read_csv(expected_path, dtype=str)
    prediction = pd.read_csv(prediction_path, dtype=str)
    truth = pd.read_csv(truth_path, dtype=str)
    joined = expected[['sample']].merge(prediction, on='sample', validate='one_to_one')
    joined = joined.merge(truth, on='sample', validate='one_to_one')
    assert len(joined) == len(expected)
    values = []
    for row in joined.to_dict('records'):
        estimate = np.asarray(json.loads(row['result_json']))
        reference = np.asarray(json.loads(row['reference_json']))
        values.append(1.0 / (1.0 + float(np.mean(np.abs(estimate-reference)))))
    value = float(np.mean(values))
    return {'metric': value, 'maximize': True, 'metrics': {'inverse_mae': value}, 'n': len(expected)}

def score_val():
    spec = prediction_spec()
    return score(spec['validation_artifact'], os.environ['BIOMAKER_SCORE_VAL_PATH'], spec['score_split'])

def validate_submission():
    expected = pd.read_csv('input/biomaker_splits/agent_test.csv', dtype=str)
    prediction = pd.read_csv('working/submission.csv', dtype=str)
    assert set(expected['sample']) == set(prediction['sample'])
    assert len(expected) == len(prediction)
    return {'valid': True}

def self_test():
    Path('working').mkdir(exist_ok=True)
    for split, artifact in [('score_val', 'validation_predictions'), ('agent_test', 'submission')]:
        rows = pd.read_csv('input/biomaker_splits/' + split + '.csv', dtype=str)
        rows = rows[['sample']].copy()
        rows['result_json'] = json.dumps(np.full((8, 8), 0.5).tolist())
        rows.to_csv('working/' + artifact + '.csv', index=False)
    score_val()
    validate_submission()
    print('EVALUATOR_OK')
'''


def test_unavailable_metric_uses_custom_evaluator_with_retained_probe_fixtures(run_inputs, monkeypatch):
    workspace, private = run_inputs
    path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metric_contract"] = {"primary_metric": "inverse_mae", "secondary_metrics": [], "maximize": True}
    write_json(path, manifest)
    prompts = backend_responses(monkeypatch, CUSTOM_EVALUATOR)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, workspace.parent / "logs", "Primary evaluation metric: inverse_mae.",
        model="mock-custom-model", attempts=1, timeout=30,
        hidden_score_path=private / "score_val.csv",
    )
    assert len(prompts) == 1
    assert "custom-metric evaluator" in prompts[0]["Introduction"]
    assert "Data adapter API" not in prompts[0]
    assert "Trusted metric API" in prompts[0]
    assert "MUST leave" in json.dumps(prompts[0]["Evaluator requirements"])
    spec = json.loads(prepared.prediction_spec)
    assert spec["evaluator_api_version"] == 2
    assert spec["primary_metric"] == "inverse_mae"
    assert "adapter_api_version" not in spec
    assert "trusted_score_samples" in prepared.path.read_text()
    write_candidate_predictions(workspace)
    result = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv", timeout=30,
    )
    assert result.ok, result.output + str(result.error)
    assert result.metric == pytest.approx(0.5)


@pytest.mark.parametrize("declaration,reason", [
    ({"secondary_metrics": ["ssim"]}, "secondary metrics"),
    ({"maximize": False}, "maximize"),
    ({"aggregation": "mean_groups"}, "aggregation"),
])
def test_custom_preparation_cannot_drop_manifest_protocol(run_inputs, monkeypatch, declaration, reason):
    workspace, private = run_inputs
    path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metric_contract"] = {
        "primary_metric": "inverse_mae", "secondary_metrics": [], "maximize": True,
        **declaration,
    }
    write_json(path, manifest)
    backend_responses(monkeypatch, CUSTOM_EVALUATOR)
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match=reason):
        evaluator_setup.prepare_evaluator(
            workspace, workspace.parent / "logs", "Primary evaluation metric: inverse_mae.",
            model="mock-custom-model", attempts=1, timeout=30,
            hidden_score_path=private / "score_val.csv",
        )
    assert not (workspace / "evaluator_prediction_spec.json").exists()


@pytest.mark.parametrize("alias", ["decision_threshold", "classification_threshold"])
def test_freeze_preserves_classification_threshold_alias(run_inputs, alias):
    workspace, private = run_inputs
    path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["task_heads"] = [{"type": "classification"}]
    manifest["metric_contract"] = {
        "primary_metric": "accuracy", "secondary_metrics": [], "maximize": True,
        alias: 0.7,
    }
    write_json(path, manifest)
    spec = {
        key: value for key, value in SPEC.items()
        if key not in {"restoration_evaluation", "ssim_parameters"}
    }
    spec.update(primary_metric="accuracy", secondary_metrics=[], task_type="binary", class_labels=[0, 1])
    spec[alias] = 0.7
    source = adapter_source(spec).replace(
        "np.asarray(json.loads(target_row['reference_json']), dtype=float)",
        "json.loads(target_row['reference_json'])",
    )
    expected = pd.read_csv(workspace / "input/biomaker_splits/score_val.csv", dtype=str)
    write_csv(private / "score_val.csv", [
        {"sample": key, "reference_json": "0"} for key in expected["sample"]
    ])
    write_csv(workspace / "working/validation_predictions.csv", [
        {"sample": key, "result_json": "[0.4, 0.6]"} for key in expected["sample"]
    ])
    evaluator_setup._freeze_data_adapter(
        workspace, source, "Primary evaluation metric: accuracy.", private / "score_val.csv",
    )
    code = (
        "import json, evaluator\n"
        "print(json.dumps(evaluator.prediction_spec()))\n"
        f"print(json.dumps(evaluator.score('working/validation_predictions.csv', {str(private / 'score_val.csv')!r}, 'input/biomaker_splits/score_val.csv')))\n"
    )
    result = evaluator_setup._run_python(workspace, code, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    frozen, score = map(json.loads, result.stdout.splitlines())
    assert frozen["threshold"] == 0.7
    assert alias not in frozen
    assert score["metric"] == 1.0  # A silently substituted threshold of 0.5 gives zero.


@pytest.mark.parametrize("alias", ["decision_threshold", "classification_threshold"])
def test_builtin_classification_prepares_common_runtime_without_generation(run_inputs, monkeypatch, alias):
    workspace, private = run_inputs
    split_dir = workspace / "input/biomaker_splits"
    answers = {name: {"answer_columns": ["outcome"]} for name in ("score_val", "test")}
    write_json(private / "private_gt.json", answers)
    write_json(split_dir / "split_plan.json", {
        "private_gt": answers, "primary_metric": "accuracy", "task_type": "binary",
    })
    write_csv(split_dir / "agent_train.csv", [
        {"sample": "train-no", "outcome": 0}, {"sample": "train-yes", "outcome": 1},
    ])
    write_json(split_dir / "split_manifest.json", {
        "key_columns": ["sample"], "task_heads": [{"type": "classification"}],
        "prediction_contract": {
            "prediction_columns": ["score1"], "prediction_type": "probability_columns",
            "class_labels": [0, 1], "positive_class": 1,
        },
        "metric_contract": {
            "primary_metric": "accuracy", "secondary_metrics": ["auc"],
            "maximize": True, alias: 0.7,
        },
    })
    labels = {}
    for split, hidden in (("score_val", "score_val"), ("agent_test", "test")):
        frame = pd.read_csv(split_dir / (split + ".csv"), dtype=str)
        labels[split] = {key: index % 2 for index, key in enumerate(frame["sample"])}
        write_csv(private / (hidden + ".csv"), [
            {"sample": key, "outcome": value} for key, value in reversed(list(labels[split].items()))
        ])
    prompts = backend_responses(monkeypatch)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, workspace.parent / "logs", "Primary evaluation metric: accuracy.\nSecondary evaluation metric: AUC.",
        model="must-not-call", attempts=1, timeout=30,
        hidden_score_path=private / "score_val.csv",
    )
    spec = json.loads(prepared.prediction_spec)
    assert not prompts
    assert spec["adapter_api_version"] == 1
    assert spec["scoring_backend"] == "trusted_metrics_v1"
    assert spec["maximize"] is True
    assert spec["threshold"] == 0.7 and alias not in spec
    assert spec["prediction_columns"] == ["score1"]
    assert spec["class_labels"] == [0, 1]
    assert "answer_columns" not in spec
    for split, artifact in (("score_val", "validation_predictions"), ("agent_test", "submission")):
        write_csv(workspace / "working" / (artifact + ".csv"), [
            {"sample": key, "score1": 0.8 if value else 0.6}
            for key, value in reversed(list(labels[split].items()))
        ])
    validation = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv", timeout=30,
    )
    assert validation.ok, validation.output + str(validation.error)
    assert validation.details["score"]["metrics"] == {"accuracy": 1.0, "auc": 1.0}
    test = evaluate_private_outputs(workspace, private)
    assert test is not None and "error" not in test, test
    assert test["results"][0]["metrics"] == {"accuracy": 1.0, "auc": 1.0}
    assert not prompts


def test_builtin_restoration_prepares_bounded_common_runtime_without_generation(tmp_path, monkeypatch):
    workspace, private = tmp_path / "workspace", tmp_path / "private_gt"
    split_dir = workspace / "input/biomaker_splits"
    frames = workspace / "input/sensor_arrays"
    frames.mkdir(parents=True)
    private.mkdir()
    np.save(frames / "training_reference.npy", np.full((8, 8), 20.0))
    np.save(private / "validation_reference.npy", np.full((8, 8), 20.0))
    np.save(private / "test_reference.npy", np.full((8, 8), 19.0))
    for index in range(6):
        np.save(frames / f"sensor_{index}.npy", np.full((8, 8), 20.0 + index))
    for split, hidden, reference in (("score_val", "score_val", "validation_reference.npy"),
                                     ("agent_test", "test", "test_reference.npy")):
        rows = [
            {"case": f"{hidden}-{index}", "sensor_capture": f"input/sensor_arrays/sensor_{index}.npy"}
            for index in range(6)
        ]
        write_csv(split_dir / (split + ".csv"), rows)
        write_csv(private / (hidden + ".csv"), [
            {"case": row["case"], "reference_file": str(private / reference)}
            for row in reversed(rows)
        ])
    write_csv(split_dir / "agent_train.csv", [{
        "case": "training", "sensor_capture": "input/sensor_arrays/sensor_0.npy",
        "reference_file": "input/sensor_arrays/training_reference.npy",
    }])
    answers = {name: {"answer_columns": ["reference_file"]} for name in ("score_val", "test")}
    write_json(private / "private_gt.json", answers)
    write_json(split_dir / "split_plan.json", {"private_gt": answers, "primary_metric": "psnr"})
    write_json(split_dir / "split_manifest.json", {
        "key_columns": ["case"], "task_heads": [{"type": "image_restoration"}],
        "prediction_contract": {"prediction_columns": ["estimate_file"], "file_reference_columns": ["estimate_file"]},
        "metric_contract": {
            "primary_metric": "psnr", "secondary_metrics": ["ssim"], "maximize": True,
            "restoration_evaluation": {
                **RESTORATION, "input_column": "sensor_capture", "target_column": "reference_file",
            },
            "ssim_parameters": {"window_size": 7},
        },
    })
    prompts = backend_responses(monkeypatch)
    probe_counts = []
    real_self_test = evaluator_setup._run_self_test

    def observe_real_probe(*args, **kwargs):
        output = real_self_test(*args, **kwargs)
        for line in output.splitlines():
            if line.startswith("EVALUATOR_PROBE_JSON="):
                probe_counts.append(json.loads(line.split("=", 1)[1])["score"]["n"])
        return output

    monkeypatch.setattr(evaluator_setup, "_run_self_test", observe_real_probe)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", TASK, model="must-not-call", attempts=1,
        hidden_score_path=private / "score_val.csv", timeout=30,
    )
    spec = json.loads(prepared.prediction_spec)
    assert not prompts
    assert probe_counts == [4]  # Preparation scores a small subset of the six rows.
    assert spec["adapter_api_version"] == 1
    assert spec["scoring_backend"] == "trusted_metrics_v1"
    assert spec["ssim_parameters"]["win_size"] == 7
    assert "window_size" not in spec["ssim_parameters"]
    assert spec["prediction_columns"] == ["estimate_file"]
    assert spec["restoration_evaluation"]["input_column"] == "sensor_capture"
    assert not (workspace / "working/.adapter_probe").exists()
    frozen = prepared.path.read_bytes()
    np.save(workspace / "working/estimate.npy", np.full((8, 8), 21.0))
    for split, artifact in (("score_val", "validation_predictions"), ("agent_test", "submission")):
        frame = pd.read_csv(split_dir / (split + ".csv"), dtype=str)
        write_csv(workspace / "working" / (artifact + ".csv"), [
            {"case": key, "estimate_file": "working/estimate.npy"} for key in reversed(frame["case"].tolist())
        ])
    validation = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv", timeout=30,
    )
    assert validation.ok, validation.output + str(validation.error)
    assert validation.metric == pytest.approx(20 * np.log10(255))
    assert validation.details["score"]["n"] == 6
    test = evaluate_private_outputs(workspace, private)
    assert test is not None and "error" not in test, test
    assert test["results"][0]["value"] == pytest.approx(20 * np.log10(255 / 2))
    assert test["results"][0]["n"] == 6
    assert prepared.path.read_bytes() == frozen
    assert not prompts
