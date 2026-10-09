"""Metric names are a shared semantic contract, admitted before candidate search."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from biomaker.utils import agentic_data_planner as planner
from biomaker.utils import evaluator_setup as evaluator
from biomaker.utils.experiment_contract import load_experiment_contract
from biomaker.utils.metric_registry import (
    METRIC_CONTRACT_REQUIREMENTS,
    canonical_metric_name,
    metric_info,
    require_supported_metric,
)
from biomaker.utils.prediction_contract import primary_metric_from_spec


def _spec(metric="auc", **changes):
    return {
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "key_columns": ["id"],
        "prediction_columns": ["prob_0", "prob_1"],
        "primary_metric": metric,
        "task_type": "multiclass",
        "maximize": True,
        **changes,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "name",
    [
        "auc",
        "AUROC",
        "macro_auroc",
        "macro_one_vs_rest_auroc",
        "macro_ovr_roc_auc",
        "macro_roc_auc_ovr",
        "Macro one-vs-rest AUROC",
        "macro_auroc_one_versus_rest",
    ],
)
def test_equivalent_macro_ovr_names_have_one_canonical_contract(name):
    assert canonical_metric_name(name) == "auc"
    assert metric_info(name) == metric_info("auc")
    assert primary_metric_from_spec({"primary_metric": name}) == "auc"
    assert (
        primary_metric_from_spec({"test_submission": {"primary_metric": name}}) == "auc"
    )
    normalized = json.loads(
        evaluator._normalize_prediction_spec_json(json.dumps(_spec(name)))
    )
    assert normalized["primary_metric"] == "auc"


@pytest.mark.parametrize(
    "name",
    [
        "micro_auc",
        "weighted_auroc",
        "macro_ovo_auroc",
        "macro_one_vs_one_auroc",
        "macro_one_vs_rest_weighted_auroc",
        "auc_pr",
        "balanced_accuracy",
        "top_5_accuracy",
        "custom_auroc",
    ],
)
def test_non_equivalent_or_unknown_metrics_never_become_auc_or_accuracy(name):
    assert canonical_metric_name(name) == name
    assert metric_info(name) is None
    assert (
        evaluator._primary_metric_from_task_desc(
            f"Primary validation metric: {name}", "multilabel"
        )
        == name
    )
    with pytest.raises(
        evaluator.EvaluatorSetupError, match="Unsupported primary metric"
    ):
        evaluator._normalize_prediction_spec_json(json.dumps(_spec(name)))


@pytest.mark.parametrize(
    "metric,task",
    [
        ("macro_one_vs_rest_auroc", "multilabel"),
        ("mean_roc_auc", "multiclass"),
        ("accuracy", "multilabel"),
        ("auc", "segmentation"),
        ("binary_dice", "classification"),
        ("accuracy", "regression"),
    ],
)
def test_metric_task_mismatch_is_rejected_before_freezing(metric, task):
    with pytest.raises(
        evaluator.EvaluatorSetupError, match="unsupported for task_type"
    ):
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec(metric, task_type=task))
        )


@pytest.mark.parametrize(
    "metric,task",
    [
        ("auc", "binary"),
        ("accuracy", "classification"),
        ("mean_roc_auc", "multilabel"),
        ("mean_binary_accuracy", "multilabel"),
        ("binary_dice", "segmentation_3d"),
        ("aji", "segmentation_2d"),
        ("hd95", "segmentation"),
    ],
)
def test_existing_supported_task_semantics_remain_explicit(metric, task):
    info = require_supported_metric(metric, task)
    normalized = json.loads(
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec(metric, task_type=task, maximize=info.maximize))
        )
    )
    assert normalized["primary_metric"] == metric
    assert normalized["maximize"] == info.maximize


@pytest.mark.parametrize("source", ["spec", "nested_spec", "manifest", "plan"])
def test_historical_contract_readers_share_canonical_names(tmp_path, source):
    metric = "macro_one_vs_rest_auroc"
    split = tmp_path / "input/biomaker_splits"
    if source == "spec":
        _write_json(tmp_path / "evaluator_prediction_spec.json", _spec(metric))
    elif source == "nested_spec":
        _write_json(
            tmp_path / "evaluator_prediction_spec.json",
            {"test_submission": {"primary_metric": metric}},
        )
    elif source == "manifest":
        _write_json(
            split / "split_manifest.json",
            {"metric_contract": {"primary_metric": metric}},
        )
    else:
        _write_json(split / "split_plan.json", {"primary_metric": metric})
    contract = load_experiment_contract(tmp_path)
    assert contract.prediction.primary_metric == contract.metric.primary_metric == "auc"


def test_unknown_saved_metric_is_not_replaced_by_manifest_default(tmp_path):
    _write_json(tmp_path / "evaluator_prediction_spec.json", _spec("micro_auroc"))
    _write_json(
        tmp_path / "input/biomaker_splits/split_manifest.json",
        {"metric_contract": {"primary_metric": "accuracy"}},
    )
    assert load_experiment_contract(tmp_path).metric.primary_metric == "micro_auroc"


def test_generated_metric_must_match_manifest_task_and_requested_objective(tmp_path):
    path = tmp_path / "input/biomaker_splits/split_manifest.json"
    _write_json(
        path,
        {
            "task_heads": ["classification"],
            "metric_contract": {"primary_metric": "macro_one_vs_rest_auroc"},
        },
    )
    assert (
        json.loads(
            evaluator._normalize_prediction_spec_json(
                json.dumps(_spec()), workspace_dir=tmp_path
            )
        )["primary_metric"]
        == "auc"
    )
    with pytest.raises(evaluator.EvaluatorSetupError, match="differs from requested"):
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec("accuracy")), workspace_dir=tmp_path
        )
    with pytest.raises(
        evaluator.EvaluatorSetupError, match="Unsupported primary metric"
    ):
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec()), workspace_dir=tmp_path, requested_metric="micro_auc"
        )
    _write_json(
        path,
        {"task_heads": ["multilabel"], "metric_contract": {"primary_metric": "auc"}},
    )
    with pytest.raises(
        evaluator.EvaluatorSetupError, match="unsupported for task_type"
    ):
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec()), workspace_dir=tmp_path
        )


def test_primary_metric_direction_must_match_canonical_semantics():
    with pytest.raises(evaluator.EvaluatorSetupError, match="maximize disagrees"):
        evaluator._normalize_prediction_spec_json(json.dumps(_spec(maximize=False)))


def _stub_spec_process(monkeypatch):
    # Admission/retry tests inspect the real generated file while isolating the
    # process-lifecycle layer, which has separate integration tests. The tiny
    # fixture module contains only a literal return value.
    def load(workspace):
        tree = ast.parse((workspace / "evaluator.py").read_text())
        return json.dumps(ast.literal_eval(tree.body[0].body[0].value))

    monkeypatch.setattr(evaluator, "_load_prediction_spec", load)


def _evaluator_code(spec):
    return (
        "def prediction_spec():\n    return "
        + repr(spec)
        + "\ndef self_test():\n    print('EVALUATOR_OK')\n"
    )


def test_generated_unsupported_metric_enters_retry_before_self_test_or_freeze(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queries, admitted = [], []
    _stub_spec_process(monkeypatch)
    metrics = iter(["micro_auroc", "macro_one_vs_rest_auroc"])

    def query(**kwargs):
        queries.append(kwargs["system_message"])
        return "```python\n" + _evaluator_code(_spec(next(metrics))) + "```"

    monkeypatch.setattr(evaluator.backend, "query", query)
    monkeypatch.setattr(
        evaluator, "_write_fixed_classification_evaluator", lambda **_: None
    )
    monkeypatch.setattr(
        evaluator, "_write_fixed_segmentation_evaluator", lambda **_: None
    )
    monkeypatch.setattr(
        evaluator, "_run_self_test", lambda *_, **__: admitted.append("self_test")
    )
    monkeypatch.setattr(
        evaluator,
        "_run_public_artifact_contract_test",
        lambda *_, **__: admitted.append("public_contract"),
    )
    prepared = evaluator.prepare_evaluator(
        workspace,
        tmp_path / "logs",
        "Primary validation metric: auc",
        "synthetic",
        attempts=2,
    )
    assert len(queries) == 2
    assert "Unsupported primary metric" in queries[1]["Previous evaluator errors"]
    assert admitted == ["self_test", "public_contract"]
    assert json.loads(prepared.prediction_spec)["primary_metric"] == "auc"
    assert (
        json.loads((workspace / "evaluator_prediction_spec.json").read_text())[
            "primary_metric"
        ]
        == "auc"
    )
    assert "micro_auroc" not in prepared.path.read_text()


@pytest.mark.parametrize(
    "cached_metric,actual_metric,accepted",
    [
        ("micro_auroc", "micro_auroc", False),
        ("auc", "accuracy", False),
        ("auc", "macro_one_vs_rest_auroc", True),
    ],
)
def test_cached_evaluator_is_readmitted_against_its_actual_contract(
    tmp_path, monkeypatch, cached_metric, actual_metric, accepted
):
    cache, workspace, logs = (
        tmp_path / name for name in ["cache", "workspace", "logs"]
    )
    cache.mkdir()
    workspace.mkdir()
    _stub_spec_process(monkeypatch)
    _write_json(
        cache / "manifest.json", {"cache_version": evaluator._EVALUATOR_CACHE_VERSION}
    )
    _write_json(cache / "evaluator_prediction_spec.json", _spec(cached_metric))
    (cache / "evaluator.py").write_text(_evaluator_code(_spec(actual_metric)))
    calls = []
    monkeypatch.setattr(
        evaluator, "_run_self_test", lambda *_, **__: calls.append("self_test")
    )
    monkeypatch.setattr(
        evaluator,
        "_run_public_artifact_contract_test",
        lambda *_, **__: calls.append("public_contract"),
    )
    result = evaluator._try_reuse_cached_evaluator(
        cache, workspace, logs, 30, tmp_path / "hidden.csv"
    )
    assert (result is not None) is accepted
    assert calls == (["self_test", "public_contract"] if accepted else [])


def test_metric_semantics_and_separate_manifest_participate_in_cache_identity(tmp_path):
    split = tmp_path / "input/biomaker_splits/split_manifest.json"
    _write_json(split, {"metric_contract": {"primary_metric": "auc"}})
    before = evaluator._stable_evaluator_cache_key({}, "synthetic", tmp_path, None)
    _write_json(split, {"metric_contract": {"primary_metric": "accuracy"}})
    assert before != evaluator._stable_evaluator_cache_key(
        {}, "synthetic", tmp_path, None
    )
    metadata = evaluator._evaluator_cache_metadata({}, "synthetic", tmp_path, None)
    assert all(
        metadata["code_fingerprints"][name]
        for name in [
            "metric_registry.py",
            "prediction_contract.py",
            "experiment_contract.py",
        ]
    )
    assert metadata["cache_version"] == evaluator._EVALUATOR_CACHE_VERSION


def test_evaluator_and_builder_prompts_require_supported_canonical_metrics(tmp_path):
    assert (
        METRIC_CONTRACT_REQUIREMENTS
        in evaluator._evaluator_prompt("task", tmp_path)["Evaluator requirements"]
    )
    # The builder's source field is the same shared requirement, so planner and
    # evaluator cannot independently invent different metric naming contracts.
    assert (
        METRIC_CONTRACT_REQUIREMENTS
        in planner._builder_user_prompt("task", {})[
            "Required split_manifest.json Fields"
        ]
    )


@pytest.mark.parametrize("task", ["volume_segmentation", "medical_volume_segmentation"])
def test_existing_volume_task_names_remain_supported(task):
    normalized = evaluator._normalize_prediction_spec_json(
        json.dumps(_spec("binary_dice", task_type=task))
    )
    assert json.loads(normalized)["primary_metric"] == "binary_dice"


@pytest.mark.parametrize("task", ["multilabel", "multi_label_classification", None])
def test_precise_multilabel_manifest_takes_priority_over_class_label_hint(
    tmp_path, task
):
    _write_json(
        tmp_path / "input/biomaker_splits/split_manifest.json",
        {
            "task_heads": ["multi_label_classification"],
            "metric_contract": {"primary_metric": "mean_roc_auc"},
        },
    )
    value = _spec("mean_roc_auc", task_type=task, class_labels=["a", "b"])
    assert (
        json.loads(
            evaluator._normalize_prediction_spec_json(
                json.dumps(value), workspace_dir=tmp_path
            )
        )["primary_metric"]
        == "mean_roc_auc"
    )


def test_generic_classification_manifest_allows_explicit_multilabel_subtype(tmp_path):
    _write_json(
        tmp_path / "input/biomaker_splits/split_manifest.json",
        {
            "task_heads": ["classification"],
            "metric_contract": {"primary_metric": "mean_roc_auc"},
        },
    )
    value = _spec("mean_roc_auc", task_type="multilabel")
    assert (
        json.loads(
            evaluator._normalize_prediction_spec_json(
                json.dumps(value), workspace_dir=tmp_path
            )
        )["primary_metric"]
        == "mean_roc_auc"
    )


@pytest.mark.parametrize("source", ["manifest", "plan"])
@pytest.mark.parametrize("direction", [False, "true", 1])
def test_split_direction_cannot_disagree_with_admitted_metric(
    tmp_path, source, direction
):
    split = tmp_path / "input/biomaker_splits"
    content = {"primary_metric": "auc", "maximize": direction}
    _write_json(
        split / ("split_manifest.json" if source == "manifest" else "split_plan.json"),
        {"metric_contract": content} if source == "manifest" else content,
    )
    with pytest.raises(
        evaluator.EvaluatorSetupError, match="Split metric contract maximize"
    ):
        evaluator._normalize_prediction_spec_json(
            json.dumps(_spec()), workspace_dir=tmp_path
        )


def test_fixed_segmentation_never_ignores_explicit_accuracy_request(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        evaluator,
        "_infer_fixed_segmentation_spec",
        lambda *_: _spec("binary_dice", task_type="segmentation"),
    )
    monkeypatch.setattr(
        evaluator,
        "_write_fixed_evaluator",
        lambda **_: pytest.fail(
            "Must reject the incompatible fixed scorer before freezing"
        ),
    )
    assert (
        evaluator._write_fixed_segmentation_evaluator(
            tmp_path,
            tmp_path / "logs",
            tmp_path / "hidden.csv",
            30,
            task_desc="Primary validation metric: accuracy",
        )
        is None
    )


def test_admitted_hd95_direction_is_explicit_even_when_generator_omits_it():
    value = _spec("hd95", task_type="segmentation")
    value.pop("maximize")
    normalized = json.loads(
        evaluator._normalize_prediction_spec_json(json.dumps(value))
    )
    assert normalized["maximize"] is False


def test_metric_name_alone_does_not_establish_task_compatibility():
    value = _spec("binary_dice")
    value.pop("task_type")
    with pytest.raises(evaluator.EvaluatorSetupError, match="must declare task_type"):
        evaluator._normalize_prediction_spec_json(json.dumps(value))
