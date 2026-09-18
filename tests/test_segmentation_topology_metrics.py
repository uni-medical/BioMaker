"""Independent ATM26 references and canonical volume adapter contracts.

Golden values below were checked against the official evaluator at commit
6af4a24529977413a8f5216a0eeb2a8ee8a1e1d3, using NumPy 1.26.4, SciPy 1.17.1
and scikit-image 0.24.0 (voxel references rechecked with 0.22.0). Sources:
evaluation/T1_evaluation/metrics.py and
evaluation/T2_evaluation/{metrics,voxel_metrics,projection,graph_utils}.py in
https://github.com/EndoluminalSurgicalVision-IMR/Airway-Tree-Modeling-26 .
They follow evaluate.py's raw-vs-largest-component call order, including its
class-20 behavior (which differs from an obsolete voxel_metrics docstring).
The fixture and constants are self-contained: tests never fetch upstream,
read a temporary checkout, or use the implementation under test as an oracle.
"""

import builtins
import copy
import weakref

import numpy as np
import pytest

from biomaker.utils.segmentation_metrics import segmentation_score_case, validate_tree_sample
from biomaker.utils.trusted_metrics import (
    metric_capabilities,
    normalize_metric_spec,
    score_samples,
    validate_prediction,
)


T1 = ["binary_dice", "cldice", "tree_length_detected", "branch_detected", "betti_error"]
T2 = ["branch_accuracy", "branch_f1", "subtree_consistency", "topological_distance",
      "multiclass_dice", "multiclass_cldice"]
OFFICIAL = dict(dice_smooth=1e-5, topology_smooth=1e-5,
                branch_coverage_threshold=0.8, topology_component_policy="largest",
                connectivity=3, betti_reference="one", subtree_ignore_labels=[19],
                missing_label_penalty=50.0, label_ids=list(range(21)))


def spec(names, protocol=None):
    return dict(OFFICIAL if protocol is None else protocol,
                metrics=[{"name": name, "maximize": name not in ("topological_distance", "betti_error")}
                         for name in names])


def case_scores(sample, names, protocol=None):
    return score_samples([sample], spec(names, protocol))["metrics"]


def validate(sample, names, require_target=True):
    return validate_tree_sample(sample, names, spec=OFFICIAL, require_target=require_target)


def anatomical_line():
    """Three unequal-size branches and one disconnected false-positive voxel.

    GT node labels are [19, 1, 20]; majority voting gives [19, 1, 1]. The
    third branch's two-versus-two tie selects label 1. Only the graph path
    removes the isolated false positive: voxel metrics must retain it.
    """
    target = np.zeros((5, 5, 12), dtype=np.uint8)
    target[2, 2, 1:3] = 19
    target[2, 2, 3:7] = 1
    target[2, 2, 7:11] = 20
    prediction = target.copy()
    prediction[2, 2, 5:7] = 20
    prediction[2, 2, 9:11] = 1
    prediction[0, 0, 0] = 20
    parsing = np.zeros_like(target)
    parsing[2, 2, 1:3] = 1
    parsing[2, 2, 3:7] = 2
    parsing[2, 2, 7:11] = 3
    return dict(prediction=prediction, target=target, parsing=parsing,
                skeleton=(target > 0).astype(np.uint8),
                graph=np.array([[0, 1], [1, 2]], dtype=np.int64))


@pytest.mark.parametrize("names,golden", [
    (T1, [0.9523809750566785, 0.9523809523809523, 1.0, 1.0, 1.0]),
    (T2, [2 / 3, 5 / 9, 1.0, 1 / 3,
          0.6481485622423092, 0.6481481481481481]),
])
def test_official_golden_all_metrics_and_registration(names, golden):
    sample = anatomical_line()
    expected = dict(zip(names, golden))
    generic_names = [name for name in names if name != "binary_dice"]
    assert segmentation_score_case(sample, generic_names, OFFICIAL) == pytest.approx(
        {name: expected[name] for name in generic_names}, abs=1e-12)
    assert score_samples(iter([sample]), spec(names))["metrics"] == pytest.approx(expected, abs=1e-12)
    registry = metric_capabilities()["metrics"]
    for name in names:
        assert registry[name]["family"] == "segmentation"
        assert registry[name]["maximize"] is (name not in ("topological_distance", "betti_error"))


def test_t1_largest_component_branch_threshold_and_raw_dice():
    target = np.zeros((5, 5, 17), dtype=np.uint8)
    target[2, 2, 1:6] = target[2, 2, 10:15] = 1
    prediction = np.zeros_like(target)
    prediction[2, 2, 1:5] = prediction[2, 2, 10:14] = 1
    parsing = target.copy()
    parsing[2, 2, 10:15] = 2
    got = case_scores(dict(prediction=prediction, target=target,
                                 parsing=parsing, skeleton=target), T1)
    # Two size-four components tie: retain the first, exactly 80% of branch 1.
    # Dice/clDice still measure both components, whereas TLD/BD use only one.
    assert got == pytest.approx(dict(zip(T1, [0.8888889506172497, 8 / 9,
                                             0.40000059999939996, 0.5, 1.0])), abs=1e-12)
    prediction[2, 2, 4] = 0
    prediction[2, 2, 13] = 0
    assert case_scores(dict(prediction=prediction, target=target,
                                  parsing=parsing, skeleton=target), [T1[3]])[T1[3]] == 0


def test_t1_betti_uses_26_connectivity_and_fixed_one_not_gt_components():
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction[1, 1, 1] = prediction[2, 2, 2] = 1
    target = np.zeros_like(prediction)
    target[0, 0, 0] = target[4, 4, 4] = 1
    name = T1[-1]
    assert case_scores(dict(prediction=prediction, target=target), [name]) == {name: 0.0}
    assert case_scores(dict(prediction=np.zeros_like(target), target=target), [name]) == {name: 1.0}


def test_empty_masks_keep_track_specific_conventions():
    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    sample = dict(prediction=empty, target=empty, skeleton=empty, parsing=empty)
    # Official T1 smooths Dice and TLD but not clDice; fixed-one Betti stays 1.
    assert case_scores(sample, T1) == dict(zip(T1, [1.0, 0.0, 1.0, 0.0, 1.0]))
    assert case_scores(sample, [T2[4], T2[5]]) == {T2[4]: 0.0, T2[5]: 0.0}
    sample["target"] = empty.copy()
    sample["target"][1, 1, 1] = 20
    smooth_score = 1e-5 / (1 + 1e-5)
    assert case_scores(sample, [T1[0]])[T1[0]] == pytest.approx(smooth_score, abs=1e-15)
    assert case_scores(sample, [T2[4]])[T2[4]] == pytest.approx(smooth_score, abs=1e-15)


def test_t2_empty_prediction_background_consistency_and_missing_label_distance():
    sample = anatomical_line()
    sample["prediction"] = np.zeros_like(sample["prediction"])
    names = T2[:4]
    # Background is not excluded from SC. With no GT background branch, TD=50.
    assert case_scores(sample, names) == dict(zip(names, [0.0, 0.0, 1.0, 50.0]))


def test_t2_projection_ties_include_background_and_graph_not_needed_for_acc():
    sample = anatomical_line()
    sample["prediction"] = sample["target"].copy()
    sample["prediction"][2, 2, 3:5] = 0
    # Remaining connected foreground (six voxels) is the LCC. Branch 2 has
    # two zeros and two ones; background wins the tie. Root 19 is also lost.
    for key in ("graph", "skeleton"):
        sample.pop(key)
    assert case_scores(sample, [T2[0]]) == {T2[0]: 1 / 3}


def test_t2_cldice_excludes_classes_without_a_reference_skeleton():
    sample = anatomical_line()
    sample["skeleton"][sample["target"] == 20] = 0
    # Only class 19 (1.0) and class 1 (0.5) remain in the topology mean.
    assert case_scores(sample, [T2[5]]) == {T2[5]: 0.75}


def test_t2_td_preserves_finite_distances_greater_than_fifty():
    count = 53
    target = np.zeros((3, 3, count + 2), dtype=np.uint8)
    target[1, 1, 1:-1] = 1
    target[1, 1, -2] = 20
    prediction = (target > 0).astype(np.uint8) * 20
    parsing = np.zeros_like(target, dtype=np.int64)
    parsing[1, 1, 1:-1] = np.arange(1, count + 1)
    graph = np.array([np.arange(count - 1), np.arange(1, count)])
    # Every prediction points at the last GT node: mean(0,1,...,52) = 26.
    got = case_scores(dict(prediction=prediction, target=target,
                                 parsing=parsing, graph=graph), [T2[3]])
    assert got == {T2[3]: 26.0}


def test_branched_tree_distance_uses_nearest_matching_gt_node():
    target = np.zeros((7, 7, 7), dtype=np.uint8)
    parsing = np.zeros_like(target)
    branches = [((3, 3, 1), (3, 3, 2)), ((3, 2, 3), (3, 1, 4)),
                ((3, 4, 3), (3, 5, 4)), ((3, 1, 5), (3, 1, 6))]
    for branch, (coordinates, label) in enumerate(zip(branches, [19, 1, 20, 1]), 1):
        for point in coordinates:
            target[point], parsing[point] = label, branch
    prediction = target.copy()
    prediction[prediction == 20] = 1
    graph = np.array([[0, 0, 1], [1, 2, 3]])
    # Mislabelled node 2 is two hops from GT node 1, three from GT node 3;
    # both carry class 1. The subtree rooted at node 1 remains consistent.
    sample = dict(prediction=prediction, target=target, parsing=parsing, graph=graph)
    assert case_scores(sample, T2[:4]) == pytest.approx(dict(zip(T2[:4], [0.75, 0.6, 1.0, 0.5])))


@pytest.mark.parametrize("name,required", [
    (T1[0], ()), (T1[1], ("skeleton",)), (T1[2], ("skeleton",)),
    (T1[3], ("skeleton", "parsing")), (T1[4], ()),
    (T2[0], ("parsing",)), (T2[1], ("parsing",)),
    (T2[2], ("parsing", "graph")), (T2[3], ("parsing", "graph")),
    (T2[4], ()), (T2[5], ("skeleton",)),
])
def test_only_requested_metrics_require_their_references(name, required):
    sample = anatomical_line()
    sample = {key: value for key, value in sample.items()
              if key in ("prediction", "target") + required}
    assert set(case_scores(sample, [name])) == {name}
    for key in required:
        incomplete = dict(sample)
        incomplete.pop(key)
        with pytest.raises(ValueError, match=key):
            validate(incomplete, [name])


def test_dice_does_not_import_skimage_or_validate_unused_references(monkeypatch):
    original_import = builtins.__import__

    def reject_skimage(name, *args, **kwargs):
        if name.startswith("skimage"):
            raise AssertionError("Dice must not load skeletonization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_skimage)
    sample = anatomical_line()
    sample.update(skeleton="unused", parsing="unused", graph="unused",
                  skeleton_geometry="unused", parsing_geometry="unused")
    assert set(case_scores(sample, [T2[4]])) == {T2[4]}


def test_frozen_metrics_execute_without_installed_biomaker_imports(monkeypatch):
    from biomaker.utils.metric_source import frozen_metric_source

    source = frozen_metric_source()
    original_import = builtins.__import__

    def reject_biomaker(name, *args, **kwargs):
        if name == "biomaker" or name.startswith("biomaker.") or kwargs.get("level", 0):
            raise AssertionError("Frozen evaluator must not import the installed method package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_biomaker)
    namespace = {}
    exec(source, namespace)
    result = namespace["score_samples"]([anatomical_line()], spec(T2))
    expected = dict(zip(T2, [2 / 3, 5 / 9, 1.0, 1 / 3,
                            0.6481485622423092, 0.6481481481481481]))
    assert result["metrics"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("field", ["prediction", "target"])
@pytest.mark.parametrize("value,dtype", [
    (-1, np.int64), (0.5, np.float64), (np.nan, np.float64),
    (np.inf, np.float64), (21, np.int64), (256, np.int64),
    (2 ** 64 - 1, np.uint64), (2 ** 63, np.float64),
])
def test_t2_label_validation_precedes_any_integer_cast(field, value, dtype):
    sample = anatomical_line()
    sample[field] = sample[field].astype(dtype)
    sample[field][0, 0, 0] = value
    with pytest.raises(ValueError):
        validate(sample, [T2[4]])


@pytest.mark.parametrize("shape", [(5, 12), (1, 5, 5, 12), (5, 5, 13), (0, 5, 12)])
def test_segmentation_shape_must_be_nonempty_and_match_target(shape):
    sample = anatomical_line()
    sample["prediction"] = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        validate(sample, [T2[4]])


@pytest.mark.parametrize("edges", [
    np.array([[0, 1, 2]]), np.array([[0, 1], [1, 2]], dtype=float) + 0.5,
    np.array([[0, -1], [1, 2]]), np.array([[0, 1], [1, 3]]),
    np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 0]]),
    np.array([[0, 0], [0, 2]]), np.empty((2, 0), dtype=int),
])
def test_graph_rejects_malformed_edges_before_traversal(edges):
    sample = anatomical_line()
    sample["graph"] = edges
    with pytest.raises(ValueError):
        validate(sample, [T2[2], T2[3]])


def test_graph_cycle_with_plausible_node_and_edge_counts_is_rejected():
    sample = anatomical_line()
    sample["parsing"][2, 2, 9:11] = 4
    # N=4, E=N-1, every node represented, but disconnected and cyclic.
    sample["graph"] = np.array([[0, 1, 2], [1, 0, 3]])
    with pytest.raises(ValueError):
        validate(sample, [T2[2]])


def test_generic_single_branch_tree_has_empty_edges_but_valid_graph_metrics():
    target = np.zeros((3, 3, 5), dtype=np.uint8)
    target[1, 1, 1:4] = 42
    sample = dict(prediction=target.copy(), target=target,
                  parsing=(target > 0).astype(np.uint8),
                  graph=np.empty((2, 0), dtype=np.int64))
    assert case_scores(sample, T2[:4], {}) == dict(zip(T2[:4], [1.0, 1.0, 0.0, 0.0]))


@pytest.mark.parametrize("field,change", [
    ("skeleton", "outside"), ("skeleton", "fractional"),
    ("parsing", "fractional"), ("parsing", "missing_id"),
    ("parsing", "zero_on_skeleton"),
])
def test_reference_arrays_are_validated_without_silent_repair(field, change):
    sample = anatomical_line()
    if change == "outside":
        sample[field][0, 0, 0] = 1
    elif change == "fractional":
        sample[field] = sample[field].astype(float)
        sample[field][2, 2, 1] = 0.5
    elif change == "missing_id":
        sample[field][sample[field] == 2] = 4
    else:
        sample[field][2, 2, 1] = 0
    with pytest.raises(ValueError):
        validate(sample, [T1[3]])


def geometry(shape):
    return dict(size=list(reversed(shape)), spacing=[0.7, 0.8, 1.2],
                origin=[1.0, 2.0, 3.0], direction=np.eye(3).ravel().tolist())


@pytest.mark.parametrize("field", ["size", "spacing", "origin", "direction"])
def test_optional_geometry_catches_original_grid_mismatches(field):
    sample = anatomical_line()
    reference = geometry(sample["target"].shape)
    sample.update({key + "_geometry": copy.deepcopy(reference)
                   for key in ("input", "prediction", "target", "skeleton", "parsing")})
    validate(sample, T1[1:])
    sample["prediction_geometry"][field][0] += 1
    with pytest.raises(ValueError):
        validate(sample, T1[1:])


def test_public_prediction_validation_does_not_read_private_targets_or_references():
    sample = anatomical_line()
    public_geometry = geometry(sample["prediction"].shape)
    sample.update(input_geometry=public_geometry, prediction_geometry=public_geometry)
    for name in ("target", "skeleton", "parsing", "graph"):
        sample[name] = object()
        sample[name + "_geometry"] = object()
    validate(sample, T2, require_target=False)
    assert validate_prediction(sample, spec(T2)) == {"valid": True}


@pytest.mark.parametrize("prediction_only", [False, True])
def test_declared_physical_policy_requires_geometry_from_custom_adapters(prediction_only):
    sample = anatomical_line()
    contract = spec(["binary_dice"], {"geometry_policy": "physical"})
    # Merely omitting all geometry fields must not weaken a frozen physical
    # comparison into a shape-only comparison in an agent-written reader.
    with pytest.raises(ValueError, match="geometry"):
        if prediction_only:
            validate_prediction(sample, contract)
        else:
            score_samples([sample], contract)


def test_equal_case_weighting_not_pooled_voxels_and_streamed_arrays():
    small = np.ones((1, 1, 1), dtype=np.uint8)
    large = np.ones((3, 3, 3), dtype=np.uint8)
    result = score_samples(iter([dict(prediction=small, target=small),
                                 dict(prediction=large * 0, target=large)]), spec([T1[0]]))
    assert result["n"] == 2
    assert result["metrics"][T1[0]] == pytest.approx((1 + 1e-5 / (27 + 1e-5)) / 2)
    references = []

    def volumes():
        for _ in range(12):
            prediction = np.ones((3, 3, 3), dtype=np.uint8)
            target = np.ones_like(prediction)
            references.extend((weakref.ref(prediction), weakref.ref(target)))
            assert sum(ref() is not None for ref in references) <= 4
            yield dict(prediction=prediction, target=target)

    assert score_samples(volumes(), spec([T1[0]]))["n"] == 12


def test_official_direction_is_fixed_and_generic_dice_keeps_existing_semantics():
    for name in (T1[-1], T2[3]):
        with pytest.raises(ValueError):
            normalize_metric_spec({"metrics": [{"name": name, "maximize": True}]})
    sample = dict(prediction=np.zeros((1, 1, 1), dtype=np.uint8),
                  target=np.ones((1, 1, 1), dtype=np.uint8))
    assert score_samples([sample], {"primary_metric": "dice"})["metric"] == 0.0
    assert score_samples([sample], spec([T1[0]]))["metrics"][T1[0]] > 0.0


def test_legacy_optional_input_shape_none_remains_valid():
    sample = dict(prediction=np.ones((2, 2), dtype=np.uint8),
                  target=np.ones((2, 2), dtype=np.uint8), input_shape=None)
    assert score_samples([sample], {"primary_metric": "binary_dice"})["metric"] == 1.0


def test_generic_multiclass_metrics_accept_labels_outside_airway_taxonomy():
    target = np.zeros((3, 3, 3), dtype=np.uint16)
    target[1, 1, 1] = 42
    sample = dict(prediction=target.copy(), target=target, skeleton=target > 0)
    names = ["multiclass_dice", "multiclass_cldice"]
    assert case_scores(sample, names, {}) == dict.fromkeys(names, 1.0)
    restricted = {"label_ids": [0, 1, 20]}
    with pytest.raises(ValueError):
        case_scores(sample, names, restricted)


def test_generic_metrics_accept_matching_two_dimensional_masks():
    target = np.zeros((5, 7), dtype=np.uint8)
    target[2, 1:6] = 42
    sample = dict(prediction=target.copy(), target=target, skeleton=target > 0)
    names = ["multiclass_dice", "multiclass_cldice", "cldice", "tree_length_detected"]
    assert case_scores(sample, names, {}) == dict.fromkeys(names, 1.0)


def test_scoring_does_not_modify_prediction_or_reference_arrays():
    sample = anatomical_line()
    before = {key: value.copy() for key, value in sample.items()}
    for value in sample.values():
        value.flags.writeable = False
    case_scores(sample, T2)
    for key, value in sample.items():
        assert np.array_equal(value, before[key])


def test_generic_betti_reference_is_target_unless_explicitly_fixed_one():
    target = np.zeros((5, 5, 5), dtype=np.uint8)
    target[0, 0, 0] = target[4, 4, 4] = 1
    sample = dict(prediction=target.copy(), target=target)
    assert case_scores(sample, ["betti_error"], {}) == {"betti_error": 0.0}
    assert case_scores(sample, ["betti_error"], {"betti_reference": "one"}) == {"betti_error": 1.0}


def test_generic_subtree_consistency_has_no_special_label_nineteen():
    sample = anatomical_line()
    # Projected classes [19, 1, 1]. Generic SC counts the inconsistent root
    # and the consistent child subtree; explicitly ignoring 19 gives 1.
    assert case_scores(sample, ["subtree_consistency"], {}) == {"subtree_consistency": 0.5}
    assert case_scores(sample, ["subtree_consistency"], {"subtree_ignore_labels": [19]}) == {
        "subtree_consistency": 1.0}


def test_generic_component_policy_connectivity_threshold_and_missing_penalty():
    sample = anatomical_line()
    sample["prediction"] = np.zeros_like(sample["prediction"])
    assert case_scores(sample, ["topological_distance"], {"missing_label_penalty": 7}) == {
        "topological_distance": 7.0}
    target = np.zeros((5, 5, 5), dtype=np.uint8)
    target[1, 1, 1] = target[2, 2, 2] = 1
    sample = dict(prediction=target.copy(), target=target,
                  skeleton=target.copy(), parsing=target.copy())
    assert case_scores(sample, ["tree_length_detected"], {"connectivity": 1}) == {
        "tree_length_detected": 0.5}
    assert case_scores(sample, ["tree_length_detected"], {"connectivity": 3}) == {
        "tree_length_detected": 1.0}
    assert case_scores(sample, ["tree_length_detected"], {
        "connectivity": 1, "topology_component_policy": "raw"}) == {"tree_length_detected": 1.0}
    sample["prediction"][2, 2, 2] = 0
    assert case_scores(sample, ["branch_detected"], {"branch_coverage_threshold": 0.5}) == {
        "branch_detected": 1.0}
    assert case_scores(sample, ["branch_detected"], {"branch_coverage_threshold": 0.8}) == {
        "branch_detected": 0.0}


@pytest.mark.parametrize("protocol", [
    {"dice_smooth": -1}, {"dice_smooth": float("nan")},
    {"topology_smooth": -1}, {"topology_smooth": float("inf")},
    {"branch_coverage_threshold": 1.1}, {"branch_coverage_threshold": -0.1},
    {"topology_component_policy": "silently_fill_holes"},
    {"connectivity": 0}, {"connectivity": 4},
    {"betti_reference": "guess"}, {"subtree_ignore_labels": [-1]},
    {"missing_label_penalty": -1}, {"label_ids": [0, 1, 1]},
])
def test_invalid_generic_protocol_is_rejected(protocol):
    with pytest.raises(ValueError):
        normalize_metric_spec(spec(["multiclass_dice"], protocol))
