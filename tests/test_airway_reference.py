"""Small-volume checks of the pinned tree parser, geometry and cache provenance."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from biomaker.utils.airway_reference import (
    ATM22_SOURCE_SHA256,
    TreeReferenceError,
    _SOURCE_PATH,
    _derive_reference,
    _load_upstream,
    _validate_label,
    _validate_tree,
    prepare_tree_reference,
)


@pytest.fixture
def reference_dependencies():
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("nibabel")
    pytest.importorskip("skimage")
    if importlib.metadata.version("scikit-image") != "0.22.0":
        pytest.skip("Pinned ATM22 preparation requires scikit-image==0.22.0")
    return sitk


@pytest.fixture
def branching_label_xyz():
    # Unequal dimensions and an asymmetric bifurcation expose axis swaps.
    skeleton = np.zeros((97, 83, 89), dtype=bool)
    skeleton[48, 41, 5:46] = True
    for i in range(31):
        skeleton[48 - i, 41, 45 + i] = True
        skeleton[48 + i, 41, 45 + i] = True
    label = (ndimage.distance_transform_edt(~skeleton) <= 2).astype(np.uint8)
    # Anatomical classes must survive outside the binary working copy. Class
    # 37 intentionally verifies that this helper is not tied to ATM26's 20.
    label[60:, :, :] *= 37
    return label


def write_label(sitk, path, array_xyz):
    image = sitk.GetImageFromArray(array_xyz.transpose(2, 1, 0))
    image.SetSpacing((0.7, 0.9, 1.3))
    image.SetOrigin((12.5, -23.25, 41.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))
    return image


def test_vendored_original_is_byte_exact():
    assert hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest() == ATM22_SOURCE_SHA256


def test_bifurcation_retains_full_skeleton(reference_dependencies, branching_label_xyz):
    skeleton, parsing, graph, metadata = _derive_reference(branching_label_xyz)
    assert metadata["node_count"] == 3
    assert metadata["edge_count"] == 2
    assert metadata["root_zero_based"] == 1
    np.testing.assert_array_equal(graph, [[1, 1], [0, 2]])
    assert skeleton.dtype == np.uint8
    assert np.count_nonzero(skeleton) == 101
    upstream = _load_upstream()
    pruned, _, _ = upstream.skeleton_parsing(skeleton)
    assert np.count_nonzero(pruned) < np.count_nonzero(skeleton)
    # Full skeleton, including junctions removed only for branch enumeration,
    # remains covered by final parsing and ORIGINAL labels.
    assert np.all(parsing[skeleton > 0] > 0)
    assert np.all(branching_label_xyz[skeleton > 0] > 0)


def test_output_geometry_labels_and_cache_reuse(reference_dependencies, branching_label_xyz, tmp_path):
    sitk = reference_dependencies
    label_path = tmp_path / "label.mha"
    image = write_label(sitk, label_path, branching_label_xyz)
    original_bytes = label_path.read_bytes()
    result = prepare_tree_reference(label_path, tmp_path / "references")
    assert result["include_graph"] is True
    assert result["node_count"] == 3
    for key in ("skeleton_path", "parsing_path"):
        output = sitk.ReadImage(result[key])
        assert output.GetSize() == image.GetSize()
        assert output.GetSpacing() == image.GetSpacing()
        assert output.GetOrigin() == image.GetOrigin()
        assert output.GetDirection() == image.GetDirection()
    skeleton_xyz = sitk.GetArrayFromImage(sitk.ReadImage(result["skeleton_path"])).transpose(2, 1, 0)
    assert np.count_nonzero(skeleton_xyz) == 101
    assert np.all(branching_label_xyz[skeleton_xyz > 0] > 0)
    graph = np.load(result["graph_path"], allow_pickle=False)
    np.testing.assert_array_equal(graph, [[1, 1], [0, 2]])
    paths = [Path(result[key]) for key in ("skeleton_path", "parsing_path", "graph_path", "metadata_path")]
    mtimes = {p: p.stat().st_mtime_ns for p in paths}
    assert prepare_tree_reference(label_path, tmp_path / "references") == result
    assert {p: p.stat().st_mtime_ns for p in paths} == mtimes
    assert label_path.read_bytes() == original_bytes
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert metadata["identity"]["input_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert metadata["identity"]["dependencies"]["scikit-image"] == "0.22.0"
    assert set(metadata["output_sha256"]) == {"skeleton.mha", "parsing.mha", "graph.npy", "preparation.log"}
    assert not list((tmp_path / "references").glob(".tree-*"))


def test_changed_label_gets_new_entry_and_corruption_fails(reference_dependencies, branching_label_xyz, tmp_path):
    sitk = reference_dependencies
    label_path = tmp_path / "label.mha"
    write_label(sitk, label_path, branching_label_xyz)
    result = prepare_tree_reference(label_path, tmp_path / "references", include_graph=False)
    assert result["graph_path"] is None
    changed = branching_label_xyz.copy()
    changed[changed == 37] = 20
    write_label(sitk, label_path, changed)
    new = prepare_tree_reference(label_path, tmp_path / "references", include_graph=False)
    assert new["cache_key"] != result["cache_key"]
    assert Path(result["metadata_path"]).is_file()
    with Path(new["parsing_path"]).open("ab") as file:
        file.write(b"damaged")
    with pytest.raises(TreeReferenceError, match="has changed"):
        prepare_tree_reference(label_path, tmp_path / "references", include_graph=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 0.5])
def test_invalid_labels_rejected_before_integer_cast(value):
    label = np.ones((5, 7, 9), dtype=float)
    label[0, 0, 0] = value
    with pytest.raises(TreeReferenceError, match="finite integer|nonnegative"):
        _validate_label(label)


def test_empty_and_too_short_tree_fail_without_partial_cache(reference_dependencies, tmp_path):
    sitk = reference_dependencies
    label_path = tmp_path / "short.mha"
    empty = np.zeros((9, 11, 13), dtype=np.uint8)
    with pytest.raises(TreeReferenceError, match="empty foreground"):
        _derive_reference(empty)
    empty[4, 5, 5:8] = 1
    write_label(sitk, label_path, empty)
    with pytest.raises(TreeReferenceError, match="removed every"):
        prepare_tree_reference(label_path, tmp_path / "references")
    assert list((tmp_path / "references").iterdir()) == []


def test_single_branch_is_valid_generic_tree(reference_dependencies):
    label = np.zeros((9, 11, 31), dtype=np.uint8)
    label[3:6, 4:7, 3:28] = 1
    _, _, graph, meta = _derive_reference(label)
    assert meta["node_count"] == 1
    assert graph.shape == (2, 0)
    assert graph.dtype == np.int64


def test_refinement_preserves_the_original_full_skeleton(reference_dependencies):
    centerline = np.zeros((97, 83, 89), dtype=bool)
    centerline[48, 41, 5:46] = True
    for i in range(31):
        centerline[48 + i, 41, 45 + i] = True
    for i in range(5):
        centerline[48 - i, 41, 45 + i] = True
    label = (ndimage.distance_transform_edt(~centerline) <= 2).astype(np.uint8)
    skeleton, parsing, graph, meta = _derive_reference(label)
    assert meta["initial_branch_count"] == 2
    assert meta["refinement_iterations"] == 1
    assert meta["node_count"] == 1
    assert np.count_nonzero(skeleton) == 75
    assert np.all(parsing[skeleton > 0] == 1)
    assert graph.shape == (2, 0)


def test_hole_filling_cannot_silently_change_ground_truth(reference_dependencies):
    label = np.zeros((9, 11, 31), dtype=np.uint8)
    label[3:6, 4:7, 3:28] = 1
    label[4, 5, 15] = 0
    with pytest.raises(TreeReferenceError, match="ORIGINAL"):
        _derive_reference(label)


def test_original_label_containment_and_disconnected_graph_rejected():
    label = np.ones((3, 3, 3), dtype=np.uint8)
    skeleton = label.copy()
    parsing = label.copy()
    label[0, 0, 0] = 0
    with pytest.raises(TreeReferenceError, match="ORIGINAL"):
        _validate_tree(skeleton, parsing, np.ones((1, 1)), np.zeros((1, 1)), label, 1)

    # An isolated root plus a disconnected 2-cycle meets the simple N-1 edge
    # count and per-node indegree conditions; reachability must reject it.
    label[:] = 1
    parsing[1] = 2
    parsing[2] = 3
    children = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.uint8)
    parent = children.T.copy()
    parent[0, 0] = 1
    with pytest.raises(TreeReferenceError, match="not fully reachable"):
        _validate_tree(skeleton, parsing, parent, children, label, 3)
