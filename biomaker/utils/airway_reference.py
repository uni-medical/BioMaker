"""Prepare private tree references once, separately from model scoring.

``prepare_tree_reference(label_path, output_dir, include_graph=True)`` returns paths to a
full skeleton, final branch parsing and (when requested) a zero-based directed tree.
The source anatomical label is never modified. Cache entries are content
addressed, provenance checked and published atomically only after validation.

This is the local ``atm22refs-v1`` protocol; it does not assert equivalence to
the unpublished ATM26 reference-generation procedure. Scoring functions must
load these already-prepared references, never generate them during scoring.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

import numpy as np


PROTOCOL_ID = "atm22refs-v1"
ATM22_COMMIT = "522081149dfde905192046bd7237b8ace7819600"
ATM22_SOURCE_SHA256 = "1fce3c18073e03c398b471100f3867d025a33704a7654e6b40d825302c753099"
ATM22_SOURCE_URL = (
    "https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/blob/"
    f"{ATM22_COMMIT}/evaluation/tree_parse.py"
)
_SOURCE_PATH = Path(__file__).parent / "_vendor" / "atm22" / "tree_parse.py"
_DEPENDENCIES = ("numpy", "scipy", "scikit-image", "nibabel", "SimpleITK")


class TreeReferenceError(ValueError):
    """A label or derived reference cannot satisfy the declared protocol."""


def _sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for dependency in _DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"ATM22 reference preparation requires {dependency}. Install the "
                "BioMaker airway evaluation dependencies in a preparation environment."
            ) from exc
    if versions["scikit-image"] != "0.22.0":
        raise RuntimeError(
            "atm22refs-v1 requires scikit-image==0.22.0 (the original "
            "skeletonize_3d API); do not substitute another skeleton algorithm."
        )
    return versions


def _load_upstream() -> Any:
    if _sha256_file(_SOURCE_PATH) != ATM22_SOURCE_SHA256:
        raise RuntimeError("Vendored ATM22 tree_parse.py does not match its pinned SHA256")
    return importlib.import_module("biomaker.utils._vendor.atm22.tree_parse")


def _geometry(image: Any) -> dict[str, Any]:
    components = (
        image.GetNumberOfComponentsPerPixel()
        if hasattr(image, "GetNumberOfComponentsPerPixel")
        else image.GetNumberOfComponents()
    )
    if image.GetDimension() != 3 or components != 1:
        raise TreeReferenceError("Reference labels must be scalar 3D medical volumes")
    result = {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }
    values = result["spacing_xyz"] + result["origin_xyz"] + result["direction"]
    if not np.isfinite(values).all() or any(v <= 0 for v in result["spacing_xyz"]):
        raise TreeReferenceError("Reference label geometry is not finite with positive spacing")
    if any(v <= 0 for v in result["size_xyz"]):
        raise TreeReferenceError("Reference label has an empty spatial dimension")
    return result


def _validate_label(label: np.ndarray) -> None:
    if label.ndim != 3 or label.dtype.kind not in "buif":
        raise TreeReferenceError("Reference labels must be numeric scalar 3D arrays")
    if not np.isfinite(label).all() or not np.equal(label, np.floor(label)).all():
        raise TreeReferenceError("Reference labels must contain finite integer values")
    if np.any(label < 0):
        raise TreeReferenceError("Reference label values must be nonnegative")
    if not np.any(label > 0):
        raise TreeReferenceError("ATM22 tree parsing cannot derive references from empty foreground")


def _validate_tree(
    skeleton: np.ndarray,
    parsing: np.ndarray,
    parent_map: np.ndarray,
    children_map: np.ndarray,
    original_label: np.ndarray,
    num: int,
) -> tuple[np.ndarray, int]:
    """Validate final results without calling the mutating upstream predicate."""
    if skeleton.shape != original_label.shape or parsing.shape != original_label.shape:
        raise TreeReferenceError("Derived reference shapes differ from the original label")
    if num < 1 or not np.any(skeleton):
        raise TreeReferenceError("ATM22 removed every branch or returned an empty skeleton")
    if not np.isin(skeleton, (0, 1)).all():
        raise TreeReferenceError("The pinned ATM22 skeleton must contain only 0 and 1")
    if not np.isfinite(parsing).all() or not np.equal(parsing, np.floor(parsing)).all():
        raise TreeReferenceError("Final parsing must contain finite integer branch IDs")
    actual_ids = np.unique(parsing[parsing > 0])
    if np.any(parsing < 0) or not np.array_equal(actual_ids, np.arange(1, num + 1)):
        raise TreeReferenceError("Final parsing IDs must be contiguous 1..N")
    foreground_skeleton = skeleton > 0
    if np.any(foreground_skeleton & (original_label <= 0)):
        raise TreeReferenceError(
            "ATM22 skeleton extends outside the ORIGINAL label (for example after hole "
            "filling); it cannot be clipped or the original label changed silently"
        )
    if np.any(foreground_skeleton & (parsing <= 0)):
        raise TreeReferenceError("Every full skeleton voxel must have a final parsing branch ID")
    if parent_map.shape != (num, num) or children_map.shape != (num, num):
        raise TreeReferenceError("Final graph matrices disagree with the branch count")
    if not np.isin(parent_map, (0, 1)).all() or not np.isin(children_map, (0, 1)).all():
        raise TreeReferenceError("Final graph matrices must be binary")
    if np.any(np.diag(children_map)):
        raise TreeReferenceError("Final children map must not contain self loops")
    roots = np.flatnonzero(np.diag(parent_map))
    if len(roots) != 1:
        raise TreeReferenceError("Final parent map must mark exactly one root")
    root = int(roots[0])
    expected_parent = children_map.T.copy()
    expected_parent[root, root] = 1
    if not np.array_equal(parent_map, expected_parent):
        raise TreeReferenceError("Final parent and children matrices are inconsistent")
    indegree = np.sum(children_map, axis=0)
    if indegree[root] != 0 or np.any(np.delete(indegree, root) != 1):
        raise TreeReferenceError("Each non-root branch must have exactly one parent")
    graph = np.asarray(np.nonzero(children_map), dtype=np.int64)
    if graph.shape != (2, num - 1):
        raise TreeReferenceError("Final graph must have exactly N-1 directed edges")
    visited: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in visited:
            raise TreeReferenceError("Final graph contains a cycle or repeated child")
        visited.add(node)
        pending.extend(int(v) for v in np.flatnonzero(children_map[node]))
    if len(visited) != num:
        raise TreeReferenceError("Final graph is not fully reachable from its root")
    return graph, root


def _derive_reference(label_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    upstream = _load_upstream()
    _validate_label(label_xyz)
    binary_working_label = (label_xyz > 0).astype(np.uint8)
    binary_working_label = upstream.large_connected_domain(binary_working_label)
    skeleton = upstream.skeletonize_3d(binary_working_label)
    skeleton_parse, component_ids, num = upstream.skeleton_parsing(skeleton)
    num = int(num)
    if num < 1:
        raise TreeReferenceError("ATM22 short-branch filtering removed every skeleton branch")
    parsing = upstream.tree_parsing_func(skeleton_parse, binary_working_label, component_ids)
    del skeleton_parse, component_ids, binary_working_label
    trachea = upstream.loc_trachea(parsing, num)
    adjacency = upstream.adjacent_map(parsing, num)
    parent_map, children_map, _ = upstream.parent_children_map(adjacency, trachea, num)
    refinement_count = 0
    initial_count = num
    # IMPORTANT: whether_refinement mutates parsing. Preserve the original
    # driver's exact condition/body order, including its final False call.
    while upstream.whether_refinement(parent_map, children_map, parsing, num, trachea) is True:
        old_num = num
        parsing, num = upstream.tree_refinement(parent_map, children_map, parsing, num, trachea)
        num = int(num)
        if num < 1 or num >= old_num:
            raise TreeReferenceError("ATM22 tree refinement did not reduce the positive branch count")
        refinement_count += 1
        trachea = upstream.loc_trachea(parsing, num)
        adjacency = upstream.adjacent_map(parsing, num)
        parent_map, children_map, _ = upstream.parent_children_map(adjacency, trachea, num)
    graph, root = _validate_tree(skeleton, parsing, parent_map, children_map, label_xyz, num)
    return skeleton, parsing, graph, {
        "node_count": num,
        "edge_count": int(graph.shape[1]),
        "root_zero_based": root,
        "initial_branch_count": initial_count,
        "refinement_iterations": refinement_count,
        "skeleton_voxels": int(np.count_nonzero(skeleton)),
    }


def _cache_result(cache_dir: Path, identity: dict, signature: str) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity or metadata["cache_key"] != signature:
            raise TreeReferenceError("Cached airway reference identity differs from the requested input")
        expected_names = {"skeleton.mha", "parsing.mha", "preparation.log"}
        if identity["include_graph"]:
            expected_names.add("graph.npy")
        if set(metadata["output_sha256"]) != expected_names:
            raise TreeReferenceError("Cached airway reference artifact list is incomplete or unexpected")
        for name in sorted(expected_names):
            if _sha256_file(cache_dir / name) != metadata["output_sha256"][name]:
                raise TreeReferenceError(f"Cached airway reference has changed: {cache_dir / name}")
        node_count = metadata["tree"]["node_count"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TreeReferenceError(f"Invalid/incomplete airway reference cache at {cache_dir}: {exc}") from exc
    return {
        "protocol_id": PROTOCOL_ID,
        "include_graph": identity["include_graph"],
        "skeleton_path": str(cache_dir / "skeleton.mha"),
        "parsing_path": str(cache_dir / "parsing.mha"),
        "graph_path": str(cache_dir / "graph.npy") if identity["include_graph"] else None,
        "metadata_path": str(metadata_path),
        "cache_key": signature,
        "node_count": node_count,
    }


def prepare_tree_reference(
    label_path: str | Path, output_dir: str | Path, *, include_graph: bool = True
) -> dict[str, Any]:
    """Prepare one immutable reference entry and return its artifact paths.

    Input is one scalar MHA/NIfTI label. Nonnegative integer anatomical
    labels are preserved; the ATM22 algorithm works on a separate binary copy in XYZ
    order. Outputs retain the original geometry. Existing entries are reused
    only after verifying input, source, dependency and output hashes. Corrupt
    entries raise rather than being overwritten. Temporary partial output is
    removed on failure, and no models or full-dataset jobs are launched.
    """
    if not isinstance(include_graph, bool):
        raise TreeReferenceError("include_graph must be a boolean")
    label_path = Path(label_path).expanduser().resolve(strict=True)
    if not str(label_path).lower().endswith((".mha", ".nii", ".nii.gz")):
        raise TreeReferenceError("Use a self-contained .mha, .nii or .nii.gz label for cache provenance")
    versions = _dependency_versions()
    _load_upstream()  # Verify the pinned source even on a cache hit.
    sitk = importlib.import_module("SimpleITK")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(label_path))
    reader.ReadImageInformation()
    geometry = _geometry(reader)
    identity = {
        "protocol_id": PROTOCOL_ID,
        "include_graph": include_graph,
        "input_sha256": _sha256_file(label_path),
        "geometry": geometry,
        "source_commit": ATM22_COMMIT,
        "source_sha256": ATM22_SOURCE_SHA256,
        "builder_sha256": _sha256_file(Path(__file__)),
        "dependencies": versions,
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    output_dir = Path(output_dir).expanduser().resolve()
    cache_dir = output_dir / f"tree-{signature}"
    if cache_dir.exists():
        return _cache_result(cache_dir, identity, signature)

    image = sitk.ReadImage(str(label_path))
    # SimpleITK arrays use ZYX; ATM22's nibabel driver uses XYZ.
    label_xyz = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    _validate_label(label_xyz)
    if _geometry(image) != geometry:
        raise TreeReferenceError("Input label geometry changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".tree-{signature}-", dir=output_dir))
    try:
        # Upstream traversal emits debug prints. A preparation log preserves
        # them without contaminating the CLI's machine-readable JSON result.
        with (temporary / "preparation.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                skeleton, parsing, graph, tree = _derive_reference(label_xyz)
        del label_xyz
        for name, array in (("skeleton.mha", skeleton), ("parsing.mha", parsing)):
            # Explicit types preserve the full skeleton and branch IDs exactly.
            dtype = np.uint8 if name == "skeleton.mha" else np.uint32
            volume = sitk.GetImageFromArray(array.transpose(2, 1, 0).astype(dtype))
            volume.CopyInformation(image)
            sitk.WriteImage(volume, str(temporary / name), useCompression=True)
            del volume
        del skeleton, parsing, image
        names = ["skeleton.mha", "parsing.mha", "preparation.log"]
        if include_graph:
            np.save(temporary / "graph.npy", graph, allow_pickle=False)
            names.append("graph.npy")
        if _sha256_file(label_path) != identity["input_sha256"]:
            raise TreeReferenceError("Input label content changed during preparation")
        metadata = {
            "identity": identity,
            "cache_key": signature,
            "input_path": str(label_path),
            "source_url": ATM22_SOURCE_URL,
            "reference_origin": "local ATM22-derived; not ATM26 organizer-provided references",
            "tree": tree,
            "output_sha256": {name: _sha256_file(temporary / name) for name in names},
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Same-filesystem directory rename: readers never see partial output.
        try:
            os.rename(temporary, cache_dir)
        except OSError:
            if not cache_dir.exists():
                raise
            # Another producer may have published the same input meanwhile;
            # accept it only after the same strict provenance/digest checks.
            return _cache_result(cache_dir, identity, signature)
        return _cache_result(cache_dir, identity, signature)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="One original label MHA/NIfTI")
    parser.add_argument("--output-dir", required=True, help="Private reference cache directory")
    parser.add_argument("--include-graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = prepare_tree_reference(args.label, args.output_dir, include_graph=args.include_graph)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
