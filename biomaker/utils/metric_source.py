"""Freeze the same trusted kernels for installed calls and portable evaluators."""

import ast
from pathlib import Path

METRIC_SOURCE_FILES = ("segmentation_metrics.py", "trusted_metrics.py")


def frozen_metric_source() -> str:
    """Bundle dependency source, removing only its package-relative import.

    No installed BioMaker import remains in the generated evaluator. Including
    both files in evaluator cache identity keeps changed kernels from reusing an
    old compiled evaluator.
    """
    root = Path(__file__).parent
    dependency = (root / "segmentation_metrics.py").read_text(encoding="utf-8")
    source = (root / "trusted_metrics.py").read_text(encoding="utf-8")
    imports = [node for node in ast.parse(source).body
               if isinstance(node, ast.ImportFrom) and node.level == 1
               and node.module == 'segmentation_metrics']
    expected = {'segmentation_score_case', 'validate_tree_sample', 'normalize_segmentation_parameters'}
    if (len(imports) != 1 or {item.name for item in imports[0].names} != expected
            or any(item.asname is not None for item in imports[0].names)):
        raise RuntimeError("Trusted segmentation dependency binding changed; update source bundling.")
    # Remove the import structurally so formatting it across lines cannot break
    # evaluator compilation. All other source remains byte-for-byte intact.
    lines = source.splitlines(keepends=True)
    binding = imports[0]
    lines[binding.lineno - 1:binding.end_lineno] = []
    return dependency + "\n" + ''.join(lines)
