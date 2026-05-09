#!/usr/bin/env python
"""Run the agentic split builder on one dataset and write the proposed package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biomaker.utils.agentic_data_planner import run_agentic_split_builder_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-split-dir", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Write observed data facts, generated builder responses, stdout, and verifier JSON under _debug/.",
    )
    args = parser.parse_args()

    result = run_agentic_split_builder_probe(
        args.data_dir.resolve(),
        args.task_file.resolve(),
        args.out_dir.resolve(),
        model=args.model,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        reference_split_dir=(
            args.reference_split_dir.resolve() if args.reference_split_dir else None
        ),
        debug_artifacts=args.debug_artifacts,
    )

    summary = {
        "output_dir": str(result.output_dir),
        "split_dir": str(result.split_dir),
        "private_gt_dir": str(result.private_gt_dir),
        "private_gt": result.verifier.get("private_gt"),
        "row_counts": result.verifier.get("row_counts"),
        "warnings": result.verifier.get("warnings"),
        "comparison": result.comparison,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
