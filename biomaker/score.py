"""Rescore saved predictions with their existing frozen evaluator, without training.

Example: python -m biomaker.score --workspace RUN --prediction predictions.csv
         --truth private_answers.csv --expected expected.csv --timeout 3600
         --workers 2 --output private_logs/rescore.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from .utils.evaluator_setup import score_artifacts_with_evaluator


def positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--truth', required=True, type=Path)
    parser.add_argument('--expected', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='New private result file; existing files are never overwritten')
    parser.add_argument('--timeout', type=positive_integer, default=3600)
    parser.add_argument('--workers', type=positive_integer, default=1)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    progress = output.with_suffix(output.suffix + '.progress.jsonl')
    if output.exists() or progress.exists():
        parser.error('Choose a new output path; previous scoring evidence is preserved')
    protected = [args.prediction, args.truth, args.expected,
                 args.workspace / 'evaluator.py',
                 args.workspace / 'evaluator_prediction_spec.json']
    if output in [p.resolve() for p in protected] or progress in [p.resolve() for p in protected]:
        parser.error('Result paths must not overwrite scoring inputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both paths, so retry never overwrites a previous interrupted attempt.
    progress.touch(exist_ok=False)
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),
                    workspace=str(args.workspace.resolve()),
                    prediction=str(args.prediction.resolve()),
                    timeout_seconds=args.timeout, workers=args.workers,
                    progress_path=str(progress), training_started=False,
                    journal_updated=False)
    with output.open('x') as handle:
        json.dump(dict(metadata, status='running'), handle, indent=2)
    started = time.monotonic()
    try:
        result = score_artifacts_with_evaluator(
            args.workspace.resolve(), args.prediction.resolve(), args.truth.resolve(),
            args.expected.resolve(), timeout=args.timeout,
            env_overrides={'BIOMAKER_EVALUATOR_WORKERS': str(args.workers),
                           'BIOMAKER_EVALUATOR_PROGRESS_PATH': str(progress)},
        )
        payload = dict(metadata, status='completed' if result.ok else 'failed',
                       elapsed_seconds=time.monotonic() - started, result=asdict(result))
        exit_code = 0 if result.ok else 124 if result.failure_kind == 'timeout' else 1
    except Exception as error:
        payload = dict(metadata, status='failed', elapsed_seconds=time.monotonic() - started,
                       error_type=type(error).__name__, error=str(error))
        exit_code = 1
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=output.parent,
            prefix=output.name + '.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, default=str)
            handle.write('\n')
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(dict(status=payload['status'], output=str(output), exit_code=exit_code)))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
