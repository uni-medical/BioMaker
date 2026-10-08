"""Result summaries preserve historical CSV semantics without retaining all rows."""

import csv
import io
import math
import random
import tracemalloc

import pytest

from webui.artifacts import read_results_summary


def previous_summary(text):
    """Materializing oracle from the pre-streaming response contract."""
    rows = list(csv.DictReader(io.StringIO(text, newline=""), delimiter="\t"))
    comparable = []
    for row in rows:
        try:
            metric = float(row.get("metric", ""))
        except (ValueError, TypeError):
            continue
        if math.isfinite(metric):
            comparable.append((metric, row))
    best = None
    if comparable:
        maximize = str(comparable[-1][1].get("maximize", "true")).lower() != "false"
        best = (max if maximize else min)(value for value, _ in comparable)
    latest = None if not rows else {
        key: rows[-1].get(key)
        for key in ("step", "status", "metric", "outcome", "exc_type", "error")
    }
    return {"count": len(rows), "best_metric": best, "latest": latest}


def source(settings, text):
    directory = settings.logs_root / "results-streaming"
    directory.mkdir(exist_ok=True)
    (directory / "results.tsv").write_text(text, encoding="utf-8", newline="")
    return {"id": directory.name, "log_dir": str(directory)}


@pytest.mark.parametrize("seed", [5, 31, 997])
def test_mixed_rows_match_previous_summary_contract(settings, seed):
    rng = random.Random(seed)
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\r\n")
    writer.writerow(["step", "metric", "maximize", "status", "error"])
    for step in range(500):
        writer.writerow([
            step,
            rng.choice(["NaN", "Infinity", "-inf", "", "bad", -3, 0.2, 0.9]),
            rng.choice(["true", "TRUE", "false", "False", "", "unexpected"]),
            rng.choice(["succeeded", "failed", ""]),
            rng.choice(["", "诊断\t字段", "line1\r\nline2", 'quoted "field"']),
        ])
    # The latest invalid score must not replace the last finite row's direction.
    writer.writerow([500, "NaN", "false", "failed", "nonfinite last row"])
    text = output.getvalue()
    run = source(settings, text)
    assert read_results_summary(settings, run) == previous_summary(text)


@pytest.mark.parametrize("text", [
    "", "metric\n", "metric\n\n\n", "step\n0\n1\n",
    "metric\tmaximize\n1\ttrue\n-1\tfalse\nNaN\ttrue\n",
    "metric\tmetric\tmaximize\n0.1\t0.7\tfalse\n",
    "metric\tmaximize\n0.8\n0.2\tfalse\textra\n",
    "metric\tmaximize\nNaN\tfalse\n-inf\ttrue\n",
])
def test_empty_missing_duplicate_and_extra_fields_preserve_summary(settings, text):
    assert read_results_summary(settings, source(settings, text)) == previous_summary(text)


def test_late_csv_failure_discards_partial_summary(settings):
    limit = csv.field_size_limit()
    run = source(settings, "metric\n0.5\n" + "x" * (limit + 1) + "\n")
    assert read_results_summary(settings, run) == {"count": 0, "best_metric": None, "latest": None}
    with pytest.raises(csv.Error):
        read_results_summary(settings, run, strict=True)


def test_many_rows_do_not_accumulate_python_row_objects(settings):
    run = source(settings, "step\tmetric\tmaximize\tstatus\n")
    path = settings.logs_root / run["id"] / "results.tsv"
    with path.open("a", encoding="utf-8") as handle:
        for step in range(50_000):
            handle.write(f"{step}\t0.5\ttrue\tsucceeded\n")
    tracemalloc.start()
    try:
        result = read_results_summary(settings, run)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["count"] == 50_000 and result["best_metric"] == 0.5
    assert result["latest"]["step"] == "49999"
    # This small-row fixture previously retained tens of MiB of row dictionaries.
    # The allowance covers parser buffers and the current row, not all possible
    # memory costs for arbitrarily large individual fields.
    assert peak < 4 * 1024 * 1024
