import json
import re

import black

ERROR_KEYWORDS = (
    "traceback",
    "error",
    "exception",
    "dependency",
    "modulenotfounderror",
    "importerror",
    "filenotfounderror",
    "runtimeerror",
    "valueerror",
    "typeerror",
    "attributeerror",
    "shape",
    "dtype",
    "cuda",
    "memory",
    "submission",
)

STATUS_PREFIXES = (
    "BioMakerExternalValidation:",
    "EVALUATOR_RESULT_JSON=",
    "EVALUATOR_ARTIFACT_REPORT_JSON=",
    "EVALUATOR_RESULT_JSON:",
    "EVALUATOR_ARTIFACT_REPORT_JSON:",
    "Resource usage:",
    "Execution time:",
    "Process exited with return code",
    "TimeoutError:",
)
PROTECTED_STATUS_PREFIXES = STATUS_PREFIXES

TRACEBACK_HEADER = "Traceback (most recent call last):"
ARTIFACT_REPORT_PREFIX = "EVALUATOR_ARTIFACT_REPORT_JSON="
RESULT_PREFIX = "EVALUATOR_RESULT_JSON="


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def truncate_text(text: str, max_chars: int = 6000, marker: str = "truncated") -> str:
    """Keep the beginning and end of long text with an explicit truncation marker."""
    if len(text) <= max_chars:
        return text
    marker_text = f"\n... [{len(text) - max_chars} characters {marker}] ...\n"
    available_chars = max_chars - len(marker_text)
    if available_chars <= 0:
        return text[:max_chars]
    head_chars = available_chars // 2
    tail_chars = available_chars - head_chars
    truncated_len = len(text) - head_chars - tail_chars
    marker_text = f"\n... [{truncated_len} characters {marker}] ...\n"
    available_chars = max_chars - len(marker_text)
    head_chars = available_chars // 2
    tail_chars = available_chars - head_chars
    truncated_text = (
        f"{text[:head_chars]}\n"
        f"... [{truncated_len} characters {marker}] ...\n"
        f"{text[-tail_chars:]}"
    )
    return truncated_text[:max_chars]


def trim_execution_output(text: str, max_chars: int = 5000) -> str:
    """Compact execution output while preserving debug and evaluator evidence.

    Most oversized logs come from progress bars or machine-readable evaluator
    JSON, not from useful diagnostic text. Normalize those first, then only use
    bounded selection as a final prompt-size guard.
    """
    text = _normalize_status_boundaries(text)
    lines, progress_count = _compact_noisy_lines(text.splitlines())
    compact_text = "\n".join(lines)
    if progress_count:
        compact_text = (
            f"[collapsed {progress_count} progress/download lines]\n" + compact_text
        )
        lines = compact_text.splitlines()

    if len(compact_text) <= max_chars:
        return compact_text

    protected_lines = _latest_status_lines(lines)
    protected_set = set(protected_lines)
    traceback_lines = [
        line for line in _last_traceback_block(lines) if line not in protected_set
    ]
    important_lines = _bounded_lines(
        [
            line
            for line in lines
            if line not in protected_set
            and any(keyword in line.lower() for keyword in ERROR_KEYWORDS)
        ],
        head=20,
        tail=40,
    )
    tail_lines = (
        []
        if traceback_lines
        else _bounded_lines(
            [line for line in lines if line not in protected_set], head=0, tail=60
        )
    )
    selected = []
    seen = set()
    for line in traceback_lines + important_lines + tail_lines:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)

    trimmed = "\n".join(selected)
    if not trimmed:
        trimmed = "\n".join(tail_lines)
    return _append_protected_status(
        trimmed,
        protected_lines,
        max_chars=max_chars,
        marker="execution output trimmed",
    )


def strip_execution_status_lines(text: str) -> str:
    """Remove BioMaker/evaluator status records from candidate-visible stdout."""
    lines = []
    for line in _normalize_status_boundaries(str(text or "")).splitlines():
        if _matched_status_prefix(line) is not None:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def format_human_execution_output(text: str, max_chars: int = 12000) -> str:
    """Return candidate program output for human inspection."""
    return truncate_text(
        strip_execution_status_lines(trim_execution_output(text, max_chars=max_chars)),
        max_chars=max_chars,
        marker="execution output trimmed",
    )


def format_prompt_execution_output(
    text: str, *, is_buggy: bool, max_chars: int = 4000
) -> str:
    """Return the execution evidence useful as context for future nodes."""
    cleaned = strip_execution_status_lines(trim_execution_output(text, max_chars=max_chars))
    if is_buggy:
        return _extract_failure_evidence(cleaned, max_chars=max_chars)
    return "Execution completed successfully. Detailed stdout is omitted from tree context."


def _compact_noisy_lines(lines: list[str]) -> tuple[list[str], int]:
    compacted = []
    progress_count = 0
    previous_blank = False
    for line in lines:
        clean = line.rstrip()
        if _is_progress_noise(clean):
            progress_count += 1
            continue
        clean = _summarize_status_line(clean)
        if not clean:
            if previous_blank:
                continue
            previous_blank = True
            compacted.append(clean)
            continue
        previous_blank = False
        compacted.append(clean)
    return compacted, progress_count


def _normalize_status_boundaries(text: str) -> str:
    normalized = str(text or "").replace("\r", "\n")
    for prefix in STATUS_PREFIXES:
        normalized = re.sub(rf"(?<!\n)({re.escape(prefix)})", r"\n\1", normalized)
    return normalized


def _latest_status_lines(lines: list[str]) -> list[str]:
    latest: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        prefix = _matched_status_prefix(line)
        if prefix is None:
            continue
        if prefix not in latest:
            order.append(prefix)
        latest[prefix] = line
    return [latest[prefix] for prefix in order]


def _matched_status_prefix(line: str) -> str | None:
    for prefix in PROTECTED_STATUS_PREFIXES:
        if line.startswith(prefix):
            return prefix
    return None


def _append_protected_status(
    body: str, protected_lines: list[str], max_chars: int, marker: str
) -> str:
    protected = "\n".join(line for line in protected_lines if line)
    if not protected:
        return truncate_text(body, max_chars=max_chars, marker=marker)
    if not body:
        return protected

    separator = "\n\n"
    budget = max_chars - len(separator) - len(protected)
    if budget <= 0:
        return protected
    return truncate_text(body, max_chars=budget, marker=marker) + separator + protected


def _extract_failure_evidence(text: str, max_chars: int) -> str:
    lines = str(text or "").splitlines()
    traceback_lines = _last_traceback_block(lines)
    if traceback_lines:
        return truncate_text(
            "\n".join(traceback_lines),
            max_chars=max_chars,
            marker="traceback trimmed",
        )
    error_lines = [
        line
        for line in lines
        if any(keyword in line.lower() for keyword in ERROR_KEYWORDS)
    ]
    if error_lines:
        return truncate_text(
            "\n".join(_bounded_lines(error_lines, head=20, tail=40)),
            max_chars=max_chars,
            marker="error output trimmed",
        )
    return truncate_text(
        "\n".join(_bounded_lines(lines, head=0, tail=80)),
        max_chars=max_chars,
        marker="execution output trimmed",
    )


def _is_progress_noise(line: str) -> bool:
    if "█" in line:
        return True
    if re.search(r"\b\d{1,3}%\|", line):
        return True
    if re.search(r"\b\d+(?:\.\d+)?[KMGT]?/[0-9.]+[KMGT]?", line) and "/s" in line:
        return True
    return False


def _summarize_status_line(line: str) -> str:
    if line.startswith(ARTIFACT_REPORT_PREFIX):
        payload = line[len(ARTIFACT_REPORT_PREFIX) :].strip()
        try:
            report = json.loads(payload)
        except Exception:
            return ARTIFACT_REPORT_PREFIX + "<artifact report JSON omitted>"
        if not isinstance(report, dict):
            return ARTIFACT_REPORT_PREFIX + "<artifact report JSON omitted>"
        parts = [
            f"available={report.get('available')}",
            f"warnings={len(report.get('warnings') or [])}",
        ]
        artifact = report.get("validation_artifact") or {}
        if isinstance(artifact, dict):
            parts.append(f"validation_rows={artifact.get('rows')}")
        alignment = report.get("alignment") or {}
        if isinstance(alignment, dict):
            parts.append(f"alignment={alignment.get('method')}")
            if alignment.get("matched_rows") is not None:
                parts.append(f"matched_rows={alignment.get('matched_rows')}")
        return "EVALUATOR_ARTIFACT_REPORT_JSON: " + ", ".join(parts)
    if line.startswith(RESULT_PREFIX):
        payload = line[len(RESULT_PREFIX) :].strip()
        try:
            result = json.loads(payload)
        except Exception:
            return line
        score = result.get("score") if isinstance(result, dict) else None
        if not isinstance(score, dict):
            return line
        parts = []
        for key in ("metric", "accuracy", "macro_ovr_roc_auc", "n_rows_scored"):
            if key in score:
                parts.append(f"{key}={score[key]}")
        validation = result.get("submission_validation")
        if isinstance(validation, dict) and "ok" in validation:
            parts.append(f"submission_ok={validation['ok']}")
        return "EVALUATOR_RESULT_JSON: " + ", ".join(parts)
    return line


def _last_traceback_block(lines: list[str]) -> list[str]:
    for idx in range(len(lines) - 1, -1, -1):
        if TRACEBACK_HEADER in lines[idx]:
            return lines[idx:]
    return []


def _bounded_lines(lines: list[str], head: int, tail: int) -> list[str]:
    if len(lines) <= head + tail:
        return lines
    return lines[:head] + lines[-tail:]


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
