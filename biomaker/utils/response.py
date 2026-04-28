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
    "Execution time:",
    "Process exited with return code",
    "TimeoutError:",
)

TRACEBACK_HEADER = "Traceback (most recent call last):"


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
    """Trim noisy execution output while preserving debug and evaluator evidence."""
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    traceback_lines = _last_traceback_block(lines)
    status_lines = [line for line in lines if line.startswith(STATUS_PREFIXES)]
    important_lines = _bounded_lines(
        [
            line
            for line in lines
            if any(keyword in line.lower() for keyword in ERROR_KEYWORDS)
        ],
        head=20,
        tail=40,
    )
    tail_lines = [] if traceback_lines else _bounded_lines(lines, head=0, tail=60)
    selected = []
    seen = set()
    for line in traceback_lines + status_lines + important_lines + tail_lines:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)

    trimmed = "\n".join(selected)
    if not trimmed:
        trimmed = "\n".join(tail_lines)
    return truncate_text(
        trimmed, max_chars=max_chars, marker="execution output trimmed"
    )


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
