"""Small local Python probes for generated-code planning."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from ..utils.environment import validate_code_environment
from ..utils.response import truncate_text

MAX_PROBE_CHARS = 2500
BLOCKED_IMPORTS = {
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
BLOCKED_CALLS = {"eval", "exec", "open", "compile", "__import__"}


def _guard_probe_code(code: str) -> str | None:
    if len(code) > 2000:
        return "ProbeRejected: code is too long; keep API probes short."
    if "?" in code:
        return "ProbeRejected: use valid Python code, not IPython magic such as `?` or `??`."

    preflight = validate_code_environment(code, include_medical=True)
    if not preflight.ok:
        return preflight.message

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"ProbeRejected: SyntaxError: {exc.msg} at line {exc.lineno}."
    if not tree.body:
        return "ProbeRejected: probe code has no executable Python statements."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BLOCKED_IMPORTS:
                    return f"ProbeRejected: import `{alias.name}` is not allowed in API probes."
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in BLOCKED_IMPORTS:
                return f"ProbeRejected: import `{node.module}` is not allowed in API probes."
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                return f"ProbeRejected: call `{func.id}()` is not allowed in API probes."
            if isinstance(func, ast.Attribute) and func.attr in {
                "system",
                "popen",
                "remove",
                "removedirs",
                "unlink",
                "rmdir",
                "mkdir",
                "makedirs",
                "rename",
                "replace",
                "rmtree",
                "write",
                "writelines",
                "chmod",
                "chown",
            }:
                return f"ProbeRejected: call `.{func.attr}()` is not allowed in API probes."
    return None


def _echo_last_expression(code: str) -> str:
    """Print repr(last_expression) for notebook-style probes."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code

    tree.body[-1] = ast.Assign(
        targets=[ast.Name(id="__probe_result", ctx=ast.Store())],
        value=tree.body[-1].value,
    )
    tree.body.append(
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Constant(value="__probe_result__="),
                        op=ast.Add(),
                        right=ast.Call(
                            func=ast.Name(id="repr", ctx=ast.Load()),
                            args=[ast.Name(id="__probe_result", ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                ],
                keywords=[],
            )
        )
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def python_probe(
    code: str, timeout: int = 20, cwd: Path | str | None = None
) -> str:
    """Run a short Python snippet for package/API and workspace introspection."""

    rejection = _guard_probe_code(code)
    if rejection:
        return rejection

    run_cwd = Path(cwd).resolve() if cwd is not None else None
    runnable_code = _echo_last_expression(code)
    try:
        result = subprocess.run(
            [sys.executable, "-c", runnable_code],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return truncate_text(
            f"ProbeTimeout after {timeout}s.\n{output.strip()}",
            max_chars=MAX_PROBE_CHARS,
            marker="probe output trimmed",
        )

    output = "\n".join(
        part
        for part in [
            f"exit_code={result.returncode}",
            "stdout:\n" + result.stdout.strip() if result.stdout.strip() else "",
            "stderr:\n" + result.stderr.strip() if result.stderr.strip() else "",
        ]
        if part
    )
    return truncate_text(
        output or "exit_code=0\n(no output)",
        max_chars=MAX_PROBE_CHARS,
        marker="probe output trimmed",
    )
