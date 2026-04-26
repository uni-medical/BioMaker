"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

from .utils.environment import validate_code_environment

logger = logging.getLogger("biomaker")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def _infer_exception_type(output: str, returncode: int) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if (
            stripped.endswith("Error")
            or "Error:" in stripped
            or "Exception:" in stripped
        ):
            return stripped.split(":", 1)[0]
    if returncode < 0:
        return "ProcessTerminated"
    return "RuntimeError"


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        enforce_environment: bool = True,
        include_medical_packages: bool = True,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.enforce_environment = enforce_environment
        self.include_medical_packages = include_medical_packages

    def cleanup_session(self):
        return

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug("Executing generated script")

        if self.enforce_environment:
            preflight = validate_code_environment(
                code, include_medical=self.include_medical_packages
            )
            if not preflight.ok:
                return ExecutionResult(
                    term_out=[preflight.message],
                    exec_time=0.0,
                    exc_type=preflight.error_type,
                    exc_info=preflight.exc_info,
                    exc_stack=None,
                )

        script_path = self.working_dir / self.agent_file_name
        script_path.write_text(code, encoding="utf-8")
        start_time = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, self.agent_file_name],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            exec_time = time.time() - start_time
            output_text = (completed.stdout or "") + (completed.stderr or "")
            output = [output_text]
            if completed.returncode == 0:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
                    f"(time limit is {humanize.naturaldelta(self.timeout)})."
                )
                return ExecutionResult(output, exec_time, None, None, None)

            exc_type = _infer_exception_type(output_text, completed.returncode)
            output.append(f"Process exited with return code {completed.returncode}.")
            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                {"returncode": str(completed.returncode)},
                None,
            )
        except subprocess.TimeoutExpired as e:
            output_text = (e.stdout or "") + (e.stderr or "")
            output = [
                output_text,
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}",
            ]
            return ExecutionResult(
                output,
                float(self.timeout),
                "TimeoutError",
                {"timeout": str(self.timeout)},
                None,
            )
        finally:
            try:
                script_path.unlink()
            except FileNotFoundError:
                pass
