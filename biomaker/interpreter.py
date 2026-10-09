"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import codecs
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin

from .utils.environment import validate_code_environment, validate_process_containment
from .utils.process_tree import (
    ProcessTreeInspectionError,
    ProcessTreeTracker,
    drain_uncaptured_process_group,
)

logger = logging.getLogger("biomaker")


def _decode_process_output(value: bytes | str | None) -> str:
    """Keep the universal-newline behavior previously supplied by Popen(text=True)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "").replace("\r\n", "\n").replace("\r", "\n")


class _StreamingLogRedactor:
    """Redact before logging, including secrets divided between pipe reads.

    Only a possible secret prefix is held back; progress without a newline is
    emitted immediately. Long API/Bearer tokens are consumed without buffering.
    The original candidate output remains in its existing execution record.
    """

    def __init__(self, environment: Mapping[str, str]):
        self.secrets = sorted(
            {
                value
                for key, value in environment.items()
                if re.search(
                    r"(?:API_?KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)", key, re.I
                )
                and len(value) >= 6
            },
            key=len,
            reverse=True,
        )
        self.pending = ""
        self.token_kind: str | None = None

    @staticmethod
    def _token_character(character: str, kind: str) -> bool:
        alphabet = "._~+/=-" if kind == "bearer" else "_-"
        return character.isascii() and (
            character.isalnum() or character in alphabet
        )

    def feed(self, value: str, *, final: bool = False) -> str:
        value = self.pending + value
        self.pending = ""
        output = []
        position = 0
        while position < len(value):
            remaining_length = len(value) - position
            matched = False
            # Known credentials may contain punctuation outside the generic token
            # alphabet. Match (or retain) them before consuming any generic prefix,
            # including credentials embedded in an already-redacted token.
            for secret in self.secrets:
                if value.startswith(secret, position):
                    if self.token_kind is None:
                        output.append("[REDACTED]")
                    position += len(secret)
                    # A known short key may prefix a longer unknown API token.
                    # Continue consuming that token without retaining its body.
                    for marker, kind in (("sk-", "api"), ("bearer ", "bearer")):
                        if secret.lower().startswith(marker) and all(
                            self._token_character(character, kind)
                            for character in secret[len(marker) :]
                        ):
                            self.token_kind = kind
                            break
                    matched = True
                    break
                if (
                    not final
                    and remaining_length < len(secret)
                    and secret.startswith(value[position:])
                ):
                    self.pending = value[position:]
                    return "".join(output)
            if matched:
                continue
            if self.token_kind is not None:
                if self._token_character(value[position], self.token_kind):
                    position += 1
                    continue
                self.token_kind = None
            for marker, kind in (("sk-", "api"), ("bearer ", "bearer")):
                lowered = value[position : position + len(marker)].lower()
                if lowered == marker:
                    output.append(
                        "Bearer [REDACTED]" if kind == "bearer" else "[REDACTED]"
                    )
                    position += len(marker)
                    self.token_kind = kind
                    matched = True
                    break
                if not final and marker.startswith(lowered):
                    self.pending = value[position:]
                    return "".join(output)
            if matched:
                continue
            output.append(value[position])
            position += 1
        if final:
            self.token_kind = None
        return "".join(output)


class _LiveProcessOutput:
    """Forward only the new bytes in communicate()'s cumulative snapshots."""

    def __init__(self, label: str, channel: str, environment: Mapping[str, str]):
        self.label = label
        self.channel = channel
        self.offset = 0
        self.decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True
        )
        self.redactor = _StreamingLogRedactor(environment)

    def update(self, captured: bytes | str | None, *, final: bool = False) -> None:
        data = (
            captured.encode("utf-8") if isinstance(captured, str) else captured or b""
        )
        fresh = data[self.offset :]
        self.offset = len(data)
        # Bound each decode/redaction/logging operation even for very noisy scripts.
        for offset in range(0, len(fresh), 4096):
            self._emit(self.decoder.decode(fresh[offset : offset + 4096]))
        if final:
            self._emit(self.decoder.decode(b"", final=True), final=True)

    def _emit(self, value: str, *, final: bool = False) -> None:
        safe = self.redactor.feed(value, final=final)
        lines = (
            part
            for line in safe.splitlines(keepends=True)
            for part in (
                line[index : index + 4096] for index in range(0, len(line), 4096)
            )
        )
        for line in lines:
            logger.info(
                "[candidate %s %s] %s",
                self.label,
                self.channel,
                line.rstrip("\n"),
                extra={
                    "markup": False,
                    "highlighter": None,
                    "biomaker_candidate_output": True,
                    "candidate_output": line,
                    "candidate_output_label": self.label,
                    "candidate_output_stream": self.channel,
                },
            )


_IPYTHON_TRACEBACK_WRAPPER = r"""
import runpy
import sys
import traceback

try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except SystemExit:
    raise
except BaseException:
    etype, evalue, tb = sys.exc_info()
    try:
        from IPython.core.ultratb import VerboseTB

        VerboseTB(color_scheme="NoColor")(etype, evalue, tb)
    except Exception:
        traceback.print_exception(etype, evalue, tb)
    sys.exit(1)
"""


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
    resource_usage: dict | None = None
    stdout: str = ""
    stderr: str = ""
    candidate_output: str = ""
    candidate_exec_time: float | None = None
    candidate_returncode: int | None = None
    candidate_exc_type: str | None = None
    candidate_exc_info: dict | None = None
    failure_phase: str | None = None
    artifact_validation: dict[str, Any] | None = None
    evaluator_result: dict[str, Any] | None = None
    system_outputs: list[dict[str, Any]] = field(default_factory=list)

    def append_system_output(
        self,
        phase: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        append_to_term_out: bool = True,
    ) -> None:
        """Record BioMaker-owned post-processing output separately from candidate stdout."""
        record: dict[str, Any] = {"phase": phase, "message": str(message)}
        if details is not None:
            record["details"] = details
        self.system_outputs.append(record)
        if append_to_term_out and message:
            self.term_out.append("\n" + str(message))

    def execution_record(self) -> dict[str, Any]:
        """Return the structured execution facts used by prompts and run exports."""
        candidate_failure = None
        if self.candidate_exc_type is not None:
            candidate_failure = {
                "phase": (
                    self.failure_phase
                    if self.failure_phase in {"candidate_run", "environment_preflight"}
                    else "candidate_run"
                ),
                "type": self.candidate_exc_type,
                "info": self.candidate_exc_info,
                "returncode": self.candidate_returncode,
            }
        final_failure = None
        if self.exc_type is not None:
            final_failure = {
                "phase": self.failure_phase or "candidate_run",
                "type": self.exc_type,
                "info": self.exc_info,
            }
        return {
            "schema_version": 1,
            "candidate": {
                "stdout": self.stdout,
                "stderr": self.stderr,
                "output": self.candidate_output,
                "returncode": self.candidate_returncode,
                "runtime_s": (
                    self.candidate_exec_time
                    if self.candidate_exec_time is not None
                    else self.exec_time
                ),
                "resources": self.resource_usage,
                "failure": candidate_failure,
            },
            "node_runtime_s": self.exec_time,
            "artifact_validation": self.artifact_validation,
            "evaluator": self.evaluator_result,
            "final_failure": final_failure,
            "system_outputs": list(self.system_outputs),
        }


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


def _descendant_pids(pid: int) -> list[int]:
    """Return child process ids recursively using platform tools."""
    try:
        completed = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=0.2,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    children = []
    for line in completed.stdout.splitlines():
        try:
            child = int(line.strip())
        except ValueError:
            continue
        children.append(child)
        children.extend(_descendant_pids(child))
    return children


def _sample_cuda_gpu_memory_mb(pids: list[int]) -> float | None:
    """Sample CUDA memory used by this process tree via nvidia-smi, when available."""
    if not pids or shutil.which("nvidia-smi") is None:
        return None
    tracked = {int(process_id) for process_id in pids}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None

    total_mb = 0.0
    matched = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            app_pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if app_pid in tracked:
            total_mb += used_mb
            matched = True
    return total_mb if matched else None


def _sample_cuda_total_gpu_memory_mb() -> float | None:
    """Sample total CUDA memory currently used on visible NVIDIA GPUs."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None

    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _sample_process_usage(pid: int, sample_gpu: bool = True) -> dict | None:
    """Sample process-tree CPU, system memory, and accelerator memory."""
    pids = [pid, *_descendant_pids(pid)]
    pid_args = ",".join(str(p) for p in pids)
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=,%cpu=", "-p", pid_args],
            capture_output=True,
            text=True,
            timeout=0.2,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        rss_mb = 0.0
        cpu_percent = 0.0
        sampled_processes = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            rss_mb += int(parts[0]) / 1024.0
            cpu_percent += float(parts[1])
            sampled_processes += 1
        sample = {
            "rss_mb": rss_mb,
            "cpu_percent": cpu_percent,
            "process_count": sampled_processes,
        }
        gpu_memory_mb = _sample_cuda_gpu_memory_mb(pids) if sample_gpu else None
        if gpu_memory_mb is not None:
            sample["gpu_memory_mb"] = gpu_memory_mb
            sample["gpu_backend"] = "cuda"
            sample["gpu_memory_source"] = "nvidia-smi_compute_apps_process_tree"
        else:
            total_gpu_memory_mb = _sample_cuda_total_gpu_memory_mb()
            if total_gpu_memory_mb is not None:
                sample["total_gpu_memory_mb"] = total_gpu_memory_mb
                sample["gpu_backend"] = "cuda"
        return sample
    except ValueError:
        return None


def _summarize_usage(
    samples: list[dict], gpu_baseline_mb: float | None = None
) -> dict | None:
    if not samples:
        return None
    rss_values = [float(s["rss_mb"]) for s in samples if "rss_mb" in s]
    cpu_values = [float(s["cpu_percent"]) for s in samples if "cpu_percent" in s]
    summary: dict[str, float | int | str] = {"sample_count": len(samples)}
    if rss_values:
        summary["peak_rss_mb"] = max(rss_values)
    if cpu_values:
        summary["avg_cpu_percent"] = sum(cpu_values) / len(cpu_values)
        summary["peak_cpu_percent"] = max(cpu_values)
    gpu_values = [
        float(s["gpu_memory_mb"])
        for s in samples
        if isinstance(s.get("gpu_memory_mb"), (int, float))
    ]
    if gpu_values:
        summary["peak_gpu_memory_mb"] = max(gpu_values)
        summary["gpu_backend"] = "cuda"
        summary["gpu_memory_source"] = "nvidia-smi_compute_apps_process_tree"
    else:
        total_gpu_values = [
            float(s["total_gpu_memory_mb"])
            for s in samples
            if isinstance(s.get("total_gpu_memory_mb"), (int, float))
        ]
        if total_gpu_values:
            peak_total_gpu = max(total_gpu_values)
            baseline = gpu_baseline_mb if gpu_baseline_mb is not None else 0.0
            summary["peak_gpu_memory_mb"] = max(0.0, peak_total_gpu - baseline)
            summary["peak_total_gpu_memory_mb"] = peak_total_gpu
            if gpu_baseline_mb is not None:
                summary["gpu_memory_baseline_mb"] = gpu_baseline_mb
            summary["gpu_backend"] = "cuda"
            summary["gpu_memory_source"] = "nvidia-smi_total_used_delta_approx"
    if "peak_gpu_memory_mb" not in summary and rss_values and _mps_available():
        summary["peak_gpu_memory_mb"] = max(rss_values)
        summary["gpu_backend"] = "mps"
        summary["gpu_memory_source"] = "unified_memory_rss_proxy"
    process_counts = [int(s["process_count"]) for s in samples if "process_count" in s]
    if process_counts:
        summary["peak_process_count"] = max(process_counts)
    summary["scope"] = "generated_script_process_tree"
    return summary


def _format_usage_line(resource_usage: dict | None) -> str | None:
    if not resource_usage:
        return None
    parts = []
    peak_rss = resource_usage.get("peak_rss_mb")
    if isinstance(peak_rss, (int, float)):
        parts.append(f"peak_rss={peak_rss:.1f} MB")
    avg_cpu = resource_usage.get("avg_cpu_percent")
    if isinstance(avg_cpu, (int, float)):
        parts.append(f"avg_cpu={avg_cpu:.1f}%")
    peak_gpu = resource_usage.get("peak_gpu_memory_mb")
    if isinstance(peak_gpu, (int, float)):
        backend = resource_usage.get("gpu_backend")
        name = f"peak_{backend}_memory" if backend else "peak_gpu_memory"
        parts.append(f"{name}={peak_gpu:.1f} MB")
    peak_processes = resource_usage.get("peak_process_count")
    if isinstance(peak_processes, (int, float)) and peak_processes > 1:
        parts.append(f"peak_processes={int(peak_processes)}")
    if not parts:
        return None
    return "Resource usage: " + ", ".join(parts) + "."


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        enforce_environment: bool = True,
        include_medical_packages: bool = True,
        environment: Mapping[str, str] | None = None,
        output_label: str | None = None,
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
        if not self.working_dir.exists():
            raise FileNotFoundError(
                f"Working directory {self.working_dir} does not exist"
            )
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.enforce_environment = enforce_environment
        self.include_medical_packages = include_medical_packages
        self.environment = dict(environment or {})
        self.output_label = re.sub(
            r"[^A-Za-z0-9_.-]", "_", output_label or uuid.uuid4().hex
        )[:12]

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

        preflight = (
            validate_code_environment(
                code, include_medical=self.include_medical_packages
            )
            if self.enforce_environment
            else validate_process_containment(code)
        )
        if not preflight.ok:
            return ExecutionResult(
                term_out=[preflight.message],
                exec_time=0.0,
                exc_type=preflight.error_type,
                exc_info=preflight.exc_info,
                exc_stack=None,
                resource_usage=None,
                stderr=preflight.message,
                candidate_output=preflight.message,
                candidate_exec_time=0.0,
                candidate_exc_type=preflight.error_type,
                candidate_exc_info=preflight.exc_info,
                failure_phase="environment_preflight",
            )

        script_path = self.working_dir / self.agent_file_name
        script_path.write_text(code, encoding="utf-8")
        start_time = time.time()
        usage_samples: list[dict] = []
        gpu_baseline_mb = _sample_cuda_total_gpu_memory_mb()
        command = (
            [
                sys.executable,
                "-u",
                "-c",
                _IPYTHON_TRACEBACK_WRAPPER,
                self.agent_file_name,
            ]
            if self.format_tb_ipython
            else [sys.executable, "-u", self.agent_file_name]
        )
        environment = {**os.environ, **self.environment, "PYTHONUNBUFFERED": "1"}
        live_stdout = _LiveProcessOutput(self.output_label, "stdout", environment)
        live_stderr = _LiveProcessOutput(self.output_label, "stderr", environment)
        process: subprocess.Popen[bytes] | None = None
        process_tree: ProcessTreeTracker | None = None
        try:
            popen_kwargs: dict[str, Any] = {
                "cwd": self.working_dir,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                # TimeoutExpired always exposes bytes; binary mode makes cumulative
                # offsets consistent with the final communicate() result, including
                # CRLF and multibyte characters split over multiple reads.
                "text": False,
                "env": environment,
            }
            if os.name == "posix":
                # A candidate-owned PGID survives leader exit, while remaining in the
                # top-level run session for session-wide Stop handling.
                popen_kwargs["process_group"] = 0
            process = subprocess.Popen(command, **popen_kwargs)
            try:
                process_tree = ProcessTreeTracker.capture(process.pid)
            except (
                OSError,
                ProcessLookupError,
                ProcessTreeInspectionError,
                ValueError,
            ):
                drain_uncaptured_process_group(process)
                raise RuntimeError(
                    "Could not capture generated process identity"
                ) from None
            stdout = ""
            stderr = ""
            timed_out = False
            next_gpu_sample_time = 0.0
            while True:
                process_tree.refresh_best_effort()
                if process.poll() is not None:
                    # Descendants may inherit the PIPE file descriptors after the
                    # leader exits. Drain the owned PGID first so communicate()
                    # observes EOF instead of turning a successful leader into a
                    # false timeout.
                    process_tree.terminate_and_wait(root_process=process)
                    stdout, stderr = process.communicate()
                    break

                now = time.time()
                sample_gpu = now >= next_gpu_sample_time
                sample = _sample_process_usage(process.pid, sample_gpu=sample_gpu)
                if sample_gpu:
                    next_gpu_sample_time = time.time() + 1.0
                if sample is not None:
                    usage_samples.append(sample)

                elapsed = time.time() - start_time
                remaining = self.timeout - elapsed
                if remaining <= 0:
                    timed_out = True
                    process_tree.terminate_and_wait(
                        root_process=process,
                    )
                    stdout, stderr = process.communicate()
                    break

                try:
                    stdout, stderr = process.communicate(
                        timeout=min(0.5, max(0.05, remaining))
                    )
                    break
                except subprocess.TimeoutExpired as pending:
                    live_stdout.update(pending.output)
                    live_stderr.update(pending.stderr)
                    continue

            live_stdout.update(stdout, final=True)
            live_stderr.update(stderr, final=True)
            stdout = _decode_process_output(stdout)
            stderr = _decode_process_output(stderr)
            exec_time = time.time() - start_time
            resource_usage = _summarize_usage(
                usage_samples, gpu_baseline_mb=gpu_baseline_mb
            )
            output_text = (stdout or "") + (stderr or "")
            output = [output_text]
            usage_line = _format_usage_line(resource_usage)
            if usage_line:
                output.append(usage_line)

            if timed_out:
                output.append(
                    f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
                return ExecutionResult(
                    output,
                    float(self.timeout),
                    "TimeoutError",
                    {"timeout": str(self.timeout)},
                    None,
                    resource_usage,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    candidate_output=output_text,
                    candidate_exec_time=float(self.timeout),
                    candidate_returncode=process.returncode,
                    candidate_exc_type="TimeoutError",
                    candidate_exc_info={"timeout": str(self.timeout)},
                    failure_phase="candidate_run",
                )

            if process.returncode == 0:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
                    f"(time limit is {humanize.naturaldelta(self.timeout)})."
                )
                return ExecutionResult(
                    output,
                    exec_time,
                    None,
                    None,
                    None,
                    resource_usage,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    candidate_output=output_text,
                    candidate_exec_time=exec_time,
                    candidate_returncode=process.returncode,
                )

            exc_type = _infer_exception_type(output_text, process.returncode)
            output.append(f"Process exited with return code {process.returncode}.")
            exc_info = {"returncode": str(process.returncode)}
            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                exc_info,
                None,
                resource_usage,
                stdout=stdout or "",
                stderr=stderr or "",
                candidate_output=output_text,
                candidate_exec_time=exec_time,
                candidate_returncode=process.returncode,
                candidate_exc_type=exc_type,
                candidate_exc_info=exc_info,
                failure_phase="candidate_run",
            )
        except subprocess.TimeoutExpired as e:
            live_stdout.update(e.stdout, final=True)
            live_stderr.update(e.stderr, final=True)
            stdout = _decode_process_output(e.stdout)
            stderr = _decode_process_output(e.stderr)
            resource_usage = _summarize_usage(
                usage_samples, gpu_baseline_mb=gpu_baseline_mb
            )
            output_text = stdout + stderr
            output = [
                output_text,
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}",
            ]
            usage_line = _format_usage_line(resource_usage)
            if usage_line:
                output.insert(1, usage_line)
            return ExecutionResult(
                output,
                float(self.timeout),
                "TimeoutError",
                {"timeout": str(self.timeout)},
                None,
                resource_usage,
                stdout=stdout,
                stderr=stderr,
                candidate_output=output_text,
                candidate_exec_time=float(self.timeout),
                candidate_exc_type="TimeoutError",
                candidate_exc_info={"timeout": str(self.timeout)},
                failure_phase="candidate_run",
            )
        finally:
            if process_tree is not None:
                process_tree.terminate_and_wait(root_process=process)
            elif process is not None and process.poll() is None:
                # Only used when identity capture itself failed. The unreaped Popen
                # child still owns its PID, so it cannot refer to an unrelated process.
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
                process.wait()
            try:
                script_path.unlink()
            except FileNotFoundError:
                pass
