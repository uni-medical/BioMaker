"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import signal
import shutil
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
    resource_usage: dict | None = None


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
    process_counts = [
        int(s["process_count"]) for s in samples if "process_count" in s
    ]
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
                    resource_usage=None,
                )

        script_path = self.working_dir / self.agent_file_name
        script_path.write_text(code, encoding="utf-8")
        start_time = time.time()
        usage_samples: list[dict] = []
        gpu_baseline_mb = _sample_cuda_total_gpu_memory_mb()
        try:
            process = subprocess.Popen(
                [sys.executable, self.agent_file_name],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout = ""
            stderr = ""
            timed_out = False
            next_gpu_sample_time = 0.0
            while True:
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
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        process.kill()
                    stdout, stderr = process.communicate()
                    break

                try:
                    stdout, stderr = process.communicate(
                        timeout=min(0.5, max(0.05, remaining))
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue

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
                )

            if process.returncode == 0:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
                    f"(time limit is {humanize.naturaldelta(self.timeout)})."
                )
                return ExecutionResult(
                    output, exec_time, None, None, None, resource_usage
                )

            exc_type = _infer_exception_type(output_text, process.returncode)
            output.append(f"Process exited with return code {process.returncode}.")
            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                {"returncode": str(process.returncode)},
                None,
                resource_usage,
            )
        except subprocess.TimeoutExpired as e:
            resource_usage = _summarize_usage(
                usage_samples, gpu_baseline_mb=gpu_baseline_mb
            )
            output_text = (e.stdout or "") + (e.stderr or "")
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
            )
        finally:
            try:
                script_path.unlink()
            except FileNotFoundError:
                pass
