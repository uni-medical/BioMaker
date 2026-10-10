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
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin

from .utils.environment import validate_code_environment, validate_process_containment
from .utils.process_tree import (
    ProcessTreeCleanupError,
    ProcessTreeInspectionError,
    ProcessTreeTracker,
    drain_uncaptured_process_group,
)

logger = logging.getLogger("biomaker")

_SANDBOX_MODE_ENV = "BIOMAKER_EXECUTION_SANDBOX"
_SANDBOX_READONLY_PATHS_ENV = "BIOMAKER_SANDBOX_READONLY_PATHS"
_SANDBOX_NETWORK_ENV = "BIOMAKER_SANDBOX_NETWORK"
_SANDBOX_POLICY_VERSION = 2
_SANDBOX_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/sys"),
)
_SANDBOX_SECRET_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GITHUB_TOKEN",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "JINA_API_KEY",
        "LARK_APP_SECRET",
        "OPENAI_API_KEY",
        "SERPER_API_KEY",
        "SSH_AUTH_SOCK",
        "WANDB_API_KEY",
    }
)
_SANDBOX_SECRET_ENV_SUFFIXES = (
    "_ACCESS_TOKEN",
    "_API_KEY",
    "_APP_SECRET",
    "_AUTH_TOKEN",
    "_CLIENT_SECRET",
    "_CREDENTIALS",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)
_SANDBOX_CONTROL_ENV_NAMES = frozenset(
    {_SANDBOX_MODE_ENV, _SANDBOX_READONLY_PATHS_ENV, _SANDBOX_NETWORK_ENV}
)
_SANDBOX_FORBIDDEN_PATH_PARTS = frozenset(
    {".git", ".ssh", "private_gt", "_split_builder"}
)


class SandboxConfigurationError(RuntimeError):
    """Raised before execution when a requested candidate sandbox is unsafe."""


@dataclass(frozen=True)
class _ReadOnlyBind:
    source: Path
    destination: Path
    kind: str

    def to_json(self) -> dict[str, str]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class _SandboxLaunchPlan:
    command_prefix: tuple[str, ...]
    environment: dict[str, str]
    receipt: dict[str, Any]


def _merged_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    if overrides:
        merged.update({str(key): str(value) for key, value in overrides.items()})
    return merged


def _is_sensitive_environment_name(name: str) -> bool:
    upper = str(name).upper()
    return upper in _SANDBOX_SECRET_ENV_NAMES or upper.endswith(
        _SANDBOX_SECRET_ENV_SUFFIXES
    )


def _scrub_sandbox_environment(
    environment: Mapping[str, str],
    *,
    working_dir: Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    scrubbed_names = sorted(
        name for name in environment if _is_sensitive_environment_name(name)
    )
    result = {
        str(name): str(value)
        for name, value in environment.items()
        if name not in scrubbed_names and name not in _SANDBOX_CONTROL_ENV_NAMES
    }
    if working_dir is not None:
        sandbox_root = working_dir / ".biomaker_sandbox"
        home_dir = sandbox_root / "home"
        cache_dir = sandbox_root / "cache"
        result.update(
            {
                "HOME": str(home_dir),
                "XDG_CACHE_HOME": str(cache_dir / "xdg"),
                "HF_HOME": str(cache_dir / "huggingface"),
                "TORCH_HOME": str(cache_dir / "torch"),
                "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
                "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
                "TMPDIR": "/tmp",
                "TMP": "/tmp",
                "TEMP": "/tmp",
            }
        )
    return result, scrubbed_names


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Path, *, parent: Path | None = None) -> Path:
    if not path.is_absolute():
        if parent is None:
            raise ValueError(f"relative path has no parent: {path}")
        path = parent / path
    return Path(os.path.abspath(os.fspath(path)))


def _uv_python_root(path: Path) -> Path | None:
    """Return the uv managed-Python root needed by absolute venv symlinks."""

    parts = path.parts
    for index in range(len(parts) - 3):
        if tuple(parts[index : index + 4]) == (".local", "share", "uv", "python"):
            return Path(*parts[: index + 4])
    return None


def _runtime_readonly_roots() -> list[Path]:
    candidates = [
        _lexical_absolute(Path(sys.prefix)),
        _lexical_absolute(Path(sys.base_prefix)),
    ]
    try:
        resolved_executable = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError):
        resolved_executable = _lexical_absolute(Path(sys.executable))
    for candidate in (resolved_executable, Path(sys.base_prefix)):
        uv_root = _uv_python_root(_lexical_absolute(candidate))
        if uv_root is not None:
            candidates.append(uv_root)

    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            raise SandboxConfigurationError(
                f"Python runtime path required by bwrap does not exist: {candidate}"
            )
        if any(
            _path_is_within(candidate, system_root)
            for system_root in _SANDBOX_SYSTEM_ROOTS
        ):
            continue
        if any(_path_is_within(candidate, root) for root in roots):
            continue
        roots = [root for root in roots if not _path_is_within(root, candidate)]
        roots.append(candidate)
    return sorted(roots, key=str)


def _configured_readonly_roots(environment: Mapping[str, str]) -> list[Path]:
    raw = str(environment.get(_SANDBOX_READONLY_PATHS_ENV, "")).strip()
    if not raw:
        return []
    roots: list[Path] = []
    broad_roots = {
        Path("/"),
        Path("/home"),
        Path("/mnt"),
        Path("/tmp"),
        Path("/var"),
        Path.home().resolve(),
    }
    for item in raw.split(os.pathsep):
        if not item.strip():
            continue
        candidate = Path(item).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise SandboxConfigurationError(
                f"Configured sandbox read-only path is unavailable: {candidate}: {exc}"
            ) from exc
        if candidate in broad_roots:
            raise SandboxConfigurationError(
                f"Configured sandbox read-only path is too broad: {candidate}"
            )
        if any(
            part.casefold() in _SANDBOX_FORBIDDEN_PATH_PARTS for part in candidate.parts
        ):
            raise SandboxConfigurationError(
                f"Configured sandbox read-only path is evaluator-private: {candidate}"
            )
        if candidate not in roots:
            roots.append(candidate)
    return sorted(roots, key=str)


def _bwrap_identity() -> tuple[Path, str]:
    raw_path = shutil.which("bwrap")
    if not raw_path:
        raise SandboxConfigurationError(
            "BIOMAKER_EXECUTION_SANDBOX=bwrap requires bubblewrap on PATH"
        )
    bwrap_path = Path(raw_path).resolve()
    try:
        completed = subprocess.run(
            [str(bwrap_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        raise SandboxConfigurationError(
            f"Could not query bubblewrap version at {bwrap_path}: {exc}"
        ) from exc
    version = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0 or not version:
        raise SandboxConfigurationError(
            f"Could not query bubblewrap version at {bwrap_path}"
        )
    return bwrap_path, version


def execution_sandbox_receipt(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the immutable sandbox policy/runtime facts without launching code.

    Main-9 uses this read-only helper for its protocol fingerprint.  The helper
    deliberately returns environment *names* only; it never records values.
    """

    merged = _merged_environment(environment)
    mode = str(merged.get(_SANDBOX_MODE_ENV, "")).strip().casefold()
    if mode in {"", "0", "disabled", "none"}:
        return {
            "schema_version": 1,
            "policy_version": _SANDBOX_POLICY_VERSION,
            "enabled": False,
            "mode": "disabled",
        }
    if mode != "bwrap":
        raise SandboxConfigurationError(
            f"Unsupported {_SANDBOX_MODE_ENV} value: {mode!r}"
        )
    if platform.system() != "Linux":
        raise SandboxConfigurationError(
            "BIOMAKER_EXECUTION_SANDBOX=bwrap is supported only on Linux"
        )

    bwrap_path, bwrap_version = _bwrap_identity()
    network = str(merged.get(_SANDBOX_NETWORK_ENV, "host")).strip().casefold()
    if network not in {"host", "none"}:
        raise SandboxConfigurationError(
            f"Unsupported {_SANDBOX_NETWORK_ENV} value: {network!r}"
        )
    runtime_roots = _runtime_readonly_roots()
    authorized_roots = _configured_readonly_roots(merged)
    _, scrubbed_names = _scrub_sandbox_environment(merged)
    return {
        "schema_version": 1,
        "policy_version": _SANDBOX_POLICY_VERSION,
        "enabled": True,
        "mode": "bwrap",
        "bwrap_path": str(bwrap_path),
        "bwrap_version": bwrap_version,
        "network": network,
        "pid_namespace": "private",
        "proc": "private",
        "device_policy": "host-dev-bind-for-cuda",
        "workspace_policy": "writable-with-read-only-agent-input-overlay",
        "tmp_policy": "tmpfs",
        "system_readonly_roots": [
            str(path) for path in _SANDBOX_SYSTEM_ROOTS if path.exists()
        ],
        "runtime_readonly_roots": [str(path) for path in runtime_roots],
        "authorized_input_roots": [str(path) for path in authorized_roots],
        "scrubbed_environment_names": scrubbed_names,
        "removed_control_environment_names": sorted(_SANDBOX_CONTROL_ENV_NAMES),
        "secret_name_suffix_policy": list(_SANDBOX_SECRET_ENV_SUFFIXES),
    }


def _symlink_bind(link: Path, *, kind: str) -> _ReadOnlyBind:
    raw_target = Path(os.readlink(link))
    destination = _lexical_absolute(raw_target, parent=link.parent)
    try:
        source = link.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise SandboxConfigurationError(
            f"Agent-visible input contains an unresolved symlink: {link}: {exc}"
        ) from exc
    return _ReadOnlyBind(source=source, destination=destination, kind=kind)


def _contains_forbidden_path_part(path: Path) -> bool:
    return any(part.casefold() in _SANDBOX_FORBIDDEN_PATH_PARTS for part in path.parts)


def _collect_agent_input_binds(
    working_dir: Path,
    authorized_roots: list[Path],
) -> list[_ReadOnlyBind]:
    input_path = working_dir / "input"
    if not input_path.exists():
        raise SandboxConfigurationError(
            f"Sandboxed execution requires an agent-visible input directory: {input_path}"
        )

    if input_path.is_symlink():
        input_bind = _symlink_bind(input_path, kind="agent-input-root")
        if input_bind.source.name != "input":
            raise SandboxConfigurationError(
                "The execution-workspace input symlink does not target the trusted "
                f"run input directory: {input_bind.source}"
            )
    else:
        try:
            input_source = input_path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise SandboxConfigurationError(
                f"Could not resolve agent-visible input directory {input_path}: {exc}"
            ) from exc
        input_bind = _ReadOnlyBind(
            source=input_source,
            destination=input_source,
            kind="agent-input-root",
        )
    if not input_bind.source.is_dir() or _contains_forbidden_path_part(
        input_bind.source
    ):
        raise SandboxConfigurationError(
            f"Unsafe agent-visible input root for sandbox: {input_bind.source}"
        )

    binds = [input_bind]
    stack = [input_bind.source]
    while stack:
        directory = stack.pop()
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise SandboxConfigurationError(
                f"Could not inspect agent-visible input {directory}: {exc}"
            ) from exc
        for child in children:
            if child.is_symlink():
                binding = _symlink_bind(child, kind="agent-input-target")
                if _path_is_within(binding.source, input_bind.source):
                    continue
                if _contains_forbidden_path_part(binding.source):
                    raise SandboxConfigurationError(
                        f"Agent-visible input points at evaluator-private data: {child}"
                    )
                if not any(
                    _path_is_within(binding.source, root) for root in authorized_roots
                ):
                    raise SandboxConfigurationError(
                        "Agent-visible input symlink target is outside "
                        f"{_SANDBOX_READONLY_PATHS_ENV}: {child} -> {binding.source}"
                    )
                binds.append(binding)
                continue
            if child.is_dir():
                stack.append(child)

    by_destination: dict[Path, _ReadOnlyBind] = {}
    for binding in binds:
        previous = by_destination.get(binding.destination)
        if previous is not None and previous.source != binding.source:
            raise SandboxConfigurationError(
                "Conflicting agent-visible sandbox mounts for "
                f"{binding.destination}: {previous.source} vs {binding.source}"
            )
        by_destination[binding.destination] = binding
    if len(by_destination) > 4096:
        raise SandboxConfigurationError(
            "Agent-visible input needs too many sandbox bind mounts; materialize "
            "the public input view before formal execution"
        )
    return sorted(by_destination.values(), key=lambda item: str(item.destination))


def _collect_framework_readonly_binds(working_dir: Path) -> list[_ReadOnlyBind]:
    """Keep the evaluator contract visible without exposing its parent run tree."""

    binds: list[_ReadOnlyBind] = []
    for name in ("evaluator.py", "evaluator_prediction_spec.json"):
        path = working_dir / name
        if not path.exists():
            continue
        if path.is_symlink():
            binding = _symlink_bind(path, kind="framework-contract")
        else:
            binding = _ReadOnlyBind(
                source=path.resolve(strict=True),
                destination=path.resolve(strict=True),
                kind="framework-contract",
            )
        if (
            not binding.source.is_file()
            or binding.source.name != name
            or _contains_forbidden_path_part(binding.source)
        ):
            raise SandboxConfigurationError(
                f"Unsafe framework contract file for sandbox: {path}"
            )
        binds.append(binding)
    return binds


def _system_readonly_binds() -> list[_ReadOnlyBind]:
    binds = [
        _ReadOnlyBind(path, path, "system-runtime")
        for path in _SANDBOX_SYSTEM_ROOTS
        if path.exists()
    ]
    resolv_conf = Path("/etc/resolv.conf")
    if resolv_conf.is_symlink():
        binding = _symlink_bind(resolv_conf, kind="network-resolution")
        if not any(
            _path_is_within(binding.source, root) for root in _SANDBOX_SYSTEM_ROOTS
        ):
            binds.append(binding)
    return binds


def _sandbox_launch_plan(
    working_dir: Path,
    command: list[str],
    environment: Mapping[str, str],
) -> _SandboxLaunchPlan | None:
    receipt = execution_sandbox_receipt(environment)
    if not receipt["enabled"]:
        return None

    working_dir = working_dir.resolve(strict=True)
    authorized_roots = [
        Path(value) for value in receipt.get("authorized_input_roots", [])
    ]
    input_binds = _collect_agent_input_binds(working_dir, authorized_roots)
    framework_binds = _collect_framework_readonly_binds(working_dir)
    system_binds = _system_readonly_binds()
    runtime_binds = [
        _ReadOnlyBind(Path(path), Path(path), "python-runtime")
        for path in receipt["runtime_readonly_roots"]
    ]

    sandbox_root = working_dir / ".biomaker_sandbox"
    writable_dirs = [
        sandbox_root / "home",
        sandbox_root / "cache" / "xdg",
        sandbox_root / "cache" / "huggingface",
        sandbox_root / "cache" / "torch",
        sandbox_root / "cache" / "matplotlib",
        sandbox_root / "cache" / "numba",
    ]
    for path in writable_dirs:
        path.mkdir(parents=True, exist_ok=True)

    sandbox_environment, scrubbed_names = _scrub_sandbox_environment(
        environment,
        working_dir=working_dir,
    )
    prefix = [
        str(receipt["bwrap_path"]),
        "--die-with-parent",
        "--unshare-pid",
        "--proc",
        "/proc",
        "--dev-bind",
        "/dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    if receipt["network"] == "none":
        # This affects candidate execution only. Host-side provider/retrieval
        # calls retain their network, but candidates cannot fetch sibling test
        # frames or public ground truth outside the sealed TaskPack.
        prefix.append("--unshare-net")
    for binding in [*system_binds, *runtime_binds]:
        prefix.extend(["--ro-bind", str(binding.source), str(binding.destination)])
    prefix.extend(["--bind", str(working_dir), str(working_dir)])
    for binding in [*input_binds, *framework_binds]:
        prefix.extend(["--ro-bind", str(binding.source), str(binding.destination)])
    prefix.extend(["--chdir", str(working_dir), "--"])

    execution_receipt = {
        **receipt,
        "workspace": str(working_dir),
        "writable_roots": [str(working_dir)],
        "agent_input_readonly_binds": [bind.to_json() for bind in input_binds],
        "framework_readonly_binds": [bind.to_json() for bind in framework_binds],
        "effective_scrubbed_environment_names": scrubbed_names,
        "sandbox_home": str(sandbox_root / "home"),
        "sandbox_cache": str(sandbox_root / "cache"),
        "command_argv": [str(value) for value in command],
    }
    return _SandboxLaunchPlan(
        command_prefix=tuple(prefix),
        environment=sandbox_environment,
        receipt=execution_receipt,
    )


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
    """Decode and redact output incrementally without waiting for the process."""

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

    def read_available(self, stream, *, final: bool = False) -> None:
        # pread does not move the shared file offset used by the child writer.
        # Reading a binary spool also preserves partial UTF-8 sequences.
        end = os.fstat(stream.fileno()).st_size
        while self.offset < end:
            data = os.pread(stream.fileno(), min(65536, end - self.offset), self.offset)
            if not data:
                break
            self.offset += len(data)
            for start in range(0, len(data), 4096):
                self._emit(self.decoder.decode(data[start:start + 4096]))
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
    sandbox_receipt: dict[str, Any] | None = None
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
            "sandbox": self.sandbox_receipt,
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
        source = resource_usage.get("gpu_memory_source")
        if source == "unified_memory_rss_proxy":
            parts.append("GPU memory not independently measured (unified-memory RSS proxy)")
        else:
            if source == "nvidia-smi_total_used_delta_approx":
                name = "peak_gpu_total_delta_approx"
            elif source == "nvidia-smi_compute_apps_process_tree":
                name = "peak_gpu_process_tree"
            else:
                name = f"recorded_peak_{backend or 'gpu'} (source unconfirmed)"
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
        self.output_label = output_label or self.working_dir.name

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
                code, include_medical=self.include_medical_packages, working_dir=self.working_dir
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

        command = (
            [sys.executable, "-u", "-c", _IPYTHON_TRACEBACK_WRAPPER, self.agent_file_name]
            if self.format_tb_ipython
            else [sys.executable, "-u", self.agent_file_name]
        )
        merged_environment = {**_merged_environment(self.environment), "PYTHONUNBUFFERED": "1"}
        try:
            sandbox_plan = _sandbox_launch_plan(
                self.working_dir,
                command,
                merged_environment,
            )
        except SandboxConfigurationError as exc:
            message = f"SandboxConfigurationError: {exc}"
            return ExecutionResult(
                term_out=[message],
                exec_time=0.0,
                exc_type="SandboxConfigurationError",
                exc_info={"message": str(exc)},
                exc_stack=None,
                resource_usage=None,
                stderr=message,
                candidate_output=message,
                candidate_exec_time=0.0,
                candidate_exc_type="SandboxConfigurationError",
                candidate_exc_info={"message": str(exc)},
                failure_phase="environment_preflight",
            )
        if sandbox_plan is not None:
            command = [*sandbox_plan.command_prefix, *command]

        script_path = self.working_dir / self.agent_file_name
        script_path.write_text(code, encoding="utf-8")
        start_time = time.time()
        usage_samples: list[dict] = []
        gpu_baseline_mb = _sample_cuda_total_gpu_memory_mb()
        process: subprocess.Popen[str] | None = None
        process_tree: ProcessTreeTracker | None = None
        stdout_capture: Any | None = None
        stderr_capture: Any | None = None
        uncaptured_drained = False
        cleanup_failed = False
        live_stdout = _LiveProcessOutput(self.output_label, "stdout", merged_environment)
        live_stderr = _LiveProcessOutput(self.output_label, "stderr", merged_environment)
        capture_stack = ExitStack()
        try:
            stdout_capture = capture_stack.enter_context(
                tempfile.TemporaryFile(mode="w+b")
            )
            stderr_capture = capture_stack.enter_context(
                tempfile.TemporaryFile(mode="w+b")
            )
            popen_kwargs: dict[str, Any] = {
                "cwd": self.working_dir,
                "stdout": stdout_capture,
                "stderr": stderr_capture,
                "text": False,
                "env": (
                    sandbox_plan.environment
                    if sandbox_plan is not None
                    else (
                        merged_environment
                    )
                ),
            }
            if os.name == "posix":
                # A candidate-owned PGID survives leader exit, while remaining in the
                # top-level run session for session-wide Stop handling.
                popen_kwargs["process_group"] = 0
            process = subprocess.Popen(command, **popen_kwargs)
            try:
                process_tree = ProcessTreeTracker.capture(process.pid)
            except ProcessLookupError:
                # A fast child may exit before macOS exposes its full identity.
                # Keep the real exit result only after its group is drained.
                uncaptured_drained = True
                drain_uncaptured_process_group(process)
            except (
                OSError,
                ProcessTreeInspectionError,
                ValueError,
            ):
                drain_uncaptured_process_group(process)
                uncaptured_drained = True
                raise RuntimeError(
                    "Could not capture generated process identity"
                ) from None
            stdout = ""
            stderr = ""
            timed_out = False
            next_gpu_sample_time = 0.0
            while True:
                live_stdout.read_available(stdout_capture)
                live_stderr.read_available(stderr_capture)
                if process_tree is not None:
                    process_tree.refresh_best_effort()
                if uncaptured_drained or process_tree.root_has_exited(process):
                    if process_tree is not None:
                        process_tree.terminate_and_wait(root_process=process)
                    live_stdout.read_available(stdout_capture, final=True)
                    live_stderr.read_available(stderr_capture, final=True)
                    stdout = _decode_process_output(os.pread(stdout_capture.fileno(), os.fstat(stdout_capture.fileno()).st_size, 0))
                    stderr = _decode_process_output(os.pread(stderr_capture.fileno(), os.fstat(stderr_capture.fileno()).st_size, 0))
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
                    live_stdout.read_available(stdout_capture, final=True)
                    live_stderr.read_available(stderr_capture, final=True)
                    stdout = _decode_process_output(os.pread(stdout_capture.fileno(), os.fstat(stdout_capture.fileno()).st_size, 0))
                    stderr = _decode_process_output(os.pread(stderr_capture.fileno(), os.fstat(stderr_capture.fileno()).st_size, 0))
                    break

                time.sleep(min(0.25, max(0.05, remaining)))

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
                    sandbox_receipt=(
                        sandbox_plan.receipt if sandbox_plan is not None else None
                    ),
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
                    sandbox_receipt=(
                        sandbox_plan.receipt if sandbox_plan is not None else None
                    ),
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
                sandbox_receipt=(
                    sandbox_plan.receipt if sandbox_plan is not None else None
                ),
            )
        except ProcessTreeCleanupError:
            cleanup_failed = True
            raise
        finally:
            try:
                if cleanup_failed:
                    pass
                elif process_tree is not None:
                    process_tree.terminate_and_wait(root_process=process)
                elif process is not None and not uncaptured_drained:
                    drain_uncaptured_process_group(process)
            except ProcessTreeCleanupError:
                cleanup_failed = True
                raise
            finally:
                capture_stack.close()
                if not cleanup_failed:
                    try:
                        script_path.unlink()
                    except FileNotFoundError:
                        pass
