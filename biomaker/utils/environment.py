"""Environment policy for generated solution code."""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import platform
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageSpec:
    import_name: str
    distribution: str
    display_name: str
    medical: bool = False


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    message: str = ""
    error_type: str | None = None
    exc_info: dict | None = None


PACKAGE_SPECS = [
    PackageSpec("numpy", "numpy", "numpy"),
    PackageSpec("pandas", "pandas", "pandas"),
    PackageSpec("sklearn", "scikit-learn", "scikit-learn"),
    PackageSpec("scipy", "scipy", "scipy"),
    PackageSpec("skimage", "scikit-image", "scikit-image", medical=True),
    PackageSpec("matplotlib", "matplotlib", "matplotlib"),
    PackageSpec("torch", "torch", "torch"),
    PackageSpec("torchvision", "torchvision", "torchvision"),
    PackageSpec("timm", "timm", "timm"),
    PackageSpec("xgboost", "xgboost", "xgboost"),
    PackageSpec("lightgbm", "lightgbm", "lightgbm"),
    PackageSpec("PIL", "Pillow", "Pillow"),
    PackageSpec("openpyxl", "openpyxl", "openpyxl"),
    PackageSpec("monai", "monai", "monai", medical=True),
    PackageSpec("SimpleITK", "SimpleITK", "SimpleITK", medical=True),
    PackageSpec("nibabel", "nibabel", "nibabel", medical=True),
    PackageSpec("pydicom", "pydicom", "pydicom", medical=True),
    PackageSpec("lifelines", "lifelines", "lifelines", medical=True),
]

BLOCKED_EXAMPLES = [
    "albumentations",
    "cv2",
    "fastai",
    "tensorflow",
    "keras",
    "segmentation_models_pytorch",
]

LOCAL_GENERATED_MODULES = {"evaluator"}

INSTALL_COMMAND_RE = re.compile(
    r"\b(pip|conda|mamba|apt-get|brew)\s+(install|add)\b|\buv\s+(pip\s+)?(install|add)\b",
    re.IGNORECASE,
)
PROCESS_ESCAPE_COMMAND_RE = re.compile(
    r"\b(?:setsid|setpgid|setpgrp|start_new_session|process_group|preexec_fn)\b"
)


def allowed_package_specs(include_medical: bool = True) -> list[PackageSpec]:
    return [spec for spec in PACKAGE_SPECS if include_medical or not spec.medical]


def allowed_import_names(include_medical: bool = True) -> set[str]:
    return {spec.import_name for spec in allowed_package_specs(include_medical)}


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _torch_accelerator_info() -> str:
    policy = "Use the best available PyTorch device in this order: CUDA, then Apple MPS, then CPU."
    try:
        import torch
    except Exception:  # noqa: BLE001 - optional framework probing is best effort
        return f"PyTorch accelerator: unavailable because torch is not importable. {policy}"

    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            devices.append(f"{idx}: {props.name}, {total_gb:.1f} GB")
        return (
            "PyTorch accelerator: CUDA is available with "
            f"{torch.cuda.device_count()} device(s) ({'; '.join(devices)}). "
            f"{policy}"
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return (
            "PyTorch accelerator: Apple MPS is available on this machine. "
            "Apple MPS uses unified system memory; see Host resources for the memory pool shared by the OS, Python, data loading, and model tensors. "
            f"{policy}"
        )
    return f"PyTorch accelerator: no GPU backend detected on this machine. {policy}"


def _system_resource_info() -> str:
    parts = [
        f"machine={platform.machine() or 'unknown'}",
        f"cpu_cores={os.cpu_count() or 'unknown'}",
    ]
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_gb = pages * page_size / (1024**3)
        parts.append(f"system_memory={total_gb:.1f} GB")
    except (AttributeError, ValueError, OSError):
        parts.append("system_memory=unknown")
    return "Host resources: " + ", ".join(parts) + "."


def environment_manifest_text(include_medical: bool = True) -> str:
    packages = [
        f"- {spec.display_name} (import `{spec.import_name}`): {_package_version(spec.distribution)}"
        for spec in allowed_package_specs(include_medical)
    ]
    blocked = ", ".join(f"`{name}`" for name in BLOCKED_EXAMPLES)
    return "\n".join(
        [
            f"Python: {sys.version.split()[0]}",
            _system_resource_info(),
            _torch_accelerator_info(),
            "Allowed third-party packages:",
            *packages,
            "This list describes what is available in the environment; it is not a recommendation to use any specific package.",
            "",
            "Allowed local generated modules:",
            "- `evaluator`: frozen BioMaker evaluator generated before solution search. Candidate code may import it for validation monitoring and artifact-format guidance when it exists.",
            "",
            "Dependency policy:",
            "- Use only the allowed third-party packages above and the Python standard library.",
            "- Do not import unlisted third-party packages.",
            "- Do not install packages or call package managers from generated code.",
            "- Keep subprocesses inside BioMaker's managed process group. Do not call setsid/setpgid, request a new subprocess session or process group, or use preexec_fn.",
            "- Downloading pretrained model weights is allowed when it is done through an installed library and does not install or modify Python packages.",
            "- If a useful package is unavailable, implement the smallest replacement with allowed packages.",
            "- PyTorch code must be portable across CUDA, Apple MPS, and CPU. Select the device dynamically and never hard-code a single accelerator.",
            f"- Common unavailable examples to avoid: {blocked}.",
        ]
    )


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names: list[str] = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [item.id for item in handler.type.elts if isinstance(item, ast.Name)]
    return any(
        name in {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
        for name in names
    )


class _ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.required: set[str] = set()
        self.optional: set[str] = set()
        self.opaque_dynamic_import_lines: list[int] = []
        self._optional_depth = 0

    def _add(self, name: str) -> None:
        target = self.optional if self._optional_depth else self.required
        target.add(name.split(".", 1)[0])

    def _add_dynamic_module_name(self, node: ast.AST, name_node: ast.AST) -> None:
        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            module_name = name_node.value.lstrip(".")
            if module_name:
                self._add(module_name)
            return
        self.opaque_dynamic_import_lines.append(getattr(node, "lineno", 0))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._add(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name in {"__import__", "importlib.import_module"} and node.args:
            self._add_dynamic_module_name(node, node.args[0])
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        catches_import_error = any(
            _handler_catches_import_error(handler) for handler in node.handlers
        )
        if catches_import_error:
            self._optional_depth += 1
            for child in node.body:
                self.visit(child)
            self._optional_depth -= 1
        else:
            for child in node.body:
                self.visit(child)
        for handler in node.handlers:
            for child in handler.body:
                self.visit(child)
        for child in [*node.orelse, *node.finalbody]:
            self.visit(child)


def _top_level_imports(tree: ast.AST) -> tuple[set[str], set[str], list[int]]:
    collector = _ImportCollector()
    collector.visit(tree)
    return (
        collector.required,
        collector.optional - collector.required,
        collector.opaque_dynamic_import_lines,
    )


def _is_stdlib_import(name: str) -> bool:
    if name in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return name in stdlib_names


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _constant_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
    if isinstance(node, (ast.List, ast.Tuple)):
        strings: list[str] = []
        for item in node.elts:
            strings.extend(_constant_strings(item))
        return strings
    return []


def _assigned_name_targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _resolved_static_strings(
    node: ast.AST,
    bindings: dict[str, list[str]],
) -> list[str]:
    if isinstance(node, ast.Name):
        return list(bindings.get(node.id, []))
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for item in node.elts:
            values.extend(_resolved_static_strings(item, bindings))
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolved_static_strings(node.left, bindings)
        right = _resolved_static_strings(node.right, bindings)
        if len(left) == 1 and len(right) == 1:
            return [left[0] + right[0]]
    return []


def _static_string_bindings(
    tree: ast.AST,
    *,
    before_line: int | None = None,
) -> dict[str, list[str]]:
    bindings: dict[str, list[str]] = {}
    assignments = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )
    for assignment in assignments:
        if before_line is not None and getattr(assignment, "lineno", 0) > before_line:
            continue
        value = assignment.value
        if value is None:
            continue
        strings = _resolved_static_strings(value, bindings)
        for name in _assigned_name_targets(assignment):
            if strings:
                bindings[name] = strings
            else:
                bindings.pop(name, None)
    return bindings


def _contains_install_command(tree: ast.AST) -> bool:
    shell_call_names = {
        "os.system",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in {"pip.main", "pip._internal.main"}:
            return True
        if call_name not in shell_call_names:
            continue
        for arg in node.args:
            strings = _constant_strings(arg)
            if any(INSTALL_COMMAND_RE.search(text) for text in strings):
                return True
            if strings and INSTALL_COMMAND_RE.search(" ".join(strings)):
                return True
    return False


_PROCESS_ESCAPE_CALLS = {"setsid", "setpgid", "setpgrp"}
_SUBPROCESS_LAUNCHERS = {"Popen", "call", "check_call", "check_output", "run"}
_OS_LAUNCHERS = {"popen", "system"}


def _getattr_escape_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    call_name = _call_name(node.func) or ""
    if call_name.rsplit(".", 1)[-1] != "getattr":
        return None
    attribute = node.args[1]
    if not isinstance(attribute, ast.Constant) or not isinstance(attribute.value, str):
        return None
    return attribute.value if attribute.value in _PROCESS_ESCAPE_CALLS else None


def _escape_callable_aliases(tree: ast.AST) -> set[str]:
    aliases = set(_PROCESS_ESCAPE_CALLS)
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"os", "posix"}:
            for imported in node.names:
                if imported.name in _PROCESS_ESCAPE_CALLS:
                    aliases.add(imported.asname or imported.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)

    # Resolve straightforward chains such as ``detach = os.setsid`` and
    # ``again = detach`` without pretending to perform general data-flow analysis.
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            call_name = _call_name(value)
            is_escape = (
                bool(
                    call_name
                    and (
                        call_name in aliases
                        or call_name.rsplit(".", 1)[-1] in _PROCESS_ESCAPE_CALLS
                    )
                )
                or _getattr_escape_name(value) is not None
            )
            if not is_escape:
                continue
            for name in _assigned_name_targets(assignment):
                if name not in aliases:
                    aliases.add(name)
                    changed = True
    return aliases


def _process_launcher_names(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    subprocess_modules = {"subprocess"}
    os_modules = {"os"}
    direct_launchers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "subprocess":
                    subprocess_modules.add(imported.asname or imported.name)
                elif imported.name == "os":
                    os_modules.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                for imported in node.names:
                    if imported.name in _SUBPROCESS_LAUNCHERS:
                        direct_launchers.add(imported.asname or imported.name)
            elif node.module == "os":
                for imported in node.names:
                    if imported.name in _OS_LAUNCHERS:
                        direct_launchers.add(imported.asname or imported.name)
    return subprocess_modules, os_modules, direct_launchers


def _root_call_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_process_launcher(
    node: ast.Call,
    *,
    subprocess_modules: set[str],
    os_modules: set[str],
    direct_launchers: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in direct_launchers
    if not isinstance(node.func, ast.Attribute):
        return False
    root = _root_call_name(node.func)
    return (root in subprocess_modules and node.func.attr in _SUBPROCESS_LAUNCHERS) or (
        root in os_modules and node.func.attr in _OS_LAUNCHERS
    )


def _process_containment_violation(tree: ast.AST) -> tuple[str, int] | None:
    blocked_keywords = {"start_new_session", "process_group", "preexec_fn"}
    escape_aliases = _escape_callable_aliases(tree)
    subprocess_modules, os_modules, direct_launchers = _process_launcher_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func) or ""
        escape_name = _getattr_escape_name(node.func)
        if (
            call_name in escape_aliases
            or call_name.rsplit(".", 1)[-1] in _PROCESS_ESCAPE_CALLS
            or escape_name is not None
        ):
            displayed_name = escape_name or call_name
            return (
                f"`{displayed_name}` would move a child outside BioMaker's managed process group.",
                getattr(node, "lineno", 0),
            )
        for keyword in node.keywords:
            if keyword.arg not in blocked_keywords:
                continue
            safe_default = (
                keyword.arg == "start_new_session"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            ) or (
                keyword.arg in {"process_group", "preexec_fn"}
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            )
            if not safe_default:
                return (
                    f"`{keyword.arg}` would let a child escape BioMaker's managed process group.",
                    getattr(node, "lineno", 0),
                )
        if not _is_process_launcher(
            node,
            subprocess_modules=subprocess_modules,
            os_modules=os_modules,
            direct_launchers=direct_launchers,
        ):
            continue
        string_bindings = _static_string_bindings(
            tree,
            before_line=getattr(node, "lineno", 0),
        )
        for arg in node.args:
            if any(
                PROCESS_ESCAPE_COMMAND_RE.search(text)
                for text in _resolved_static_strings(arg, string_bindings)
            ):
                return (
                    "a nested process command can escape BioMaker's managed process group.",
                    getattr(node, "lineno", 0),
                )
    return None


def validate_process_containment(code: str) -> PreflightResult:
    """Reject generated code that can leave its candidate-owned process group."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PreflightResult(
            ok=False,
            error_type="SyntaxError",
            message=(
                f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}."
            ),
            exc_info={
                "msg": exc.msg,
                "lineno": str(exc.lineno),
                "offset": str(exc.offset),
            },
        )
    violation = _process_containment_violation(tree)
    if violation is None:
        return PreflightResult(ok=True)
    reason, line = violation
    return PreflightResult(
        ok=False,
        error_type="ProcessContainmentError",
        message=(
            f"ProcessContainmentError at line {line}: {reason} "
            "Use ordinary subprocesses; they inherit the assigned compute slot and "
            "BioMaker will clean them before releasing it."
        ),
        exc_info={"lineno": str(line), "reason": reason},
    )


def _is_truthy_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


def _is_not_none(node: ast.AST) -> bool:
    return not (isinstance(node, ast.Constant) and node.value is None)


def _uses_external_weight_download(tree: ast.AST) -> bool:
    segmentation_model_names = {
        "deeplabv3_mobilenet_v3_large",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "fcn_resnet50",
        "fcn_resnet101",
        "lraspp_mobilenet_v3_large",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        keyword_names = {keyword.arg for keyword in node.keywords}
        for keyword in node.keywords:
            if keyword.arg == "pretrained" and _is_truthy_constant(keyword.value):
                return True
            if keyword.arg == "weights" and _is_not_none(keyword.value):
                return True
            if keyword.arg in {
                "weights_backbone",
                "pretrained_backbone",
            } and _is_not_none(keyword.value):
                return True

        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if (
            func_name in segmentation_model_names
            and "weights_backbone" not in keyword_names
        ):
            return True

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "hub"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "torch"
        ):
            return True
    return False


def _portable_device_violation(tree: ast.AST) -> tuple[str, int] | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "cuda":
            return (
                "Use dynamic device selection and `.to(device)` instead of `.cuda()`, so the code also runs on MPS and CPU.",
                getattr(node, "lineno", 0),
            )

    return None


def validate_code_environment(
    code: str, include_medical: bool = True
) -> PreflightResult:
    """Validate generated code against the environment contract before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return PreflightResult(
            ok=False,
            error_type="SyntaxError",
            message=f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}.",
            exc_info={"msg": e.msg, "lineno": str(e.lineno), "offset": str(e.offset)},
        )

    containment_error = _process_containment_violation(tree)
    if containment_error is not None:
        reason, line = containment_error
        return PreflightResult(
            ok=False,
            error_type="ProcessContainmentError",
            message=(
                f"ProcessContainmentError at line {line}: {reason} "
                "Use ordinary subprocesses; they inherit the assigned compute slot and "
                "BioMaker will clean them before releasing it."
            ),
            exc_info={"lineno": str(line), "reason": reason},
        )

    if _contains_install_command(tree):
        return PreflightResult(
            ok=False,
            error_type="DependencyPolicyError",
            message=(
                "DependencyPolicyError: generated code attempted to install packages. "
                "Package installation is disabled. Remove package-manager calls and use only allowed packages."
            ),
            exc_info={"reason": "package installation is disabled"},
        )

    portable_device_error = _portable_device_violation(tree)
    if portable_device_error is not None:
        message, lineno = portable_device_error
        return PreflightResult(
            ok=False,
            error_type="DevicePortabilityError",
            message=(
                f"DevicePortabilityError at line {lineno}: {message} "
                "Use a dynamic fallback such as CUDA if available, else MPS if available, else CPU, "
                "and move model/tensors through the selected `device`."
            ),
            exc_info={"lineno": str(lineno), "reason": message},
        )

    allowed_imports = allowed_import_names(include_medical)
    required_imports, optional_imports, opaque_dynamic_import_lines = (
        _top_level_imports(tree)
    )
    if opaque_dynamic_import_lines:
        line = opaque_dynamic_import_lines[0]
        return PreflightResult(
            ok=False,
            error_type="DependencyPolicyError",
            message=(
                f"DependencyPolicyError at line {line}: dynamic import with a non-literal module name is disabled. "
                "Use normal import statements or literal module names so BioMaker can verify the dependency policy."
            ),
            exc_info={
                "lineno": str(line),
                "reason": "dynamic import module name is not statically verifiable",
            },
        )
    for import_name in sorted(required_imports):
        if _is_stdlib_import(import_name):
            continue

        if import_name in LOCAL_GENERATED_MODULES:
            continue

        if import_name in allowed_imports:
            if importlib.util.find_spec(import_name) is not None:
                continue
            return PreflightResult(
                ok=False,
                error_type="DependencyPolicyError",
                message=(
                    f"DependencyPolicyError: package `{import_name}` is allowed by policy but is not installed "
                    "in the current Python environment. Install BioMaker requirements before running, or replace "
                    "this import with an installed allowed package."
                ),
                exc_info={
                    "package": import_name,
                    "reason": "allowed package not installed",
                },
            )

        allowed = ", ".join(sorted(f"`{name}`" for name in allowed_imports))
        return PreflightResult(
            ok=False,
            error_type="DependencyPolicyError",
            message=(
                f"DependencyPolicyError: package `{import_name}` is not allowed in this environment. "
                f"Allowed third-party imports are: {allowed}. "
                "Replace the unavailable package with Python standard library or an allowed package."
            ),
            exc_info={
                "package": import_name,
                "allowed_imports": sorted(allowed_imports),
            },
        )

    for import_name in sorted(optional_imports):
        if _is_stdlib_import(import_name) or import_name in LOCAL_GENERATED_MODULES:
            continue
        if import_name in allowed_imports:
            continue
        allowed = ", ".join(sorted(f"`{name}`" for name in allowed_imports))
        return PreflightResult(
            ok=False,
            error_type="DependencyPolicyError",
            message=(
                f"DependencyPolicyError: optional fallback import `{import_name}` is not allowed in this environment. "
                f"Allowed third-party imports are: {allowed}. "
                "Remove the unavailable package import and keep the fallback implementation."
            ),
            exc_info={
                "package": import_name,
                "allowed_imports": sorted(allowed_imports),
            },
        )

    return PreflightResult(ok=True)
