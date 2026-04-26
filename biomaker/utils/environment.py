"""Environment policy for generated solution code."""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
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
    "skimage",
]

LOCAL_GENERATED_MODULES = {"evaluator"}

INSTALL_COMMAND_RE = re.compile(
    r"\b(pip|conda|mamba|uv|apt-get|brew)\s+(install|add)\b", re.IGNORECASE
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
    except Exception:
        return f"PyTorch accelerator: unavailable because torch is not importable. {policy}"

    if torch.cuda.is_available():
        return (
            f"PyTorch accelerator: CUDA is available with {torch.cuda.device_count()} device(s). "
            f"{policy}"
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return f"PyTorch accelerator: Apple MPS is available on this machine. {policy}"
    return f"PyTorch accelerator: no GPU backend detected on this machine. {policy}"


def environment_manifest_text(include_medical: bool = True) -> str:
    packages = [
        f"- {spec.display_name} (import `{spec.import_name}`): {_package_version(spec.distribution)}"
        for spec in allowed_package_specs(include_medical)
    ]
    blocked = ", ".join(f"`{name}`" for name in BLOCKED_EXAMPLES)
    return "\n".join(
        [
            f"Python: {sys.version.split()[0]}",
            _torch_accelerator_info(),
            "Allowed third-party packages:",
            *packages,
            "",
            "Allowed local generated modules:",
            "- `evaluator`: frozen BioMaker evaluator generated before solution search. Candidate code may import it for validation monitoring and artifact-format guidance when it exists.",
            "",
            "Dependency policy:",
            "- Use only the allowed third-party packages above and the Python standard library.",
            "- Do not import unlisted third-party packages.",
            "- Do not install packages or call package managers from generated code.",
            "- Downloading pretrained model weights is allowed when it uses installed libraries such as torchvision, timm, torch.hub, or MONAI. Do not download or install new Python packages.",
            "- If a useful package is unavailable, implement the smallest replacement with allowed packages.",
            f"- Common unavailable examples to avoid: {blocked}.",
        ]
    )


def _top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def _is_stdlib_import(name: str) -> bool:
    if name in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return name in stdlib_names


def _contains_install_command(code: str) -> bool:
    return INSTALL_COMMAND_RE.search(code) is not None


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


def validate_code_environment(
    code: str, include_medical: bool = True
) -> PreflightResult:
    """Validate generated code against the environment contract before execution."""
    if _contains_install_command(code):
        return PreflightResult(
            ok=False,
            error_type="DependencyPolicyError",
            message=(
                "DependencyPolicyError: generated code attempted to install packages. "
                "Package installation is disabled. Remove package-manager calls and use only allowed packages."
            ),
            exc_info={"reason": "package installation is disabled"},
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return PreflightResult(
            ok=False,
            error_type="SyntaxError",
            message=f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}.",
            exc_info={"msg": e.msg, "lineno": str(e.lineno), "offset": str(e.offset)},
        )

    allowed_imports = allowed_import_names(include_medical)
    for import_name in sorted(_top_level_imports(tree)):
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

    return PreflightResult(ok=True)
