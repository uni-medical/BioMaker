"""Local candidate imports are real source, never global package exemptions."""

from pathlib import Path

import pytest

from biomaker.interpreter import Interpreter
from biomaker.utils.environment import validate_code_environment


def _generated(source="VALUE = 42\n"):
    # Same creation/import pattern as the failed real candidate; synthetic code.
    return (
        "from pathlib import Path\nimport sys\n"
        "WORK_DIR = Path('./working')\n"
        f"INFERENCE_SRC = {source!r}\n"
        "def main():\n"
        "    WORK_DIR.mkdir(exist_ok=True)\n"
        "    (WORK_DIR / 'inference.py').write_text(INFERENCE_SRC, encoding='utf-8')\n"
        "    work_abs = str(WORK_DIR.resolve())\n"
        "    if work_abs not in sys.path:\n"
        "        sys.path.insert(0, work_abs)\n"
        "    import inference as inference_mod\n"
        "    print(inference_mod.VALUE)\n"
        "main()\n"
    )


def _check(code, tmp_path):
    return validate_code_environment(code, working_dir=tmp_path)


def _text_written(source="VALUE = 42\n", *, pathlib=False, mode="w", body=None):
    path = "Path(WORK_DIR).joinpath('inference.py')" if pathlib else "os.path.join(WORK_DIR, 'inference.py')"
    opener = f"inference_path.open({mode!r}, encoding='utf-8')" if pathlib else f"open(inference_path, {mode!r})"
    body = body or "f.write(inference_src)"
    return (
        "import os, sys, importlib\nfrom pathlib import Path\n"
        "WORK_DIR = './working'\n"
        "def main():\n"
        "    os.makedirs(WORK_DIR, exist_ok=True)\n"
        f"    inference_src = {source!r}\n"
        f"    inference_path = {path}\n"
        f"    with {opener} as f:\n"
        + "".join(f"        {line}\n" for line in body.splitlines())
        + "    try:\n"
        "        sys.path.insert(0, os.path.abspath(WORK_DIR))\n"
        "        inf = importlib.import_module('inference')\n"
        "        importlib.reload(inf)\n"
        "        print(inf.VALUE)\n"
        "    except ImportError:\n"
        "        raise\n"
        "main()\n"
    )


@pytest.mark.parametrize("pathlib", [False, True])
@pytest.mark.parametrize("mode", ["w", "wt"])
def test_proven_closed_text_module_is_admitted_without_writes(tmp_path, pathlib, mode):
    result = _check(_text_written(pathlib=pathlib, mode=mode), tmp_path)
    assert result.ok, result.message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("pathlib", [False, True])
def test_interpreter_executes_closed_text_module(tmp_path, monkeypatch, pathlib):
    from biomaker import interpreter
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    monkeypatch.setattr(interpreter, "_sample_process_usage", lambda *a, **k: None)
    result = Interpreter(tmp_path, timeout=10).run(
        "import time\ntime.sleep(0.5)\n" + _text_written(pathlib=pathlib)
    )
    assert result.exc_type is None, result.term_out
    assert "42" in result.stdout
    assert (tmp_path / "working/inference.py").read_text() == "VALUE = 42\n"


@pytest.mark.parametrize("pathlib", [False, True])
@pytest.mark.parametrize("source", ["import cv2", "import helper", "import os\nos.setsid()", "def broken(:"])
def test_text_written_source_is_recursively_checked(tmp_path, pathlib, source):
    result = _check(_text_written(source, pathlib=pathlib), tmp_path)
    assert not result.ok and "local module" in result.message


def test_multiple_literal_text_writes_are_checked_as_complete_source(tmp_path):
    result = _check(_text_written(body="f.write('VALUE = ')\nf.write('42\\n')"), tmp_path)
    assert result.ok, result.message
    result = _check(_text_written(body="f.write('import c')\nf.write('v2\\n')"), tmp_path)
    assert not result.ok and "cv2" in result.message


@pytest.mark.parametrize("body", [
    "f.write(input())", "f.write(inference_src)\nf.write(input())",
    "if input():\n    f.write(inference_src)",
    "f.write(inference_src)\nf.seek(0)",
    "f.write(inference_src)\nf.truncate(1)",
    "g = f\ng.write(inference_src)",
    "f.write('import cv2')\nPath(inference_path).write_text(inference_src)",
])
def test_uncertain_text_writer_cannot_trust_existing_source(tmp_path, body):
    (tmp_path / "working").mkdir()
    (tmp_path / "working/inference.py").write_text("VALUE=42")
    result = _check(_text_written(body=body), tmp_path)
    assert not result.ok and "not statically verifiable" in result.message


@pytest.mark.parametrize("mode", ["a", "x", "wb", "w+", "a+"])
def test_unproven_text_modes_are_not_admitted(tmp_path, mode):
    result = _check(_text_written(mode=mode), tmp_path)
    assert not result.ok


def test_unknown_join_or_source_and_path_escape_remain_blocked(tmp_path):
    assert not _check(_text_written().replace("WORK_DIR = './working'", "WORK_DIR = input()"), tmp_path).ok
    assert not _check(_text_written().replace("inference_src = 'VALUE = 42\\n'", "inference_src = input()"), tmp_path).ok
    assert not _check(_text_written().replace("WORK_DIR = './working'", "WORK_DIR = '../outside'"), tmp_path).ok


def test_conditional_closed_writer_is_not_proven_after_branch(tmp_path):
    code = (
        "if input():\n"
        "    with open('helper.py', 'w') as f:\n"
        "        f.write('VALUE=42')\n"
        "import helper\n"
    )
    assert not _check(code, tmp_path).ok


def test_custom_opener_cannot_claim_to_write_the_named_module(tmp_path):
    code = _text_written().replace(
        "open(inference_path, 'w')", "open(inference_path, 'w', opener=custom_opener)"
    )
    assert not _check(code, tmp_path).ok


def _numeric_template_code(*, template=None, conversion="float", pathlib=False):
    template = template or (
        "VALUE = __SIZE__\nLABELS = __LABELS__\n"
        "MEAN = __MEAN__\nSTD = __STD__\n"
    )
    code = _text_written(source=template, pathlib=pathlib)
    return code.replace(
        f"    inference_src = {template!r}\n",
        f"    template = {template!r}\n"
        "    LABELS = [0, 1, 2]\n"
        "    SIZE = 42\n"
        "    samples = [sum([1, 2, 3]) / 3, sum([4, 5, 6]) / 3]\n"
        "    inference_src = (template.replace('__LABELS__', repr(LABELS))\n"
        "        .replace('__SIZE__', repr(SIZE))\n"
        f"        .replace('__MEAN__', repr([{conversion}(v) for v in samples]))\n"
        f"        .replace('__STD__', repr([{conversion}(v) for v in samples])))\n",
    )


@pytest.mark.parametrize("conversion", ["float", "int"])
@pytest.mark.parametrize("pathlib", [False, True])
def test_runtime_numeric_template_is_proven_without_fabricated_parameters(tmp_path, conversion, pathlib):
    from biomaker.utils.local_modules import LocalModuleDiscovery, _SourceTemplate
    import ast
    code = _numeric_template_code(conversion=conversion, pathlib=pathlib)
    inspected = []

    def record(source, child):
        assert any(isinstance(value, _SourceTemplate) for value in child.sources.values())
        inspected.append(source)
        return _check(source, tmp_path)

    discovery = LocalModuleDiscovery(tmp_path.resolve(), record)
    discovery.visit(ast.parse(code))
    assert inspected and "MEAN = __MEAN__" in inspected[0]
    assert "STD = __STD__" in inspected[0]
    # The fixed AST preserves unknown-value identifiers; no invented 0 data.
    assert "VALUE = 42" in inspected[0] and "LABELS = [0, 1, 2]" in inspected[0]
    assert _check(code, tmp_path).ok
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("pathlib", [False, True])
def test_real_subprocess_writes_runtime_numeric_parameters_then_imports(tmp_path, monkeypatch, pathlib):
    from biomaker import interpreter
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    monkeypatch.setattr(interpreter, "_sample_process_usage", lambda *a, **k: None)
    result = Interpreter(tmp_path, timeout=10).run(
        "import time\ntime.sleep(0.5)\n" + _numeric_template_code(pathlib=pathlib)
    )
    assert result.exc_type is None, result.term_out
    source = (tmp_path / "working/inference.py").read_text()
    assert "MEAN = [2.0, 5.0]" in source
    assert "STD = [2.0, 5.0]" in source
    assert "42" in result.stdout


@pytest.mark.parametrize("bad_source", [
    "import cv2\n", "import importlib\nimportlib.import_module('cv2')\n",
    "import subprocess\nsubprocess.run(['pip', 'install', 'cv2'])\n",
    "import os\nos.setsid()\n", "x.cuda()\n",
])
def test_numeric_template_keeps_all_source_policies(tmp_path, bad_source):
    template = bad_source + "VALUE = __SIZE__\nLABELS = __LABELS__\nMEAN = __MEAN__\nSTD = __STD__\n"
    result = _check(_numeric_template_code(template=template), tmp_path)
    assert not result.ok and "local module" in result.message


@pytest.mark.parametrize("replacement", [
    "repr(samples)", "repr(object())", "str(samples)",
    "repr([v for v in samples])", "repr([str(v) for v in samples])",
    "repr([helper(v) for v in samples])", "repr((float(v) for v in samples))",
])
def test_unproven_parameter_serialization_stays_blocked(tmp_path, replacement):
    code = _numeric_template_code().replace("repr([float(v) for v in samples])", replacement)
    result = _check(code, tmp_path)
    assert not result.ok and "not statically verifiable" in result.message


@pytest.mark.parametrize("shadow", [
    "repr = lambda x: 'import cv2'", "float = lambda x: x", "int = lambda x: x",
    "def repr(x):\n    return str(x)", "from os import path as float",
    "import builtins", "globals()['float'] = lambda x: x",
    "__builtins__['repr'] = str", "builtins = __import__('builtins')",
    "from math import *", "import sys\nsys.__dict__['float'] = object()",
    "import sys\nsys.modules['builtins'].float = object()",
    "match object():\n    case float:\n        pass",
])
def test_shadowed_or_mutable_builtin_namespace_has_no_template_proof(tmp_path, shadow):
    conversion = "int" if shadow.startswith("int =") else "float"
    result = _check(shadow + "\n" + _numeric_template_code(conversion=conversion), tmp_path)
    assert not result.ok


def test_comprehension_target_and_function_parameter_cannot_shadow_conversion(tmp_path):
    code = _numeric_template_code()
    assert not _check(code.replace("for v in samples", "for float in samples"), tmp_path).ok
    assert not _check(code.replace("def main():", "def main(float=None):"), tmp_path).ok


@pytest.mark.parametrize("slot_line", [
    "MEAN = '__MEAN__'", "import __MEAN__", "from __MEAN__ import x",
    "__MEAN__ = 1", "MEAN = prefix__MEAN__", "MEAN = [__MEAN__]",
    "MEAN = f'__MEAN__'", "PATH = 'working/__MEAN__.py'",
    "MEAN = __MEAN__ # __MEAN__", "MEAN = __MEAN__\nOTHER = '__MEAN__'",
])
def test_dynamic_slots_are_only_complete_unquoted_assignment_expressions(tmp_path, slot_line):
    template = "VALUE = __SIZE__\nLABELS = __LABELS__\n" + slot_line + "\nSTD = __STD__\n"
    assert not _check(_numeric_template_code(template=template), tmp_path).ok


@pytest.mark.parametrize("transform", [
    ".replace('0', '__import__(\"cv2\")')",
    ".replace('MEAN = ', 'import ')",
    ".replace('__MEAN__', 'import cv2')", ".strip()", " + '\\n'",
])
def test_unknown_template_provenance_cannot_be_dropped_or_rewritten(tmp_path, transform):
    code = _numeric_template_code().replace(
        "    inference_path =", f"    inference_src = inference_src{transform}\n    inference_path ="
    )
    assert not _check(code, tmp_path).ok


def test_template_cannot_be_used_as_a_path(tmp_path):
    code = _numeric_template_code().replace(
        "inference_path = os.path.join(WORK_DIR, 'inference.py')",
        "inference_path = str(inference_src)",
    )
    assert not _check(code, tmp_path).ok


def test_literal_repr_and_replace_are_exact_and_bounded(tmp_path):
    code = (
        "from pathlib import Path\nvalues = [-1, 2.5, True, None]\n"
        "src = 'DATA = TOKEN'.replace('TOKEN', repr(values))\n"
        "Path('helper.py').write_text(src)\nimport helper\n"
    )
    assert _check(code, tmp_path).ok
    # This would expand to >1MB if evaluated; the proof must reject first.
    code = (
        "from pathlib import Path\n"
        f"src = {'x' * 2000!r}.replace('x', {'y' * 2000!r})\n"
        "Path('helper.py').write_text(src)\nimport helper\n"
    )
    assert not _check(code, tmp_path).ok


@pytest.mark.parametrize("mutation", [
    "values.append(object())", "values[0] = object()", "values += unknown",
    "alias = values\nalias.append(object())", "mutate(values)",
    "alias = values.append\nalias(object())", "del values[0]",
    "holder.items = values", "holder['items'] = values",
    "items, = (values,)\nitems.append(object())",
    "holder = {'items': values}\nmutate(holder)",
    "alias = [values]\nalias[0].append(object())",
    "def mutate():\n    values.append(object())\nmutate()",
    "def len(x):\n    x.append(object())\nlen(values)",
    "def print(x):\n    x.append(object())\nprint(values)",
])
def test_mutable_literal_data_cannot_keep_a_stale_repr_proof(tmp_path, mutation):
    code = (
        "from pathlib import Path\nvalues = [0, 1, 2]\n" + mutation + "\n"
        "src = 'DATA = TOKEN'.replace('TOKEN', repr(values))\n"
        "Path('helper.py').write_text(src)\nimport helper\n"
    )
    result = _check(code, tmp_path)
    assert not result.ok and "not statically verifiable" in result.message


def test_readonly_builtin_uses_and_later_list_mutations_preserve_completed_source(tmp_path):
    code = (
        "from pathlib import Path\nvalues = [0, 1, 2]\n"
        "print(values)\nsize = len(values)\n"
        "src = 'DATA = TOKEN'.replace('TOKEN', repr(values))\n"
        "values.append(object())\n"
        "Path('helper.py').write_text(src)\nimport helper\n"
    )
    assert _check(code, tmp_path).ok


def test_literal_module_is_admitted_without_preflight_writes(tmp_path):
    result = _check(_generated(), tmp_path)
    assert result.ok, result.message
    assert list(tmp_path.iterdir()) == []


def test_interpreter_executes_generated_module(tmp_path, monkeypatch):
    from biomaker import interpreter
    monkeypatch.setattr(interpreter, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    monkeypatch.setattr(interpreter, "_sample_process_usage", lambda *a, **k: None)
    # Retain real subprocess execution/containment; keep the child available for
    # the macOS process identity snapshot even on very fast machines.
    code = "import time\ntime.sleep(0.5)\n" + _generated()
    result = Interpreter(tmp_path, timeout=10).run(code)
    assert result.exc_type is None, result.term_out
    assert "42" in result.stdout
    assert (tmp_path / "working/inference.py").read_text() == "VALUE = 42\n"


@pytest.mark.parametrize("code", ["import inference", "import cv2", "try:\n import cv2\nexcept ImportError:\n pass"])
def test_unknown_packages_remain_blocked(tmp_path, code):
    result = _check(code, tmp_path)
    assert not result.ok and result.error_type == "DependencyPolicyError"


@pytest.mark.parametrize("source", [
    "import cv2\nVALUE=42", "import importlib\nimportlib.import_module('cv2')",
    "import importlib\nname='cv2'\nimportlib.import_module(name)",
    "import subprocess\nsubprocess.run(['pip','install','cv2'])",
    "import os\nos.setsid()", "import torch\nx.cuda()", "def broken(:",
])
def test_generated_module_has_same_dependency_and_execution_policy(tmp_path, source):
    result = _check(_generated(source), tmp_path)
    assert not result.ok and "local module" in result.message


@pytest.mark.parametrize("statement", ["import helper", "import sys, helper", "from helper import VALUE", "import importlib\nimportlib.import_module('helper')"])
def test_existing_local_source_is_inspected(tmp_path, statement):
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE=42\n")
    assert _check(statement, tmp_path).ok
    helper.write_text("import cv2\nVALUE=42\n")
    result = _check(statement, tmp_path)
    assert not result.ok and "cv2" in result.message


def test_nested_existing_package_dependencies_are_checked(tmp_path):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text("from .helper import VALUE\n")
    (package / "helper.py").write_text("VALUE=42\n")
    assert _check("from mypackage import VALUE", tmp_path).ok
    (package / "helper.py").write_text("import cv2\nVALUE=42\n")
    assert not _check("from mypackage import VALUE", tmp_path).ok


def test_cyclic_local_imports_terminate_and_still_check_dependencies(tmp_path):
    (tmp_path / "one.py").write_text("import two\n")
    (tmp_path / "two.py").write_text("import one\n")
    assert _check("import one", tmp_path).ok
    (tmp_path / "two.py").write_text("import one\nimport cv2\n")
    assert not _check("import one", tmp_path).ok


@pytest.mark.parametrize("prefix", [
    "if False:\n    Path('helper.py').write_text('VALUE=1')\n",
    "def never_called():\n    Path('helper.py').write_text('VALUE=1')\n",
    "for i in []:\n    Path('helper.py').write_text('VALUE=1')\n",
    "try:\n    Path('helper.py').write_text('VALUE=1')\nexcept Exception:\n    pass\n",
])
def test_unproven_writes_do_not_allow_unknown_import(tmp_path, prefix):
    result = _check("from pathlib import Path\n" + prefix + "import helper", tmp_path)
    assert not result.ok


def test_write_after_import_or_missing_search_path_is_not_local(tmp_path):
    assert not _check("from pathlib import Path\nimport helper\nPath('helper.py').write_text('VALUE=1')", tmp_path).ok
    assert not _check("from pathlib import Path\nPath('working/helper.py').write_text('VALUE=1')\nimport helper", tmp_path).ok


def test_unknown_source_overwrite_does_not_trust_existing_file(tmp_path):
    (tmp_path / "helper.py").write_text("VALUE=42")
    result = _check("from pathlib import Path\nPath('helper.py').write_text(input())\nimport helper", tmp_path)
    assert not result.ok and "not statically verifiable" in result.message


@pytest.mark.parametrize("mutation", [
    "if input():\n    Path('helper.py').write_text(input())",
    "for i in range(1):\n    Path('helper.py').write_text(input())",
    "try:\n    Path('helper.py').write_text(input())\nexcept Exception:\n    pass",
    "with open('helper.py', 'w') as f:\n    f.write(input())",
    "Path('helper.py').write_bytes(b'import cv2')",
    "Path('helper.py').unlink()",
])
def test_uncertain_or_unsupported_overwrite_does_not_trust_old_source(tmp_path, mutation):
    (tmp_path / "helper.py").write_text("VALUE=42")
    result = _check("from pathlib import Path\n" + mutation + "\nimport helper", tmp_path)
    assert not result.ok and "not statically verifiable" in result.message


def test_outside_and_symlinked_modules_are_not_candidate_owned(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-external")
    outside.mkdir()
    (outside / "helper.py").write_text("VALUE=42")
    (tmp_path / "helper.py").symlink_to(outside / "helper.py")
    assert not _check("import helper", tmp_path).ok
    assert not _check(f"import sys\nsys.path.insert(0, {str(outside)!r})\nimport helper", tmp_path).ok


def test_internal_symlink_cannot_hide_a_source_overwrite(tmp_path):
    (tmp_path / "actual.py").write_text("VALUE=42")
    (tmp_path / "helper.py").symlink_to(tmp_path / "actual.py")
    assert _check("import helper", tmp_path).ok
    assert not _check("from pathlib import Path\nPath('helper.py').write_text('import cv2')\nimport helper", tmp_path).ok


def test_deep_import_graph_has_bounded_inspection(tmp_path):
    for index in range(20):
        (tmp_path / f"helper{index}.py").write_text(f"import helper{index + 1}" if index < 19 else "VALUE=42")
    result = _check("import helper0", tmp_path)
    assert not result.ok and "inspection limit" in result.message


def test_policy_without_workspace_does_not_guess_current_directory(tmp_path, monkeypatch):
    (tmp_path / "helper.py").write_text("VALUE=42")
    monkeypatch.chdir(tmp_path)
    assert not validate_code_environment("import helper").ok


def test_large_local_source_is_rejected_without_execution(tmp_path):
    (tmp_path / "helper.py").write_text("#" * 1_000_001)
    result = _check("import helper", tmp_path)
    assert not result.ok and "size limit" in result.message
