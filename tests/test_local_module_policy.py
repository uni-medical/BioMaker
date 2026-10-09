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
