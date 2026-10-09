"""Conservative, non-executing discovery of candidate-owned Python imports.

Only workspace files and literal source writes on the import's execution path
count as local modules. Unknown paths/source strings remain subject to the normal
third-party policy. This is a dependency preflight, not a Python sandbox.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


class LocalModuleError(ValueError):
    pass


class LocalModuleDiscovery(ast.NodeVisitor):
    def __init__(self, root: Path, validate, *, paths=None, sources=None, checked=None, depth=0):
        self.root = root
        self.validate = validate
        self.paths = list(paths if paths is not None else [root])
        self.sources = dict(sources or {})
        self.bindings = {}
        self.checked = checked if checked is not None else set()
        self.depth = depth
        self.local_nodes: set[tuple[int, str]] = set()

    def value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Div):
                    return Path(left) / right
                if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                    return left + right
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "os.path.join" and node.args and not node.keywords:
                parts = [self.value(arg) for arg in node.args]
                if all(part is not None for part in parts):
                    return os.path.join(*parts)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath" and not node.keywords:
                base = self.value(node.func.value)
                parts = [self.value(arg) for arg in node.args]
                if base is not None and all(part is not None for part in parts):
                    return Path(base).joinpath(*parts)
            if name in {"Path", "pathlib.Path", "str", "os.path.abspath"} and len(node.args) == 1:
                value = self.value(node.args[0])
                if value is not None:
                    if name == "str":
                        return str(value)
                    return self.path(value) if name == "os.path.abspath" else Path(value)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"} and not node.args:
                value = self.value(node.func.value)
                return self.path(value) if value is not None else None
        return None

    def path(self, value):
        try:
            path = Path(value)
            path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        except (ValueError, OSError, RuntimeError):
            return None
        return path if path.is_relative_to(self.root) else None

    def visit_Assign(self, node):
        self.visit(node.value)
        value = self.value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bindings[target.id] = value

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.bindings[node.target.id] = self.value(node.value)

    def _opened_file(self, node):
        """Resolve a known open call without opening or evaluating anything."""
        name = ast.unparse(node.func)
        value, mode = None, "r"
        if name == "open" and node.args:
            value = self.value(node.args[0])
            if len(node.args) > 1:
                mode = self.value(node.args[1])
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            value = self.value(node.func.value)
            if node.args:
                mode = self.value(node.args[0])
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode = self.value(keyword.value)
        path = self.path(value) if value is not None else None
        return path, mode

    def visit_Call(self, node):
        self.generic_visit(node)
        name = ast.unparse(node.func)
        # Opening for writing invalidates existing source. A proven text-write
        # context below restores it only after all writes have been inspected.
        path, mode = self._opened_file(node)
        if path is not None and path.suffix == ".py" and (not isinstance(mode, str) or any(c in mode for c in "wax+")):
            self.sources[path] = None
        if name in {"__import__", "importlib.import_module"} and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            module = node.args[0].value.split(".", 1)[0]
            if module and self.check_module(module):
                self.local_nodes.add((id(node), module))
        if name in {"sys.path.insert", "sys.path.append"} and node.args:
            value = self.value(node.args[-1])
            path = self.path(value) if value is not None else None
            if path is not None and path not in self.paths:
                self.paths.insert(0, path) if name.endswith("insert") else self.paths.append(path)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"write_text", "write_bytes", "unlink"}:
            value = self.value(node.func.value)
            path = self.path(value) if value is not None else None
            if path is not None and path.suffix == ".py":
                # A nonliteral overwrite invalidates even an existing file.
                source = self.value(node.args[0]) if node.args and node.func.attr == "write_text" else None
                self.sources[path] = source if isinstance(source, str) else None

    def visit_With(self, node):
        # Recognize the ordinary closed text-file form. Keeping this restricted
        # to direct writes prevents seek/truncate, handle aliases, conditional
        # writes or unknown source from accidentally trusting stale file bytes.
        if len(node.items) != 1:
            self.generic_visit(node)
            return
        item = node.items[0]
        call = item.context_expr
        if not isinstance(call, ast.Call) or not isinstance(item.optional_vars, ast.Name):
            self.generic_visit(node)
            return
        path, mode = self._opened_file(call)
        self.visit(call)
        parts = []
        proven = path is not None and path.suffix == ".py" and mode in {"w", "wt"}
        # A custom opener can redirect the descriptor away from the named file.
        opener_nodes = [kw.value for kw in call.keywords if kw.arg == "opener"]
        if ast.unparse(call.func) == "open" and len(call.args) > 7:
            opener_nodes.append(call.args[7])
        if any(not isinstance(value, ast.Constant) or value.value is not None for value in opener_nodes):
            proven = False
        for statement in node.body:
            write = statement.value if isinstance(statement, ast.Expr) else None
            if (
                isinstance(write, ast.Call)
                and isinstance(write.func, ast.Attribute)
                and isinstance(write.func.value, ast.Name)
                and write.func.value.id == item.optional_vars.id
                and write.func.attr == "write"
                and len(write.args) == 1
                and not write.keywords
            ):
                source = self.value(write.args[0])
                if isinstance(source, str):
                    parts.append(source)
                else:
                    proven = False
            else:
                proven = False
            self.visit(statement)
        if path is not None and path.suffix == ".py" and (
            not isinstance(mode, str) or any(c in mode for c in "wax+")
        ):
            self.sources[path] = "".join(parts) if proven else None

    def _isolated_body(self, body, *, argument_names=()):
        bindings, paths, sources = self.bindings, self.paths, self.sources
        self.bindings, self.paths, self.sources = dict(bindings), list(paths), dict(sources)
        for name in argument_names:
            self.bindings.pop(name, None)
        for child in body:
            self.visit(child)
        result = self.bindings, self.paths, self.sources
        self.bindings, self.paths, self.sources = bindings, paths, sources
        return result

    def visit_FunctionDef(self, node):
        args = node.args
        names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
        names += [a.arg for a in [args.vararg, args.kwarg] if a is not None]
        self._isolated_body(node.body, argument_names=names)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._isolated_body(node.body)

    def visit_If(self, node):
        self.visit(node.test)
        yes = self._isolated_body(node.body)
        no = self._isolated_body(node.orelse)
        self.bindings = {k: v for k, v in yes[0].items() if k in no[0] and no[0][k] == v}
        self.paths = [p for p in yes[1] if p in no[1]]
        self.sources = {
            key: yes[2][key] if key in yes[2] and key in no[2] and yes[2][key] == no[2][key] else None
            for key in yes[2].keys() | no[2].keys()
        }
        # The conventional "if path not in sys.path: insert(path)" guarantees
        # that path is present after either branch.
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.NotIn) and ast.unparse(test.comparators[0]) == "sys.path":
            value = self.value(test.left)
            path = self.path(value) if value is not None else None
            if path in yes[1] and path not in self.paths:
                self.paths.insert(0, path)

    def visit_Try(self, node):
        variants = [self._isolated_body(node.body)]
        for handler in node.handlers:
            variants.append(self._isolated_body(handler.body))
        variants.append(self._isolated_body(node.orelse))
        self._invalidate_uncertain(variants)
        # A write in a try block is not guaranteed if an exception is caught.
        for child in node.finalbody:
            self.visit(child)

    def visit_For(self, node):
        self._invalidate_uncertain([
            self._isolated_body(node.body), self._isolated_body(node.orelse),
        ])

    def _invalidate_uncertain(self, variants):
        for bindings, _paths, sources in variants:
            for key, value in sources.items():
                if key not in self.sources or self.sources[key] != value:
                    self.sources[key] = None
            for key, value in bindings.items():
                if key not in self.bindings or self.bindings[key] != value:
                    self.bindings[key] = None

    visit_AsyncFor = visit_For
    visit_While = visit_For

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name.split(".", 1)[0]
            if self.check_module(name):
                self.local_nodes.add((id(node), name))

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module and self.check_module(node.module.split(".", 1)[0]):
            self.local_nodes.add((id(node), node.module.split(".", 1)[0]))

    def check_module(self, name):
        if name == "evaluator":
            return False  # Frozen evaluator remains governed by its own admission.
        for directory in self.paths:
            module = directory / (name + ".py")
            package = directory / name / "__init__.py"
            for path in (package, module):
                is_package = path == package
                # Never follow a module/package symlink outside the candidate.
                path = self.path(path)
                if path is None:
                    continue
                if path in self.sources or path.is_file():
                    files = [path]
                    if is_package:
                        files = set()
                        for source_path in path.parent.rglob("*.py"):
                            files.add(source_path)
                            if len(files) > 128:
                                raise LocalModuleError("local package exceeds the preflight inspection limit")
                        files = sorted(files | {
                            p for p in self.sources if p.is_relative_to(path.parent)
                        })
                    for source_path in files:
                        self.check_source(source_path)
                    return True
        return False

    def check_source(self, path):
        path = self.path(path)
        if path is None:
            raise LocalModuleError("local Python source escapes the candidate workspace")
        try:
            if path in self.sources:
                source = self.sources[path]
            else:
                if path.stat().st_size > 1_000_000:
                    raise LocalModuleError("local Python source exceeds the preflight size limit")
                source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise LocalModuleError("local Python source cannot be read") from exc
        if source is None:
            raise LocalModuleError("local Python source is not statically verifiable")
        key = path, source
        if key in self.checked:
            return
        if len(self.checked) >= 128 or len(source) > 1_000_000 or self.depth >= 16:
            raise LocalModuleError("local Python imports exceed the preflight inspection limit")
        self.checked.add(key)
        child = LocalModuleDiscovery(self.root, self.validate, paths=self.paths, sources=self.sources, checked=self.checked, depth=self.depth + 1)
        result = self.validate(source, child)
        if not result.ok:
            raise LocalModuleError(f"local module `{path.relative_to(self.root)}`: {result.message}")
