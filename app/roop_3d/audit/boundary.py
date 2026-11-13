"""Fail-closed isolation checks for the standalone roop-3d checkout (what ``verify_isolation.py --report`` really runs).

Five checks, every count COMPUTED from what was inspected, none constant:

``sibling_imports``   no ``.py`` under ``app/`` imports a sibling-checkout package (``roop_ultimate``/``roop-ultimate``, and
                      ``roop`` when this checkout has no ``app/roop`` of its own), ``sys.path``-inserts, ``importlib``-loads,
                      ``runpy``-runs or ``spec_from_file_location``-loads a path that names a sibling checkout or climbs out of
                      the checkout (``Path(__file__).parents[k]`` / ``.parent`` chains / ``os.path.dirname`` nesting).
``sibling_writes``    no write-mode ``open``, ``write_text/write_bytes``, ``shutil``/``os`` mutation, ``mkdir``/``touch``/``unlink``
                      whose target expression is built from a sibling root (``legacy_root(...)``, a literal naming a sibling, or a
                      variable assigned from one).
``interpreter``       ``sys.prefix`` resolves inside ``app/roop_3d/.venv`` of this checkout.
``git_origin``        ``git remote get-url origin`` names the ``roop-3d`` repository.
``tracked_artefacts`` no tracked file lives under a venv/cache/runtime directory or is a model-weight/engine file.

Limits (stated, not hidden): checks (a)/(b) are a static AST heuristic. They see literal strings, ``legacy_root`` calls and simple
single-assignment variable flow within a file; they do NOT see paths read from config/env/JSON at runtime, string concatenation
through helper functions, ``exec``/``eval`` or dynamic ``__import__`` of computed names. ``.runtime/`` and ignored files are out
of scope for (a)/(b) because git ignores them. A file that cannot be parsed is itself a violation (fail closed).
Standard library only; no legacy code is imported.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

SIBLING_NAMES = ("roop-ultimate", "roop_ultimate")
LEGACY_ROOT_CALLS = frozenset({"legacy_root"})
IMPORT_CALLS = frozenset({"import_module", "spec_from_file_location", "run_path", "run_module", "SourceFileLoader", "load_source"})
PATH_MUTATORS = frozenset({"write_text", "write_bytes", "mkdir", "touch", "unlink", "rmdir", "rename", "replace", "symlink_to"})
SHUTIL_WRITERS = frozenset({"copy", "copy2", "copyfile", "copytree", "move", "rmtree", "copymode", "copystat"})
OS_WRITERS = frozenset({"replace", "rename", "remove", "unlink", "rmdir", "mkdir", "makedirs", "removedirs", "symlink", "link", "truncate"})
TRACKED_FORBIDDEN_DIRS = frozenset({".venv", "venv", "site-packages", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                                    ".cache", ".runtime", "node_modules"})
WEIGHT_SUFFIXES = frozenset({".onnx", ".pth", ".pt", ".safetensors", ".ckpt", ".engine", ".plan", ".trt", ".gguf", ".tflite",
                             ".h5", ".caffemodel", ".pkl"})
PRUNE = frozenset({".git", ".venv", "venv", ".runtime", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"})


@dataclass(frozen=True)
class Violation:
    check: str
    path: str
    line: int
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "detail": self.detail}


# ----------------------------------------------------------------------------------------------------------------- files
def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


def python_files(app_dir: Path) -> tuple[list[Path], str]:
    """Tracked + untracked-not-ignored ``.py`` files under ``app_dir`` (``git ls-files``), else a pruned directory walk.

    Returns ``(absolute paths, source)`` with source ``"git"`` or ``"walk"``."""
    run = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"], app_dir)
    if run is not None and run.returncode == 0:
        names = sorted({n for n in run.stdout.split("\0") if n})
        return [app_dir / n for n in names if (app_dir / n).is_file()], "git"
    files: list[Path] = []
    for base, directories, filenames in os.walk(app_dir, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in PRUNE)
        files.extend(Path(base) / n for n in sorted(filenames) if n.endswith(".py"))
    return files, "walk"


# ------------------------------------------------------------------------------------------------------------------ AST
def _tokens(own_name: str, siblings: Iterable[str], own_dirs: frozenset[str] = frozenset()) -> tuple[re.Pattern[str], tuple[str, ...]]:
    names = {n.casefold() for n in SIBLING_NAMES} | {s.casefold() for s in siblings
                                                      if len(s) >= 4 and s.casefold() != own_name.casefold() and s.casefold() not in own_dirs}
    pattern = re.compile(r"(?:^|[\\/])(" + "|".join(re.escape(n) for n in sorted(names)) + r")(?:$|[\\/])")
    return pattern, tuple(sorted(names))


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _strings(node: ast.AST) -> Iterable[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _names(node: ast.AST) -> set[str]:
    return {c.id for c in ast.walk(node) if isinstance(c, ast.Name)} | {
        c.attr for c in ast.walk(node) if isinstance(c, ast.Attribute)}


def _file_ups(node: ast.AST) -> int:
    """How many directory levels above ``__file__``'s own directory this expression climbs; ``-1`` when not rooted at ``__file__``."""
    best = -1
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Attribute) and child.value.attr == "parents":
            index = child.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int) and _rooted_in_file(child.value.value):
                best = max(best, index.value)
        elif isinstance(child, ast.Attribute) and child.attr == "parent" and _rooted_in_file(child.value):
            depth, cursor = 0, child
            while isinstance(cursor, ast.Attribute) and cursor.attr == "parent":
                depth += 1
                cursor = cursor.value
            if _rooted_in_file(cursor):
                best = max(best, depth - 1)
        elif isinstance(child, ast.Call) and _call_name(child) == "dirname" and child.args and _rooted_in_file(child.args[0]):
            depth, cursor = 0, child
            while isinstance(cursor, ast.Call) and _call_name(cursor) == "dirname" and cursor.args:
                depth += 1
                cursor = cursor.args[0]
            if _rooted_in_file(cursor):
                best = max(best, depth - 1)
    return best


def _rooted_in_file(node: ast.AST) -> bool:
    return any(isinstance(c, ast.Name) and c.id == "__file__" for c in ast.walk(node))


class _Scanner(ast.NodeVisitor):
    def __init__(self, relative: str, depth_to_parent_of_checkout: int, pattern: re.Pattern[str], own_roop_package: bool):
        self.relative, self.limit, self.pattern, self.own_roop = relative, depth_to_parent_of_checkout, pattern, own_roop_package
        self.imports: list[Violation] = []
        self.writes: list[Violation] = []
        self.imports_checked = 0
        self.import_calls = 0
        self.write_calls = 0
        self.tainted: set[str] = set()
        self._module_taint: set[str] = set()

    # ---- taint: names assigned from a sibling-root expression. Scope-aware: module-level assignments taint the module,
    # function-level ones only that function (fixpoint inside each scope), so a common name such as ``path`` in one
    # function does not taint unrelated uses elsewhere.
    def _expr_tainted(self, node: ast.AST) -> bool:
        ids = {c.id for c in ast.walk(node) if isinstance(c, ast.Name)}
        if ids & self.tainted or _names(node) & LEGACY_ROOT_CALLS:
            return True
        return any(self.pattern.search(text.replace("\\", "/") + "/") for text in _strings(node))

    def _taint_scope(self, body_nodes: list[ast.AST]) -> set[str]:
        assignments: list[ast.AST] = []

        def collect(node: ast.AST, root: bool) -> None:
            if not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return                                  # nested functions get their own scope
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(node, "value", None) is not None:
                assignments.append(node)
            for child in ast.iter_child_nodes(node):
                collect(child, False)

        for top in body_nodes:
            collect(top, True)
        found: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node in assignments:
                if self._expr_tainted(node.value):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in (c.id for c in ast.walk(target) if isinstance(c, ast.Name)):
                            if name not in self.tainted:
                                self.tainted.add(name)
                                found.add(name)
                                changed = True
        return found

    def prime(self, tree: ast.Module) -> None:
        """Taint module-level names (statements outside any function/class body walk are included via ast.walk of the
        top-level non-function statements)."""
        top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self._module_taint = self._taint_scope(top)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        outer = set(self.tainted)
        self._taint_scope([node])
        self.generic_visit(node)
        self.tainted = outer

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _top(self, module: str) -> str:
        return module.split(".")[0]

    def _banned_module(self, module: str) -> bool:
        top = self._top(module).replace("-", "_").casefold()
        return top == "roop_ultimate" or (top == "roop" and not self.own_roop)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports_checked += 1
            if self._banned_module(alias.name):
                self.imports.append(Violation("sibling_imports", self.relative, node.lineno, f"import {alias.name}"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports_checked += 1
        if node.level == 0 and node.module and self._banned_module(node.module):
            self.imports.append(Violation("sibling_imports", self.relative, node.lineno, f"from {node.module} import ..."))
        self.generic_visit(node)

    def _argument_nodes(self, node: ast.Call) -> list[ast.AST]:
        return [*node.args, *(k.value for k in node.keywords)]

    def _path_escape(self, nodes: list[ast.AST]) -> bool:
        return any(_file_ups(n) >= self.limit for n in nodes)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        func = node.func
        owner = func.value if isinstance(func, ast.Attribute) else None
        owner_names = _names(owner) if owner is not None else set()
        args = self._argument_nodes(node)
        sys_path_call = name in {"insert", "append", "extend"} and owner is not None and "path" in owner_names and "sys" in owner_names
        if sys_path_call or name in IMPORT_CALLS:
            self.import_calls += 1
            if any(self._expr_tainted(a) for a in args) or self._path_escape(args):
                self.imports.append(Violation("sibling_imports", self.relative, node.lineno,
                                              f"{'sys.path.' + name if sys_path_call else name}() reaches a sibling checkout"))
        write_target: list[ast.AST] = []
        if isinstance(func, ast.Name) and func.id == "open" or (isinstance(func, ast.Attribute) and func.attr == "open" and owner is not None):
            mode = None
            positional = node.args[1:] if isinstance(func, ast.Name) else node.args[:1]
            if positional and isinstance(positional[0], ast.Constant):
                mode = positional[0].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if isinstance(mode, str) and set(mode) & set("wax+"):
                write_target = [node.args[0]] if isinstance(func, ast.Name) and node.args else ([owner] if owner is not None else [])
        elif name in SHUTIL_WRITERS and "shutil" in owner_names:
            write_target = args
        elif name in OS_WRITERS and "os" in owner_names and "path" not in owner_names:
            write_target = args
        elif name in PATH_MUTATORS and owner is not None and owner_names and not owner_names & {"shutil", "os"}:
            if name in {"rename", "replace"}:
                if len(node.args) == 1 and not node.keywords:        # Path.replace(target); str.replace has two arguments
                    write_target = [owner, *args]
            else:
                write_target = [owner]
        if write_target:
            self.write_calls += 1
            if any(self._expr_tainted(t) for t in write_target) or self._path_escape(write_target):
                self.writes.append(Violation("sibling_writes", self.relative, node.lineno, f"{name}() targets a sibling-checkout path"))
        self.generic_visit(node)


def scan_sources(app_dir: Path, *, siblings: Iterable[str] = ()) -> dict[str, Any]:
    """Checks (a) and (b) over every Python file under ``app_dir``."""
    app_dir = Path(app_dir).absolute()
    checkout = app_dir.parent
    own_dirs = frozenset(p.name.casefold() for base in (app_dir, checkout) for p in base.iterdir() if p.is_dir())
    pattern, token_names = _tokens(checkout.name, siblings, own_dirs)
    own_roop = (app_dir / "roop").is_dir()
    files, source = python_files(app_dir)
    imports: list[Violation] = []
    writes: list[Violation] = []
    unparsable: list[Violation] = []
    imports_checked = import_calls = write_calls = 0
    for path in files:
        relative = path.relative_to(app_dir).as_posix()
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError, OSError) as error:
            unparsable.append(Violation("sibling_imports", relative, getattr(error, "lineno", 0) or 0, f"unparsable: {error}"))
            continue
        # levels from this file's directory up to the checkout's parent (where sibling checkouts live)
        depth = len(path.parent.relative_to(app_dir).parts) + 2     # app_dir -> checkout -> parent of checkout
        scanner = _Scanner(relative, depth, pattern, own_roop)
        scanner.prime(tree)
        scanner.visit(tree)
        imports += scanner.imports
        writes += scanner.writes
        imports_checked += scanner.imports_checked
        import_calls += scanner.import_calls
        write_calls += scanner.write_calls
    return {"files_scanned": len(files), "file_source": source, "sibling_tokens": list(token_names),
            "imports_checked": imports_checked, "import_calls": import_calls, "write_calls": write_calls,
            "imports": [*imports, *unparsable], "writes": writes}


# ---------------------------------------------------------------------------------------------------------- other checks
def check_interpreter(app_dir: Path, prefix: str | Path | None = None) -> dict[str, Any]:
    expected = (Path(app_dir) / "roop_3d" / ".venv").absolute()
    actual = Path(prefix if prefix is not None else sys.prefix)
    try:
        inside = actual.resolve().is_relative_to(expected.resolve())
    except OSError:
        inside = False
    violations = [] if inside else [{"detail": f"interpreter prefix {actual} is not inside {expected}"}]
    return {"ok": inside, "checked": 1, "violations": violations, "prefix": str(actual), "expected": str(expected)}


def _repo_name(url: str) -> str:
    tail = re.split(r"[\\/:]", url.strip().rstrip("/\\"))[-1]
    return tail[:-4] if tail.endswith(".git") else tail


def check_git_origin(repo_root: Path, expected_name: str = "roop-3d") -> dict[str, Any]:
    run = _git(["remote", "get-url", "origin"], repo_root)
    if run is None or run.returncode != 0 or not run.stdout.strip():
        reason = "git is unavailable" if run is None else (run.stderr.strip() or "no origin remote")
        return {"ok": False, "checked": 1, "violations": [{"detail": f"cannot determine git origin: {reason}"}], "origin": None}
    url = run.stdout.strip()
    ok = _repo_name(url) == expected_name
    return {"ok": ok, "checked": 1, "origin": url,
            "violations": [] if ok else [{"detail": f"origin {url} is not the {expected_name} repository"}]}


def check_tracked_artefacts(repo_root: Path) -> dict[str, Any]:
    run = _git(["ls-files", "-z"], repo_root)
    if run is None or run.returncode != 0:
        return {"ok": False, "checked": 0, "violations": [{"detail": "git ls-files failed; tracked files cannot be inspected"}]}
    tracked = [n for n in run.stdout.split("\0") if n]
    violations = []
    for name in tracked:
        parts = name.replace("\\", "/").split("/")
        if TRACKED_FORBIDDEN_DIRS & set(parts[:-1]):
            violations.append({"path": name, "detail": "tracked under a venv/cache/runtime directory"})
        elif Path(name).suffix.casefold() in WEIGHT_SUFFIXES:
            violations.append({"path": name, "detail": "tracked model-weight/engine file"})
    return {"ok": not violations, "checked": len(tracked), "violations": violations}


def run_checks(app_dir: Path, *, prefix: str | Path | None = None, siblings: Iterable[str] | None = None,
               expected_origin_name: str = "roop-3d") -> dict[str, Any]:
    """Run all five checks; ``ok`` only when every one passes. ``siblings`` defaults to the directory names next to the checkout."""
    app_dir = Path(app_dir).absolute()
    repo_root = app_dir.parent
    if siblings is None:
        try:
            siblings = [p.name for p in repo_root.parent.iterdir() if p.is_dir() and p != repo_root]
        except OSError:
            siblings = []
    scan = scan_sources(app_dir, siblings=siblings)
    imports_check = {"ok": not scan["imports"], "checked": scan["imports_checked"] + scan["import_calls"],
                     "files_scanned": scan["files_scanned"], "file_source": scan["file_source"], "sibling_tokens": scan["sibling_tokens"],
                     "violations": [v.as_dict() for v in scan["imports"]]}
    writes_check = {"ok": not scan["writes"], "checked": scan["write_calls"], "files_scanned": scan["files_scanned"],
                    "violations": [v.as_dict() for v in scan["writes"]]}
    checks = {"sibling_imports": imports_check, "sibling_writes": writes_check,
              "interpreter": check_interpreter(app_dir, prefix), "git_origin": check_git_origin(repo_root, expected_origin_name),
              "tracked_artefacts": check_tracked_artefacts(repo_root)}
    violations = [{"check": name, **v} for name, c in checks.items() for v in c["violations"]]
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "violations": violations,
            "limits": "static AST heuristic for sibling_imports/sibling_writes; see roop_3d/audit/boundary.py"}
