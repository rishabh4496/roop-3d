"""Architecture gate: the ACTIVE render path contains zero 2D swappers, zero 5-point crop-paste, zero 2.5D geometry
and zero post-crop 2D GAN restoration.

Nothing here is a hard-coded module list. The set of "active" modules is computed from the import graph with ``ast``:
every module reachable (including lazy, function-level and ``importlib.import_module("roop_3d...")`` imports) from the live
roots below. ``tests/``, ``docs/``, virtualenvs and runtime output are excluded, and so is this test.

Every violation message names ``file:line`` (repo-relative). Set ``ZERO3D_POLICY_REPORT=<path>`` to also dump the complete
violation list of every rule as JSON.

Rules
  a) no 2D swapper identifiers / model ids in code, catalogs, registries or schemas
  b) no 5-point crop-paste (``cv2.estimateAffinePartial2D``; ``cv2.warpAffine`` whose matrix derives from alignment/paste)
  c) no 2.5D (weak-perspective / orthographic cameras, 3DDFA_V2 frontal strip, per-triangle ``warpAffine`` warpers)
  d) no 2D post-crop GAN calls (CodeFormer/GFPGAN/GPEN/UltraMax...); only UV-texture builders operating on UV maps
  e) runtime: ``True3DArchitectureGuard`` accepts a valid full-head mesh + K + [R|t] and rejects the 2D/2.5D shapes
  f) FastAPI OpenAPI schema and ``contracts.ts`` expose no removed swapper/enhancer options
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

PKG = "roop_3d"
PKG_ROOT = Path(__file__).resolve().parents[1]          # .../app/roop_3d  (the `roop_3d` package directory)
REPO_ROOT = PKG_ROOT.parents[1]                         # .../roop-3d
UI_SRC = REPO_ROOT / "app" / "roop_3d_ui" / "src"

# --------------------------------------------------------------------------------------------------------------------
# Live roots. Packages expand to every descendant module; roots that no longer exist are skipped (``server.app`` must exist).
# --------------------------------------------------------------------------------------------------------------------
LIVE_ROOTS = (
    "roop_3d.server",                      # the ASGI app and every router it mounts
    "roop_3d.engine.registry",             # engine dispatch
    "roop_3d.engine.pipeline_runner",      # reference engine runner
    "roop_3d.engine.volumetric",           # per-frame pipeline session
    "roop_3d.jobs",                        # job worker
    "roop_3d.pipeline",                    # photoreal engine + job runner
    "roop_3d.compositor",
    "roop_3d.enhancers",
    "roop_3d.optimized_processor",         # historical default stream engine (gone after the purge)
)
MUST_EXIST_ROOT = "roop_3d.server.app"

# Directory names never scanned (at any depth) and package-root-only names.
SKIP_DIRS_ANY = {".venv", "venv", "env", "tests", "docs", "__pycache__", ".runtime", ".omp", ".pytest_cache", "node_modules", "dist"}
SKIP_DIRS_TOP = {"app", "roop_3d", "outputs", "cache"}   # nested `app/roop_3d` mirror, runtime outputs, kernel caches

# --------------------------------------------------------------------------------------------------------------------
# Banned vocabulary
# --------------------------------------------------------------------------------------------------------------------
_SWAP_MODELS = (r"inswapper|reswapper|simswap|blendswap|uniface|hyperswap|realswap|hififace"
                r"|ghost[_\- ]?(?:\d+|unet|swap\w*|net|face)")
SWAP_MODEL_RE = re.compile(_SWAP_MODELS, re.I)
SWAP_WORD_RE = re.compile(r"swapper|dfm_morph|\.dfm\b|^ghost$", re.I)               # identifiers / schema keys only
GAN_NAME_RE = re.compile(r"codeformer|gfpgan|gpen|ultramax|restoreformer|styleganex|restore_aligned|restore_hires"
                         r"|restore_in_crop_space|hires_handoff|decoupledrestorer|onnxfacerestorer", re.I)
GAN_OPTION_RE = re.compile(r"restorer_model|face_restoration_policy|restoration_blend|fidelity_weight", re.I)
CAMERA_25D_RE = re.compile(r"weak[_\- ]?perspective|orthographic|scaled[_\- ]?ortho|orthographicpose|project_model5"
                           r"|estimate_pose_5pt|affine[_\- ]?camera", re.I)
GEOM_25D_RE = re.compile(r"3ddfa[_\- ]?v?2|warp_delaunay|piecewise_affine|build_canonical_head_mesh|canonical68|pose_blend"
                         r"|mesh_blend|frontaliz|tps_expression|thinplatesplinewarper|estimated_2_5d|standard_2d"
                         r"|(?<![0-9a-z])2[_.]?5[_.]?d(?![a-z0-9])|frontal[_\- ]?strip|jaw_fit|pose_frontalization", re.I)

# Where a neural detailer may legitimately live: UV-texture builders only.
UV_BUILDER_RE = re.compile(r"(^|/)(uv_[a-z0-9_]+|uv_baker)\.py$")
FRAME_ARG_RE = re.compile(r"frame|crop|aligned|portrait|swapped|face_img|bgr|roi|patch|target|photo|image(?!_?atlas)", re.I)
UV_ARG_RE = re.compile(r"uv|atlas|texel|albedo", re.I)
GAN_CALL_RE = re.compile(r"codeformer|gfpgan|gpen|ultramax|restoreformer|restore_aligned|restore_hires|restore_in_crop|restorer", re.I)

ALIGN_MARKER_RE = re.compile(r"estimateaffine|similarity|umeyama|align|registration|template|arcface|ffhq|kps|5pt|five"
                             r"|crop_matrix|inverse|invert|\binv\b|paste|replacement|crop_to_frame|frame_to_crop|swap", re.I)

# --------------------------------------------------------------------------------------------------------------------
# Explicit, justified allowlists. Keep them tiny; every entry needs a reason and is checked for staleness.
# Keys: (package-relative posix path, function qualname).
# --------------------------------------------------------------------------------------------------------------------
ALLOWED_ESTIMATE_AFFINE: dict[tuple[str, str], str] = {
    ("faceset/recognizer.py", "Recognizer.align_face"):
        "ArcFace needs its 112x112 five-point-aligned crop as NETWORK INPUT for an identity embedding; nothing is pasted back.",
    ("faceset_loader.py", "aligned_crop_512"):
        "Offline FaceSet ingestion: ArcFace-aligned 512 px crop feeds the landmark/identity networks; never composited into a frame.",
    ("faceset_synthesizer.py", "register_uv"):
        "Offline FaceSet UV map registration from generated portrait; operates in canonical UV space, never on video frames.",
}
ALLOWED_WARP_AFFINE: dict[tuple[str, str], str] = {
    ("faceset/recognizer.py", "Recognizer.align_face"): "Applies the embedding-input alignment above; output is an ArcFace network input, never pasted.",
    ("media/video_reader.py", "*"): "Generic resize/rotate/display-matrix handling of decoded video frames; no face geometry.",
    ("engine/detection/dense_landmarks.py", "*"): "Fixed-size network-input crop for a 106/203-point landmark regressor; output is landmarks, not pixels.",
    ("engine/detection/robust_detector.py", "*"): "Rotation test-time augmentation for the detector; results are mapped back as boxes/points.",
    ("reconstruction/native_backbone.py", "*"): "Crop for the dense-geometry regressor input; the regressor outputs geometry, not pixels pasted into frames.",
    ("reconstruction/morphable_backbone.py", "*"): "Crop for the morphable-model regressor input; the regressor outputs geometry, not pixels pasted into frames.",
    ("faceset_loader.py", "*"): "Landmark-network input crops while loading FaceSet photographs; offline asset ingestion.",
    ("engine/volumetric/angle_diagnostics.py", "*"): "Diagnostics-only thumbnails of already-detected faces; never reaches a rendered frame.",
    ("faceset_synthesizer.py", "register_uv"): "Offline FaceSet UV map registration from generated portrait; operates in canonical UV space, never on video frames.",
    ("engine/volumetric/preprocess.py", "aligned_crop"): "ArcFace-template 128x128 aligned crop for feature extractor network input; never composited or pasted back.",
    ("reconstruction/full_head.py", "FullHeadEngine.process"): "FullHeadEngine detail extraction crops a target ROI patch from the frame for comparison; does not paste into video frames.",
    ("engine/edit_plan.py", "*"): "Transforms occluder plane masks into target frame coordinates; does not paste face crops.",
    ("pipeline/session_adapter.py", "*"): "Maps semantic parser masks between face analysis coordinates and frame space; no face crop pasting.",
    ("pipeline/photorealistic_swap_core.py", "*"): "Places rendered 3D head layer coverage plane into ROI coordinates; no 2D swapper pasting.",
    ("compositor/pipeline.py", "*"): "Compositor transforms semantic analysis crops and mask planes for anatomical blending; no 2D swapper pasting.",
    ("compositor/boundary_fusion.py", "*"): "Transforms anatomical boundary masks for multi-band fusion; no 2D swapper pasting.",
    ("compositor/robust_fusion.py", "*"): "Transforms robust occlusion masks for boundary fusion; no 2D swapper pasting.",
    ("media/vfx.py", "*"): "Transforms 3D geometry pass buffers (normal, depth, lighting) for VFX export; no 2D swapper pasting.",
    ("vfx_export.py", "*"): "Transforms 3D geometry pass buffers (normal, depth, lighting) for VFX export; no 2D swapper pasting.",
}
# Same function-key convention; applies to the estimateAffine rule only when the call is *not* in one of the sets above.

# Informational: JSON data that mirrors machine state or protects *another* project and is never a render-path catalog.
JSON_SKIP_NAMES = re.compile(r"(verified_local_models|local_models_discovered|protected_(all_)?paths|STAGE\d+|_REPORT|risks|FINAL_)", re.I)
JSON_AUDIT_KEEP = re.compile(r"manifest|catalog|registry|sources", re.I)

# Files that implement or document the ban itself; their vocabulary is the enforcement, not a violation.
ENFORCEMENT_FILES = {"core/true3d_guard.py"}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str      # repo-relative posix
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


# --------------------------------------------------------------------------------------------------------------------
# Module index + import graph (ast only; nothing is imported or executed)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Module:
    name: str
    path: Path
    rel: str            # package-relative posix path
    is_pkg: bool
    tree: ast.Module


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _walk_py(root: Path):
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name in SKIP_DIRS_ANY or entry.name.startswith(".") or (root == PKG_ROOT and entry.name in SKIP_DIRS_TOP):
                continue
            if not (entry / "__init__.py").exists():
                continue
            yield from _walk_py(entry)
        elif entry.suffix == ".py":
            yield entry


@lru_cache(maxsize=1)
def module_index() -> tuple[dict[str, Module], list[Violation]]:
    modules: dict[str, Module] = {}
    parse_errors: list[Violation] = []
    for path in _walk_py(PKG_ROOT):
        rel = path.relative_to(PKG_ROOT).as_posix()
        parts = rel[:-3].split("/")
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        name = ".".join([PKG, *parts]) if parts else PKG
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except FileNotFoundError:        # another agent deleted it between listing and reading
            continue
        except SyntaxError as error:
            parse_errors.append(Violation("parse", _repo_rel(path), error.lineno or 1, f"syntax error: {error.msg}"))
            continue
        modules[name] = Module(name, path, rel, is_pkg, tree)
    return modules, parse_errors


def _resolve_relative(module: Module, level: int, target: str | None) -> str | None:
    package = module.name if module.is_pkg else module.name.rpartition(".")[0]
    for _ in range(level - 1):
        package = package.rpartition(".")[0]
    if not package:
        return None
    return f"{package}.{target}" if target else package


def _prefixes(dotted: str):
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        yield ".".join(parts[:i])


@dataclass
class ImportRef:
    module: str            # dotted module that was named (absolute)
    names: tuple[str, ...]  # names imported from it (empty for plain `import x`)
    line: int


def imports_of(module: Module) -> list[ImportRef]:
    refs: list[ImportRef] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PKG or alias.name.startswith(PKG + "."):
                    refs.append(ImportRef(alias.name, (), node.lineno))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(module, node.level, node.module) if node.level else node.module
            if base and (base == PKG or base.startswith(PKG + ".")):
                refs.append(ImportRef(base, tuple(a.name for a in node.names), node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if callee in {"import_module", "__import__"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and (arg.value == PKG or arg.value.startswith(PKG + ".")):
                    refs.append(ImportRef(arg.value, (), node.lineno))
    return refs


@lru_cache(maxsize=1)
def import_graph() -> dict[str, set[str]]:
    modules, _ = module_index()
    graph: dict[str, set[str]] = {}
    for module in modules.values():
        edges: set[str] = set()
        for ref in imports_of(module):
            for target in (ref.module, *(f"{ref.module}.{n}" for n in ref.names)):
                for prefix in _prefixes(target):          # importing a.b.c runs a/__init__ and a/b/__init__
                    if prefix in modules:
                        edges.add(prefix)
        edges.discard(module.name)
        graph[module.name] = edges
    return graph


@lru_cache(maxsize=1)
def active_modules() -> frozenset[str]:
    modules, _ = module_index()
    seeds: set[str] = set()
    for root in LIVE_ROOTS:
        for name in modules:
            if name == root or name.startswith(root + "."):
                seeds.add(name)
    graph = import_graph()
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph.get(name, ()))
        for prefix in _prefixes(name):                      # parent package __init__ modules execute too
            if prefix in modules and prefix not in seen:
                stack.append(prefix)
    return frozenset(seen)


def active_list() -> list[Module]:
    modules, _ = module_index()
    return [modules[n] for n in sorted(active_modules()) if modules[n].rel not in ENFORCEMENT_FILES]


# --------------------------------------------------------------------------------------------------------------------
# Token extraction
# --------------------------------------------------------------------------------------------------------------------
IDLIKE_RE = re.compile(r"^[\w.\-:/\[\]]{1,120}$")


def _docstring_ids(tree: ast.AST) -> set[int]:
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                skip.add(id(body[0].value))
    return skip


def tokens_of(tree: ast.AST):
    """Yield (kind, text, line): kind is 'ident' (names, attributes, defs, args), 'idstr' (single-token string such as a
    model id or schema key) or 'prose' (any other string). Docstrings and comments are not code and are skipped."""
    skip = _docstring_ids(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            yield "ident", node.id, line
        elif isinstance(node, ast.Attribute):
            yield "ident", node.attr, line
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "ident", node.name, line
        elif isinstance(node, ast.arg):
            yield "ident", node.arg, line
        elif isinstance(node, ast.keyword) and node.arg:
            yield "ident", node.arg, getattr(node.value, "lineno", line)
        elif isinstance(node, ast.alias):
            yield "ident", node.name.split(".")[-1], line
            if node.asname:
                yield "ident", node.asname, line
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            text = node.value
            yield ("idstr" if IDLIKE_RE.match(text) else "prose"), text, line


def _clip(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def scan_tokens(module: Module, rule: str, patterns: dict[str, tuple[re.Pattern, tuple[str, ...]]],
                *, skip_uv: bool = False) -> list[Violation]:
    """patterns: label -> (regex, kinds it applies to)."""
    found: list[Violation] = []
    if skip_uv and UV_BUILDER_RE.search(module.rel):
        return found
    seen: set[tuple[int, str, str]] = set()
    for kind, text, line in tokens_of(module.tree):
        for label, (regex, kinds) in patterns.items():
            if kind in kinds and regex.search(text):
                key = (line, label, text)
                if key in seen:
                    continue
                seen.add(key)
                found.append(Violation(rule, _repo_rel(module.path), line, f"{label}: {kind} {_clip(text)!r}"))
    return found


# --------------------------------------------------------------------------------------------------------------------
# Call-site analysis (rules b, c, d)
# --------------------------------------------------------------------------------------------------------------------
def _callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")


def _callee_text(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:  # pragma: no cover
        return _callee_name(call)


def _names_in(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


class _FuncCtx:
    def __init__(self, node, qualname: str):
        self.node = node
        self.qualname = qualname
        self.assigns: dict[str, list[ast.AST]] = {}
        self.params = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        for sub in ast.walk(node):
            targets: list[ast.AST] = []
            value = None
            if isinstance(sub, ast.Assign):
                targets, value = list(sub.targets), sub.value
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)) and sub.value is not None:
                targets, value = [sub.target], sub.value
            elif isinstance(sub, ast.NamedExpr):
                targets, value = [sub.target], sub.value
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        self.assigns.setdefault(t.id, []).append(value)

    def matrix_text(self, node: ast.AST, depth: int = 3, seen: set[str] | None = None) -> str:
        seen = seen if seen is not None else set()
        chunks = [ast.unparse(node)]
        if depth <= 0:
            return " ".join(chunks)
        for name in _names_in(node):
            if name in seen:
                continue
            seen.add(name)
            chunks.append(name)
            for value in self.assigns.get(name, []):
                chunks.append(self.matrix_text(value, depth - 1, seen))
        return " ".join(chunks)


@dataclass
class CallSite:
    module: Module
    call: ast.Call
    qualname: str
    ctx: _FuncCtx | None
    in_loop: bool

    @property
    def where(self) -> tuple[str, int]:
        return _repo_rel(self.module.path), self.call.lineno


def call_sites(module: Module) -> list[CallSite]:
    sites: list[CallSite] = []

    def visit(node: ast.AST, scope: list[str], ctx: _FuncCtx | None, in_loop: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join([*scope, child.name])
                visit(child, [*scope, child.name], _FuncCtx(child, qual), False)
            elif isinstance(child, ast.ClassDef):
                visit(child, [*scope, child.name], ctx, False)
            else:
                loop = in_loop or isinstance(child, (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
                if isinstance(child, ast.Call):
                    sites.append(CallSite(module, child, ".".join(scope) or "<module>", ctx, in_loop))
                visit(child, scope, ctx, loop)

    visit(module.tree, [], None, False)
    return sites


def _allowed(table: dict[tuple[str, str], str], rel: str, qualname: str) -> str | None:
    return table.get((rel, qualname)) or table.get((rel, "*"))


def check_crop_paste(module: Module) -> list[Violation]:
    found: list[Violation] = []
    for site in call_sites(module):
        name = _callee_name(site.call)
        path, line = site.where
        if name in {"estimateAffinePartial2D", "estimateAffine2D", "estimateRigidTransform"}:
            if _allowed(ALLOWED_ESTIMATE_AFFINE, module.rel, site.qualname):
                continue
            found.append(Violation("b", path, line, f"cv2.{name}() fits a 5-point similarity; 5-point alignment feeds crop-paste "
                                                    f"(in {site.qualname}); allow-list only for embedding-input alignment"))
        elif name in {"warpAffine", "warp_affine"}:
            if _allowed(ALLOWED_WARP_AFFINE, module.rel, site.qualname):
                continue
            args = site.call.args
            matrix = args[1] if len(args) > 1 else next((k.value for k in site.call.keywords if k.arg in {"M", "matrix"}), None)
            text = site.ctx.matrix_text(matrix) if (matrix is not None and site.ctx is not None) else (ast.unparse(matrix) if matrix is not None else "")
            context = f"{site.qualname.rpartition('.')[2]} {text}"      # function leaf only: class names are not evidence
            hit = ALIGN_MARKER_RE.search(context)
            if hit:
                found.append(Violation("b", path, line, f"{_callee_text(site.call)}() with a matrix derived from alignment/paste "
                                                        f"(marker {hit.group(0)!r}; in {site.qualname}); not on the justified allow-list"))
    return found


def check_25d_calls(module: Module) -> list[Violation]:
    found: list[Violation] = []
    funcs_with_triangle_fit: set[str] = set()
    for site in call_sites(module):
        if _callee_name(site.call) == "getAffineTransform":
            funcs_with_triangle_fit.add(site.qualname)
            path, line = site.where
            found.append(Violation("c", path, line, f"cv2.getAffineTransform() is the per-2D-triangle warp primitive (in {site.qualname})"))
    for site in call_sites(module):
        if _callee_name(site.call) in {"warpAffine", "warp_affine"} and site.qualname in funcs_with_triangle_fit:
            path, line = site.where
            found.append(Violation("c", path, line, f"warpAffine in the same function as getAffineTransform: per-triangle piecewise warper ({site.qualname})"))
    return found


def check_gan_calls(module: Module) -> list[Violation]:
    """Call sites of GAN restorers. Outside UV builders any call is a violation; inside, no frame/crop-looking argument."""
    found: list[Violation] = []
    in_uv = bool(UV_BUILDER_RE.search(module.rel))
    for site in call_sites(module):
        callee = _callee_text(site.call)
        if not GAN_CALL_RE.search(callee):
            continue
        path, line = site.where
        if not in_uv:
            found.append(Violation("d", path, line, f"call {callee}(...) to a 2D GAN restorer outside a UV-texture builder (in {site.qualname})"))
            continue
        arg_names: set[str] = set()
        for arg in [*site.call.args, *(k.value for k in site.call.keywords)]:
            arg_names |= _names_in(arg)
        frameish = sorted(n for n in arg_names if FRAME_ARG_RE.search(n) and not UV_ARG_RE.search(n))
        if frameish or not any(UV_ARG_RE.search(n) for n in arg_names):
            found.append(Violation("d", path, line, f"call {callee}(...) in a UV builder does not operate on a UV map "
                                                    f"(frame/crop-like args: {frameish or 'none; no uv/atlas argument either'}; in {site.qualname})"))
    return found


# --------------------------------------------------------------------------------------------------------------------
# JSON catalogs / registries
# --------------------------------------------------------------------------------------------------------------------
def catalog_json_files() -> list[Path]:
    files: list[Path] = []

    def walk(root: Path) -> None:
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                if entry.name in SKIP_DIRS_ANY or entry.name.startswith(".") or (root == PKG_ROOT and entry.name in SKIP_DIRS_TOP):
                    continue
                walk(entry)
            elif entry.suffix == ".json":
                if JSON_SKIP_NAMES.search(entry.name):
                    continue
                rel = entry.relative_to(PKG_ROOT).as_posix()
                if rel.startswith("audit/") and not JSON_AUDIT_KEEP.search(entry.name):
                    continue
                files.append(entry)

    walk(PKG_ROOT)
    return files


def scan_json_text(path: Path, rule: str, regexes: dict[str, re.Pattern]) -> list[Violation]:
    found: list[Violation] = []
    rel = _repo_rel(path)
    for number, text in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for label, regex in regexes.items():
            m = regex.search(text)
            if m:
                found.append(Violation(rule, rel, number, f"{label}: {_clip(text.strip(), 90)!r}"))
                break
    return found


# --------------------------------------------------------------------------------------------------------------------
# Collection per rule
# --------------------------------------------------------------------------------------------------------------------
IDENT_KINDS = ("ident", "idstr")
ALL_KINDS = ("ident", "idstr", "prose")


@lru_cache(maxsize=None)
def rule_violations(rule: str) -> tuple[Violation, ...]:
    out: list[Violation] = []
    mods = active_list()
    if rule == "a":
        pats = {"2D swapper model id": (SWAP_MODEL_RE, ALL_KINDS), "2D swapper identifier": (SWAP_WORD_RE, IDENT_KINDS)}
        for m in mods:
            out += scan_tokens(m, "a", pats)
        out += [v for p in catalog_json_files() for v in scan_json_text(p, "a", {"2D swapper model id": SWAP_MODEL_RE,
                                                                                 "swapper catalog key": re.compile(r'"[^"]*swapper[^"]*"\s*:|"swapper(_2d)?"', re.I)})]
    elif rule == "b":
        for m in mods:
            out += check_crop_paste(m)
    elif rule == "c":
        pats = {"weak-perspective/orthographic camera": (CAMERA_25D_RE, ALL_KINDS), "2.5D geometry": (GEOM_25D_RE, IDENT_KINDS)}
        for m in mods:
            out += scan_tokens(m, "c", pats)
            out += check_25d_calls(m)
        out += [v for p in catalog_json_files() for v in scan_json_text(p, "c", {"2.5D": GEOM_25D_RE, "weak-perspective": CAMERA_25D_RE})]
    elif rule == "d":
        pats = {"2D GAN restorer": (GAN_NAME_RE, ALL_KINDS), "2D restoration option": (GAN_OPTION_RE, IDENT_KINDS)}
        for m in mods:
            out += scan_tokens(m, "d", pats, skip_uv=True)
            out += check_gan_calls(m)
        out += [v for p in catalog_json_files() for v in scan_json_text(p, "d", {"2D GAN restorer": GAN_NAME_RE})]
    else:  # pragma: no cover
        raise KeyError(rule)
    unique = {(v.path, v.line, v.detail): v for v in out}
    return tuple(sorted(unique.values(), key=lambda v: (v.path, v.line, v.detail)))


def _dump_report() -> None:
    target = os.environ.get("ZERO3D_POLICY_REPORT")
    if not target:
        return
    payload = {
        "active_module_count": len(active_modules()),
        "violations": {r: [str(v) for v in rule_violations(r)] for r in ("a", "b", "c", "d")},
    }
    Path(target).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _assert_clean(rule: str, title: str, limit: int = 60) -> None:
    _dump_report()
    violations = rule_violations(rule)
    if violations:
        shown = "\n  ".join(str(v) for v in violations[:limit])
        more = f"\n  ... and {len(violations) - limit} more (set ZERO3D_POLICY_REPORT=<file> for all)" if len(violations) > limit else ""
        files = len({v.path for v in violations})
        pytest.fail(f"{title}: {len(violations)} violation(s) in {files} file(s) of {len(active_modules())} active modules:\n  {shown}{more}", pytrace=False)


# --------------------------------------------------------------------------------------------------------------------
# Meta tests: the analysis itself is sound
# --------------------------------------------------------------------------------------------------------------------
def test_active_set_is_computed_from_the_import_graph() -> None:
    modules, parse_errors = module_index()
    assert not parse_errors, "unparseable modules:\n  " + "\n  ".join(map(str, parse_errors))
    assert MUST_EXIST_ROOT in modules, f"{MUST_EXIST_ROOT} is the ASGI app root and must exist"
    active = active_modules()
    assert MUST_EXIST_ROOT in active
    assert len(active) >= 20, f"only {len(active)} active modules found; the import-graph walk is broken"
    assert not [n for n in active if ".tests" in n or n.endswith("test_zero_2d_and_25d_policy")]


def test_scanner_detects_planted_violations() -> None:
    """The gate must be able to fail: plant every banned shape in synthetic source and require a hit for each."""
    src = '''
import cv2, numpy as np
from roop_3d.core.settings import swapper_model_id
MODEL = "inswapper_128"
CATALOG = {"simswap_256": 1, "ghost_256_unet_1": 2, "blendswap_256": 3, "uniface_256": 4}
class WeakPerspectiveCamera: ...
def align(pts, ref):
    M, _ = cv2.estimateAffinePartial2D(pts, ref)
    return cv2.warpAffine(np.zeros((8, 8)), M, (8, 8))
def paste(frame, crop, inverse):
    return cv2.warpAffine(crop, inverse, frame.shape[:2])
def warp_delaunay_piecewise_affine(img, tris):
    for t in tris:
        m = cv2.getAffineTransform(t, t)
        img = cv2.warpAffine(img, m, (4, 4))
def post(frame, aligned_crop, codeformer):
    return codeformer.restore_aligned(aligned_crop)
'''
    module = Module("roop_3d.synthetic", Path("synthetic.py"), "synthetic.py", False, ast.parse(src))
    a = scan_tokens(module, "a", {"id": (SWAP_MODEL_RE, ALL_KINDS), "word": (SWAP_WORD_RE, IDENT_KINDS)})
    assert {"inswapper_128", "simswap_256", "ghost_256_unet_1", "blendswap_256", "uniface_256", "swapper_model_id"} <= {
        re.search(r"'([^']+)'", v.detail).group(1) for v in a}
    assert len(check_crop_paste(module)) >= 3
    assert len(check_25d_calls(module)) >= 2
    c = scan_tokens(module, "c", {"cam": (CAMERA_25D_RE, ALL_KINDS), "geo": (GEOM_25D_RE, IDENT_KINDS)})
    assert any("WeakPerspectiveCamera" in v.detail for v in c) and any("warp_delaunay" in v.detail for v in c)
    assert check_gan_calls(module) and scan_tokens(module, "d", {"gan": (GAN_NAME_RE, ALL_KINDS)})


def test_uv_builder_may_call_a_detailer_only_on_uv_maps() -> None:
    good = Module("roop_3d.rendering.uv_x", Path("uv_x.py"), "rendering/uv_x.py", False, ast.parse(
        "def build(uv_atlas):\n    return detailer.restore_hires(uv_atlas)\n"))
    bad = Module("roop_3d.rendering.uv_y", Path("uv_y.py"), "rendering/uv_y.py", False, ast.parse(
        "def build(frame):\n    return detailer.restore_hires(frame)\n"))
    assert check_gan_calls(good) == []
    assert len(check_gan_calls(bad)) == 1


def test_allow_lists_have_reasons_and_are_not_stale() -> None:
    modules, _ = module_index()
    by_rel = {m.rel: m for m in modules.values()}
    problems: list[str] = []
    for table, call_names in ((ALLOWED_ESTIMATE_AFFINE, {"estimateAffinePartial2D", "estimateAffine2D"}),
                              (ALLOWED_WARP_AFFINE, {"warpAffine", "warp_affine"})):
        for (rel, qual), reason in table.items():
            if len(reason.strip()) < 25:
                problems.append(f"{rel}::{qual}: allow-list entry needs a real reason")
            module = by_rel.get(rel)
            if module is None:
                problems.append(f"{rel}::{qual}: stale allow-list entry (module no longer exists)")
                continue
            sites = [s for s in call_sites(module) if _callee_name(s.call) in call_names]
            if not any(qual in {"*", s.qualname} for s in sites):
                problems.append(f"{rel}::{qual}: stale allow-list entry (no matching call site any more)")
    assert not problems, "\n".join(problems)


def test_no_dangling_internal_imports_in_active_modules() -> None:
    """A purge must not leave an active module importing a deleted module or a deleted name (`file:line` for each)."""
    modules, _ = module_index()
    tops: dict[str, tuple[set[str], bool]] = {}

    def top_names(module: Module) -> tuple[set[str], bool]:
        if module.name in tops:
            return tops[module.name]
        names: set[str] = set()
        star = False

        def collect(body: list[ast.stmt]) -> None:
            nonlocal star
            for stmt in body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(stmt.name)
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                        names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
                elif isinstance(stmt, ast.Import):
                    names.update((a.asname or a.name.split(".")[0]) for a in stmt.names)
                elif isinstance(stmt, ast.ImportFrom):
                    for a in stmt.names:
                        if a.name == "*":
                            star = True
                        names.add(a.asname or a.name)
                elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for attr in ("body", "orelse", "finalbody"):
                        collect(getattr(stmt, attr, []))
                    for handler in getattr(stmt, "handlers", []):
                        collect(handler.body)

        collect(module.tree.body)
        if "__getattr__" in names:
            star = True
        tops[module.name] = (names, star)
        return tops[module.name]

    problems: list[str] = []
    for module in active_list():
        for ref in imports_of(module):
            target = modules.get(ref.module)
            where = f"{_repo_rel(module.path)}:{ref.line}"
            if target is None:
                if ref.module.startswith(PKG + ".") and not any(ref.module == n or n.startswith(ref.module + ".") for n in modules):
                    problems.append(f"{where}: imports {ref.module!r}, which does not exist")
                continue
            names, star = top_names(target)
            for name in ref.names:
                if name == "*" or star or name in names or f"{ref.module}.{name}" in modules:
                    continue
                problems.append(f"{where}: `from {ref.module} import {name}` but {target.rel} defines no {name!r}")
    assert not problems, f"{len(problems)} dangling import(s):\n  " + "\n  ".join(problems[:80])


# --------------------------------------------------------------------------------------------------------------------
# The gate: rules a-d over the active render path
# --------------------------------------------------------------------------------------------------------------------
def test_a_zero_2d_swapper_identifiers_and_model_ids() -> None:
    _assert_clean("a", "2D swapper identifiers/model ids (inswapper_128, simswap, ghost, reswapper, blendswap, uniface, hyperswap, realswap, hififace)")


def test_b_zero_five_point_crop_paste() -> None:
    _assert_clean("b", "5-point crop-paste (estimateAffinePartial2D / alignment-derived warpAffine)")


def test_c_zero_25d_geometry() -> None:
    _assert_clean("c", "2.5D geometry (weak-perspective/orthographic cameras, 3DDFA_V2 strip, per-triangle warpAffine)")


def test_d_zero_2d_post_crop_gan_calls() -> None:
    _assert_clean("d", "2D post-crop GAN restorers (only UV-texture builders on UV maps may use them)")


def test_residual_purged_modules_are_gone() -> None:
    """Modules whose only purpose was a 2D/2.5D path must not linger on disk, reachable or not."""
    must_not_exist = [
        "swappers", "enhancers/styleganex_age.py", "engine/volumetric/identity_swapper.py", "engine/volumetric/swapper_engine.py",
        "pose_matrix.py", "models/emap_cache.py", "hires_handoff.py", "reconstruction/pose_blend.py", "reconstruction/jaw_fit.py",
        "reconstruction/expression_tps.py", "reconstruction/frontalization.py", "reconstruction/tps_warp.py",
        "engine/stage16_pipeline.py",
    ]
    present = [p for p in must_not_exist if (PKG_ROOT / p).exists()]
    assert not present, "purged paths still on disk (app/roop_3d/<path>):\n  " + "\n  ".join(present)


# --------------------------------------------------------------------------------------------------------------------
# e) Runtime: True3DArchitectureGuard (core/true3d_guard.py)
# --------------------------------------------------------------------------------------------------------------------
def _uv_sphere(rings: int, segments: int, radii=(0.075, 0.115, 0.095)):
    """Closed, consistently wound ellipsoid in the head frame (x right, y DOWN, face towards -z)."""
    import numpy as np
    verts = [(0.0, -radii[1], 0.0)]                              # crown (y is down, so the top is -y)
    for i in range(1, rings):
        theta = np.pi * i / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            verts.append((radii[0] * np.sin(theta) * np.cos(phi), -radii[1] * np.cos(theta), radii[2] * np.sin(theta) * np.sin(phi)))
    verts.append((0.0, radii[1], 0.0))
    faces = []
    for j in range(segments):
        faces.append((0, 1 + (j + 1) % segments, 1 + j))
    for i in range(rings - 2):
        for j in range(segments):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            c, d = a + segments, b + segments
            faces += [(a, b, c), (b, d, c)]
    last = len(verts) - 1
    base = 1 + (rings - 2) * segments
    for j in range(segments):
        faces.append((last, base + j, base + (j + 1) % segments))
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def _head_fixture():
    """Watertight full-head stand-in with region labels, two eyeball spheres and an oral cavity, all inside the head."""
    import numpy as np
    v, f = _uv_sphere(40, 56)
    y = v[:, 1]
    drop = (y - y.min()) / float(np.ptp(y))                       # y is down: 0 at the crown, 1 at the lowest vertex
    front = v[:, 2] < 0.0                                          # the face points to -z
    lateral = np.abs(v[:, 0]) > 0.55 * np.abs(v[:, 0]).max()
    regions = {
        "cranium": np.flatnonzero(drop < 0.30),
        "forehead_dome": np.flatnonzero((drop < 0.36) & front & ~lateral),
        "temples": np.flatnonzero((drop > 0.14) & (drop < 0.42) & lateral),
        "cheekbones": np.flatnonzero((drop > 0.36) & (drop < 0.60) & front & lateral),
        "mandible": np.flatnonzero((drop > 0.52) & (drop < 0.88) & (front | lateral)),
        "submental_neck": np.flatnonzero(drop > 0.80),
    }
    rng = np.random.default_rng(7)

    def ball(center, radius, n=96):
        d = rng.normal(size=(n, 3))
        return np.asarray(center) + radius * d / np.linalg.norm(d, axis=1, keepdims=True)

    eyes = np.vstack([ball((-0.032, -0.025, -0.060), 0.012), ball((0.032, -0.025, -0.060), 0.012)])
    mouth = rng.uniform((-0.02, 0.040, -0.075), (0.02, 0.060, -0.040), size=(80, 3))
    return v, f, regions, {"eyeballs": eyes, "oral_cavity": mouth}


def _valid_pose():
    import numpy as np
    K = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]])
    angle = np.deg2rad(20.0)
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return K, R, np.array([0.0, 0.0, 0.7])


@pytest.fixture(scope="module")
def guard():
    from roop_3d.core.true3d_guard import True3DArchitectureGuard
    return True3DArchitectureGuard()


def test_e_guard_accepts_valid_full_head_with_perspective_camera(guard) -> None:
    v, f, regions, parts = _head_fixture()
    K, R, t = _valid_pose()
    report = guard.assert_true_3d(mesh=dict(vertices=v, faces=f, regions=regions, parts=parts),
                                  camera=dict(K=K, R=R, t=t, image_size=(1280, 720)),
                                  config={"projection": "perspective", "rasterizer": "z_buffer",
                                          "shader": "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV"})
    assert report.mesh.watertight and report.mesh.ok and not report.mesh.problems
    assert report.camera.ok and report.camera.min_depth > 0.5
    assert report.config.ok


def test_e_guard_rejects_frontal_strip_mesh(guard) -> None:
    import numpy as np
    from roop_3d.core.true3d_guard import HeadMeshError, True3DArchitectureError
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(3)
    xy = rng.uniform(-0.08, 0.08, size=(76, 2))                       # 68 landmarks + boundary points, Delaunay on (x, y)
    strip_v = np.column_stack([xy, np.full(76, -0.01)])
    strip_f = Delaunay(xy).simplices.astype(np.int64)
    with pytest.raises(HeadMeshError):
        guard.validate_head_mesh(strip_v, strip_f)
    # A DENSE but flat height-field (2.5D) must fail too: vertex count alone must not be enough.
    gx, gy = np.meshgrid(np.linspace(-0.08, 0.08, 50), np.linspace(-0.11, 0.11, 50))
    flat_v = np.column_stack([gx.ravel(), gy.ravel(), -0.02 * np.exp(-(gx.ravel() ** 2 + gy.ravel() ** 2) / 0.004)])
    flat_f = Delaunay(flat_v[:, :2]).simplices.astype(np.int64)
    with pytest.raises(True3DArchitectureError):
        guard.validate_head_mesh(flat_v, flat_f)


def test_e_guard_rejects_non_watertight_mesh(guard) -> None:
    import numpy as np
    from roop_3d.core.true3d_guard import HeadMeshError
    v, f, regions, parts = _head_fixture()
    holed = np.delete(f, np.arange(200, 212), axis=0)                  # punch a hole in the surface
    with pytest.raises(HeadMeshError, match="watertight"):
        guard.validate_head_mesh(v, holed, regions=regions, parts=parts)


def test_e_guard_rejects_front_only_shell_of_a_real_head(guard) -> None:
    """Dropping the back half of a closed head leaves an open, front-only shell: not a complete head."""
    import numpy as np
    from roop_3d.core.true3d_guard import HeadMeshError
    v, f, regions, parts = _head_fixture()
    keep = v[:, 2] <= 0.0
    mask = keep[f].all(axis=1)
    remap = -np.ones(len(v), np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    with pytest.raises(HeadMeshError):
        guard.validate_head_mesh(v[keep], remap[f[mask]], parts=parts)


def test_e_guard_rejects_affine_orthographic_cameras(guard) -> None:
    import numpy as np
    from roop_3d.core.true3d_guard import AffineCameraError, RenderConfigError
    K, R, t = _valid_pose()
    scaled_ortho = np.array([[800.0, 0.0, 0.0], [0.0, 800.0, 0.0]])        # 2x3 weak-perspective projection
    with pytest.raises(AffineCameraError):
        guard.validate_camera(scaled_ortho)
    affine_k = K.copy()
    affine_k[2] = (0.0, 0.0, 0.0)
    with pytest.raises(AffineCameraError):
        guard.validate_camera(affine_k, R, t)
    affine_p = np.hstack([K @ R, (K @ t).reshape(3, 1)])
    affine_p[2, :3] = 0.0                                                     # last row (0, 0, 0, s): orthographic
    with pytest.raises(AffineCameraError):
        guard.validate_camera(K, R, t, projection_matrix=affine_p)
    for banned in ("orthographic", "weak_perspective", "affine"):
        with pytest.raises(RenderConfigError):
            guard.validate_render_config({"projection": banned, "rasterizer": "z_buffer", "shader": "pbr subsurface"})


def test_e_guard_rejects_non_se3_pose_and_head_behind_camera(guard) -> None:
    import numpy as np
    from roop_3d.core.true3d_guard import CameraPoseError
    K, R, t = _valid_pose()
    v, *_ = _head_fixture()
    for bad_r in (R * 1.3,                                                   # scaled (weak-perspective style) rotation
                  R @ np.diag([1.0, 1.0, -1.0]),                             # reflection, det = -1
                  np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])):   # shear
        with pytest.raises(CameraPoseError):
            guard.validate_camera(K, bad_r, t)
    with pytest.raises(CameraPoseError):
        guard.validate_camera(K, R, np.array([0.0, np.nan, 0.7]))
    with pytest.raises(CameraPoseError):
        guard.validate_camera(K, R, t * np.array([1, 1, -1]), vertices=v)   # head behind the camera
    with pytest.raises(CameraPoseError):
        guard.validate_camera(K, R, None, vertices=v)                        # a pose needs both R and t


def test_e_guard_rejects_banned_render_config_keys(guard) -> None:
    from roop_3d.core.true3d_guard import RenderConfigError
    base = {"projection": "perspective", "rasterizer": "z_buffer", "shader": "3D-Gaussian-Head-Avatar"}
    assert guard.validate_render_config(base).ok
    for key in ("swapper_model_id", "restorer_model", "crop_paste", "mesh_blend", "frontalization"):
        with pytest.raises(RenderConfigError):
            guard.validate_render_config({**base, key: "on"})
    with pytest.raises(RenderConfigError):
        guard.validate_render_config({**base, "rasterizer": "painters_algorithm"})


# --------------------------------------------------------------------------------------------------------------------
# f) Schemas and UI contracts
# --------------------------------------------------------------------------------------------------------------------
OPTION_PATTERNS = {
    "2D swapper model id": SWAP_MODEL_RE,
    "swapper option": re.compile(r"swapper|dfm_morph", re.I),
    "2D GAN restorer": GAN_NAME_RE,
    "2D restoration option": GAN_OPTION_RE,
    "2.5D / weak-perspective option": re.compile(CAMERA_25D_RE.pattern + "|" + GEOM_25D_RE.pattern, re.I),
}
PROSE_PATTERNS = {"2D swapper model id": SWAP_MODEL_RE, "2D GAN restorer": GAN_NAME_RE}


def _source_location(needle: str) -> str:
    for module in active_list():
        for number, text in enumerate(module.path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if needle in text:
                return f"{_repo_rel(module.path)}:{number}"
    return "(not found in active python source; see catalogs/JSON)"


def _walk_schema(node, pointer: str, found: list[tuple[str, str, str]]) -> None:
    def check(text: str, where: str, *, key: bool) -> None:
        patterns = OPTION_PATTERNS if (key or IDLIKE_RE.match(text)) else PROSE_PATTERNS
        for label, regex in patterns.items():
            if regex.search(text):
                found.append((where, label, text))
                return

    if isinstance(node, dict):
        for key, value in node.items():
            check(str(key), f"{pointer}/{key}", key=True)
            _walk_schema(value, f"{pointer}/{key}", found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk_schema(value, f"{pointer}/{index}", found)
    elif isinstance(node, str):
        check(node, pointer, key=False)


def test_f_openapi_schema_exposes_no_removed_options(tmp_path) -> None:
    from roop_3d.core.settings import Settings
    from roop_3d.server.app import create_app
    schema = create_app(Settings(), repo_root=tmp_path, worker_enabled=False).openapi()
    found: list[tuple[str, str, str]] = []
    _walk_schema(schema, "#", found)
    if found:
        lines = [f"{_source_location(text.split('/')[-1] if '/' in text else text)}: [f] openapi {where} -> {label}: {_clip(text, 70)!r}"
                 for where, label, text in found[:60]]
        pytest.fail(f"OpenAPI schema exposes {len(found)} removed swapper/enhancer/2.5D option(s):\n  " + "\n  ".join(lines), pytrace=False)


_COMMENT_RE = re.compile(r"/\*.*?\*/|(?<![:\"'])//[^\n]*", re.S)


def _ui_violations(files: list[Path]) -> list[Violation]:
    found: list[Violation] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        stripped = _COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)     # keeps line numbers
        for number, line in enumerate(stripped.splitlines(), 1):
            for label, regex in OPTION_PATTERNS.items():
                if regex.search(line):
                    found.append(Violation("f", _repo_rel(path), number, f"{label}: {_clip(line.strip(), 90)!r}"))
                    break
    return found


def _fail_if(violations: list[Violation], title: str) -> None:
    if violations:
        shown = "\n  ".join(map(str, violations[:60]))
        more = f"\n  ... and {len(violations) - 60} more" if len(violations) > 60 else ""
        pytest.fail(f"{title}: {len(violations)} violation(s) in {len({v.path for v in violations})} file(s):\n  {shown}{more}", pytrace=False)


def test_f_ui_contracts_expose_no_removed_options() -> None:
    contracts = UI_SRC / "contracts.ts"
    assert contracts.exists(), f"{_repo_rel(contracts)} missing"
    _fail_if(_ui_violations([contracts]), "UI contracts.ts exposes removed swapper/enhancer/2.5D options")


def _ui_source_files() -> list[Path]:
    out: list[Path] = []
    for path in sorted(UI_SRC.rglob("*")):
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx"} or not path.is_file():
            continue
        parts = path.relative_to(UI_SRC).parts
        if any(p in {"__tests__", "node_modules", "dist"} for p in parts) or re.search(r"\.(test|spec|e2e)\.", path.name):
            continue
        out.append(path)
    return out


def test_f_ui_sources_expose_no_removed_options() -> None:
    files = _ui_source_files()
    assert len(files) > 20, "UI source tree not found"
    _fail_if(_ui_violations(files), "UI source exposes removed swapper/enhancer/2.5D options")
