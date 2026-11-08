"""Standalone install integrity: real (non-linked), gitignored local directories and in-sync npm manifests.

Layout mapping for this checkout: the Python environment is `app/roop_3d/.venv`, FaceSets live in
`app/roop_3d/.runtime/facesets`, and model weights are read-only external inputs reached through the
hashed registry, so no weights directory exists inside the repository by design (AGENTS.md).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from roop_3d.audit.isolation import safe_path
from roop_3d.audit.project_paths import application_root, project_root

WEIGHT_SUFFIXES = {".onnx", ".engine", ".plan", ".trt", ".pth", ".pt", ".safetensors", ".ckpt", ".bin"}
RUNTIME_DIRECTORIES = ("roop_3d/.venv", "roop_3d/.runtime", "roop_3d/.runtime/facesets")


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(project_root()), *arguments], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("relative", RUNTIME_DIRECTORIES)
def test_local_directories_are_real_directories_inside_the_checkout(relative: str) -> None:
    root = application_root()
    path = root / relative
    if not path.exists():
        pytest.skip(f"{relative} is created on first use")
    # safe_path rejects symlink/junction traversal and anything resolving outside the owned root.
    assert safe_path(root, relative) == path.absolute()
    assert path.is_dir() and not path.is_symlink()
    assert path.resolve().is_relative_to(project_root().resolve())


def test_virtual_environment_is_dedicated() -> None:
    environment = application_root() / "roop_3d/.venv"
    assert environment.is_dir()
    assert (environment / "pyvenv.cfg").is_file()


@pytest.mark.parametrize("relative", ("app/roop_3d/.venv", "app/roop_3d/.runtime/facesets", "app/roop_3d_ui/node_modules",
                                      "app/roop_3d_ui/dist"))
def test_local_state_is_gitignored(relative: str) -> None:
    assert _git("check-ignore", "-q", "--no-index", relative).returncode == 0, f"{relative} must be gitignored"
    assert _git("ls-files", "--", relative).stdout.strip() == "", f"{relative} must not be tracked"


def test_no_model_weights_or_engines_are_tracked() -> None:
    tracked = _git("ls-files").stdout.splitlines()
    assert tracked, "git ls-files returned nothing"
    assert [name for name in tracked if Path(name).suffix.lower() in WEIGHT_SUFFIXES] == []


def test_frontend_lockfile_matches_package_manifest() -> None:
    """A drifted lockfile is what makes `npm ci` fail; compare the manifest to the lock's root entry."""
    ui = application_root() / "roop_3d_ui"
    manifest = json.loads((ui / "package.json").read_text("utf-8"))
    lock = json.loads((ui / "package-lock.json").read_text("utf-8"))
    root_entry = lock["packages"][""]
    for section in ("dependencies", "devDependencies", "engines"):
        assert root_entry.get(section, {}) == manifest.get(section, {}), section
    assert manifest["engines"]["node"] == "^20.19.0 || >=22.12.0"
    assert lock["lockfileVersion"] >= 3


def test_frontend_references_no_remote_font_or_asset_hosts() -> None:
    ui = application_root() / "roop_3d_ui"
    offenders = []
    for path in [ui / "index.html", *sorted((ui / "src").rglob("*.css"))]:
        text = path.read_text("utf-8")
        offenders += [f"{path.name}: {host}" for host in ("fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net",
                                                          "unpkg.com", "cdnjs.cloudflare.com") if host in text]
    assert offenders == []
    assert "@fontsource-variable/inter" in (ui / "src/main.tsx").read_text("utf-8")


# -- Stage 27 certification: standalone means real directories and nothing fetched from the network ------------------

APP_STATE_DIRECTORIES = ("env", "models", "facesets")
REPARSE_POINT = 0x400   # FILE_ATTRIBUTE_REPARSE_POINT: Windows junctions and symlinks both carry it


@pytest.mark.parametrize("name", APP_STATE_DIRECTORIES)
def test_app_state_directories_are_real_directories_not_links(name: str) -> None:
    """`app/env`, `app/models` and `app/facesets` must hold their own files; a symlink or junction would make the install depend on another folder."""
    import os
    path = application_root() / name
    if not path.exists():
        pytest.skip(f"app/{name} is created on first use")
    info = os.lstat(path)
    assert not path.is_symlink(), f"app/{name} is a symlink"
    assert not getattr(info, "st_file_attributes", 0) & REPARSE_POINT, f"app/{name} is a junction / reparse point"
    assert path.is_dir()
    assert path.resolve().is_relative_to(project_root().resolve()), f"app/{name} resolves outside the checkout"
    for child in path.rglob("*"):   # nothing inside may be a link either (bounded: these folders hold cache bundles, not trees)
        assert not child.is_symlink(), f"{child.relative_to(project_root())} is a symlink"


@pytest.mark.parametrize("name", APP_STATE_DIRECTORIES)
def test_app_state_directories_are_gitignored_and_untracked(name: str) -> None:
    assert _git("check-ignore", "-q", "--no-index", f"app/{name}").returncode == 0, f"app/{name} must be gitignored"
    assert _git("ls-files", "--", f"app/{name}").stdout.strip() == "", f"app/{name} must not be tracked"


# Hosts that appear in production UI text but are never fetched: the SVG / XML namespace identifiers.
NAMESPACE_HOSTS = {"www.w3.org"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
KNOWN_CDNS = ("cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com", "fonts.googleapis.com", "fonts.gstatic.com", "ajax.googleapis.com",
              "code.jquery.com", "stackpath.bootstrapcdn.com", "maxcdn.bootstrapcdn.com", "use.fontawesome.com", "esm.sh", "cdn.skypack.dev")
TEST_PATH_PARTS = {"e2e", "harness", "mock-server", "acceptance", "node_modules", "dist", "test-results"}


def _production_sources() -> list[Path]:
    ui = application_root() / "roop_3d_ui"
    files = [ui / "index.html"]
    for suffix in ("*.ts", "*.tsx", "*.js", "*.jsx", "*.css", "*.html"):
        files += [p for p in (ui / "src").rglob(suffix) if ".test." not in p.name and not TEST_PATH_PARTS & set(p.parts)]
    return sorted(set(files))


def _external_hosts(text: str) -> set[str]:
    import re
    return {match.group(1).lower() for match in re.finditer(r"""(?:https?|wss?)://(\[[0-9a-f:]+\]|[A-Za-z0-9._-]+)""", text)}


def test_production_ui_source_references_no_external_host() -> None:
    sources = _production_sources()
    assert len(sources) > 100, "the source scan must actually cover the UI"
    found = {f"{p.relative_to(application_root() / 'roop_3d_ui').as_posix()}: {host}"
             for p in sources for host in _external_hosts(p.read_text("utf-8", errors="replace")) - NAMESPACE_HOSTS - LOCAL_HOSTS}
    assert sorted(found) == []


def test_built_ui_bundle_loads_nothing_remote() -> None:
    import re
    dist = application_root() / "roop_3d_ui/dist"
    if not (dist / "index.html").is_file():
        pytest.skip("run `npm run build` to produce dist/")
    html = (dist / "index.html").read_text("utf-8")
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", html), "index.html loads a remote script or stylesheet"
    for css in dist.rglob("*.css"):
        text = css.read_text("utf-8", errors="replace")
        assert not re.search(r"""url\(\s*["']?https?://""", text) and "@import" not in text, f"{css.name} imports a remote resource"
    runtime_fetch = re.compile(r"""(?:fetch|importScripts|import|new Worker|new WebSocket|new EventSource)\(\s*["'](?:https?|wss?)://(?!localhost|127\.0\.0\.1)""")
    for script in dist.rglob("*.js"):
        text = script.read_text("utf-8", errors="replace")
        assert not runtime_fetch.search(text), f"{script.name} fetches a remote URL at runtime"
        assert not [cdn for cdn in KNOWN_CDNS if cdn in text], f"{script.name} references a CDN"
