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
