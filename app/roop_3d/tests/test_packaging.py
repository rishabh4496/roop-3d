"""The ``[tool.setuptools] packages`` list decides what a built wheel contains. A package missing from it works from a source
checkout (the launcher runs from source) and then fails at import in an installed wheel, so the list is pinned here.

``KNOWN_UNLISTED`` is a PRE-EXISTING gap, written down instead of silently changed: those packages are real, imported by listed
code (``color`` by ``core.workstation`` and the compositor), and absent from the list. Fixing it means adding them to
``pyproject.toml`` and emptying this set; until then a NEW package cannot join them unnoticed.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
KNOWN_UNLISTED = {"color", "enhancers", "evaluation", "identity", "installer", "orchestrator", "security", "swappers"}
SKIPPED_DIRS = {"tests", ".venv", ".runtime", "__pycache__", "docs"}


def listed() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:                                    # the project's venv predates tomllib (3.11): read the one list literal ourselves there
        import tomllib
    except ImportError:
        match = re.search(r"^packages\s*=\s*(\[.*?\])\s*$", text, re.M | re.S)
        assert match, "pyproject.toml has no [tool.setuptools] packages list"
        return ast.literal_eval(match.group(1))
    return tomllib.loads(text)["tool"]["setuptools"]["packages"]


def on_disk() -> set[str]:
    return {d.name for d in ROOT.iterdir() if d.is_dir() and d.name not in SKIPPED_DIRS and (d / "__init__.py").is_file()}


def test_tracking_and_editing_are_in_the_package_list():
    packages = listed()
    assert "roop_3d.tracking" in packages and "roop_3d.editing" in packages


def test_every_package_on_disk_is_listed_or_a_documented_gap():
    unlisted = {name for name in on_disk() if f"roop_3d.{name}" not in listed()}
    assert unlisted == KNOWN_UNLISTED, (
        f"new unlisted package(s): {sorted(unlisted - KNOWN_UNLISTED)}; fixed gap(s) to remove from KNOWN_UNLISTED: "
        f"{sorted(KNOWN_UNLISTED - unlisted)}")


def test_the_listed_packages_all_exist_on_disk():
    assert {p.removeprefix("roop_3d.") for p in listed() if "." in p and p.count(".") == 1} <= on_disk()


def test_setuptools_would_collect_the_tracking_and_editing_modules_into_a_wheel():
    setuptools = pytest.importorskip("setuptools")
    distribution = setuptools.Distribution({"packages": listed(), "package_dir": {"roop_3d": str(ROOT)}, "script_name": "setup.py"})
    build = distribution.get_command_obj("build_py")
    build.ensure_finalized()
    collected = {(package, module) for package, module, _ in build.find_all_modules()}
    assert ("roop_3d.tracking", "video_tracker") in collected and ("roop_3d.tracking", "filters") in collected
    assert ("roop_3d.editing", "expression_audio_director") in collected
