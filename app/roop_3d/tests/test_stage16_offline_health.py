"""Unit tests for Stage 16 offline dependency cache and offline readiness verification."""
from __future__ import annotations

from pathlib import Path
import pytest

from roop_3d.installer.offline_cache import (
    check_offline_dependency_cache,
    check_offline_readiness,
)
from roop_3d.models.registry import ModelRecord, ModelRegistry


def test_offline_dependency_cache(tmp_path: Path) -> None:
    # Test with empty root
    res_empty = check_offline_dependency_cache(tmp_path)
    assert res_empty["status"] == "WARN"
    assert res_empty["venv_present"] is False

    # Test with venv present
    venv_dir = tmp_path / "app/roop_3d/.venv"
    venv_dir.mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text("home = /bin", encoding="utf-8")

    res_venv = check_offline_dependency_cache(tmp_path)
    assert res_venv["status"] == "READY"
    assert res_venv["venv_present"] is True


def test_offline_readiness_reporting(tmp_path: Path) -> None:
    registry = ModelRegistry()
    res = check_offline_readiness(tmp_path, registry)
    assert "offline_ready" in res
    assert "core_models" in res
    assert "missing_models" in res
    assert res["zero_remote_calls"] is True
    assert len(res["core_models"]) >= 3
