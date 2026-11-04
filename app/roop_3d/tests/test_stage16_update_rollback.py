"""Unit tests for Stage 16 update checking, configuration backup, and safe rollback."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from roop_3d.installer.update_manager import (
    check_update_status,
    create_configuration_backup,
    rollback_to_backup,
)


def test_check_update_status(tmp_path: Path) -> None:
    # Test update checker in a repository
    from roop_3d.audit.project_paths import project_root
    res = check_update_status(project_root())
    assert "status" in res
    assert "current_commit" in res
    assert "current_branch" in res
    assert "changed_components" in res


def test_configuration_backup_and_rollback(tmp_path: Path) -> None:
    # Setup test workspace
    runtime_dir = tmp_path / "roop_3d/.runtime"
    runtime_dir.mkdir(parents=True)
    config_file = runtime_dir / "config.json"
    config_file.write_text(json.dumps({"schema_version": 2, "hardware_profile": "12gb"}), encoding="utf-8")

    proj_file = tmp_path / ".roop3d-project.json"
    proj_file.write_text(json.dumps({"app_id": "roop-3d"}), encoding="utf-8")

    # Outputs and facesets to preserve
    facesets_dir = tmp_path / "roop_3d/outputs/facesets"
    facesets_dir.mkdir(parents=True)
    (facesets_dir / "preserve_me.fsz").write_text("faceset_binary_data", encoding="utf-8")

    outputs_dir = tmp_path / "roop_3d/outputs"
    (outputs_dir / "rendered_output.mp4").write_text("video_binary_data", encoding="utf-8")

    with patch("roop_3d.installer.update_manager._git", return_value="test_commit_hash_12345678"):
        # 1. Create backup
        backup_res = create_configuration_backup(tmp_path)
        assert backup_res["status"] == "PASS"
        assert "backup_id" in backup_res
        assert "config.json" in backup_res["copied_files"]

        # Modify config to simulate an update change
        config_file.write_text(json.dumps({"schema_version": 2, "hardware_profile": "modified_profile"}), encoding="utf-8")

        # 2. Rollback
        rollback_res = rollback_to_backup(tmp_path, backup_res["backup_id"])
        assert rollback_res["status"] == "PASS"

        # Verify config was restored
        restored_data = json.loads(config_file.read_text("utf-8"))
        assert restored_data["hardware_profile"] == "12gb"

        # Verify FaceSets and outputs were preserved
        assert (facesets_dir / "preserve_me.fsz").read_text("utf-8") == "faceset_binary_data"
        assert (outputs_dir / "rendered_output.mp4").read_text("utf-8") == "video_binary_data"


def test_backup_pruning(tmp_path: Path) -> None:
    # Setup backups directory with 8 dummy backups
    backups_dir = tmp_path / "cache/backups"
    backups_dir.mkdir(parents=True)

    for i in range(8):
        d = backups_dir / f"backup_20260101_00000{i}_commit{i}"
        d.mkdir()
        (d / "backup_manifest.json").write_text(json.dumps({"backup_id": d.name}), encoding="utf-8")

    with patch("roop_3d.installer.update_manager._git", return_value="dummy_commit"):
        create_configuration_backup(tmp_path)

    # After create_backup, should be pruned to MAX_BACKUPS_RETAINED (5)
    remaining = [d.name for d in backups_dir.iterdir() if d.is_dir()]
    assert len(remaining) == 5
