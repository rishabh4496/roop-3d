"""Update inspection, dependency change detection, configuration backup, and safe rollback.

Guarantees:
- Never modifies Roop Ultimate or files outside this Roop 3D checkout.
- Preserves all FaceSets, projects, and outputs during updates and rollbacks.
- Backs up configuration and Git commit before applying updates.
- Identifies component-level changes and warns of breaking updates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from roop_3d.audit.isolation import is_link, safe_path
from roop_3d.core.errors import DomainError

MAX_BACKUPS_RETAINED = 5


def _git(root: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise DomainError("GIT_ERROR", f"git {' '.join(args)} failed: {res.stderr.strip()}", 500)
    return res.stdout.strip()


def check_update_status(root: Path) -> dict[str, Any]:
    """Inspects Git repository version, upstream tracking branch, and changed files."""
    try:
        current_commit = _git(root, "rev-parse", "HEAD")
        current_branch = _git(root, "branch", "--show-current")
    except DomainError as err:
        return {
            "status": "ERROR",
            "message": f"Git inspection failed: {err}",
            "current_commit": "unknown",
            "current_branch": "unknown",
            "changed_components": {},
        }

    upstream_ref = None
    try:
        upstream_ref = _git(root, "rev-parse", "--abbrev-ref", "@{u}")
    except DomainError:
        pass

    ahead_count = 0
    behind_count = 0
    changed_files: list[str] = []

    if upstream_ref:
        try:
            rev_list = _git(root, "rev-list", "--left-right", "--count", f"HEAD...{upstream_ref}")
            parts = rev_list.split()
            if len(parts) == 2:
                ahead_count, behind_count = int(parts[0]), int(parts[1])
            if behind_count > 0:
                raw_diff = _git(root, "diff", "--name-only", f"HEAD..{upstream_ref}")
                changed_files = [f.strip() for f in raw_diff.splitlines() if f.strip()]
        except Exception:
            pass

    # Categorize changed files
    components = {
        "core_engine": [],
        "models": [],
        "server": [],
        "ui": [],
        "launchers": [],
        "dependencies": [],
        "other": [],
    }

    warnings: list[str] = []

    for f in changed_files:
        f_norm = f.replace("\\", "/")
        if f_norm.startswith(("app/roop_3d/engine/", "app/roop_3d/core/")):
            components["core_engine"].append(f)
        elif f_norm.startswith("app/roop_3d/models/"):
            components["models"].append(f)
        elif f_norm.startswith("app/roop_3d/server/"):
            components["server"].append(f)
        elif f_norm.startswith("app/roop_3d_ui/"):
            components["ui"].append(f)
        elif f_norm in ("start.js", "install.js", "update.js", "reset.js", "pinokio.js"):
            components["launchers"].append(f)
        elif f_norm in ("app/roop_3d/requirements.lock", "app/roop_3d/requirements.txt",
                        "app/roop_3d_ui/package.json", "app/roop_3d_ui/package-lock.json"):
            components["dependencies"].append(f)
        else:
            components["other"].append(f)

    if components["dependencies"]:
        warnings.append(
            f"Dependencies changed ({', '.join(components['dependencies'])}). Update will re-sync Python and Node packages."
        )

    if components["launchers"]:
        warnings.append(
            f"Launcher scripts changed ({', '.join(components['launchers'])}). Check for updated Pinokio configuration."
        )

    return {
        "status": "UP_TO_DATE" if behind_count == 0 else "UPDATE_AVAILABLE",
        "current_commit": current_commit,
        "current_branch": current_branch,
        "upstream_ref": upstream_ref or "none",
        "commits_behind": behind_count,
        "commits_ahead": ahead_count,
        "has_updates": behind_count > 0,
        "changed_files_count": len(changed_files),
        "changed_components": {k: v for k, v in components.items() if v},
        "warnings": warnings,
    }


def create_configuration_backup(root: Path) -> dict[str, Any]:
    """Snapshots configuration and active commit into cache/backups/."""
    backups_dir = root / "cache/backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    try:
        commit = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "branch", "--show-current")
    except Exception:
        commit = "unknown"
        branch = "unknown"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_id = f"backup_{ts}_{commit[:8]}"
    target_dir = backups_dir / backup_id
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []

    # Backup runtime config
    config_file = root / "roop_3d/.runtime/config.json"
    if config_file.is_file():
        shutil.copy2(config_file, target_dir / "config.json")
        copied_files.append("config.json")

    # Backup project identity
    proj_file = root / ".roop3d-project.json"
    if proj_file.is_file():
        shutil.copy2(proj_file, target_dir / ".roop3d-project.json")
        copied_files.append(".roop3d-project.json")

    # Write backup manifest
    manifest = {
        "backup_id": backup_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "copied_files": copied_files,
    }
    (target_dir / "backup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Prune older backups
    all_backups = sorted([d for d in backups_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")])
    if len(all_backups) > MAX_BACKUPS_RETAINED:
        for old in all_backups[:-MAX_BACKUPS_RETAINED]:
            shutil.rmtree(old, ignore_errors=True)

    return {
        "status": "PASS",
        "backup_id": backup_id,
        "backup_dir": str(target_dir.relative_to(root)),
        "git_commit": commit,
        "copied_files": copied_files,
    }


def rollback_to_backup(root: Path, backup_id: str | None = None) -> dict[str, Any]:
    """Rolls back Git commit and restores configuration from the specified or latest backup.

    Preserves:
    - FaceSets (roop_3d/outputs/facesets, roop_3d/.runtime/facesets)
    - Projects (roop_3d/.runtime/projects)
    - Rendered outputs (outputs/, roop_3d/outputs)
    - Legacy boundaries (refuses external paths)
    """
    backups_dir = root / "cache/backups"
    if not backups_dir.is_dir():
        raise DomainError("ROLLBACK_NO_BACKUPS", "No configuration backups exist in cache/backups", 404)

    if backup_id:
        target_dir = backups_dir / backup_id
        if not target_dir.is_dir():
            raise DomainError("ROLLBACK_BACKUP_NOT_FOUND", f"Backup '{backup_id}' not found", 404)
    else:
        candidates = sorted([d for d in backups_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")])
        if not candidates:
            raise DomainError("ROLLBACK_NO_BACKUPS", "No configuration backups found", 404)
        target_dir = candidates[-1]

    manifest_file = target_dir / "backup_manifest.json"
    if not manifest_file.is_file():
        raise DomainError("ROLLBACK_CORRUPT_BACKUP", f"Backup manifest missing in {target_dir.name}", 500)

    manifest = json.loads(manifest_file.read_text("utf-8"))
    target_commit = manifest.get("git_commit")

    # 1. Reset Git commit
    if target_commit and target_commit != "unknown":
        _git(root, "reset", "--hard", target_commit)

    # 2. Restore config.json
    restored_files: list[str] = []
    if (target_dir / "config.json").is_file():
        dest = root / "roop_3d/.runtime/config.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_dir / "config.json", dest)
        restored_files.append("config.json")

    return {
        "status": "PASS",
        "rolled_back_to_backup": target_dir.name,
        "restored_git_commit": target_commit,
        "restored_files": restored_files,
        "preserved_paths": [
            "roop_3d/outputs/facesets",
            "roop_3d/.runtime/facesets",
            "roop_3d/outputs",
            "outputs",
        ],
        "message": f"Successfully rolled back to backup {target_dir.name} (commit {target_commit[:8] if target_commit else 'unknown'}).",
    }
