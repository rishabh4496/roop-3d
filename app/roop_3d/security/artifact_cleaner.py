"""Safe deletion and secure lifecycle management of temporary media and biometric artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.security.content_validator import validate_safe_path


class SafeArtifactCleaner:
    """Provides validated deletion of temporary media, caches, and biometric assets."""

    def __init__(self, root: Path):
        self.root = root

    def purge_upload(self, upload_id: str) -> bool:
        """Safely delete partial upload and receipt."""
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise DomainError("UPLOAD_ID_INVALID", "Invalid upload identifier", 422)

        deleted = False
        for ext in (".part", ".json", ".lock"):
            rel = f"roop_3d/.runtime/uploads/{upload_id}{ext}"
            try:
                target = owned(self.root, rel)
                validate_safe_path(target, self.root, allow_symlinks=False)
                if target.is_file():
                    target.unlink()
                    deleted = True
            except (DomainError, OSError):
                pass
        return deleted

    def purge_job_workspace(self, job_id: str) -> bool:
        """Safely delete job work directory, checkpoints, frames, and pose caches."""
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise DomainError("JOB_ID_INVALID", "Invalid job identifier", 422)

        rel = f"roop_3d/.runtime/jobs/{job_id}"
        target_dir = owned(self.root, rel)
        if not target_dir.exists():
            return False

        validate_safe_path(target_dir, self.root, allow_symlinks=False)
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise DomainError("JOB_WORKDIR_INVALID", "Job workspace is not a valid owned directory", 409)

        # Bottom-up deletion with symlink detection
        for item in sorted(target_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if item.is_symlink():
                raise DomainError("SYMLINK_ATTACK_DETECTED", f"Symlink encountered inside job directory: {item}", 403)
            validate_safe_path(item, self.root, allow_symlinks=False)
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()

        target_dir.rmdir()
        return True

    def purge_faceset_biometrics(self, faceset_id: str) -> dict[str, Any]:
        """Safely wipe FaceSet manifests, coverage, identity assets (3D meshes/embeddings), and archives."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
            raise DomainError("FACESET_ID_INVALID", "Invalid FaceSet identifier", 422)

        purged_items: list[str] = []

        # 1. Manifest
        manifest = owned(self.root, f"roop_3d/.runtime/facesets/{faceset_id}.json")
        if manifest.is_file():
            manifest.unlink(missing_ok=True)
            purged_items.append("manifest")

        # 2. Coverage
        coverage = owned(self.root, f"roop_3d/.runtime/coverage/{faceset_id}.json")
        if coverage.is_file():
            coverage.unlink(missing_ok=True)
            purged_items.append("coverage")

        # 3. Archives
        for base in ("outputs", ".runtime"):
            archive = owned(self.root, f"roop_3d/{base}/facesets/{faceset_id}.fsz")
            if archive.is_file():
                archive.unlink(missing_ok=True)
                purged_items.append(f"archive_{base}")

        # 4. Identity assets (3D mesh, texture atlas, embeddings)
        assets_dir = owned(self.root, f"roop_3d/.runtime/identity_assets/{faceset_id}")
        if assets_dir.exists():
            validate_safe_path(assets_dir, self.root, allow_symlinks=False)
            for item in sorted(assets_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if item.is_symlink():
                    raise DomainError("SYMLINK_ATTACK_DETECTED", f"Symlink rejected in identity asset: {item}", 403)
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            assets_dir.rmdir()
            purged_items.append("identity_assets_3d")

        return {"faceset_id": faceset_id, "purged": purged_items}

    def purge_stale_temp_artifacts(
        self,
        *,
        max_upload_age_seconds: float = 3600 * 24,
        max_cache_age_seconds: float = 3600 * 48,
    ) -> dict[str, int]:
        """Clean up orphaned uploads, inspections, and cache files exceeding age limits."""
        now = time.time()
        counts = {"uploads": 0, "caches": 0}

        # Stale uploads
        uploads_dir = self.root / "roop_3d/.runtime/uploads"
        if uploads_dir.is_dir():
            for p in uploads_dir.iterdir():
                if p.is_file() and (now - p.stat().st_mtime) > max_upload_age_seconds:
                    p.unlink(missing_ok=True)
                    counts["uploads"] += 1

        # Stale runtime cache
        cache_dir = self.root / "roop_3d/.runtime/cache"
        if cache_dir.is_dir():
            for p in cache_dir.glob("*"):
                if p.is_file() and (now - p.stat().st_mtime) > max_cache_age_seconds:
                    p.unlink(missing_ok=True)
                    counts["caches"] += 1

        return counts
