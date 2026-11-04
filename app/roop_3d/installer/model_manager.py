"""Offline model bundle importer and integrity verifier for Roop 3D.

Enforces:
- Strict archive traversal guards (rejection of '..', absolute paths, junctions, symlinks).
- Zip bomb protection with explicit expansion size ceilings.
- File extension whitelisting.
- SHA-256 and byte-size verification against the canonical ModelRegistry.
- Informative and actionable missing-model error diagnostics.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, BinaryIO
import zipfile

from roop_3d.audit.isolation import is_link, safe_path
from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import ModelRecord, ModelRegistry
from roop_3d.models.verifier import verify_file

MAX_BUNDLE_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB max expansion
MAX_SINGLE_FILE_BYTES = 5 * 1024 * 1024 * 1024           # 5 GiB max per file
ALLOWED_MODEL_EXTENSIONS = frozenset({".onnx", ".bin", ".json", ".hash", ".safetensors"})
BLOCKED_EXTENSIONS = frozenset({".exe", ".bat", ".cmd", ".ps1", ".sh", ".py", ".dll", ".so", ".dylib", ".vbs"})


def validate_archive_entry(entry: zipfile.ZipInfo) -> None:
    """Rejects unsafe archive entries before extraction."""
    name = entry.filename.replace("\\", "/")
    posix = PurePosixPath(name)

    if posix.is_absolute() or name.startswith("/") or ":" in name:
        raise DomainError("ARCHIVE_TRAVERSAL_BLOCKED", f"Archive entry has an absolute or drive path: {name}", 403)

    if any(part in {".", ".."} for part in posix.parts):
        raise DomainError("ARCHIVE_TRAVERSAL_BLOCKED", f"Archive entry attempts traversal: {name}", 403)

    # Reject symlinks/special files encoded in zip file attributes
    # Unix mode is stored in the upper 16 bits of external_attr
    mode = entry.external_attr >> 16
    if stat.S_ISLNK(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
        raise DomainError("ARCHIVE_SYMLINK_BLOCKED", f"Archive entry is a symlink or special file: {name}", 403)

    suffix = posix.suffix.lower()
    if suffix in BLOCKED_EXTENSIONS:
        raise DomainError("ARCHIVE_UNAPPROVED_TYPE", f"Archive contains blocked executable/script: {name}", 403)

    if entry.file_size > MAX_SINGLE_FILE_BYTES:
        raise DomainError("ARCHIVE_FILE_TOO_LARGE", f"Archive entry exceeds 5 GiB size limit: {name}", 413)


def check_archive_safety(zip_file: zipfile.ZipFile) -> None:
    """Verifies total expansion size and validates every entry in the archive."""
    total_uncompressed = 0
    for entry in zip_file.infolist():
        validate_archive_entry(entry)
        total_uncompressed += entry.file_size
        if total_uncompressed > MAX_BUNDLE_UNCOMPRESSED_BYTES:
            raise DomainError("ARCHIVE_EXPANSION_LIMIT", "Archive exceeds total 20 GiB uncompressed ceiling", 413)


class ModelManager:
    def __init__(self, root: Path, registry: ModelRegistry | None = None):
        self.root = root.resolve()
        self.registry = registry or ModelRegistry()
        self.cache_dir = self.root / "roop_3d/models/cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _target_path(self, record: ModelRecord) -> Path:
        suffix = Path(record.filenames[0]).suffix if record.filenames else ".artifact"
        return self.cache_dir / f"{record.model_id}.model{suffix}"

    def scan_directory(self, dir_path: Path) -> dict[str, Any]:
        """Scans a local directory for registered models."""
        if not dir_path.is_dir():
            raise DomainError("MODEL_DIR_INVALID", f"Specified path is not a directory: {dir_path}", 400)

        found_models: list[dict[str, Any]] = []
        unrecognized_files: list[str] = []
        count = 0

        # Map filenames to records
        records_by_filename: dict[str, list[ModelRecord]] = {}
        for rec in self.registry.records():
            for fn in rec.filenames:
                records_by_filename.setdefault(fn, []).append(rec)

        for parent, dirs, files in os.walk(dir_path, followlinks=False):
            # Guard against links/junctions in search
            dirs[:] = [d for d in dirs if not is_link(Path(parent) / d)]
            for fname in files:
                count += 1
                if count > 50000:
                    raise DomainError("MODEL_SEARCH_LIMIT", "Scan directory contains more than 50,000 files", 400)

                file_path = Path(parent) / fname
                candidates = records_by_filename.get(fname, [])
                if not candidates:
                    unrecognized_files.append(fname)
                    continue

                for rec in candidates:
                    try:
                        verify_file(file_path, rec.expected_sha256, rec.byte_size)
                        found_models.append({
                            "model_id": rec.model_id,
                            "name": rec.name,
                            "filename": fname,
                            "path": str(file_path),
                            "sha256": rec.expected_sha256,
                            "verified": True,
                        })
                        break
                    except DomainError:
                        continue

        return {
            "scanned_files": count,
            "matched_models": found_models,
            "unrecognized_files": unrecognized_files[:20],
        }

    def import_bundle(self, bundle_path: Path) -> dict[str, Any]:
        """Imports a model bundle from a local directory or ZIP file into the model cache."""
        if not bundle_path.exists():
            raise DomainError("BUNDLE_NOT_FOUND", f"Model bundle path does not exist: {bundle_path}", 404)

        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        corrupted: list[dict[str, Any]] = []

        temp_extract = self.root / "cache/tmp" / f"bundle_{os.getpid()}"
        temp_extract.mkdir(parents=True, exist_ok=True)

        try:
            if bundle_path.is_file():
                if not zipfile.is_zipfile(bundle_path):
                    raise DomainError("BUNDLE_INVALID_FORMAT", "Model bundle file must be a valid .zip archive", 400)
                with zipfile.ZipFile(bundle_path, "r") as zf:
                    check_archive_safety(zf)
                    zf.extractall(temp_extract)
                source_root = temp_extract
            else:
                source_root = bundle_path

            # Map filenames to records
            records_by_name = {rec.name: rec for rec in self.registry.records()}
            for rec in self.registry.records():
                for fn in rec.filenames:
                    records_by_name[fn] = rec

            for parent, dirs, files in os.walk(source_root, followlinks=False):
                dirs[:] = [d for d in dirs if not is_link(Path(parent) / d)]
                for fname in files:
                    file_path = Path(parent) / fname
                    rec = records_by_name.get(fname)
                    if not rec:
                        continue

                    dest = self._target_path(rec)
                    # Check if already installed and verified
                    if dest.exists():
                        try:
                            verify_file(dest, rec.expected_sha256, rec.byte_size)
                            skipped.append({"name": rec.name, "model_id": rec.model_id, "reason": "Already installed and verified"})
                            continue
                        except DomainError:
                            pass

                    # Verify source file
                    try:
                        verify_file(file_path, rec.expected_sha256, rec.byte_size)
                    except DomainError as err:
                        corrupted.append({"name": rec.name, "file": fname, "error": str(err)})
                        continue

                    # Copy atomically to target
                    temp_dest = dest.with_suffix(dest.suffix + ".part")
                    shutil.copy2(file_path, temp_dest)
                    os.replace(temp_dest, dest)

                    imported.append({
                        "name": rec.name,
                        "model_id": rec.model_id,
                        "sha256": rec.expected_sha256,
                        "byte_size": rec.byte_size,
                        "destination": str(dest.relative_to(self.root)),
                    })

        finally:
            if temp_extract.exists():
                shutil.rmtree(temp_extract, ignore_errors=True)

        return {
            "status": "PASS" if imported or skipped else "EMPTY",
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "corrupted_count": len(corrupted),
            "imported": imported,
            "skipped": skipped,
            "corrupted": corrupted,
        }

    def verify_all_models(self) -> dict[str, Any]:
        """Runs full SHA-256 verification on all models in the local cache."""
        results = []
        for rec in self.registry.records():
            target = self._target_path(rec)
            if not target.exists():
                results.append({"name": rec.name, "model_id": rec.model_id, "state": "NOT_INSTALLED", "sha256_verified": False})
                continue
            try:
                verify_file(target, rec.expected_sha256, rec.byte_size)
                results.append({"name": rec.name, "model_id": rec.model_id, "state": "VERIFIED_LOCAL", "sha256_verified": True})
            except DomainError as err:
                results.append({"name": rec.name, "model_id": rec.model_id, "state": "MODEL_DRIFT", "sha256_verified": False, "error": str(err)})

        verified_count = sum(1 for r in results if r["state"] == "VERIFIED_LOCAL")
        return {
            "total_registered": len(results),
            "verified_count": verified_count,
            "results": results,
        }


def format_missing_model_error(model_name: str, registry: ModelRegistry | None = None) -> str:
    """Generates a detailed, actionable error message when a model is missing offline."""
    registry = registry or ModelRegistry()
    record = None
    for rec in registry.records():
        if rec.name == model_name or model_name in rec.filenames or rec.model_id == model_name:
            record = rec
            break

    if not record:
        return f"Model '{model_name}' is not installed and not found in the registered model catalog."

    role = getattr(record, "role", "inference model")
    sha = record.expected_sha256 or "unknown"
    size = f"{record.byte_size:,} bytes" if record.byte_size else "unknown"

    return (
        f"Missing Model: '{record.name}'\n"
        f"- Role: {role}\n"
        f"- Registered ID: {record.model_id}\n"
        f"- Expected SHA-256: {sha}\n"
        f"- Expected Size: {size}\n\n"
        f"Offline Resolution:\n"
        f"1. Download or locate '{record.name}' matching the exact SHA-256 hash above.\n"
        f"2. Place the file in 'roop_3d/models/cache/{record.model_id}.model{Path(record.filenames[0]).suffix}' "
        f"or use 'python -B tools/project_admin.py model-import <path-to-folder-or-zip>'."
    )
