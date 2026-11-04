"""Unit tests for Stage 16 model manager, bundle import, archive traversal security, and hash verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.installer.model_manager import (
    ModelManager,
    check_archive_safety,
    format_missing_model_error,
    validate_archive_entry,
)
from roop_3d.models.registry import ModelRecord, ModelRegistry


@pytest.fixture
def test_registry():
    registry = ModelRegistry()
    base_record = registry.records()[0]
    test_content = b"TEST_ONNX_MODEL_CONTENT_12345"
    test_sha = hashlib.sha256(test_content).hexdigest()
    test_record = base_record.model_copy(update={
        "model_id": "test_model_001",
        "name": "test_model.onnx",
        "expected_sha256": test_sha,
        "byte_size": len(test_content),
        "filenames": ("test_model.onnx",),
        "weight_filenames": ("test_model.onnx",),
    })
    from roop_3d.models.registry import Manifest
    test_manifest = Manifest(
        evidence_scope=registry.manifest.evidence_scope,
        source_documents=registry.manifest.source_documents,
        models=(*registry.manifest.models, test_record),
    )
    custom_registry = ModelRegistry(manifest=test_manifest)
    return custom_registry, test_content, test_record


def test_archive_traversal_blocked(tmp_path: Path) -> None:
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../etc/passwd.onnx", b"bad")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(DomainError) as exc_info:
            check_archive_safety(zf)
        assert exc_info.value.code == "ARCHIVE_TRAVERSAL_BLOCKED"


def test_archive_absolute_path_blocked(tmp_path: Path) -> None:
    zip_path = tmp_path / "absolute.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("/root/evil.onnx", b"bad")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(DomainError) as exc_info:
            check_archive_safety(zf)
        assert exc_info.value.code == "ARCHIVE_TRAVERSAL_BLOCKED"


def test_archive_executable_blocked(tmp_path: Path) -> None:
    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("payload.exe", b"executable_data")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(DomainError) as exc_info:
            check_archive_safety(zf)
        assert exc_info.value.code == "ARCHIVE_UNAPPROVED_TYPE"


def test_archive_expansion_limit(tmp_path: Path) -> None:
    zip_path = tmp_path / "bomb.zip"
    info = zipfile.ZipInfo("large.onnx")
    info.file_size = 25 * 1024 * 1024 * 1024  # 25 GiB claim

    with pytest.raises(DomainError) as exc_info:
        validate_archive_entry(info)
    assert exc_info.value.code == "ARCHIVE_FILE_TOO_LARGE"


def test_model_import_from_directory(tmp_path: Path, test_registry) -> None:
    registry, content, record = test_registry
    manager = ModelManager(tmp_path, registry)

    source_dir = tmp_path / "source_models"
    source_dir.mkdir()
    (source_dir / "test_model.onnx").write_bytes(content)

    res = manager.import_bundle(source_dir)
    assert res["status"] == "PASS"
    assert res["imported_count"] == 1
    assert res["corrupted_count"] == 0

    # Verify model is now installed and verified
    verify_res = manager.verify_all_models()
    matched = next((r for r in verify_res["results"] if r["model_id"] == record.model_id), None)
    assert matched is not None
    assert matched["state"] == "VERIFIED_LOCAL"


def test_model_import_from_zip(tmp_path: Path, test_registry) -> None:
    registry, content, record = test_registry
    manager = ModelManager(tmp_path, registry)

    zip_file = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("test_model.onnx", content)

    res = manager.import_bundle(zip_file)
    assert res["status"] == "PASS"
    assert res["imported_count"] == 1
    assert res["corrupted_count"] == 0


def test_model_import_corrupted_file_rejected(tmp_path: Path, test_registry) -> None:
    registry, _, record = test_registry
    manager = ModelManager(tmp_path, registry)

    source_dir = tmp_path / "bad_models"
    source_dir.mkdir()
    (source_dir / "test_model.onnx").write_bytes(b"WRONG_HASH_BYTES")

    res = manager.import_bundle(source_dir)
    assert res["imported_count"] == 0
    assert res["corrupted_count"] == 1


def test_missing_model_error_message(test_registry) -> None:
    registry, _, record = test_registry
    msg = format_missing_model_error("test_model.onnx", registry)
    assert "Missing Model: 'test_model.onnx'" in msg
    assert record.expected_sha256 in msg
    assert "Offline Resolution" in msg
