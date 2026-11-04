"""Complete Security, Privacy, and Synthetic Media Protection Test Suite for Stage 13."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import stat
import zipfile
import pytest

from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.faceset.schemas import ConsentMetadata
from roop_3d.faceset.validator import (
    MAX_COMPRESSION_RATIO,
    MAX_ENTRIES,
    MAX_ENTRY_UNCOMPRESSED_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    ArchiveValidator,
)
from roop_3d.jobs.project_store import ProjectDraft, ProjectManifest, ProjectStore
from roop_3d.jobs.schema import JobRequest
from roop_3d.media.provenance import (
    capabilities,
    recipe_claim,
    verify_manifest,
)
from roop_3d.models.governance import ModelCompatibilityScanner
from roop_3d.models.registry import ModelRecord
from roop_3d.security.artifact_cleaner import SafeArtifactCleaner
from roop_3d.security.content_validator import (
    validate_faceset_archive_bytes,
    validate_image_header,
    validate_safe_filename,
    validate_safe_path,
    validate_video_container,
)
from roop_3d.security.log_sanitizer import (
    SecurityRedactionFilter,
    sanitize_log_message,
)
from roop_3d.security.privacy import (
    ConsentRecord,
    ConsentVerifier,
    TargetFootagePermission,
)
from roop_3d.security.synthetic_media import (
    AuditRecord,
    IdentityHonestyPolicy,
    SafetyGuard,
    SyntheticMediaDisclosure,
    build_provenance_manifest,
)
from roop_3d.server.app import create_app
from roop_3d.server.security import is_loopback_peer


# ---------------------------------------------------------------------------
# Requirement 1: Bind processing services to loopback by default
# ---------------------------------------------------------------------------
def test_default_service_binds_to_loopback():
    settings = Settings()
    assert settings.host in {"127.0.0.1", "localhost"}
    assert settings.port == 8030
    assert settings.remote_processing_enabled is False


def test_is_loopback_peer_validation():
    assert is_loopback_peer(("127.0.0.1", 12345)) is True
    assert is_loopback_peer(("::1", 12345)) is True
    assert is_loopback_peer(("192.168.1.100", 12345)) is False
    assert is_loopback_peer(("8.8.8.8", 12345)) is False
    assert is_loopback_peer(None) is False


# ---------------------------------------------------------------------------
# Requirement 2: Prevent unintended external media transmission
# ---------------------------------------------------------------------------
def test_external_egress_policy_blocks_non_loopback():
    with pytest.raises(DomainError) as exc_info:
        ConsentVerifier.verify_egress_policy("https://api.external-cloud.com")
    assert exc_info.value.code == "REMOTE_EGRESS_BLOCKED"

    # Loopback is permitted
    ConsentVerifier.verify_egress_policy("http://127.0.0.1:8188")
    ConsentVerifier.verify_egress_policy("http://localhost:8188")


def test_hidden_transfer_audit():
    with pytest.raises(DomainError) as exc:
        ConsentVerifier.audit_no_hidden_transfer({
            "remote_processing_enabled": False,
            "destination_host": "api.remote-transfer.com",
        })
    assert exc.value.code == "HIDDEN_TRANSFER_DETECTED"


# ---------------------------------------------------------------------------
# Requirement 3: Validate file types and actual file contents (magic bytes)
# ---------------------------------------------------------------------------
def test_validate_image_header_magic_bytes():
    # PNG valid
    assert validate_image_header(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...") == "png"
    # JPEG valid
    assert validate_image_header(b"\xff\xd8\xff\xe0\x00\x10JFIF...") == "jpeg"
    # WebP valid
    assert validate_image_header(b"RIFF\x20\x00\x00\x00WEBPVP8 ...") == "webp"

    # Disallowed binary disguises
    with pytest.raises(DomainError) as exc:
        validate_image_header(b"MZ\x90\x00\x03\x00\x00\x00")
    assert exc.value.code == "DISALLOWED_PAYLOAD"

    with pytest.raises(DomainError) as exc:
        validate_image_header(b"#!/bin/bash\nrm -rf /")
    assert exc.value.code == "DISALLOWED_PAYLOAD"

    # Mismatch with expected format
    with pytest.raises(DomainError) as exc:
        validate_image_header(b"\x89PNG\r\n\x1a\n", expected_format="jpeg")
    assert exc.value.code == "IMAGE_FORMAT_MISMATCH"


def test_validate_video_container_magic_bytes():
    # MP4 with ftyp
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"
    assert validate_video_container(mp4_bytes, "sample.mp4") == "mp4"

    # MKV with EBML header
    mkv_bytes = b"\x1a\x45\xdf\xa3\x93\x42\x82\x88matroska"
    assert validate_video_container(mkv_bytes, "sample.mkv") == "mkv"

    # AVI with RIFF AVI
    avi_bytes = b"RIFF\x40\x00\x00\x00AVI LIST\x00\x00\x00\x00"
    assert validate_video_container(avi_bytes, "sample.avi") == "avi"

    # TS with sync byte 0x47
    ts_bytes = b"\x47\x40\x00\x10" + b"\x00" * 184
    assert validate_video_container(ts_bytes, "sample.ts") == "ts"

    # Malicious executable disguised as mp4
    fake_mp4 = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 100
    with pytest.raises(DomainError) as exc:
        validate_video_container(fake_mp4, "malicious.mp4")
    assert exc.value.code == "DISALLOWED_PAYLOAD"


def test_faceset_archive_magic_bytes():
    # Valid ZIP header
    validate_faceset_archive_bytes(b"PK\x03\x04\x14\x00\x00\x00")

    # Invalid header
    with pytest.raises(DomainError) as exc:
        validate_faceset_archive_bytes(b"NOT_A_ZIP_HEADER")
    assert exc.value.code == "FACESET_ARCHIVE_INVALID"


# ---------------------------------------------------------------------------
# Requirement 4 & 7: Enforce upload & decompression limits, safe archive handling
# ---------------------------------------------------------------------------
def test_archive_validator_entry_content_magic_check():
    # ZIP with fake PNG containing python script
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("0.png", b"import os; os.system('calc')")
    bio.seek(0)
    with zipfile.ZipFile(bio, "r") as zf:
        with pytest.raises(DomainError) as exc:
            ArchiveValidator.validate_zip(zf)
        assert exc.value.code == "FACESET_SECURITY_VIOLATION"


def test_archive_validator_zip_bomb_defense():
    uncompressed = 1024 * 1024  # 1 MiB zeros
    data = b"\x00" * uncompressed
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.png", data)
    bio.seek(0)
    with zipfile.ZipFile(bio, "r") as zf:
        with pytest.raises(DomainError) as exc:
            ArchiveValidator.validate_zip(zf)
        assert exc.value.code == "FACESET_SECURITY_VIOLATION"
        assert "zip bomb" in exc.value.message.lower()


# ---------------------------------------------------------------------------
# Requirement 5: Prevent directory traversal and symlink attacks
# ---------------------------------------------------------------------------
def test_validate_safe_filename():
    assert validate_safe_filename("valid_image_01.png") == "valid_image_01.png"

    with pytest.raises(DomainError):
        validate_safe_filename("../escape.png")

    with pytest.raises(DomainError):
        validate_safe_filename("CON.png")

    with pytest.raises(DomainError):
        validate_safe_filename("image\x00.png")


def test_validate_safe_path_traversal(tmp_path: Path):
    root = tmp_path / "app_root"
    root.mkdir()
    safe_sub = root / "allowed"
    safe_sub.mkdir()
    target = safe_sub / "file.txt"
    target.write_text("ok")

    assert validate_safe_path(target, root) == target.resolve()

    # Traversal escape
    outside = tmp_path / "outside.txt"
    outside.write_text("bad")
    with pytest.raises(DomainError) as exc:
        validate_safe_path(outside, root)
    assert exc.value.code == "PATH_TRAVERSAL_DETECTED"


# ---------------------------------------------------------------------------
# Requirement 6: Protect project and output directories
# ---------------------------------------------------------------------------
def test_project_store_relative_path_enforces_hex():
    assert ProjectStore.relative("a" * 32) == f"roop_3d/.runtime/projects/{'a'*32}.r3dp"
    with pytest.raises(DomainError) as exc:
        ProjectStore.relative("../traversal")
    assert exc.value.code == "PROJECT_ID_INVALID"


# ---------------------------------------------------------------------------
# Requirement 8: Protect against malicious model files (pickle rejection)
# ---------------------------------------------------------------------------
def test_pickle_serialization_rejected_for_inference():
    from roop_3d.models.registry import ModelRegistry
    scanner = ModelCompatibilityScanner()
    record = ModelRegistry().records()[0].model_copy(update={"serialization_format": "pickle"})
    result = scanner.scan(record)
    assert any("Pickle-based serialization is not admitted" in issue for issue in result["issues"])
    assert result["lifecycle_state"] in {"INCOMPATIBLE", "FAILED"}


# ---------------------------------------------------------------------------
# Requirement 9: Sanitize errors and application logs
# ---------------------------------------------------------------------------
def test_log_sanitizer_redacts_tokens_and_biometrics():
    raw_log = "User authenticated with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret and roop3d_token=abc123secret"
    sanitized = sanitize_log_message(raw_log)
    assert "Bearer [REDACTED]" in sanitized
    assert "secret=[REDACTED]" in sanitized
    assert "abc123secret" not in sanitized

    # Biometric array redaction
    embedding_str = str([0.12345] * 512)
    embedding_log = f"Extracted vector: {embedding_str}"
    sanitized_emb = sanitize_log_message(embedding_log)
    assert "[BIOMETRIC_EMBEDDING_REDACTED]" in sanitized_emb
    assert "0.12345" not in sanitized_emb


def test_security_redaction_filter():
    logger = logging.getLogger("test_security_redaction")
    filt = SecurityRedactionFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Error with token=super_secret_token_123",
        args=(),
        exc_info=None,
    )
    assert filt.filter(record) is True
    assert "secret=[REDACTED]" in record.msg
    assert "super_secret_token_123" not in record.msg


# ---------------------------------------------------------------------------
# Requirement 10: Keep credentials outside project manifests
# ---------------------------------------------------------------------------
def test_project_manifest_rejects_credential_injection(tmp_path: Path):
    project_path = tmp_path / "project_with_creds.r3dp"
    project_with_creds = {
        "schema_version": 1,
        "project_id": "a" * 32,
        "name": "Malicious Project",
        "saved_at": "2026-11-04T00:00:00Z",
        "api_key": "secret_api_key_12345",
        "request": {},
    }
    project_path.write_text(json.dumps(project_with_creds))

    store = ProjectStore(tmp_path, media=None, catalog=None, jobs=None)
    rel_dir = tmp_path / "roop_3d/.runtime/projects"
    rel_dir.mkdir(parents=True)
    target = rel_dir / f"{'a'*32}.r3dp"
    target.write_text(json.dumps(project_with_creds))

    with pytest.raises(DomainError) as exc:
        store.load("a" * 32)
    assert exc.value.code == "PROJECT_CREDENTIALS_REJECTED"


# ---------------------------------------------------------------------------
# Requirement 11: Validate API origins and WebSocket connections
# ---------------------------------------------------------------------------
def test_api_origin_validation(tmp_path: Path):
    app = create_app(Settings(port=8030), repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app)

    # Valid loopback origin
    res = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:8030", "Host": "127.0.0.1:8030"})
    assert res.status_code == 200

    # Malicious foreign origin
    bad_res = client.get("/api/v1/health", headers={"Origin": "http://evil-attacker.com", "Host": "127.0.0.1:8030"})
    assert bad_res.status_code == 403
    assert bad_res.json()["error"]["code"] == "ORIGIN_DENIED"


# ---------------------------------------------------------------------------
# Requirement 14: Safely delete temporary media and biometric artifacts
# ---------------------------------------------------------------------------
def test_safe_artifact_cleaner(tmp_path: Path):
    cleaner = SafeArtifactCleaner(tmp_path)

    # Create dummy biometric assets
    faceset_id = "test_faceset_123"
    fsz_dir = tmp_path / f"roop_3d/.runtime/identity_assets/{faceset_id}"
    fsz_dir.mkdir(parents=True)
    (fsz_dir / "mesh.obj").write_text("v 0 0 0")
    faceset_meta_dir = tmp_path / "roop_3d/.runtime/facesets"
    faceset_meta_dir.mkdir(parents=True, exist_ok=True)
    (faceset_meta_dir / f"{faceset_id}.json").write_text("{}")
    coverage_meta_dir = tmp_path / "roop_3d/.runtime/coverage"
    coverage_meta_dir.mkdir(parents=True, exist_ok=True)
    (coverage_meta_dir / f"{faceset_id}.json").write_text("{}")

    result = cleaner.purge_faceset_biometrics(faceset_id)
    assert "identity_assets_3d" in result["purged"]
    assert not fsz_dir.exists()
    assert not (tmp_path / f"roop_3d/.runtime/facesets/{faceset_id}.json").exists()


# ---------------------------------------------------------------------------
# Privacy: Biometric permission and target footage consent
# ---------------------------------------------------------------------------
def test_privacy_consent_verifier():
    # Consent granted
    valid_consent = ConsentRecord(
        subject_name="Alice",
        consent_granted=True,
        permitted_operations=("swap", "reenactment"),
    )
    ConsentVerifier.verify_source_consent(valid_consent, operation="swap")

    # Consent not granted
    unconsented = ConsentRecord(
        subject_name="Bob",
        consent_granted=False,
    )
    with pytest.raises(DomainError) as exc:
        ConsentVerifier.verify_source_consent(unconsented, operation="swap")
    assert exc.value.code == "IDENTITY_CONSENT_DENIED"

    # Unpermitted operation
    with pytest.raises(DomainError) as exc:
        ConsentVerifier.verify_source_consent(valid_consent, operation="3d_reconstruction")
    assert exc.value.code == "OPERATION_UNAUTHORIZED"


# ---------------------------------------------------------------------------
# Synthetic Media: Honest integrity vs identity proof, disclosure & C2PA
# ---------------------------------------------------------------------------
def test_identity_honesty_rejects_sha256_as_identity_proof():
    # Truthful integrity statement
    statement = IdentityHonestyPolicy.honest_integrity_statement("a" * 64)
    assert statement["verification_type"] == "artifact_byte_integrity"
    assert "does not establish legal or biometric person identity" in statement["limitation"]

    # Dishonest claims asserting hash proves human identity
    with pytest.raises(DomainError) as exc:
        IdentityHonestyPolicy.audit_claim({"claim": "cryptographic proof of person identity via SHA-256"})
    assert exc.value.code == "DISHONEST_IDENTITY_CLAIM"

    with pytest.raises(DomainError) as exc:
        IdentityHonestyPolicy.audit_claim({"proves_person_identity": True})
    assert exc.value.code == "DISHONEST_IDENTITY_CLAIM"


def test_synthetic_media_disclosure_and_manifest():
    manifest = build_provenance_manifest(
        input_sha256="0" * 64,
        output_sha256="1" * 64,
        output_path="outputs/media/test.mp4",
        job_id="j" * 32,
        engine_id="reference_onnx",
        hardware_profile="cpu",
        model_digests={"inswapper_128.onnx": "2" * 64},
        render_settings={"swap_mode": "full"},
    )
    assert manifest["disclosure"]["synthetic_media"] is True
    assert manifest["disclosure"]["processor"] == "Roop 3D"
    assert manifest["integrity_notice"]["verification_type"] == "artifact_byte_integrity"


def test_safety_guard_blocks_ncii_and_deceptive_impersonation():
    # Prohibited NCII term
    with pytest.raises(DomainError) as exc:
        SafetyGuard.inspect_request(notes="Generating deepnude content")
    assert exc.value.code == "SAFETY_POLICY_VIOLATION"

    # Non-consensual leak intent
    with pytest.raises(DomainError) as exc:
        SafetyGuard.inspect_request(face_labels=("target_leak", "revenge_porn"))
    assert exc.value.code == "SAFETY_POLICY_VIOLATION"

    # Clean benign request
    SafetyGuard.inspect_request(notes="Creative visual reenactment for animation project")


def test_security_policy_endpoint(tmp_path: Path):
    app = create_app(Settings(port=8030), repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app)
    res = client.get("/api/v1/security/policy", headers={"Origin": "http://127.0.0.1:8030", "Host": "127.0.0.1:8030"})
    assert res.status_code == 200
    data = res.json()
    assert data["loopback_bound"] is True
    assert data["remote_processing_enabled"] is False
    assert data["synthetic_media_notice_required"] is True
    assert data["log_sanitization"] is True


def test_purge_artifacts_endpoint(tmp_path: Path):
    app = create_app(Settings(port=8030), repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app)
    res = client.post(
        "/api/v1/security/purge-artifacts",
        json={"upload_ids": ["a" * 32], "purge_stale_caches": True},
        headers={"Origin": "http://127.0.0.1:8030", "Host": "127.0.0.1:8030"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "purged"


def test_job_store_rejects_ncii_safety_violation(tmp_path: Path):
    from roop_3d.jobs.store import JobStore
    from roop_3d.media.schema import TrackToFaceSetMap
    store = JobStore(tmp_path)
    req = JobRequest(
        target_path="roop_3d/.runtime/media/target.mp4",
        inspection_id="1" * 32,
        face_mappings=(TrackToFaceSetMap(track_id="track_0000", faceset_id="deepnude_unconsented"),),
    )
    with pytest.raises(DomainError) as exc:
        store.create(req, "idempotency_key_123")
    assert exc.value.code == "SAFETY_POLICY_VIOLATION"


def test_checkpoint_restoration_verification(tmp_path: Path):
    from roop_3d.engine.pipeline_runner import PipelineCheckpoint, read_checkpoint_npz, checkpoint_npz_bytes
    cp = PipelineCheckpoint(
        job_id="a" * 32,
        input_sha256="1" * 64,
        request_sha256="2" * 64,
        settings_sha256="3" * 64,
        completed_frame=10,
        total_frames=100,
        pose_cache={"view_0": [0.1, 0.2, 0.3]},
    )
    data = checkpoint_npz_bytes(cp)
    loaded = read_checkpoint_npz(data)
    assert loaded.job_id == cp.job_id
    assert loaded.completed_frame == 10

    tampered = data.replace(b"roop3d.checkpoint.npz/1", b"invalid.format.npz/1")
    with pytest.raises(ValueError):
        read_checkpoint_npz(tampered)
