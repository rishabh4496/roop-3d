from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from roop_3d.audit.isolation import AuditError
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import Manifest, ModelRecord, ModelRegistry, authorize, license_info
from roop_3d.models.verifier import verified_stream, verify_file


def artifact(data: bytes = b"local fixture weights", **changes) -> ModelRecord:
    return ModelRecord.model_validate({
        "model_id": "fixture", "name": "fixture.onnx", "role": "identity_swapper",
        "role_evidence": "Test fixture only", "filenames": ["fixture.onnx"],
        "expected_sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data),
        "input_output_contract": {"state": "UNKNOWN"}, "status": "verified_local",
        "evidence": "Synthetic fixture bytes; no inference claim", "license_notice": "Fixture terms",
        **changes})


def registry_for(record: ModelRecord) -> ModelRegistry:
    return ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=(record,)))


def test_manifest_preserves_all_observed_identities_without_inventing_rights():
    root = application_root()
    old = json.loads((root / "roop_3d/audit/verified_local_models.json").read_text())
    registry = ModelRegistry()
    local = [item for item in registry.records() if item.status == "verified_local"]
    assert len(local) == len(old["models"]) == 95
    assert {(m.expected_sha256, m.byte_size) for m in local} == {
        (m["sha256"], m["size_bytes"]) for m in old["models"]}
    for record in local:
        assert record.upstream_url is None and record.commercial_use == "unknown"
        assert record.license_id == "UNKNOWN"
        assert record.input_output_contract == next(m["graph"] for m in old["models"]
                                                   if m["sha256"] == record.expected_sha256)
    assert {"liveportrait_appearance", "liveportrait_motion", "liveportrait_warp",
            "liveportrait_stitch", "detector", "recognizer", "identity_swapper",
            "face_parser", "occlusion_mask", "face_restorer"} <= {m.role for m in local}
    assert registry.get("flame-2020").status == "blocked"
    assert registry.get("wav2lip").expected_sha256 is None


def test_verifier_detects_digest_size_and_permission_errors(tmp_path, monkeypatch):
    path = tmp_path / "fixture.onnx"
    data = b"local fixture weights"
    path.write_bytes(data)
    record = artifact()
    assert verify_file(path, record.expected_sha256, len(data)) == path
    with verified_stream(path, record.expected_sha256, len(data)) as stream:
        assert stream.mode == "rb" and stream.read() == data
        with pytest.raises(OSError):
            stream.write(b"no")
    for digest, size in [("0" * 64, len(data)), (record.expected_sha256, len(data) + 1)]:
        with pytest.raises(DomainError, match="Artifact"):
            verify_file(path, digest, size)
    original = Path.open
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("fixture denial")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(DomainError) as failure:
        verify_file(path, record.expected_sha256, len(data))
    assert failure.value.code == "MODEL_UNREADABLE"


def test_local_provider_returns_original_path_and_never_writes(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    path = source / "fixture.onnx"
    path.write_bytes(b"local fixture weights")
    before = path.stat()
    record = artifact(locator={"scope": "legacy", "path": "fixture.onnx"})
    provider = LocalWeightProvider(registry_for(record), scopes={"legacy": source})
    assert provider.resolve("fixture") == path
    with provider.open("fixture") as stream:
        assert stream.read() == b"local fixture weights"
    assert list(source.iterdir()) == [path]
    assert (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_mode) == (
        before.st_size, before.st_mtime_ns, before.st_mode)
    path.write_bytes(b"wrong fixture weight")
    with pytest.raises(DomainError) as failure:
        provider.resolve("fixture")
    assert failure.value.code == "MODEL_DRIFT"


def test_search_paths_match_by_name_and_digest_and_source_drift_is_fatal(tmp_path):
    base = tmp_path / "roop_3d/.runtime/weights"
    for child, data in [("a", b"unrelated weight"), ("b", b"local fixture weights")]:
        folder = base / child
        folder.mkdir(parents=True)
        (folder / "fixture.onnx").write_bytes(data)
    record = artifact()
    provider = LocalWeightProvider(registry_for(record), root=tmp_path,
                                   search_paths=lambda: (str(base),))
    assert provider.resolve(record) == base / "b/fixture.onnx"
    record = artifact(locator={"scope": "legacy", "path": "a/fixture.onnx"})
    provider = LocalWeightProvider(registry_for(record), root=tmp_path, scopes={"legacy": base},
                                   search_paths=lambda: (str(base),))
    with pytest.raises(DomainError):
        provider.resolve(record)


def test_missing_legacy_root_can_fall_back_to_configured_weights(tmp_path):
    base = tmp_path / "roop_3d/.runtime/weights"
    base.mkdir(parents=True)
    (base / "fixture.onnx").write_bytes(b"local fixture weights")
    record = artifact(locator={"scope": "legacy", "path": "fixture.onnx"})
    provider = LocalWeightProvider(registry_for(record), root=tmp_path,
                                   scopes={"legacy": tmp_path / "missing-source"},
                                   search_paths=lambda: (str(base),))
    assert provider.resolve(record) == base / "fixture.onnx"


@pytest.mark.parametrize("change", [
    {"model_id": "../escaped"}, {"model_id": "con"}, {"filenames": ["../bad.onnx"]},
    {"filenames": ["C:/bad.onnx"]}, {"filenames": ["bad:stream"]},
    {"filenames": ["NUL.onnx"]}, {"expected_sha256": "not-a-digest"},
    {"byte_size": 0}, {"expected_sha256": None}])
def test_registry_rejects_unsafe_identity(change):
    with pytest.raises((ValueError, AuditError)):
        artifact(**change)


def test_duplicate_ids_and_discovery_tampering_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        Manifest(evidence_scope="fixture", models=(artifact(), artifact()))
    folder = tmp_path / "roop_3d/models"
    folder.mkdir(parents=True)
    manifest = Manifest(evidence_scope="fixture", models=(artifact(),),
                        source_documents={"roop_3d/models/source.json": "0" * 64})
    (folder / "source.json").write_text("{}")
    (folder / "model_manifest.json").write_text(manifest.model_dump_json())
    with pytest.raises(DomainError) as failure:
        ModelRegistry(tmp_path)
    assert failure.value.code == "MODEL_REGISTRY_INVALID"


def test_license_gate_blocks_restrictions_without_converting_acknowledgement_to_rights():
    record = artifact(status="blocked", manual_acceptance_required=True)
    for action in ("resolve", "activate", "download"):
        with pytest.raises(DomainError) as failure:
            authorize(record, action=action, accepted_license=True)
        assert failure.value.code == "MODEL_BLOCKED"
    restricted = artifact(manual_acceptance_required=True, license_id="LicenseRef-Fixture")
    with pytest.raises(DomainError) as failure:
        authorize(restricted, action="activate")
    assert failure.value.code == "LICENSE_ACCEPTANCE_REQUIRED"
    authorize(restricted, action="activate", accepted_license=True)
    with pytest.raises(DomainError):
        authorize(restricted, action="activate", accepted_license=True, commercial=True)
    assert not license_info(artifact())["automatic_download_allowed"]
    assert license_info(artifact())["download_gate"] == "LICENSE_UNVERIFIED"


def test_provider_manual_acceptance_enforced_before_open(tmp_path):
    path = tmp_path / "fixture.onnx"
    path.write_bytes(b"local fixture weights")
    record = artifact(locator={"scope": "legacy", "path": path.name}, manual_acceptance_required=True)
    provider = LocalWeightProvider(registry_for(record), scopes={"legacy": tmp_path})
    with pytest.raises(DomainError):
        with provider.open(record):
            pytest.fail("License gate allowed activation")
    with provider.open(record, accepted_license=True) as stream:
        assert stream.read() == path.read_bytes()
