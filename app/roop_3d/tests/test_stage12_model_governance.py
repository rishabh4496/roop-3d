from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import httpx

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.downloader import Downloader
from roop_3d.models.downloads import ModelDownloadManager
from roop_3d.models.governance import ModelCompatibilityScanner
from roop_3d.models.registry import Manifest, ModelRecord, ModelRegistry
from roop_3d.tests.test_stage03_registry import artifact


def governed_record(data: bytes, *, model_id: str = "fixture_v1", **changes) -> ModelRecord:
    defaults = {
        "license_id": "Apache-2.0", "license_url": "https://example.org/LICENSE",
        "commercial_use": "allow", "redistribution_permission": "allow",
        "attribution_requirements": "Preserve upstream notice",
        "original_repository": "LOCAL_TEST_FIXTURE", "source_revision": "LOCAL_GENERATED_FIXTURE",
        "version_family": "fixture", "version": "1.0.0", "architecture": "FixtureNet",
        "required_base_model": "NONE", "hardware_requirements": {"state": "VERIFIED"},
        "precision_requirements": ["FP32"],
        "dependency_compatibility": {"state": "COMPATIBLE", "requirements": []},
        "security_status": "VERIFIED", "serialization_format": "ONNX",
    }
    defaults.update(changes)
    locator = defaults.pop("locator", {"scope": "fixture", "path": f"{model_id}.onnx"})
    return artifact(data, model_id=model_id, locator=locator, **defaults)


def test_all_canonical_and_supplemental_records_materialize_governance_fields():
    registry = ModelRegistry()
    required = {"model_id", "name", "version", "original_repository", "source_revision",
                "architecture", "required_base_model", "weight_filenames", "expected_sha256",
                "license_id", "license_url", "commercial_use", "redistribution_permission",
                "attribution_requirements", "base_model_license_id", "base_model_license_url",
                "base_model_commercial_use", "base_model_redistribution_permission",
                "base_model_attribution_requirements", "hardware_requirements", "precision_requirements",
                "dependency_compatibility", "security_status"}
    for record in registry.records():
        materialized = record.model_dump()
        assert required <= materialized.keys()
        assert materialized["weight_filenames"] == materialized["filenames"]
        # A revision can be a pinned commit, an upstream release tag, or the explicit
        # local acknowledgement marker; a local acknowledgement is not a commit SHA.
        assert materialized["source_revision"] == record.source_revision
        assert record.source_revision in {"UNKNOWN", "LOCAL-UNVERIFIED", "buffalo_l"} or (
            len(record.source_revision) == 40 and all(c in "0123456789abcdefABCDEF" for c in record.source_revision))
        if record.source_revision == "LOCAL-UNVERIFIED":
            assert record.commercial_use == "deny" and record.redistribution_permission == "deny"


def test_scanner_requires_integrity_compatibility_license_and_security_evidence():
    data = b"fixture"
    scanner = ModelCompatibilityScanner()
    good = governed_record(data)
    result = scanner.scan(good, installed=True, integrity_verified=True)
    assert result["lifecycle_state"] == "READY"
    assert result["issues"] == []

    unknown = artifact(data)
    result = scanner.scan(unknown, installed=True, integrity_verified=True)
    assert result["lifecycle_state"] == "LICENSE_RESTRICTED"
    assert "License and redistribution terms are not fully reviewed" in result["issues"]

    mismatch = governed_record(data, dependency_compatibility={"state": "INCOMPATIBLE", "requirements": []})
    assert scanner.scan(mismatch, installed=True, integrity_verified=True)["lifecycle_state"] == "INCOMPATIBLE"
    assert scanner.scan(good, installed=True, integrity_verified=False)["lifecycle_state"] == "QUARANTINED"


def test_unknown_license_record_cannot_become_an_inference_input(tmp_path):
    from roop_3d.models.local_provider import LocalWeightProvider
    data = b"local model bytes"
    record = artifact(data, locator={"scope": "fixture", "path": "fixture.onnx"})
    (tmp_path / "fixture.onnx").write_bytes(data)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="fixture", models=(record,)))
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry)
    catalog.provider = LocalWeightProvider(registry, root=tmp_path, scopes={"fixture": tmp_path})
    with pytest.raises(DomainError) as failure:
        catalog.resolve_for_inference(record.model_id)
    assert failure.value.code == "LICENSE_UNVERIFIED"


def test_corrupt_managed_cache_artifact_is_moved_to_quarantine(tmp_path):
    record = governed_record(b"expected")
    downloader = Downloader(tmp_path)
    final, _, _ = downloader.paths(record)
    path = tmp_path / final
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    with pytest.raises(DomainError) as failure:
        downloader.cached(record)
    assert failure.value.code == "MODEL_QUARANTINED"
    assert not path.exists()
    quarantined = list(path.parent.glob(path.name + ".*.quarantined"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"tampered"


def test_version_pinning_and_rollback_keep_artifact_versions_immutable(tmp_path):
    first_data, second_data = b"version one", b"version two"
    records = []
    for model_id, data, version in (("fixture_v1", first_data, "1.0.0"),
                                    ("fixture_v2", second_data, "2.0.0")):
        record = governed_record(data, model_id=model_id, version=version)
        (tmp_path / f"{model_id}.onnx").write_bytes(data)
        records.append(record)
    registry = ModelRegistry(root=tmp_path, manifest=Manifest(evidence_scope="fixture", models=tuple(records)))
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry)
    catalog.provider._scopes = {"fixture": tmp_path}

    catalog.pin_version("fixture", "fixture_v1")
    catalog.pin_version("fixture", "fixture_v2")
    assert catalog.pinned_version("fixture").model_id == "fixture_v2"
    assert catalog.select_version(records[0]).model_id == "fixture_v2"
    assert catalog.rollback_version("fixture")["active_model_id"] == "fixture_v1"
    assert catalog.pinned_version("fixture").model_id == "fixture_v1"
    assert catalog.select_version(records[1]).model_id == "fixture_v1"
    assert (tmp_path / "fixture_v1.onnx").read_bytes() == first_data
    assert (tmp_path / "fixture_v2.onnx").read_bytes() == second_data


def test_model_manifest_governance_does_not_rewrite_pinned_csv_evidence():
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "models/model_manifest.json").read_text("utf-8"))
    csv_path = root / "audit/model_and_license_register.csv"
    source_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert manifest["source_documents"]["roop_3d/audit/model_and_license_register.csv"] == source_digest
    assert all("security_status" in model and "redistribution_permission" in model for model in manifest["models"])


def test_restricted_torch_loader_fails_closed_without_weights_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda stream, map_location=None: "unsafe"))
    from roop_3d.models.serialization import load_restricted_torch_weights
    import io
    with pytest.raises(DomainError) as failure:
        load_restricted_torch_weights(io.BytesIO(b"checkpoint"))
    assert failure.value.code == "UNSAFE_CHECKPOINT_LOADER"


def test_restricted_torch_loader_never_falls_back_to_pickle(monkeypatch):
    calls = []
    def safe_load(stream, *, map_location, weights_only):
        calls.append((map_location, weights_only))
        return {"tensor": "fixture"}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=safe_load))
    from roop_3d.models.serialization import load_restricted_torch_weights
    import io
    assert load_restricted_torch_weights(io.BytesIO(b"checkpoint")) == {"tensor": "fixture"}
    assert calls == [("cpu", True)]


def test_download_lifecycle_reaches_ready_only_after_hash_and_governance_checks(tmp_path):
    data = b"verified download"
    record = governed_record(data, status="verified_remote", upstream_url="https://github.com/fixture/model.onnx",
                             provenance_state="VERIFIED_REMOTE", locator=None)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="fixture", models=(record,)))
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry)
    catalog.downloader.transport = httpx.MockTransport(lambda request: httpx.Response(200, content=data))
    manager = ModelDownloadManager(catalog)
    try:
        started = manager.start(record.model_id)
        import time
        deadline = time.monotonic() + 3
        result = manager.get(started["download_id"])
        while result["state"] not in {"completed", "failed", "cancelled"} and time.monotonic() < deadline:
            time.sleep(.01)
            result = manager.get(started["download_id"])
        assert result["state"] == "completed"
        assert result["lifecycle_state"] == "READY"
    finally:
        manager.close()
