"""Verified public sources and the unpublished-rights acknowledgement path (offline)."""
import copy
import json

import pytest

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.downloader import validate_url
from roop_3d.models.hub import ModelHubService
from roop_3d.models.registry import ModelRegistry, _apply_verified_sources, authorize


def _unpublished(registry):
    return [r for r in registry.records() if r.rights_unpublished]


def test_downloader_tls_keeps_verification_and_trusts_system_store():
    import ssl
    from roop_3d.models.downloader import tls_context
    context = tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert len(context.get_ca_certs()) > 0


def test_overlay_sources_match_registered_identity_and_pass_url_policy():
    registry = ModelRegistry()
    overlay = json.loads((application_root() / "roop_3d/models/verified_sources.json").read_text("utf-8"))
    assert overlay["sources"]
    for model_id, source in overlay["sources"].items():
        record = registry.get(model_id)
        assert (record.expected_sha256, record.byte_size) == (source["sha256"], source["byte_size"])
        assert record.upstream_url == source["upstream_url"] and record.provenance_state == "VERIFIED_REMOTE"
        assert validate_url(record.upstream_url) == source["upstream_url"]


def test_mismatching_overlay_is_rejected():
    registry = ModelRegistry()
    overlay = json.loads((application_root() / "roop_3d/models/verified_sources.json").read_text("utf-8"))
    broken = copy.deepcopy(overlay)
    first = next(iter(broken["sources"]))
    broken["sources"][first]["byte_size"] += 1
    with pytest.raises(ValueError):
        _apply_verified_sources(registry.manifest, broken)


def test_unpublished_rights_need_acknowledgement_and_never_grant_more():
    records = _unpublished(ModelRegistry())
    assert records
    for record in records:
        with pytest.raises(DomainError) as error:
            authorize(record, action="download")
        assert error.value.code == "LICENSE_ACCEPTANCE_REQUIRED"
        authorize(record, action="download", accepted_license=True)
        with pytest.raises(DomainError) as error:
            authorize(record, action="download", accepted_license=True, commercial=True)
        assert error.value.code == "COMMERCIAL_USE_NOT_VERIFIED"
        with pytest.raises(DomainError) as error:
            authorize(record, action="activate", accepted_license=True)
        assert error.value.code == "LICENSE_UNVERIFIED"


def test_hub_explains_every_model_that_cannot_be_downloaded():
    models = ModelHubService(ModelCatalog()).snapshot()["models"]
    for model in models:
        if model["download_allowed"]:
            assert model["status"] == "NOT_INSTALLED" and model["license"]["acceptance_required"]
        else:
            assert model["download_blocker"]
        if model["status"] == "NOT_INSTALLED" and model["weight_files"][0]["source_url"]:
            assert model["download_allowed"]


def test_github_release_redirect_is_allowed_only_to_release_storage():
    from roop_3d.models.downloader import validate_hf_redirect
    source = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
    good = "https://release-assets.githubusercontent.com/github-production-release-asset/1/abc?sp=r"
    assert validate_hf_redirect(source, good) == good
    for bad in ("https://evil.example/resnet18.onnx", "http://release-assets.githubusercontent.com/x",
                "https://objects.githubusercontent.com/x"):
        with pytest.raises(DomainError):
            validate_hf_redirect(source, bad)
    with pytest.raises(DomainError):
        validate_hf_redirect("https://example.com/x", good)


def test_published_licenses_are_recorded_and_unlock_activation_checks():
    registry = ModelRegistry()
    by_name = {r.name: r for r in registry.records() if r.upstream_url}
    for name, license_id in (("face_detection_yunet_2023mar.onnx", "MIT"), ("face_recognition_sface_2021dec.onnx", "Apache-2.0")):
        record = by_name[name]
        assert record.license_id == license_id and record.license_url and not record.rights_unpublished
        authorize(record, action="download")  # no acknowledgement needed for a published permissive license


def test_unsourced_models_report_missing_public_source():
    models = ModelHubService(ModelCatalog()).snapshot()["models"]
    unsourced = [m for m in models if m["weight_files"] and not m["weight_files"][0]["source_url"]]
    assert unsourced
    assert all(m["download_blocker"] for m in unsourced)
