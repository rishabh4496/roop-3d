"""Unit tests for FaceSet schemas, contracts, and V2 export functionality."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.faceset.faceset_reader import export_v2
from roop_3d.faceset.schemas import (
    ConsentMetadata,
    FaceSetDocument,
    FaceSetV2Manifest,
    FaceView,
    PoseBinSummary,
    QualityMetrics,
    ReferenceMatch,
)
from roop_3d.faceset.validator import ArchiveValidator


def _make_dummy_doc(faceset_id: str = "alice") -> FaceSetDocument:
    view = FaceView(
        index=0,
        image_bytes=b"\x89PNG\r\n\x1a\nfakeimage",
        embedding=tuple([1.0] + [0.0] * 511),
        landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)),
        bbox=(0.0, 0.0, 50.0, 50.0),
        yaw=5.0,
        pitch=2.0,
        roll=0.0,
        expression={"openness": 0.1},
        quality=QualityMetrics(sharpness=0.8, detector_confidence=0.99, score=0.85),
        pose_bin="center_center",
        is_synthetic=False,
        uv_map_path=None,
    )
    bin_summary = PoseBinSummary(
        yaw_bin="center",
        pitch_bin="center",
        members=(0,),
        support=1,
        representative_index=0,
        embedding=view.embedding,
    )
    return FaceSetDocument(
        faceset_id=faceset_id,
        name=faceset_id.capitalize(),
        source_sha256="a" * 64,
        archive_format="v2_standard",
        canonical_embedding=view.embedding,
        medoid_embedding=view.embedding,
        views=(view,),
        pose_bins={"center_center": bin_summary},
        metadata={"created_by": "test"},
    )


def test_schema_immutability():
    doc = _make_dummy_doc()
    with pytest.raises(Exception):
        doc.faceset_id = "other"  # type: ignore
    with pytest.raises(Exception):
        doc.views[0].yaw = 10.0  # type: ignore


def test_extra_fields_forbidden():
    with pytest.raises(Exception):
        QualityMetrics(sharpness=1.0, extra_unexpected_field=123)  # type: ignore


def test_angle_bounds():
    # Yaw out of bounds
    with pytest.raises(Exception):
        FaceView(
            index=0, image_bytes=b"1", embedding=tuple([1.0] + [0.0] * 511),
            landmarks=(), bbox=(0, 0, 1, 1), yaw=95.0, pitch=0.0, roll=0.0,
            quality=QualityMetrics(), pose_bin="center_center", is_synthetic=False,
        )


def test_is_synthetic_honest_provenance():
    # is_synthetic must be strictly False
    with pytest.raises(Exception):
        FaceView(
            index=0, image_bytes=b"1", embedding=tuple([1.0] + [0.0] * 511),
            landmarks=(), bbox=(0, 0, 1, 1), yaw=0.0, pitch=0.0, roll=0.0,
            quality=QualityMetrics(), pose_bin="center_center", is_synthetic=True,  # type: ignore
        )


def test_manifest_v2_serialization():
    doc = _make_dummy_doc("bob")
    consent = ConsentMetadata(
        consent_granted=True,
        consent_timestamp="2026-10-30T12:00:00Z",
        commercial_use_allowed=False,
        notes="Subject test consent",
    )
    manifest = FaceSetV2Manifest(
        schema_version=2,
        schema_type="roop.fsz.v2",
        faceset_id=doc.faceset_id,
        source_name=doc.name,
        created_at="2026-10-30T12:00:00Z",
        consent=consent,
        identity={"mean_embedding": list(doc.canonical_embedding)},
        pose_bins={},
        sources=[],
        provenance={"pipeline": "stage04"},
    )
    dumped = manifest.model_dump(by_alias=True)
    assert dumped["schema"] == "roop.fsz.v2"
    assert dumped["schema_version"] == 2
    assert dumped["consent"]["consent_granted"] is True


def test_export_v2_creates_valid_archive(tmp_path: Path):
    doc = _make_dummy_doc("carol")
    out_dir = tmp_path / "roop_3d/outputs/facesets"
    out_dir.mkdir(parents=True)
    out_file = out_dir / "carol.fsz"

    consent = ConsentMetadata(consent_granted=True, commercial_use_allowed=True)
    exported = export_v2(doc, out_file, consent=consent, app_root=tmp_path)
    assert exported == out_file
    assert out_file.exists()

    # Verify that the output archive is valid
    with zipfile.ZipFile(out_file, "r") as zf:
        res = ArchiveValidator.validate_zip(zf)
        assert res.is_valid is True
        assert res.has_metadata is True
        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        assert meta["schema"] == "roop.fsz.v2"
        assert meta["faceset_id"] == "carol"
        assert meta["consent"]["consent_granted"] is True
        assert meta["consent"]["commercial_use_allowed"] is True
        assert "0.png" in zf.namelist()


def test_export_v2_refuses_overwrite(tmp_path: Path):
    doc = _make_dummy_doc("david")
    out_dir = tmp_path / "roop_3d/outputs/facesets"
    out_dir.mkdir(parents=True)
    out_file = out_dir / "david.fsz"
    out_file.write_bytes(b"existing_file")

    with pytest.raises(DomainError) as exc_info:
        export_v2(doc, out_file, app_root=tmp_path)
    assert exc_info.value.code == "FACESET_EXISTS"
