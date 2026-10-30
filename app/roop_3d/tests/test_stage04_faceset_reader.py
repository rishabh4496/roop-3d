"""Unit and integration tests for FaceSetReader."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset.pose_selector import PoseSelector


def _create_synthetic_face_image(yaw_offset: float = 0.0) -> bytes:
    """Create a 128x128 synthetic image with detectable face-like features."""
    img = np.full((128, 128, 3), 220, dtype=np.uint8)
    # Draw simple facial features (eyes, nose, mouth)
    left_eye_x = int(45 + yaw_offset)
    right_eye_x = int(85 + yaw_offset)
    cv2.circle(img, (left_eye_x, 45), 6, (40, 40, 40), -1)
    cv2.circle(img, (right_eye_x, 45), 6, (40, 40, 40), -1)
    cv2.circle(img, (int(65 + yaw_offset * 1.5), 70), 5, (60, 60, 60), -1)
    cv2.ellipse(img, (int(65 + yaw_offset), 95), (20, 8), 0, 0, 180, (50, 50, 50), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _create_v1_zip(images: list[bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, img_data in enumerate(images):
            zf.writestr(f"{i}.png", img_data)
    return bio.getvalue()


def _create_v2_zip(images: list[bytes], metadata: dict) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata).encode("utf-8"))
        for i, img_data in enumerate(images):
            zf.writestr(f"{i}.png", img_data)
    return bio.getvalue()


def test_reader_legacy_v2(tmp_path: Path):
    img0 = _create_synthetic_face_image(0.0)
    img1 = _create_synthetic_face_image(-10.0)
    meta = {
        "version": 2,
        "schema": "roop.fsz",
        "source_name": "test_person",
        "sources": [
            {
                "source_index": 0,
                "filename": "0.png",
                "pose": [0.0, 0.0, 0.0],
                "geometry": {
                    "bbox": [10.0, 10.0, 90.0, 90.0],
                    "landmarks_2d": [[45.0, 45.0], [85.0, 45.0], [65.0, 70.0], [45.0, 95.0], [85.0, 95.0]],
                    "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                },
                "identity": {"arcface_embedding": [1.0] + [0.0] * 511},
                "quality": {"sharpness": 0.9, "detector_confidence": 0.98, "score": 0.92},
            },
            {
                "source_index": 1,
                "filename": "1.png",
                "pose": [-30.0, 0.0, 0.0],
                "geometry": {
                    "bbox": [10.0, 10.0, 90.0, 90.0],
                    "landmarks_2d": [[35.0, 45.0], [75.0, 45.0], [50.0, 70.0], [35.0, 95.0], [75.0, 95.0]],
                    "yaw": -30.0, "pitch": 0.0, "roll": 0.0,
                },
                "identity": {"arcface_embedding": [0.0, 1.0] + [0.0] * 510},
                "quality": {"sharpness": 0.85, "detector_confidence": 0.95, "score": 0.88},
            },
        ],
    }

    archive_bytes = _create_v2_zip([img0, img1], meta)
    archive_file = tmp_path / "person_v2.fsz"
    archive_file.write_bytes(archive_bytes)
    initial_sha = hashlib.sha256(archive_bytes).hexdigest()

    reader = FaceSetReader(cache_root=tmp_path / "cache")
    doc = reader.read(archive_file)

    assert doc.faceset_id == "person_v2"
    assert doc.name == "test_person"
    assert doc.archive_format == "legacy_v2"
    assert len(doc.views) == 2
    assert len(doc.canonical_embedding) == 512
    assert len(doc.medoid_embedding) == 512
    assert doc.views[0].is_synthetic is False
    assert doc.views[1].is_synthetic is False

    # Verify input file was never mutated
    final_sha = hashlib.sha256(archive_file.read_bytes()).hexdigest()
    assert initial_sha == final_sha


def test_reader_legacy_v1_lazy_computation_and_cache(tmp_path: Path):
    img0 = _create_synthetic_face_image(0.0)
    img1 = _create_synthetic_face_image(5.0)
    archive_bytes = _create_v1_zip([img0, img1])
    archive_file = tmp_path / "person_v1.fsz"
    archive_file.write_bytes(archive_bytes)
    initial_sha = hashlib.sha256(archive_bytes).hexdigest()

    cache_dir = tmp_path / "cache"
    reader = FaceSetReader(cache_root=cache_dir)
    doc = reader.read(archive_file)

    assert doc.faceset_id == "person_v1"
    assert doc.archive_format == "legacy_v1"
    assert len(doc.views) == 2
    assert len(doc.canonical_embedding) == 512
    assert len(doc.medoid_embedding) == 512

    # Check cache was created
    cached_meta = cache_dir / "facesets" / initial_sha / "metadata.json"
    assert cached_meta.exists()

    # Re-reading should hit cache successfully
    doc_cached = reader.read(archive_file)
    assert len(doc_cached.views) == 2
    assert doc_cached.metadata.get("cached") is True

    # Verify input file was never mutated
    final_sha = hashlib.sha256(archive_file.read_bytes()).hexdigest()
    assert initial_sha == final_sha


def test_reader_image_directory(tmp_path: Path):
    img_dir = tmp_path / "face_photos"
    img_dir.mkdir()
    (img_dir / "0.png").write_bytes(_create_synthetic_face_image(0.0))
    (img_dir / "1.png").write_bytes(_create_synthetic_face_image(2.0))

    reader = FaceSetReader(cache_root=tmp_path / "cache")
    doc = reader.read(img_dir, faceset_id="my_photos")

    assert doc.faceset_id == "my_photos"
    assert doc.archive_format == "image_directory"
    assert len(doc.views) == 2


def test_reader_not_found():
    reader = FaceSetReader()
    with pytest.raises(DomainError) as exc_info:
        reader.read("non_existent_archive.fsz")
    assert exc_info.value.code == "FACESET_NOT_FOUND"
