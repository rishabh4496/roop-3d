"""Stage 3 Phase A: FaceSet quality / coverage analysis, plus the FaceSet-reader landmark fixes it relies on."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402

from roop_3d.faceset.faceset_reader import FaceSetReader  # noqa: E402
from roop_3d.reconstruction.faceset_quality import analyze_faceset  # noqa: E402


@pytest.fixture(scope="module")
def subject():
    return W.make_subject("quality", W.LAYOUT_5, 11)


@pytest.fixture(scope="module")
def other():
    return W.make_subject("other", W.LAYOUT_5, 12, width=0.02, nose_depth=0.002)


def _view_reasons(report, index):
    return next(q for q in report.views if q.index == index).exclusion_reasons


def test_front_only_reports_missing_angles_and_asks_for_images(subject):
    report = analyze_faceset(W.make_faceset("front", [W.make_view(subject, 0, 0.0)]))
    assert report.readiness == "front_only"
    assert report.usable_count == 1 and report.yaw_span_deg == 0.0
    assert {"three_quarter_image_left", "three_quarter_image_right", "profile_image_left", "profile_image_right"} <= set(report.missing)
    text = " ".join(report.requests)
    assert "[required]" in text and "image LEFT" in text and "image RIGHT" in text and "profile" in text
    assert report.pitch_coverage["level"] == [0]


def test_three_view_and_full_coverage_levels(subject):
    three = analyze_faceset(W.full_set(subject, (0, -42, 42)))
    assert three.readiness == "full_coverage" and three.usable_count == 3
    assert set(three.missing) >= {"profile_image_left", "profile_image_right"}      # profiles still requested
    assert not any(r.startswith("[required]") for r in three.requests)
    full = analyze_faceset(W.full_set(subject, (0, -42, 42, -74, 74)))
    assert full.readiness == "full_coverage" and not {"profile_image_left", "profile_image_right"} & set(full.missing)
    assert full.yaw_range_deg is not None and full.yaw_range_deg[0] < -60 and full.yaw_range_deg[1] > 60


def test_missing_one_side_is_named(subject):
    report = analyze_faceset(W.full_set(subject, (0, 40, 74)))
    assert report.readiness == "partial_coverage"
    assert "three_quarter_image_left" in report.missing and "profile_image_left" in report.missing


def test_pitch_distribution_is_measured(subject):
    doc = W.full_set(subject, (0, 0, 0), pitches=[-20.0, 0.0, 20.0])
    report = analyze_faceset(doc)
    assert report.pitch_coverage["looking_down"] and report.pitch_coverage["looking_up"] and report.pitch_coverage["level"]
    assert report.pitch_span_deg > 30


def test_exact_and_near_duplicates_are_excluded(subject):
    base = W.make_view(subject, 0, 0.0)
    exact = base.model_copy(update={"index": 1})
    reencoded = cv2.imdecode(np.frombuffer(base.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    ok, jpg = cv2.imencode(".jpg", reencoded, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    near = base.model_copy(update={"index": 2, "image_bytes": jpg.tobytes()})
    report = analyze_faceset(W.make_faceset("dups", [base, exact, near, W.make_view(subject, 3, -40.0)]))
    assert report.usable_count == 2
    assert _view_reasons(report, 1) == ["DUPLICATE_IMAGE"] and _view_reasons(report, 2) == ["DUPLICATE_IMAGE"]
    assert report.duplicate_groups and set(report.duplicate_groups[0]) == {0, 1, 2}


def test_single_foreign_view_is_excluded_as_identity_outlier(subject, other):
    views = [W.make_view(subject, i, y) for i, y in enumerate((0, -40, 40, -72))]
    views.append(W.make_view(other, 4, 20.0))
    report = analyze_faceset(W.make_faceset("outlier", views))
    assert report.identity["status"] == "consistent"
    assert 4 in report.identity["excluded_outliers"] and "IDENTITY_OUTLIER" in _view_reasons(report, 4)
    assert report.usable_count == 4


def test_mixed_identities_block_reconstruction(subject, other):
    views = [W.make_view(subject, 0, 0.0), W.make_view(subject, 1, -40.0), W.make_view(other, 2, 0.0), W.make_view(other, 3, 40.0)]
    report = analyze_faceset(W.make_faceset("mixed", views))
    assert report.identity["status"] == "inconsistent"
    assert report.readiness == "blocked_inconsistent_identity" and report.usable_count == 0
    assert report.requests[0].startswith("[blocking]") and "more than one person" in report.requests[0]


def test_invalid_poses_are_rejected(subject):
    mismatch = W.make_view(subject, 1, 0.0, stored_pose=(60.0, 0.0, 0.0))     # landmarks say frontal, pose says turned
    extreme = W.make_view(subject, 2, 0.0, stored_pose=(89.5, 0.0, 0.0))
    report = analyze_faceset(W.make_faceset("poses", [W.make_view(subject, 0, 0.0), mismatch, extreme]))
    assert "POSE_LANDMARK_MISMATCH" in _view_reasons(report, 1)
    assert {"POSE_OUT_OF_RANGE", "POSE_LANDMARK_MISMATCH"} & set(_view_reasons(report, 2))
    assert report.usable_indices == [0]


def test_fabricated_landmarks_are_never_treated_as_observations(subject):
    good = W.make_view(subject, 0, 0.0)
    fixed = good.model_copy(update={"index": 1, "landmarks": ((16.0, 32.0), (48.0, 32.0), (32.0, 48.0), (20.0, 60.0), (44.0, 60.0))})
    flagged = good.model_copy(update={"index": 2, "landmarks_trusted": False})
    report = analyze_faceset(W.make_faceset("fake", [good, fixed, flagged]))
    assert "LANDMARKS_NOT_OBSERVED" in _view_reasons(report, 1) and "LANDMARKS_NOT_OBSERVED" in _view_reasons(report, 2)


def test_blur_exposure_size_and_truncation(subject):
    clean = W.make_view(subject, 0, 0.0)
    img = cv2.imdecode(np.frombuffer(clean.image_bytes, np.uint8), cv2.IMREAD_COLOR)

    def variant(index, image, **update):
        ok, png = cv2.imencode(".png", image)
        assert ok
        return clean.model_copy(update={"index": index, "image_bytes": png.tobytes(), **update})

    blurred = variant(1, cv2.GaussianBlur(img, (0, 0), 25))
    dark = variant(2, (img * 0.05).astype(np.uint8))
    blown = variant(3, np.clip(img.astype(np.float32) * 6, 0, 255).astype(np.uint8))
    truncated = variant(4, img, bbox=(-250.0, 0.0, 130.0, 300.0))
    tiny = variant(5, img, bbox=(100.0, 100.0, 120.0, 130.0))
    report = analyze_faceset(W.make_faceset("blurry", [clean, blurred, dark, blown, truncated, tiny]))
    assert "TOO_BLURRY" in _view_reasons(report, 1)
    assert "BAD_EXPOSURE" in _view_reasons(report, 2) and "BAD_EXPOSURE" in _view_reasons(report, 3)
    assert "FACE_TRUNCATED_BY_IMAGE_EDGE" in _view_reasons(report, 4)
    assert "FACE_TOO_SMALL" in _view_reasons(report, 5)
    q = next(v for v in report.views if v.index == 0)
    assert q.usable and 0.0 < q.exposure_score <= 1.0 and q.sharpness > 0.03


def test_occlusion_is_only_reported_when_measurable(subject):
    view = W.make_view(subject, 0, 0.0)
    plain = analyze_faceset(W.make_faceset("occ", [view]))
    assert plain.views[0].occlusion_fraction is None
    assert "unavailable" in plain.capabilities["occlusion"]
    mask = np.full((W.IMG_H, W.IMG_W), 255, np.uint8)
    x1, y1, x2, y2 = (int(v) for v in view.bbox)
    mask[y1:y1 + int((y2 - y1) * 0.7), x1:x2] = 0
    blocked = analyze_faceset(W.make_faceset("occ", [view]), occlusion_masks={0: mask})
    assert blocked.views[0].occlusion_fraction == pytest.approx(0.7, abs=0.05)
    assert "FACE_HEAVILY_OCCLUDED" in _view_reasons(blocked, 0)


def test_expression_spread_is_reported(subject):
    views = [W.make_view(subject, 0, 0.0), W.make_view(subject, 1, -40.0, expression="open")]
    report = analyze_faceset(W.make_faceset("expr", views))
    assert report.expression_spread["openness_max"] - report.expression_spread["openness_min"] > 0.5


# ----------------------------------------------------------------------------- FaceSet reader landmark handling
def _archive(sources: list[dict], images: list[bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps({"version": 2, "schema": "roop.fsz", "source_name": "x", "sources": sources}))
        for i, data in enumerate(images):
            zf.writestr(f"{i}.png", data)
    return bio.getvalue()


def test_reader_derives_five_points_from_a_68_point_set_and_flags_fabricated_landmarks(tmp_path: Path):
    subject = W.make_subject("rd", W.LAYOUT_68, 5)
    view = W.make_view(subject, 0, 0.0)
    dense = [list(p) for p in view.landmarks_68]
    embedding = [1.0] + [0.0] * 511
    sources = [
        {"source_index": 0, "filename": "0.png", "pose": [0.0, 0.0, 0.0], "identity": {"arcface_embedding": embedding},
         "geometry": {"bbox": list(view.bbox), "landmarks_2d": dense}, "quality": {"score": 0.9}},
        {"source_index": 1, "filename": "1.png", "pose": [0.0, 0.0, 0.0], "identity": {"arcface_embedding": embedding},
         "geometry": {"bbox": list(view.bbox)}, "quality": {"score": 0.9}},
    ]
    path = tmp_path / "dense.fsz"
    path.write_bytes(_archive(sources, [view.image_bytes, view.image_bytes]))
    doc = FaceSetReader(cache_root=tmp_path / "cache").read(path)
    first, second = doc.views
    # Before the fix these were the first five JAW points of the 68-point set.
    assert np.allclose(first.landmarks, W.five_from_dense(np.asarray(dense)), atol=1e-6)
    assert first.landmarks_68 is not None and len(first.landmarks_68) == 68 and first.landmarks_trusted
    assert second.landmarks_trusted is False                      # the reader invents stand-in landmarks: flagged
    assert analyze_faceset(doc).views[1].exclusion_reasons.count("LANDMARKS_NOT_OBSERVED") == 1
