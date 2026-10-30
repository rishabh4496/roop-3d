"""Unit tests for pose-bin classification and nearest-view reference selector."""
from __future__ import annotations

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.faceset.pose_selector import PoseSelector, classify_pose_bin
from roop_3d.faceset.schemas import (
    FaceSetDocument,
    FaceView,
    PoseBinSummary,
    QualityMetrics,
)


def _make_dummy_view(
    index: int,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
    openness: float = 0.0,
    quality_score: float = 1.0,
) -> FaceView:
    pose_bin, _, _ = classify_pose_bin(yaw, pitch)
    return FaceView(
        index=index,
        image_bytes=b"dummy_crop_bytes",
        embedding=tuple([1.0] + [0.0] * 511),
        landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)),
        bbox=(0.0, 0.0, 32.0, 32.0),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        expression={"openness": openness},
        quality=QualityMetrics(score=quality_score, sharpness=quality_score, detector_confidence=1.0),
        pose_bin=pose_bin,
        is_synthetic=False,
    )


def _make_document(views: list[FaceView]) -> FaceSetDocument:
    pose_bins = {}
    for v in views:
        pose_bins[v.pose_bin] = PoseBinSummary(
            yaw_bin="center",
            pitch_bin="center",
            members=(v.index,),
            support=1,
            representative_index=v.index,
            embedding=v.embedding,
        )
    return FaceSetDocument(
        faceset_id="test_faceset",
        name="Test FaceSet",
        source_sha256="0" * 64,
        archive_format="v2_standard",
        canonical_embedding=tuple([1.0] + [0.0] * 511),
        medoid_embedding=tuple([1.0] + [0.0] * 511),
        views=tuple(views),
        pose_bins=pose_bins,
    )


def test_classify_pose_bin():
    assert classify_pose_bin(0.0, 0.0)[0] == "center_center"
    assert classify_pose_bin(-30.0, 0.0)[0] == "left_center"
    assert classify_pose_bin(30.0, 0.0)[0] == "right_center"
    assert classify_pose_bin(0.0, -25.0)[0] == "center_up"
    assert classify_pose_bin(0.0, 25.0)[0] == "center_down"
    assert classify_pose_bin(-40.0, -20.0)[0] == "left_up"
    assert classify_pose_bin(40.0, 20.0)[0] == "right_down"


def test_empty_faceset_raises():
    doc = _make_document([])
    with pytest.raises(DomainError) as exc_info:
        PoseSelector(doc)
    assert exc_info.value.code == "FACESET_EMPTY"


def test_coverage():
    v0 = _make_dummy_view(0, yaw=0.0, pitch=0.0)
    v1 = _make_dummy_view(1, yaw=-30.0, pitch=0.0)
    v2 = _make_dummy_view(2, yaw=30.0, pitch=0.0)
    doc = _make_document([v0, v1, v2])
    selector = PoseSelector(doc)

    cov = selector.coverage()
    assert cov.get("center_center") == 1
    assert cov.get("left_center") == 1
    assert cov.get("right_center") == 1
    assert selector.view_count == 3


def test_select_best_reference_by_angle():
    v_frontal = _make_dummy_view(0, yaw=0.0, pitch=0.0)
    v_left = _make_dummy_view(1, yaw=-45.0, pitch=0.0)
    v_right = _make_dummy_view(2, yaw=45.0, pitch=0.0)
    doc = _make_document([v_frontal, v_left, v_right])
    selector = PoseSelector(doc)

    # Query frontal
    m_front = selector.select_best_reference(target_yaw=2.0, target_pitch=1.0)
    assert m_front.view_index == 0
    assert m_front.yaw == 0.0
    assert m_front.is_synthetic is False
    assert 0.0 <= m_front.confidence <= 1.0

    # Query left
    m_left = selector.select_best_reference(target_yaw=-40.0, target_pitch=0.0)
    assert m_left.view_index == 1
    assert m_left.yaw == -45.0

    # Query right
    m_right = selector.select_best_reference(target_yaw=50.0, target_pitch=0.0)
    assert m_right.view_index == 2
    assert m_right.yaw == 45.0


def test_select_best_reference_expression_tiebreaker():
    v_closed = _make_dummy_view(0, yaw=0.0, pitch=0.0, openness=0.0)
    v_open = _make_dummy_view(1, yaw=0.0, pitch=0.0, openness=0.9)
    doc = _make_document([v_closed, v_open])
    selector = PoseSelector(doc)

    m_open = selector.select_best_reference(target_yaw=0.0, target_pitch=0.0, target_expression=0.85)
    assert m_open.view_index == 1

    m_closed = selector.select_best_reference(target_yaw=0.0, target_pitch=0.0, target_expression=0.05)
    assert m_closed.view_index == 0


def test_select_best_reference_quality_preference():
    v_blurry = _make_dummy_view(0, yaw=10.0, pitch=0.0, quality_score=0.2)
    v_sharp = _make_dummy_view(1, yaw=12.0, pitch=0.0, quality_score=0.95)
    doc = _make_document([v_blurry, v_sharp])
    selector = PoseSelector(doc)

    # Both views are very close to target yaw=11.0, but sharp view should be preferred
    match = selector.select_best_reference(target_yaw=11.0, target_pitch=0.0)
    assert match.view_index == 1
    assert match.quality_score >= 0.9
