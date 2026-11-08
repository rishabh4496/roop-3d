"""Acceptance checks for Stage 09 mask, restoration and frame fusion behavior."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop, warp_crop
from roop_3d.compositor.color_match import ColorStats, apply_lab_stats
from roop_3d.compositor.masks import build_face_mask, landmark_hull_mask
from roop_3d.compositor.occlusion import subtract_occluders
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.compositor.restoration import blend_restored_roi
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.engine.volumetric.session import ReferencePipeline


def _stats(offset: float = 0.0) -> ColorStats:
    return ColorStats((50 + offset,) * 3, (8, 8, 8), (60 + offset,) * 3, (10, 10, 10))


def _landmarks() -> np.ndarray:
    return np.array(((10, 10), (54, 10), (54, 55), (10, 55), (32, 8), (32, 57)), np.float32)


def test_semantic_intersection_removes_glasses_and_hull_fallback_is_soft():
    probabilities = np.zeros((64, 64, 19), np.float32)
    probabilities[10:56, 10:55, 1] = 1.0
    probabilities[26:34, 12:53, 1] = 0.0
    probabilities[26:34, 12:53, 6] = 1.0
    parsed = build_face_mask((64, 64), _landmarks(), probabilities=probabilities, feather_px=2)
    assert parsed.source.startswith("bisenet")
    assert parsed.alpha[20, 20] > 0.8
    assert parsed.alpha[30, 30] < 0.1

    fallback = build_face_mask((64, 64), _landmarks(), probabilities=None, feather_px=2)
    assert fallback.source == "landmark_hull"
    assert 0 < fallback.alpha[10, 32] < 1
    assert fallback.alpha[32, 32] > 0.9


def test_subtracted_occluder_remains_byte_identical_after_composition():
    target = np.full((80, 80, 3), 31, np.uint8)
    replacement = np.full((40, 40, 3), 220, np.uint8)
    alpha = np.ones((40, 40), np.float32)
    front = np.zeros_like(alpha)
    front[14:26, 14:26] = 1.0
    alpha = subtract_occluders(alpha, front)
    inverse = np.array(((1, 0, 20), (0, 1, 20)), np.float32)
    result = fuse_crop(target, replacement, inverse, alpha)
    np.testing.assert_array_equal(result[34:46, 34:46], target[34:46, 34:46])
    assert np.any(result != target)


def test_feather_fusion_never_changes_pixels_outside_dilated_blend_roi():
    target = np.full((128, 144, 3), (17, 35, 53), np.uint8)
    replacement = np.full((42, 46, 3), (210, 160, 120), np.uint8)
    alpha = np.zeros((42, 46), np.float32)
    cv2.ellipse(alpha, (23, 21), (17, 19), 0, 0, 360, 1, -1)
    alpha = cv2.GaussianBlur(alpha, (15, 15), 2)
    inverse = np.array(((1, 0, 47), (0, 1, 39)), np.float32)
    _, warped = warp_crop(replacement, inverse, target.shape, alpha)
    support = cv2.dilate((warped > 1e-5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    result = fuse_crop(target, replacement, inverse, alpha, mode=BlendMode.FEATHER)
    np.testing.assert_array_equal(result[~support], target[~support])
    assert np.isfinite(result).all()


@pytest.mark.parametrize("mode", [BlendMode.FEATHER, BlendMode.MULTI_BAND, BlendMode.POISSON])
def test_all_boundary_modes_return_finite_bgr(mode):
    target = np.full((72, 72, 3), 25, np.uint8)
    crop = np.full((40, 40, 3), 180, np.uint8)
    alpha = landmark_hull_mask((40, 40), np.array(((4, 4), (35, 4), (35, 36), (4, 36))), feather_px=2)
    matrix = np.array(((1, 0, 16), (0, 1, 16)), np.float32)
    result = fuse_crop(target, crop, matrix, alpha, mode=mode, levels=3)
    assert result.dtype == np.uint8 and result.shape == target.shape
    assert np.isfinite(result).all()
    assert int(result.max()) <= 255
    _, warped_alpha = warp_crop(crop, matrix, target.shape, alpha)
    radius = 8 if mode == BlendMode.MULTI_BAND else 2
    allowed = cv2.dilate((warped_alpha > 1e-5).astype(np.uint8),
                         np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    np.testing.assert_array_equal(result[~allowed], target[~allowed])


def test_restoration_and_lab_transfer_preserve_pixels_outside_face_mask():
    source = np.full((32, 32, 3), 45, np.uint8)
    restored = np.full_like(source, 200)
    mask = np.zeros((32, 32), np.float32)
    mask[8:24, 8:24] = 1
    result = blend_restored_roi(source, restored, mask, 0.4)
    np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
    assert np.all(result[mask == 1] > source[mask == 1])

    stats = ColorStats((40, 130, 140), (8, 10, 10), (55, 140, 150), (9, 12, 12))
    matched = apply_lab_stats(source, stats, mask)
    np.testing.assert_array_equal(matched[mask == 0], source[mask == 0])
    with pytest.raises(ValueError):
        blend_restored_roi(source, restored, mask, 1.1)


def test_temporal_filter_reduces_mask_jitter_and_resets_lost_tracks():
    smoother = TemporalSmoother(max_tracks=4, max_cache_bytes=2 * 1024 * 1024)
    base = np.zeros((64, 64), np.float32)
    cv2.ellipse(base, (32, 32), (20, 24), 0, 0, 360, 1, -1)
    raw_deltas, filtered_deltas, raw_luma_deltas, filtered_luma_deltas = [], [], [], []
    previous_raw = previous_filtered = None
    previous_raw_luma = previous_filtered_luma = None
    matrix = np.array(((1, 0, 8), (0, 1, 8)), np.float32)
    for frame_index in range(12):
        shift = 1 if frame_index % 2 else -1
        raw = cv2.warpAffine(base, np.array(((1, 0, shift), (0, 1, 0)), np.float32), (64, 64))
        smoother.begin_frame(frame_index, active_track_ids={7})
        offset = 8.0 if frame_index % 2 else -8.0
        result = smoother.update(7, matrix, raw, _stats(offset), frame_index=frame_index)
        if previous_raw is not None:
            raw_deltas.append(float(np.mean(np.abs(raw - previous_raw))))
            filtered_deltas.append(float(np.mean(np.abs(result.alpha - previous_filtered))))
            raw_luma_deltas.append(abs(offset - previous_raw_luma))
            filtered_luma_deltas.append(abs(result.color_stats.target_mean[0] - previous_filtered_luma))
        previous_raw, previous_filtered = raw, result.alpha
        previous_raw_luma, previous_filtered_luma = offset, result.color_stats.target_mean[0]
    assert np.mean(filtered_deltas) < np.mean(raw_deltas)
    assert np.mean(filtered_luma_deltas) < np.mean(raw_luma_deltas)
    assert smoother.track_count == 1 and smoother.cache_bytes <= smoother.max_cache_bytes

    smoother.begin_frame(12, active_track_ids=set())
    assert smoother.track_count == 0
    smoother.update(7, matrix, base, _stats(), frame_index=13)
    smoother.begin_frame(14, active_track_ids={7}, scene_cut=True)
    assert smoother.track_count == 0


def test_temporal_cache_is_strictly_bounded():
    smoother = TemporalSmoother(max_tracks=2, max_cache_bytes=12_000, max_mask_side=512)
    matrix = np.array(((1, 0, 0), (0, 1, 0)), np.float32)
    mask = np.ones((128, 128), np.float32)
    for track_id in (1, 2, 3, 4):
        smoother.update(track_id, matrix, mask, _stats())
        assert smoother.track_count <= 2
        assert smoother.cache_bytes <= 12_000


class _SyntheticParser:
    def predict(self, image: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((*image.shape[:2], 19), np.float32)
        probabilities[8:-8, 8:-8, 1] = 1.0
        return probabilities


def test_frame_compositor_fuses_replacement_and_protects_explicit_foreground():
    target = np.full((96, 96, 3), 32, np.uint8)
    crop = np.full((64, 64, 3), 190, np.uint8)
    inverse = np.array(((1, 0, 16), (0, 1, 16)), np.float32)
    replacement = SimpleNamespace(track_id=3, image=crop, inverse=inverse)
    points = np.array(((25, 24), (70, 24), (70, 70), (25, 70), (47, 22)), np.float32)
    foreground = np.zeros((64, 64), np.float32)
    foreground[26:34, 15:49] = 1.0
    compositor = FrameCompositor(parser=_SyntheticParser(), temporal=TemporalSmoother())
    result = compositor.compose(target, (replacement,), {3: points},
                                occluder_masks={3: foreground},
                                options=CompositeOptions(feather_px=2, color_strength=0))
    assert result.shape == target.shape and result.dtype == np.uint8
    assert np.any(result != target)
    np.testing.assert_array_equal(result[42:50, 31:65], target[42:50, 31:65])
    assert compositor.last_reports[3]["temporal_detail"]["backend"] == "opencv_dis_cpu"
    assert compositor.last_reports[3]["mask_watchdog"]["reset_reason"] == "INITIAL"


def test_reference_pipeline_exposes_full_frame_compositing_hook():
    target = np.full((96, 96, 3), 24, np.uint8)
    crop = np.full((64, 64, 3), 175, np.uint8)
    inverse = np.array(((1, 0, 16), (0, 1, 16)), np.float32)
    replacement = SimpleNamespace(track_id=5, image=crop, inverse=inverse)
    points = np.array(((26, 24), (68, 24), (68, 70), (26, 70), (47, 22)), np.float32)
    observation = SimpleNamespace(track_id=5, observation=SimpleNamespace(landmarks106=points))
    result = SimpleNamespace(observations=(observation,), replacements=(replacement,), scene_cut=False)
    session = ReferencePipeline()
    session.compositor.parser = _SyntheticParser()
    session.frames_processed = 1
    try:
        composed = session.composite_frame(target, result,
                                           options=CompositeOptions(feather_px=2, color_strength=0))
        assert composed.shape == target.shape
        assert np.any(composed != target)
    finally:
        session.release_session()
