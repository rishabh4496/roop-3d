from __future__ import annotations

import cv2
import numpy as np

from roop_3d.compositor.detail_temporal import TemporalDetailStabilizer
from roop_3d.compositor.stream_restoration import MaskSeamWatchdog
from roop_3d.core.pose import rotation_from_euler
from roop_3d.temporal import TrackTemporalStabilizer


def test_so3_pose_filter_damps_roll_jitter_and_takes_short_arc_across_wrap():
    stabilizer = TrackTemporalStabilizer(min_cutoff=.15, beta=.04)
    outputs = []
    for frame, roll in enumerate((179.0, -179.0, 179.5, -179.5, 179.0, -179.0)):
        outputs.append(stabilizer.filter_pose(4, "pose", (0.0, 0.0, roll), frame / 30.0))
    # The representation can choose either side of +/-180, but must never interpolate through 0.
    assert all(abs(float(value[2])) > 170 for value in outputs)

    noisy = [stabilizer.filter_pose(7, "pose", (0, 0, value), i / 30.0)[2]
             for i, value in enumerate((0.8, -0.8) * 15)]
    assert np.std(noisy[4:]) < .5


def test_so3_filter_resets_on_head_whip_and_accepts_scene_reset():
    stabilizer = TrackTemporalStabilizer()
    stabilizer.filter_rotation_matrix(1, "camera", rotation_from_euler(0, 0, 0), 0.0)
    whip = rotation_from_euler(45, 0, 0)
    current = stabilizer.filter_rotation_matrix(1, "camera", whip, 1 / 30)
    np.testing.assert_allclose(current, whip, atol=1e-8)
    stabilizer.reset()
    np.testing.assert_allclose(stabilizer.filter_rotation_matrix(1, "camera", whip, 2 / 30), whip, atol=1e-8)


def _texture(side: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(128, 36, (side, side)).clip(0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(noise, (0, 0), .8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_detail_stabilizer_is_identity_on_first_frame_and_resets_without_cut_ghosts():
    side = 96
    base = np.full((side, side, 3), 128, np.uint8)
    guide = _texture(side)
    enhanced = np.clip(base.astype(np.int16) + np.random.default_rng(2).integers(-15, 16, base.shape), 0, 255).astype(np.uint8)
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96)
    initial, first_report = stabilizer.update(3, enhanced, base, guide, frame_index=0)
    np.testing.assert_array_equal(initial, enhanced)
    assert first_report["backend"] == "opencv_dis_cpu"

    current = np.clip(base.astype(np.int16) + np.random.default_rng(3).integers(-15, 16, base.shape), 0, 255).astype(np.uint8)
    cut_output, report = stabilizer.update(3, current, base, _texture(side, 55), frame_index=1, scene_cut=True)
    np.testing.assert_array_equal(cut_output, current)
    assert report["reset_reason"] == "SCENE_CUT"
    assert stabilizer.cache_bytes > 0
    stabilizer.retain(set())
    assert stabilizer.track_count == 0 and stabilizer.cache_bytes == 0


def test_detail_stabilizer_never_imports_history_under_current_occluder():
    side = 128
    base = np.full((side, side, 3), 128, np.uint8)
    guide = _texture(side, 12)
    rng = np.random.default_rng(22)
    first = np.clip(base.astype(np.int16) + rng.integers(-18, 19, base.shape), 0, 255).astype(np.uint8)
    second = np.clip(base.astype(np.int16) + rng.integers(-18, 19, base.shape), 0, 255).astype(np.uint8)
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96)
    stabilizer.update(1, first, base, guide, frame_index=0)
    visible = np.ones((side, side), np.float32)
    visible[20:70, 20:70] = 0.0
    output, report = stabilizer.update(1, second, base, guide, valid_mask=visible, frame_index=1)
    np.testing.assert_array_equal(output[20:70, 20:70], second[20:70, 20:70])
    assert report["influenced_fraction"] > 0


def test_mask_watchdog_reports_single_frame_drop_with_per_face_thread_isolation():
    watchdog = MaskSeamWatchdog()
    face = np.zeros((128, 128), np.float32)
    cv2.ellipse(face, (64, 64), (38, 46), 0, 0, 360, 1.0, -1)
    assert watchdog.observe_temporal(5, face, frame_index=0)["anomaly"] is False
    drop = watchdog.observe_temporal(5, np.zeros_like(face), frame_index=1)
    assert drop["anomaly"] is True
    assert drop["iou"] == 0.0 and drop["area_jump"] > .12
    assert drop["recovery"] == "not_attempted_no_lookahead"
    # A different face and a scene cut cannot inherit the first face's matte baseline.
    other = watchdog.observe_temporal(6, np.zeros_like(face), frame_index=1)
    cut = watchdog.observe_temporal(5, face, frame_index=2, scene_cut=True)
    assert other["reset_reason"] == "INITIAL"
    assert cut["reset_reason"] == "SCENE_CUT" and cut["anomaly"] is False


def test_120_frame_temporal_benchmark_reports_real_metrics_and_zero_cut_ghosting():
    from roop_3d.tools.temporal_stability import benchmark_temporal_stability

    result = benchmark_temporal_stability(frames=120, seed=7, side=128, cut_frame=60)
    assert result["frames"] == 120 and result["synthetic_only"] is True
    assert result["scene_cut_ghosting_free"] is True
    assert result["scene_cut_max_pixel_delta"] == 0
    assert result["laplacian_flicker"]["stabilized"] < result["laplacian_flicker"]["raw"]
    assert result["requested_70_percent_gate_met"] == (
        result["laplacian_flicker"]["reduction_percent"] >= 70.0)
