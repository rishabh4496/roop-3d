from __future__ import annotations

import cv2
import numpy as np

from roop_3d.compositor.detail_temporal import TemporalDetailStabilizer
from roop_3d.compositor.pipeline import FrameCompositor
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
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96, use_cuda_flow=False)
    initial, first_report = stabilizer.update(3, enhanced, base, guide, frame_index=0)
    np.testing.assert_array_equal(initial, enhanced)
    assert first_report["backend"] == "none_no_history"
    assert first_report["temporal_alpha"] == .25 and first_report["history_weight"] == 0.0

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
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96, use_cuda_flow=False)
    stabilizer.update(1, first, base, guide, frame_index=0)
    visible = np.ones((side, side), np.float32)
    visible[20:70, 20:70] = 0.0
    output, report = stabilizer.update(1, second, base, guide, valid_mask=visible, frame_index=1)
    np.testing.assert_array_equal(output[20:70, 20:70], second[20:70, 20:70])
    assert report["influenced_fraction"] > 0
    assert report["temporal_alpha"] == .25 and 0.0 < report["history_weight"] <= .75


def test_detail_flow_resolution_can_exceed_residual_storage_cap():
    side = 96
    base = np.full((side, side, 3), 128, np.uint8)
    guide = _texture(side, 18)
    stabilizer = TemporalDetailStabilizer(flow_side=128, max_side=96, use_cuda_flow=False)
    stabilizer.update(4, base, base, guide, frame_index=0)
    _, report = stabilizer.update(4, base, base, guide, frame_index=1)
    assert report["backend"] == "opencv_dis_cpu"


def test_cuda_flow_unavailable_falls_back_to_cpu_dis(monkeypatch):
    try:
        import torch
    except ModuleNotFoundError:
        expected_reason = "ModuleNotFoundError"
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        expected_reason = "CUDA_RUNTIME_UNAVAILABLE"
    side = 96
    base = np.full((side, side, 3), 128, np.uint8)
    guide = _texture(side, 19)
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96)
    stabilizer.update(2, base, base, guide, frame_index=0)
    _, report = stabilizer.update(2, base, base, guide, frame_index=1)
    assert report["backend"] == "opencv_dis_cpu"
    assert report["gpu_fallback_reason"].startswith(expected_reason)


def test_torch_flow_fusion_keeps_warp_and_history_on_tensor_device():
    import pytest
    torch = pytest.importorskip("torch")

    class _TensorFlow:
        def pair(self, previous, current):
            assert previous.device == current.device
            flow = torch.zeros((1, 16, 16, 2), device=previous.device, dtype=torch.float32)
            return flow, flow

    stabilizer = TemporalDetailStabilizer(flow_side=16, max_side=16, use_cuda_flow=False)
    stabilizer._gpu_flow = _TensorFlow()
    guide = torch.arange(16 * 16, dtype=torch.uint8).reshape(16, 16)
    previous_residual = torch.ones((1, 3, 16, 16), dtype=torch.float16)
    state = type("State", (), {"guide": guide, "residual": previous_residual})()
    current = np.zeros((16, 16, 3), np.float32)
    valid_work = np.ones((16, 16), np.float32)
    valid_full = np.ones((32, 32), np.float32)
    fused, confidence, influenced, history, correction, reason = stabilizer._fuse_cuda(
        state, guide, current, valid_work, valid_full, (16, 16), (32, 32))
    assert reason is None and confidence > .99 and influenced == 1.0
    assert history == pytest.approx(.75, abs=1e-5)
    assert fused.device.type == "cpu" and fused.shape == (1, 3, 16, 16)
    assert correction.shape == (32, 32, 3)
    assert np.mean(correction) == pytest.approx(.75, abs=.02)


def test_nvidia_flow_bridge_uses_gpu_pair_contract_and_releases(monkeypatch):
    import sys
    from types import SimpleNamespace

    import roop_3d.optical_flow_tracker as flow_module

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def to(self, _device):
            return self

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    created = []

    class FakeNvidiaFlow:
        device = 0

        def __init__(self, size, *, device):
            assert size == (96, 96) and device == 0
            self.closed = False
            created.append(self)

        def pair(self, previous, current):
            assert isinstance(previous, FakeTensor) and isinstance(current, FakeTensor)
            flow = FakeTensor(np.zeros((1, 96, 96, 2), np.float32))
            return flow, flow

        def close(self):
            self.closed = True

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        from_numpy=lambda value: FakeTensor(value),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(flow_module, "NvidiaHardwareFlow", FakeNvidiaFlow)
    stabilizer = TemporalDetailStabilizer(flow_side=96, max_side=96)
    image = np.zeros((96, 96), np.uint8)
    forward, backward, backend, fallback = stabilizer._flow_pair(image, image)
    assert forward.shape == backward.shape == (96, 96, 2)
    assert backend == "opencv_nvidia_of_2_0_cpu_warp" and fallback is None
    assert stabilizer.capability()["backend_policy"] == "nvidia_of_2_0_if_available_else_cpu_dis"
    stabilizer.release()
    assert created[0].closed is True
    assert stabilizer.capability()["backend"] == "none_no_history"


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


def test_mask_watchdog_recovers_anomalous_frame_from_bidirectional_flow_context():
    watchdog = MaskSeamWatchdog()
    side = 128
    guide = np.random.default_rng(42).integers(0, 256, (side, side), dtype=np.uint8)
    face = np.zeros((side, side), np.float32)
    cv2.ellipse(face, (64, 64), (38, 46), 0, 0, 360, 1.0, -1)
    watchdog.observe_temporal(12, face, frame_index=0)
    context = {"previous_mask": face, "next_mask": face,
               "previous_guide": guide, "current_guide": guide, "next_guide": guide}
    report, repaired = watchdog.observe_and_recover(
        12, np.zeros_like(face), frame_index=1, context=context)
    assert report["anomaly"] is True
    assert report["recovery"] == "bidirectional_optical_flow"
    assert report["recovery_metrics"]["context_iou"] > .9
    assert np.mean(np.abs(repaired - face)) < .02


def test_mask_recovery_respects_current_occluder_and_scene_cut():
    watchdog = MaskSeamWatchdog()
    side = 128
    guide = _texture(side, 31)[:, :, 0]
    face = np.zeros((side, side), np.float32)
    cv2.ellipse(face, (64, 64), (38, 46), 0, 0, 360, 1.0, -1)
    watchdog.observe_temporal(13, face, frame_index=0)
    occluder = np.zeros_like(face)
    occluder[:, 62:70] = 1
    context = {"previous_mask": face, "next_mask": face,
               "previous_guide": guide, "current_guide": guide, "next_guide": guide,
               "occluder_mask": occluder}
    report, repaired = watchdog.observe_and_recover(
        13, np.zeros_like(face), frame_index=1, context=context)
    assert report["recovery"] == "bidirectional_optical_flow"
    assert np.max(repaired[:, 62:70]) == 0
    cut, unchanged = watchdog.observe_and_recover(
        13, np.zeros_like(face), frame_index=2, scene_cut=True, context=context)
    assert cut["recovery"] is None
    assert np.count_nonzero(unchanged) == 0


def test_mask_recovery_rejects_lookahead_across_appearance_cut():
    watchdog = MaskSeamWatchdog()
    face = np.zeros((128, 128), np.float32)
    cv2.ellipse(face, (64, 64), (38, 46), 0, 0, 360, 1.0, -1)
    watchdog.observe_temporal(14, face, frame_index=0)
    context = {
        "previous_mask": face,
        "next_mask": face,
        "previous_guide": np.full((128, 128), 32, np.uint8),
        "current_guide": np.full((128, 128), 224, np.uint8),
        "next_guide": np.full((128, 128), 32, np.uint8),
    }
    report, unchanged = watchdog.observe_and_recover(
        14, np.zeros_like(face), frame_index=1, context=context)
    assert report["anomaly"] is True
    assert report["recovery"] == "rejected_context"
    assert report["recovery_metrics"]["appearance_correlation_previous"] == 0.0
    assert np.count_nonzero(unchanged) == 0


def test_recovered_mask_preserves_swap_weight_and_projection_coverage():
    recovered = np.ones((8, 8), np.float32)
    coverage = np.full((8, 8), .5, np.float32)
    weighted = FrameCompositor._apply_recovered_alpha(recovered, .3, coverage)
    np.testing.assert_allclose(weighted, .15)
    zero = FrameCompositor._apply_recovered_alpha(recovered, 0.0)
    assert np.count_nonzero(zero) == 0


def test_120_frame_temporal_benchmark_reports_real_metrics_and_zero_cut_ghosting():
    from roop_3d.tools.temporal_stability import benchmark_temporal_stability

    result = benchmark_temporal_stability(frames=120, seed=7, side=128, cut_frame=60)
    assert result["frames"] == 120 and result["synthetic_only"] is True
    assert result["temporal_alpha"] == .25 and result["history_weight"] == .75
    assert result["scene_cut_ghosting_free"] is True
    assert result["scene_cut_max_pixel_delta"] == 0
    assert result["temporal_warp_error"]["reduction_percent"] > 70.0
    assert result["laplacian_flicker"]["reduction_percent"] > 70.0
    assert result["requested_70_percent_gate_met"] is True


def test_weight_sensitivity_benchmark_keeps_gate_and_cut_results_explicit():
    from roop_3d.tools.temporal_stability import benchmark_temporal_stability

    result = benchmark_temporal_stability(frames=120, seed=7, side=128, cut_frame=60,
                                          temporal_alpha=.5)
    assert result["temporal_alpha"] == .5
    assert result["history_weight"] == .5
    assert result["laplacian_flicker"]["reduction_percent"] < 70.0
    assert result["requested_70_percent_gate_met"] is False
    assert result["scene_cut_max_pixel_delta"] == 0
