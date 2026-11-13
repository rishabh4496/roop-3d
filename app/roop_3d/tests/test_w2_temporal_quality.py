"""Deterministic CPU checks of the scene-cut detector, flow matte stabiliser and frequency-separated blend."""
import cv2
import numpy as np

from roop_3d.compositor.frequency_blend import FrequencyBlendOptions, frequency_blend
from roop_3d.compositor.mask_flow import FlowMaskStabilizer
from roop_3d.tracking.scene_cut import SceneCutDetector


def _scene(seed: int, size=(120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return base


def test_cut_is_reported_for_a_different_scene_and_flash_for_an_exposure_jump_that_keeps_structure():
    detector = SceneCutDetector()
    shot_a = _scene(1)
    assert detector.observe(shot_a).kind == "none"                      # first frame is never an event
    assert detector.observe(np.clip(shot_a.astype(np.int16) + 2, 0, 255).astype(np.uint8)).kind == "none"
    cut = detector.observe(_scene(2))
    assert cut.kind == "cut" and cut.to_telemetry()["cuts_total"] == 1

    flash_detector = SceneCutDetector()
    flash_detector.observe(shot_a)
    flash = flash_detector.observe(np.clip(shot_a.astype(np.int16) + 90, 0, 255).astype(np.uint8))
    assert flash.kind == "flash" and flash_detector.cuts_total == 0     # tracks survive a flash


def test_flow_stabiliser_cuts_alpha_flicker_on_a_moving_edge_and_forgets_the_old_shot_after_a_cut():
    rng = np.random.default_rng(0)
    texture = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 255, (96, 128), dtype=np.uint8), (0, 0), 2.0), None, 0, 255, cv2.NORM_MINMAX)
    stabiliser = FlowMaskStabilizer()
    raw, out = [], []
    for t in range(14):
        shift = 2 * t
        guide = np.roll(texture, shift, axis=1)
        alpha = np.zeros((96, 128), np.float32)
        alpha[:, 20 + shift: 60 + shift] = 1.0
        noisy = np.clip(alpha + rng.normal(0, 0.25, alpha.shape), 0, 1).astype(np.float32)
        raw.append(noisy)
        out.append(stabiliser.update("face", noisy, guide, frame_index=t).alpha)
    # Flicker = temporal std of pixels that are background in every frame (columns 100+: the edge never gets there).
    raw_flicker = np.std([frame[:, 100:] for frame in raw[4:]], axis=0).mean()
    out_flicker = np.std([frame[:, 100:] for frame in out[4:]], axis=0).mean()
    # The edge must not be smeared: the stabilised matte's edge stays within 1 px of the true one in the last frame.
    edge = lambda m: float(np.argmax(cv2.GaussianBlur(m, (0, 0), 1.0)[48, 20:] < 0.5))
    assert abs(edge(out[-1][:, :] ) - edge(np.pad(np.ones((96, 40), np.float32), ((0, 0), (20 + 26, 128 - 60 - 26))))) <= 1.0
    assert out_flicker < 0.7 * raw_flicker

    result = stabiliser.update("face", np.ones((96, 128), np.float32), texture, frame_index=14, scene_cut=True)
    assert result.reset_reason == "SCENE_CUT" and result.confidence == 0.0
    np.testing.assert_array_equal(result.alpha, np.ones((96, 128), np.float32))   # nothing of the old shot leaks in


def test_frequency_blend_keeps_target_detail_with_retain_hf_and_swap_detail_without():
    rng = np.random.default_rng(3)
    target = cv2.GaussianBlur(rng.integers(60, 200, (128, 128, 3), dtype=np.uint8), (0, 0), 6.0)
    grain = rng.normal(0, 14, target.shape).astype(np.float32)
    target = np.clip(target + grain, 0, 255).astype(np.uint8)
    swap = np.clip(cv2.GaussianBlur(target, (0, 0), 3.0).astype(np.float32) + 20, 0, 255).astype(np.uint8)   # smoothed, shifted
    alpha = np.zeros((128, 128), np.float32)
    alpha[32:96, 32:96] = 1.0

    def fine_energy(image):
        patch = image.astype(np.float32)[40:88, 40:88]
        return float(np.abs(patch - cv2.GaussianBlur(patch, (0, 0), 1.5)).mean())

    kept = frequency_blend(target, swap, alpha, FrequencyBlendOptions(retain_hf=1.0))
    smooth = frequency_blend(target, swap, alpha, FrequencyBlendOptions(retain_hf=0.0))
    assert abs(fine_energy(kept) - fine_energy(target)) < 0.1 * fine_energy(target)
    assert fine_energy(smooth) < 0.6 * fine_energy(target)
    np.testing.assert_allclose(kept[:16], target[:16].astype(np.float32), atol=0.5)    # outside the matte: the target
    # Reinhard base: the shifted swap takes on the target's brightness inside the matte
    assert abs(kept[40:88, 40:88].mean() - target[40:88, 40:88].mean()) < 6.0


def test_frequency_blend_is_selectable_in_boundary_fusion_and_leaves_pixels_outside_the_roi_untouched():
    from roop_3d.compositor.boundary_fusion import BlendMode, fuse_warped
    rng = np.random.default_rng(5)
    target = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    source = np.full_like(target, 200)
    alpha = np.zeros((96, 96), np.float32)
    alpha[30:66, 30:66] = 1.0
    out = fuse_warped(target, source, alpha, mode=BlendMode.FREQUENCY, levels=3)
    np.testing.assert_array_equal(out[:8], target[:8])
    assert not np.array_equal(out[40:56, 40:56], target[40:56, 40:56])
    import pytest
    with pytest.raises(ValueError):
        fuse_warped(target, source, alpha, mode=BlendMode.FREQUENCY, linear_light=True)
