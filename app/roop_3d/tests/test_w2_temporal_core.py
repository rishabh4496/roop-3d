"""Wave 2 temporal/quality gaps G6, G8-G12: numeric, synthetic, CPU-only."""
import cv2
import numpy as np
import pytest

from roop_3d.compositor.frequency_blend import FrequencyBlendOptions, frequency_blend
from roop_3d.compositor.mask_flow import FlowMaskStabilizer
from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker
from roop_3d.pipeline.subpixel import crop_box, warp_affine, warp_perspective
from roop_3d.tracking.scene_cut import CutEvent, SceneCutDetector, ssim
from roop_3d.tracking.temporal_telemetry import temporal_telemetry


# ---------------------------------------------------------------- G9 ByteTrack second pass
def _unit(seed):
    v = np.random.default_rng(seed).normal(size=512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _obs(x, embedding, score=1.0):
    box = np.array([x, 50, x + 80, 150], np.float32)
    points = np.array([[x + 20, 80], [x + 60, 80], [x + 40, 100], [x + 25, 125], [x + 55, 125]], np.float32)
    return FaceObservation(box, points, np.zeros((106, 2), np.float32), embedding, (0.0, 0.0, 0.0), score=score)


def _tracker():
    t = FaceTracker.__new__(FaceTracker)
    t._tracks, t._last_raw_box, t.stabilize, t.released = {}, {}, False, False
    t.threshold, t.low_threshold, t.low_match_iou = 0.5, 0.1, 0.4
    t.max_tracks, t.max_missed, t.next_id, t.scene_id, t.frame_index, t.policy = 8, 15, 0, 0, 0, None
    return t


def test_low_score_detection_continues_the_occluded_track_but_never_starts_one():
    t, emb = _tracker(), _unit(1)
    assert [f.track_id for f in t.assign([_obs(100, emb)])] == [0]
    occluded_embedding = _unit(2)                    # an occluded crop looks like nobody: appearance would reject it
    out = t.assign([_obs(104, occluded_embedding, score=0.25), _obs(400, _unit(3), score=0.2)])
    assert [(f.track_id, f.age) for f in out] == [(0, 2)]          # continued by IoU; the far low-score one is dropped
    np.testing.assert_array_equal(out[0].observation.embedding, emb)  # doubtful embedding not adopted
    assert t.next_id == 1                                           # nothing was spawned
    assert [f.track_id for f in t.assign([_obs(106, emb)])] == [0]  # the old ladder still re-identifies


def test_low_score_pass_uses_only_tracks_left_over_by_the_high_pass():
    t, a, b = _tracker(), _unit(1), _unit(2)
    t.assign([_obs(100, a), _obs(300, b)])
    out = t.assign([_obs(302, b), _obs(102, _unit(9), score=0.3)])
    assert sorted(f.track_id for f in out) == [0, 1]


# ---------------------------------------------------------------- G10 scene cut
def test_ssim_identity_and_flash_telemetry_keys():
    img = np.random.default_rng(0).integers(0, 255, (64, 64)).astype(np.float32)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)
    assert ssim(img, 255 - img) < 0.2
    event = CutEvent("flash", 0.5, 0.1, None, 3, cuts_total=2)
    assert event.to_telemetry() == {"scene_cut": False, "flash": True, "cuts_total": 2}


def test_pose_velocity_spike_promotes_a_borderline_change_to_a_cut():
    rng = np.random.default_rng(4)
    a = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (160, 120), interpolation=cv2.INTER_CUBIC)
    b = np.roll(a, 40, axis=1)[:, :, ::-1].copy()               # same statistics family, displaced + channel swap
    calm, spiky = SceneCutDetector(), SceneCutDetector()
    calm.observe(a)
    spiky.observe(a)
    e_calm, e_spike = calm.observe(b, pose_velocity=10.0), spiky.observe(b, pose_velocity=2000.0)
    assert e_spike.kind == "cut" or e_calm.kind == "cut"
    if e_calm.kind != "cut":
        assert 0.30 < e_calm.hist_distance and e_spike.pose_velocity == 2000.0


# ---------------------------------------------------------------- G11 mask flow
def _sequence(frames=30, speed=1.3, seed=3):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((128, 208)).astype(np.float32), (0, 0), 2.0)
    base = 255 * (base - base.min()) / (base.max() - base.min())
    x = np.arange(128, dtype=np.float32)[None, :].repeat(128, 0)
    for t in range(frames):
        guide = cv2.warpAffine(base, np.float32([[1, 0, speed * t], [0, 1, 0]]), (128, 128), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        truth = (1 / (1 + np.exp(np.clip((x - (50 + speed * t)) / 0.7, -30, 30)))).astype(np.float32)
        noisy = np.clip(truth + rng.normal(0, 0.15, truth.shape), 0, 1).astype(np.float32)
        yield np.clip(guide, 0, 255).astype(np.uint8), truth, noisy


def _edge(alpha):
    p = alpha.mean(0)
    i = int(np.argmax(p < 0.5))
    return i - 1 + (p[i - 1] - 0.5) / (p[i - 1] - p[i])


def test_flow_mask_lowers_alpha_noise_on_a_moving_edge_within_one_pixel_and_reset_leaves_no_ghost():
    stab, raw, out, err = FlowMaskStabilizer(), [], [], []
    for t, (guide, truth, noisy) in enumerate(_sequence()):
        result = stab.update("f", noisy, guide, frame_index=t)
        if t >= 5:
            raw.append(noisy[:, :40]); out.append(result.alpha[:, :40])
            err.append(abs(_edge(result.alpha) - _edge(truth)))
    assert np.var(out, axis=0).mean() < 0.6 * np.var(raw, axis=0).mean()
    assert max(err) < 1.0
    ghost = stab.update("f", np.zeros((128, 128), np.float32), guide, frame_index=30, scene_cut=True)
    assert ghost.reset_reason == "SCENE_CUT" and ghost.confidence == 0.0 and ghost.alpha.max() == 0.0
    after = stab.update("f", np.zeros((128, 128), np.float32), guide, frame_index=31)
    assert after.alpha.max() < 1e-6                       # the old shot's matte is not blended back in


# ---------------------------------------------------------------- G7 frequency blend
def _hf(image, mask):
    img = image.astype(np.float32)
    band = img - cv2.GaussianBlur(img, (0, 0), 2.0)
    return np.abs(band)[mask].mean()


def test_frequency_blend_energy_equivalence_and_low_band_colour():
    rng = np.random.default_rng(8)
    target = np.clip(cv2.GaussianBlur(rng.integers(60, 180, (160, 160, 3), dtype=np.uint8), (0, 0), 8).astype(np.float32)
                     + rng.normal(0, 10, (160, 160, 3)), 0, 255).astype(np.uint8)
    swap = np.clip(cv2.GaussianBlur(target, (0, 0), 4).astype(np.float32) + np.array([30, -20, 25]), 0, 255).astype(np.uint8)
    alpha = np.zeros((160, 160), np.float32); alpha[40:120, 40:120] = 1.0
    inner = np.zeros((160, 160), bool); inner[56:104, 56:104] = True
    kept = frequency_blend(target, swap, alpha, FrequencyBlendOptions(retain_hf=1.0))
    assert abs(_hf(kept, inner) - _hf(target, inner)) < 0.05 * _hf(target, inner)
    plain = frequency_blend(target, swap, alpha, FrequencyBlendOptions(retain_hf=0.0))
    assert abs(_hf(plain, inner) - _hf(swap, inner)) < 0.05 * _hf(target, inner)
    lab = lambda im: cv2.cvtColor(cv2.GaussianBlur(np.asarray(im, np.float32), (0, 0), 6) / 255, cv2.COLOR_BGR2LAB)[inner].mean(0)
    assert np.abs(lab(kept) - lab(target)).max() < 3.0          # Lab base of the swap moved onto the target's
    assert np.abs(lab(swap) - lab(target)).max() > 6.0          # (and it really was off before)
    np.testing.assert_array_equal(np.rint(kept[:8]).astype(np.uint8), target[:8])


# ---------------------------------------------------------------- G6 sub-pixel warp
def test_subpixel_shift_round_trip_beats_integer_crop_and_identity_is_exact():
    rng = np.random.default_rng(2)
    img = cv2.GaussianBlur(rng.integers(0, 255, (96, 96), dtype=np.uint8), (0, 0), 2.0)
    ident = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    np.testing.assert_array_equal(warp_affine(img, ident, (96, 96)), img)
    np.testing.assert_array_equal(warp_perspective(img, np.eye(3), (96, 96)), img)
    there = warp_affine(img, np.array([[1, 0, 0.37], [0, 1, 0]]), (96, 96))
    back = warp_affine(there, np.array([[1, 0, -0.37], [0, 1, 0]]), (96, 96))
    integer = np.roll(np.roll(img, round(0.37), axis=1), -round(0.37), axis=1)   # box rounded to whole pixels
    shifted_truth = warp_affine(img, np.array([[1, 0, 0.37], [0, 1, 0]]), (96, 96))
    err = lambda a, b: np.abs(a.astype(np.float32) - b.astype(np.float32))[8:-8, 8:-8].mean()
    assert err(back, img) < 1.0
    rounded_there = np.roll(img, 0, axis=1)                                      # round(0.37) == 0: the shift is lost
    assert err(shifted_truth, img) > 3 * err(back, img) and err(rounded_there, shifted_truth) > err(back, img)
    box = crop_box(img, (10.37, 10.0, 58.37, 58.0), (48, 48))
    assert box.shape == (48, 48)


# ---------------------------------------------------------------- G8 default XSeg
def test_default_visible_net_resolves_in_manifest_and_unknown_fails_with_named_code():
    from roop_3d.compositor.occlusion_analysis import DEFAULT_NETS
    from roop_3d.compositor.occlusion_model import DEFAULT_VISIBLE_NET, manifest_model_id
    assert DEFAULT_NETS[0] == DEFAULT_VISIBLE_NET
    assert len(manifest_model_id(DEFAULT_VISIBLE_NET)) == 24
    with pytest.raises(DomainError) as caught:
        manifest_model_id("xseg_99.onnx")
    assert caught.value.code == "VISIBLE_FACE_MODEL_NOT_IN_MANIFEST"


# ---------------------------------------------------------------- G12 one filter
def test_single_one_euro_definition_serves_every_consumer():
    import roop_3d.temporal as canonical
    import roop_3d.temporal_stabilizer as legacy
    from roop_3d.compositor import temporal as compositor_temporal
    from roop_3d.reconstruction import reenactment
    assert legacy.OneEuroFilter is canonical.OneEuroFilter
    assert compositor_temporal.OneEuroFilter is canonical.OneEuroFilter
    assert reenactment.OneEuroFilter is canonical.OneEuroFilter
    stabilizer = legacy.TemporalStabilizer(min_cutoff=.15, beta=.04, d_cutoff=1.0)
    rng = np.random.default_rng(1)
    out = np.array([stabilizer.filter(0, "c", 5 + rng.normal(0, .5, (5, 2)), i / 120) for i in range(600)])
    assert out[100:].std() < 0.3 * 0.5


# ---------------------------------------------------------------- telemetry
def test_temporal_telemetry_block():
    from roop_3d.temporal import OneEuroFilter
    f = OneEuroFilter(1.0, 0.5)
    f.update(np.zeros(2), 0.0); f.update(np.array([2.0, 0.0]), 0.1)
    raw = np.zeros((5, 2)); filt = raw + np.array([3.0, 4.0])
    block = temporal_telemetry(raw_landmarks=raw, filtered_landmarks=filt, cut_event=CutEvent("cut", .6, .5, None, 7, 4),
                               filters=[f], mask_flow_confidence=0.8)
    assert block["jitter_px"] == pytest.approx(5.0) and block["scene_cut"] is True and block["flash"] is False
    assert block["cuts_total"] == 4 and block["one_euro_cutoff_hz"] > 1.0
    assert block["stability_score"] == pytest.approx(np.exp(-2.5) * 0.9, abs=1e-3)
    assert set(block) == {"jitter_px", "one_euro_cutoff_hz", "scene_cut", "flash", "cuts_total",
                          "mask_flow_confidence", "stability_score"}
