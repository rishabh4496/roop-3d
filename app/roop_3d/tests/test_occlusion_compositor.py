"""Stage 5 occlusion compositor: unit and integration tests that need no model weights.

The stubs below are *test doubles injected into the analyser*, standing in for parser/network
outputs so the decision logic can be exercised deterministically. They are never used by product
code, and the real-model behaviour is covered separately in ``test_occlusion_real_models.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.compositor.face_regions import (REGION_CLASSES, build_regions, restoration_protection, teeth_estimate)
from roop_3d.compositor.matte import border_fade, decontaminate_edges, guided_filter, refine_occluder
from roop_3d.compositor.occlusion_analysis import OcclusionAnalyzer
from roop_3d.compositor.occlusion_model import VisibleFaceNet
from roop_3d.compositor.occlusion_tracker import OcclusionTracker
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor, RobustOptions
from roop_3d.compositor.quality import (blend_linear_light, local_illumination_match, match_sharpness,
                                        transfer_fine_texture)
from roop_3d.compositor.restoration import blend_restored_roi
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.core.errors import DomainError
from roop_3d.core.workstation import WorkstationParameters


def texture(side: int, seed: int = 1, sigma: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((side, side, 3)).astype(np.float32)
    if sigma > 0:
        base = cv2.GaussianBlur(base, (0, 0), sigma)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    return (60 + base * 140).astype(np.uint8)


# --------------------------------------------------------------------------- matte refinement
def test_guided_filter_snaps_soft_matte_to_the_image_edge_and_invents_nothing():
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, 32:] = 220                                  # real edge at x=32
    soft = np.zeros((64, 64), np.float32)
    soft[:, 32:] = 1.0
    soft = cv2.GaussianBlur(soft, (0, 0), 2.0)           # blurry, low-resolution matte
    refined = refine_occluder(soft, image, radius=4, eps=1e-3)
    step = lambda m: float(m[32, 32] - m[32, 31])         # transition located exactly at the real edge
    assert step(refined) > 0.9 and step(refined) > 2 * step(soft)
    # plateaus are tightened: less leftover face beside the object, less see-through inside it
    assert refined[32, 29] < soft[32, 29] and refined[32, 35] > soft[32, 35]
    # no foreground is created far from any evidence
    empty = refine_occluder(np.zeros((64, 64), np.float32), image)
    assert not empty.any()
    far = np.zeros((64, 64), np.float32)
    far[:, 56:] = 0.0
    assert refine_occluder(far, image).max() == 0.0
    with pytest.raises(DomainError):
        guided_filter(np.zeros((8, 8), np.float32), np.zeros((9, 9), np.float32))


def test_edge_decontamination_removes_baked_in_border_colour_but_keeps_opaque_core():
    source = np.full((64, 64, 3), (120, 150, 200), np.uint8)
    alpha = cv2.GaussianBlur(cv2.circle(np.zeros((64, 64), np.float32), (32, 32), 22, 1.0, -1), (0, 0), 3.0)
    ring = (alpha > 0.05) & (alpha < 0.9)
    source[ring] = (5, 5, 5)                              # contamination: dark background baked into the edge
    cleaned = decontaminate_edges(source, alpha)
    core = alpha >= 0.96
    np.testing.assert_array_equal(cleaned[core], source[core])
    before = np.abs(source[ring].astype(int) - (120, 150, 200)).mean()
    after = np.abs(cleaned[ring].astype(int) - (120, 150, 200)).mean()
    assert after < before * 0.5
    outer = (alpha > 0.05) & (alpha < 0.5)                  # the outer feather carries no baked-in colour any more
    assert np.abs(cleaned[outer].astype(int) - (120, 150, 200)).mean() < 12
    inner = (alpha >= 0.8) & (alpha < 0.96)                 # near-opaque pixels keep their own colour by design
    np.testing.assert_array_equal(cleaned[inner], source[inner])
    assert decontaminate_edges(source, np.ones((64, 64), np.float32)).tolist() == source.tolist()


# --------------------------------------------------------------------------- regions and restoration guard
def test_regions_separate_named_classes_and_teeth_are_only_a_labelled_heuristic():
    probs = np.zeros((64, 64, 19), np.float32)
    probs[..., 0] = 1
    for name, ids in REGION_CLASSES.items():
        probs[...] = 0
        probs[10:20, 10:20, ids[0]] = 1.0
        regions = build_regions(probs)
        assert regions[name][12, 12] == pytest.approx(1.0) and regions[name][40, 40] == 0
    probs[...] = 0
    probs[20:44, 20:44, 11] = 1.0                          # mouth interior
    regions = build_regions(probs)
    image = np.full((64, 64, 3), (40, 40, 140), np.uint8)   # reddish mouth
    image[28:36, 24:40] = (235, 235, 240)                   # bright, unsaturated "teeth"
    teeth = teeth_estimate(image, regions["mouth_interior"])
    assert teeth[32, 32] > 0.5 and teeth[24, 22] < 0.2
    assert not teeth_estimate(np.full((64, 64, 3), 90, np.uint8), np.zeros((64, 64), np.float32)).any()
    with pytest.raises(DomainError):
        build_regions(np.zeros((8, 8, 5), np.float32))


def test_restoration_guard_leaves_eyes_mouth_and_teeth_untouched():
    probs = np.zeros((64, 64, 19), np.float32)
    probs[..., 1] = 1.0
    probs[18:24, 14:26, 4] = 1.0
    probs[18:24, 14:26, 1] = 0.0                            # eye
    regions = build_regions(probs)
    protect = restoration_protection(regions, np.full((64, 64, 3), 100, np.uint8), grow_px=1)
    assert protect[20, 20] > 0.9 and protect[50, 50] == 0
    source = np.full((64, 64, 3), 50, np.uint8)
    restored = np.full_like(source, 220)
    mask = np.ones((64, 64), np.float32)
    guarded = blend_restored_roi(source, restored, mask, 1.0, protect=protect)
    assert np.all(guarded[20, 18:22] == 50)                 # eye pixels exactly preserved
    assert np.all(guarded[50, 50] == 220)                   # plain skin is restored
    assert restoration_protection(None) is None
    with pytest.raises(DomainError):
        blend_restored_roi(source, restored, mask, 1.0, protect=np.zeros((3, 3), np.float32))


# --------------------------------------------------------------------------- temporal occlusion tracker
def _blob(side, box, value):
    m = np.zeros((side, side), np.float32)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = value
    return m


def test_tracker_confident_new_detections_are_never_delayed_or_smeared():
    tracker = OcclusionTracker()
    image = texture(128)
    tracker.update(1, np.zeros((128, 128), np.float32), image, frame_index=0)
    hand = _blob(128, (40, 40, 90, 90), 1.0)               # a hand enters, confidently detected
    state = tracker.update(1, hand, image, frame_index=1)
    np.testing.assert_allclose(state.occluder, hand, atol=1e-6)


def test_tracker_uses_flow_to_stabilise_uncertain_pixels_where_the_object_moved():
    side = 128
    base = texture(side, seed=3, sigma=1.5)
    shifted = np.roll(base, 4, axis=1)                     # the whole scene moves 4 px right
    tracker = OcclusionTracker()
    tracker.update(1, _blob(side, (40, 40, 80, 80), 1.0), base, frame_index=0)
    # current frame: the model is unsure (0.5) about the object, which moved with the scene
    state = tracker.update(1, _blob(side, (44, 40, 84, 80), 0.5), shifted, frame_index=1)
    inside_moved = state.occluder[50:70, 50:78].mean()
    assert inside_moved > 0.6, "history warped by the flow must push the uncertain pixels toward occluded"
    assert abs(state.motion[0] - 4.0) < 1.5 and abs(state.motion[1]) < 1.5
    assert state.history_influence > 0.05
    # pixels that were never occluded stay clear (history is warped, not smeared in place)
    assert state.occluder[100:120, 10:30].max() < 0.05


def test_tracker_resets_on_scene_cut_frame_gap_appearance_jump_and_holds_with_limit():
    side = 128
    image = texture(side, seed=5)
    other = 255 - texture(side, seed=9, sigma=3.0)
    occ = _blob(side, (30, 30, 90, 90), 0.5)
    tracker = OcclusionTracker(max_hold=2)
    tracker.update(2, occ, image, frame_index=0)
    assert tracker.update(2, occ, image, frame_index=1, scene_cut=True).reset_reason == "SCENE_CUT"
    tracker.update(2, occ, image, frame_index=2)
    assert tracker.update(2, occ, image, frame_index=9).reset_reason == "FRAME_GAP"
    tracker.update(2, occ, image, frame_index=10)
    jumped = tracker.update(2, occ, other, frame_index=11)
    assert jumped.reset_reason == "APPEARANCE_CUT" and jumped.history_influence == 0.0
    # analysis failure: hold with decay, flagged, and expiry
    tracker.update(2, _blob(side, (30, 30, 90, 90), 1.0), other, frame_index=12)
    first = tracker.update(2, None, other, frame_index=13)
    second = tracker.update(2, None, other, frame_index=14)
    assert first.held and second.held and second.occluder.max() < first.occluder.max()
    third = tracker.update(2, None, other, frame_index=15)
    assert third.occluder is None and third.reset_reason == "HOLD_EXPIRED"
    tracker.retain(set())
    assert tracker.track_count == 0
    with pytest.raises(DomainError):
        tracker.update(3, np.zeros((5, 5), np.float32), image, frame_index=0)


# --------------------------------------------------------------------------- quality helpers
def test_local_illumination_follows_target_shading_without_double_counting_global_gain():
    flat = np.full((96, 96, 3), 150, np.uint8)
    gradient = np.tile(np.linspace(70, 200, 96, dtype=np.float32)[None, :, None], (96, 1, 3)).astype(np.uint8)
    alpha = np.ones((96, 96), np.float32)
    matched = local_illumination_match(flat, gradient, alpha, strength=1.0)
    luma = matched[..., 1].astype(float)
    assert luma[:, :10].mean() < luma[:, -10:].mean() - 20          # now darker on the dark side
    assert abs(luma.mean() - 150) < 12                                # mean brightness not shifted a second time
    np.testing.assert_array_equal(local_illumination_match(flat, gradient, alpha, strength=0.0), flat)


def test_sharpness_matching_blurs_to_target_but_never_sharpens_and_ignores_the_warp_border():
    sharp = texture(96, seed=2, sigma=0.7)
    blurred_target = cv2.GaussianBlur(sharp, (0, 0), 1.6)
    alpha = np.ones((96, 96), np.float32)
    foot = np.ones((96, 96), np.float32)
    matched, info = match_sharpness(sharp, blurred_target, alpha, foot)
    energy = lambda im: float((np.hypot(*np.gradient(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)))).mean())
    assert info["sigma"] > 0.4
    assert abs(energy(matched) - energy(blurred_target)) < abs(energy(sharp) - energy(blurred_target)) * 0.5
    same, info2 = match_sharpness(blurred_target, sharp, alpha, foot)          # target sharper: untouched
    np.testing.assert_array_equal(same, blurred_target)
    assert info2 == {"sigma": 0.0, "motion_length": 0.0}
    # footprint hole (black border outside the warped crop) must not be dragged into the face
    holed = sharp.copy()
    holed[:, :16] = 0
    foot2 = foot.copy()
    foot2[:, :16] = 0
    out, _ = match_sharpness(holed, blurred_target, alpha, foot2)
    assert out[40, 24:40].mean() > sharp[40, 24:40].mean() * 0.7


def test_motion_blur_matching_is_directional():
    sharp = texture(96, seed=4, sigma=0.7)
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, :] = 1 / 9
    target = cv2.filter2D(sharp, -1, kernel)                       # horizontal motion blur
    alpha, foot = np.ones((96, 96), np.float32), np.ones((96, 96), np.float32)
    matched, info = match_sharpness(sharp, target, alpha, foot, motion_px=(14.0, 0.0), shutter=0.6)
    assert info["motion_length"] > 4
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    horizontal = lambda im: float(np.abs(np.diff(gray(im), axis=1)).mean())
    vertical = lambda im: float(np.abs(np.diff(gray(im), axis=0)).mean())
    assert horizontal(matched) < horizontal(sharp) * 0.8
    assert vertical(matched) / vertical(sharp) > horizontal(matched) / horizontal(sharp)


def test_fine_texture_comes_from_the_target_only_on_skin_and_only_when_present():
    target = texture(96, seed=7, sigma=0.0)                         # strongly textured target skin
    smooth = cv2.GaussianBlur(target, (0, 0), 3.0)
    alpha, foot = np.ones((96, 96), np.float32), np.ones((96, 96), np.float32)
    skin = np.zeros((96, 96), np.float32)
    skin[:, :48] = 1.0                                              # eyes/mouth/etc. would be outside this
    out, gain = transfer_fine_texture(smooth, target, alpha, skin, foot, strength=1.0)
    hf = lambda im: float(np.abs(im.astype(np.float32) - cv2.GaussianBlur(im, (0, 0), 2.0)).mean())
    assert gain > 0.05 and hf(out[:, :48]) > hf(smooth[:, :48]) * 1.5
    np.testing.assert_array_equal(out[:, 52:], smooth[:, 52:])      # off-skin pixels untouched
    # a motion-blurred / smooth target has no fine band: nothing is invented
    unchanged, none = transfer_fine_texture(smooth, smooth, alpha, np.ones((96, 96), np.float32), foot)
    assert none < 0.02 and np.abs(unchanged.astype(int) - smooth.astype(int)).max() <= 1


def test_linear_light_blend_is_physically_correct_at_half_coverage():
    black = np.zeros((4, 4, 3), np.uint8)
    white = np.full((4, 4, 3), 255, np.uint8)
    mid = blend_linear_light(black, white, np.full((4, 4), 0.5, np.float32))
    assert mid[0, 0, 0] == pytest.approx(188, abs=1.5)               # 0.5 linear -> sRGB 188, not 128
    full = blend_linear_light(black, white, np.ones((4, 4), np.float32))
    assert full[0, 0, 0] == pytest.approx(255, abs=0.5)


# --------------------------------------------------------------------------- analyser decisions (test doubles)
class _StubNet(VisibleFaceNet):
    def __init__(self, name, fn, *, available=True):
        self.model_name, self._fn, self._available = name, fn, available
        self.model, self.inference_verified, self.last_error, self.channel_order = None, False, None, "bgr"

    def available(self):
        return self._available

    def predict(self, image):
        self.inference_verified = True
        return self._fn(image.shape[:2]).astype(np.float32)


class _StubParser:
    def __init__(self, face=True, background=False):
        self.face, self.background = face, background

    def predict(self, image):
        h, w = image.shape[:2]
        p = np.zeros((h, w, 19), np.float32)
        if self.background:
            p[..., 0] = 1.0
        else:
            p[..., 0] = 1.0
            p[h // 6: -h // 6, w // 6: -w // 6, 0] = 0.0
            p[h // 6: -h // 6, w // 6: -w // 6, 1] = 1.0
        return p


def _analyser(parser, *nets):
    analyser = OcclusionAnalyzer(parser=parser)
    analyser.nets = list(nets)
    return analyser


POINTS = np.array(((40, 40), (216, 40), (216, 216), (40, 216)), np.float32)


def _visible(hole=None):
    def fn(shape):
        m = np.zeros(shape, np.float32)
        m[36:220, 36:220] = 1.0                      # a face-sized region, like a real network output
        if hole:
            x0, y0, x1, y1 = hole
            m[y0:y1, x0:x1] = 0.0
        return m
    return fn


def test_analyser_full_tier_marks_a_hand_as_occluder_and_keeps_the_face_shape():
    hand = (100, 110, 170, 160)
    analyser = _analyser(_StubParser(), _StubNet("xseg_3.onnx", _visible(hand)), _StubNet("face_occluder.onnx", _visible(hand)))
    result = analyser.analyze(texture(256), POINTS, feather_px=3)
    assert result.tier == "full" and result.parser_trusted and result.confidence > 0.8
    assert result.occluder[130, 130] > 0.9 and result.occluder[70, 70] < 0.1
    assert result.face_alpha[130, 130] > 0.9                              # shape is independent of occlusion
    assert result.regions is not None and result.protect is not None
    assert result.nets_used == ("xseg_3.onnx", "face_occluder.onnx")


def test_analyser_disagreeing_networks_preserve_target_pixels_instead_of_guessing():
    only_a = _visible((100, 110, 170, 160))
    analyser = _analyser(_StubParser(), _StubNet("xseg_3.onnx", only_a), _StubNet("face_occluder.onnx", _visible()))
    result = analyser.analyze(texture(256), POINTS, feather_px=3)
    assert result.occluder[130, 130] > 0.9, "if either network says occluded, the original pixels are kept"
    assert result.confidence < 0.8                                         # and the disagreement is reflected


def test_analyser_distrusts_a_parser_that_contradicts_the_networks():
    """Real failure mode: BiSeNet 'confidently' finds no face on a profile while XSeg sees it."""
    analyser = _analyser(_StubParser(background=True), _StubNet("xseg_3.onnx", _visible()))
    result = analyser.analyze(texture(256), POINTS, feather_px=3)
    assert result.tier == "neural" and not result.parser_trusted
    assert "PARSER_DISAGREES_WITH_NETWORKS" in result.reasons
    assert result.face_alpha[128, 128] > 0.9, "the swap must not vanish just because the parser failed"
    assert result.regions is None and result.protect is None and not result.skin.any()


def test_parser_trust_has_hysteresis_per_track():
    analyser = _analyser(_StubParser(), _StubNet("xseg_3.onnx", _visible()))
    tiers = [analyser.analyze(texture(256), POINTS, track_id=4).tier for _ in range(4)]
    assert tiers == ["neural", "neural", "full", "full"]                   # earned after 3 consistent frames
    analyser.parser = _StubParser(background=True)                         # parser breaks: trust is lost at once
    assert analyser.analyze(texture(256), POINTS, track_id=4).tier == "neural"
    analyser.parser = _StubParser()
    assert analyser.analyze(texture(256), POINTS, track_id=4).tier == "neural"   # and has to be earned again
    analyser.retain(())
    assert analyser._trust_streak == {}


def test_collapsed_landmark_hull_hands_the_face_shape_to_the_networks():
    """Real failure mode: on a face rolled 90 degrees the 106-point hull collapsed to a small cluster."""
    tiny = np.array(((100, 100), (130, 100), (130, 130), (100, 130)), np.float32)
    analyser = _analyser(_StubParser(), _StubNet("xseg_3.onnx", _visible()), _StubNet("face_occluder.onnx", _visible()))
    result = analyser.analyze(texture(256), tiny, feather_px=3)
    assert "HULL_IMPLAUSIBLE_USING_NETWORKS" in result.reasons and result.tier == "neural"
    assert result.face_alpha[60, 60] > 0.9 and result.face_alpha[10, 10] < 0.05   # the network's face, not the tiny hull
    assert result.regions is None
    healthy = analyser.analyze(texture(256), POINTS, feather_px=3)
    assert "HULL_IMPLAUSIBLE_USING_NETWORKS" not in healthy.reasons


def test_analyser_without_any_evidence_is_geometric_and_says_occlusion_is_unverified():
    analyser = _analyser(None)
    result = analyser.analyze(texture(256), POINTS, feather_px=3)
    assert result.tier == "geometric" and result.occluder is None and result.confidence <= 0.3
    assert "OCCLUSION_UNVERIFIED" in result.reasons
    manual = np.zeros((256, 256), np.float32)
    manual[100:130, 100:130] = 1.0
    assert analyser.analyze(texture(256), POINTS, explicit_front=manual).occluder[110, 110] == 1.0


def test_analyser_reports_missing_networks_and_does_not_retry_them():
    broken = _StubNet("xseg_3.onnx", _visible(), available=False)
    analyser = _analyser(_StubParser(), broken)
    result = analyser.analyze(texture(256), POINTS)
    assert result.tier == "semantic" and "NETWORKS_UNAVAILABLE" in result.reasons
    assert analyser.capability()["unavailable"] == ["xseg_3.onnx"]
    assert "HEURISTIC" in analyser.capability()["teeth"]


def test_visible_face_net_rejects_unsupported_models_and_reports_missing_weights():
    with pytest.raises(ValueError):
        VisibleFaceNet(model_name="not_a_model.onnx")
    with pytest.raises(ValueError):
        VisibleFaceNet(model_name="xseg_3.onnx", channel_order="gbr")


# --------------------------------------------------------------------------- compositor integration
def _scene(side=256):
    frame = texture(side, seed=11, sigma=2.0)
    return frame


def _replacement(track_id, colour, inverse, side=128):
    return SimpleNamespace(track_id=track_id, image=np.full((side, side, 3), colour, np.uint8),
                           inverse=np.asarray(inverse, np.float32))


def _compositor(*nets, parser=None):
    compositor = FrameCompositor(parser=parser or _StubParser(), temporal=TemporalSmoother())
    compositor._analyzer = _analyser(compositor.parser, *nets)
    return compositor


ROBUST = CompositeOptions(occlusion_mode="robust", feather_px=2, color_strength=0,
                          robust=RobustOptions(local_illumination=0, texture_strength=0))


def test_robust_compositor_keeps_the_hand_but_replaces_the_face_around_it():
    frame = _scene()
    inverse = ((1.6, 0, 20), (0, 1.6, 20))
    replacement = _replacement(1, (30, 200, 30), inverse)
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    hand = (100, 110, 170, 160)                                         # analysis-crop (256) coordinates
    compositor = _compositor(_StubNet("xseg_3.onnx", _visible(hand)), _StubNet("face_occluder.onnx", _visible(hand)))
    for index in range(3):                                              # parser trust is earned over 3 frames
        out = compositor.compose(frame, (replacement,), {1: points}, frame_index=index, options=ROBUST)
    # analysis crop 256 -> frame: scale 1.6/2 = 0.8, offset 20
    hx0, hy0, hx1, hy1 = (int(20 + v * 0.8) for v in hand)
    np.testing.assert_array_equal(out[hy0 + 6:hy1 - 6, hx0 + 6:hx1 - 6], frame[hy0 + 6:hy1 - 6, hx0 + 6:hx1 - 6])
    face_point = out[int(20 + 70 * 0.8), int(20 + 70 * 0.8)]
    assert face_point[1] > 150 and face_point[0] < 90                   # green replacement visible elsewhere
    report = compositor.last_reports[1]
    assert report["tier"] == "full" and report["fusion"]["occluded_fraction"] > 0.05
    assert report["analysis_ms"]["total"] > 0
    assert report["temporal_detail"]["backend"] in {
        "none_no_history", "opencv_dis_cpu", "opencv_nvidia_of_2_0_cpu_warp"}
    assert 0 <= report["mask_watchdog"]["iou"] <= 1


def test_robust_and_standard_modes_are_selectable_and_standard_is_the_default():
    assert CompositeOptions().occlusion_mode == "standard"
    with pytest.raises(ValueError):
        CompositeOptions(occlusion_mode="magic")
    with pytest.raises(ValueError):
        RobustOptions(analysis_size=8)
    frame = _scene()
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, 20), (0, 1.6, 20)))
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    standard = FrameCompositor(parser=_StubParser(), temporal=TemporalSmoother()).compose(
        frame, (replacement,), {1: points}, options=CompositeOptions(feather_px=2, color_strength=0))
    assert np.any(standard != frame)


def test_nearer_face_is_never_painted_over_by_a_farther_swapped_face():
    frame = _scene(320)
    far = _replacement(1, (20, 20, 230), ((1.0, 0, 30), (0, 1.0, 60)))      # smaller = farther, red
    near = _replacement(2, (230, 30, 30), ((1.5, 0, 70), (0, 1.5, 40)))     # bigger = nearer, blue
    far_pts = np.array(((38, 70), (150, 70), (150, 170), (38, 170)), np.float32)
    near_pts = np.array(((100, 45), (250, 45), (250, 215), (100, 215)), np.float32)
    nets = lambda: (_StubNet("xseg_3.onnx", _visible()), _StubNet("face_occluder.onnx", _visible()))
    results = []
    for order in ((far, near), (near, far)):
        compositor = _compositor(*nets())
        results.append(compositor.compose(frame, order, {1: far_pts, 2: near_pts}, frame_index=0, options=ROBUST))
    np.testing.assert_array_equal(results[0], results[1])                      # draw order is depth, not list order
    out = results[0]
    # inside the near face's own area nothing of the far (red-dominant... BGR: 20,20,230) swap may remain
    inner = out[100:200, 150:230]
    assert not np.any((inner[..., 2] > 200) & (inner[..., 0] < 60)), "far face leaked through the nearer face"
    assert "OCCLUDED_BY_NEARER_FACE" in compositor.last_reports[1]["reasons"]


def test_partially_outside_frame_is_reported_and_reflected_pixels_never_count_as_foreground():
    frame = _scene(160)
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, -50), (0, 1.6, 10)))      # left part of the crop is off-frame
    points = np.array(((10, 40), (200, 40), (200, 210), (10, 210)), np.float32)
    # the network (like any model fed reflected pixels) calls the off-frame band "not face"
    edge = _visible((0, 0, 70, 256))
    compositor = _compositor(_StubNet("xseg_3.onnx", edge), _StubNet("face_occluder.onnx", edge))
    out = compositor.compose(frame, (replacement,), {1: points}, frame_index=0, options=ROBUST)
    report = compositor.last_reports[1]
    assert "PARTIALLY_OUTSIDE_FRAME" in report["reasons"] and report["outside_frame_fraction"] > 0.05
    assert out.shape == frame.shape and np.isfinite(out).all()


def test_scene_cut_and_lost_tracks_reset_occlusion_history():
    frame = _scene()
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, 20), (0, 1.6, 20)))
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    compositor = _compositor(_StubNet("xseg_3.onnx", _visible((90, 90, 150, 150))))
    compositor.compose(frame, (replacement,), {1: points}, frame_index=0, options=ROBUST)
    compositor.compose(frame, (replacement,), {1: points}, frame_index=1, options=ROBUST)
    assert compositor.last_reports[1]["temporal"]["reset"] is None
    compositor.compose(frame, (replacement,), {1: points}, frame_index=2, scene_cut=True, options=ROBUST)
    assert compositor.last_reports[1]["temporal"]["reset"] == "SCENE_CUT"
    compositor.compose(frame, (), {}, frame_index=3, options=ROBUST)
    assert compositor.occlusion_tracker.track_count == 0


def test_face_motion_is_measured_in_frame_pixels_not_from_the_stabilised_crop():
    frame = _scene()
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    compositor = _compositor(_StubNet("xseg_3.onnx", _visible()))
    for index, shift in enumerate((20, 26, 32)):                       # the face moves 6 px/frame to the right
        replacement = _replacement(1, (30, 200, 30), ((1.6, 0, shift), (0, 1.6, 20)))
        compositor.compose(frame, (replacement,), {1: points}, frame_index=index, options=ROBUST)
    motion = compositor.last_reports[1]["temporal"]["face_motion_px"]
    assert motion == pytest.approx([6.0, 0.0], abs=1e-3)


def test_partial_occlusion_does_not_flatten_or_recolour_the_swapped_face_interior():
    """Regression: decontamination must follow the geometric face edge, not the post-occlusion alpha.

    With half-transparent stripes over the face, the old behaviour treated the stripes as "edge", replaced their
    colour by the nearby opaque stripes and changed the interior by ~2 grey levels after blending (0.14 now).
    """
    from roop_3d.compositor.robust_fusion import RobustFusionOptions, fuse_robust

    frame = _scene()
    crop = texture(128, seed=21, sigma=0.8)
    inverse = np.array(((1.6, 0, 20), (0, 1.6, 20)), np.float32)
    face = cv2.GaussianBlur(cv2.circle(np.zeros((128, 128), np.float32), (64, 64), 50, 1.0, -1), (0, 0), 3.0)
    front = np.zeros((128, 128), np.float32)
    for x in range(0, 128, 16):
        front[:, x + 8:x + 16] = 0.45                                   # alternating clear / half-occluded stripes
    options = RobustFusionOptions(match_sharpness=False, texture_strength=0.0, linear_light=False, refine_matte=False)
    out, report = fuse_robust(frame, crop, inverse, face, front, None, options=options)
    size = (frame.shape[1], frame.shape[0])
    alpha = (cv2.warpAffine(face, inverse, size) * (1 - cv2.warpAffine(front, inverse, size)))[..., None]
    expected = (1 - alpha) * frame.astype(np.float32) + alpha * cv2.warpAffine(crop, inverse, size).astype(np.float32)
    centre = int(20 + 64 * 1.6)
    window = (slice(centre - 30, centre + 30), slice(centre - 30, centre + 30))
    assert 0.1 < report.occluded_fraction < 0.4
    assert np.abs(out[window].astype(np.float32) - expected[window]).mean() < 1.0


def test_face_alpha_fades_out_at_the_crop_border_instead_of_ending_in_a_hard_rectangle():
    """Real failure: a chin / forehead leaving the aligned crop produced a rectangular seam in the output."""
    fade = border_fade((256, 256))
    assert fade[0, 100] == 0 and fade[100, 0] == 0 and fade[128, 128] == 1.0
    assert 0 < fade[6, 128] < 1 and np.all(np.diff(fade[:20, 128]) >= 0)
    with pytest.raises(ValueError):
        border_fade((2, 2))
    frame = _scene()
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, 20), (0, 1.6, 20)))
    everything = np.array(((0, 0), (255, 0), (255, 255), (0, 255)), np.float32)      # a hull that reaches the crop border
    compositor = _compositor(_StubNet("xseg_3.onnx", lambda shape: np.ones(shape, np.float32)))
    out = compositor.compose(frame, (replacement,), {1: everything}, frame_index=0, options=ROBUST)
    painted = np.abs(out.astype(int) - frame.astype(int)).max(axis=2) > 12
    row = painted[20 + 100 * 1]                                                         # a frame row through the crop
    edge_strength = np.abs(out.astype(int) - frame.astype(int))[20 + 2, 20:20 + 200, 1].mean()   # 2 px inside the top edge
    middle_strength = np.abs(out.astype(int) - frame.astype(int))[20 + 100, 20 + 60:20 + 140, 1].mean()
    assert edge_strength < 0.3 * middle_strength and row.any()


def test_geometric_tier_still_composites_and_says_so_instead_of_faking_occlusion():
    frame = _scene()
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, 20), (0, 1.6, 20)))
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    compositor = FrameCompositor(parser=None, temporal=TemporalSmoother())
    compositor._analyzer = _analyser(None)
    out = compositor.compose(frame, (replacement,), {1: points}, frame_index=0, options=ROBUST)
    assert np.any(out != frame)
    report = compositor.last_reports[1]
    assert report["tier"] == "geometric" and "OCCLUSION_UNVERIFIED" in report["reasons"]
    assert report["fusion"]["occluded_fraction"] == 0.0


def test_fully_occluded_face_leaves_the_frame_untouched():
    frame = _scene()
    replacement = _replacement(1, (30, 200, 30), ((1.6, 0, 20), (0, 1.6, 20)))
    points = np.array(((40, 40), (210, 40), (210, 210), (40, 210)), np.float32)
    hidden = lambda shape: np.zeros(shape, np.float32)
    compositor = _compositor(_StubNet("xseg_3.onnx", hidden), _StubNet("face_occluder.onnx", hidden))
    out = compositor.compose(frame, (replacement,), {1: points}, frame_index=0, options=ROBUST)
    # allow only the feathered ring of the hull outside the net-interior weighting
    inner = out[70:170, 70:170]
    np.testing.assert_array_equal(inner, frame[70:170, 70:170])
    assert "FACE_MOSTLY_HIDDEN" in compositor.last_reports[1]["reasons"]


def test_workstation_parameter_plumbing_is_backward_compatible():
    default = WorkstationParameters()
    assert default.occlusion_mode == "standard" and "occlusion_mode" not in default.model_dump()
    assert default.composite_options().occlusion_mode == "standard"
    robust = WorkstationParameters(occlusion_mode="robust")
    assert robust.model_dump()["occlusion_mode"] == "robust"
    assert robust.composite_options().occlusion_mode == "robust"
    with pytest.raises(ValueError):
        WorkstationParameters(occlusion_mode="other")


def test_capability_reports_robust_components_only_after_they_exist():
    compositor = FrameCompositor(parser=_StubParser(), temporal=TemporalSmoother())
    assert "robust_occlusion" not in compositor.capability()
    assert compositor.capability()["occlusion_modes"] == {"default": "standard", "available": ["standard", "robust"]}
    compositor._analyzer = _analyser(_StubParser(), _StubNet("xseg_3.onnx", _visible()))
    assert "visible_face_networks" in compositor.capability()["robust_occlusion"]
