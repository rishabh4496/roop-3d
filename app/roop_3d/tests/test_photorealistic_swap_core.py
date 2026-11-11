"""Stage 17: numeric contracts of the photorealistic swap core and its primitives (synthetic scenes, no model weights)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.pipeline import image_ops as ops
from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore, SwapCoreOptions
from roop_3d.reconstruction.expression_tps import BLOCKS, layout_is_plausible
from roop_3d.tests.swap_core_support import landmarks106, make_scene


# ------------------------------------------------------------------------------------------------------ primitives
def test_synthetic_layout_is_plausible_and_head_frame_matches_construction():
    points = landmarks106(200, 200, 190, 200)
    assert layout_is_plausible(points)
    frame = ops.head_frame(points)
    assert frame.down[1] > 0.99 and abs(frame.across[0]) > 0.99
    assert frame.width == pytest.approx(190, rel=0.02)


def test_head_polygon_covers_jaw_and_reaches_above_the_brows():
    points = landmarks106(200, 200, 190, 200)
    polygon = ops.head_polygon(points, lift=0.42)
    brow_top = points[np.r_[BLOCKS["brow_a"], BLOCKS["brow_b"]]][:, 1].min()
    assert polygon[:, 1].min() < brow_top - 0.3 * 200 * 0.42
    mask = ops.polygon_mask(polygon, (400, 400))
    assert mask[200, 200] == 1.0 and mask[10, 10] == 0.0


def test_similarity_fit_recovers_rotation_scale_translation_exactly():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, (20, 2))
    angle, scale, shift = 0.4, 1.7, np.array([13.0, -7.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = scale * src @ rot.T + shift
    matrix = ops.similarity_fit(src, dst)
    assert np.allclose(ops.apply_affine(matrix, src), dst, atol=1e-8)
    with pytest.raises(ValueError):
        ops.similarity_fit(np.zeros((5, 2)), dst[:5])


def test_quotient_ratio_is_exactly_one_for_equal_lights_and_bounded_otherwise():
    rng = np.random.default_rng(2)
    shading = rng.uniform(0.2, 1.0, (64, 64, 3)).astype(np.float32)
    assert np.allclose(ops.quotient_ratio(shading, shading, sigma=0), 1.0, atol=1e-3)
    wild = ops.quotient_ratio(shading * 50, shading * 0.001, clip=(0.4, 2.2), sigma=0)
    assert wild.min() >= 0.4 - 1e-6 and wild.max() <= 2.2 + 1e-6               # float32 2.2 is a hair above the double


def test_quotient_ratio_is_smooth_so_it_cannot_carry_pore_detail():
    rng = np.random.default_rng(3)
    noisy_t = rng.uniform(0.4, 0.6, (96, 96, 3)).astype(np.float32)
    noisy_s = rng.uniform(0.4, 0.6, (96, 96, 3)).astype(np.float32)
    raw = ops.quotient_ratio(noisy_t, noisy_s, sigma=0)
    smooth = ops.quotient_ratio(noisy_t, noisy_s, sigma=4.0)
    assert ops.laplacian_variance(smooth[..., 1]) < 0.01 * ops.laplacian_variance(raw[..., 1])


def test_silhouette_normals_face_the_camera_in_the_middle_and_lean_outward_at_the_rim():
    mask = np.zeros((200, 200), np.float32)
    cv2.circle(mask, (100, 100), 70, 1.0, -1)
    normals = ops.silhouette_normals(mask)
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-4)
    assert normals[100, 100, 2] > 0.99
    assert normals[100, 165, 0] > 0.3 and normals[100, 35, 0] < -0.3          # +x on the right rim, -x on the left
    assert normals[5, 5, 2] == 1.0                                              # outside the mask: facing the camera


@pytest.mark.parametrize("size", [(240, 320), (241, 333)])
def test_laplacian_blend_reproduces_background_at_alpha_zero_and_foreground_at_one(size):
    rng = np.random.default_rng(4)
    fg = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    bg = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    zero = ops.laplacian_pyramid_blend(fg, bg, np.zeros(size, np.float32), levels=4)
    one = ops.laplacian_pyramid_blend(fg, bg, np.ones(size, np.float32), levels=4)
    assert np.abs(zero.astype(int) - bg).max() <= 1
    assert np.abs(one.astype(int) - fg).max() <= 1


def test_laplacian_blend_has_no_halo_across_a_soft_seam():
    h, w = 200, 200
    fg = np.full((h, w, 3), 200, np.uint8)
    bg = np.full((h, w, 3), 60, np.uint8)
    alpha = np.zeros((h, w), np.float32)
    cv2.circle(alpha, (100, 100), 50, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
    out = ops.laplacian_pyramid_blend(fg, bg, alpha, levels=4).astype(np.float32)[..., 0]
    assert out.min() >= 58 and out.max() <= 202                                # no ringing past either flat colour
    row = out[100, 20:100]
    assert (np.diff(row) >= -1.0).all()                                        # monotonic ramp into the face


def test_tps_flow_is_identity_for_equal_points_and_moves_only_the_requested_points():
    pts = np.array([[40, 40], [160, 40], [40, 160], [160, 160], [100, 100], [100, 40]], np.float64)
    mx, my, kept, det = ops.tps_flow(pts, pts, (200, 200))
    gx, gy = np.meshgrid(np.arange(200), np.arange(200))
    assert np.abs(mx - gx).max() < 0.6 and np.abs(my - gy).max() < 0.6 and kept == 1.0
    moved = pts.copy()
    moved[5] += (0, 14)
    mx, my, kept, det = ops.tps_flow(pts, moved, (200, 200))
    assert det >= 0.25
    # backward map: output pixel at the new location samples the old content
    assert my[int(moved[5, 1]), int(moved[5, 0])] == pytest.approx(pts[5, 1], abs=1.5)
    assert abs(mx[0, 0] - 0) < 0.5 and abs(my[0, 0] - 0) < 0.5                  # border pinned


def test_tps_flow_scales_a_folding_request_back_until_it_is_fold_free():
    pts = np.array([[60, 100], [140, 100], [100, 60], [100, 140], [100, 100]], np.float64)
    target = pts.copy()
    target[0] = (140, 100)                                                      # drags a point through its neighbour
    *_, kept, det = ops.tps_flow(pts, target, (200, 200), min_jacobian=0.25)
    assert kept < 1.0 and det >= 0.25 - 1e-3


def test_match_focus_only_softens_and_add_grain_is_seeded():
    rng = np.random.default_rng(5)
    sharp = rng.integers(60, 200, (120, 120, 3), dtype=np.uint8)
    alpha = np.ones((120, 120), np.float32)
    soft, sigma = ops.match_focus(sharp, alpha, ops.laplacian_variance(ops.luma(cv2.GaussianBlur(sharp, (0, 0), 1.0))))
    assert sigma > 0 and ops.laplacian_variance(ops.luma(soft)) < ops.laplacian_variance(ops.luma(sharp))
    same, none = ops.match_focus(soft, alpha, ops.laplacian_variance(ops.luma(sharp)) * 5)
    assert none == 0.0 and same is soft
    a, b = ops.add_grain(soft, alpha, 3.0, seed=7), ops.add_grain(soft, alpha, 3.0, seed=7)
    assert np.array_equal(a, b) and not np.array_equal(a, ops.add_grain(soft, alpha, 3.0, seed=8))


# --------------------------------------------------------------------------------------------------- the core, 4 steps
@pytest.fixture(scope="module")
def full_run():
    scene = make_scene()
    core = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=0.0), restorer=lambda crop, lm5: crop, restorer_name="identity-double")
    out, report = core.process_face(scene.frame, scene.packet, frame_index=3)
    return scene, out, report


def test_frame_is_untouched_outside_the_region_of_interest(full_run):
    scene, out, report = full_run
    x0, y0, x1, y1 = report.geometry["roi"]
    mask = np.ones(scene.frame.shape[:2], bool)
    mask[y0:y1, x0:x1] = False
    assert np.array_equal(out[mask], scene.frame[mask])
    assert not np.array_equal(out[y0:y1, x0:x1], scene.frame[y0:y1, x0:x1])


def test_step1_old_jaw_overhang_is_erased_to_the_true_background(full_run):
    scene, out, report = full_run
    assert report.geometry["mode"] == "landmark_silhouette" and report.geometry["max_contour_shift_px"] > 20
    silhouette = report.geometry["silhouette"]
    assert silhouette["applied"] and silhouette["inpaint"]["overhang_pixels"] > 500
    # pixels the old head covered but the source-locked skull does not: now background (tolerance covers inpaint + grain)
    locked = _locked_mask(scene, report)
    overhang = scene.target_head & ~(cv2.dilate(locked.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    overhang &= cv2.erode(scene.target_head.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert overhang.sum() > 300
    error = np.abs(out[overhang].astype(float) - scene.background[overhang].astype(float)).mean()
    old_error = np.abs(scene.frame[overhang].astype(float) - scene.background[overhang].astype(float)).mean()
    assert error < old_error and error < 50.0


def test_step2_swapped_face_reaches_the_source_locked_jaw_not_the_target_jaw(full_run):
    scene, out, report = full_run
    assert report.rectify["tps"]["applied"] and report.rectify["tps"]["min_jacobian"] >= 0.25
    assert report.rectify["patch"]["used"]
    centre = out[250 - 10: 250 + 10, 240 - 10: 240 + 10].astype(float).mean((0, 1))
    assert abs(centre - np.array([150, 172, 214])).max() < 30                       # the source's skin tone, not the target's


def test_step3_relight_follows_the_target_light_direction_without_blurring_pores():
    scene = make_scene(light_left=0.45)
    flat = Photorealistic3DSwapCore(SwapCoreOptions(relight_strength=0.0, skin_tone_strength=0.0, detail_gain=0.0, match_focus=False, match_grain=False))
    lit = Photorealistic3DSwapCore(SwapCoreOptions(relight_strength=1.0, skin_tone_strength=0.0, detail_gain=0.0, match_focus=False, match_grain=False))
    a, _ = flat.process_face(scene.frame, scene.packet)
    b, rep = lit.process_face(scene.frame, scene.packet)

    def left_right(image):
        luma = ops.luma(image)
        return luma[220:300, 170:210].mean() - luma[220:300, 270:310].mean()

    assert left_right(b) > left_right(a) + 6.0                                       # the left side got brighter relative to the right
    assert rep.relight["normals"] == "silhouette_ellipsoid" and rep.relight["inner"]["ratio_max"] > 1.02
    # pores: fine-scale energy of the face region is preserved by the smooth ratio
    region = (slice(225, 290), slice(215, 265))
    hf_a = ops.high_pass(ops.luma(a)[region], 1.2).std()
    hf_b = ops.high_pass(ops.luma(b)[region], 1.2).std()
    assert hf_b > 0.85 * hf_a


def test_step3_same_light_means_no_relight():
    scene = make_scene(light_left=0.0)
    base = Photorealistic3DSwapCore(SwapCoreOptions(relight_strength=0.0, skin_tone_strength=0.0, match_focus=False, match_grain=False, detail_gain=0.0))
    relit = Photorealistic3DSwapCore(SwapCoreOptions(relight_strength=1.0, skin_tone_strength=0.0, match_focus=False, match_grain=False, detail_gain=0.0))
    a, _ = base.process_face(scene.frame, scene.packet)
    b, rep = relit.process_face(scene.frame, scene.packet)
    assert rep.relight["inner"]["ratio_min"] > 0.8 and rep.relight["inner"]["ratio_max"] < 1.25
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 6.0


def test_step3_trust_gain_is_bounded_chromatic_only_and_scaled_by_illuminant_confidence():
    scene = make_scene()
    core = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=1.0, skin_tone_max_gain=1.15))
    _, report = core.process_face(scene.frame, scene.packet)
    trust = report.relight["trust"]
    assert trust["applied"] and all(1 / 1.15 - 1e-3 <= g <= 1.15 + 1e-3 for g in trust["gain_bgr"])
    assert float(np.asarray(trust["gain_bgr"]) @ ops.LUMA_BGR) == pytest.approx(1.0, abs=0.02)    # exposure is not TRUST's job
    assert 0.0 < trust["illuminant_confidence"] <= 1.0
    # an identical source and swap tone must not be "corrected" into a different one
    identical = make_scene()
    identical.packet.source_view = identical.swap_texture if False else identical.packet.source_view
    off = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=0.0, match_focus=False, match_grain=False)).process_face(scene.frame, scene.packet)[0]
    on = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=1.0, match_focus=False, match_grain=False)).process_face(scene.frame, scene.packet)[0]
    assert abs(ops.luma(on)[240:260, 230:250].mean() - ops.luma(off)[240:260, 230:250].mean()) < 6.0   # no brightness shift from TRUST


def test_nothing_is_written_outside_the_source_locked_outline_so_there_is_no_halo(full_run):
    scene, out, report = full_run
    locked = _locked_mask(scene, report)
    outside = ~(cv2.dilate(locked.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    x0, y0, x1, y1 = report.geometry["roi"]
    region = np.zeros_like(outside)
    region[y0:y1, x0:x1] = True
    ring = outside & region & (cv2.dilate(scene.target_head.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    # outside the new head the frame is background plate only: no skin-tinted glow left over from a feathered alpha
    glow = np.abs(out[ring].astype(float) - scene.background[ring].astype(float)).mean()
    assert glow < 14.0, glow


def test_silhouette_normals_have_no_medial_axis_crease_on_a_tall_head():
    mask = np.zeros((300, 300), np.float32)
    cv2.ellipse(mask, (150, 150), (80, 130), 0, 0, 360, 1.0, -1)
    normals = ops.silhouette_normals(mask)
    inside = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    jump = np.abs(np.diff(normals[..., 0], axis=1))[inside[:, 1:] & inside[:, :-1]].max()
    assert jump < 0.06, jump                                                       # a distance-transform ridge would jump here


def test_step4_restorer_runs_in_the_aligned_crop_and_is_blended_by_alpha():
    scene = make_scene()
    seen = {}

    def restorer(crop, lm5):
        seen["shape"], seen["lm5"] = crop.shape, np.asarray(lm5).shape
        return cv2.resize(np.full_like(crop, 255), (512, 512))                       # a restorer that paints everything white

    core = Photorealistic3DSwapCore(SwapCoreOptions(restoration_blend=0.5, skin_tone_strength=0.0, match_focus=False, match_grain=False),
                                    restorer=restorer, restorer_name="white-double")
    out, report = core.process_face(scene.frame, scene.packet)
    assert seen == {"shape": (128, 128, 3), "lm5": (5, 2)} and report.restore["ran"] and report.restore["native_restorer_px"] == 512
    centre = out[240:260, 230:250].astype(float).mean()
    reference = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=0.0, match_focus=False, match_grain=False)).process_face(scene.frame, scene.packet)[0]
    assert centre > reference[240:260, 230:250].astype(float).mean() + 40            # ~half way to white
    # a restorer must hand back a square uint8 image
    bad = Photorealistic3DSwapCore(restorer=lambda crop, lm5: crop.astype(np.float32))
    with pytest.raises(DomainError) as caught:
        bad.process_face(scene.frame, scene.packet)
    assert caught.value.code == "SWAP_CORE_RESTORER_INVALID"


def test_step4_occluder_keeps_the_target_pixels_in_front_of_the_face():
    scene = make_scene()
    occluder = np.zeros(scene.frame.shape[:2], np.float32)
    occluder[230:290, 200:250] = 1.0                                                 # a hand over the left cheek
    core = Photorealistic3DSwapCore(SwapCoreOptions(skin_tone_strength=0.0, match_focus=False, match_grain=False))
    packet = scene.packet
    packet.occluder = occluder
    out, report = core.process_face(scene.frame, packet)
    inside = (slice(240, 280), slice(210, 240))
    assert np.abs(out[inside].astype(int) - scene.frame[inside]).mean() < 4.0
    assert report.blend["occluder_applied"]


def test_step4_detail_injection_never_exceeds_the_source_photograph():
    scene = make_scene()
    soft = scene.packet
    soft.swapped_crop = cv2.GaussianBlur(soft.swapped_crop, (0, 0), 2.0)             # a soft swap: the photograph has more detail
    on = Photorealistic3DSwapCore(SwapCoreOptions(detail_gain=1.0, skin_tone_strength=0.0, match_focus=False, match_grain=False))
    off = Photorealistic3DSwapCore(SwapCoreOptions(detail_gain=0.0, skin_tone_strength=0.0, match_focus=False, match_grain=False))
    a, rep_on = on.process_face(scene.frame, soft)
    b, _ = off.process_face(scene.frame, soft)
    region = (slice(225, 290), slice(215, 265))
    hf = lambda image: ops.high_pass(ops.luma(image)[region], 1.4).std()           # noqa: E731
    assert rep_on.restore["detail"]["applied"] and hf(a) > hf(b)
    assert hf(a) <= rep_on.restore["detail"]["source_hf"] * 1.15


def test_inner_face_scope_leaves_the_silhouette_alone():
    scene = make_scene()
    core = Photorealistic3DSwapCore(SwapCoreOptions(swap_scope="inner_face", skin_tone_strength=0.0))
    out, report = core.process_face(scene.frame, scene.packet)
    assert report.geometry["mode"] == "skipped:SCOPE_KEEPS_TARGET_SILHOUETTE" and "silhouette" not in report.geometry
    assert not report.rectify["tps"]["applied"] and not report.rectify["patch"]["used"]
    assert any(item.startswith("geometry:") for item in report.fallbacks)


def test_missing_source_geometry_degrades_and_says_so():
    scene = make_scene()
    scene.packet.source_landmarks = None
    out, report = Photorealistic3DSwapCore().process_face(scene.frame, scene.packet)
    assert report.geometry["mode"] == "skipped:NO_SOURCE_GEOMETRY"
    assert "geometry:NO_SOURCE_GEOMETRY" in report.fallbacks and "restore:NO_RESTORER" in report.fallbacks


def test_adaptive_skull_lock_interpolates_between_target_and_source_jaw():
    scene = make_scene()
    shifts = {}
    for blend in (0.0, 0.5, 1.0):
        core = Photorealistic3DSwapCore(SwapCoreOptions(swap_scope="adaptive_skull", skull_blend=blend, skin_tone_strength=0.0))
        _, report = core.process_face(scene.frame, scene.packet)
        shifts[blend] = report.geometry["max_contour_shift_px"]
    assert shifts[0.0] < 0.5 < shifts[0.5] < shifts[1.0]
    assert shifts[0.5] == pytest.approx(shifts[1.0] / 2, rel=0.15)


def test_implausible_target_layout_is_refused_not_guessed():
    scene = make_scene()
    scene.packet.target_landmarks = scene.packet.target_landmarks[::-1].copy()
    with pytest.raises(DomainError) as caught:
        Photorealistic3DSwapCore().process_face(scene.frame, scene.packet)
    assert caught.value.code == "SWAP_CORE_LANDMARKS_INVALID"


def test_process_face_is_deterministic_and_does_not_mutate_its_input():
    scene = make_scene()
    original = scene.frame.copy()
    core = Photorealistic3DSwapCore()
    a, _ = core.process_face(scene.frame, scene.packet, frame_index=5)
    b, _ = core.process_face(scene.frame, scene.packet, frame_index=5)
    assert np.array_equal(a, b) and np.array_equal(scene.frame, original)


def _locked_mask(scene, report):
    """Rebuild the source-locked outline from the report's own inputs (independent of the core's private state)."""
    from roop_3d.pipeline.image_ops import apply_affine, forehead_arc, head_polygon, polygon_mask, similarity_fit, inner_anchor_indices
    target = scene.packet.target_landmarks
    source = scene.packet.source_landmarks
    anchors = inner_anchor_indices()
    p_src = apply_affine(similarity_fit(source[anchors], target[anchors]), source)
    return polygon_mask(head_polygon(p_src, 0.42), scene.frame.shape[:2]) > 0.5
