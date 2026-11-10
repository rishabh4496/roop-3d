"""StrandAwareOcclusionCompositor: depth z-test, N.V grazing falloff, semantic and object cues, and the directional strand matte.

The strand matte is verified against a synthetic scene with KNOWN strand pixels (anti-aliased 1.25 px curves hanging from a
hair region over noisy skin), plus the negative controls that matter more than the positive one: isotropic skin grain, a
skin line that touches no hair, and unrelated textured skin far from the hair must NOT be promoted to hair.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.compositor.face_regions import FaceRegions
from roop_3d.core.errors import DomainError
from roop_3d.rendering.strand_occlusion import (
    OcclusionInputs, StrandAwareOcclusionCompositor, StrandOcclusionOptions, gabor_bank, strand_evidence)

SKIN = np.array((150, 170, 205), np.float32)
HAIR = np.array((30, 40, 60), np.float32)
H = W = 256
ROWS = np.arange(H)[:, None] * np.ones((1, W))


def scene(seed=0, noise=3.0, strands=True, orphan=False, grain=14.0):
    """Skin with isotropic grain, hair on rows < 64, optional thin hair strands hanging over the face, optional orphan skin line."""
    rng = np.random.default_rng(seed)
    img = np.tile(SKIN, (H, W, 1))
    img += (cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2) * grain)[..., None]
    hair_map = np.zeros((H, W), np.float32)
    hair_map[:64] = 1.0
    hair_map = cv2.GaussianBlur(hair_map, (0, 0), 1.5)
    img = img * (1 - hair_map[..., None]) + HAIR * hair_map[..., None]
    big = np.zeros((H * 4, W * 4), np.uint8)
    if strands:
        for x0 in (60, 90, 120, 150, 185):
            pts = np.array([[x0 + 6 * np.sin(t / 18.0) + 0.1 * (t - 60) * (1 if x0 % 2 else -1), t] for t in range(58, 112)], np.float32)
            cv2.polylines(big, [np.round(pts * 4).astype(np.int32)], False, 255, 5, cv2.LINE_AA)
    strand_mask = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    orphan_mask = np.zeros((H, W), np.float32)
    if orphan:                                                   # a dark skin crease 74 px below the hair, touching nothing
        big2 = np.zeros((H * 4, W * 4), np.uint8)
        cv2.polylines(big2, [np.round(np.array([[80, 138], [110, 141], [150, 138], [180, 141]], np.float32) * 4).astype(np.int32)],
                      False, 255, 5, cv2.LINE_AA)
        orphan_mask = cv2.resize(big2, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    line = np.maximum(strand_mask, orphan_mask)
    img = img * (1 - line[..., None]) + HAIR * line[..., None]
    img += rng.normal(0, noise, img.shape)
    swap = np.zeros((H, W), np.float32)
    cv2.ellipse(swap, (128, 170), (95, 110), 0, 0, 360, 1.0, -1)
    swap = cv2.GaussianBlur(swap, (0, 0), 2)
    return np.clip(img, 0, 255).astype(np.uint8), swap, hair_map, strand_mask, orphan_mask


def regions_of(hair_map, **others):
    return FaceRegions({"hair": hair_map, **others}, hair_map.shape)


@pytest.fixture(scope="module")
def strand_scene():
    img, swap, hair_map, strand_mask, _ = scene()
    comp = StrandAwareOcclusionCompositor()
    return img, swap, hair_map, strand_mask, comp.evaluate(OcclusionInputs(img, swap, regions=regions_of(hair_map)))


# =========================================================================== strand matte
def test_strands_hanging_from_the_hairline_are_matted_and_skin_between_them_is_not(strand_scene):
    img, swap, hair_map, line, res = strand_scene
    strand = res.components["strand"]
    band = (ROWS > 72) & (ROWS < 108) & (swap > 0.9)
    on = (line > 0.5) & band
    between = (cv2.dilate((line > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0) & band
    far = (swap > 0.9) & (ROWS > 150)
    assert on.sum() > 150
    assert strand[on].mean() > 0.5 and strand[on].min() > 0.2          # measured on this scene: 0.64 mean, 0.34 min
    assert strand[between].max() < 0.02                                # measured: exactly 0
    assert strand[far].max() < 0.02                                    # grain far from the hair is never promoted
    # what that buys: the swap is withheld on the strands and kept on the skin between them
    assert res.alpha[on].mean() < 0.55 and res.alpha[between].mean() > 0.95
    assert res.report.strand_pixels > 100 and "strand" in res.report.cues_available


def test_strand_cue_is_what_removes_the_swap_there_not_the_parser(strand_scene):
    img, swap, hair_map, line, with_strands = strand_scene
    off = StrandAwareOcclusionCompositor(StrandOcclusionOptions(strand_enabled=False)).evaluate(
        OcclusionInputs(img, swap, regions=regions_of(hair_map)))
    on = (line > 0.5) & (ROWS > 72) & (ROWS < 108) & (swap > 0.9)
    assert off.components["strand"].max() == 0 and off.alpha[on].mean() > 0.95      # semantic hair alone leaves them painted over
    assert with_strands.alpha[on].mean() < off.alpha[on].mean() - 0.4


def test_isotropic_grain_next_to_hair_is_not_mistaken_for_strands():
    img, swap, hair_map, _, _ = scene(strands=False, grain=30.0, noise=8.0)
    res = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap, regions=regions_of(hair_map)))
    band = (ROWS > 66) & (ROWS < 100) & (swap > 0.9)
    assert res.components["strand"][band].mean() < 0.02 and res.components["strand"][band].max() < 0.25


def test_colour_gate_rejects_thin_bright_threads_that_are_not_hair():
    """A specular highlight or a light thread is just as thin and oriented as a strand, but lies AWAY from the hair colour."""
    dark_img, swap, hair_map, line, _ = scene()
    base, _, _, _, _ = scene(strands=False)
    weight = line[..., None]
    bright = np.clip(base.astype(np.float32) * (1 - weight) + 250.0 * weight, 0, 255).astype(np.uint8)
    on = (line > 0.5) & (ROWS > 72) & (ROWS < 108) & (swap > 0.9)
    gated = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(bright, swap, regions=regions_of(hair_map)))
    ungated = StrandAwareOcclusionCompositor(StrandOcclusionOptions(colour_gate=False)).evaluate(
        OcclusionInputs(bright, swap, regions=regions_of(hair_map)))
    dark = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(dark_img, swap, regions=regions_of(hair_map)))
    assert ungated.components["strand"][on].mean() > 0.5               # the ridge detector alone WOULD fire on the bright thread ...
    assert gated.components["strand"][on].mean() < 0.02                # ... the hair-colour line says it is not hair
    assert dark.components["strand"][on].mean() > 0.5                  # while the same geometry in hair colour is kept


def test_orientation_coherence_rejects_strong_isotropic_speckle_next_to_hair():
    """Strong speckle fires the ridge filter bank in every direction; only the structure tensor tells it from a line."""
    rng = np.random.default_rng(5)
    img, swap, hair_map, _, _ = scene(strands=False, grain=0.0, noise=0.0)
    z = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
    img = np.clip(img.astype(np.float32) + (z / z.std() * 90.0)[..., None], 0, 255).astype(np.uint8)
    band = (ROWS > 66) & (ROWS < 100) & (swap > 0.9)
    options = StrandOcclusionOptions()
    skin = np.clip(swap * (1 - hair_map), 0, 1)
    strand, maps = strand_evidence(img, hair_map, skin, options)
    assert maps["ridge"][band].mean() > options.strand_response_high             # the ridge response alone is above the firing level
    assert maps["coherence"][band].mean() < 0.4                                  # ... but the neighbourhood has no dominant direction
    assert strand[band].mean() < 0.02 and (strand[band] > 0.3).mean() < 0.01


def test_a_skin_line_that_touches_no_hair_is_never_promoted_but_the_attached_strands_are():
    img, swap, hair_map, line, orphan = scene(orphan=True)
    res = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap, regions=regions_of(hair_map)))
    strand = res.components["strand"]
    assert strand[(orphan > 0.5)].max() < 0.05                       # 74 px from the hair and connected to nothing
    assert strand[(line > 0.5) & (ROWS > 72) & (ROWS < 108) & (swap > 0.9)].mean() > 0.5
    # the same crease linked to the hair by a strand IS hair: hysteresis follows connectivity, not position
    img2 = img.copy()
    cv2.line(img2, (80, 64), (80, 138), tuple(int(c) for c in HAIR), 1, cv2.LINE_AA)
    attached = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img2, swap, regions=regions_of(hair_map)))
    assert attached.components["strand"][112:134, 78:83].max() > 0.3          # the far part of the link, > 50 px from the hair
    assert attached.components["strand"][orphan > 0.5].max() > 0.3             # and the crease it reaches
    # ... but only up to strand_max_length_px: a link longer than that is not followed
    short = StrandAwareOcclusionCompositor(StrandOcclusionOptions(strand_max_length_px=30.0)).evaluate(
        OcclusionInputs(img2, swap, regions=regions_of(hair_map)))
    assert short.components["strand"][112:134, 78:83].max() < 0.05


def test_strand_matte_needs_a_hair_prior_and_says_so():
    img, swap, _, _, _ = scene()
    res = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap))
    assert res.components["strand"].max() == 0 and "strand" not in res.report.cues_available
    res = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap, regions=regions_of(np.zeros((H, W), np.float32))))
    assert any("no hair" in note for note in res.report.notes)


def test_facial_hair_prior_drives_the_same_matte_on_the_lower_face():
    rng = np.random.default_rng(3)
    img = np.tile(SKIN, (H, W, 1)) + (cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2) * 10)[..., None]
    beard = np.zeros((H, W), np.float32)
    beard[196:] = 1.0                                              # stubble/beard zone at the bottom of the swap
    beard = cv2.GaussianBlur(beard, (0, 0), 1.5)
    img = img * (1 - beard[..., None]) + HAIR * beard[..., None]
    big = np.zeros((H * 4, W * 4), np.uint8)
    for x0 in (90, 128, 166):                                      # moustache-like strands reaching up from the beard
        pts = np.array([[x0 + 3 * np.sin(t / 9.0), t] for t in range(150, 200)], np.float32)
        cv2.polylines(big, [np.round(pts * 4).astype(np.int32)], False, 255, 5, cv2.LINE_AA)
    mask = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = np.clip(img * (1 - mask[..., None]) + HAIR * mask[..., None] + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    swap = np.zeros((H, W), np.float32)
    cv2.ellipse(swap, (128, 150), (80, 100), 0, 0, 360, 1.0, -1)
    comp = StrandAwareOcclusionCompositor()
    with_prior = comp.evaluate(OcclusionInputs(img, swap, facial_hair=beard))
    without = comp.evaluate(OcclusionInputs(img, swap))
    on = (mask > 0.5) & (ROWS > 156) & (ROWS < 192) & (swap > 0.9)
    assert on.sum() > 60 and with_prior.components["strand"][on].mean() > 0.4 and without.components["strand"].max() == 0


def test_gabor_bank_is_zero_mean_normalised_and_orientation_selective():
    thetas, kernels = gabor_bank(12, 4.0, 1.7)
    assert len(kernels) == 12 and thetas[0] == 0 and thetas[-1] < np.pi
    for k in kernels:
        assert abs(float(k.sum())) < 1e-5 and float(np.abs(k).sum()) == pytest.approx(1.0, abs=1e-5)
    # a vertical dark line is best matched by the kernel whose wave runs horizontally (theta ~ 0)
    img = np.ones((41, 41), np.float32)
    img[:, 20] = 0.0
    responses = [abs(float(cv2.filter2D(img, cv2.CV_32F, k)[20, 20])) for k in kernels]
    assert int(np.argmax(responses)) in (0, 11) and max(responses) > 3 * min(responses)


# =========================================================================== depth, grazing, semantic, object cues
def flat_inputs(lo=16, hi=48, **kw):
    """A featureless grey frame: the guide image carries no edges, so every value in `alpha` comes from the cues alone."""
    img = np.full((64, 64, 3), 128, np.uint8)
    swap = np.zeros((64, 64), np.float32)
    swap[lo:hi, lo:hi] = 1.0
    return OcclusionInputs(img, swap, **kw), swap


def test_grazing_falloff_follows_n_dot_v_between_the_two_thresholds():
    comp = StrandAwareOcclusionCompositor(StrandOcclusionOptions(grazing_cos_low=0.1, grazing_cos_high=0.5))
    cos = np.array([-0.3, 0.0, 0.1, 0.3, 0.5, 0.9])
    normals = np.stack([np.sqrt(1 - cos ** 2), np.zeros_like(cos), cos], axis=-1)
    f = comp.grazing_falloff(normals)
    assert f[0] == f[1] == f[2] == 0 and f[4] == f[5] == 1 and 0.4 < f[3] < 0.6
    assert np.all(np.diff(f) >= 0)                                   # monotone in N.V
    assert comp.grazing_falloff(np.zeros((2, 3)))[0] == 0              # a degenerate normal is faded, not trusted
    tilted = comp.grazing_falloff(np.array([[1.0, 0.0, 0.0]]), view_direction=(1.0, 0.0, 0.0))
    assert tilted[0] == 1                                            # same surface, camera looking along +x


def test_grazing_fades_the_swap_but_is_not_counted_as_an_occluder():
    normals = np.zeros((64, 64, 3), np.float32)
    cos = np.clip((np.arange(64) - 16) / 31.0, 0.0, 1.0)[None, :] * 0.9 * np.ones((64, 1))     # N.V = 0 at x=16 .. 0.9 at x=47
    normals[..., 2], normals[..., 0] = cos, np.sqrt(1 - cos ** 2)
    inputs, swap = flat_inputs(face_normals=normals)
    res = StrandAwareOcclusionCompositor().evaluate(inputs)
    row = res.alpha[32]
    assert row[17] < 0.05 and row[46] > 0.95 and np.all(np.diff(row[17:47]) >= -1e-3)   # faded at grazing, untouched face-on
    assert np.all(res.alpha <= swap + 1e-6) and res.occluder.max() == 0
    assert res.report.grazing_share > 0.05 and res.report.occluded_share == 0


def test_depth_test_uses_tolerance_and_ignores_unknown_depth():
    face = np.full((64, 64), 0.50, np.float32)
    occluder = np.full((64, 64), np.inf, np.float32)
    occluder[:, :22] = 0.46                                           # 40 mm in front  -> occluder
    occluder[:, 22:30] = 0.497                                        # 3 mm in front   -> inside the 6 mm tolerance
    occluder[:, 30:40] = 0.55                                         # behind the face -> not an occluder
    occluder[:, 40:44] = np.nan                                       # unknown
    inputs, _ = flat_inputs(face_depth=face, occluder_depth=occluder, lo=8, hi=56)
    res = StrandAwareOcclusionCompositor().evaluate(inputs)
    d = res.components["depth"]
    assert d[32, 12] == pytest.approx(1.0) and d[32, 26] == 0 and d[32, 35] == 0 and d[32, 42] == 0 and d[32, 60] == 0
    assert res.alpha[32, 12] < 0.05 and res.alpha[32, 26] > 0.95 and res.alpha[32, 35] > 0.95 and res.alpha[32, 42] > 0.95
    assert "depth" in res.report.cues_available


def test_depth_cue_without_face_depth_is_skipped_with_a_note():
    inputs, _ = flat_inputs(occluder_depth=np.full((64, 64), 0.1, np.float32))
    res = StrandAwareOcclusionCompositor().evaluate(inputs)
    assert res.components["depth"].max() == 0 and any("face_depth" in n for n in res.report.notes)


def test_semantic_cues_are_weighted_and_object_maps_pass_through():
    glasses = np.zeros((64, 64), np.float32)
    glasses[20:28, 16:48] = 1.0
    hat = np.zeros((64, 64), np.float32)
    hat[40:44, 16:48] = 0.5
    hair = np.zeros((64, 64), np.float32)
    hair[30:34, 16:48] = 1.0
    obj = np.zeros((64, 64), np.float32)
    obj[:, 44:48] = 1.0
    regions = FaceRegions({"glasses": glasses, "hat": hat, "hair": hair}, (64, 64))
    inputs, swap = flat_inputs(regions=regions, object_occluder=obj)
    res = StrandAwareOcclusionCompositor(StrandOcclusionOptions(hair_weight=0.5, strand_enabled=False)).evaluate(inputs)
    assert res.components["semantic"][24, 30] == pytest.approx(1.0)
    assert res.components["semantic"][42, 30] == pytest.approx(0.5)
    assert res.components["semantic"][32, 30] == pytest.approx(0.5)               # hair weight 0.5
    assert res.alpha[24, 30] < 0.05 and res.alpha[32, 30] == pytest.approx(0.5, abs=0.06) and res.alpha[36, 30] > 0.95
    assert res.alpha[36, 46] < 0.05                                                  # the hand/microphone map
    assert {"semantic", "object"} <= set(res.report.cues_available)


def test_union_of_cues_is_a_noisy_or_not_a_sum():
    regions = FaceRegions({"glasses": np.full((64, 64), 0.5, np.float32)}, (64, 64))
    obj = np.full((64, 64), 0.5, np.float32)
    inputs, _ = flat_inputs(regions=regions, object_occluder=obj)
    res = StrandAwareOcclusionCompositor().evaluate(inputs)
    assert res.occluder[32, 32] == pytest.approx(0.75, abs=1e-6)


def test_boundary_snap_moves_a_soft_matte_onto_a_real_image_edge_but_never_blurs_a_flat_guide():
    # an object matte whose soft edge sits 3 px left of the real edge in the picture (dark | light at x = 30)
    img = np.full((64, 64, 3), 200, np.uint8)
    img[:, :30] = 40
    ramp = np.clip((27.0 + 5.0 - np.arange(64)) / 10.0, 0.0, 1.0)            # 1 -> 0 across x = 22..32, centred on 27
    obj = np.tile(ramp.astype(np.float32), (64, 1))
    swap = np.zeros((64, 64), np.float32)
    swap[8:56, 8:56] = 1.0
    snapped = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap, object_occluder=obj)).alpha[32]
    plain = StrandAwareOcclusionCompositor(StrandOcclusionOptions(boundary_snap=False)).evaluate(
        OcclusionInputs(img, swap, object_occluder=obj)).alpha[32]

    def crossing(row):                        # sub-pixel x where alpha rises through 0.5
        i = int(np.nonzero((row[:-1] < 0.5) & (row[1:] >= 0.5))[0][0])
        return i + (0.5 - row[i]) / (row[i + 1] - row[i])
    assert abs(crossing(snapped) - 30.0) < abs(crossing(plain) - 30.0) - 0.5      # the transition moved toward the image edge
    window = slice(20, 36)                                                              # the matte's transition, clear of the swap's own edge
    assert np.abs(np.diff(snapped[window])).max() > 2.0 * np.abs(np.diff(plain[window])).max()   # and got sharper (steepest step)
    # flat guide: nothing to snap to, so a thin occluder keeps its exact computed value
    flat = np.full((64, 64, 3), 128, np.uint8)
    thin = np.zeros((64, 64), np.float32)
    thin[30:34, :] = 1.0
    a = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(flat, swap, object_occluder=thin)).alpha
    assert a[31:33, 20:44].max() < 1e-6 and a[40, 20:44].min() > 0.999


# =========================================================================== contract
def test_alpha_never_exceeds_the_geometric_coverage_and_is_local():
    img, swap, hair_map, _, _ = scene()
    comp = StrandAwareOcclusionCompositor()
    res = comp.evaluate(OcclusionInputs(img, swap, regions=regions_of(hair_map)))
    assert np.all(res.alpha <= swap * res.grazing + 1e-6) and res.alpha.min() >= 0 and res.alpha.max() <= 1
    # computed on the swap's bounding box: embedding the same frame in a larger canvas gives the same alpha inside it
    pad = 40
    img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    wide = comp.evaluate(OcclusionInputs(img_p, np.pad(swap, pad),
                                         regions=regions_of(np.pad(hair_map, pad, mode="edge"))))
    inner = wide.alpha[pad:-pad, pad:-pad]
    # (the swap is cut off by the bottom edge of the small canvas, where its filters replicate instead of seeing zeros: compare
    # away from the canvas border, which is the only place the two can legitimately differ)
    interior = np.zeros((H, W), bool)
    interior[8:-8, 8:-8] = True
    assert np.abs(inner - res.alpha)[interior].max() < 0.02
    assert np.abs(wide.components["strand"][pad:-pad, pad:-pad] - res.components["strand"]).max() < 0.02


def test_empty_swap_and_invalid_inputs():
    img = np.zeros((32, 32, 3), np.uint8)
    res = StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, np.zeros((32, 32), np.float32)))
    assert res.alpha.max() == 0 and res.report.roi is None and res.occlusion_in(np.ones((32, 32))) == 0.0
    comp = StrandAwareOcclusionCompositor()
    ok = np.ones((32, 32), np.float32)
    for bad in (OcclusionInputs(img.astype(np.float32), ok), OcclusionInputs(img, np.ones((31, 32), np.float32)),
                OcclusionInputs(img, ok * np.nan), OcclusionInputs(img, ok, face_normals=np.zeros((32, 32, 2), np.float32)),
                OcclusionInputs(img, ok, occluder_depth=np.zeros((5, 5), np.float32)),
                OcclusionInputs(img, ok, regions=FaceRegions({}, (8, 8))),
                OcclusionInputs(img, ok, view_direction=(0.0, 0.0, 0.0))):
        with pytest.raises(DomainError):
            comp.evaluate(bad)
    with pytest.raises(ValueError):
        StrandAwareOcclusionCompositor(StrandOcclusionOptions(grazing_cos_low=0.5, grazing_cos_high=0.4))
    with pytest.raises(ValueError):
        StrandOcclusionOptions(gabor_orientations=2)


def test_inputs_are_not_mutated():
    img, swap, hair_map, _, _ = scene()
    snapshot = (img.copy(), swap.copy(), hair_map.copy())
    StrandAwareOcclusionCompositor().evaluate(OcclusionInputs(img, swap, regions=regions_of(hair_map)))
    assert np.array_equal(img, snapshot[0]) and np.array_equal(swap, snapshot[1]) and np.array_equal(hair_map, snapshot[2])


def test_occlusion_in_region_reports_the_removed_share():
    obj = np.zeros((64, 64), np.float32)
    obj[:, 32:] = 1.0
    inputs, swap = flat_inputs(object_occluder=obj)
    res = StrandAwareOcclusionCompositor().evaluate(inputs)
    left, right = np.zeros((64, 64)), np.zeros((64, 64))
    left[:, :32], right[:, 32:] = 1, 1
    assert res.occlusion_in(left) == pytest.approx(0.0, abs=1e-3) and res.occlusion_in(right) == pytest.approx(1.0, abs=1e-3)
    assert 0.45 < res.occlusion_in(np.ones((64, 64))) < 0.55


def test_composite_keeps_the_target_on_strands_and_outside_the_swap_byte_identical(strand_scene):
    img, swap, hair_map, line, _ = strand_scene
    swapped = np.full_like(img, (60, 200, 60))                       # an unmistakable green face
    comp = StrandAwareOcclusionCompositor()
    inputs = OcclusionInputs(img, swap, regions=regions_of(hair_map))
    out, res = comp.composite(inputs, swapped)
    assert out.dtype == np.uint8 and out.shape == img.shape
    outside = cv2.dilate((swap > 1e-5).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    assert np.array_equal(out[outside], img[outside])
    on = (line > 0.5) & (ROWS > 72) & (ROWS < 108) & (swap > 0.9)
    between = (cv2.dilate((line > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0) & (ROWS > 72) & (ROWS < 108) & (swap > 0.9)
    # share of the way from the target pixel to the swap pixel (green channel): 1 = fully replaced, 0 = untouched
    weight = lambda m: ((out[m][:, 1].astype(float) - img[m][:, 1]) / (200.0 - img[m][:, 1])).mean()
    assert weight(between) > 0.9 and weight(on) < 0.6                  # strand pixels stay (mostly) the target's own hair
    with pytest.raises(DomainError):
        comp.composite(inputs, swapped[:, :-1])
