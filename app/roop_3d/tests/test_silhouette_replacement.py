"""Stage 6B: silhouette replacement. A different head shape must not leave the target's old jaw behind (narrower source) or
collide with its hair / ears (wider source).

Scenes are synthetic on purpose: the true background behind the head is known exactly, so "the overhang was filled with
background" is a measurement, not an opinion. Nothing here claims behaviour on real footage."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.rendering.perimeter_cage import (CageOptions, PerimeterCageWarper, mesh_perimeter_pairs, perimeter_from_masks,
                                              perimeter_vertex_ids)
from roop_3d.rendering.silhouette_inpaint import DonorFrame, InpaintOptions, SilhouetteDeltaInpainter, overhang_ring
from roop_3d.rendering.silhouette_replacement import (DonorHistory, SWAP_SCOPES, SilhouetteReplacer, resolve_scope)

SIDE = 160
CENTRE = (80, 80)
SKIN = (180, 140, 100)
HAIR = (30, 30, 40)


_NOISE = cv2.GaussianBlur(np.random.default_rng(11).normal(0, 1, (SIDE + 32, SIDE + 32)).astype(np.float32), (0, 0), 2.5)
_NOISE /= _NOISE.std()


def background(side: int = SIDE, shift: tuple[int, int] = (0, 0), texture: float = 0.0) -> np.ndarray:
    """Smooth gradient plus a gentle ripple: predictable enough to continue. ``texture`` adds blurred noise (fixed in the world,
    so a camera ``shift`` moves it) - enough structure for the background-motion fit to lock onto."""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    xx, yy = xx + shift[0], yy + shift[1]
    base = np.stack((60 + 0.35 * xx + 0.1 * yy, 90 + 0.2 * xx + 0.3 * yy, 120 + 0.1 * xx + 0.25 * yy), -1)
    base += 10 * np.sin(xx / 11.0)[..., None] * np.array([1.0, 0.7, 0.4])
    if texture:
        base += texture * _NOISE[16 + shift[1]:16 + shift[1] + side, 16 + shift[0]:16 + shift[0] + side][..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def ellipse(axes: tuple[int, int], centre: tuple[int, int] = CENTRE, side: int = SIDE) -> np.ndarray:
    plane = np.zeros((side, side), np.uint8)
    cv2.ellipse(plane, centre, axes, 0, 0, 360, 1, -1)
    return plane.astype(np.float32)


def hair_ring(axes: tuple[int, int], thickness: int = 10) -> np.ndarray:
    """Hair / ear structure hugging the head on its top and flanks (not below the cheek line)."""
    outer = ellipse((axes[0] + thickness, axes[1] + thickness))
    ring = np.clip(outer - ellipse(axes), 0, 1)
    ring[CENTRE[1] + 28:] = 0
    return ring


def head_scene(target_axes, hair_axes=None, side: int = SIDE):
    bg = background(side)
    frame = bg.copy()
    target = ellipse(target_axes)
    hair = hair_ring(hair_axes or target_axes)
    frame[hair > 0] = HAIR
    frame[target > 0] = SKIN
    return bg, frame, target, hair


# ================================================================ overhang ring
def test_overhang_is_dilated_target_minus_source():
    target, source = ellipse((46, 52)), ellipse((30, 52))
    ring = overhang_ring(target, source, dilate_px=4, source_overlap_px=0)
    assert ring[80, 36] and ring[80, 124]                                           # old cheek pixels, both sides
    assert not ring[80, 80] and not ring[80, 70]                                    # inside the new face
    assert not ring[5, 5]                                                           # far background
    assert ring[80, 32] and not ring[80, 20]                                        # the dilated rim, but only 4 px of it
    assert not (ring & (source > 0.5)).any()
    # With overlap the hole runs under the new face's edge, never past it by more than that overlap.
    wider = overhang_ring(target, source, dilate_px=4, source_overlap_px=3)
    assert wider.sum() > ring.sum() and (wider & ~ring).sum() > 0


def test_overhang_is_empty_when_source_covers_target():
    assert not overhang_ring(ellipse((30, 40)), ellipse((46, 52)), dilate_px=0, source_overlap_px=0).any()


def test_masks_must_match_frame():
    with pytest.raises(DomainError):
        SilhouetteDeltaInpainter().fill(background(), np.zeros((10, 10)), np.zeros((SIDE, SIDE)))


# ================================================================ tier (a): temporal donors
def _wide_to_slim():
    """Bare head on a known background (no hair: the inpainter alone is under test; the replacer excludes hair itself)."""
    bg = background()
    target, source = ellipse((46, 52)), ellipse((30, 52))
    frame = bg.copy()
    frame[target > 0] = SKIN
    return bg, frame, target, source, np.zeros_like(target)


def _overhang_core(target, source):
    """Interior of the overhang (away from its rim) where the true background is known and nothing else may remain."""
    ring = overhang_ring(target, source, dilate_px=0, source_overlap_px=0)
    return ring & (cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) \
        & ~(cv2.dilate(source.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)


def test_temporal_donor_recovers_the_real_background_behind_the_old_jaw():
    bg, frame, target, source, _ = _wide_to_slim()
    # Neighbouring frame: the head has moved away (a nod), so the pixels the old jaw hides are visible there.
    donor_head = ellipse((46, 52), centre=(80, -60))
    donor = bg.copy()
    donor[donor_head > 0] = SKIN
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(donor, donor_head)])
    core = _overhang_core(target, source)
    assert core.sum() > 500
    error = np.abs(out.astype(int) - bg.astype(int)).sum(-1)[core]
    assert error.max() <= 6, "the overhang must show the TRUE background, not a guess"
    assert report.tier == "temporal" and report.temporal_pixels > 0 and report.spatial_pixels == 0
    assert report.donors_used == 1 and report.overhang_pixels == report.filled_pixels
    assert not np.any(np.all(out[core] == SKIN, axis=-1)), "old facial skin must be gone"


def test_pixels_no_donor_saw_fall_through_to_the_spatial_tier():
    bg, frame, target, source, _ = _wide_to_slim()
    donor_head = ellipse((46, 52), centre=(80, 8))                      # the donor's head still hides the upper part of our jaw
    donor = bg.copy()
    donor[donor_head > 0] = SKIN
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(donor, donor_head)])
    assert report.tier == "temporal+spatial" and report.temporal_pixels > 0 and report.spatial_pixels > 0
    assert report.temporal_pixels + report.spatial_pixels == report.overhang_pixels
    seen_by_donor = _overhang_core(target, source) & (donor_head == 0) & (cv2.dilate(donor_head.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0)
    assert np.abs(out.astype(int) - bg.astype(int)).sum(-1)[seen_by_donor].max() <= 6
    assert not np.any(np.all(out[_overhang_core(target, source)] == SKIN, axis=-1))


def test_temporal_donor_is_registered_through_the_background_motion():
    bg = background(texture=28)
    target, source = ellipse((46, 52)), ellipse((30, 52))
    frame = bg.copy()
    frame[target > 0] = SKIN
    # The camera panned 3 px right / 2 px down between the frames: a donor pasted without registration is visibly displaced.
    donor = background(shift=(3, 2), texture=28)
    donor_head = np.zeros((SIDE, SIDE), np.float32)
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(donor, donor_head)])
    core = _overhang_core(target, source)
    error = np.abs(out.astype(int) - bg.astype(int)).sum(-1)[core]
    naive = np.abs(donor.astype(int) - bg.astype(int)).sum(-1)[core]
    assert report.donors_used == 1 and report.tier == "temporal"
    assert np.median(naive) >= 12, "the scene must make a missing registration visible"
    assert np.median(error) <= 6 and np.median(error) < 0.5 * np.median(naive)


def test_donor_exposure_drift_is_corrected_on_the_surround():
    bg, frame, target, source, _ = _wide_to_slim()
    donor = np.clip(bg.astype(int) + 9, 0, 255).astype(np.uint8)         # the donor frame was a little brighter
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(donor, None)])
    # donors without a head mask are only usable where the current surround reproduces them: here they do, after the bias
    core = _overhang_core(target, source)
    assert np.abs(out.astype(int) - bg.astype(int)).sum(-1)[core].max() <= 9
    assert report.donors_used == 1


def test_unrelated_donor_is_rejected_and_the_spatial_tier_takes_over():
    bg, frame, target, source, _ = _wide_to_slim()
    rng = np.random.default_rng(3)
    wrong = rng.integers(0, 255, bg.shape, dtype=np.uint8)
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(wrong, None)])
    assert report.donors_used == 0 and report.tier == "spatial" and report.spatial_pixels > 0
    assert sum(report.rejected.values()) == 1
    core = _overhang_core(target, source)
    assert not np.any(np.all(out[core] == SKIN, axis=-1))


def test_donor_never_donates_where_it_had_the_head_itself():
    bg, frame, target, source, _ = _wide_to_slim()
    donor = bg.copy()
    donor[target > 0] = (0, 0, 255)                                      # the donor's head sits exactly where ours does
    out, report = SilhouetteDeltaInpainter().fill(frame, target, source, donors=[DonorFrame(donor, target)])
    assert not np.any(np.all(out[_overhang_core(target, source)] == (0, 0, 255), axis=-1))
    assert report.tier != "temporal"


# ================================================================ tier (b): spatial fill
def test_spatial_fill_continues_the_background_and_never_uses_skin():
    bg, frame, target, source, _ = _wide_to_slim()
    out, report = SilhouetteDeltaInpainter(InpaintOptions(grain_strength=0)).fill(frame, target, source)
    core = _overhang_core(target, source)
    error = np.abs(out.astype(int) - bg.astype(int)).sum(-1)[core]
    assert report.tier == "spatial" and report.spatial_method == "multiscale_ns" and report.donors_offered == 0
    assert error.mean() <= 24, "a smooth background must be continued, not smeared with skin"
    assert not np.any(np.all(out[core] == SKIN, axis=-1))
    # outside the ring nothing moves
    far = (cv2.dilate(target.astype(np.uint8), np.ones((31, 31), np.uint8)) == 0)
    np.testing.assert_array_equal(out[far], frame[far])


def test_spatial_fill_continues_a_background_edge():
    bg = background()
    bg[:, 90:] = (200, 60, 60)                                           # a hard vertical edge to the right of the head
    frame = bg.copy()
    target, source = ellipse((46, 52)), ellipse((30, 52))
    frame[target > 0] = SKIN
    out, _ = SilhouetteDeltaInpainter().fill(frame, target, source)
    # Left of the edge the fill is the gradient, right of it the red: the edge column must not bleed across (>8 px away from it).
    right = _overhang_core(target, source) & (np.arange(SIDE)[None, :] >= 97)
    left = _overhang_core(target, source) & (np.arange(SIDE)[None, :] <= 83)
    assert right.any() and left.any()
    assert np.abs(out.astype(int)[right] - np.array([200, 60, 60])).sum(-1).mean() < 40
    assert np.abs(out.astype(int)[left] - bg.astype(int)[left]).sum(-1).mean() < 40


def test_occluder_is_never_painted_over_or_used():
    bg, frame, target, source, _ = _wide_to_slim()
    front = np.zeros((SIDE, SIDE), np.float32)
    front[70:90, 30:60] = 1
    frame[front > 0] = (10, 220, 10)
    out, _ = SilhouetteDeltaInpainter().fill(frame, target, source, occluder=front)
    np.testing.assert_array_equal(out[front > 0], frame[front > 0])
    assert not np.any(np.all(out[(target > 0) & (source == 0) & (front == 0)] == (10, 220, 10), axis=-1))


def test_neural_inpainter_is_used_and_a_failure_falls_back():
    bg, frame, target, source, _ = _wide_to_slim()
    seen = {}

    def neural(image, mask):
        seen["mask_fraction"] = float((mask > 0).mean())
        return np.full_like(image, (7, 7, 7))

    out, report = SilhouetteDeltaInpainter(neural_inpainter=neural).fill(frame, target, source)
    assert report.spatial_method == "neural" and seen["mask_fraction"] > 0
    assert np.all(out[_overhang_core(target, source)] == (7, 7, 7))

    def broken(image, mask):
        raise RuntimeError("no weights")

    out, report = SilhouetteDeltaInpainter(neural_inpainter=broken).fill(frame, target, source)
    assert report.spatial_method == "multiscale_ns" and "NEURAL_INPAINTER_FAILED" in report.notes[0]


def test_no_overhang_is_a_no_op():
    bg, frame, _, _ = head_scene((30, 52))
    # (With the default 6 px dilation the poles of a same-height target still leave a thin ring; here the rim is off.)
    inpainter = SilhouetteDeltaInpainter(InpaintOptions(dilate_px=0, source_overlap_px=0))
    out, report = inpainter.fill(frame, ellipse((30, 52)), ellipse((46, 52)))
    np.testing.assert_array_equal(out, frame)
    assert report.overhang_pixels == 0 and report.tier == "none"


def test_fully_excluded_frame_is_reported_not_invented():
    frame = background()
    everything = np.ones((SIDE, SIDE), np.float32)
    with pytest.raises(DomainError) as error:
        SilhouetteDeltaInpainter().fill(frame, everything, np.zeros_like(everything))
    assert error.value.code == "BACKGROUND_DONORS_UNAVAILABLE"


# ================================================================ cage warp
def test_cage_moves_the_perimeter_onto_the_source_and_leaves_the_far_field_alone():
    warper = PerimeterCageWarper()
    # wide target -> slim source: content at the target outline must land on the slim outline
    src, tgt = perimeter_from_masks(ellipse((30, 52)), ellipse((46, 52)))
    field = warper.estimate(src, tgt, (SIDE, SIDE))
    assert not field.identity and field.report.fold_scale > 0 and field.report.min_jacobian >= 0.25
    # a marker at the target's left outline ends up at the source's left outline (within a few px)
    marker = np.zeros((SIDE, SIDE), np.float32)
    marker[78:83, 80 - 46 - 2:80 - 46 + 3] = 1
    moved = warper.warp(field, marker)
    ys, xs = np.nonzero(moved > 0.5)
    assert abs(xs.mean() - (80 - 30)) < 6
    # the far corners do not move at all
    assert np.abs(field.dx[:12, :12]).max() < 0.5 and np.abs(field.dy[-12:, -12:]).max() < 0.5
    assert field.report.control_residual_px < 0.25 * field.report.max_displacement_px + 2


def test_cage_is_identity_when_the_silhouettes_agree():
    pts, same = perimeter_from_masks(ellipse((40, 52)), ellipse((40, 52)))
    field = PerimeterCageWarper().estimate(pts, same, (SIDE, SIDE))
    assert field.identity and field.report.reason == "SILHOUETTES_ALREADY_MATCH"
    image = background()
    np.testing.assert_array_equal(PerimeterCageWarper().warp(field, image), image)


def test_cage_never_folds():
    # Wildly contradictory correspondences (neighbours sent in opposite directions) would fold an unconstrained interpolant.
    rng = np.random.default_rng(1)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.column_stack((80 + 40 * np.cos(angles), 80 + 40 * np.sin(angles)))
    jitter = ring + rng.normal(0, 14, ring.shape)
    field = PerimeterCageWarper(CageOptions(max_displacement_fraction=0.6)).estimate(ring, jitter, (SIDE, SIDE))
    if not field.identity:
        h, w = field.dx.shape
        det = (1 + np.gradient(field.dx, axis=1)) * (1 + np.gradient(field.dy, axis=0)) \
            - np.gradient(field.dx, axis=0) * np.gradient(field.dy, axis=1)
        assert det.min() > 0.05, "a kept field must be fold-free"
    else:
        assert field.report.reason == "CAGE_WOULD_FOLD"


def test_cage_validates_inputs():
    warper = PerimeterCageWarper()
    with pytest.raises(DomainError):
        warper.estimate(np.zeros((8, 2)), np.zeros((7, 2)), (SIDE, SIDE))
    with pytest.raises(DomainError):
        warper.estimate(np.full((8, 2), np.nan), np.zeros((8, 2)), (SIDE, SIDE))
    assert warper.estimate(np.zeros((2, 2)), np.ones((2, 2)), (SIDE, SIDE)).report.reason == "TOO_FEW_PERIMETER_POINTS"


def test_mesh_perimeter_is_the_boundary_ring_of_the_region_and_projects_pairwise():
    # a 5x5 vertex grid, region = the inner 3x3 vertices: its boundary ring is the 8 outer vertices of the region
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    verts = np.column_stack((xs.ravel(), ys.ravel()))
    faces = []
    for r in range(4):
        for c in range(4):
            a = r * 5 + c
            faces += [(a, a + 1, a + 5), (a + 1, a + 6, a + 5)]
    weights = np.zeros(25)
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            weights[r * 5 + c] = 1
    ids = perimeter_vertex_ids(weights, np.asarray(faces))
    assert sorted(ids.tolist()) == [6, 7, 8, 11, 13, 16, 17, 18]
    source_px, target_px = verts * 2.0, verts * 3.0
    s, t = mesh_perimeter_pairs(ids, source_px, target_px, visible=np.ones(25, bool))
    np.testing.assert_allclose(t, s * 1.5)
    s2, _ = mesh_perimeter_pairs(ids, source_px, target_px, visible=np.arange(25) != 6)
    assert len(s2) == 7


def test_perimeter_from_masks_orders_source_inside_target_for_a_slimmer_source():
    src, tgt = perimeter_from_masks(ellipse((30, 52)), ellipse((46, 52)))
    assert len(src) == len(tgt) >= 24
    centre = np.array(CENTRE, float)
    left = np.abs(np.arctan2(*(tgt - centre)[:, ::-1].T)) > 2.9                  # rays pointing left/right
    assert (np.linalg.norm(src - centre, axis=1)[left] < np.linalg.norm(tgt - centre, axis=1)[left]).all()


# ================================================================ replacer
def _near_hair(image: np.ndarray) -> np.ndarray:
    """Hair pixels and their anti-aliased rim (anything more than ~a quarter hair), grown by 2 px."""
    hairish = (np.abs(image.astype(int) - HAIR).sum(-1) < 200).astype(np.uint8)
    return cv2.dilate(hairish, np.ones((5, 5), np.uint8)) > 0


def _hair_cols(image: np.ndarray, rows=slice(30, 80)) -> np.ndarray:
    dark = np.abs(image.astype(int) - HAIR).sum(-1) < 20
    return np.nonzero(dark[rows].any(axis=0))[0]


def test_replacer_pulls_hair_in_and_clears_the_old_jaw_for_a_narrower_source():
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((30, 52))
    structure = hair
    out, changed, report = SilhouetteReplacer().apply(frame, target_mask=target, source_mask=source,
                                                      plan=resolve_scope("full_craniofacial"), structure_mask=structure)
    before, after = _hair_cols(frame), _hair_cols(out)
    assert after.min() > before.min() + 6 and after.max() < before.max() - 6, "hair must contract toward the slimmer skull"
    assert report.applied and report.perimeter_source == "silhouettes" and not report.cage.identity
    assert report.inpaint.overhang_pixels > 500 and report.vacated_pixels > 0
    skin = np.all(out == SKIN, axis=-1) & ~(cv2.dilate(source.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    assert skin.sum() == 0, "no old facial skin may remain outside the new face"
    # the vacated hair area shows background, not leftover hair
    old_hair_left = (hair > 0) & (np.arange(SIDE)[None, :] < 80 - 46 - 4) & (np.arange(SIDE)[:, None] > 50) & (np.arange(SIDE)[:, None] < 80)
    assert not np.any(np.all(np.abs(out.astype(int) - HAIR).sum(-1)[..., None] < 20, axis=-1) & old_hair_left)
    assert changed.max() == 1 and changed[0, 0] == 0
    np.testing.assert_array_equal(out[:6, :6], frame[:6, :6])


def test_replacer_pushes_hair_out_for_a_wider_source():
    bg, frame, target, hair = head_scene((30, 52))
    source = ellipse((46, 52))
    out, _, report = SilhouetteReplacer().apply(frame, target_mask=target, source_mask=source,
                                                plan=resolve_scope("full_craniofacial"), structure_mask=hair)
    before, after = _hair_cols(frame), _hair_cols(out)
    assert after.min() < before.min() - 8 and after.max() > before.max() + 8, "hair must expand around the wider skull"
    # (only the thin rim the 6 px dilation leaves at the poles of two equally tall heads, never a cheek / jaw overhang)
    assert report.inpaint.overhang_pixels < 0.02 * SIDE * SIDE and not report.cage.identity
    flank = (target > 0) & (np.abs(np.arange(SIDE)[None, :] - CENTRE[0]) > 20)
    assert (report.inpaint.overhang_pixels == 0) or not np.any(flank & (np.abs(out.astype(int) - frame.astype(int)).sum(-1) > 0) & (source == 0))


def test_replacer_leaves_the_frame_alone_for_the_legacy_scope():
    bg, frame, target, hair = head_scene((46, 52))
    out, changed, report = SilhouetteReplacer().apply(frame, target_mask=target, source_mask=ellipse((30, 52)),
                                                      plan=resolve_scope("inner_face"), structure_mask=hair)
    np.testing.assert_array_equal(out, frame)
    assert not changed.any() and not report.applied and report.reason == "SCOPE_KEEPS_TARGET_SILHOUETTE"


def test_replacer_keeps_foreground_occluders_fixed():
    bg, frame, target, hair = head_scene((46, 52))
    front = np.zeros((SIDE, SIDE), np.float32)
    front[40:60, 36:60] = 1
    frame[front > 0] = (10, 220, 10)
    out, _, _ = SilhouetteReplacer().apply(frame, target_mask=target, source_mask=ellipse((30, 52)),
                                           plan=resolve_scope("full_craniofacial"), structure_mask=hair, occluder=front)
    np.testing.assert_array_equal(out[front > 0], frame[front > 0])


def test_replacer_honours_explicit_mesh_perimeter_pairs():
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((30, 52))
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    tgt = np.column_stack((80 + 56 * np.cos(angles), 80 + 62 * np.sin(angles)))
    src = np.column_stack((80 + 40 * np.cos(angles), 80 + 62 * np.sin(angles)))
    _, _, report = SilhouetteReplacer().apply(frame, target_mask=target, source_mask=source, plan=resolve_scope("full_craniofacial"),
                                              structure_mask=hair, perimeter=(src, tgt))
    assert report.perimeter_source == "mesh" and not report.cage.identity


# ================================================================ scope + history
def test_scope_plans():
    full = resolve_scope()
    assert (full.scope, full.source_identity_lock, full.replace_silhouette, full.confine_to_target) == ("full_craniofacial", 1.0, True, False)
    adaptive = resolve_scope("adaptive_skull", 0.35)
    assert (adaptive.source_identity_lock, adaptive.replace_silhouette) == (0.35, True)
    legacy = resolve_scope("inner_face", 0.9)
    assert (legacy.source_identity_lock, legacy.replace_silhouette, legacy.confine_to_target) == (0.0, False, True)
    assert SWAP_SCOPES[0] == "full_craniofacial"                         # the default is listed first
    for bad in (("nope", 1.0), ("adaptive_skull", 1.5), ("adaptive_skull", float("nan"))):
        with pytest.raises(DomainError):
            resolve_scope(*bad)


def test_donor_history_keeps_recent_frames_and_ignores_seeks_and_replays():
    history = DonorHistory(depth=2)
    frame = background()
    for index in (10, 11, 12):
        history.push(7, frame + index % 3, np.zeros((SIDE, SIDE)), index)
    assert len(history.donors(7, (SIDE, SIDE))) == 2                      # depth 2: the oldest was dropped
    near = history.donors(7, (SIDE, SIDE), frame_index=13, max_gap=4)
    assert len(near) == 2
    assert history.donors(7, (SIDE, SIDE), frame_index=12, max_gap=4).__len__() == 1      # the frame itself never donates to itself
    assert history.donors(7, (SIDE, SIDE), frame_index=400, max_gap=4) == []              # a seek
    assert history.donors(7, (SIDE + 1, SIDE)) == []
    history.push(7, frame, None, 12)                                                      # re-render replaces, not duplicates
    assert len(history.donors(7, (SIDE, SIDE))) == 2
    history.retain({1})
    assert history.donors(7, (SIDE, SIDE)) == []


# ================================================================ end to end through the real compositor
def _compose(frame, target, source, hair, scope="full_craniofacial", skull_blend=1.0, history=(), background_donors=None):
    """``history`` = earlier frames as (image, target, source, hair) tuples, each with ITS OWN masks, composed first by the same
    compositor so it accumulates its donor history exactly as a video job would."""
    from roop_3d.compositor.occlusion_analysis import FaceAnalysis
    from roop_3d.compositor.face_regions import FaceRegions
    from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor, RobustOptions
    zeros = np.zeros((SIDE, SIDE), np.float32)

    current = {}

    class Analyzer:
        def retain(self, tracks): pass
        def release(self): pass
        def analyze(self, crop, *args, **kwargs):
            regions = FaceRegions({"hair": current["hair"], "skin": current["target"]}, (SIDE, SIDE))
            return FaceAnalysis(current["target"].copy(), zeros.copy(), current["target"].copy(), regions, None, 1.0, "test",
                                parser_trusted=True)

    def item_for(source_mask, target_mask):
        return SimpleNamespace(track_id=1, image=np.full_like(frame, 210), inverse=np.eye(2, 3), coverage=source_mask,
                               full_head=True, target_silhouette=target_mask, source_silhouette=source_mask)
    compositor = FrameCompositor(analyzer=Analyzer())
    options = CompositeOptions(color_strength=0, feather_px=0, swap_scope=scope, skull_blend=skull_blend,
                               robust=RobustOptions(analysis_size=SIDE, refine_matte=False, edge_decontamination=False,
                                                    sharpness_matching=False, local_illumination=0, texture_strength=0))
    try:
        for index, (image, t_mask, s_mask, h_mask) in enumerate(history):
            current.update(target=t_mask, hair=h_mask)
            compositor.compose(image, [item_for(s_mask, t_mask)], options=options, frame_index=index)
        current.update(target=target, hair=hair)
        result = compositor.compose(frame, [item_for(source, target)], options=options, frame_index=len(history),
                                    background_donors=background_donors)
        report = dict(compositor.last_reports[1])
    finally:
        compositor.release()
    return result, report


def test_e2e_wide_target_to_slim_source_removes_the_old_jaw_and_contracts_the_hair():
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((30, 52))
    result, report = _compose(frame, target, source, hair)
    inside = result[source > 0.99][:, :]
    assert np.all(np.abs(inside.astype(int) - 210).max(-1) <= 2), "the source face is composited"
    leftover = np.all(result == SKIN, axis=-1) & ~(cv2.dilate(source.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    assert leftover.sum() == 0, "no pixel of the target's old cheeks / jaw may survive around the slimmer face"
    before, after = _hair_cols(frame), _hair_cols(result)
    assert after.min() > before.min() + 6 and after.max() < before.max() - 6
    note = report["silhouette_completion"]
    assert note["applied"] and note["scope"] == "full_craniofacial" and note["inpaint"]["overhang_pixels"] > 500
    np.testing.assert_array_equal(result[:6, :6], frame[:6, :6])
    np.testing.assert_array_equal(result[-6:, -6:], frame[-6:, -6:])


def test_e2e_slim_target_to_wide_source_expands_the_hair_instead_of_colliding_with_it():
    bg, frame, target, hair = head_scene((30, 52))
    source = ellipse((46, 52))
    result, report = _compose(frame, target, source, hair)
    assert np.all(np.abs(result[source > 0.99].astype(int) - 210).max(-1) <= 2)
    before, after = _hair_cols(frame), _hair_cols(result)
    assert after.min() < before.min() - 8 and after.max() > before.max() + 8, "hair must hug the wider skull"
    # hair now sits just outside the new face's flank rather than being buried under it
    ring = (cv2.dilate(source.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0) & (source == 0)
    ring[CENTRE[1] + 20:] = False
    hair_like = np.abs(result.astype(int) - HAIR).sum(-1) < 40
    assert (hair_like & ring).sum() > 0.5 * ring[:CENTRE[1] + 20].sum() * 0.3
    note = report["silhouette_completion"]
    assert note["applied"] and note["inpaint"]["overhang_pixels"] < 0.02 * SIDE * SIDE and not note["cage"]["identity"]


def test_e2e_video_donor_history_supplies_real_background():
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((30, 52))
    # Earlier frames: the same head still on its way in from above (so it hides only the top of the old jaw), with its own masks.
    nothing = np.zeros((SIDE, SIDE), np.float32)
    previous = []
    for dy in (-90, -80):
        t_prev = ellipse((46, 52), centre=(80, 80 + dy))
        h_prev = hair_ring((46, 52)) * 0
        f_prev = bg.copy()
        f_prev[t_prev > 0] = SKIN
        previous.append((f_prev, t_prev, ellipse((30, 52), centre=(80, 80 + dy)), h_prev))
    result, report = _compose(frame, target, source, hair, history=previous)
    inpaint = report["silhouette_completion"]["inpaint"]
    assert inpaint["donors_offered"] == 2 and inpaint["donors_used"] == 2 and inpaint["temporal_pixels"] > 0
    core = _overhang_core(target, source)
    core &= ~_near_hair(result)                                          # the contracted hair now occupies part of the old flank
    assert core.sum() > 300
    assert np.abs(result.astype(int) - bg.astype(int)).sum(-1)[core].max() <= 6


def test_e2e_caller_supplied_future_frames_are_donors_too():
    from roop_3d.rendering.silhouette_inpaint import DonorFrame
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((30, 52))
    future = DonorFrame(bg.copy(), None)                                 # t+1 of an offline job: the head has already left
    result, report = _compose(frame, target, source, hair, background_donors={1: [future]})
    inpaint = report["silhouette_completion"]["inpaint"]
    assert inpaint["donors_used"] == 1 and inpaint["tier"].startswith("temporal")
    core = _overhang_core(target, source)
    core &= ~_near_hair(result)                                          # the contracted hair now occupies part of the old flank
    assert np.abs(result.astype(int) - bg.astype(int)).sum(-1)[core].max() <= 6


def test_e2e_legacy_inner_face_scope_keeps_the_target_outline():
    bg, frame, target, hair = head_scene((46, 52))
    source = ellipse((60, 60))                                           # a much wider source head than the target's own face
    result, report = _compose(frame, target, source, hair, scope="inner_face")
    assert report["silhouette_completion"] == {"scope": "inner_face", "applied": False, "reason": "SCOPE_KEEPS_TARGET_SILHOUETTE"}
    outside_target = (target == 0) & (hair == 0)
    np.testing.assert_array_equal(result[outside_target], frame[outside_target])   # nothing spills past the target's face
    np.testing.assert_array_equal(result[hair > 0], frame[hair > 0])               # hair untouched: no warp, no fill


def test_e2e_adaptive_scope_runs_the_same_silhouette_step():
    bg, frame, target, hair = head_scene((46, 52))
    result, report = _compose(frame, target, ellipse((38, 52)), hair, scope="adaptive_skull", skull_blend=0.5)
    assert report["silhouette_completion"]["scope"] == "adaptive_skull" and report["silhouette_completion"]["applied"]


# ================================================================ parameters + wire format
def test_workstation_parameters_default_to_full_craniofacial_and_stay_off_the_wire_until_set():
    from roop_3d.core.workstation import WorkstationParameters
    plain = WorkstationParameters()
    assert plain.swap_scope == "full_craniofacial" and plain.skull_blend == 1.0
    assert "swap_scope" not in plain.model_dump() and "skull_blend" not in plain.model_dump()
    assert plain.composite_options().swap_scope == "full_craniofacial"
    touched = WorkstationParameters(swap_scope="adaptive_skull", skull_blend=0.4)
    assert touched.model_dump()["swap_scope"] == "adaptive_skull" and touched.model_dump()["skull_blend"] == 0.4
    options = touched.composite_options()
    assert (options.swap_scope, options.skull_blend) == ("adaptive_skull", 0.4)
    for bad in ({"swap_scope": "everything"}, {"skull_blend": 1.1}, {"skull_blend": -0.1}):
        with pytest.raises(ValueError):
            WorkstationParameters(**bad)


def test_job_fingerprint_is_unchanged_until_the_scope_is_touched():
    from roop_3d.jobs.schema import JobRequest
    base = dict(target_path="roop_3d/.runtime/media/t.mp4", inspection_id="a" * 32,
                face_mappings=[{"track_id": "track_0000", "faceset_id": "alice"}])
    plain = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4}}).identity_payload()["workstation_parameters"]
    touched = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4, "swap_scope": "inner_face"}}
                                        ).identity_payload()["workstation_parameters"]
    assert "swap_scope" not in plain and "skull_blend" not in plain
    assert touched["swap_scope"] == "inner_face"


def test_live_preview_cache_key_changes_only_when_the_scope_is_set():
    from pydantic import ValidationError
    from roop_3d.server.live_preview import LivePreviewPayload
    base = dict(engine_id="reference_onnx", target_path="roop_3d/.runtime/media/t.mp4", inspection_id="a" * 32, frame_index=3,
                target_face_id="track_0000", source_face_ids=["alice"])
    plain = LivePreviewPayload(**base).config_digest()
    assert "swap_scope" not in plain and "skull_blend" not in plain
    touched = LivePreviewPayload(**base, swap_scope="adaptive_skull", skull_blend=0.4).config_digest()
    assert touched["swap_scope"] == "adaptive_skull" and touched["skull_blend"] == 0.4 and touched != plain
    for bad in ({"swap_scope": "everything"}, {"skull_blend": 1.5}):
        with pytest.raises(ValidationError):
            LivePreviewPayload(**base, **bad)


def test_composite_options_reject_an_invalid_scope():
    from roop_3d.compositor.pipeline import CompositeOptions
    with pytest.raises(ValueError):
        CompositeOptions(swap_scope="bogus")
    with pytest.raises(ValueError):
        CompositeOptions(silhouette_history=99)
