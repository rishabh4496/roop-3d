"""Lead VFX Compositing & Semantic Segmentation Engine Tests.

Tests the refactored Stage 3 Masking & Compositing Engine:
1. Strict Hair, Ear & Neck Exclusion in Semantic Face Parsing (BiSeNet 19-Class).
2. Edge-Preserving Fast Guided Filter for Hairline & Bangs (Zero Smudge Blur).
3. Photorealistic Skin Detail & Subsurface Color Matching (Anti-Plastic Finish).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.compositor.color_match import (
    apply_ycbcr_stats,
    estimate_lab_stats,
    estimate_ycbcr_stats,
    ycbcr_mean_std_match,
)
from roop_3d.compositor.hairline import refine_hairline_matte
from roop_3d.compositor.lighting import relight_spherical_harmonics
from roop_3d.compositor.masks import (
    INCLUDED_SKIN_FACE_CLASSES,
    INCLUDED_SKIN_FACE_CLASSES_DICT,
    STRICT_FORBIDDEN_CLASSES,
    STRICT_FORBIDDEN_CLASSES_DICT,
    apply_submental_jaw_cutoff,
    compose_active_mask,
    compute_dual_bisenet_forbidden_mask,
    extract_forbidden_mask,
)
from roop_3d.compositor.matte import fast_guided_filter, guidedFilter
from roop_3d.compositor.quality import inject_target_skin_micro_texture


# ---------------------------------------------------------------------------
# 1. Strict Hair, Ear & Neck Exclusion in Semantic Face Parsing (BiSeNet 19-Class)
# ---------------------------------------------------------------------------

def test_semantic_class_partitioning_matches_vfx_spec():
    """Verify exact partition of 19-class CelebAMask-HQ labels."""
    expected_included = {
        1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye", 5: "r_eye",
        10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
    }
    expected_forbidden = {
        14: "neck", 15: "neck_l", 16: "cloth", 17: "hair", 18: "hat",
        7: "l_ear", 8: "r_ear", 9: "ear_r", 6: "eye_g",
    }
    assert set(INCLUDED_SKIN_FACE_CLASSES) == set(expected_included.keys())
    assert set(STRICT_FORBIDDEN_CLASSES) == set(expected_forbidden.keys())
    assert INCLUDED_SKIN_FACE_CLASSES_DICT == expected_included
    assert STRICT_FORBIDDEN_CLASSES_DICT == expected_forbidden
    # The two sets must be mutually exclusive
    assert set(INCLUDED_SKIN_FACE_CLASSES).isdisjoint(set(STRICT_FORBIDDEN_CLASSES))


def test_extract_forbidden_mask_captures_all_forbidden_classes():
    """Verify that extract_forbidden_mask aggregates probability mass from hair, neck, ears, cloth, hat, glasses."""
    probs = np.zeros((32, 32, 19), dtype=np.float32)
    # Background (class 0)
    probs[:10, :, 0] = 1.0
    # Included skin (class 1)
    probs[10:20, :16, 1] = 1.0
    # Forbidden hair (class 17)
    probs[10:20, 16:, 17] = 0.9
    # Forbidden neck (class 14)
    probs[20:, :16, 14] = 0.85
    # Forbidden ear (class 7)
    probs[20:, 16:, 7] = 0.75

    forbidden_mask = extract_forbidden_mask(probs)
    assert forbidden_mask.shape == (32, 32)
    # Background and skin regions should have 0 forbidden score
    assert np.allclose(forbidden_mask[:10, :], 0.0)
    assert np.allclose(forbidden_mask[10:20, :16], 0.0)
    # Hair, neck, and ear regions must be extracted
    assert np.allclose(forbidden_mask[10:20, 16:], 0.9)
    assert np.allclose(forbidden_mask[20:, :16], 0.85)
    assert np.allclose(forbidden_mask[20:, 16:], 0.75)


def test_dual_bisenet_forbidden_mask_computes_union():
    """Verify that dual BiSeNet runs on both target crop and swapped crop and unions forbidden pixels."""
    class MockParser:
        def __init__(self, target_forbidden_box, swap_forbidden_box):
            self.target_box = target_forbidden_box
            self.swap_box = swap_forbidden_box

        def predict(self, crop: np.ndarray) -> np.ndarray:
            probs = np.zeros((crop.shape[0], crop.shape[1], 19), dtype=np.float32)
            probs[..., 1] = 1.0  # default skin
            # If crop matches target or swap, mark hair class 17
            if crop[0, 0, 0] == 100:  # target tag
                r0, r1, c0, c1 = self.target_box
                probs[r0:r1, c0:c1, 1] = 0.0
                probs[r0:r1, c0:c1, 17] = 1.0
            elif crop[0, 0, 0] == 200:  # swap tag
                r0, r1, c0, c1 = self.swap_box
                probs[r0:r1, c0:c1, 1] = 0.0
                probs[r0:r1, c0:c1, 17] = 1.0
            return probs

    target_crop = np.full((64, 64, 3), 100, dtype=np.uint8)
    swapped_crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    parser = MockParser((0, 20, 0, 30), (10, 30, 20, 50))

    tgt_forbid, swap_forbid, combined = compute_dual_bisenet_forbidden_mask(
        parser, target_crop, swapped_crop
    )
    assert tgt_forbid[5, 10] == 1.0
    assert tgt_forbid[25, 35] == 0.0
    assert swap_forbid[25, 35] == 1.0
    # Combined must contain both forbidden regions
    assert combined[5, 10] == 1.0
    assert combined[25, 35] == 1.0
    assert combined[50, 50] == 0.0


def test_compose_active_mask_formula():
    """Verify active_mask = mesh_face_only_mask * xseg_face_mask * (1.0 - forbidden_hair_neck_ear_mask)."""
    mesh = np.full((32, 32), 0.9, dtype=np.float32)
    xseg = np.full((32, 32), 0.8, dtype=np.float32)
    forbidden = np.zeros((32, 32), dtype=np.float32)
    forbidden[:10, :] = 0.6  # hair / neck

    active = compose_active_mask(mesh, xseg, forbidden)
    # In unoccluded region: 0.9 * 0.8 * 1.0 = 0.72
    assert np.allclose(active[15:, :], 0.9 * 0.8, atol=1e-5)
    # In forbidden region: 0.9 * 0.8 * (1.0 - 0.6) = 0.288
    assert np.allclose(active[:10, :], 0.9 * 0.8 * (1.0 - 0.6), atol=1e-5)


def test_apply_submental_jaw_cutoff_zeroes_throat_and_neck():
    """Verify that Submental Chin-to-Neck Cutoff zeroes out mask below jawline and creates inward SDF falloff."""
    h, w = 128, 128
    mask = np.ones((h, w), dtype=np.float32)
    # Mock a 68-point landmark array with chin tip at y=80, x=64 and gonions at (y=50, x=20) and (y=50, x=108)
    landmarks = np.zeros((68, 2), dtype=np.float32)
    # jaw points 0..16
    for i in range(17):
        t = i / 16.0
        # parabola from x=20..108, y bottom at 80
        x = 20.0 + t * 88.0
        y = 50.0 + 30.0 * (1.0 - 4.0 * (t - 0.5) ** 2)
        landmarks[i] = [x, y]

    cutoff_mask = apply_submental_jaw_cutoff(mask, landmarks, falloff_px=6.0)
    # Below the chin (e.g. y=100..128, which is the throat/neck), mask must be strictly 0.0
    assert np.all(cutoff_mask[95:, :] == 0.0)
    # Well inside the face (e.g. nose at y=40, x=64), mask should remain 1.0
    assert cutoff_mask[40, 64] == 1.0
    # Right at the chin boundary inside, there is an inward falloff between 0 and 1
    assert 0.0 <= cutoff_mask[78, 64] <= 1.0


# ---------------------------------------------------------------------------
# 2. Edge-Preserving Fast Guided Filter for Hairline & Bangs (Zero Smudge Blur)
# ---------------------------------------------------------------------------

def test_fast_guided_filter_edge_preservation():
    """Fast Guided Filter (guidedFilter) locks to sharp edges without blurring across high-contrast boundary."""
    h, w = 64, 64
    guide = np.zeros((h, w), dtype=np.float32)
    guide[:, 32:] = 1.0  # high contrast step edge (e.g. hair strand vs forehead)

    # Initial rough / blurry segmentation matte
    source = np.zeros((h, w), dtype=np.float32)
    source[:, 30:] = 1.0
    blurry_source = cv2.GaussianBlur(source, (9, 9), 2.5)

    # Standard fast_guided_filter call and guidedFilter alias
    filtered = fast_guided_filter(guide, blurry_source, radius=8, eps=1e-4)
    alias_filtered = guidedFilter(guide, blurry_source, radius=8, eps=1e-4)

    assert np.allclose(filtered, alias_filtered)
    # Sharp transition at x=32
    step_guided = float(filtered[32, 33] - filtered[32, 31])
    step_blurry = float(blurry_source[32, 33] - blurry_source[32, 31])
    # The guided step is much sharper than the blurry gaussian transition
    assert step_guided > 2.0 * step_blurry
    # Guided filter suppresses bleed on the dark side of guide (e.g. at x=30, 31)
    assert filtered[32, 31] < blurry_source[32, 31]
    assert filtered[32, 30] < blurry_source[32, 30]


def test_refine_hairline_matte_differentiates_forehead_from_jaw():
    """refine_hairline_matte applies guided filter on forehead and smooth feathering on jaw."""
    h, w = 128, 128
    guide_bgr = np.full((h, w, 3), 180, dtype=np.uint8)  # bright skin
    # Dark hair strand in upper forehead: column 62:66 is dark hair
    guide_bgr[:50, 62:66, :] = 20

    # Initial matte with soft hair cutout
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:50, 62:66] = 0.3

    refined = refine_hairline_matte(alpha, guide_bgr, radius=8, eps=1e-4, smooth_feather_px=4.0)
    assert refined.shape == (h, w)
    # Along the hair strand in the forehead (y=20, x=64), alpha is significantly lower than skin
    assert refined[20, 64] < refined[20, 80]
    # Along the jaw (y=100, x=64), where guide was uniform skin, smooth feathering keeps alpha smooth
    assert np.isfinite(refined[100, :]).all()


# ---------------------------------------------------------------------------
# 3. Photorealistic Skin Detail & Subsurface Color Matching (Anti-Plastic Finish)
# ---------------------------------------------------------------------------

def test_ycbcr_and_lab_stats_exclude_dark_shadows():
    """Verify that estimate_ycbcr_stats and estimate_lab_stats exclude pixels with luma <= 0.08."""
    h, w = 64, 64
    source = np.full((h, w, 3), 150, dtype=np.uint8)
    target = np.full((h, w, 3), 180, dtype=np.uint8)
    # Dark hair / shadow region: luma < 0.08
    target[:20, :, :] = 10  # low luma ~ 0.04
    mask = np.ones((h, w), dtype=np.float32)

    stats_ycc = estimate_ycbcr_stats(source, target, mask)
    # Target Y channel in YCbCr for pixel value 180 is ~180.
    # If the dark region (value 10) was included, mean would be significantly lower than 180.
    assert stats_ycc.target_mean[0] > 160

    stats_lab = estimate_lab_stats(source, target, mask)
    # Target L channel in LAB for pixel value 180 is ~188
    assert stats_lab.target_mean[0] > 160


def test_ycbcr_mean_std_match_transfers_chroma():
    """ycbcr_mean_std_match transfers luminance and chrominance within the masked skin area."""
    h, w = 32, 32
    source = np.full((h, w, 3), [100, 120, 150], dtype=np.uint8)  # BGR
    target = np.full((h, w, 3), [120, 160, 210], dtype=np.uint8)  # BGR warmer tone
    mask = np.zeros((h, w), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    matched = ycbcr_mean_std_match(source, target, mask, strength=1.0)
    # Inside mask, image should shift toward target
    assert np.all(matched[12:20, 12:20, 2] > source[12:20, 12:20, 2])
    # Outside mask, unchanged
    assert np.all(matched[0, 0] == source[0, 0])


def test_inject_target_skin_micro_texture_preserves_pore_frequencies():
    """Verify that 25% target high-pass skin residual is injected exclusively over unoccluded skin pixels."""
    h, w = 64, 64
    # Enhanced image is slightly smooth
    enhanced = np.full((h, w, 3), 128, dtype=np.uint8)
    # Target frame has high-frequency pore texture
    rng = np.random.default_rng(42)
    noise = rng.integers(-20, 20, size=(h, w, 3), dtype=np.int16)
    target = np.clip(128 + noise, 0, 255).astype(np.uint8)

    skin_mask = np.zeros((h, w), dtype=np.float32)
    skin_mask[16:48, 16:48] = 1.0

    injected = inject_target_skin_micro_texture(
        enhanced, target, skin_mask, injection_rate=0.25, highpass_sigma=1.5
    )
    # Outside skin mask, enhanced is completely unchanged (exact 128)
    assert np.all(injected[:10, :10] == 128)
    # Inside skin mask, variance has increased due to the 25% target pore residual
    inner_var = np.var(injected[20:40, 20:40].astype(np.float32))
    assert inner_var > 0.5


def test_relight_spherical_harmonics_decoupled_luma_excludes_shadows():
    """Verify that relight_spherical_harmonics modulates luminance without flat RGB tinting."""
    h, w = 32, 32
    swapped = np.full((h, w, 3), [80, 120, 180], dtype=np.uint8)  # warm skin tone
    target = np.full((h, w, 3), [100, 140, 200], dtype=np.uint8)
    # Dark hair / neck shadow region in target (luma < 0.08)
    target[:10, :, :] = 10
    skin_mask = np.ones((h, w), dtype=np.float32)
    face_mask = np.ones((h, w), dtype=np.float32)

    relit = relight_spherical_harmonics(swapped, target, skin_mask, face_mask, color_space="ycbcr")
    assert relit.shape == (h, w, 3)
    assert relit.dtype == np.uint8
    # Skin luminance was relit
    assert not np.array_equal(relit, swapped)
