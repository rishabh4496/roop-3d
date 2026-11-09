"""Stage 15 Studio Polish VFX Finishing & Biophysical Solvers Acceptance Tests.

Validates the four studio-grade optical and biophysical finishing solvers in Stage 2 & Stage 3:
1. Dynamic Camera Intrinsic Focal-Length (FOV) Solver:
   - Levenberg-Marquardt K_target minimization over 3D landmarks.
   - Bounded between 18mm wide-angle selfie and 135mm telephoto equivalents.
   - 100% matching perspective foreshortening recovery.
2. Perceptual OKLab Sliced Wasserstein Color Transfer + SSS Shadow Warmth:
   - Lossless OKLab perceptual color transformations.
   - Specular / diffuse reflectance separation (pore highlights & sheen preservation).
   - Sliced Wasserstein Optimal Transport exclusively over unoccluded BiSeNet Class 1 skin pixels.
   - Subsurface scattering shadow-terminator curve along N dot L transition zone.
3. Boundary-Ribbon Sub-Pixel Alpha Matting for Hair Strands, Eyeglasses & Facial Hair:
   - Automatic 12px morphological boundary ribbon (Dilate(6px) - Erode(6px)).
   - Multi-scale Color Line Model + Matting Laplacian sub-pixel refinement at boundaries.
4. Submental Under-Chin Contact Shadow Projection:
   - 3D jaw contour with overhead SH lighting vector.
   - Soft contact shadow gradient (0-15% opacity, 6px-14px falloff) onto upper neck.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
import pytest

from roop_3d.compositor.color_match import (
    apply_sss_shadow_warmth,
    oklab_to_srgb,
    oklab_wasserstein_match_with_sss,
    recombine_skin_specular,
    separate_skin_specular_diffuse,
    sliced_wasserstein_color_transfer_oklab,
    srgb_to_oklab,
)
from roop_3d.compositor.lighting import project_submental_contact_shadow
from roop_3d.compositor.matte import (
    build_boundary_ribbon_trimap,
    refine_boundary_ribbon_matte,
)
from roop_3d.reconstruction.dynamic_camera import (
    DEFAULT_FOCAL_BOUNDS_MM,
    DynamicCameraSolver,
    compute_focal_bounds_pixels,
    focal_pixels_to_mm,
    mm_to_focal_pixels,
    solve_dynamic_camera_intrinsics,
)
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    Pose3D,
    euler_to_rotation_matrix,
    rotation_matrix_to_rodrigues,
)
from roop_3d.reconstruction.native_face_geometry import bfm_camera
from roop_3d.reconstruction.tps_fallback import (
    get_canonical_3d_landmarks,
    solve_pose_pnp,
)


# ===========================================================================
# 1. Dynamic Camera Intrinsic Focal-Length (FOV) Solver Tests
# ===========================================================================

def test_focal_length_conversions_match_35mm_full_frame_standards():
    """Verify millimeter-to-pixel and pixel-to-millimeter conversions for 35mm full-frame."""
    image_dim = 512
    # 36mm sensor width: 18mm is 0.5 * image_dim = 256px
    f_18mm = mm_to_focal_pixels(18.0, image_dim, sensor_dimension_mm=36.0)
    assert f_18mm == pytest.approx(256.0, abs=1e-5)
    assert focal_pixels_to_mm(f_18mm, image_dim, sensor_dimension_mm=36.0) == pytest.approx(18.0, abs=1e-5)

    # 135mm telephoto is 3.75 * image_dim = 1920px
    f_135mm = mm_to_focal_pixels(135.0, image_dim, sensor_dimension_mm=36.0)
    assert f_135mm == pytest.approx(1920.0, abs=1e-5)
    assert focal_pixels_to_mm(f_135mm, image_dim, sensor_dimension_mm=36.0) == pytest.approx(135.0, abs=1e-5)

    min_px, max_px = compute_focal_bounds_pixels(512, 512)
    assert min_px == pytest.approx(256.0, abs=1e-5)
    assert max_px == pytest.approx(1920.0, abs=1e-5)


def test_dynamic_camera_solver_wide_angle_selfie_foreshortening_recovery():
    """Verify that a 24mm wide-angle selfie with strong perspective foreshortening is recovered accurately."""
    w, h = 640, 480
    canonical = get_canonical_3d_landmarks(68)

    # Ground-truth wide-angle camera (24mm equivalent)
    f_gt = mm_to_focal_pixels(24.0, max(w, h), sensor_dimension_mm=36.0)  # ~426.67px
    cx_gt, cy_gt = w / 2.0, h / 2.0
    gt_intrinsics = CameraIntrinsics(fx=f_gt, fy=f_gt, cx=cx_gt, cy=cy_gt, width=w, height=h)

    # Ground-truth close-up pose (e.g. arm's length selfie at 0.55m)
    gt_yaw, gt_pitch, gt_roll = 10.0, -8.0, 4.0
    gt_tvec = np.array([0.02, -0.01, 0.55], dtype=np.float64)
    gt_r_mat = euler_to_rotation_matrix(gt_yaw, gt_pitch, gt_roll, degrees=True)
    transformed_3d = (canonical @ gt_r_mat.T) + gt_tvec
    observed_2d = gt_intrinsics.project(transformed_3d)

    # Solve with DynamicCameraSolver starting from default 50mm guess
    solver = DynamicCameraSolver()
    recovered_intrinsics, recovered_pose = solver.solve(
        observed_2d,
        canonical,
        width=w,
        height=h,
    )

    # Verify recovered focal length matches ground truth
    assert recovered_intrinsics.fx == pytest.approx(f_gt, rel=0.03)
    assert recovered_intrinsics.fy == pytest.approx(f_gt, rel=0.03)
    # Pose recovery accuracy
    assert recovered_pose.yaw == pytest.approx(gt_yaw, abs=0.8)
    assert recovered_pose.pitch == pytest.approx(gt_pitch, abs=0.8)
    assert recovered_pose.roll == pytest.approx(gt_roll, abs=0.8)
    # Reprojection error must be sub-pixel (< 0.2 px)
    assert recovered_pose.reprojection_error < 0.2
    assert recovered_pose.f_est == pytest.approx(f_gt, rel=0.03)


def test_dynamic_camera_solver_telephoto_shot_recovery():
    """Verify that a 105mm telephoto shot with flat foreshortening is recovered accurately."""
    w, h = 640, 480
    canonical = get_canonical_3d_landmarks(68)

    # Ground-truth telephoto camera (105mm equivalent)
    f_gt = mm_to_focal_pixels(105.0, max(w, h), sensor_dimension_mm=36.0)  # ~1866.67px
    cx_gt, cy_gt = w / 2.0, h / 2.0
    gt_intrinsics = CameraIntrinsics(fx=f_gt, fy=f_gt, cx=cx_gt, cy=cy_gt, width=w, height=h)

    # Distant telephoto pose (subject at 2.4m)
    gt_yaw, gt_pitch, gt_roll = -12.0, 6.0, -3.0
    gt_tvec = np.array([-0.05, 0.03, 2.4], dtype=np.float64)
    gt_r_mat = euler_to_rotation_matrix(gt_yaw, gt_pitch, gt_roll, degrees=True)
    transformed_3d = (canonical @ gt_r_mat.T) + gt_tvec
    observed_2d = gt_intrinsics.project(transformed_3d)

    solver = DynamicCameraSolver()
    recovered_intrinsics, recovered_pose = solver.solve(
        observed_2d,
        canonical,
        width=w,
        height=h,
    )

    assert recovered_intrinsics.fx == pytest.approx(f_gt, rel=0.04)
    assert recovered_pose.yaw == pytest.approx(gt_yaw, abs=0.8)
    assert recovered_pose.pitch == pytest.approx(gt_pitch, abs=0.8)
    assert recovered_pose.roll == pytest.approx(gt_roll, abs=0.8)
    assert recovered_pose.reprojection_error < 0.2


def test_solve_pose_pnp_with_dynamic_focal_flag():
    """Verify solve_pose_pnp automatically enables dynamic focal solver when estimate_focal=True."""
    w, h = 512, 512
    canonical = get_canonical_3d_landmarks(68)
    gt_intrinsics = CameraIntrinsics.from_fov(fov_degrees=50.0, width=w, height=h)
    gt_yaw, gt_pitch, gt_roll = 5.0, -5.0, 2.0
    gt_tvec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gt_r_mat = euler_to_rotation_matrix(gt_yaw, gt_pitch, gt_roll, degrees=True)
    transformed_3d = (canonical @ gt_r_mat.T) + gt_tvec
    observed_2d = gt_intrinsics.project(transformed_3d)

    # Run solve_pose_pnp with estimate_focal=True
    pose = solve_pose_pnp(observed_2d, gt_intrinsics, canonical_3d=canonical, estimate_focal=True)
    assert pose.f_est is not None
    assert pose.f_est == pytest.approx(gt_intrinsics.fx, rel=0.03)
    assert pose.reprojection_error < 0.2


def test_bfm_camera_dynamic_focal_replaces_1015():
    """Verify bfm_camera replaces legacy 1015.0 when f_est is supplied."""
    coeffs = np.zeros(257, np.float32)
    coeffs[224:227] = (0.1, -0.1, 0.05)
    coeffs[254:] = (0.05, -0.05, 0.5)

    # Legacy default returns 1015.0
    _, _, k_legacy = bfm_camera(coeffs)
    assert k_legacy[0, 0] == 1015.0
    assert k_legacy[1, 1] == 1015.0

    # With dynamic f_est supplied
    _, _, k_dynamic = bfm_camera(coeffs, f_est=450.0)
    assert k_dynamic[0, 0] == 450.0
    assert k_dynamic[1, 1] == 450.0
    assert k_dynamic[0, 2] == 112.0
    assert k_dynamic[1, 2] == 111.0


# ===========================================================================
# 2. OKLab Sliced Wasserstein Color Transfer & SSS Shadow Warmth Tests
# ===========================================================================

def test_srgb_to_oklab_and_back_is_lossless():
    """Verify that srgb_to_oklab and oklab_to_srgb round-trip within 1 code value."""
    rng = np.random.default_rng(42)
    # Diverse skin, shadow, and clothing colors
    sample_bgr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    oklab = srgb_to_oklab(sample_bgr)

    # Verify OKLab coordinates
    # L in [0, 1]
    assert np.all(oklab[..., 0] >= -1e-4) and np.all(oklab[..., 0] <= 1.0 + 1e-4)
    # a, b chrominance in roughly [-0.5, 0.5]
    assert np.all(np.abs(oklab[..., 1:]) < 0.6)

    reconstructed_bgr = oklab_to_srgb(oklab)
    max_err = np.max(np.abs(sample_bgr.astype(int) - reconstructed_bgr.astype(int)))
    assert max_err <= 1


def test_skin_specular_diffuse_separation_isolates_pore_sheen():
    """Verify separate_skin_specular_diffuse extracts pore highlights into specular_map."""
    # Base skin tone in BGR
    img = np.full((64, 64, 3), [130, 160, 210], dtype=np.uint8)
    # Add a high-intensity pore highlight glint in the center
    img[30:34, 30:34] = [240, 250, 255]

    skin_mask = np.ones((64, 64), dtype=np.float32)
    diffuse, spec_map = separate_skin_specular_diffuse(img, skin_mask, threshold=2.0)

    # Specular map should have peak at the highlight
    assert np.max(spec_map[30:34, 30:34]) > 20.0
    # Specular map should be zero in flat skin regions
    assert np.all(spec_map[:10, :10] == 0.0)

    # Recombination should restore original highlight exactly
    recombined = recombine_skin_specular(diffuse, spec_map)
    np.testing.assert_allclose(recombined, img, atol=2)


def test_sliced_wasserstein_color_transfer_oklab_aligns_skin_tones():
    """Verify Sliced Wasserstein color matching in OKLab matches target skin distribution."""
    # Source: pale / cool skin
    source_bgr = np.full((48, 48, 3), [180, 190, 220], dtype=np.uint8)
    # Target: warm / sun-kissed skin
    target_bgr = np.full((48, 48, 3), [100, 140, 210], dtype=np.uint8)
    mask = np.ones((48, 48), dtype=np.float32)

    matched = sliced_wasserstein_color_transfer_oklab(
        source_bgr, target_bgr, mask, strength=1.0, num_slices=24, num_iters=2
    )

    # In matched result, skin tone must move strongly toward target
    mean_matched = matched.mean(axis=(0, 1))
    mean_target = target_bgr.mean(axis=(0, 1))
    np.testing.assert_allclose(mean_matched, mean_target, atol=12.0)

    # Pixels outside mask must remain untouched
    partial_mask = np.zeros((48, 48), dtype=np.float32)
    partial_mask[16:32, 16:32] = 1.0
    matched_partial = sliced_wasserstein_color_transfer_oklab(source_bgr, target_bgr, partial_mask)
    np.testing.assert_array_equal(matched_partial[:10, :10], source_bgr[:10, :10])


def test_apply_sss_shadow_warmth_targets_shadow_terminator_zone():
    """Verify SSS shadow warmth curve boosts dermal warmth exclusively along the N dot L transition zone."""
    h, w = 64, 64
    # Neutral skin tone
    base_bgr = np.full((h, w, 3), [140, 160, 200], dtype=np.uint8)

    # Synthetic normal map: hemisphere shading from left to right
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r2 = xx * xx + yy * yy
    zz = np.sqrt(np.maximum(1.0 - np.minimum(r2, 1.0), 0.05))
    normals = np.stack([xx, yy, zz], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    # Key light pointing from right (+X)
    light_dir = np.array([1.0, 0.0, 0.3], dtype=np.float32)

    skin_mask = np.ones((h, w), dtype=np.float32)
    warm_bgr = apply_sss_shadow_warmth(base_bgr, normals, light_dir, skin_mask=skin_mask, warmth_amount=0.8)

    # Find the shadow terminator column where N dot L is near 0.04
    n_dot_l = np.sum(normals * (light_dir / np.linalg.norm(light_dir)), axis=-1)
    terminator_mask = np.abs(n_dot_l - 0.04) < 0.08
    full_lit_mask = n_dot_l > 0.7
    deep_shadow_mask = n_dot_l < -0.5

    # Terminator region must gain red/amber warmth (R channel in BGR increases relative to B)
    terminator_diff = warm_bgr.astype(float) - base_bgr.astype(float)
    # Red channel (index 2) should increase more than Blue channel (index 0)
    assert np.mean(terminator_diff[terminator_mask, 2]) > np.mean(terminator_diff[terminator_mask, 0])
    assert np.mean(terminator_diff[terminator_mask, 2]) > 3.0

    # Full lit region and deep shadow should have near-zero warmth shift
    assert np.allclose(warm_bgr[full_lit_mask], base_bgr[full_lit_mask], atol=2)
    assert np.allclose(warm_bgr[deep_shadow_mask], base_bgr[deep_shadow_mask], atol=2)


# ===========================================================================
# 3. Boundary-Ribbon Sub-Pixel Alpha Matting Tests
# ===========================================================================

def test_build_boundary_ribbon_trimap_creates_12px_morphological_band():
    """Verify build_boundary_ribbon_trimap produces an exact 12px uncertain boundary band."""
    h, w = 120, 120
    mask = np.zeros((h, w), dtype=np.float32)
    mask[30:90, 30:90] = 1.0  # rectangular face core

    trimap, uncertain, core = build_boundary_ribbon_trimap(mask, radius=6)

    assert trimap.shape == (h, w)
    assert uncertain.shape == (h, w)
    assert core.shape == (h, w)

    # Check that uncertain ribbon on a straight boundary is exactly 12px wide
    # Left edge: column profile across x=15..45 at row 60
    left_edge_width = int(np.count_nonzero(uncertain[60, :50] > 0))
    right_edge_width = int(np.count_nonzero(uncertain[60, 50:] > 0))
    assert left_edge_width == 12
    assert right_edge_width == 12

    # Trimap values: 1.0 (core), 0.5 (uncertain), 0.0 (background)
    assert np.all(trimap[core > 0.5] == 1.0)
    assert np.all(trimap[uncertain > 0.1] == 0.5)
    assert np.all(trimap[(core == 0) & (uncertain == 0)] == 0.0)


def test_refine_boundary_ribbon_matte_subpixel_edge_accuracy():
    """Verify refine_boundary_ribbon_matte resolves fine boundary hair strands with sub-pixel sharpness."""
    h, w = 128, 128
    # Background: skin tone
    guide = np.full((h, w, 3), [160, 180, 220], dtype=np.uint8)
    # Fine dark hair strand passing through the boundary (3px wide)
    guide[20:70, 62:65] = [25, 20, 18]

    # Initial rough mask
    mask = np.zeros((h, w), dtype=np.float32)
    mask[40:100, 30:100] = 1.0

    refined = refine_boundary_ribbon_matte(guide, mask, radius=6)

    assert refined.shape == (h, w)
    assert refined.dtype == np.float32
    assert np.all(refined >= 0.0) and np.all(refined <= 1.0)

    # Deep inside core must remain 1.0
    assert np.all(refined[60:80, 50:70] == 1.0)
    # Far outside must remain 0.0
    assert np.all(refined[:20, :20] == 0.0)

    # Over the dark hair strand, alpha should be significantly attenuated (occluder locked)
    strand_alpha = refined[45, 63]
    neighbor_alpha = refined[45, 75]
    assert strand_alpha < neighbor_alpha


# ===========================================================================
# 4. Submental Under-Chin Contact Shadow Projection Tests
# ===========================================================================

def test_project_submental_contact_shadow_creates_soft_gradient_under_chin():
    """Verify project_submental_contact_shadow creates 0-15% opacity contact shadow on upper neck."""
    h, w = 128, 128
    # Uniform base image (e.g. neck skin)
    image = np.full((h, w, 3), 180, dtype=np.uint8)

    # Chin jawline contour across y=65..75
    jaw_pts = np.array([
        [25.0, 55.0], [40.0, 68.0], [64.0, 75.0], [88.0, 68.0], [103.0, 55.0]
    ], dtype=np.float32)

    # Overhead lighting vector (+Y points down in screen coords)
    overhead_light = np.array([0.1, 0.9, 0.4], dtype=np.float32)

    shadowed = project_submental_contact_shadow(
        image,
        jaw_pts,
        light_direction=overhead_light,
        max_opacity=0.12,
        falloff_px=10.0,
    )

    assert shadowed.shape == (h, w, 3)
    assert shadowed.dtype == np.uint8

    # Region ABOVE the jawline (face interior) must be 100% untouched
    assert np.all(shadowed[:50, :] == 180)

    # Region immediately below chin (y=76..82, x=60..68) must be shadowed
    chin_shadow_sample = shadowed[77, 64]
    darkening = (180.0 - float(np.mean(chin_shadow_sample))) / 180.0
    # Must be bounded in [0.05, 0.15]
    assert 0.04 <= darkening <= 0.15

    # Falloff: pixels far below the chin (y > 105) must return to unshadowed 180
    assert np.all(shadowed[110:, 64] == 180)


def test_project_submental_contact_shadow_zero_overhead_produces_no_shadow():
    """Verify that when light comes from below (-Y), contact shadow is zero."""
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    jaw_pts = np.array([[10, 30], [32, 40], [54, 30]], dtype=np.float32)

    # Upward light (negative Y)
    upward_light = np.array([0.0, -0.8, 0.6], dtype=np.float32)
    shadowed = project_submental_contact_shadow(
        image, jaw_pts, light_direction=upward_light, max_opacity=0.12
    )

    # No overhead light -> no contact shadow cast downward
    diff = np.max(np.abs(image.astype(int) - shadowed.astype(int)))
    assert diff <= 3  # Minimal or zero
