"""Stage 10 physical-camera optics, relighting, assets, and UV denoising tests."""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.rendering.physical_camera import (
    DominantKeyLightAligner,
    PhysicalCameraMatcher,
    TemporalSpatialSwapDenoiser,
    load_cc0_preview_assets,
    sample_micro_pore_normal,
)


def _linear_to_srgb(values):
    values = np.asarray(values, np.float32)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055)


def test_projection_and_focal_estimation_recover_physical_focal_length():
    height, width = 720, 1280
    points = np.array([
        [-0.09, -0.06, 0.48], [-0.05, 0.03, 0.52], [0.00, -0.02, 0.45],
        [0.04, 0.05, 0.57], [0.09, -0.04, 0.49], [-0.02, 0.08, 0.62],
        [0.07, 0.09, 0.66], [-0.08, 0.07, 0.59],
    ], dtype=np.float64)
    focal_mm = 72.0
    projected = PhysicalCameraMatcher.project_points(points, (height, width, 3), focal_mm)
    estimated = PhysicalCameraMatcher.estimate_focal_length_mm(points, projected, (height, width, 3))
    np.testing.assert_allclose(estimated, focal_mm, atol=1e-4)
    wide = PhysicalCameraMatcher.project_points(points, (height, width, 3), 18.0)
    tele = PhysicalCameraMatcher.project_points(points, (height, width, 3), 135.0)
    center = np.array([width / 2, height / 2])
    assert np.linalg.norm(tele[0] - center) > 7.0 * np.linalg.norm(wide[0] - center)


def test_radial_distortion_fit_recovers_k1_and_k2():
    points = np.array([[x, y, 0.5 + 0.1 * x] for x in (-0.16, -0.08, 0, 0.08, 0.16)
                       for y in (-0.12, -0.04, 0.04, 0.12)], dtype=np.float64)
    image_shape = (800, 1200, 3)
    expected = (-0.16, 0.045)
    observed = PhysicalCameraMatcher.project_points(points, image_shape, 48.0, distortion=(*expected, 0.0, 0.0, 0.0))
    fitted = PhysicalCameraMatcher.estimate_radial_distortion(points, observed, 48.0, image_shape)
    np.testing.assert_allclose(fitted, expected, atol=0.01)


def test_chromatic_aberration_measurement_detects_opposed_color_edge_shifts():
    height, width = 96, 128
    green = np.zeros((height, width), np.uint8)
    green[20:76, 26:100] = 255
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radial_x = (xx - (width - 1) * 0.5) / width
    radial_y = (yy - (height - 1) * 0.5) / width
    blue = cv2.remap(green, xx - radial_x * 8.0, yy - radial_y * 8.0, cv2.INTER_LINEAR)
    red = cv2.remap(green, xx + radial_x * 8.0, yy + radial_y * 8.0, cv2.INTER_LINEAR)
    frame = np.stack((blue, green, red), axis=-1)  # BGR
    measured = PhysicalCameraMatcher.measure_target_optics(frame)
    assert abs(measured.blue_edge_shift_px) > 0.25
    assert abs(measured.red_edge_shift_px) > 0.25
    assert measured.blue_edge_shift_px * measured.red_edge_shift_px < 0


def test_optical_matching_adds_dof_motion_ca_and_measured_sensor_grain():
    height, width = 96, 128
    checker = (((np.indices((height, width)).sum(axis=0) // 4) % 2) * 200 + 25).astype(np.uint8)
    sharp = np.repeat(checker[..., None], 3, axis=2)
    target = cv2.GaussianBlur(sharp, (0, 0), 1.2)
    target = np.clip(target.astype(np.int16) + np.random.default_rng(2).normal(0, 4, target.shape), 0, 255).astype(np.uint8)
    previous = np.roll(target, 3, axis=1)
    alpha = np.zeros((height, width), np.uint8)
    alpha[12:-12, 16:-16] = 255
    rgba = np.concatenate((sharp, alpha[..., None]), axis=2)
    matcher = PhysicalCameraMatcher(50.0)
    matched, info = matcher.match_rendered_head(rgba, target, previous_frame_bgr=previous, alpha=alpha,
                                                target_mask=alpha, random_seed=4)
    mask = alpha > 0
    before_lap = cv2.Laplacian(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    after_lap = cv2.Laplacian(cv2.cvtColor(matched[..., :3], cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    before = before_lap[mask].var()
    after = after_lap[mask].var()
    assert after < before
    assert matched.shape == rgba.shape and matched.dtype == np.uint8
    assert np.linalg.norm(info.measurements.motion_blur_vector_px) > 0.5
    assert info.directional_blur_length_px > 0
    assert info.measurements.grain_sigma > 0
    assert info.applied_grain_sigma == info.measurements.grain_sigma


def test_depth_of_field_blurs_out_of_focus_head_surface_more():
    height, width = 128, 160
    checker = (((np.indices((height, width)).sum(axis=0) // 3) % 2) * 200 + 25).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    alpha = np.zeros((height, width), np.uint8)
    alpha[12:-12, 8:-8] = 255
    rgba = np.concatenate((image, alpha[..., None]), axis=2)
    depth = np.ones((height, width), np.float32)
    depth[:, width // 2:] = 2.0
    output, info = PhysicalCameraMatcher(50.0).match_rendered_head(
        rgba, image, alpha=alpha, depth_m=depth, focus_distance_m=1.0, f_number=2.0,
        target_mask=alpha,
    )
    gray = cv2.cvtColor(output[..., :3], cv2.COLOR_BGR2GRAY)
    near_variance = cv2.Laplacian(gray[24:-24, 12:width // 2 - 8], cv2.CV_32F).var()
    far_variance = cv2.Laplacian(gray[24:-24, width // 2 + 8:-12], cv2.CV_32F).var()
    assert info.depth_of_field_sigma_px > 0
    assert far_variance < near_variance * 0.25


def test_sh_estimate_recovers_key_direction_and_rim_lobes_are_structured():
    height, width = 96, 96
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    z = np.sqrt(np.maximum(1.0 - x * x - y * y, 0.03))
    normals = np.stack((x * 0.5, y * 0.35, z), axis=-1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-6)
    truth = np.array([0.78, -0.18, 0.60], np.float32)
    truth /= np.linalg.norm(truth)
    irradiance = 0.24 + 0.72 * np.maximum(normals @ truth, 0.0)
    bgr_linear = irradiance[..., None] * np.array([0.82, 0.92, 1.0], np.float32)
    image = np.rint(np.clip(_linear_to_srgb(bgr_linear), 0, 1) * 255).astype(np.uint8)
    mask = (x * x + y * y) < 0.88
    estimate = DominantKeyLightAligner().estimate(image, normals, mask=mask)
    assert np.dot(estimate.direction_xyz, truth) > 0.78
    assert -180 <= estimate.azimuth_deg <= 180
    assert -90 <= estimate.elevation_deg <= 90
    assert 2000 <= estimate.kelvin_temp <= 12000
    assert estimate.sh_coefficients.shape == (9, 3)
    assert isinstance(estimate.rim_lobes, tuple)


def test_temporal_spatial_denoiser_reduces_uv_texture_flicker_without_washing_edges():
    height, width = 80, 96
    truth = np.full((height, width), 58, np.float32)
    truth[:, width // 2:] = 184
    base = np.repeat(truth[..., None], 3, axis=2)
    rng = np.random.default_rng(11)
    previous = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    current = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    denoiser = TemporalSpatialSwapDenoiser(temporal_strength=0.75, spatial_sigma=1.0)
    denoiser.process(previous, np.zeros((height, width, 2), np.float32))
    output = denoiser.process(current, np.zeros((height, width, 2), np.float32))
    current_error = np.mean(np.abs(current.astype(np.float32) - base))
    output_error = np.mean(np.abs(output.astype(np.float32) - base))
    edge_jump = float(output[:, width // 2 + 2].mean() - output[:, width // 2 - 2].mean())
    assert output_error < current_error * 0.78
    assert edge_jump > 90.0
    denoiser.reset()
    first = denoiser.process(current, np.zeros((height, width, 2), np.float32))
    assert np.mean(np.abs(first.astype(np.float32) - current)) < 5.0


def test_bundled_cc0_environment_probes_and_pore_normals_are_tileable():
    assets = load_cc0_preview_assets()
    assert len(assets.probes) == 5
    assert {probe.name for probe in assets.probes} == {
        "Neutral Studio", "Warm Indoor", "Cool Overcast", "Golden Hour", "Cinematic Rim"
    }
    assert all(probe.license_id == "CC0-1.0" and probe.sh_coefficients.shape == (9, 3) for probe in assets.probes)
    assert assets.metadata["license"] == "CC0-1.0"
    assert assets.micro_pore_normal.shape == (256, 256, 3)
    uv = np.array([[0.21, 0.48], [0.78, 0.81]], np.float32)
    a = sample_micro_pore_normal(uv, assets.micro_pore_normal, repeat=9.0)
    b = sample_micro_pore_normal(uv + 1.0 / 9.0, assets.micro_pore_normal, repeat=9.0)
    np.testing.assert_allclose(a, b, atol=2e-5)
