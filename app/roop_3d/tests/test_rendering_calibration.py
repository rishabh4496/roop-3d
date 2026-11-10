"""Color-temperature, energy-budget, and intrinsic-detail regressions."""
import numpy as np

from roop_3d.rendering.intrinsic_texture import IntrinsicTextureDecomposer
from roop_3d.rendering.physical_buffers import evaluate_sh_irradiance, shade_lambertian_sh
from roop_3d.rendering.skin_calibration import (
    SceneIllumination,
    UnbiasedSkinToneCalibrator,
    linear_to_srgb,
)
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer


def _lit_face(albedo_bgr, illumination, side=48):
    linear = np.broadcast_to(np.asarray(albedo_bgr, np.float32), (side, side, 3))
    return np.rint(np.clip(linear_to_srgb(linear * illumination.linear_bgr_gain), 0, 1) * 255).astype(np.uint8)


def test_skin_albedo_is_invariant_to_scene_color_temperature_and_exposure():
    calibrator = UnbiasedSkinToneCalibrator()
    intrinsic = np.array([0.27, 0.39, 0.56], np.float32)
    warm = SceneIllumination.from_temperature(2850, 0.68)
    cool = SceneIllumination.from_temperature(8200, 1.35)
    warm_face, cool_face = _lit_face(intrinsic, warm), _lit_face(intrinsic, cool)
    mask = np.ones(warm_face.shape[:2], np.uint8)

    warm_estimate = calibrator.decompose(warm_face, skin_mask=mask, illumination=warm)
    cool_estimate = calibrator.decompose(cool_face, skin_mask=mask, illumination=cool)

    np.testing.assert_allclose(np.median(warm_estimate.albedo_linear_bgr, axis=(0, 1)), intrinsic, atol=0.008)
    np.testing.assert_allclose(np.median(cool_estimate.albedo_linear_bgr, axis=(0, 1)), intrinsic, atol=0.008)
    np.testing.assert_allclose(warm_estimate.lab_median, cool_estimate.lab_median, atol=1.2)
    assert warm_estimate.melanin_median >= 0 and warm_estimate.hemoglobin_median >= 0


def test_albedo_preservation_ratio_controls_intrinsic_tone_before_target_light():
    calibrator = UnbiasedSkinToneCalibrator()
    source_albedo = np.array([0.20, 0.30, 0.45], np.float32)
    target_albedo = np.array([0.38, 0.28, 0.19], np.float32)
    source_light = SceneIllumination.from_temperature(3000, 0.8)
    target_light = SceneIllumination.from_temperature(7200, 1.1)
    source, target = _lit_face(source_albedo, source_light), _lit_face(target_albedo, target_light)
    mask = np.ones(source.shape[:2], np.uint8)

    preserve = calibrator.calibrate(source, target, source_skin_mask=mask, target_skin_mask=mask,
        source_illumination=source_light, target_illumination=target_light, albedo_preservation_ratio=1.0)
    adapt = calibrator.calibrate(source, target, source_skin_mask=mask, target_skin_mask=mask,
        source_illumination=source_light, target_illumination=target_light, albedo_preservation_ratio=0.0)
    preserve_albedo = calibrator.decompose(preserve, skin_mask=mask, illumination=target_light).albedo_linear_bgr
    adapt_albedo = calibrator.decompose(adapt, skin_mask=mask, illumination=target_light).albedo_linear_bgr

    np.testing.assert_allclose(np.median(preserve_albedo, axis=(0, 1)), source_albedo, atol=0.015)
    np.testing.assert_allclose(np.median(adapt_albedo, axis=(0, 1)), target_albedo, atol=0.015)


def test_sh_diffuse_and_specular_reflection_stay_within_incoming_energy():
    yy, xx = np.mgrid[-1:1:17j, -1:1:19j]
    normals = np.stack((xx, yy, np.ones_like(xx) * 0.75), axis=-1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    albedo = np.broadcast_to(np.array([0.24, 0.48, 0.72], np.float32), normals.shape).copy()
    sh = np.zeros((9, 3), np.float32)
    sh[0] = np.array([0.55, 0.62, 0.70], np.float32) / 0.886227
    sh[1] = np.array([0.10, 0.06, 0.02], np.float32) / 1.023328
    sh[3] = np.array([0.16, 0.19, 0.22], np.float32) / 1.023328
    incoming = np.maximum(evaluate_sh_irradiance(normals, sh), 0)
    outgoing = shade_lambertian_sh(albedo, normals, sh, specular_strength=1.0, shininess=20)

    assert np.all(outgoing >= 0)
    assert np.all(outgoing <= incoming + 1e-6)
    assert outgoing.sum() < incoming.sum()
    doubled = shade_lambertian_sh(albedo, normals, sh * 2, specular_strength=1.0, shininess=20)
    np.testing.assert_allclose(doubled, outgoing * 2, atol=1e-5)


def test_intrinsic_uv_cleanup_keeps_four_observed_detail_bands():
    side = 64
    y, x = np.mgrid[0:side, 0:side]
    smooth_light = 0.55 + 0.35 * x / (side - 1)
    pores = 0.018 * (((x + y) % 3) - 1)
    value = np.clip(smooth_light + pores, 0.08, 0.95)
    texture = np.rint(np.stack((value * 210, value * 185, value * 155), axis=-1)).astype(np.uint8)
    mask = np.ones((side, side), np.uint8)
    mask[:4] = 0

    result = IntrinsicTextureDecomposer().decompose(texture, mask)

    assert result.albedo_bgr.shape == texture.shape
    assert len(result.laplacian_bands) == len(result.displacement_bands_m) == 4
    assert result.displacement_uv_m.shape == mask.shape
    assert np.max(np.abs(result.displacement_uv_m)) < 0.002
    assert np.all(result.albedo_bgr[:4] == 0)


def test_uv_synthesis_exposes_opt_in_intrinsic_texture_bands():
    mesh = Mesh3D(
        np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
        uvs=np.array([[0, 0], [1, 0], [0.5, 1]], np.float32),
    )
    synthesizer = UVSynthesizer(
        mesh, uv_size=64, intrinsic_texture_decomposer=IntrinsicTextureDecomposer())
    y, x = np.mgrid[:64, :64]
    value = np.clip(0.45 + 0.35 * x / 63 + 0.015 * ((x + y) % 2), 0, 1)
    texture = np.rint(np.stack((value, value * 0.9, value * 0.8), axis=-1) * 255).astype(np.uint8)

    canonical, displacement, bands = synthesizer._intrinsic_texture(texture)

    assert canonical.shape == texture.shape
    assert displacement.shape == (64, 64)
    assert len(bands) == 4
    assert np.all(canonical[~synthesizer.face_texel_mask] == 0)


def test_physical_renderer_rasterizes_aligned_buffers_on_gpu():
    from roop_3d.core.pose import rotation_from_euler
    from roop_3d.reconstruction.full_head import HeadPose, head_roi
    from roop_3d.rendering.physical_buffers import PhysicalBufferRenderer
    from roop_3d.tests.test_full_head import head_asset

    asset = head_asset()
    pose = HeadPose(rotation_from_euler(0, 0, 0), np.array([0, 0, 0.9]),
                    np.array([[800, 0, 64], [0, 800, 64], [0, 0, 1]], np.float32))
    inverse = head_roi(pose.project(asset.vertices), 128)
    sh = np.zeros((9, 3), np.float32)
    sh[0] = 0.8 / 0.886227
    renderer = PhysicalBufferRenderer()
    try:
        buffers = renderer.render(asset.vertices, asset.faces, asset.uvs, asset.texture,
            asset.texel_confidence, pose, inverse, sh, side=128)
    finally:
        renderer.release()

    active = buffers.coverage > 0.5
    assert active.sum() > 100
    assert buffers.normals.shape == buffers.albedo.shape == buffers.sh_shaded.shape == (128, 128, 3)
    assert buffers.z_depth.shape == buffers.coverage.shape == (128, 128)
    np.testing.assert_allclose(np.linalg.norm(buffers.normals[active], axis=-1), 1, atol=2e-5)
    assert np.all(np.isfinite(buffers.z_depth[active]))
    assert np.all(np.isfinite(buffers.albedo[active]))
    assert np.all(np.isfinite(buffers.sh_shaded[active]))
    assert np.all(np.isinf(buffers.z_depth[~active]))
