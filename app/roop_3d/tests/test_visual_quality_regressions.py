"""Deterministic compositing contracts on synthetic BGR images and semantic maps.

These fixtures bound arithmetic, mask support and relative seam/detail behavior.
They do not establish real-video Delta-E/SSIM/identity scores, subpixel landmark
accuracy, source-pigment recovery, tongue detection or GPU residency/throughput.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.compositor.boundary_fusion import (
    BlendMode, feather_anatomical_mask, fuse_crop, fuse_warped, warp_crop,
)
from roop_3d.compositor.color_match import estimate_lab_stats, estimate_ycbcr_stats
from roop_3d.compositor.detail_temporal import TemporalDetailStabilizer
from roop_3d.compositor.occlusion_analysis import OcclusionAnalyzer
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor, RobustOptions
from roop_3d.compositor.quality import inject_target_skin_micro_texture, transfer_fine_texture
from roop_3d.core.errors import DomainError


class SemanticMap:
    """A supplied parser-output fixture at the external semantic inference boundary."""
    def __init__(self, labels):
        self.labels = np.asarray(labels, np.uint8)

    def predict(self, image):
        labels = cv2.resize(self.labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return np.eye(19, dtype=np.float32)[labels]


def labels_with_skin_and_nose(side=128):
    labels = np.full((side, side), 4, np.uint8)  # protected from color/detail sampling, not from identity replacement
    labels[32:96, 32:64] = 1
    labels[32:96, 64:96] = 10
    labels[80:96, 100:116] = 12
    return labels


def compose_observed(target, crop, labels, *, mode="standard", options=None, foreground=None, method="identity_swap"):
    parser = SemanticMap(labels)
    analyzer = OcclusionAnalyzer(parser=parser, net_names=()) if mode == "robust" else None
    compositor = FrameCompositor(parser=parser, analyzer=analyzer)
    compositor.detail_temporal = TemporalDetailStabilizer(use_cuda_flow=False)
    side = crop.shape[0]
    item = SimpleNamespace(track_id=7, image=crop, inverse=np.eye(2, 3, dtype=np.float32), method=method)
    points = np.array(((8, 8), (side - 9, 8), (side - 9, side - 9), (8, side - 9)), np.float32)
    options = options or CompositeOptions()
    options = replace(
        options, occlusion_mode=mode, neck_protection=False, hair_matting=False, dense_shading=0,
        robust=RobustOptions(
            analysis_size=128, mask_flow=False, linear_light=False, refine_matte=False,
            edge_decontamination=False, sharpness_matching=False, local_illumination=0, texture_strength=0))
    observed = {}

    def observe(stage, image, inverse, alpha):
        if stage == "color":
            observed["color"] = image.copy()

    try:
        # Robust parser trust needs three consistent frames; no flow history or
        # learned network is fabricated to bypass that consumer-visible gate.
        for index in range(3 if mode == "robust" else 1):
            output = compositor.compose(target, (item,), {7: points}, options=options, frame_index=index,
                                        occluder_masks={} if foreground is None else {7: foreground}, observer=observe)
        return output, observed["color"], dict(compositor.last_reports[7])
    finally:
        compositor.release()


def test_anatomical_sdf_feather_is_inward_smoothstep_with_a_known_flat_edge():
    mask = np.zeros((96, 96), np.float32)
    mask[12:84, 12:84] = 1
    alpha = feather_anatomical_mask(mask, 8)
    np.testing.assert_allclose(alpha[48, [12, 13, 15, 19]], [0.04296875, 0.15625, 0.5, 1.0], atol=1e-6)
    np.testing.assert_array_equal(alpha[mask == 0], np.zeros(np.count_nonzero(mask == 0), np.float32))
    assert np.all(np.diff(alpha[48, 12:20]) > 0)
    assert alpha[48, 48] == 1.0
    uncertain = mask * 0.7
    assert np.all(feather_anatomical_mask(uncertain, 8) <= uncertain)
    clipped = feather_anatomical_mask(np.ones((32, 32), np.float32), 8)
    assert 0 < clipped[0, 16] < clipped[1, 16] < clipped[16, 16] == 1


def seam_scene():
    y, x = np.mgrid[:160, :160]
    target = np.repeat((45 + x // 8 + 5 * (y % 2))[..., None], 3, axis=2).astype(np.uint8)
    source = np.repeat((175 + y // 8 + 8 * (x % 3 == 0))[..., None], 3, axis=2).astype(np.uint8)
    mask = np.zeros((160, 160), np.float32)
    mask[32:128, 32:128] = 1
    return target, source, mask


def test_four_level_multiband_is_the_default_and_softens_the_synthetic_seam():
    target, source, support = seam_scene()
    alpha = feather_anatomical_mask(support, 8)
    defaults = CompositeOptions()
    assert defaults.mode == BlendMode.MULTI_BAND and defaults.pyramid_levels == 4
    actual = fuse_warped(target, source, alpha)
    explicit = fuse_warped(target, source, alpha, mode=BlendMode.MULTI_BAND, levels=4)
    one_level = fuse_warped(target, source, alpha, mode=BlendMode.MULTI_BAND, levels=1)
    np.testing.assert_array_equal(actual, explicit)
    assert np.max(np.abs(actual.astype(int) - one_level.astype(int))) > 1
    hard = fuse_warped(target, source, support, mode=BlendMode.FEATHER)
    edge_jump = lambda image: np.abs(np.diff(image[80, 28:44, 0].astype(float))).max()
    assert edge_jump(actual) < edge_jump(hard)
    assert actual[80, 80, 0] > target[80, 80, 0] + 50
    np.testing.assert_array_equal(actual[alpha == 0], target[alpha == 0])


def test_frame_compositor_uses_the_default_four_level_fusion_in_its_rendered_pixels():
    target, crop, support = seam_scene()
    labels = support.astype(np.uint8)
    options = CompositeOptions(color_strength=0, skin_texture_retention=0)
    implicit, _, _ = compose_observed(target, crop, labels, options=options)
    explicit, _, _ = compose_observed(
        target, crop, labels, options=replace(options, mode=BlendMode.MULTI_BAND, pyramid_levels=4))
    shallow, _, _ = compose_observed(target, crop, labels, options=replace(options, pyramid_levels=1))
    np.testing.assert_array_equal(implicit, explicit)
    assert np.max(np.abs(implicit.astype(int) - shallow.astype(int))) > 1
    np.testing.assert_array_equal(implicit[support == 0], target[support == 0])


@pytest.mark.parametrize("linear_light", [False, True])
def test_multiband_preserves_every_zero_alpha_pixel_including_subpixel_occluder_holes(linear_light):
    target = np.random.default_rng(91).integers(20, 220, (160, 160, 3), dtype=np.uint8)
    crop = np.full((96, 96, 3), (205, 150, 75), np.uint8)
    mask = np.zeros((96, 96), np.float32)
    mask[8:88, 8:88] = 1
    mask[32:64, 32:64] = 0
    matrix = np.array(((1, 0, 24.5), (0, 1, 24.5)), np.float32)
    _, alpha = warp_crop(crop, matrix, target.shape, mask)
    actual = fuse_crop(target, crop, matrix, mask, linear_light=linear_light)
    np.testing.assert_array_equal(actual[alpha == 0], target[alpha == 0])
    np.testing.assert_array_equal(actual[59:86, 59:86], target[59:86, 59:86])
    assert np.any(actual[40:48, 40:48] != target[40:48, 40:48])


def test_hidden_source_colors_cannot_contaminate_the_multiband_seam():
    target, source, support = seam_scene()
    support[68:92, 68:92] = 0
    alpha = feather_anatomical_mask(support, 6)
    black_border, bright_border = source.copy(), source.copy()
    black_border[alpha == 0] = 0
    bright_border[alpha == 0] = (255, 0, 255)
    first = fuse_warped(target, black_border, alpha)
    second = fuse_warped(target, bright_border, alpha)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[68:92, 68:92], target[68:92, 68:92])


@pytest.mark.parametrize("mode", ["standard", "robust"])
@pytest.mark.parametrize("method", ["lab", "ycbcr", "histogram"])
def test_color_sampling_and_correction_use_skin_plus_nose_not_eyes_or_lips(mode, method):
    labels = labels_with_skin_and_nose()
    skin = (labels == 1) | (labels == 10)
    crop = np.full((128, 128, 3), (70, 90, 110), np.uint8)
    target = np.full_like(crop, (240, 25, 210))
    target[skin] = (120, 150, 180)
    changed_features = target.copy()
    changed_features[~skin] = (25, 220, 40)
    options = CompositeOptions(color_method=method, color_strength=1, skin_texture_retention=0)
    _, corrected, report = compose_observed(target, crop, labels, mode=mode, options=options)
    _, other, _ = compose_observed(changed_features, crop, labels, mode=mode, options=options)
    np.testing.assert_array_equal(corrected, other)
    np.testing.assert_array_equal(corrected[~skin], crop[~skin])
    before = np.abs(crop[skin].astype(float) - target[skin]).mean()
    for class_id in (1, 10):
        selected = labels == class_id
        assert np.abs(corrected[selected].astype(float) - target[selected]).mean() < before / 4
    assert report["color_matching"]["state"] == "APPLIED"
    assert report["color_matching"]["classes"] == [1, 10]
    if mode == "robust":
        assert report["parser_trusted"]


@pytest.mark.parametrize("mode", ["standard", "robust"])
@pytest.mark.parametrize("samples", [0, 1])
def test_insufficient_skin_keeps_swap_colors_and_reports_unavailable_without_face_wide_fallback(mode, samples):
    labels = np.full((128, 128), 4, np.uint8)
    if samples:
        labels[64, 64] = 1
    crop = np.full((128, 128, 3), (65, 90, 120), np.uint8)
    target = np.full_like(crop, (160, 180, 200))
    _, observed, report = compose_observed(target, crop, labels, mode=mode,
                                          options=CompositeOptions(color_strength=1))
    np.testing.assert_array_equal(observed, crop)
    color = report["color_matching"]
    assert color["state"] == "UNAVAILABLE" and color["samples"] < 2
    assert color["reason"] == "SKIN_NOSE_STATISTICS_UNAVAILABLE"


@pytest.mark.parametrize("estimate", [estimate_lab_stats, estimate_ycbcr_stats])
def test_sparse_color_statistics_never_reintroduce_forbidden_or_shadow_samples(estimate):
    source = np.full((16, 16, 3), 100, np.uint8)
    target = np.full_like(source, 150)
    skin = np.zeros((16, 16), np.float32)
    skin[8, 4:12] = 1
    forbidden = skin.copy()
    forbidden[8, 4:6] = 0
    target[8, 5] = 0  # only one unoccluded, non-shadow sample remains
    with pytest.raises(DomainError) as failure:
        estimate(source, target, skin, forbidden)
    assert failure.value.code == "COLOR_MATCH_MASK_EMPTY"


def detail_image(side=128):
    _, x = np.mgrid[:side, :side]
    band = 24 * np.cos(2 * np.pi * x / 6)
    return np.repeat((128 + band)[..., None], 3, axis=2).round().astype(np.uint8)


def retain_detail(source, target, skin, alpha, method):
    if method == "post_restoration":
        return inject_target_skin_micro_texture(source, target, skin, alpha, injection_rate=1)
    return transfer_fine_texture(source, target, alpha, skin, np.ones(skin.shape, np.float32), strength=1)[0]


@pytest.mark.parametrize("method", ["post_restoration", "robust_fusion"])
def test_target_dog_detail_is_retained_only_in_unoccluded_skin(method):
    source = np.full((128, 128, 3), 128, np.uint8)
    target = detail_image()
    skin = np.zeros((128, 128), np.float32)
    skin[16:112, 16:112] = 1
    alpha = np.ones_like(skin)
    alpha[48:80, 48:80] = 0
    actual = retain_detail(source, target, skin, alpha, method)
    trusted = (skin * alpha) > 0.5
    residual = actual[..., 0].astype(float) - 128
    assert residual[trusted].std() > 1
    assert np.corrcoef(residual[trusted], target[..., 0][trusted].astype(float) - 128)[0, 1] > 0.9
    np.testing.assert_array_equal(actual[~trusted], source[~trusted])
    contaminated = target.copy()
    contaminated[~trusted] = (0, 255, 0)
    np.testing.assert_array_equal(retain_detail(source, contaminated, skin, alpha, method), actual)


@pytest.mark.parametrize("method", ["post_restoration", "robust_fusion"])
def test_target_detail_does_not_double_existing_texture_or_invent_it_from_feature_edges(method):
    textured = detail_image()
    ones = np.ones((128, 128), np.float32)
    np.testing.assert_array_equal(retain_detail(textured, textured, ones, ones, method), textured)
    smooth = np.full_like(textured, 128)
    skin = np.zeros_like(ones)
    skin[24:104, 24:104] = 1
    target = smooth.copy()
    target[skin == 0] = (0, 255, 0)  # high-contrast eyes/lips/foreground are not skin detail
    np.testing.assert_array_equal(retain_detail(smooth, target, skin, ones, method), smooth)


@pytest.mark.parametrize("mode", ["standard", "robust"])
def test_compositor_reinjects_target_detail_into_skin_and_nose_but_not_protected_features(mode):
    labels = labels_with_skin_and_nose()
    skin = (labels == 1) | (labels == 10)
    crop = np.full((128, 128, 3), 128, np.uint8)
    target = detail_image()
    foreground = np.zeros((128, 128), np.float32)
    foreground[56:72, 56:88] = 1
    options = CompositeOptions(color_strength=0, skin_texture_retention=1)
    output, retained, report = compose_observed(
        target, crop, labels, mode=mode, foreground=foreground, options=options,
        method="dense_3d_correspondence_neural_ultramax")
    np.testing.assert_array_equal(retained[~skin], crop[~skin])
    np.testing.assert_array_equal(retained[foreground == 1], crop[foreground == 1])
    for class_id in (1, 10):
        trusted = (labels == class_id) & (foreground == 0)
        assert retained[..., 0][trusted].std() > 1
    np.testing.assert_array_equal(output[60:68, 60:84], target[60:68, 60:84])
    assert report["skin_microtexture"]["source"] == "target_frame"
    assert report["skin_microtexture"]["method"] == "skin_normalized_dog"
