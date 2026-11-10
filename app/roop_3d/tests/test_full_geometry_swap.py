"""Stage 5B source-geometry lock and full-head UV completion contracts."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from roop_3d.reconstruction.lbs import ArticulatedHeadLayer
else:
    ArticulatedHeadLayer = None

from roop_3d.reconstruction.uv_completion import FullHeadUVCompleter

requires_torch = pytest.mark.skipif(torch is None, reason="the articulated geometry layer requires PyTorch")


def _geometry_layer():
    vertices = torch.tensor([
        [-0.075, 0.075, 0.000], [-0.045, 0.105, 0.008], [0.045, 0.105, 0.008], [0.075, 0.075, 0.000],
        [-0.085, 0.015, 0.010], [0.085, 0.015, 0.010], [-0.055, -0.035, 0.000], [0.055, -0.035, 0.000],
        [0.000, 0.082, 0.015], [0.000, 0.122, 0.000],
    ], dtype=torch.float32)
    weights = torch.zeros((len(vertices), 5), dtype=torch.float32)
    weights[:, 0] = 1.0
    weights[6] = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0))
    weights[7] = torch.tensor((0.0, 0.0, 0.0, 0.0, 1.0))
    weights[8] = torch.tensor((0.4, 0.0, 0.6, 0.0, 0.0))
    weights[9] = torch.tensor((0.0, 0.0, 1.0, 0.0, 0.0))
    joints = torch.tensor([
        [0.0, 0.0, 0.0], [0.0, -0.025, 0.0], [0.0, 0.065, 0.0],
        [-0.03, 0.01, 0.0], [0.03, 0.01, 0.0],
    ], dtype=torch.float32)
    shape = torch.zeros((300, len(vertices), 3), dtype=torch.float32)
    shape[0, :, 0] = vertices[:, 0] * 0.30
    shape[1, :, 0] = torch.sign(vertices[:, 0]) * 0.006
    shape[2, :, 1] = vertices[:, 1] * 0.25
    expression = torch.zeros((100, len(vertices), 3), dtype=torch.float32)
    expression[0, 8, 1] = -0.012  # mouth opening moves soft tissue, not jaw/cheek bone widths
    expression[0, 9, 1] = 0.004
    layer = ArticulatedHeadLayer(vertices, weights, joints, shape_directions=shape,
                                 expression_directions=expression)
    return layer


def _vertices(layer, beta_source, beta_target=None, psi=None):
    batch = 1
    root = torch.tensor([[0.04, -0.16, 0.03]], dtype=torch.float32)
    neck = torch.tensor([[0.01, 0.02, -0.015]], dtype=torch.float32)
    jaw = torch.tensor([[0.08, 0.0, 0.0]], dtype=torch.float32)
    eyes = torch.tensor([[0.0, 0.12, -0.03, 0.001, -0.001, 0.0]], dtype=torch.float32)
    psi = torch.zeros((batch, 100), dtype=torch.float32) if psi is None else psi
    result, _ = layer(beta_source, psi, root, neck, jaw, eyes, beta_target=beta_target)
    return result[0].detach().cpu().numpy()


def _project(points):
    camera = np.asarray(points, np.float64).copy()
    camera[:, 2] += 1.25
    focal = 1450.0
    return np.column_stack((focal * camera[:, 0] / camera[:, 2] + 512.0,
                            focal * camera[:, 1] / camera[:, 2] + 512.0))


def _relative_error(actual, expected):
    return abs(actual - expected) / max(abs(expected), 1e-9)


@pytest.mark.parametrize("target_values", [
    (2.8, -2.1, 1.9),
    (-2.4, 3.0, -2.7),
    (0.0, 0.0, 0.0),
])
@requires_torch
def test_projected_source_jaw_and_cheek_widths_ignore_target_shape(target_values):
    layer = _geometry_layer()
    assert layer.source_identity_lock == 1.0
    beta_source = torch.zeros((1, 300), dtype=torch.float32)
    beta_source[0, :3] = torch.tensor((0.65, -0.35, 0.20))
    beta_target = torch.zeros((1, 300), dtype=torch.float32)
    beta_target[0, :3] = torch.tensor(target_values)
    psi_target = torch.zeros((1, 100), dtype=torch.float32)
    psi_target[0, 0] = 0.8

    source_neutral = _vertices(layer, beta_source)
    swapped = _vertices(layer, beta_source, beta_target=beta_target, psi=psi_target)
    source_pixels = _project(source_neutral)
    swapped_pixels = _project(swapped)

    source_jaw_width = np.ptp(source_pixels[[0, 1, 2, 3], 0])
    source_cheek_width = np.ptp(source_pixels[[4, 5], 0])
    swapped_jaw_width = np.ptp(swapped_pixels[[0, 1, 2, 3], 0])
    swapped_cheek_width = np.ptp(swapped_pixels[[4, 5], 0])
    assert _relative_error(swapped_jaw_width, source_jaw_width) < 0.005
    assert _relative_error(swapped_cheek_width, source_cheek_width) < 0.005


@requires_torch
def test_target_expression_is_converted_to_source_metric_scale():
    layer = _geometry_layer()
    beta_source = torch.zeros((1, 300), dtype=torch.float32)
    beta_target = torch.zeros((1, 300), dtype=torch.float32)
    beta_source[0, 0] = -0.8
    beta_target[0, 0] = 1.7
    psi = torch.zeros((1, 100), dtype=torch.float32)
    psi[0, 0] = 1.0

    normalized = layer.normalize_expression_for_source(psi, beta_source, beta_target)

    assert normalized.shape == psi.shape
    assert normalized[0, 0] == pytest.approx(
        float(layer._bone_scale(beta_source)[0] / layer._bone_scale(beta_target)[0]), rel=1e-6)


def test_full_head_uv_completer_stitches_and_tags_generated_texels():
    height = width = 32
    footprint = np.ones((height, width), bool)
    positions = np.zeros((height, width, 3), np.float32)
    positions[..., 0] = np.linspace(-0.05, 0.05, width)[None, :]
    positions[..., 1] = np.linspace(0.0, 0.08, height)[:, None]
    colors = np.zeros((2, height, width, 3), np.float32)
    colors[0] = (100, 135, 180)
    colors[1] = (104, 139, 184)
    weights = np.zeros((2, height, width), np.float32)
    weights[0, :, :19] = 1.0
    weights[1, :, 13:25] = 1.0

    result = FullHeadUVCompleter().complete(
        colors, weights, footprint, positions, mirror_x=0.0)

    assert result.texture.shape == (height, width, 4)
    assert np.all(result.texture[..., 3] == 255)
    assert np.all(result.source[weights.max(axis=0) > 0] == 1)
    assert np.all(result.source[:, 25:] == 2)
    assert result.stats["view_count"] == 2
    assert result.stats["completion_pipeline"] == "multiband+poisson+skin_symmetry+harmonic"
