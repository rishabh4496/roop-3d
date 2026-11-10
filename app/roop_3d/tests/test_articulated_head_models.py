# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Contract tests for the optional, clean-room articulated reconstruction graph."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="the articulated reconstruction extra requires PyTorch")
if torch is not None:
    from roop_3d.reconstruction.lbs import ArticulatedHeadLayer
    from roop_3d.reconstruction.monocular_regressors import (
        ArcFaceToMetricShapeAdapter,
        AsymmetricExpressionEncoder,
        LipPhonemeRegressor,
        MonocularArticulatedHead,
        export_articulated_head_onnx,
    )


def _layer(*, shape_directions=None, expression_directions=None):
    vertices = torch.tensor([
        [-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 1.0, 0.0],
    ])
    weights = torch.zeros((6, 5))
    weights[:, 0] = 1.0
    weights[5] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    joints = torch.tensor([
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.8, 0.0],
        [-0.5, 0.2, 0.0], [0.5, 0.2, 0.0],
    ])
    contour = torch.tensor([
        [[0, 1]],
        [[2, 3]],
    ])
    return ArticulatedHeadLayer(
        vertices, weights, joints,
        shape_directions=shape_directions,
        expression_directions=expression_directions,
        contour_candidates=contour,
    ), vertices


def _neutral_inputs(batch=1):
    return (
        torch.zeros(batch, 300), torch.zeros(batch, 100),
        torch.zeros(batch, 3), torch.zeros(batch, 3),
        torch.zeros(batch, 3), torch.zeros(batch, 6),
    )


def test_lbs_root_pose_keeps_template_shape_and_marches_yaw_contour():
    layer, template = _layer()
    beta, psi, root, neck, jaw, eyes = _neutral_inputs(batch=2)
    root[0, 1] = -0.2
    root[1, 1] = 0.2

    vertices, contour = layer(beta, psi, root, neck, jaw, eyes)

    torch.testing.assert_close(vertices[:, 4], template[4].unsqueeze(0).expand(2, -1), atol=1e-6, rtol=0)
    extent = 2.0 * torch.cos(torch.tensor(0.2))
    torch.testing.assert_close(contour[:, 0, 0], torch.stack((-extent, extent)))


def test_shape_and_expression_directions_are_applied_in_metric_vertex_space():
    shape = torch.zeros(300, 6, 3)
    shape[0, 0, 0] = 0.01
    expression = torch.zeros(100, 6, 3)
    expression[0, 1, 1] = -0.02
    layer, template = _layer(shape_directions=shape, expression_directions=expression)
    beta, psi, root, neck, jaw, eyes = _neutral_inputs()
    beta[0, 0] = 2.0
    psi[0, 0] = 0.5

    vertices, _ = layer(beta, psi, root, neck, jaw, eyes)

    torch.testing.assert_close(vertices[0, 0], template[0] + torch.tensor([0.02, 0.0, 0.0]))
    torch.testing.assert_close(vertices[0, 1], template[1] + torch.tensor([0.0, -0.01, 0.0]))


def test_jaw_joint_moves_only_vertices_with_jaw_skinning_weight():
    layer, template = _layer()
    beta, psi, root, neck, jaw, eyes = _neutral_inputs()
    jaw[0, 0] = 0.5

    vertices, _ = layer(beta, psi, root, neck, jaw, eyes)

    torch.testing.assert_close(vertices[0, :5], template[:5])
    assert not torch.allclose(vertices[0, 5], template[5])


def test_neck_rotation_activates_pose_corrective_basis():
    pose_directions = torch.zeros(36, 6, 3)
    pose_directions[5, 0, 2] = 0.1  # neck rotation matrix row 1, column 2
    layer, template = _layer()
    # Rebuild with the supplied pose-corrective basis.
    layer = ArticulatedHeadLayer(
        layer.template_vertices, layer.skinning_weights, layer.rest_joints,
        pose_directions=pose_directions,
    )
    beta, psi, root, neck, jaw, eyes = _neutral_inputs()
    neck[0, 0] = 0.2

    vertices, _ = layer(beta, psi, root, neck, jaw, eyes)

    assert vertices[0, 0, 2] < template[0, 2]


def test_arcface_pool_rejects_embedding_outlier_and_normalizes_result():
    adapter = ArcFaceToMetricShapeAdapter(hidden_dim=32)
    embeddings = torch.zeros(1, 3, 512)
    embeddings[0, 0, 0] = 1.0
    embeddings[0, 1, 0] = 1.0
    embeddings[0, 2, 0] = -1.0
    pooled = adapter.pool_views(embeddings, torch.ones(1, 3), torch.tensor([[-30.0, 30.0, 0.0]]))

    assert pooled.shape == (1, 512)
    assert pooled[0, 0] > 0.999
    torch.testing.assert_close(pooled.norm(dim=-1), torch.ones(1))
    coefficients = adapter(embeddings, torch.ones(1, 3), torch.zeros(1, 3))
    assert coefficients.shape == (1, 300)
    assert coefficients.abs().max() <= adapter.coefficient_limit
    views = [
        SimpleNamespace(embedding=tuple(embeddings[0, index].tolist()),
                        quality=SimpleNamespace(sharpness=1.0), yaw=(-20.0 + index * 20.0))
        for index in range(3)
    ]
    assert adapter.forward_faceset(SimpleNamespace(views=views)).shape == (1, 300)


def test_expression_and_lip_heads_keep_direct_asymmetric_residual_outputs():
    expression = AsymmetricExpressionEncoder(3, hidden_dim=16)
    with torch.no_grad():
        expression.local_residual.bias[0:3] = torch.tensor([0.2, -0.1, 0.05])
    psi, local = expression(torch.zeros(1, 3, 64, 64))
    assert psi.shape == (1, 100)
    torch.testing.assert_close(local[0, 0], torch.tensor([0.2, -0.1, 0.05]))

    lips = LipPhonemeRegressor(2, hidden_dim=16)
    with torch.no_grad():
        lips.jaw_pose.weight.zero_()
        lips.jaw_pose.bias.copy_(torch.tensor([0.1, 0.0, -0.2]))
        lips.inner_lip_residual.bias[0:3] = torch.tensor([-0.03, 0.0, 0.01])
    jaw, inner_lip, logits = lips(torch.zeros(1, 3, 48, 64))
    torch.testing.assert_close(jaw[0], torch.tensor([0.1, 0.0, -0.2]))
    torch.testing.assert_close(inner_lip[0, 0], torch.tensor([-0.03, 0.0, 0.01]))
    assert logits.shape == (1, len(LipPhonemeRegressor.PHONEMES))


def test_full_image_identity_lbs_graph_exports_and_passes_onnx_checker(tmp_path):
    onnx = pytest.importorskip("onnx", reason="ONNX is required only for export validation")
    layer, _ = _layer()
    model = MonocularArticulatedHead(layer, inner_lip_vertex_indices=torch.tensor([4, 5]), hidden_dim=16)
    destination = export_articulated_head_onnx(
        model, tmp_path / "articulated.onnx", face_size=32, mouth_size=32, reference_views=2,
    )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    assert {value.name for value in graph.graph.output} >= {
        "vertices", "cheek_contour", "beta", "psi", "theta_jaw", "phoneme_logits",
    }
