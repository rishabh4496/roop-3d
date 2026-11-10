# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Original trainable heads for metric identity and asymmetric articulation.

These modules implement general architectural ideas found across public 3D
face research; they do not include third-party code, checkpoints, or datasets.
The project supplies ArcFace embeddings and geometric bases as inputs. Learned
weights must be trained or supplied by the application before predictions can
be treated as fitted measurements.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from roop_3d.reconstruction.lbs import ArticulatedHeadLayer


class ArcFaceToMetricShapeAdapter(nn.Module):
    """Pool a FaceSet's ArcFace features and regress 300 metric shape coefficients.

    Inputs are ``embeddings (B, N, 512)``, normalized sharpness in ``[0, 1]``,
    and FaceSet yaw in degrees. Repeated yaw bins receive lower weight, frontal
    views retain a modest quality preference, and embedding outliers below a
    cosine threshold are rejected. A best-view fallback keeps the operation
    defined when every view falls below the threshold.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        shape_dim: int = 300,
        hidden_dim: int = 512,
        yaw_bandwidth_degrees: float = 18.0,
        outlier_cosine_threshold: float = 0.25,
        coefficient_limit: float = 3.0,
    ) -> None:
        super().__init__()
        if embedding_dim != 512 or shape_dim != 300:
            raise ValueError("the FaceSet adapter contract is 512-D ArcFace to 300 shape coefficients")
        if yaw_bandwidth_degrees <= 0 or not -1 <= outlier_cosine_threshold < 1 or coefficient_limit <= 0:
            raise ValueError("invalid pooling or coefficient configuration")
        self.embedding_dim = embedding_dim
        self.shape_dim = shape_dim
        self.yaw_bandwidth_degrees = float(yaw_bandwidth_degrees)
        self.outlier_cosine_threshold = float(outlier_cosine_threshold)
        self.coefficient_limit = float(coefficient_limit)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, shape_dim),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def pool_views(self, embeddings: Tensor, sharpness: Tensor, yaw_degrees: Tensor) -> Tensor:
        """Return the robust, yaw-balanced unit-length FaceSet feature."""
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embeddings must have shape (B, N, 512)")
        if sharpness.shape != embeddings.shape[:2] or yaw_degrees.shape != embeddings.shape[:2]:
            raise ValueError("sharpness and yaw_degrees must both have shape (B, N)")
        vectors = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        sharp = sharpness.clamp(0.0, 1.0).clamp_min(0.05)
        yaw = yaw_degrees.clamp(-90.0, 90.0)

        yaw_delta = (yaw.unsqueeze(2) - yaw.unsqueeze(1)).abs()
        yaw_density = torch.exp(-yaw_delta / self.yaw_bandwidth_degrees).sum(dim=-1).clamp_min(1.0)
        diversity_weight = torch.rsqrt(yaw_density)
        frontal_weight = 0.25 + 0.75 * torch.cos(yaw * (math.pi / 180.0)).square()
        base = sharp * diversity_weight * frontal_weight
        first = F.normalize((vectors * base.unsqueeze(-1)).sum(dim=1), p=2, dim=-1, eps=1e-8)
        similarity = (vectors * first.unsqueeze(1)).sum(dim=-1)
        maximum_similarity = similarity.max(dim=1, keepdim=True).values
        keep = (similarity >= self.outlier_cosine_threshold) | (similarity >= maximum_similarity - 1e-7)
        robust_weight = base * keep.to(dtype=base.dtype)
        pooled = (vectors * robust_weight.unsqueeze(-1)).sum(dim=1)
        return F.normalize(pooled, p=2, dim=-1, eps=1e-8)

    def forward(self, embeddings: Tensor, sharpness: Tensor, yaw_degrees: Tensor) -> Tensor:
        pooled = self.pool_views(embeddings, sharpness, yaw_degrees)
        return torch.tanh(self.regressor(pooled)) * self.coefficient_limit

    def forward_faceset(self, faceset_or_views) -> Tensor:
        """Regress beta directly from a parsed ``.fsz`` document or its views.

        ``FaceView.embedding`` is the FaceSet's stored ArcFace feature;
        sharpness and yaw are read from that same view. This convenience path
        prepares tensors only; the tensor-only ``forward`` remains the ONNX
        seam.
        """
        views = getattr(faceset_or_views, "views", faceset_or_views)
        views = tuple(views)
        if not views:
            raise ValueError("a FaceSet must contain at least one reference view")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        embeddings = torch.as_tensor(
            [[list(view.embedding) for view in views]], device=device, dtype=dtype)
        sharpness = torch.as_tensor(
            [[float(view.quality.sharpness) for view in views]], device=device, dtype=dtype)
        yaw = torch.as_tensor(
            [[float(view.yaw) for view in views]], device=device, dtype=dtype)
        return self(embeddings, sharpness, yaw)


class _ImageEncoder(nn.Module):
    """Small spatial encoder with a fixed, TensorRT-friendly operation set."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.ReLU(inplace=False),
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1),
        )
        self.projection = nn.Sequential(nn.Linear(192, hidden_dim), nn.ReLU(inplace=False))

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.features(image))


class DisentangledFaceEncoder(nn.Module):
    """Encode a face crop into independent identity, expression, pose and light codes.

    The heads are deliberately separate so the renderer can consume only the
    factors needed for a frame. This is an architecture contract; meaningful
    predictions still require supplied or trained weights.
    """

    def __init__(self, *, identity_dim: int = 512, expression_dim: int = 100,
                 pose_dim: int = 6, illumination_dim: int = 27, hidden_dim: int = 256) -> None:
        super().__init__()
        dimensions = (identity_dim, expression_dim, pose_dim, illumination_dim, hidden_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Encoder dimensions must be positive integers")
        self.identity_dim, self.expression_dim = identity_dim, expression_dim
        self.pose_dim, self.illumination_dim = pose_dim, illumination_dim
        self.encoder = _ImageEncoder(hidden_dim)
        self.identity = nn.Linear(hidden_dim, identity_dim)
        self.expression = nn.Linear(hidden_dim, expression_dim)
        self.pose = nn.Linear(hidden_dim, pose_dim)
        self.illumination = nn.Linear(hidden_dim, illumination_dim)

    def forward(self, face_roi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.encoder(face_roi)
        return (F.normalize(self.identity(features), p=2, dim=-1, eps=1e-8),
                torch.tanh(self.expression(features)), torch.tanh(self.pose(features)),
                torch.tanh(self.illumination(features)))


class AsymmetricExpressionEncoder(nn.Module):
    """Regress 100 expression coefficients and unsmoothed local vertex offsets.

    The per-vertex residual head is intentionally direct: no temporal averaging,
    symmetry constraint, or attenuation is applied, allowing one-sided facial
    actions to survive the model output. Inputs are normalized face crops
    ``(B, 3, H, W)`` and outputs are ``(B, 100)`` and ``(B, V, 3)``.
    """

    def __init__(self, vertex_count: int, *, expression_dim: int = 100, hidden_dim: int = 256) -> None:
        super().__init__()
        if vertex_count <= 0 or expression_dim != 100:
            raise ValueError("vertex_count must be positive and expression_dim must be 100")
        self.vertex_count = vertex_count
        self.encoder = _ImageEncoder(hidden_dim)
        self.expression = nn.Linear(hidden_dim, expression_dim)
        self.local_residual = nn.Linear(hidden_dim, vertex_count * 3)
        nn.init.zeros_(self.expression.weight)
        nn.init.zeros_(self.expression.bias)
        nn.init.zeros_(self.local_residual.weight)
        nn.init.zeros_(self.local_residual.bias)

    def forward(self, face_roi: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(face_roi)
        psi = self.expression(features)
        residual = self.local_residual(features).reshape(-1, self.vertex_count, 3)
        return psi, residual


class LipPhonemeRegressor(nn.Module):
    """Dedicated frame-local mouth branch for jaw motion and inner-lip offsets.

    No recurrent state or temporal filter is present, so a plosive or opening
    is not delayed by model-side smoothing. ``phoneme_logits`` order is
    silence, P, B, M, A, E, I, O, U. The jaw and vertex outputs are direct.
    """

    PHONEMES = ("silence", "P", "B", "M", "A", "E", "I", "O", "U")

    def __init__(self, inner_lip_vertex_count: int, *, hidden_dim: int = 192) -> None:
        super().__init__()
        if inner_lip_vertex_count <= 0:
            raise ValueError("inner_lip_vertex_count must be positive")
        self.inner_lip_vertex_count = inner_lip_vertex_count
        self.encoder = _ImageEncoder(hidden_dim)
        self.jaw_pose = nn.Linear(hidden_dim, 3)
        self.inner_lip_residual = nn.Linear(hidden_dim, inner_lip_vertex_count * 3)
        self.phoneme = nn.Linear(hidden_dim, len(self.PHONEMES))
        nn.init.zeros_(self.jaw_pose.weight)
        nn.init.zeros_(self.jaw_pose.bias)
        nn.init.zeros_(self.inner_lip_residual.weight)
        nn.init.zeros_(self.inner_lip_residual.bias)
        nn.init.zeros_(self.phoneme.weight)
        nn.init.zeros_(self.phoneme.bias)

    def forward(self, mouth_roi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encoder(mouth_roi)
        jaw = self.jaw_pose(features)
        residual = self.inner_lip_residual(features).reshape(-1, self.inner_lip_vertex_count, 3)
        return jaw, residual, self.phoneme(features)


class ArticulatedPoseEncoder(nn.Module):
    """Regress root, neck and shared six-DoF eye poses from a face crop."""

    def __init__(
        self, *, hidden_dim: int = 192, pose_limit_radians: float = math.pi,
        eye_translation_limit_metres: float = 0.02,
    ) -> None:
        super().__init__()
        self.pose_limit_radians = float(pose_limit_radians)
        self.eye_translation_limit_metres = float(eye_translation_limit_metres)
        self.encoder = _ImageEncoder(hidden_dim)
        self.pose = nn.Linear(hidden_dim, 12)
        nn.init.zeros_(self.pose.weight)
        nn.init.zeros_(self.pose.bias)

    def forward(self, face_roi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = torch.tanh(self.pose(self.encoder(face_roi)))
        root = raw[:, :3] * self.pose_limit_radians
        neck = raw[:, 3:6] * self.pose_limit_radians
        eyes = torch.cat((
            raw[:, 6:9] * self.pose_limit_radians,
            raw[:, 9:12] * self.eye_translation_limit_metres,
        ), dim=1)
        return root, neck, eyes


class MonocularArticulatedHead(nn.Module):
    """Exportable image/ArcFace-to-LBS graph for one frame and one FaceSet.

    The graph accepts normalized face and mouth crops plus the FaceSet's
    precomputed 512-D ArcFace embeddings and quality metadata. Its outputs are
    vertices, marched cheek landmarks, beta, psi, jaw pose, and phoneme logits.
    Supplying joint/shape bases and trained weights remains an application
    responsibility; the class does not claim to ship recovered subject shape.
    """

    def __init__(
        self,
        skinning_layer: ArticulatedHeadLayer,
        *,
        inner_lip_vertex_indices: Tensor,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        indices = torch.as_tensor(inner_lip_vertex_indices, dtype=torch.long).reshape(-1)
        if indices.numel() == 0 or (indices < 0).any() or (indices >= skinning_layer.template_vertices.shape[0]).any():
            raise ValueError("inner_lip_vertex_indices must contain valid template vertex indices")
        self.skinning_layer = skinning_layer
        self.register_buffer("inner_lip_basis", F.one_hot(
            indices, num_classes=skinning_layer.template_vertices.shape[0]).to(dtype=torch.float32).transpose(0, 1).contiguous())
        self.identity = ArcFaceToMetricShapeAdapter(hidden_dim=hidden_dim)
        self.expression = AsymmetricExpressionEncoder(
            skinning_layer.template_vertices.shape[0], hidden_dim=hidden_dim)
        self.lips = LipPhonemeRegressor(indices.numel(), hidden_dim=hidden_dim)
        self.pose = ArticulatedPoseEncoder(hidden_dim=hidden_dim)

    def forward(
        self,
        face_roi: Tensor,
        mouth_roi: Tensor,
        reference_embeddings: Tensor,
        reference_sharpness: Tensor,
        reference_yaw_degrees: Tensor,
        beta_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        beta = self.identity(reference_embeddings, reference_sharpness, reference_yaw_degrees)
        psi, asymmetric_residual = self.expression(face_roi)
        jaw, inner_lip, phoneme_logits = self.lips(mouth_roi)
        global_pose, neck_pose, theta_eyes = self.pose(face_roi)
        lip_residual = torch.einsum("vl,blc->bvc", self.inner_lip_basis, inner_lip)
        vertices, contour = self.skinning_layer(
            beta, psi, global_pose, neck_pose, jaw, theta_eyes,
            vertex_residual=asymmetric_residual + lip_residual,
            beta_target=beta_target,
        )
        return vertices, contour, beta, psi, jaw, phoneme_logits


def export_articulated_head_onnx(
    model: MonocularArticulatedHead,
    output_path: str | Path,
    *,
    face_size: int = 224,
    mouth_size: int = 96,
    reference_views: int = 4,
    opset_version: int = 17,
) -> Path:
    """Export the full image, identity-pooling, regressor and LBS graph.

    Batch and reference-view dimensions are dynamic. Image dimensions are fixed
    to the example resolution; this gives TensorRT a stable optimization
    profile while allowing variable FaceSet sizes. TensorRT can consume the
    resulting ONNX directly; TensorRT itself remains an optional runtime extra.
    The export accepts ``beta_target`` explicitly so target expression scale
    can be normalized while the output mesh stays locked to the FaceSet beta.
    """
    if face_size <= 0 or mouth_size <= 0 or reference_views <= 0:
        raise ValueError("export example dimensions must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    examples = (
        torch.zeros((1, 3, face_size, face_size), device=device, dtype=dtype),
        torch.zeros((1, 3, mouth_size, mouth_size), device=device, dtype=dtype),
        torch.zeros((1, reference_views, 512), device=device, dtype=dtype),
        torch.ones((1, reference_views), device=device, dtype=dtype),
        torch.zeros((1, reference_views), device=device, dtype=dtype),
        torch.zeros((1, model.skinning_layer.SHAPE_DIM), device=device, dtype=dtype),
    )
    torch.onnx.export(
        model,
        examples,
        str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=("face_roi", "mouth_roi", "reference_embeddings", "reference_sharpness",
                     "reference_yaw_degrees", "beta_target"),
        output_names=("vertices", "cheek_contour", "beta", "psi", "theta_jaw", "phoneme_logits"),
        dynamic_axes={
            "face_roi": {0: "batch"},
            "mouth_roi": {0: "batch"},
            "reference_embeddings": {0: "batch", 1: "reference_views"},
            "reference_sharpness": {0: "batch", 1: "reference_views"},
            "reference_yaw_degrees": {0: "batch", 1: "reference_views"},
            "beta_target": {0: "batch"},
            "vertices": {0: "batch"},
            "cheek_contour": {0: "batch"},
            "beta": {0: "batch"},
            "psi": {0: "batch"},
            "theta_jaw": {0: "batch"},
            "phoneme_logits": {0: "batch"},
        },
        dynamo=False,
    )
    return path


def _export_dynamic_batch(model: nn.Module, output_path: str | Path, examples: tuple[Tensor, ...], *,
                          input_names: tuple[str, ...], output_names: tuple[str, ...],
                          opset_version: int = 17) -> Path:
    """Export a fixed-spatial, dynamic-batch model for TensorRT profile building."""
    if len(examples) != len(input_names) or not examples or not output_names:
        raise ValueError("ONNX inputs and names must have matching nonempty contracts")
    if any(example.ndim < 1 or example.shape[0] != 1 for example in examples):
        raise ValueError("Export examples must have batch size one")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    torch.onnx.export(model.eval(), examples, str(destination), export_params=True,
                      opset_version=opset_version, do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, dynamo=False)
    return destination


def export_disentangled_face_encoder_onnx(model: DisentangledFaceEncoder, output_path: str | Path, *,
                                          face_size: int = 224, opset_version: int = 17) -> Path:
    if face_size < 16:
        raise ValueError("face_size must be at least 16")
    parameter = next(model.parameters())
    example = torch.zeros((1, 3, face_size, face_size), device=parameter.device, dtype=parameter.dtype)
    return _export_dynamic_batch(model, output_path, (example,), input_names=("face_roi",),
        output_names=("identity_code", "expression_code", "pose_code", "illumination_code"),
        opset_version=opset_version)


def export_lip_phoneme_regressor_onnx(model: LipPhonemeRegressor, output_path: str | Path, *,
                                      mouth_size: int = 96, opset_version: int = 17) -> Path:
    if mouth_size < 16:
        raise ValueError("mouth_size must be at least 16")
    parameter = next(model.parameters())
    example = torch.zeros((1, 3, mouth_size, mouth_size), device=parameter.device, dtype=parameter.dtype)
    return _export_dynamic_batch(model, output_path, (example,), input_names=("mouth_roi",),
        output_names=("jaw_pose", "inner_lip_residual", "phoneme_logits"), opset_version=opset_version)
