# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Differentiable articulated head skinning with export-friendly tensor operations.

The rig has four articulated groups (root, neck, jaw, and conjugate eyes) and
five skinning transforms: the shared eye pose is applied around each eyeball's
own pivot. Geometry is supplied by the caller; this module contains no model
assets or code from third-party face projects.

Tensor contract (all coordinates in the template's metric frame):
* beta: (B, 300), psi: (B, 100)
* global_pose, neck_pose, theta_jaw: axis-angle rotations (B, 3), radians
* theta_eyes: conjugate eye rotation then translation (B, 6), radians/metres
* vertices: output (B, V, 3); dynamic contour landmarks: (B, K, 3)

The forward graph uses fixed topology and ordinary PyTorch tensor ops, so the
same module can be exported to ONNX and used as a TensorRT graph component.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def axis_angle_to_matrix(rotation: Tensor) -> Tensor:
    """Convert (..., 3) axis-angle vectors to rotation matrices without Python loops."""
    x, y, z = rotation.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*rotation.shape[:-1], 3, 3)
    angle2 = (rotation * rotation).sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle2 + 1e-12)
    # Taylor limits keep the identity branch well behaved and differentiable.
    a = torch.where(angle2 > 1e-8, torch.sin(angle) / angle, 1.0 - angle2 / 6.0)
    b = torch.where(angle2 > 1e-8, (1.0 - torch.cos(angle)) / (angle2 + 1e-12), 0.5 - angle2 / 24.0)
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * torch.matmul(skew, skew)


class ArticulatedHeadLayer(nn.Module):
    """Apply shape, expression, pose correctives, five LBS transforms and cheek contours.

    ``skinning_weights`` has one column for root, neck, jaw, left eye and right
    eye. ``pose_directions`` uses 36 pose features (the 3x3 local rotation
    offsets for each non-root transform). Blendshape arrays are optional; a
    missing basis contributes exactly zero and therefore supports a neutral
    template while model assets are being trained or supplied.

    ``contour_candidates`` is an optional integer table shaped (2, K, C). Its
    first axis contains candidate vertex sets for negative and nonnegative
    yaw. Each set is marched to its outermost projected-x vertex, producing a
    yaw-dependent 3D cheek contour without data-dependent Python control flow.
    Candidate entries padded with -1 are ignored.
    """

    SHAPE_DIM = 300
    EXPRESSION_DIM = 100
    JOINT_COUNT = 5

    def __init__(
        self,
        template_vertices: Tensor,
        skinning_weights: Tensor,
        rest_joints: Tensor,
        *,
        shape_directions: Tensor | None = None,
        expression_directions: Tensor | None = None,
        pose_directions: Tensor | None = None,
        joint_regressor: Tensor | None = None,
        contour_candidates: Tensor | None = None,
    ) -> None:
        super().__init__()
        vertices = torch.as_tensor(template_vertices, dtype=torch.float32)
        weights = torch.as_tensor(skinning_weights, dtype=torch.float32)
        joints = torch.as_tensor(rest_joints, dtype=torch.float32)
        if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.shape[0] == 0:
            raise ValueError("template_vertices must have shape (V, 3)")
        if weights.shape != (vertices.shape[0], self.JOINT_COUNT):
            raise ValueError("skinning_weights must have shape (V, 5)")
        if joints.shape != (self.JOINT_COUNT, 3):
            raise ValueError("rest_joints must have shape (5, 3): root, neck, jaw, left eye, right eye")
        if not torch.isfinite(vertices).all() or not torch.isfinite(weights).all() or not torch.isfinite(joints).all():
            raise ValueError("template geometry and skinning weights must be finite")
        if (weights < 0).any() or (weights.sum(dim=-1) <= 0).any():
            raise ValueError("skinning weights must be nonnegative and nonzero for every vertex")
        weights = weights / weights.sum(dim=-1, keepdim=True)
        self.register_buffer("template_vertices", vertices.contiguous())
        self.register_buffer("skinning_weights", weights.contiguous())
        self.register_buffer("rest_joints", joints.contiguous())

        self._register_basis("shape_directions", shape_directions, self.SHAPE_DIM, vertices.shape[0])
        self._register_basis("expression_directions", expression_directions, self.EXPRESSION_DIM, vertices.shape[0])
        self._register_basis("pose_directions", pose_directions, 36, vertices.shape[0])

        if joint_regressor is not None:
            regressor = torch.as_tensor(joint_regressor, dtype=torch.float32)
            if regressor.shape != (self.JOINT_COUNT, vertices.shape[0]):
                raise ValueError("joint_regressor must have shape (5, V)")
            self.register_buffer("joint_regressor", regressor.contiguous())
        else:
            self.register_buffer("joint_regressor", None)

        if contour_candidates is None:
            candidates = torch.full((2, 0, 1), -1, dtype=torch.long)
        else:
            candidates = torch.as_tensor(contour_candidates, dtype=torch.long)
            if candidates.ndim != 3 or candidates.shape[0] != 2:
                raise ValueError("contour_candidates must have shape (2, K, C)")
            if (candidates >= vertices.shape[0]).any() or (candidates < -1).any():
                raise ValueError("contour candidate indices must be -1 or valid vertex indices")
            if candidates.numel() and (candidates >= 0).sum(dim=-1).eq(0).any():
                raise ValueError("each contour landmark candidate set must contain a vertex")
        self.register_buffer("contour_candidates", candidates.contiguous())

    def _register_basis(self, name: str, value: Tensor | None, count: int, vertex_count: int) -> None:
        if value is None:
            self.register_buffer(name, None)
            return
        basis = torch.as_tensor(value, dtype=torch.float32)
        if basis.shape != (count, vertex_count, 3):
            raise ValueError(f"{name} must have shape ({count}, V, 3)")
        self.register_buffer(name, basis.contiguous())

    def forward(
        self,
        beta: Tensor,
        psi: Tensor,
        global_pose: Tensor,
        neck_pose: Tensor,
        theta_jaw: Tensor,
        theta_eyes: Tensor,
        global_translation: Tensor | None = None,
        vertex_residual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return skinned vertices and yaw-marched cheek landmarks."""
        batch = beta.shape[0]
        if beta.shape != (batch, self.SHAPE_DIM) or psi.shape != (batch, self.EXPRESSION_DIM):
            raise ValueError("beta and psi must have shapes (B, 300) and (B, 100)")
        if global_pose.shape != (batch, 3) or neck_pose.shape != (batch, 3) or theta_jaw.shape != (batch, 3):
            raise ValueError("global_pose, neck_pose and theta_jaw must have shape (B, 3)")
        if theta_eyes.shape != (batch, 6):
            raise ValueError("theta_eyes must have shape (B, 6): shared eye axis-angle and translation")

        shaped = self.template_vertices.unsqueeze(0).expand(batch, -1, -1)
        if self.shape_directions is not None:
            shaped = shaped + torch.einsum("bp,pvc->bvc", beta, self.shape_directions)
        if self.expression_directions is not None:
            shaped = shaped + torch.einsum("bp,pvc->bvc", psi, self.expression_directions)

        if self.joint_regressor is None:
            rest = self.rest_joints.unsqueeze(0).expand(batch, -1, -1)
        else:
            # Regress pivots from identity/expression geometry, before pose
            # correctives and local residuals can shift the skeleton itself.
            rest = torch.einsum("jv,bvc->bjc", self.joint_regressor, shaped)

        local_rotations = torch.stack((
            axis_angle_to_matrix(global_pose),
            axis_angle_to_matrix(neck_pose),
            axis_angle_to_matrix(theta_jaw),
            axis_angle_to_matrix(theta_eyes[:, :3]),
            axis_angle_to_matrix(theta_eyes[:, :3]),
        ), dim=1)
        if self.pose_directions is not None:
            identity = torch.eye(3, dtype=beta.dtype, device=beta.device)
            pose_feature = (local_rotations[:, 1:] - identity).flatten(start_dim=1)
            shaped = shaped + torch.einsum("bp,pvc->bvc", pose_feature, self.pose_directions)
        if vertex_residual is not None:
            shaped = shaped + vertex_residual

        # Parent order: root; neck->root; jaw/eyes->neck. Both eyes share the
        # same six DoF and rotate about their distinct rest pivots.
        root_translation = torch.zeros_like(global_pose) if global_translation is None else global_translation
        posed_joint_positions = [torch.matmul(local_rotations[:, 0], rest[:, 0].unsqueeze(-1)).squeeze(-1) + root_translation]
        global_rotations = [local_rotations[:, 0]]
        eye_translation = theta_eyes[:, 3:]
        for joint_index in range(1, self.JOINT_COUNT):
            parent_rotation = global_rotations[0] if joint_index == 1 else global_rotations[1]
            parent_position = posed_joint_positions[0] if joint_index == 1 else posed_joint_positions[1]
            offset = rest[:, joint_index] - rest[:, 0 if joint_index == 1 else 1]
            if joint_index >= 3:
                offset = offset + eye_translation
            posed_joint_positions.append(parent_position + torch.matmul(parent_rotation, offset.unsqueeze(-1)).squeeze(-1))
            global_rotations.append(torch.matmul(parent_rotation, local_rotations[:, joint_index]))

        joints_posed = torch.stack(posed_joint_positions, dim=1)
        rotations = torch.stack(global_rotations, dim=1)
        relative = shaped[:, None, :, :] - rest[:, :, None, :]
        transformed = torch.matmul(rotations[:, :, None, :, :], relative.unsqueeze(-1)).squeeze(-1) + joints_posed[:, :, None, :]
        vertices = torch.einsum("vj,bjvc->bvc", self.skinning_weights, transformed)

        yaw = global_pose[:, 1]
        contour = self._march_contour(vertices, yaw)
        return vertices, contour

    def _march_contour(self, vertices: Tensor, yaw: Tensor) -> Tensor:
        if self.contour_candidates.shape[1] == 0:
            return vertices[:, :0, :]
        side = (yaw >= 0).to(torch.long)
        candidates = self.contour_candidates.index_select(0, side)
        valid = candidates >= 0
        candidate_indices = candidates.clamp_min(0)
        expanded_vertices = vertices.unsqueeze(1).expand(-1, candidates.shape[1], -1, -1)
        gather_indices = candidate_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        points = torch.gather(expanded_vertices, 2, gather_indices)
        x = points[..., 0]
        # Negative yaw marches toward the left silhouette; nonnegative yaw to right.
        fill = torch.finfo(x.dtype).max
        score = x
        score = torch.where(valid, score, torch.where(side[:, None, None] == 0, fill, -fill))
        best = torch.where(side[:, None] == 0, score.argmin(dim=-1), score.argmax(dim=-1))
        gather_index = best[..., None, None].expand(-1, -1, 1, 3)
        return points.gather(2, gather_index).squeeze(2)
