"""Acceptance tests for Stage 08 visibility-weighted UV texture synthesis and confidence gating."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    Pose3D,
)
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh
from roop_3d.reconstruction.uv_synthesizer import (
    UVSynthesizer,
    ViewObservation,
)


def test_uv_synthesizer_single_view():
    mesh = build_canonical_head_mesh(landmark_count=68)
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=640, height=480)

    # Frontal camera view: looking directly at face at distance 1.0m
    pose = Pose3D(
        rvec=(0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 1.0),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )

    # Synthetic face image (skin-toned BGR: [140, 160, 220])
    img = np.ones((480, 640, 3), dtype=np.uint8)
    img[:, :] = [140, 160, 220]

    obs = ViewObservation(
        image=img,
        pose=pose,
        intrinsics=intrinsics,
        sharpness=1.0,
    )

    synthesizer = UVSynthesizer(mesh=mesh, uv_size=128)
    result = synthesizer.synthesize([obs])

    assert result.texture_map.shape == (128, 128, 3)
    assert result.confidence_mask.shape == (128, 128)
    assert result.observed_ratio > 0.3  # Substantial portion of face observed

    # Check observed texels have correct skin tone
    observed_mask = result.confidence_mask == int(ConfidenceState.OBSERVED)
    assert np.any(observed_mask)
    obs_pixels = result.texture_map[observed_mask]
    np.testing.assert_allclose(np.mean(obs_pixels, axis=0), [140, 160, 220], atol=10.0)

    # Unseen texels outside mesh geometry must be strictly zero
    unseen_mask = result.confidence_mask == int(ConfidenceState.UNSEEN)
    assert np.any(unseen_mask)
    assert np.all(result.texture_map[unseen_mask] == 0)


def test_unseen_back_of_head_never_blended():
    # Construct a synthetic mesh with front face (facing +Z) and back face (facing -Z)
    vertices = np.array([
        # Front triangle (normal +Z in model, facing camera)
        [-0.05, -0.05,  0.05],
        [ 0.05, -0.05,  0.05],
        [ 0.00,  0.05,  0.05],
        # Back triangle (normal -Z in model, facing away from camera)
        [ 0.05, -0.05, -0.05],
        [-0.05, -0.05, -0.05],
        [ 0.00,  0.05, -0.05],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],  # Front
        [3, 4, 5],  # Back
    ], dtype=np.int32)

    uvs = np.array([
        [0.1, 0.1], [0.4, 0.1], [0.25, 0.4],  # Front UV patch
        [0.6, 0.6], [0.9, 0.6], [0.75, 0.9],  # Back UV patch
    ], dtype=np.float32)

    mesh = Mesh3D(vertices=vertices, faces=faces, uvs=uvs)
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=400, height=400)

    # Frontal camera view: camera at (0, 0, 1.0), looking along -Z
    pose = Pose3D(
        rvec=(0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 1.0),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )

    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    obs = ViewObservation(image=img, pose=pose, intrinsics=intrinsics)

    synthesizer = UVSynthesizer(mesh=mesh, uv_size=128)
    result = synthesizer.synthesize([obs])

    # Front UV patch (top-left in image raster space)
    front_uv_pixel_y, front_uv_pixel_x = int(0.2 * 128), int(0.25 * 128)
    assert result.confidence_mask[front_uv_pixel_y, front_uv_pixel_x] == int(ConfidenceState.OBSERVED)

    # Back UV patch (bottom-right in image raster space)
    # The back face points away from the camera, so normal dot product is <= 0!
    # It must NEVER be blended as observed texture.
    back_uv_pixel_y, back_uv_pixel_x = int(0.75 * 128), int(0.75 * 128)
    assert result.confidence_mask[back_uv_pixel_y, back_uv_pixel_x] != int(ConfidenceState.OBSERVED)


def test_uv_synthesizer_occlusion_mask():
    mesh = build_canonical_head_mesh(landmark_count=68)
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=640, height=480)
    pose = Pose3D(
        rvec=(0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 1.0),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )

    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Binary occlusion mask: completely black (0 = occluded)
    occlusion_mask = np.zeros((480, 640), dtype=np.uint8)

    obs = ViewObservation(
        image=img,
        pose=pose,
        intrinsics=intrinsics,
        occlusion_mask=occlusion_mask,
    )

    synthesizer = UVSynthesizer(mesh=mesh, uv_size=128)
    result = synthesizer.synthesize([obs])

    # Since all pixels are occluded, observed ratio must be 0
    assert result.observed_ratio == 0.0
    assert not np.any(result.confidence_mask == int(ConfidenceState.OBSERVED))
