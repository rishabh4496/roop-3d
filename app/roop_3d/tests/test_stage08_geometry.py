"""Acceptance tests for Stage 08 3D geometry contracts and coordinate conventions."""
from __future__ import annotations

import math
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    Pose3D,
    UVOrigin,
    euler_to_rotation_matrix,
    opencv_to_opengl_camera,
    opengl_to_opencv_camera,
    pixel_to_uv,
    quaternion_to_rotation_matrix,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rodrigues,
    uv_to_pixel,
)


def test_camera_intrinsics_projection_and_unprojection():
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=640, height=480)
    assert intrinsics.fx == pytest.approx(intrinsics.fy, rel=1e-5)
    assert intrinsics.cx == 320.0
    assert intrinsics.cy == 240.0

    points_3d = np.array([
        [0.0, 0.0, 1.5],
        [0.1, -0.05, 1.2],
        [-0.15, 0.2, 2.0],
    ], dtype=np.float32)

    projected_2d = intrinsics.project(points_3d)
    assert projected_2d.shape == (3, 2)
    # The origin point (0, 0, 1.5) must project exactly to the principal point (cx, cy)
    assert projected_2d[0, 0] == pytest.approx(320.0, abs=1e-4)
    assert projected_2d[0, 1] == pytest.approx(240.0, abs=1e-4)

    unprojected_3d = intrinsics.unproject(projected_2d, points_3d[:, 2])
    np.testing.assert_allclose(unprojected_3d, points_3d, rtol=1e-5, atol=1e-5)


def test_camera_intrinsics_invalid_fov():
    with pytest.raises(DomainError):
        CameraIntrinsics.from_fov(fov_degrees=-10.0, width=640, height=480)
    with pytest.raises(DomainError):
        CameraIntrinsics.from_fov(fov_degrees=185.0, width=640, height=480)


def test_opencv_to_opengl_conversion_involution():
    pts_cv = np.array([[1.0, 2.0, 3.0], [-0.5, 0.8, 1.5]], dtype=np.float32)
    pts_gl = opencv_to_opengl_camera(pts_cv)
    assert pts_gl[0, 0] == pts_cv[0, 0]
    assert pts_gl[0, 1] == -pts_cv[0, 1]
    assert pts_gl[0, 2] == -pts_cv[0, 2]

    round_trip = opengl_to_opencv_camera(pts_gl)
    np.testing.assert_allclose(round_trip, pts_cv)


def test_rotation_representations_round_trip():
    test_euler_angles = [
        (0.0, 0.0, 0.0),
        (25.0, -15.0, 10.0),
        (-35.0, 20.0, -18.0),
        (15.0, 0.0, 0.0),
        (0.0, -25.0, 0.0),
    ]

    for yaw, pitch, roll in test_euler_angles:
        r_mat = euler_to_rotation_matrix(yaw, pitch, roll, degrees=True)
        # Check orthogonality: R * R^T = I and det(R) = +1
        np.testing.assert_allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(r_mat) == pytest.approx(1.0, abs=1e-6)

        # Round trip to Rodrigues
        rvec = rotation_matrix_to_rodrigues(r_mat)
        r_mat_from_rodrigues = rodrigues_to_rotation_matrix(rvec)
        np.testing.assert_allclose(r_mat_from_rodrigues, r_mat, atol=1e-6)

        # Round trip to Quaternion
        quat = rotation_matrix_to_quaternion(r_mat)
        norm_q = math.sqrt(sum(x * x for x in quat))
        assert norm_q == pytest.approx(1.0, abs=1e-6)
        r_mat_from_quat = quaternion_to_rotation_matrix(quat)
        np.testing.assert_allclose(r_mat_from_quat, r_mat, atol=1e-6)

        # Round trip to Euler
        y_rec, p_rec, r_rec = rotation_matrix_to_euler(r_mat, degrees=True)
        assert y_rec == pytest.approx(yaw, abs=1e-4)
        assert p_rec == pytest.approx(pitch, abs=1e-4)
        assert r_rec == pytest.approx(roll, abs=1e-4)


def test_pose_3d_transform_points():
    pose = Pose3D(
        rvec=(0.0, 0.0, 0.0),
        tvec=(0.1, -0.2, 1.0),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        reprojection_error=0.12,
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    transformed = pose.transform_points(points)
    expected = np.array([[0.1, -0.2, 1.0], [1.1, 0.8, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(transformed, expected, atol=1e-5)


def test_mesh_3d_winding_and_normals():
    # Triangle in XY plane with vertices in CCW order when looking along +Z:
    # v0 = (0, 0, 0), v1 = (1, 0, 0), v2 = (0, 1, 0)
    # e1 = (1, 0, 0), e2 = (0, 1, 0)
    # e1 x e2 = (0, 0, 1) -> normal points in +Z
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    mesh = Mesh3D(vertices=vertices, faces=faces, uvs=uvs)
    face_normals = mesh.compute_face_normals()
    assert face_normals.shape == (1, 3)
    np.testing.assert_allclose(face_normals[0], [0.0, 0.0, 1.0], atol=1e-6)

    vertex_normals = mesh.compute_vertex_normals()
    assert vertex_normals.shape == (3, 3)
    for i in range(3):
        np.testing.assert_allclose(vertex_normals[i], [0.0, 0.0, 1.0], atol=1e-6)


def test_mesh_3d_validation_failures():
    # Degenerate NaN vertex
    with pytest.raises(DomainError):
        Mesh3D(
            vertices=np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
    # Out of bounds face index
    with pytest.raises(DomainError):
        Mesh3D(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 5]]),
        )


def test_uv_to_pixel_round_trip():
    width, height = 512, 256
    uv_coords = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
    ], dtype=np.float32)

    for origin in (UVOrigin.BOTTOM_LEFT, UVOrigin.TOP_LEFT):
        pixels = uv_to_pixel(uv_coords, width, height, origin=origin)
        recovered_uv = pixel_to_uv(pixels, width, height, origin=origin)
        np.testing.assert_allclose(recovered_uv, uv_coords, atol=1e-5)
