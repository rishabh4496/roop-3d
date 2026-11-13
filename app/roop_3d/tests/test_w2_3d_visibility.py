"""G3: real depth-buffered self-occlusion (no hard-coded index masks)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.mesh_raycaster import MeshRaycaster
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, Mesh3D
from roop_3d.reconstruction.makehuman_template import makehuman_template
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh

ASSET = Path(__file__).resolve().parents[1] / "reconstruction" / "assets" / "makehuman_head_v1.npz"
INTR = CameraIntrinsics(fx=500.0, fy=500.0, cx=256.0, cy=256.0, width=512, height=512)
RAY = MeshRaycaster()


def _vis(mesh, yaw, **pose):
    return RAY.compute_visibility_map(mesh, {"yaw": yaw, **pose}, intrinsics=INTR)


@pytest.fixture(scope="module")
def canonical():
    return build_canonical_head_mesh(68)


def _sphere_head(rings=40, segments=48) -> Mesh3D:
    """Closed, x-symmetric head-sized shell (repo frame: +z toward the viewer) with a nose bump on +z."""
    verts = []
    for i in range(1, rings):
        phi = np.pi * i / rings
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            p = np.array([np.sin(phi) * np.sin(theta), -np.cos(phi), np.sin(phi) * np.cos(theta)]) * np.array([0.075, 0.1, 0.09])
            p[2] += 0.03 * np.exp(-((p[0] / 0.015) ** 2 + ((p[1] - 0.01) / 0.02) ** 2)) * (p[2] > 0)
            verts.append(p)
    top, bottom = len(verts), len(verts) + 1
    verts += [[0, -0.1, 0], [0, 0.1, 0]]
    faces = []
    for i in range(rings - 2):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = (i + 1) * segments + j, (i + 1) * segments + (j + 1) % segments
            faces += [[a, c, b], [b, c, d]]
    for j in range(segments):
        faces += [[top, j, (j + 1) % segments]]
        base = (rings - 2) * segments
        faces += [[bottom, base + (j + 1) % segments, base + j]]
    mesh = Mesh3D(np.array(verts, np.float32), np.array(faces, np.int32))
    # make sure the winding is outward (normal . radial > 0) for the test mesh itself
    outward = np.einsum("ij,ij->i", mesh.compute_face_normals(), mesh.vertices[mesh.faces].mean(axis=1))
    return mesh if (outward > 0).mean() > 0.9 else Mesh3D(mesh.vertices, mesh.faces[:, [0, 2, 1]])


def test_frontal_canonical_head_is_almost_fully_visible(canonical):
    result = _vis(canonical, 0.0)
    assert result.vertex_visible.mean() >= 0.95
    assert result.vertex_depth_tested.all()
    assert result.visibility_map.max() > 0.9 and result.execution_time_ms > 0


@pytest.mark.parametrize("yaw", [60.0, 75.0, 90.0])
def test_far_side_is_occluded_and_near_side_visible_on_both_yaw_signs(canonical, yaw):
    # canonical x>0 is the subject's left cheek/eye (image right); +yaw (nose to image right) hides it.
    far_cheek, near_cheek, far_eye, near_eye, nose = 15, 1, 45, 36, 30
    pos, neg = _vis(canonical, yaw).vertex_visible, _vis(canonical, -yaw).vertex_visible
    assert not pos[far_cheek] and pos[near_cheek] and pos[near_eye]
    assert not neg[near_cheek] and neg[far_cheek] and neg[far_eye]
    if yaw >= 75.0:      # at 60 deg the outer eye corner is still genuinely in view; from 75 deg the nose hides it
        assert not pos[far_eye] and not neg[near_eye]
    if yaw < 90:
        assert pos[nose] and neg[nose]
    assert _vis(canonical, yaw).self_occluded_fraction > 0.25


def test_visibility_is_mirror_symmetric_on_a_symmetric_mesh():
    mesh = _sphere_head()
    verts = mesh.vertices.astype(np.float64)
    mirrored = verts * np.array([-1, 1, 1])
    mirror_index = np.argmin(((verts[None] - mirrored[:, None]) ** 2).sum(-1), axis=1)
    assert np.abs(verts[mirror_index] - mirrored).max() < 1e-6
    for yaw in (30.0, 60.0, 75.0, 90.0):
        a, b = _vis(mesh, yaw).vertex_visible, _vis(mesh, -yaw).vertex_visible
        assert (a != b[mirror_index]).sum() <= 0.015 * len(a), yaw  # only silhouette-edge ties (pixel rounding) may differ
        assert 0.3 < 1 - a.mean() < 0.7, "about half of a closed shell is self-occluded at profile"


def test_depth_test_hides_a_vertex_behind_a_front_surface():
    # two parallel quads facing the camera: the back one is fully behind the front one
    front = [[-.05, -.05, 0.0], [.05, -.05, 0.0], [.05, .05, 0.0], [-.05, .05, 0.0]]
    back = [[v[0], v[1], -0.05] for v in front]
    verts = np.array(front + back, np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int32)
    mesh = Mesh3D(verts, faces)
    assert mesh.compute_face_normals()[:, 2].min() > 0, "outward is +z toward the viewer"
    res = _vis(mesh, 0.0)
    assert res.vertex_visible[:4].all() and not res.vertex_visible[4:].any()
    # eps: a surface 2 mm behind is within tolerance and stays visible
    verts[4:, 2] = -0.002
    assert _vis(Mesh3D(verts, faces), 0.0).vertex_visible.all()


def test_makehuman_asset_head_occludes_far_cheek_at_profile():
    mesh = makehuman_template(ASSET)
    v, n = mesh.vertices, mesh.normals
    front_facing = n[:, 2] > 0.3
    frontal = _vis(mesh, 0.0, tz=0.7).vertex_visible
    # interior eye/mouth geometry sits behind skin even frontally; the outer surface is visible
    assert frontal[front_facing].mean() >= 0.75
    for yaw in (75.0, -75.0, 90.0, -90.0):
        vis = _vis(mesh, yaw, tz=0.7).vertex_visible
        sign = 1.0 if yaw > 0 else -1.0
        far = (v[:, 0] * sign > 0.05) & (n[:, 0] * sign > 0.3)       # +x side bulging away from the camera at +yaw
        near = (v[:, 0] * sign < -0.05) & (n[:, 0] * sign < -0.3)
        assert far.sum() > 50 and near.sum() > 50
        assert vis[far].mean() < 0.05, yaw
        assert vis[near].mean() > 0.75, yaw


def test_accepts_solved_pose_and_rejects_unusable_input(canonical):
    from roop_3d.reconstruction.pose_solver import solve_head_pose_6dof
    from roop_3d.reconstruction.pose_solver import _FLIP
    from roop_3d.core.pose import rotation_from_euler
    import cv2
    from roop_3d.reconstruction.pose_solver import rotation_from_standard, default_intrinsics_matrix
    from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks
    model = get_canonical_3d_landmarks(68).astype(np.float64) @ _FLIP
    k = default_intrinsics_matrix(640, 480)
    rotation = rotation_from_standard(rotation_from_euler(70.0, 0.0, 0.0))
    pts = cv2.projectPoints(model, cv2.Rodrigues(rotation)[0], np.array([[0.0], [0.0], [0.7]]), k, None)[0].reshape(-1, 2)
    pose = solve_head_pose_6dof(pts, image_size=(640, 480))
    via_pose = RAY.compute_visibility_map(canonical, pose, resolution=(480, 640)).vertex_visible
    via_dict = RAY.compute_visibility_map(canonical, {"yaw": pose.yaw_deg, "pitch": pose.pitch_deg, "roll": pose.roll_deg,
                                                      "tx": pose.t[0], "ty": pose.t[1], "tz": pose.t[2]},
                                          resolution=(480, 640)).vertex_visible
    assert (via_pose != via_dict).sum() <= 1
    with pytest.raises(DomainError) as bare:
        RAY.compute_visibility_map(canonical.vertices, {"yaw": 0})
    assert bare.value.code == "VISIBILITY_FACES_REQUIRED"
    with pytest.raises(DomainError) as behind:
        RAY.compute_visibility_map(canonical, {"yaw": 0, "tz": -1.0})
    assert behind.value.code == "VISIBILITY_POSE_INVALID"
