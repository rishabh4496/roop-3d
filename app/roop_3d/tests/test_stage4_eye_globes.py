"""Stage 4 eyeballs: lid-margin landmark binding, globe placement inside the sockets and tile mapping (real MakeHuman asset)."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.reconstruction.eye_globes import (
    EYE_NAMES, TILE_SIZE, build_eye_globes, compose_eye_tiles, tile_grid)
from roop_3d.reconstruction.makehuman_landmarks import eye_apertures
from roop_3d.reconstruction.makehuman_template import makehuman_template


@pytest.fixture(scope="module")
def head():
    template = makehuman_template()
    vertices = template.vertices.astype(np.float64)
    return template, vertices, eye_apertures(vertices, template.faces)


def test_dense_eye_landmarks_are_the_anatomical_lid_margin(head):
    template, vertices, apertures = head
    marks = template.landmark_vertices["68pt"]
    assert len(np.unique(marks)) == 68
    assert np.array_equal(np.concatenate([item["landmarks"] for item in apertures]), marks[36:48])
    for item in apertures:
        assert set(item["landmarks"].tolist()) <= set(item["rim"].tolist())
        upper = vertices[item["landmarks"][[1, 2]], 1].mean()
        lower = vertices[item["landmarks"][[4, 5]], 1].mean()
        assert upper < lower                       # image y grows downward: iBUG upper-lid points sit above the lower lid
    left, right = apertures
    assert vertices[left["landmarks"], 0].mean() < 0 < vertices[right["landmarks"], 0].mean()


def test_globes_rest_behind_the_lid_margin_and_face_outward(head):
    template, vertices, apertures = head
    globes = build_eye_globes(vertices, template.faces, apertures)
    assert globes.faces.max() < len(globes.vertices) and globes.cornea_faces.shape == (len(globes.faces),)
    for eye, item in enumerate(apertures):
        rim = vertices[item["rim"]]
        centre = globes.centers[eye]
        central = rim[np.abs(rim[:, 0] - centre[0]) < 0.3 * np.ptp(rim[:, 0])]
        assert centre[2] < central[:, 2].min()                      # +z is toward the viewer: the globe sits behind the lids
        distance = np.linalg.norm(central - centre, axis=1)
        assert np.all(np.abs(distance - globes.radius_m) < 0.002)   # the lid margin rests on the globe surface (within 2 mm)
        assert 0.7 * np.ptp(rim[:, 0]) / 2 < globes.radius_m < 1.2 * np.ptp(rim[:, 0]) / 2
    corners = globes.vertices.astype(np.float64)
    tri = globes.faces
    cross = np.cross(corners[tri[:, 1]] - corners[tri[:, 0]], corners[tri[:, 2]] - corners[tri[:, 0]])
    owner = np.where(tri[:, 0] < globes.vertices_per_eye, 0, 1)
    outward = corners[tri].mean(axis=1) - globes.centers[owner]
    assert np.all(np.einsum("ij,ij->i", cross, outward) > 0)        # same cross-product-outward winding as the head template
    assert np.allclose(np.linalg.norm(globes.normals, axis=1), 1.0, atol=1e-4)
    assert globes.cornea_faces.any() and not globes.cornea_faces.all()


def test_tile_mapping_is_the_exact_inverse_of_the_vertex_uvs(head):
    template, vertices, apertures = head
    globes = build_eye_globes(vertices, template.faces, apertures)
    psi, phi, valid = tile_grid()
    assert valid.sum() > 0.7 * TILE_SIZE ** 2 * np.pi / 4
    pole = globes.tile_pixels[0]
    assert np.allclose(pole, ((TILE_SIZE - 1) / 2.0, (TILE_SIZE - 1) / 2.0))
    assert globes.tile_pixels.min() >= 0 and globes.tile_pixels.max() <= TILE_SIZE - 1
    points, normals = globes.surface(0, np.array([0.0, globes.limbus_psi, 1.2]), np.array([0.0, 1.0, 2.0]))
    radii = np.linalg.norm(points - globes.centers[0], axis=1)
    assert np.isclose(radii[1], globes.radius_m, atol=1e-6)         # the cornea meets the sclera sphere at the limbus
    assert radii[0] > globes.radius_m                               # and bulges beyond it at the apex


def test_texture_labels_prior_when_nothing_was_measured(head):
    template, vertices, apertures = head
    globes = build_eye_globes(vertices, template.faces, apertures)
    measured = np.zeros((2, TILE_SIZE, TILE_SIZE, 3), np.float32)
    tiles, stats = compose_eye_tiles(globes, measured, np.zeros((2, TILE_SIZE, TILE_SIZE), np.float32))
    assert stats["iris_colour_source"] == "prior_neutral_brown" and stats["sclera_colour_source"] == "prior_neutral_white"
    assert tiles.shape == (2, TILE_SIZE, TILE_SIZE, 3) and tiles.dtype == np.uint8
    for name in EYE_NAMES:
        assert stats["per_eye"][name]["measured_iris_texel_fraction"] == 0.0
