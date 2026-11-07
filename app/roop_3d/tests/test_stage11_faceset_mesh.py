"""Stage 11: RBF deformation, welded normals, multiband blend, texture completion, padding mask, 68-point correspondence."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.identity_asset import padding_clear_mask
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.rbf_deform import deform_template_rbf, weld_ids, welded_vertex_normals
from roop_3d.reconstruction.uv_completion import complete_texture, multiband_blend, nearest_texels

needs_asset = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="MakeHuman head asset not built")


def _grid(n: int = 12) -> Mesh3D:
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, n), np.linspace(-0.1, 0.1, n))
    verts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return Mesh3D(verts, np.array(faces), uvs=verts[:, :2] + 0.5)


def test_rbf_moves_controls_and_decays_to_zero_far_away():
    mesh = _grid()
    ctrl_idx = np.array([66, 67, 78])
    ctrl = mesh.vertices[ctrl_idx].astype(float)
    target = ctrl + np.array([[0, 0, 0.004], [0, 0, 0.004], [0, 0, 0.004]])
    out, support, strength = deform_template_rbf(mesh, ctrl, target, np.ones(3), ctrl_idx, support=0.05)
    moved = np.linalg.norm(out - mesh.vertices, axis=1)
    assert moved[ctrl_idx].min() > 0.0015 and strength > 0.9
    far = np.linalg.norm(mesh.vertices - ctrl.mean(axis=0), axis=1) > 0.07
    assert moved[far].max() < 0.1 * moved[ctrl_idx].max()                 # decays away from the controls (Laplacian smoothing leaks a little)


def test_welded_normals_are_identical_across_seam_duplicates():
    mesh = _grid(4)
    shared = int(mesh.faces[0, 1])
    verts = np.vstack([mesh.vertices, mesh.vertices[[shared]] + 0.0])      # a seam copy of a shared vertex
    faces = mesh.faces.copy()
    faces[0, 1] = len(verts) - 1
    normals = welded_vertex_normals(verts, faces)
    assert np.allclose(normals[shared], normals[-1]) and np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert weld_ids(verts)[shared] == weld_ids(verts)[-1]


def test_multiband_blend_feathers_exposure_step_and_leaves_empty_texels_empty():
    h = w = 64
    colors = np.zeros((2, h, w, 3), np.float32)
    colors[0], colors[1] = 100.0, 140.0
    weights = np.zeros((2, h, w), np.float32)
    weights[0, :, :36], weights[1, :, 28:] = 1.0, 1.0                        # overlap columns 28..35, the winner flips at 28
    weights[0, :, :36] += 0.5 * (np.arange(36) < 28)
    out = multiband_blend(colors, weights, levels=4)
    row = out[32, :, 0]
    assert abs(row[5] - 100) < 3 and abs(row[52] - 140) < 6
    assert np.abs(np.diff(row[8:56])).max() < 12                              # no hard 40-level step: the seam is feathered
    empty = multiband_blend(colors, np.zeros_like(weights), levels=3)
    assert empty.max() == 0.0


def test_nearest_texels_finds_the_closest_point():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, (500, 3))
    query = pts[:20] + rng.normal(0, 3e-4, (20, 3))
    idx, dist = nearest_texels(query, pts)
    assert (idx == np.arange(20)).all() and (dist < 2e-3).all()


def test_complete_texture_has_no_holes_and_uses_symmetry_then_skin_tone():
    h = w = 64
    xs = np.linspace(-0.08, 0.08, w)
    positions = np.zeros((h, w, 3))
    positions[..., 0] = xs[None, :]
    positions[..., 1] = np.linspace(-0.1, 0.1, h)[:, None]
    footprint = np.ones((h, w), bool)
    observed = np.zeros((h, w), bool)
    observed[:, :24] = True                                                  # only the left side was seen
    texture = np.zeros((h, w, 3), np.uint8)
    texture[observed] = (90, 110, 150)
    result = complete_texture(texture, observed, footprint, positions, mirror_x=0.0)
    assert (result.source[:, 40:] == 2).mean() > 0.8                          # the mirrored side comes from symmetry projection
    assert (result.source == 3).sum() == result.stats["harmonic_ratio"] * h * w or result.stats["harmonic_ratio"] < 0.3
    assert result.texture.shape == (h, w, 4) and (result.texture[..., 3] == 255).all()
    assert result.texture[..., :3].max(axis=2).min() > 60                     # no black holes anywhere


def test_padding_mask_marks_letterbox_only():
    img = np.full((100, 100, 3), 120, np.uint8)
    img[:, :20] = 0
    img[40:50, 50:60] = 0                                                     # an interior dark patch (e.g. an eye) is content
    mask = padding_clear_mask(img)
    assert mask is not None and mask[50, 5] == 0 and mask[45, 55] == 255 and mask[50, 80] == 255
    assert padding_clear_mask(np.full((50, 50, 3), 90, np.uint8)) is None


@needs_asset
def test_makehuman_68_correspondence_is_anatomically_ordered():
    from roop_3d.reconstruction.makehuman_landmarks import correspondence_report
    from roop_3d.reconstruction.makehuman_template import makehuman_template
    t = makehuman_template()
    lm = t.vertices[t.landmark_vertices["68pt"]]
    assert lm.shape == (68, 3)
    assert lm[36:42, 1].mean() < lm[48:60, 1].mean() < lm[8, 1]               # eyes above mouth above chin (+y down)
    assert lm[36:42, 0].mean() < lm[42:48, 0].mean() and lm[0, 0] < lm[16, 0]
    assert lm[30, 2] > lm[36:48, 2].max() - 0.02                              # nose tip is the most forward feature
    report = correspondence_report(t)
    assert report["anchor_rms_mm"] < 8.0 and report["snap_mean_mm"] < 6.0
