"""3D swap mask beyond the inner face oval, the biharmonic jaw field, and the z-buffer rasteriser they share.

Geometry helpers are checked on synthetic meshes with known answers. Everything anatomical (jaw contour, chin underside, ears)
runs on the real CC0 MakeHuman head, which is skipped when the asset is not built. The 68-point correspondence on that head
is DERIVED (``makehuman_landmarks``), so these tests assert properties of the construction, not anatomical ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.rbf_deform import welded_vertex_normals
from roop_3d.rendering.swap_region_3d import (
    CHIN_HANDLES, JAW_LANDMARKS, JawTarget, SwapRegionConfig, WeldedGraph, adapt_jaw, build_swap_mask_3d,
    derive_ear_anchors, jaw_target_from_landmarks, jaw_target_from_ratios, jaw_width, polygon_signed_distance,
    rasterize_swap_mask)
from roop_3d.rendering.vertex_raster import orthographic_pixels, rasterize_vertex_attributes

needs_asset = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="MakeHuman head asset not built")

FULL = SwapRegionConfig(include_jawline=True, include_chin_underside=True, include_ears=True)


def _grid(n: int = 21, size: float = 0.1):
    xs, ys = np.meshgrid(np.linspace(-size, size, n), np.linspace(-size, size, n))
    verts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return verts, np.array(faces)


@pytest.fixture(scope="module")
def head():
    from roop_3d.reconstruction.makehuman_template import makehuman_template
    mesh = makehuman_template()
    vertices = mesh.vertices.astype(np.float64)
    landmarks = np.asarray(mesh.landmark_vertices["68pt"], np.int64)
    return vertices, np.asarray(mesh.faces, np.int64), landmarks, WeldedGraph(vertices, mesh.faces)


# =========================================================================== helpers on synthetic geometry
def test_polygon_signed_distance_is_positive_inside_and_negative_outside():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    d = polygon_signed_distance(np.array([[1.0, 1.0], [0.25, 1.0], [3.0, 1.0], [1.0, -0.5]]), square)
    assert d == pytest.approx([1.0, 0.25, -1.0, -0.5])


def test_welded_graph_shares_uv_seam_nodes_and_measures_surface_distance():
    verts, faces = _grid()
    seam = np.vstack([verts, verts[[0]]])                      # a UV-seam duplicate of vertex 0
    faces = faces.copy()
    faces[0, 0] = len(seam) - 1
    graph = WeldedGraph(seam, faces)
    assert graph.weld[0] == graph.weld[-1] and graph.n == len(verts)
    dist = graph.geodesic([0])
    straight = np.linalg.norm(verts - verts[0], axis=1)
    assert dist[0] == 0 and dist[-2] >= straight[-2] - 1e-9          # a surface path is never shorter than the chord
    assert dist[-2] <= 1.5 * straight[-2]                            # and on a grid at most the Manhattan detour (sqrt 2 .. 1.41)
    assert np.isinf(graph.geodesic([0], limit=0.02)[-2])
    path = graph.path(0, len(verts) - 1)
    assert path[0] == graph.weld[0] and path[-1] == graph.weld[len(verts) - 1]


def test_graph_rejects_invalid_meshes():
    verts, faces = _grid(4)
    with pytest.raises(DomainError):
        WeldedGraph(verts, faces + 1000)
    with pytest.raises(DomainError):
        WeldedGraph(np.full((3, 3), np.nan), np.array([[0, 1, 2]]))


def test_rasterizer_z_buffers_and_interpolates_attributes():
    pixels = np.array([[2, 2], [14, 2], [2, 14], [2, 2], [14, 2], [14, 14]], float)       # two overlapping triangles
    depth = np.array([5, 5, 5, 3, 3, 3], float)                                           # the second is nearer
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    attribute = np.array([0, 1, 0, 10, 20, 30], float)
    out = rasterize_vertex_attributes(pixels, depth, faces, attribute, (16, 16))
    assert out.depth[8, 8] == pytest.approx(3.0) and out.face_id[8, 8] == 1                # nearer triangle wins
    assert out.face_id[10, 3] == 0 and out.depth[10, 3] == pytest.approx(5.0)              # only the far one covers this pixel
    assert out.face_id[0, 0] == -1 and np.isinf(out.depth[0, 0]) and out.attributes[0, 0, 0] == 0
    # barycentric interpolation at a pixel centre of the nearer triangle (2,2),(14,2),(14,14)
    x, y = 8.5, 5.5
    a, b, c = pixels[3:6]
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    l0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det
    l1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det
    expect = l0 * 10 + l1 * 20 + (1 - l0 - l1) * 30
    assert out.attributes[5, 8, 0] == pytest.approx(expect, abs=1e-4)
    assert out.coverage.sum() == (out.face_id >= 0).sum()


def test_rasterizer_validates_inputs():
    with pytest.raises(DomainError):
        rasterize_vertex_attributes(np.zeros((3, 2)), np.zeros(2), np.array([[0, 1, 2]]), np.zeros(3), (4, 4))
    with pytest.raises(DomainError):
        rasterize_vertex_attributes(np.zeros((3, 2)), np.zeros(3), np.array([[0, 1, 5]]), np.zeros(3), (4, 4))


# =========================================================================== swap mask on the real head
@needs_asset
def test_default_mask_is_the_inner_oval_only(head):
    v, f, lm, graph = head
    mask = build_swap_mask_3d(v, f, lm, graph=graph)
    assert set(mask.components) == {"inner"} and mask.ear_anchors is None
    assert mask.weights.min() >= 0 and mask.weights.max() <= 1
    assert 0.2 < mask.area_fraction() < 0.5                                 # the frontal face, not the whole head
    nose_tip = int(lm[30])
    assert mask.weights[nose_tip] > 0.95
    assert mask.weights[int(np.argmin(v[:, 2]))] == 0                     # back of the head
    # the oval stops AT the jaw: the jaw-contour vertices themselves are on its edge, not inside it
    assert mask.weights[lm[list(JAW_LANDMARKS)]].mean() < 0.3


@needs_asset
def test_jawline_extension_only_adds_and_covers_the_contour(head):
    v, f, lm, graph = head
    base = build_swap_mask_3d(v, f, lm, graph=graph)
    ext = build_swap_mask_3d(v, f, lm, SwapRegionConfig(include_jawline=True), graph=graph)
    assert np.all(ext.weights >= base.weights - 1e-6)                      # monotonic: an extension never removes coverage
    assert ext.area_fraction() > base.area_fraction() + 0.03
    assert ext.components["jawline"][lm[list(JAW_LANDMARKS)]].min() > 0.99
    assert ext.weights[lm[list(JAW_LANDMARKS)]].min() > 0.99
    assert ext.components["jawline"][int(lm[30])] == 0                   # the band is local to the contour
    wider = build_swap_mask_3d(v, f, lm, SwapRegionConfig(include_jawline=True, jaw_band_m=0.03), graph=graph)
    assert wider.area_fraction("jawline") > ext.area_fraction("jawline")


@needs_asset
def test_chin_underside_selects_down_facing_surface_below_the_chin(head):
    v, f, lm, graph = head
    mask = build_swap_mask_3d(v, f, lm, SwapRegionConfig(include_chin_underside=True), graph=graph)
    w = mask.components["chin_underside"]
    normals = welded_vertex_normals(v, f)
    hit = w > 0.05
    assert hit.sum() > 20
    assert normals[hit, 1].min() > 0.1                                       # every selected vertex faces downwards (+y is down)
    assert v[hit, 1].min() >= v[lm[list(CHIN_HANDLES)], 1].mean() - 0.011
    assert w[int(lm[30])] == 0 and w[int(np.argmin(v[:, 1]))] == 0           # not the nose, not the crown
    assert mask.area_fraction() >= mask.area_fraction("inner")


@needs_asset
def test_ear_anchors_are_lateral_symmetric_and_the_ear_component_is_local(head):
    v, f, lm, graph = head
    anchors = derive_ear_anchors(v, lm)
    assert v[anchors[0], 0] < -0.07 and v[anchors[1], 0] > 0.07
    assert v[anchors[0], 0] == pytest.approx(-v[anchors[1], 0], abs=2e-3)
    mask = build_swap_mask_3d(v, f, lm, SwapRegionConfig(include_ears=True), graph=graph)
    ears = mask.components["ears"]
    assert ears[anchors].min() > 0.99 and ears[int(lm[30])] == 0 and 0 < mask.area_fraction("ears") < 0.15
    assert np.array_equal(mask.ear_anchors, anchors)


@needs_asset
def test_swap_mask_rasterises_depth_tested(head):
    v, f, lm, graph = head
    mask = build_swap_mask_3d(v, f, lm, FULL, graph=graph)
    pixels, depth = orthographic_pixels(v, scale=1800.0, center_px=(192.0, 230.0))
    image, raster = rasterize_swap_mask(mask, pixels, depth, f, (384, 384))
    assert image.shape == (384, 384) and image.min() >= 0 and image.max() <= 1
    nose = pixels[lm[30]].astype(int)
    assert image[nose[1], nose[0]] > 0.9
    assert (image[raster.face_id < 0] == 0).all()                            # nothing outside the mesh
    # depth test: the near face hides the back of the head, so frontal swap weights win at the nose pixel
    assert raster.depth[nose[1], nose[0]] < depth[lm[30]] + 5e-3


@needs_asset
def test_config_and_target_validation(head):
    v, f, lm, graph = head
    with pytest.raises(ValueError):
        SwapRegionConfig(jaw_band_m=0.0)
    with pytest.raises(ValueError):
        SwapRegionConfig(unknown=1)
    with pytest.raises(DomainError):
        jaw_target_from_ratios(v, lm, width_ratio=2.0)
    with pytest.raises(DomainError):
        adapt_jaw(v, f, lm, JawTarget(np.zeros((16, 3))), SwapRegionConfig(include_jawline=True), graph=graph)
    with pytest.raises(DomainError):
        adapt_jaw(v, f, lm, JawTarget(v[lm[:17]]), SwapRegionConfig(include_ears=True), graph=graph)      # ears need ear_points
    with pytest.raises(DomainError):
        build_swap_mask_3d(v, f, lm[:60], graph=graph)


# =========================================================================== Laplacian jaw adaptation
@needs_asset
def test_no_extension_leaves_the_mesh_untouched(head):
    v, f, lm, graph = head
    target = jaw_target_from_ratios(v, lm, width_ratio=1.2)
    result = adapt_jaw(v, f, lm, target, SwapRegionConfig(), graph=graph)           # inner oval only
    assert np.array_equal(result.vertices, v.astype(np.float32)) and not result.field.any()
    assert result.report.handles == 0 and result.report.folded_triangles == 0


@needs_asset
@pytest.mark.parametrize("ratio", [0.88, 1.15])
def test_jaw_field_reaches_the_target_width_without_folds_and_stays_local(head, ratio):
    v, f, lm, graph = head
    cfg = SwapRegionConfig(include_jawline=True)
    target = jaw_target_from_ratios(v, lm, width_ratio=ratio, config=cfg)
    result = adapt_jaw(v, f, lm, target, cfg, graph=graph)
    r = result.report
    assert r.jaw_width_target_m == pytest.approx(r.jaw_width_before_m * ratio, rel=1e-6)
    assert r.jaw_width_after_m == pytest.approx(r.jaw_width_target_m, rel=0.01) and r.handle_residual_m < 5e-4
    assert r.folded_triangles == 0 and r.max_displacement_m <= cfg.max_displacement_m + 1e-9 and 0 < r.free_vertices < graph.n
    # locality: untouched far from the contour (nose tip, crown, back of the head), exactly
    moved = np.linalg.norm(result.field, axis=1)
    reach = graph.geodesic(build_swap_mask_3d(v, f, lm, cfg, graph=graph).contour_vertices)
    assert moved[reach > cfg.jaw_band_m + cfg.inner_blend_m + 1e-6].max() == 0
    for far in (int(lm[30]), int(np.argmin(v[:, 1])), int(np.argmin(v[:, 2]))):
        assert moved[far] == 0
    # UV-seam copies of one position must move identically (no cracks)
    for _, group in zip(range(1), [np.nonzero(graph.weld == graph.weld[int(lm[8])])[0]]):
        assert np.ptp(result.field[group], axis=0).max() < 1e-7
    # smooth: the field changes gently across mesh edges (biharmonic, no landmark spikes)
    a, b = graph.weld[f[:, 0]], graph.weld[f[:, 1]]
    step = np.linalg.norm(result.field[f[:, 0]] - result.field[f[:, 1]], axis=1)
    assert step.max() < 0.6 * r.max_displacement_m + 1e-4


@needs_asset
def test_extreme_targets_never_fold_and_the_miss_is_reported(head):
    v, f, lm, graph = head
    cfg = SwapRegionConfig(include_jawline=True, include_chin_underside=True, include_ears=True, max_displacement_m=0.06)
    target = jaw_target_from_ratios(v, lm, width_ratio=1.6, length_ratio=0.6, config=cfg)
    result = adapt_jaw(v, f, lm, target, cfg, graph=graph)
    assert result.report.folded_triangles == 0
    assert result.report.regions == ("jawline", "chin_underside", "ears")
    assert np.isfinite(result.vertices).all()
    # if relief or the displacement cap pulled the jaw short of the target, the report says so instead of claiming success
    shortfall = result.report.jaw_width_target_m - result.report.jaw_width_after_m
    assert shortfall <= 1e-4 or result.report.handle_residual_m > 1e-4 or result.report.strength < 1.0


@needs_asset
def test_ears_follow_the_jaw_width_change_by_ear_follow(head):
    v, f, lm, graph = head
    cfg = SwapRegionConfig(include_jawline=True, include_ears=True, ear_follow=0.5)
    target = jaw_target_from_ratios(v, lm, width_ratio=1.2, config=cfg)
    result = adapt_jaw(v, f, lm, target, cfg, graph=graph)
    anchors = derive_ear_anchors(v, lm)
    expected = np.abs(v[anchors, 0]) * (1.0 + 0.5 * 0.2)
    assert np.abs(result.vertices[anchors, 0]) == pytest.approx(expected, abs=1e-3)
    off = SwapRegionConfig(include_jawline=True, include_ears=True, ear_follow=0.0)
    still = adapt_jaw(v, f, lm, jaw_target_from_ratios(v, lm, width_ratio=1.2, config=off), off, graph=graph)
    assert np.abs(still.vertices[anchors, 0]) == pytest.approx(np.abs(v[anchors, 0]), abs=1e-3)


@needs_asset
def test_chin_underside_adaptation_moves_the_chin_not_the_jaw_ends(head):
    v, f, lm, graph = head
    cfg = SwapRegionConfig(include_chin_underside=True)
    target = jaw_target_from_ratios(v, lm, length_ratio=1.25, config=cfg)
    result = adapt_jaw(v, f, lm, target, cfg, graph=graph)
    moved = np.linalg.norm(result.field, axis=1)
    assert moved[lm[8]] > 0.004                                              # the chin tip drops
    assert moved[lm[0]] == 0 and moved[lm[16]] == 0                          # the jaw ends were not handles in this mode
    assert result.report.folded_triangles == 0 and result.report.regions == ("chin_underside",)


@needs_asset
def test_jaw_target_from_landmarks_removes_pose_and_scale_before_comparing_jaws(head):
    v, f, lm, graph = head
    src = v[lm]
    narrow = src.copy()
    mid = 0.5 * (src[:17, 0].min() + src[:17, 0].max())
    narrow[:17, 0] = mid + (src[:17, 0] - mid) * 0.85                           # the target's jaw is 15 % narrower ...
    theta = np.deg2rad(7.0)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    posed = 1.07 * narrow @ rot.T + np.array([0.02, -0.01, 0.03])             # ... and it sits elsewhere, rolled and 7 % larger
    target = jaw_target_from_landmarks(src, posed)
    assert target.jaw_points == pytest.approx(narrow[:17], abs=1e-9)
    assert jaw_width(target.jaw_points) == pytest.approx(0.85 * jaw_width(src[:17]), rel=1e-6)
    with pytest.raises(DomainError):
        jaw_target_from_landmarks(src[:60], posed)
