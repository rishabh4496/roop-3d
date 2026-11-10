"""Barycentric mesh bridge: exact closest-point maths, then conversion of positions / normals / UV albedo in both directions.

Synthetic part: two *different triangulations of the same analytic head-like surface* (a lat-long grid with UV seams and an
icosphere with no UVs), so the true correspondence is known. Real part: MakeHuman <-> FLAME 2023 Open, skipped when FLAME is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import mesh_bridge as mb
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.topology import MorphableTopologyRegistry, TopologyMesh

RADII = np.array([0.085, 0.110, 0.095])


def surface_point(direction: np.ndarray) -> np.ndarray:
    """Analytic head-like surface: an ellipsoid with a nose-like bump toward +z (identity-asset frame)."""
    d = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    p = d * RADII
    bump = 0.02 * np.exp(-((d[..., 0] ** 2) + (d[..., 1] - 0.05) ** 2) / 0.03) * (d[..., 2] > 0)
    p[..., 2] = p[..., 2] + bump
    return p


def orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = float(np.einsum("fc,fc->f", normal, tri.mean(axis=1) - vertices.mean(axis=0)).sum())
    return faces if outward > 0 else faces[:, [0, 2, 1]]


def latlong_head(n_lat: int = 28, n_lon: int = 40) -> TopologyMesh:
    """Open at the poles, vertices duplicated along the seam so every vertex has exactly one UV (like the MakeHuman template)."""
    lat = np.linspace(0.12, np.pi - 0.12, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon + 1)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # direction with +z toward the viewer: longitude 0 faces +z
    d = np.stack([np.sin(la) * np.sin(lo), -np.cos(la), np.sin(la) * np.cos(lo)], axis=-1)
    vertices = surface_point(d.reshape(-1, 3))
    uvs = np.stack([(lo / (2 * np.pi)).ravel(), 1.0 - (la / np.pi).ravel()], axis=-1)
    index = np.arange(n_lat * (n_lon + 1)).reshape(n_lat, n_lon + 1)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b, c, e = index[i, j], index[i, j + 1], index[i + 1, j], index[i + 1, j + 1]
            faces += [[a, c, b], [b, c, e]]
    faces = orient_outward(vertices, np.array(faces))
    return TopologyMesh("synthetic_latlong", vertices, faces, uvs=uvs, uv_provenance="synthetic")


def icosphere_head(subdivisions: int = 3) -> TopologyMesh:
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
                  [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
                  [8, 6, 7], [9, 8, 1]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdivisions):
        cache, verts, faces = {}, list(v), []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(verts), np.array(faces)
    vertices = surface_point(v)
    return TopologyMesh("synthetic_icosphere", vertices, orient_outward(vertices, f))


@pytest.fixture(scope="module")
def pair():
    return latlong_head(), icosphere_head()


@pytest.fixture(scope="module")
def bridge(pair):
    return mb.BarycentricMeshBridge.build(pair[0], pair[1], matched_mm=8.0)


def deform(points: np.ndarray) -> np.ndarray:
    """A smooth displacement field applied identically to whichever mesh we hand it."""
    return points + np.stack([0.006 * np.sin(points[:, 1] * 40), 0.004 * points[:, 0] / 0.085, 0.005 * np.cos(points[:, 0] * 30)], axis=1)


# ------------------------------------------------------------------------------------------------------ exact maths
def test_closest_barycentric_cases():
    a, b, c = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cases = {"interior": ([0.25, 0.25, 1.0], [0.5, 0.25, 0.25]), "vertex_a": ([-1.0, -1.0, 0.3], [1, 0, 0]),
             "vertex_b": ([2.0, -1.0, 0.0], [0, 1, 0]), "vertex_c": ([-1.0, 2.0, 0.0], [0, 0, 1]),
             "edge_ab": ([0.5, -1.0, 0.2], [0.5, 0.5, 0]), "edge_ac": ([-1.0, 0.5, 0.0], [0.5, 0, 0.5]),
             "edge_bc": ([1.0, 1.0, 0.0], [0, 0.5, 0.5])}
    for name, (point, expected) in cases.items():
        w = mb._closest_bary(np.array(point), a, b, c)
        assert np.allclose(w, expected, atol=1e-12), name
        assert abs(w.sum() - 1.0) < 1e-12


def test_closest_points_match_brute_force(pair):
    mesh = pair[1]
    rng = np.random.default_rng(5)
    points = mesh.vertices[rng.choice(len(mesh.vertices), 40, replace=False)] + rng.normal(0, 0.006, (40, 3))
    ids, bary, dist = mb.closest_points_on_mesh(points, mesh.vertices, mesh.faces)
    tri = mesh.vertices[mesh.faces]
    for p, d in zip(points, dist):
        w = mb._closest_bary(np.broadcast_to(p, (len(tri), 3)), tri[:, 0], tri[:, 1], tri[:, 2])
        q = w[:, :1] * tri[:, 0] + w[:, 1:2] * tri[:, 1] + w[:, 2:] * tri[:, 2]
        assert abs(np.linalg.norm(q - p, axis=1).min() - d) < 1e-10
    assert np.allclose(bary.sum(axis=1), 1.0) and (bary >= 0).all()


def test_empty_mesh_is_rejected():
    with pytest.raises(DomainError):
        mb.closest_points_on_mesh(np.zeros((1, 3)), np.zeros((3, 3)), np.zeros((0, 3), int))


# ------------------------------------------------------------------------------------------------ synthetic bridge
def test_registration_recovers_a_known_similarity(pair):
    source = pair[0]
    scale, angle = 1.07, 0.12
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    moved = TopologyMesh("moved", scale * pair[1].vertices @ rot.T + np.array([0.004, -0.006, 0.01]), pair[1].faces)
    reg = mb.register_surfaces(source, moved, nonrigid=False)
    back = reg.apply(moved.vertices)
    _, _, d = mb.closest_points_on_mesh(back, source.vertices, source.faces)
    assert np.percentile(d, 80) < 0.0015 and abs(reg.scale - 1 / scale) < 0.03


def test_bridge_reports_its_evidence(bridge):
    r = bridge.report()
    assert r["source"] == "synthetic_latlong" and r["target"] == "synthetic_icosphere"
    for direction in ("to_target", "to_source"):
        assert {"mean_mm", "p95_mm", "max_mm", "matched_fraction", "vertices"} <= set(r[direction])
    assert r["to_target"]["mean_mm"] < 2.0 and r["to_target"]["matched_fraction"] > 0.8
    assert r["registration"]["rigid_p95_mm"] < 6.0 and "uv_overlap_fraction" in r["uv"]
    assert "not anatomical" in r["note"]


def test_derived_uv_layout_is_labelled_and_injective(bridge):
    target = bridge.target
    assert target.uvs is not None and len(target.uvs) == len(target.vertices) == target.vertex_count
    assert target.uv_provenance.startswith("derived from synthetic_latlong")
    assert bridge.uv_stats["uv_overlap_fraction"] < 0.05
    assert target.source_index is not None and np.allclose(target.vertices, bridge.target.vertices)
    assert target.uvs.min() >= -1e-9 and target.uvs.max() <= 1 + 1e-9


def test_positions_follow_the_source_deformation(bridge, pair):
    source = pair[0]
    moved_source = deform(source.vertices)
    converted = bridge.convert_positions(moved_source, "target")
    truth = deform(bridge.target_in_source)          # the same field, evaluated on the target's own rest surface
    matched = bridge.matched("target")
    err = np.linalg.norm(converted - truth, axis=1)[matched] * 1000
    # 8 mm of displacement field, ~20 mm triangles: the transfer must recover it to a small fraction of that
    assert np.median(err) < 1.0 and np.percentile(err, 95) < 3.0


def test_round_trip_is_close_to_identity(bridge, pair):
    source = pair[0]
    moved = deform(source.vertices)
    back = bridge.convert_positions(bridge.convert_positions(moved, "target"), "source")
    err = np.linalg.norm(back - moved, axis=1)[bridge.matched("source")] * 1000
    assert np.median(err) < 1.5 and np.percentile(err, 90) < 4.0


def test_rest_conversion_reproduces_the_registered_target(bridge, pair):
    rest = bridge.convert_positions(pair[0].vertices, "target")
    ok = bridge.matched("target")
    # the normal offset is carried along the Phong-interpolated normal, so only a sub-millimetre difference from the exact
    # face normal remains
    assert np.linalg.norm(rest - bridge.target_in_source, axis=1)[ok].max() < 1e-3


def test_normals_are_unit_and_outward(bridge, pair):
    n = bridge.convert_normals(pair[0].normals(), "target")
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    radial = bridge.target_in_source - bridge.target_in_source.mean(axis=0)
    cos = np.einsum("nc,nc->n", n, radial / np.linalg.norm(radial, axis=1, keepdims=True))
    assert (cos[bridge.matched("target")] > 0.5).mean() > 0.95


def test_generic_attribute_transfer_interpolates_barycentrically(bridge, pair):
    constant = np.full((len(pair[0].vertices), 2), 3.5)
    assert np.allclose(bridge.transfer(constant, "target"), 3.5)
    with pytest.raises(DomainError) as error:
        bridge.transfer(np.zeros((5, 3)), "target")
    assert error.value.code == "BRIDGE_SHAPE_MISMATCH"
    with pytest.raises(DomainError):
        bridge.transfer(constant, "sideways")


def _position_texture(mesh: TopologyMesh, positions: np.ndarray, size: int):
    tri, bary = mb.rasterize_uv(mesh.uvs, mesh.faces, size)
    covered = tri >= 0
    tex = np.zeros((size, size, 3), np.float32)
    tex[covered] = np.einsum("nc,ncd->nd", bary[covered], positions[mesh.faces[tri[covered]]])
    return mb.pad_texels(tex, covered, 6).astype(np.float32), covered


@pytest.mark.parametrize("direction", ["target", "source"])
def test_uv_albedo_is_transferred_by_geometry(bridge, pair, direction):
    """Colour == 3D position: after conversion every texel must hold (nearly) the position of the surface point it now covers."""
    size = 192
    src_mesh, dst_mesh = (bridge.source, bridge.target) if direction == "target" else (bridge.target, bridge.source)
    src_pos = bridge.source.vertices if direction == "target" else bridge.target_in_source
    dst_pos = bridge.target_in_source if direction == "target" else bridge.source.vertices
    texture, _ = _position_texture(src_mesh, src_pos, size)
    out = bridge.convert_texture(texture, direction)
    expected, covered = _position_texture(dst_mesh, dst_pos, size)
    ok = out["matched"] & covered
    assert ok.sum() > 0.5 * covered.sum()
    err = np.linalg.norm(out["texture"][ok] - expected[ok], axis=1) * 1000
    assert np.median(err) < 1.5 and np.percentile(err, 95) < 4.0


def test_texel_state_is_nearest_sampled_and_unmatched_texels_become_unseen(bridge):
    size = 96
    state = np.full((size, size), 2, np.uint8)
    out = bridge.convert_texture(np.full((size, size, 3), 200, np.uint8), "target", texel_state=state)
    assert out["texel_state"].dtype == np.uint8 and set(np.unique(out["texel_state"])) <= {0, 2}
    assert (out["texel_state"][out["matched"]] == 2).all() and (out["texel_state"][~out["matched"]] == 0).all()
    assert out["texture"].dtype == np.uint8
    assert (out["texture"][out["matched"]] == 200).all()


def test_texture_resolution_can_change(bridge):
    out = bridge.convert_texture(np.full((64, 64, 3), 90, np.uint8), "target", out_size=128)
    assert out["texture"].shape == (128, 128, 3)


def test_texture_transfer_needs_both_uv_layouts(pair):
    bare = mb.BarycentricMeshBridge.build(pair[1], pair[1].with_vertices(pair[1].vertices * 1.0))
    # neither mesh has UVs and the donor has none to give: conversion must say so rather than guess
    with pytest.raises(DomainError) as error:
        bare.convert_texture(np.zeros((16, 16, 3), np.uint8), "target")
    assert error.value.code == "UV_LAYOUT_MISSING"


def test_derive_uv_needs_a_donor_layout(pair):
    with pytest.raises(DomainError) as error:
        mb.derive_uv_layout(pair[1], pair[1], pair[1].vertices)
    assert error.value.code == "UV_DONOR_MISSING"


def test_overlap_metric_detects_a_folded_layout():
    uvs = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    assert mb.uv_overlap_fraction(uvs, np.array([[0, 1, 2], [1, 3, 2]]), 64) < 0.05          # a quad split in two: no overlap
    assert mb.uv_overlap_fraction(uvs, np.array([[0, 1, 2], [0, 1, 2]]), 64) > 0.9           # the same triangle twice: folded


# -------------------------------------------------------------------------------------------------- real assets
_registry = MorphableTopologyRegistry()
real = pytest.mark.skipif(not ((DEFAULT_ASSET_DIR / ASSET_NAME).exists() and _registry.availability("flame2023_open")["available"]),
                          reason="needs the built MakeHuman asset and FLAME 2023 Open")


@pytest.fixture(scope="module")
def real_bridge():
    return _registry.bridge("makehuman_cc0", "flame2023_open")


@real
def test_real_registration_and_coverage(real_bridge):
    r = real_bridge.report()
    assert r["registration"]["rigid_p95_mm"] < 10.0 and 0.85 < r["registration"]["similarity_scale"] < 1.1
    assert r["to_target"]["matched_fraction"] > 0.9 and r["to_source"]["matched_fraction"] > 0.85
    assert r["to_target"]["mean_mm"] < 4.0 and r["registration"]["nonrigid_folded_fraction"] < 0.005
    assert r["uv"]["uv_overlap_fraction"] < 0.03, "the derived FLAME atlas must be (nearly) injective"
    assert "derived" in r["uv_provenance"]


@real
def test_real_positions_move_with_the_source(real_bridge):
    a = real_bridge.source
    delta = np.zeros_like(a.vertices)
    jaw = a.vertices[:, 1] > np.percentile(a.vertices[:, 1], 70)
    delta[jaw, 1] = 0.01                                           # drop the whole lower head by 10 mm
    moved = real_bridge.convert_positions(a.vertices + delta, "target")
    rest = real_bridge.convert_positions(a.vertices, "target")
    t = real_bridge.target_in_source
    lower = (t[:, 1] > np.percentile(a.vertices[:, 1], 78)) & real_bridge.matched("target")
    upper = (t[:, 1] < np.percentile(a.vertices[:, 1], 40)) & real_bridge.matched("target")
    assert 0.007 < np.median((moved - rest)[lower, 1]) < 0.0115
    assert np.abs((moved - rest)[upper, 1]).max() < 0.002


@real
@pytest.mark.parametrize("direction", ["target", "source"])
def test_real_albedo_transfer_is_geometrically_consistent(real_bridge, direction):
    size = 256
    src_mesh, dst_mesh = (real_bridge.source, real_bridge.target) if direction == "target" else (real_bridge.target, real_bridge.source)
    src_pos = real_bridge.source.vertices if direction == "target" else real_bridge.target_in_source
    dst_pos = real_bridge.target_in_source if direction == "target" else real_bridge.source.vertices
    texture, _ = _position_texture(src_mesh, src_pos, size)
    out = real_bridge.convert_texture(texture, direction)
    expected, covered = _position_texture(dst_mesh, dst_pos, size)
    ok = out["matched"] & covered
    assert ok.sum() > 0.8 * covered.sum()
    err = np.linalg.norm(out["texture"][ok] - expected[ok], axis=1) * 1000
    assert np.median(err) < 2.5 and np.percentile(err, 95) < 8.0
