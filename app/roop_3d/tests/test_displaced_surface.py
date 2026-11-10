"""Analytical refinement contracts, not pretrained skin-detail quality claims."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.reconstruction.displaced_surface import DisplacementSurface, detail_levels, sample_uv_displacement


def square(levels=1):
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    faces = np.array([[0, 2, 1], [1, 2, 3]], np.int32)
    return DisplacementSurface(faces, uv, levels=levels)


@pytest.mark.parametrize("levels,vertices,triangles", [(0, 4, 2), (1, 9, 8), (2, 25, 32)])
def test_refinement_welds_shared_edges_preserves_winding_and_never_adds_nonface_anatomy(levels, vertices, triangles):
    surface = square(levels)
    base = np.column_stack((surface.uvs[:4], np.zeros(4))).astype(np.float32)
    refined = surface.interpolate(base)
    assert len(refined) == vertices
    assert len(surface.faces) == triangles
    assert surface.vertex_count == vertices
    np.testing.assert_array_equal(refined[:4], base)
    np.testing.assert_allclose(refined[:, :2], surface.uvs)
    assert np.all((refined[:, :2] >= 0) & (refined[:, :2] <= 1))
    assert np.all(refined[:, 2] == 0), "linear refinement invented coarse geometry"
    p = refined[surface.faces]
    cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    assert np.all(cross[:, 2] < 0)
    assert (-cross[:, 2]/2).sum() == pytest.approx(1)
    # No slit at the shared diagonal: every nonboundary edge has two owners.
    edges = np.sort(np.concatenate((surface.faces[:, [0, 1]], surface.faces[:, [1, 2]], surface.faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    assert np.all(counts <= 2)
    middle = refined[boundary].mean(1)[:, :2]
    assert np.all(np.any((middle == 0) | (middle == 1), axis=1))


def test_high_frequency_displacement_between_native_vertices_is_not_lost():
    surface = square()
    base = np.column_stack((surface.uvs[:4], np.zeros(4))).astype(np.float32)
    detail = np.zeros((65, 65), np.float32)
    detail[32, 32] = .003
    # Native vertices all miss this measured UV-map feature.
    assert not sample_uv_displacement(detail, surface.uvs[:4]).any()
    result = surface.displace(base, detail)
    centre = np.flatnonzero(np.all(surface.uvs == .5, axis=1))
    assert len(centre) == 1
    assert result[centre[0], 2] == pytest.approx(-.003)
    np.testing.assert_array_equal(result[:4], base)
    # Applying another expression/pose uses exactly the same correspondence IDs.
    shifted = surface.displace(base + (.1, -.1, .4), detail)
    np.testing.assert_allclose(shifted-result, np.tile((.1, -.1, .4), (9, 1)), atol=1e-7)


@pytest.mark.parametrize("requested,profile,expected", [(0, "6gb", 0), (1, "cpu", 1), (2, "6gb", 1), (2, "12gb", 2)])
def test_geometry_detail_budget_is_profile_specific(requested, profile, expected):
    assert detail_levels(requested, profile) == expected


@pytest.mark.parametrize("bad", [True, -1, 3, 1.5, "1"])
def test_refinement_rejects_unbounded_or_ambiguous_levels(bad):
    with pytest.raises(ValueError):
        detail_levels(bad, "6gb")
    with pytest.raises(ValueError):
        DisplacementSurface(np.array([[0, 1, 2]]), np.zeros((3, 2)), levels=bad)


def test_displacement_and_refinement_validate_finite_geometry_and_uvs():
    surface = square()
    with pytest.raises(ValueError, match="compact"):
        surface.displace(np.zeros((5, 3)), np.zeros((64, 64)))
    with pytest.raises(ValueError, match="UVs"):
        sample_uv_displacement(np.zeros((64, 64)), np.array([[np.nan, 0]]))
    with pytest.raises(ValueError, match="2D"):
        sample_uv_displacement(np.full((64, 64), np.inf), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="triangular"):
        DisplacementSurface(np.array([[0, 0, 1]]), np.zeros((3, 2)))


def test_refined_backbone_surface_reaches_both_photographic_warp_and_neural_output():
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
    from roop_3d.reconstruction.morphable_backbone import FaceTopology, MorphableBackbone
    from roop_3d.reconstruction.neural_face import NeuralFaceEngine
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    base = np.column_stack((uv*.6+.1, np.zeros(4))).astype(np.float32)
    faces = np.array([[0, 2, 1], [1, 2, 3]], np.int32)
    topology = FaceTopology("flame2023open", 4, np.arange(4), faces, uv)
    detail = np.zeros((1, 1, 65, 65), np.float32)
    detail[0, 0, 32, 32] = .003
    class Graph:
        def __init__(self, result):
            self.result = result
        def run(self, values):
            return self.result
    coarse = Graph({"identity_vertices_m": base[None], "rotation": np.eye(3)[None],
                    "translation_m": np.array([[0, 0, 1]]), "scale": np.ones((1, 1)),
                    "intrinsics": np.diag([224, 224, 1])[None]})
    expression = Graph({"coefficients": np.zeros((1, 50)), "expressed_vertices_m": base[None]})
    hrn = Graph({"mid_offsets_m": np.zeros((1, 4, 3)), "displacement_uv_m": detail})
    backbone = MorphableBackbone(topology, coarse, expression, hrn, expression_basis="smirk_flame50",
                                evidence={}, detail_subdivisions=1)
    points = ARCFACE_REFERENCE_5PTS.copy()
    points[:, 0] += 8
    image = np.full((128, 128, 3), 45, np.uint8)
    fit = backbone.fit(image, points)
    assert fit.vertices.shape == fit.identity.shape == fit.mid_offsets.shape == (9, 3)
    assert topology.faces.max() == 3 and backbone.surface.faces.max() == 8
    assert fit.vertices[:, 2].min() == pytest.approx(-.003)
    class Enhancer:
        def restore_hires(self, neural, landmarks):
            assert np.all(neural == 180)
            return np.full((256, 256, 3), 200, np.uint8)
    class Swapper:
        model = SimpleNamespace(resolution=128, spec={}, infer=lambda *a: None)
        def synthesize_correspondence(self, conditioning, **kwargs):
            assert conditioning.dtype == np.uint8 and conditioning.shape == image.shape
            return np.full_like(conditioning, 180)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    view = SimpleNamespace(index=0, image_bytes=encoded, landmarks=points, is_synthetic=False,
                           landmarks_trusted=True, yaw=0, pitch=0, quality=SimpleNamespace(score=1))
    document = SimpleNamespace(faceset_id="fixture", views=(view,))
    engine = NeuralFaceEngine(backbone, ReferenceSurfaceRasterizer(), Enhancer())
    engine.prepare(document)
    assert engine.sources["fixture"][0].fit.camera_vertices.shape == (9, 3)
    result = engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, Swapper(), track_id=0)
    assert np.all(result.image == 200)
    assert result.coverage.sum() > 100
    assert engine.last_fields[0].confidence.sum() > 100
    field = engine.last_original_fields[0]
    valid = field.confidence > 0
    assert valid.any() and np.isfinite(field.source_xy[valid]).all()
    assert np.all((field.source_xy[valid] >= 0).all(axis=1))
    assert np.all(field.source_xy[valid, 0] < image.shape[1])
    assert np.all(field.source_xy[valid, 1] < image.shape[0])
    assert np.all(field.source_xy[~valid] == -1)
    assert engine.last_source_view_indices == (0,)
    capability = backbone.capability()
    assert capability["native_vertex_count"] == 4
    assert capability["active_face_vertices"] == 4
    assert capability["warp_surface_vertices"] == 9
    assert capability["detail_subdivisions"] == 1
    assert not capability["raw_mesh_compositing"]
    from roop_3d.core.errors import DomainError
    with pytest.raises(DomainError, match="Prepare source"):
        engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)),
                       SimpleNamespace(faceset_id="missing"), Swapper(), track_id=0)
    assert engine.last_fields == engine.last_original_fields == engine.last_source_view_indices == ()
