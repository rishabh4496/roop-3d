"""Native-camera, articulation and UV regressions without licensed model data."""
import cv2
import numpy as np
import pytest

from roop_3d.reconstruction.dense_correspondence import project
from roop_3d.reconstruction.native_face_geometry import (
    CV_AXES, FlameExpressionRig, SurfaceTransfer, UVSurfaceMap, bfm_camera,
    bfm_rotation, deep3d_crop, sample_hrn_mid,
)


def test_native_camera_matches_independent_upstream_projection_including_tz():
    coefficients = np.zeros(257, np.float32)
    coefficients[224:227] = (.2, -.3, .1)
    coefficients[254:] = (.08, -.12, .7)
    points = np.array([[-.4, .3, .1], [.25, -.4, .2], [.1, .1, .5]], np.float32)
    rotation, translation, k = bfm_camera(coefficients)
    camera = points @ bfm_rotation(coefficients[224:227]).T + coefficients[254:]
    camera[:, 2] = 10-camera[:, 2]
    expected = camera[:, :2]/camera[:, 2:]*1015 +112
    expected[:, 1] = 223-expected[:, 1]
    actual = project((points @ CV_AXES * .1) @ rotation.T+translation, k)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-6)
    assert translation[2] == pytest.approx(.93)


def test_crop_has_exact_resize_pixel_centre_transform():
    standard = np.array([[-.3, .2, .1], [.3, .2, .1], [0, 0, .4], [-.2, -.3, 0], [.2, -.3, 0]])
    # Known scaled orthographic landmarks; intentionally require rounding.
    points = standard[:, :2] * 61.3 + (120.2, 100.8)
    points[:, 1] = 239-points[:, 1]
    y, x = np.mgrid[:240, :280]
    image = np.stack((x % 255, y % 255, (x+y) % 255), -1).astype(np.uint8)
    crop = deep3d_crop(image, points, standard)
    assert crop.image.shape == (224, 224, 3)
    # Affine and inverse return original measured image coordinates exactly.
    fitted = points @ crop.matrix[:, :2].T + crop.matrix[:, 2]
    restored = fitted @ crop.inverse[:, :2].T + crop.inverse[:, 2]
    np.testing.assert_allclose(restored, points, atol=1e-6)
    # Linear coordinate channels track the same transform within bicubic rounding.
    locations = np.array([[90, 80], [110, 110], [130, 130]], np.float64)
    original = locations @ crop.inverse[:, :2].T + crop.inverse[:, 2]
    for (px, py), point in zip(locations.astype(int), original):
        np.testing.assert_allclose(crop.image[py, px, :2], point, atol=1)


def test_surface_transfer_retains_asymmetric_displacements_and_rejects_unregistered_indices():
    transfer = SurfaceTransfer(np.array([[0, 1, 2], [1, 2, 3]]),
                               np.array([[.7, .2, .1], [0, .25, .75]]), 4, 2)
    values = np.zeros((4, 3), np.float32)
    values[0, 1], values[3, 0] = .01, -.02
    np.testing.assert_allclose(transfer(values), [[0, .007, 0], [-.015, 0, 0]], atol=1e-9)
    with pytest.raises(ValueError, match="convex"):
        SurfaceTransfer(np.zeros((1, 3), int), np.zeros((1, 3)), 4, 1)
    with pytest.raises(ValueError, match="indices"):
        SurfaceTransfer(np.array([[0, 1, 5]]), np.ones((1, 3))/3, 4, 1)


def rig_fixture():
    n, j = 5023, 5
    template = np.zeros((n, 3), np.float32)
    template[0], template[1] = (0, 1, 0), (0, -1, 0)
    expression = np.zeros((n, 3, 50), np.float32)
    expression[2, 0, 0] = .01
    regressor = np.zeros((j, n), np.float32)
    regressor[:, 3] = 1  # all joints at origin, independent of expression
    weights = np.zeros((n, j), np.float32)
    weights[:, 0], weights[0, 0], weights[0, 2] = 1, 0, 1
    eyelids = np.zeros((2, n, 3), np.float32)
    eyelids[0, 4, 1], eyelids[1, 5, 1] = -.002, -.003
    return {"template_m": template, "expression_dirs_m": expression, "smirk_to_expression": np.eye(50),
            "parents": np.array([-1, 0, 1, 1, 1]), "joint_regressor": regressor,
            "skinning_weights": weights, "pose_dirs_m": np.zeros((36, n*3)), "eyelid_dirs_m": eyelids}


def test_flame_decoder_articulates_jaw_without_mirroring_smirk_or_eyelids():
    data = rig_fixture()
    rig = FlameExpressionRig(data)
    coefficients = np.zeros(50, np.float32)
    coefficients[0] = .8
    result = rig.express(rig.template, coefficients, (0, 0, np.pi/2), (.9, 0))
    np.testing.assert_allclose(result[0], (-1, 0, 0), atol=1e-6)
    np.testing.assert_allclose(result[1], (0, -1, 0), atol=1e-6)
    assert result[2, 0] == pytest.approx(.008)
    assert result[4, 1] == pytest.approx(-.0018)
    assert np.all(result[5] == 0), "left eyelid leaked into right eyelid"
    neutral = rig.express(rig.template, np.zeros(50), (0, 0, 0), (0, 0))
    np.testing.assert_allclose(neutral, rig.template, atol=1e-7)


def test_uv_lookup_uses_native_hrn_ndc_pixel_centres_and_linear_attributes():
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    lookup = UVSurfaceMap(uv, np.array([[0, 1, 2]]), size=32)
    values = np.column_stack((uv, np.ones(3))).astype(np.float32)
    result = lookup.interpolate(values)
    y, x = np.mgrid[:32, :32]
    expected = np.stack(((x+.5)/32, 1-(y+.5)/32, np.ones_like(x)), -1)
    np.testing.assert_allclose(result[lookup.valid], expected[lookup.valid], atol=1e-6)
    assert lookup.valid.sum() > 400


def test_mid_uv_sampling_matches_upstream_hrn_formula_at_noninteger_uvs():
    rng = np.random.default_rng(82)
    deformation = rng.normal(size=(64, 64, 3)).astype(np.float32)
    uvs = rng.uniform(.02, .98, (35709, 2)).astype(np.float32)
    actual = sample_hrn_mid(deformation, uvs)
    # Scalar implementation independent of the vectorized runtime path.
    for index in (0, 99, 2048, 35708):
        x, y = uvs[index] * 64
        ix, iy = int(x), int(y)
        ax, ay = x-ix, y-iy
        expected = ((deformation[63-iy, ix]*(1-ax) + deformation[63-iy, ix+1]*ax)*ay
                    + (deformation[64-iy, ix]*(1-ax)+deformation[64-iy, ix+1]*ax)*(1-ay))
        np.testing.assert_allclose(actual[index], expected, atol=2e-6)


@pytest.mark.parametrize("family,native_count", [("bfm35709", 35709), ("flame2023open", 5023)])
@pytest.mark.parametrize("subdivisions,warp_vertices", [(0, 4), (1, 9)])
def test_native_three_tier_fit_routes_real_geometry_decoders_for_both_surface_families(family, native_count, subdivisions, warp_vertices):
    from roop_3d.reconstruction.native_backbone import NativeMorphableBackbone
    from roop_3d.reconstruction.morphable_backbone import FaceTopology
    from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
    # This is a pipeline contract fixture, not an imitation pretrained model.
    # Full native vertex counts catch BFM/FLAME index mixing and remap limits.
    mean = np.zeros((35709, 3), np.float32)
    mean[:4] = [[-.3, -.3, 0], [.3, -.3, 0], [-.3, .3, 0], [.3, .3, 0]]
    uv = np.full((35709, 2), .5, np.float32)
    uv[:4] = [[.1, .1], [.9, .1], [.1, .9], [.9, .9]]
    faces = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
    bfm_topology = FaceTopology("bfm35709", 35709, np.arange(4), faces, uv[:4])
    topology = FaceTopology(family, native_count, np.arange(4), faces, uv[:4])
    data = rig_fixture()
    data["template_m"][:] = 0
    data["template_m"][:4] = mean[:4]*.1
    data["expression_dirs_m"][:] = 0
    data["expression_dirs_m"][2, 0, 0] = .001
    rig = FlameExpressionRig(data)
    standard = np.array([[-.3, .2, .1], [.3, .2, .1], [0, 0, .4], [-.2, -.3, 0], [.2, -.3, 0]], np.float32)
    points = standard[:, :2]*100 + (128, 128)
    points[:, 1] = 255-points[:, 1]
    bfm = {"mean_shape": mean, "identity_basis": np.zeros((35709, 3, 80), np.float32),
           "expression_basis": np.zeros((35709, 3, 64), np.float32), "hrn_uvs": uv,
           "standard5": standard, "landmarks68": np.arange(68) % 4}
    def registered(n):
        indices = np.zeros((n, 3), int)
        indices[:4, 0] = np.arange(4)
        weights = np.zeros((n, 3), np.float32)
        weights[:, 0] = 1
        return indices, weights
    b2f_i, b2f_w = registered(5023)
    f2b_i, f2b_w = registered(35709)
    registration = {"bfm_to_flame_indices": b2f_i, "bfm_to_flame_weights": b2f_w,
                    "flame_to_bfm_indices": f2b_i, "flame_to_bfm_weights": f2b_w,
                    "flame_to_bfm_rotation": np.eye(3), "flame_to_bfm_translation_m": np.zeros(3),
                    "flame_to_bfm_scale": 1., "flame_hrn_uvs": uv[:5023]}
    seen = []
    class Graph:
        def __init__(self, role):
            self.role = role
        def run(self, inputs):
            seen.append((self.role, next(iter(inputs.values())).shape))
            if self.role == "coarse":
                return {"coefficients": np.zeros((1, 257), np.float32)}
            if self.role == "expression":
                expression = np.zeros((1, 50), np.float32)
                expression[0, 0] = .8
                return {"expression": expression, "jaw": np.zeros((1, 3), np.float32), "eyelids": np.array([[.9, 0]], np.float32)}
            if self.role == "hrn_mid":
                return {"deformation": np.full((1, 3, 64, 64), .01, np.float32)}
            return {"displacement": np.full((1, 1, 256, 256), .02, np.float32)}
        def release(self):
            pass
    graphs = {role: Graph(role) for role in ("coarse", "expression", "hrn_mid", "hrn_high")}
    backbone = NativeMorphableBackbone(topology, bfm_topology, bfm, rig, registration,
        graphs, ReferenceSurfaceRasterizer(), {"hrn_alpha": 1., "hrn_beta": 1.}, detail_subdivisions=subdivisions)
    try:
        result = backbone.fit(np.full((256, 256, 3), 180, np.uint8), points)
        assert seen == [("coarse", (1, 3, 224, 224)), ("expression", (1, 3, 224, 224)),
                        ("hrn_mid", (1, 6, 64, 64)), ("hrn_high", (1, 9, 256, 256))]
        assert result.vertices.shape == result.identity.shape == result.mid_offsets.shape == (warp_vertices, 3)
        assert backbone.surface.vertex_count == warp_vertices
        assert len(backbone.surface.faces) == len(faces)*4**subdivisions
        assert backbone.topology.vertex_count == native_count
        assert np.isfinite(result.camera_vertices).all()
        assert result.mid_offsets.max() > 0
        assert result.displacement_uv.max() == pytest.approx(.0002)
        assert result.expression[0] == pytest.approx(.8)
        np.testing.assert_allclose(result.eyelids, [.9, 0])
        assert result.scale == 1
    finally:
        backbone.release()
