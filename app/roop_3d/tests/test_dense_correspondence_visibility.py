"""Analytic visibility regressions only, not evidence of real-model image quality."""
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.dense_correspondence import (
    GPUSurfaceRasterizer, ReferenceSurfaceRasterizer, WarpField, correspondence_field, project,
)


SIZE = (32, 32)
FACES = np.array([[0, 1, 2]], np.int32)
K = np.eye(3, dtype=np.float32)


def scene(offset=(.25, .25), slant=False):
    pixels = np.array([[1, 1], [29, 1], [1, 29]], np.float32)
    z = 1 / (1 - .015 * pixels[:, 0] - .008 * pixels[:, 1]) if slant else np.ones(3)
    source = np.column_stack((pixels * z[:, None], z)).astype(np.float32)
    target = source.copy()
    target[:, :2] -= np.asarray(offset) * z[:, None]
    raster = ReferenceSurfaceRasterizer()
    depth = raster.rasterize(source, FACES, K, source, SIZE).depth
    return raster, source, target, depth


def field_for(raster, source, target, depth, **kwargs):
    return correspondence_field(raster, FACES, source, K, target, K,
                                SIZE, depth, SIZE, **kwargs)


@pytest.mark.parametrize("slant", [False, True])
def test_identity_and_fractional_slant_have_no_interior_holes(slant):
    raster, source, target, depth = scene(slant=slant)
    field = field_for(raster, source, target, depth, tolerance_m=1e-5)
    assert np.all(field.confidence[4:10, 4:10] == 1)
    np.testing.assert_allclose(field.source_xy[5, 5], [5.25, 5.25], atol=2e-6)
    if slant:
        assert abs(depth[5, 5] - depth[6, 6]) > .02
    identity = field_for(raster, source, source, depth, tolerance_m=1e-5)
    assert identity.confidence[5, 5] == 1
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    valid = identity.confidence > 0
    np.testing.assert_array_equal(identity.sample(image)[valid], image[valid])


@pytest.mark.parametrize("tap", [(5, 6), (6, 5), (6, 6)])
@pytest.mark.parametrize("obstruction", ["occluder", "background", "mask"])
def test_each_non_nearest_bilinear_tap_must_be_visible_and_observed(tap, obstruction):
    raster, source, target, depth = scene(slant=True)
    support = np.ones((32, 32), np.float32)
    assert field_for(raster, source, target, depth).confidence[5, 5] == 1
    if obstruction == "mask":
        support[tap] = 0
    else:
        depth[tap] = .5 if obstruction == "occluder" else np.inf
    field = field_for(raster, source, target, depth, source_support=support)
    assert field.confidence[5, 5] == 0
    assert np.all(field.source_xy[5, 5] == -1)
    assert field.confidence[10, 5] == 1


def test_actual_rasterized_occluder_boundary_rejects_other_tap():
    raster, source, target, depth = scene()
    # Tiny foreground triangle covers (6, 5) but not nearest sample (5, 5).
    front = np.array([[5.6, 4.6, 1], [6.4, 4.6, 1], [6, 5.4, 1]], np.float32) * .5
    vertices = np.vstack((source, front))
    faces = np.vstack((FACES, [3, 4, 5])).astype(np.int32)
    occluded = raster.rasterize(vertices, faces, K, vertices, SIZE).depth
    assert occluded[5, 5] == depth[5, 5]
    assert occluded[5, 6] == .5
    assert field_for(raster, source, target, occluded).confidence[5, 5] == 0


def test_zero_weight_taps_not_required_and_quantized_sampling_agrees():
    raster, source, target, depth = scene(offset=(0, 0))
    support = np.ones((32, 32), np.float32)
    support[5, 6] = support[6, 5] = support[6, 6] = 0
    assert field_for(raster, source, target, depth, source_support=support).confidence[5, 5] == 1
    raster, source, target, depth = scene(offset=(.01, .01))
    field = field_for(raster, source, target, depth, source_support=support)
    assert field.confidence[5, 5] == 1  # rounds to an integer in OpenCV's table
    image = np.zeros((32, 32), np.float32)
    image[5, 5] = 1
    assert field.sample(image)[5, 5] == 1


def test_triangle_silhouette_and_image_boundary_reject_unavailable_taps():
    raster, source, target, depth = scene()
    # (15.25, 14.25) is inside, but its (16, 15) tap is outside x+y<=30.
    assert field_for(raster, source, target, depth).confidence[14, 15] == 0
    source = np.array([[-10, -10, 1], [70, -10, 1], [-10, 70, 1]], np.float32)
    target = source.copy()
    target[:, :2] -= .25
    depth = raster.rasterize(source, FACES, K, source, SIZE).depth
    field = field_for(raster, source, target, depth)
    assert field.confidence[5, 30] == 1
    assert field.confidence[5, 31] == 0
    assert np.isfinite(field.source_xy).all()


def test_affine_crop_intrinsics_allow_rotation_skew_and_reflection():
    raster, source, _, _ = scene(offset=(0, 0))
    # A valid affine crop need not have positive diagonal focal entries.
    k = np.array([[0, -1, 31], [1, .2, 0], [0, 0, 1]], np.float32)
    depth = raster.rasterize(source, FACES, k, source, SIZE).depth
    field = correspondence_field(raster, FACES, source, k, source, k, SIZE, depth, SIZE)
    assert field.confidence.sum() > 200
    assert np.isfinite(project(source, k)).all()


@pytest.mark.parametrize("bad", ["vertices", "faces_shape", "faces_float", "negative", "overflow",
                                 "attributes", "size", "nan_k", "singular_k", "homogeneous_k"])
@pytest.mark.parametrize("gpu", [False, True])
def test_invalid_mesh_inputs_rejected_before_gl_or_cv2(bad, gpu):
    raster, vertices, _, _ = scene()
    faces, k, attributes, size = FACES, K.copy(), vertices.copy(), SIZE
    if bad == "vertices":
        vertices[0, 0] = np.nan
    elif bad == "faces_shape":
        faces = np.array([0, 1, 2])
    elif bad == "faces_float":
        faces = FACES.astype(float)
    elif bad == "negative":
        faces = np.array([[0, -1, 2]])
    elif bad == "overflow":
        faces = np.array([[0, 1, 3]])
    elif bad == "attributes":
        attributes = np.zeros((3, 2))
    elif bad == "size":
        size = (32767, 32)
    elif bad == "nan_k":
        k[0, 0] = np.nan
    elif bad == "singular_k":
        k[0] = 0
    else:
        k[2] = [0, 0, 0]
    raster = GPUSurfaceRasterizer() if gpu else raster
    with pytest.raises(ValueError):
        raster.rasterize(vertices, faces, k, attributes, size)


@pytest.mark.parametrize("bad", ["depth_shape", "depth_nan", "depth_negative", "support_shape",
                                 "support_nan", "support_range", "tolerance_nan", "tolerance_negative"])
def test_invalid_correspondence_inputs(bad):
    raster, source, target, depth = scene()
    support = np.ones((32, 32), np.float32)
    tolerance = .001
    if bad == "depth_shape":
        depth = depth[:2]
    elif bad == "depth_nan":
        depth[5, 5] = np.nan
    elif bad == "depth_negative":
        depth[5, 5] = -1
    elif bad == "support_shape":
        support = support[:2]
    elif bad == "support_nan":
        support[5, 5] = np.nan
    elif bad == "support_range":
        support[5, 5] = 2
    elif bad == "tolerance_nan":
        tolerance = np.nan
    else:
        tolerance = -1
    with pytest.raises(ValueError):
        field_for(raster, source, target, depth, source_support=support, tolerance_m=tolerance)


@pytest.mark.parametrize("bad", ["map_nan", "map_shape", "confidence_shape", "image_nan", "image_shape"])
def test_sample_rejects_invalid_inputs_before_opencv(bad):
    xy = np.zeros((4, 4, 2), np.float32)
    confidence = np.ones((4, 4), np.float32)
    image = np.zeros((8, 8, 3), np.float32)
    if bad == "map_nan":
        xy[0, 0, 0] = np.nan
    elif bad == "map_shape":
        xy = xy[..., 0]
    elif bad == "confidence_shape":
        confidence = confidence[:2]
    elif bad == "image_nan":
        image[0, 0, 0] = np.nan
    else:
        image = np.zeros(8)
    with pytest.raises(ValueError):
        WarpField(xy, confidence).sample(image)


def test_gpu_slanted_footprint_visibility_matches_reference_interior():
    _, source, target, _ = scene(slant=True)
    gpu = GPUSurfaceRasterizer()
    try:
        try:
            depth = gpu.rasterize(source, FACES, K, source, SIZE).depth
        except DomainError as error:
            pytest.skip(str(error))
        field = field_for(gpu, source, target, depth)
        assert np.all(field.confidence[4:10, 4:10] == 1)
        support = np.ones((32, 32), np.float32)
        support[6, 6] = 0
        assert field_for(gpu, source, target, depth, source_support=support).confidence[5, 5] == 0
    finally:
        gpu.release()


def test_raster_area_budget_before_allocation():
    from roop_3d.reconstruction.dense_correspondence import _size
    assert _size((4096, 4096)) == (4096, 4096)
    with pytest.raises(ValueError, match="raster budget"):
        _size((4097, 4096))
    with pytest.raises(ValueError, match="raster budget"):
        _size((np.int32(32766), np.int32(32766)))
