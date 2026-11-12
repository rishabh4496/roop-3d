"""The dense swap crop is registered to the swapper template from landmarks, never framed by the mesh silhouette."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.engine.volumetric.preprocess import aligned_crop, template_points, transform_points
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
from roop_3d.reconstruction.neural_face import NeuralFaceEngine

FACES = np.array([[0, 2, 1], [1, 2, 3]], np.int32)


def landmarks(scale=1.0, degrees=0.0, shift=(0.0, 0.0)):
    base = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
    base[:, 0] += 8
    centre = base.mean(0)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return (base - centre) @ (scale * np.array(((c, -s), (s, c)))).T + centre + shift


def make_engine(jaw_drop=0.0, size=128, template="arcface_112", enhanced=256, yaws=(0,)):
    # A quad covering the face; ``jaw_drop`` lowers the two chin vertices (an open mouth grows the silhouette bbox).
    vertices = np.array([[.2, .1, 1], [.8, .1, 1], [.2, .8 + jaw_drop, 1], [.8, .8 + jaw_drop, 1]], np.float32)

    class Backbone:
        topology = SimpleNamespace(faces=FACES)

        def fit(self, image, points):
            return SimpleNamespace(camera_vertices=vertices, intrinsics=np.diag([224, 224, 1]), rotation=np.eye(3),
                                   translation=np.zeros(3, np.float32), scale=1.0, identity=vertices.copy(),
                                   vertices=vertices.copy(), expression_vertices=vertices.copy(),
                                   alignment=aligned_crop(image, np.asarray(points), 128))

    class Enhancer:
        def restore_hires(self, crop, points):
            self.points = np.asarray(points)
            return np.full((enhanced, enhanced, 3), 200, np.uint8)

    class Model:
        resolution = size
        spec = {"alignment_template": template}

        def infer(self, crop, embedding, *, source_view):
            return crop

    class Swapper:
        model = Model()

        def synthesize_correspondence(self, conditioning, **kwargs):
            return conditioning

    image = np.full((300, 300, 3), 60, np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    views = tuple(SimpleNamespace(index=i, image_bytes=encoded, landmarks=landmarks(), is_synthetic=False,
                                  landmarks_trusted=True, yaw=yaw, pitch=0, quality=SimpleNamespace(score=1))
                  for i, yaw in enumerate(yaws))
    document = SimpleNamespace(faceset_id="f", views=views)
    engine = NeuralFaceEngine(Backbone(), ReferenceSurfaceRasterizer(), Enhancer())
    engine.prepare(document)
    return engine, document, Swapper(), image


@pytest.mark.parametrize("scale,degrees,shift", [(1.0, 0, (0, 0)), (1.6, 0, (30, -12)), (0.7, 35, (-20, 25)),
                                                 (1.2, -170, (5, 5))])
def test_landmarks_land_on_template_at_any_scale_roll_and_position(scale, degrees, shift):
    engine, document, swapper, image = make_engine()
    points = landmarks(scale, degrees, shift)
    result = engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper, track_id=0)
    # result.landmarks5 are the crop-space points handed to UltraMax, scaled to the enhanced crop (256 / 128).
    np.testing.assert_allclose(result.landmarks5 / 2.0, template_points(128), atol=1e-6)
    np.testing.assert_allclose(transform_points(result.landmarks5, result.inverse), points, atol=1e-5)


def test_crop_does_not_depend_on_visible_mesh_extent():
    points = landmarks(1.3, 12, (8, -4))
    closed, document, swapper, image = make_engine(jaw_drop=0.0)
    opened, document2, swapper2, image2 = make_engine(jaw_drop=0.45)
    face = SimpleNamespace(landmarks5=points, pose=(0, 0, 0))
    a = closed.process(image, face, document, swapper, track_id=0)
    b = opened.process(image2, face, document2, swapper2, track_id=0)
    np.testing.assert_allclose(a.inverse, b.inverse, atol=1e-9)
    np.testing.assert_allclose(closed.last_diag["crop_matrix"], opened.last_diag["crop_matrix"], atol=1e-9)
    # ... while the rasterised silhouette inside the unchanged crop does follow the mesh.
    assert (b.coverage > .5).sum() > (a.coverage > .5).sum() * 1.2


@pytest.mark.parametrize("size,template", [(128, "arcface_112"), (256, "arcface_112"), (512, "ffhq_512")])
def test_swapper_template_selection_matches_the_2d_swap_path(size, template):
    engine, document, swapper, image = make_engine(size=size, template=template, enhanced=size)
    points = landmarks(1.1, 5, (3, 3))
    engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper, track_id=0)
    registered = transform_points(points, engine.last_diag["crop_matrix"])
    np.testing.assert_allclose(registered, template_points(size, template), atol=2.0 * size / 128)


def test_source_views_hand_off_gradually_and_are_logged_while_the_head_turns():
    engine, document, swapper, image = make_engine(yaws=(-45, 0, 45, 80))
    points = landmarks()
    shares, indices = [], []
    for frame, yaw in enumerate(np.linspace(0, 70, 90)):
        engine.process(image, SimpleNamespace(landmarks5=points, pose=(float(yaw), 0, 0)), document, swapper, track_id=0,
                       frame_index=frame, timestamp=frame / 30)
        views = engine.last_diag["views"]
        assert sum(v["weight"] for v in views) == pytest.approx(1.0)
        shares.append({v["index"]: v["weight"] for v in views})
        indices.append(frozenset(shares[-1]))
    steps = [max(abs(b.get(i, 0) - a.get(i, 0)) for i in set(a) | set(b)) for a, b in zip(shares, shares[1:])]
    assert max(steps) <= 0.5                       # the blend moves through the hand-offs, never a one-frame swap
    assert sum(a != b for a, b in zip(indices, indices[1:])) <= 4
    assert 1 in indices[0] and indices[-1] == {2, 3}
