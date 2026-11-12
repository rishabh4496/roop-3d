"""Hair, neck, cloth and background texels of a source photograph are never conditioning appearance."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import aligned_crop, template_points
from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
from roop_3d.reconstruction.neural_face import NeuralFaceEngine

HAIR, SKIN, CLOTH, BACKGROUND = 17, 1, 16, 0


class Parser:
    """Skin everywhere except a hair block (upper-left), a collar strip (bottom) and the background corner."""
    def __init__(self):
        self.released = 0

    def predict(self, image):
        h, w = image.shape[:2]
        labels = np.full((h, w), SKIN)
        labels[: h // 4, : w // 2] = HAIR
        labels[-h // 6:, :] = CLOTH
        labels[:, -w // 8:] = BACKGROUND
        probabilities = np.zeros((h, w, 19), np.float32)
        np.put_along_axis(probabilities, labels[..., None], 1.0, axis=-1)
        return probabilities

    def release(self):
        self.released += 1


def build(parser):
    vertices = np.array([[.05, .05, 1], [.95, .05, 1], [.05, .95, 1], [.95, .95, 1]], np.float32)

    class Backbone:
        topology = SimpleNamespace(faces=np.array([[0, 2, 1], [1, 2, 3]], np.int32))

        def fit(self, image, points):
            return SimpleNamespace(camera_vertices=vertices, intrinsics=np.diag([224, 224, 1]), rotation=np.eye(3),
                                   translation=np.zeros(3, np.float32), scale=1.0, identity=vertices.copy(),
                                   vertices=vertices.copy(), expression_vertices=vertices.copy(),
                                   alignment=aligned_crop(image, np.asarray(points), 128))
    image = np.full((200, 200, 3), 90, np.uint8)
    view = SimpleNamespace(index=0, image_bytes=cv2.imencode(".png", image)[1].tobytes(), landmarks=template_points(128),
                           is_synthetic=False, landmarks_trusted=True, yaw=0, pitch=0, quality=SimpleNamespace(score=1))
    document = SimpleNamespace(faceset_id="f", views=(view,))
    engine = NeuralFaceEngine(Backbone(), ReferenceSurfaceRasterizer(), SimpleNamespace(), parser=parser)
    engine.prepare(document)
    return engine, document


def test_non_skin_texels_are_unobserved_and_skin_stays_observed():
    parser = Parser()
    engine, _ = build(parser)
    observed = engine.sources["f"][0].observed
    side = observed.shape[0]
    assert (observed[: side // 4 - 8, : side // 2 - 8] == 0).all()                 # hair block (dilated)
    assert (observed[-side // 6 + 8:, :] == 0).all()                               # collar strip
    assert (observed[:, -side // 8 + 8:] == 0).all()                               # background
    core = observed[side // 3: 2 * side // 3, side // 3: 2 * side // 3]
    assert core.min() == 1.0                                                       # genuine skin keeps full support
    assert parser.released == 1                                                    # parsing is a one-off; no resident model


def test_rejection_is_dilated_by_a_few_pixels_only():
    engine, _ = build(Parser())
    observed = engine.sources["f"][0].observed
    side = observed.shape[0]
    edge_row = side // 4                      # the hair block's lower boundary, away from the other regions
    column = side // 4
    assert observed[edge_row + 2, column] == 0                  # inside the dilation margin
    assert observed[edge_row + 14, column] == 1.0               # clear of it


def test_without_a_parser_support_is_the_photograph_bounds_as_before():
    engine, _ = build(None)
    observed = engine.sources["f"][0].observed
    assert observed.min() >= 0 and observed[observed.shape[0] // 2, observed.shape[1] // 2] == 1.0


def test_missing_parser_model_is_reported_not_hidden():
    class Missing:
        def predict(self, image):
            raise DomainError("MODEL_MISSING", "bisenet not installed", 503)

        def release(self):
            pass
    engine, _ = build(Missing())
    assert engine.parser_unavailable.startswith("MODEL_MISSING")
    assert engine.sources["f"][0].observed.max() == 1.0
