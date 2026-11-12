"""Replicated borders (no black wedges) and alias-free photograph caching."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.engine.volumetric.preprocess import aligned_crop, template_points
from roop_3d.reconstruction.neural_face import NeuralFaceEngine


def test_face_at_the_frame_edge_gets_replicated_not_black_borders():
    image = np.full((200, 200, 3), 180, np.uint8)
    points = template_points(128) * 0.9 + (-15, 40)          # most of the aligned window falls outside the frame
    crop = aligned_crop(image, points, 128).image
    assert crop.min() >= 175                                  # BORDER_CONSTANT left ~35 % of the crop black


def test_backbone_input_crops_replicate_the_border(monkeypatch):
    from roop_3d.reconstruction.morphable_backbone import FaceTopology, MorphableBackbone
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    topology = FaceTopology("flame2023open", 4, np.arange(4), np.array([[0, 2, 1], [1, 2, 3]], np.int32), uv)
    seen = []
    base = np.zeros((1, 4, 3), np.float32)

    class Graph:
        def __init__(self, result):
            self.result = result

        def run(self, values):
            seen.extend(v for k, v in values.items() if k in ("rgb", "rgb_hd"))
            return self.result
    coarse = Graph({"identity_vertices_m": base, "rotation": np.eye(3)[None], "translation_m": np.array([[0, 0, 1]]),
                    "scale": np.ones((1, 1)), "intrinsics": np.diag([224, 224, 1])[None]})
    expression = Graph({"coefficients": np.zeros((1, 50)), "expressed_vertices_m": base})
    detail = Graph({"mid_offsets_m": base, "displacement_uv_m": np.zeros((1, 1, 65, 65), np.float32)})
    backbone = MorphableBackbone(topology, coarse, expression, detail, expression_basis="smirk_flame50", evidence={})
    image = np.full((200, 200, 3), 180, np.uint8)
    backbone.fit(image, template_points(128) * 0.9 + (-15, 40))
    assert len(seen) >= 2 and all(blob.min() > 0.65 for blob in seen)     # 180/255 = .706; zero fill would give 0


def test_photograph_cache_averages_a_shrink_instead_of_aliasing_and_keeps_geometry():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    matrix = np.array(((0.25, 0, 0), (0, 0.25, 0)))
    cache = NeuralFaceEngine._photograph_cache(noise, matrix, 256)
    area = cv2.resize(noise, (256, 256), interpolation=cv2.INTER_AREA)
    aliased = cv2.warpAffine(noise, matrix, (256, 256), flags=cv2.INTER_LINEAR)
    # white noise averaged over a 4x4 footprint keeps 1/4 of its deviation; aliasing keeps all of it
    assert 0.5 * area.std() < cache.std() < 1.1 * area.std()
    assert aliased.std() > 3 * cache.std()
    # a horizontal ramp keeps its geometry through the pre-shrink (including a translation and the half-pixel offset)
    ramp = np.tile(np.arange(1024, dtype=np.float32), (1024, 1))
    image = np.repeat(ramp[:, :, None], 3, 2) / 4
    shifted = np.array(((0.3, 0, 20.0), (0, 0.3, 5.0)))
    cache = NeuralFaceEngine._photograph_cache(image.astype(np.float32), shifted, 256)
    x = np.arange(60, 200)
    expected = (x - 20.0) / 0.3 / 4
    assert np.abs(cache[100, x, 0] - expected).max() < 0.35


def test_upscaling_uses_the_original_pixels_directly():
    image = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    matrix = np.array(((2.0, 0, 0), (0, 2.0, 0)))
    cache = NeuralFaceEngine._photograph_cache(image, matrix, 128)
    expected = cv2.warpAffine(image, matrix, (128, 128), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(cache, expected)
