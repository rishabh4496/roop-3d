"""Synthetic scenes with known ground truth for the Stage 17 swap-core tests (no model is loaded)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.pipeline.photorealistic_swap_core import FacePacket
from roop_3d.reconstruction.expression_tps import BLOCKS


def landmarks106(cx: float, cy: float, width: float, height: float, *, jaw_width: float | None = None) -> np.ndarray:
    """A plausible 106-point face (``layout_is_plausible`` holds): contour is a half-ellipse through the chin."""
    jaw_width = width if jaw_width is None else jaw_width
    points = np.zeros((106, 2), np.float64)
    theta = np.linspace(np.pi, 0.0, 33)
    eye_y = cy - 0.10 * height
    points[BLOCKS["contour"]] = np.stack((cx + 0.5 * jaw_width * np.cos(theta), eye_y + 0.78 * height * np.sin(theta)), 1)
    for name, sign in (("eye_a", -1), ("eye_b", 1)):
        t = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        points[BLOCKS[name]] = np.stack((cx + sign * 0.20 * width + 0.07 * width * np.cos(t), eye_y + 0.03 * height * np.sin(t)), 1)
    for name, sign in (("brow_a", -1), ("brow_b", 1)):
        t = np.linspace(-1, 1, 9)
        points[BLOCKS[name]] = np.stack((cx + sign * 0.20 * width + 0.10 * width * t, eye_y - 0.14 * height - 0.02 * height * (1 - t ** 2)), 1)
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points[BLOCKS["mouth"]] = np.stack((cx + 0.15 * width * np.cos(t), cy + 0.36 * height + 0.04 * height * np.sin(t)), 1)
    points[BLOCKS["nose"]] = np.stack((cx + 0.04 * width * np.sin(np.linspace(0, 6, 15)), np.linspace(eye_y + 0.02 * height, cy + 0.22 * height, 15)), 1)
    return points


def five_points(points106: np.ndarray) -> np.ndarray:
    eyes_a = points106[BLOCKS["eye_a"]].mean(0)
    eyes_b = points106[BLOCKS["eye_b"]].mean(0)
    mouth = points106[BLOCKS["mouth"]]
    return np.array([eyes_a, eyes_b, points106[BLOCKS["nose"]][-1], mouth.min(0) * [1, 0] + mouth.mean(0) * [0, 1],
                     mouth.max(0) * [1, 0] + mouth.mean(0) * [0, 1]], np.float32)


def skin_texture(shape: tuple[int, int], base_bgr, seed: int, *, pore_sigma: float = 7.0) -> np.ndarray:
    """Skin colour + band-limited pore noise (the high-frequency detail the pipeline must not blur away)."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, pore_sigma * 2.2, shape).astype(np.float32), (0, 0), 0.9)
    out = np.empty((*shape, 3), np.float32)
    out[:] = np.asarray(base_bgr, np.float32)
    return np.clip(out + noise[:, :, None], 0, 255).astype(np.uint8)


@dataclass
class Scene:
    frame: np.ndarray
    background: np.ndarray            # the frame with no head: ground truth for the filled overhang
    packet: FacePacket
    target_head: np.ndarray           # bool mask of the original head
    swap_texture: np.ndarray          # the crop the packet carries
    source_view: np.ndarray


def make_scene(*, size: int = 480, target_jaw: float = 200.0, source_jaw: float = 150.0, light_left: float = 0.35,
               lit_source: bool = False, seed: int = 3) -> Scene:
    h = w = size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    background = np.stack((70 + xx / 8, 95 + yy / 9, 150 - xx / 12), -1).astype(np.float32)
    background += np.random.default_rng(seed).normal(0, 1.5, background.shape).astype(np.float32)
    background = np.clip(background, 0, 255).astype(np.uint8)

    cx, cy, fw, fh = w / 2, h / 2 + 10, 190.0, 200.0
    target = landmarks106(cx, cy, fw, fh, jaw_width=target_jaw)
    # The head is drawn from the SAME outline definition the core derives (jaw contour + forehead arc): the heuristic outline
    # is therefore the true outline here, which is the assumption step 1 makes about real footage and the report names.
    from roop_3d.pipeline.image_ops import head_polygon, polygon_mask
    head = polygon_mask(head_polygon(target, 0.42), (h, w)) > 0.5
    skin = skin_texture((h, w), (140, 165, 205), seed + 1, pore_sigma=2.0).astype(np.float32)
    shade = 1.0 + light_left * (-(xx - cx) / (0.5 * target_jaw))
    frame = background.astype(np.float32).copy()
    frame[head] = (skin * shade[:, :, None])[head]
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # the neural swap: a 128 px aligned crop of the SOURCE's skin (own pore detail), face-shaped coverage
    crop_size, scale = 128, 1.8
    crop = skin_texture((crop_size, crop_size), (150, 172, 214), seed + 2, pore_sigma=7.0)
    cyy, cxx = np.mgrid[:crop_size, :crop_size].astype(np.float32)
    coverage = (((cxx - 64) / 52.0) ** 2 + ((cyy - 66) / 58.0) ** 2 <= 1.0).astype(np.float32)
    coverage = cv2.GaussianBlur(coverage, (0, 0), 2.0)
    crop_to_frame = np.array([[scale, 0, cx - 64 * scale], [0, scale, cy - 64 * scale]], np.float64)

    # the FaceSet photograph of the source, in its own pixel space (narrower jaw)
    view_size = 256
    view_scale = 190.0 / 120.0
    view_pts = landmarks106(view_size / 2, view_size / 2 + 6, 120.0, 126.0, jaw_width=source_jaw / view_scale * 1.0)
    view = skin_texture((view_size, view_size), (146, 170, 212), seed + 4, pore_sigma=6.0)
    if lit_source:
        view = np.clip(view.astype(np.float32) * (1.0 + 0.4 * np.linspace(1, -1, view_size))[None, :, None], 0, 255).astype(np.uint8)

    packet = FacePacket(track_id=0, swapped_crop=crop, crop_to_frame=crop_to_frame, target_landmarks=target,
                        crop_landmarks5=five_points(target) @ np.array([[1 / scale, 0], [0, 1 / scale]]) - np.array([cx - 64 * scale, cy - 64 * scale]) / scale,
                        crop_coverage=coverage, source_landmarks=view_pts, source_view=view)
    return Scene(frame, background, packet, head, crop, view)
