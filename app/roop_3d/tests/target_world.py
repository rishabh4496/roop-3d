"""Scripted target performer for the Stage 4 reenactment tests (test-only; never imported by production code).

A performer is a synthetic 3D head (a DIFFERENT person from the source identity) whose 68 landmarks are moved by known
expression parameters and rigid pose, projected through a known camera, and laid out as the tracker's two landmark sets:
5 points and 106 points in the 2d106 block structure of ``expression_tps.BLOCKS``. Everything the engine is asked to
recover (pose, expression, timing) is therefore known exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import identity_world as W
from roop_3d.core.pose import rotation_from_euler

CANVAS = (400, 400)                  # (h, w)
SCALE = 1750.0                       # px per metre => inter-ocular distance ~ 110 px
FPS = 30.0


@dataclass
class Performer:
    subject: W.Subject
    swap_eye_blocks: bool = False    # a detector may order its eye/brow blocks either way round
    iod_m: float = field(init=False)

    def __post_init__(self):
        lm = self.subject.true_lm
        self.iod_m = float(np.linalg.norm(lm[36:42].mean(0) - lm[42:48].mean(0)))


def expression_3d(lm68: np.ndarray, iod: float, *, mouth_open=0.0, smile=0.0, blink_a=0.0, blink_b=0.0, brow=0.0,
                  mouth_width=0.0) -> np.ndarray:
    """Known expression displacement of the 68 3D landmarks (metres); parameters are in inter-ocular units."""
    p = lm68.copy()
    p[[57, 58, 59, 65, 66, 67], 1] += mouth_open * iod                      # lower lip (outer + inner) follows the jaw
    p[[6, 7, 8, 9, 10], 1] += mouth_open * iod * np.array([0.35, 0.7, 1.0, 0.7, 0.35])
    p[[48, 54], 1] -= smile * iod
    p[48, 0] -= mouth_width * iod / 2
    p[54, 0] += mouth_width * iod / 2
    for blink, upper, lower in ((blink_a, [37, 38], [40, 41]), (blink_b, [43, 44], [46, 47])):
        gap = abs(float(p[lower, 1].mean() - p[upper, 1].mean()))
        p[upper, 1] += blink * 0.5 * gap * 0.95
        p[lower, 1] -= blink * 0.5 * gap * 0.95
    p[17:27, 1] -= brow * iod
    return p


def _resample(points: np.ndarray, n: int, closed: bool = False) -> np.ndarray:
    pts = np.vstack([points, points[:1]]) if closed else points
    t = np.linspace(0, 1, len(pts))
    out_t = np.linspace(0, 1, n, endpoint=not closed)
    return np.column_stack([np.interp(out_t, t, pts[:, 0]), np.interp(out_t, t, pts[:, 1])])


def landmarks106(p2d: np.ndarray, swap_eyes: bool = False) -> np.ndarray:
    """68-point 2D landmarks -> the 106-point block layout (contour, eye_a, brow_a, mouth, nose, eye_b, brow_b)."""
    contour = _resample(p2d[0:17], 33)
    eye_r, eye_l = _resample(p2d[36:42], 10, True), _resample(p2d[42:48], 10, True)
    brow_r, brow_l = _resample(p2d[17:22], 9), _resample(p2d[22:27], 9)
    mouth = np.vstack([p2d[48:60], p2d[60:68]])
    nose = _resample(p2d[27:36], 15)
    a, b = (eye_l, eye_r) if swap_eyes else (eye_r, eye_l)
    ba, bb = (brow_l, brow_r) if swap_eyes else (brow_r, brow_l)
    return np.vstack([contour, a, ba, mouth, nose, b, bb])


def five_points(p2d: np.ndarray) -> np.ndarray:
    return np.vstack([p2d[36:42].mean(0), p2d[42:48].mean(0), p2d[30], p2d[48], p2d[54]])


def frame(performer: Performer, yaw=0.0, pitch=0.0, roll=0.0, *, t: float, noise_px=0.35, seed=0,
          sharpness: float | None = 1.0, scene_cut=False, **expression):
    """-> (TargetObservation kwargs, truth dict)."""
    s = performer.subject
    lm3d = expression_3d(s.true_lm, performer.iod_m, **expression)
    rot = rotation_from_euler(yaw, pitch, roll)
    centre = s.true_lm.mean(axis=0)
    p2d = SCALE * ((lm3d - centre) @ rot.T)[:, :2] + np.array([CANVAS[1] / 2, CANVAS[0] / 2])
    p2d = p2d + np.random.default_rng(seed).normal(0, noise_px, p2d.shape)
    return dict(landmarks5=five_points(p2d), landmarks106=landmarks106(p2d, performer.swap_eye_blocks), timestamp=t,
                scene_cut=scene_cut, sharpness=sharpness), dict(yaw=yaw, pitch=pitch, roll=roll, p2d=p2d, **expression)


def sequence(performer: Performer, script, *, noise_px=0.35, seed=0):
    """`script`: list of dicts (pose + expression + optional flags) -> list of (observation kwargs, truth)."""
    out = []
    for i, step in enumerate(script):
        step = dict(step)
        out.append(frame(performer, t=i / FPS, noise_px=noise_px, seed=seed + i, **step))
    return out
