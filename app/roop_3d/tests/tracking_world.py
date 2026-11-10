"""Test-only helpers for the head-tracking tests (never imported by production code).

Builds on ``target_world`` (a scripted performer with KNOWN pose and expression, projected through a known camera) and
``identity_world`` (a synthetic head that differs from the canonical template). The tracker under test sees only landmarks,
timestamps and pixels; the scripts are used only by the assertions.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402
import target_world as TW  # noqa: E402

from roop_3d.reconstruction.multiview_fit import batch_rotations  # noqa: E402
from roop_3d.relighting_3d import SphericalHarmonicsLighting  # noqa: E402
from roop_3d.tracking.video_tracker import FrameEvidence  # noqa: E402

FPS = TW.FPS


def performer(seed: int = 202, **geometry) -> TW.Performer:
    """A head that differs from the canonical template (default: a narrower face, flatter nose, shorter jaw)."""
    geometry = {"width": -0.08, "nose_depth": 0.004, "jaw_drop": -0.004, **geometry}
    return TW.Performer(W.make_subject(f"track{seed}", W.LAYOUT_68, seed, **geometry))


def evidence(obs: dict, **extra) -> FrameEvidence:
    extra.setdefault("scene_cut", obs.get("scene_cut", False))
    return FrameEvidence(timestamp=obs["timestamp"], landmarks5=obs["landmarks5"], landmarks106=obs["landmarks106"], **extra)


def moving_script(n: int = 150, *, yaw=35.0, pitch=10.0, roll=4.0, mouth=0.25) -> list[dict]:
    """A smooth performance: head sweep, nod and tilt, a talking mouth and a slow smile."""
    steps = []
    for i in range(n):
        t = i / FPS
        steps.append(dict(yaw=yaw * np.sin(2 * np.pi * t / 4.0), pitch=pitch * np.sin(2 * np.pi * t / 3.0 + 1),
                          roll=roll * np.sin(2 * np.pi * t / 2.5), mouth_open=mouth * max(0.0, np.sin(2 * np.pi * t / 1.5)),
                          smile=0.05 * np.sin(t)))
    return steps


def frontal_script(n: int, **expression) -> list[dict]:
    """Slight natural head motion around frontal (inside the identity lock's pose window)."""
    return [dict(yaw=6 * np.sin(i / 11.0), pitch=3 * np.sin(i / 7.0), roll=1.5 * np.sin(i / 9.0), **expression) for i in range(n)]


def run(tracker, performer_, script, *, noise=0.35, seed=0, start=0, **extra):
    """-> list of (TrackedHeadFrame, truth). Timestamps continue from ``start`` frames in."""
    out = []
    for i, step in enumerate(script):
        obs, truth = TW.frame(performer_, t=(start + i) / FPS, noise_px=noise, seed=seed + start + i, **step)
        out.append((tracker.process(evidence(obs, **extra)), truth))
    return out


def pose_errors(out, skip: int = 20) -> np.ndarray:
    """(N, 3) absolute yaw/pitch/roll errors in degrees of the tracked frames after a warm-up."""
    return np.array([[abs(f.camera.yaw_deg - t["yaw"]), abs(f.camera.pitch_deg - t["pitch"]), abs(f.camera.roll_deg - t["roll"])]
                     for f, t in out[skip:] if f.status == "tracked"])


# --------------------------------------------------------------------------- lit renders (for the lighting tests)
SH = SphericalHarmonicsLighting()
LIT_SCALE, LIT_CENTRE, LIT_SHAPE = 2000.0, (300.0, 300.0), (600, 600)


def sh_environment(direction, ambient: float = 0.35, intensity: float = 0.9) -> np.ndarray:
    d = np.asarray(direction, np.float64)
    return SH.create_directional_lighting(d / np.linalg.norm(d), ambient=ambient, intensity=intensity)


def lit_frame(subject: W.Subject, yaw: float, pitch: float, coefficients: np.ndarray, *, roll: float = 0.0):
    """Render the subject's head under a KNOWN SH environment (per-triangle flat shading, albedo 0.5, linear light -> sRGB)
    and return (BGR image, landmarks5, landmarks106) from the very same camera."""
    rot = batch_rotations(np.array([[yaw, pitch, roll]]))[0]
    centre_3d = subject.template_lm.mean(axis=0)
    mesh = subject.mesh
    vertices = mesh.vertices.astype(np.float64)
    cam = (vertices - centre_3d) @ rot.T
    xy = LIT_SCALE * cam[:, :2] + np.array(LIT_CENTRE)
    faces = mesh.faces
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    camera_normal = normal @ rot.T
    shade = np.maximum(SH.evaluate_basis(camera_normal) @ coefficients, 0.0) * 0.5
    srgb = np.where(shade <= 0.0031308, shade * 12.92, 1.055 * np.clip(shade, 0, None) ** (1 / 2.4) - 0.055)
    colours = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    image = np.full(LIT_SHAPE + (3,), 40, np.uint8)
    visible = np.where(camera_normal[:, 2] > 0.0)[0]
    for k in visible[np.argsort(cam[faces[visible]][:, :, 2].mean(axis=1))]:
        cv2.fillConvexPoly(image, np.round(xy[faces[k]] * 16).astype(np.int32), tuple(int(c) for c in colours[k]), cv2.LINE_8, 4)
    p2d = LIT_SCALE * ((subject.true_lm - centre_3d) @ rot.T)[:, :2] + np.array(LIT_CENTRE)
    return image, TW.five_points(p2d), TW.landmarks106(p2d)


def flat_image(value: int, noise: float = 2.0, seed: int = 0, shape=(120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(value, noise, shape + (3,)), 0, 255).astype(np.uint8)
