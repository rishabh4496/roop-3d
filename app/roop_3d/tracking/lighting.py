"""Per-frame second-order spherical-harmonics lighting L_t, fitted on a generic head under the tracked camera.

Method (the photometric_optimization recipe reduced to what is measurable from landmarks alone): the canonical head
template is posed with the tracked weak-perspective camera, the frame is sampled at the projected vertices that face the
camera and are skin (eyes, brows and lips are excluded), and ``photometry.fit_sh`` solves the nine RGB coefficients by
robust weighted least squares in linear light with the 0.5 diffuse-reflectance prior.

Limits that follow from the method, and that the estimate reports instead of hiding:
  * the geometry is the generic template, so normals are only right to the extent the actor's head resembles it;
  * there is no cast-shadow or self-occlusion model (the nose casts none here);
  * a single image cannot separate albedo from illumination, hence the reflectance prior.
Coefficients are in the CAMERA frame (+x right, +y down, +z toward the viewer, BGR order): independent of head pose, which is
what a compositor needs to relight a pasted face. ``key_direction`` points toward the dominant light in that frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from roop_3d.reconstruction.geometry_types import WeakPerspectiveCamera
from roop_3d.reconstruction.multiview_fit import identity_template
from roop_3d.reconstruction.photometry import fit_sh, linearize
from roop_3d.relighting_3d import SphericalHarmonicsLighting

SKIN_EXCLUDE_LANDMARKS = tuple(range(17, 27)) + tuple(range(36, 48)) + tuple(range(48, 68))   # brows, eyes, mouth
SKIN_EXCLUDE_RADIUS_M = 0.012
MIN_FACING_Z = 0.3                 # a vertex whose camera-frame normal is nearer to edge-on than this is not sampled
MIN_SAMPLES = 150
REFLECTANCE_PRIOR = 0.5            # fit_sh's default diffuse albedo


@dataclass(frozen=True)
class LightingEstimate:
    coefficients: np.ndarray       # (9, 3) BGR, camera frame, shading for REFLECTANCE_PRIOR
    key_direction: np.ndarray      # (3,) unit vector toward the dominant light
    samples: int
    residual: float                # RMS of the fit in linear light (0..1 scale)


@lru_cache(maxsize=1)
def _skin_template(subdivisions: int = 2):
    mesh = identity_template(subdivisions)
    normals = np.asarray(mesh.normals, np.float64)
    vertices = np.asarray(mesh.vertices, np.float64)
    excluded = vertices[list(SKIN_EXCLUDE_LANDMARKS)]
    distance = np.linalg.norm(vertices[:, None, :] - excluded[None], axis=2).min(axis=1)
    return vertices, normals, distance > SKIN_EXCLUDE_RADIUS_M


class MeshSHLightingEstimator:
    def __init__(self, subdivisions: int = 2, min_samples: int = MIN_SAMPLES):
        self.vertices, self.normals, self.skin = _skin_template(subdivisions)
        self.min_samples = int(min_samples)
        self._sh = SphericalHarmonicsLighting()

    def estimate(self, image_bgr: np.ndarray, camera: WeakPerspectiveCamera) -> LightingEstimate | None:
        """None when too few skin vertices face the camera inside the frame (an extreme profile, a face at the border)."""
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Lighting needs a BGR uint8 frame")
        height, width = image.shape[:2]
        pixels = camera.project(self.vertices)
        normals = camera.rotate_normals(self.normals)
        usable = (self.skin & (normals[:, 2] > MIN_FACING_Z) & (pixels[:, 0] >= 1) & (pixels[:, 0] <= width - 2)
                  & (pixels[:, 1] >= 1) & (pixels[:, 1] <= height - 2))
        if int(usable.sum()) < self.min_samples:
            return None
        points = pixels[usable].astype(np.float32)
        sampled = cv2.remap(image, points[:, :1].reshape(1, -1), points[:, 1:].reshape(1, -1), cv2.INTER_LINEAR)[0]
        linear, facing = linearize(sampled), normals[usable]
        coefficients = fit_sh(facing, linear, np.ones(len(facing), np.float32))
        predicted = np.maximum(self._sh.evaluate_basis(facing) @ coefficients, 0.0) * REFLECTANCE_PRIOR
        residual = float(np.sqrt(np.mean((predicted - linear) ** 2)))
        return LightingEstimate(coefficients.astype(np.float32), self._sh.extract_key_light_direction(coefficients),
                                int(len(facing)), residual)


def angle_between_deg(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))))
