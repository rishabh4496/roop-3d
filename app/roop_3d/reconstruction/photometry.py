"""Linear-light, second-order RGB irradiance fitting for the full-head renderer.

Nine coefficients means three SH bands (orders 0..2), not nine bands. A single
photograph cannot uniquely separate reflectance and illumination: the default
fit uses a 0.5 diffuse-reflectance prior. Albedo retains spatial identity detail;
it is an intrinsic-image estimate, not measured reflectometry.
Normals use +x right, +y down, +z towards the viewer. Images here are BGR.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.relighting_3d import SphericalHarmonicsLighting


def linearize(image):
    value = np.asarray(image, np.float32) / 255
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def fit_sh(normals, linear, mask, *, albedo=None):
    """Bounded robust weighted solve; occluded/non-skin samples have zero weight.

    Fitting is a small host-side calibration, shading remains on the GPU. No
    full-resolution HxWx9 design matrix or GPU model arena is allocated.
    """
    normals, linear = np.asarray(normals), np.asarray(linear)
    weight = np.asarray(mask, np.float32)
    valid = ((weight > .05) & np.isfinite(normals).all(-1) & np.isfinite(linear).all(-1)
             & (np.linalg.norm(normals, axis=-1) > .5))
    indices = np.flatnonzero(valid)[::max(1, int(np.ceil(valid.sum() / 8192)))]
    neutral = np.zeros((9, 3), np.float32)
    neutral[0] = 1 / .886227
    if len(indices) < 32:
        return neutral
    n = normals.reshape(-1, 3)[indices]
    b = SphericalHarmonicsLighting.evaluate_basis(n).astype(np.float64)
    pixels = linear.reshape(-1, 3)[indices]
    reflectance = (.5 if albedo is None else np.maximum(
        np.broadcast_to(np.asarray(albedo), linear.shape).reshape(-1, 3)[indices], .04))
    y = pixels / reflectance
    base = np.clip(weight.ravel()[indices], 0, 1)
    w = base.copy()
    # Preserve the DC term; penalize high orders more than the key direction.
    ridge = np.diag([1e-7, 1e-5, 1e-5, 1e-5, .02, .02, .02, .02, .02]) * len(b)
    for _ in range(4):
        coeffs = np.linalg.solve(b.T @ (b * w[:, None]) + ridge, b.T @ (y * w[:, None]))
        residual = np.linalg.norm(y - b @ coeffs, axis=1)
        scale = max(float(np.median(residual)) * 1.4826, .01)
        w = base * np.minimum(1, 1.5 * scale / np.maximum(residual, 1e-6))
    return coeffs.astype(np.float32)


@dataclass(frozen=True)
class AlbedoDecomposition:
    albedo: np.ndarray                 # linear BGR reflectance
    sh_source: np.ndarray              # (9, 3), object coordinates


def decompose_albedo(texture, uv_normals, confidence):
    linear = linearize(texture)
    # Fix the intrinsic color ambiguity with a neutral-source-illuminant prior:
    # retain the source's robust skin chromaticity in reflectance. A gray prior
    # would incorrectly bleach the identity's skin under neutral target light.
    samples = linear[np.asarray(confidence) > .05]
    color = np.percentile(samples, 70, axis=0) if len(samples) else np.ones(3)
    prior = np.clip(color * (.5 / max(float(color @ np.array([.114,.587,.299])), .01)), .04, .95)
    sh = fit_sh(uv_normals, linear, confidence, albedo=prior)
    shading = SphericalHarmonicsLighting().evaluate_shading(uv_normals, sh)
    return AlbedoDecomposition(np.clip(linear / np.maximum(shading, .04), 0, 1).astype(np.float32), sh)


def extend_neck(vertices, faces, uvs, *, distance=.008, reference_vertices=None):
    """Extrude only the open inferior boundary, at most 8 mm, with a feather.

    Closed templates are left alone. UVs are inherited from the submental edge;
    this is a transition ribbon, not an invented textured neck.
    """
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    reference = vertices if reference_vertices is None else reference_vertices
    bottom = reference[:, 1].max()
    boundary = unique[(count == 1) & (reference[unique, 1].min(axis=1) > bottom - .018)]
    feather = np.ones(len(vertices), np.float32)
    if not len(boundary):
        return vertices, faces, uvs, feather
    ids = np.unique(boundary)
    lookup = dict(zip(ids, range(len(vertices), len(vertices) + len(ids))))
    extra = vertices[ids].copy()
    extra[:, 1] += distance
    extra[:, 2] -= distance * .25
    triangles = [[a, b, lookup[b]] for a, b in boundary] + [[a, lookup[b], lookup[a]] for a, b in boundary]
    return (np.vstack((vertices, extra)), np.vstack((faces, triangles)), np.vstack((uvs, uvs[ids])),
            np.r_[feather, np.zeros(len(ids), np.float32)])


@dataclass(frozen=True)
class HeadPhotometry:
    """Per-frame snapshot; never reads mutable rasterizer.last_buffers later."""
    normals: np.ndarray
    rest_normals: np.ndarray
    target_normals: np.ndarray
    depth: np.ndarray
    target_support: np.ndarray
    sh_source: np.ndarray
    inverse: np.ndarray
    camera: np.ndarray
    detail: object = None

    def lighting_mask(self, alpha):
        if self.detail is None:
            return alpha
        excluded = self.detail.masks(alpha.shape[0])[2]
        radius = max(1, round(alpha.shape[0] * .003))
        excluded = cv2.dilate(excluded, np.ones((radius*2+1, radius*2+1), np.uint8))
        protected = cv2.GaussianBlur(excluded, (0,0), max(.5, radius*.5))
        return alpha * (1 - np.maximum(excluded, protected))

    def apply(self, image, target, skin, alpha, options):
        from roop_3d.reconstruction.photometric_gpu import relight_head
        return relight_head(self, image, target, skin, alpha, options)


def fit_target_surface(vertices, indices, points, pose):
    """Fit measured target XY onto the head's depth prior, using smooth RBFs.

    Monocular landmarks constrain projected shape, not metric target depth. The
    latter retains the MakeHuman prior. This is independent of the source matte.
    """
    camera = pose.camera_vertices(vertices)
    z = camera[indices, 2]
    desired = np.column_stack(((points[:, 0] - pose.camera[0, 2]) * z / pose.camera[0, 0],
                               (points[:, 1] - pose.camera[1, 2]) * z / pose.camera[1, 1], z))
    _, first = np.unique(indices, return_index=True)
    sites = vertices[indices[first]]
    delta = (desired - camera[indices]) @ pose.rotation / pose.scale * (1, 1, -1)
    # Gaussian RBF ridge solve avoids an additional runtime dependency. Each
    # block is bounded even for a subdivided high-resolution head.
    def kernel(a, b):
        return np.exp(-np.sum((a[:, None] - b[None]) ** 2, axis=-1) / (2 * .035**2))
    weights = np.linalg.solve(kernel(sites, sites) + np.eye(len(sites)) * .03, delta[first])
    fitted = np.vstack([kernel(block, sites) @ weights for block in np.array_split(vertices, max(1, len(vertices)//4096))])
    return vertices + np.clip(fitted, -.015, .015)
