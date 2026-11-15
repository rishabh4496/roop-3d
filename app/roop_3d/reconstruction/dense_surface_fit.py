"""Perspective bundle adjustment of measured dense landmarks on a volumetric head.

Only source observations constrain geometry. Compact-support 3D deformation leaves
unobserved cranium and ears at their MakeHuman prior; no image is warped.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import CameraIntrinsics
from roop_3d.reconstruction.pose_solver import solve_head_pose_6dof
from roop_3d.reconstruction.rbf_deform import deform_template_rbf


@dataclass(frozen=True)
class DenseSurfaceFit:
    vertices: np.ndarray
    cameras: dict
    diagnostics: dict


def fit_dense_surface(mesh, source_views, *, fov_degrees=45.0) -> DenseSurfaceFit:
    indices = np.asarray(mesh.landmark_vertices['68pt'])
    prior = mesh.vertices[indices].astype(np.float64)
    rows = []
    for view, image, _ in source_views:
        if view.landmarks_68 is None:
            continue
        observed = np.asarray(view.landmarks_68, np.float64)
        camera = CameraIntrinsics.from_fov(fov_degrees, image.shape[1], image.shape[0])
        pose = solve_head_pose_6dof(observed, intrinsics=camera, model_points=prior)
        # Pixel uncertainty scales with source resolution, not an absolute crop size.
        sigma = max(1.5, 0.008 * float(np.linalg.norm(np.ptp(observed, axis=0))))
        # The 17 jaw-contour landmarks lie on the occluding silhouette: in an oblique view they slide along the surface
        # and do not correspond to the fixed 3D vertex a frontal view observes. Their weight falls off with |yaw|.
        contour = float(np.clip(1.0 - abs(float(getattr(view, 'yaw', 0.0))) / 55.0, 0.08, 1.0))
        weights = np.ones(68)
        weights[:17] = contour
        rows.append((view, observed, camera, pose, sigma, weights))
    if len(rows) < 2:
        raise DomainError('DENSE_MULTIVIEW_REQUIRED', 'At least two measured dense source views are required', 422)
    n = len(prior)
    point_count = 3 * n
    initial = np.r_[np.zeros(point_count), np.concatenate([
        np.r_[cv2.Rodrigues(row[3].R)[0].ravel(), row[3].t] for row in rows])]
    flip = np.array((1., 1., -1.))
    pose_priors = initial[point_count:].reshape(-1, 6)

    def residual(values):
        points = (prior + values[:point_count].reshape(n, 3)) * flip
        errors = []
        for index, (view, observed, camera, _, sigma, weights) in enumerate(rows):
            pose = values[point_count + 6 * index:point_count + 6 * (index + 1)]
            projected = cv2.projectPoints(points, pose[:3], pose[3:], camera.matrix(), None)[0][:, 0]
            errors.append(((projected - observed) / sigma * np.sqrt(view.quality.score) * weights[:, None]).ravel())
        # Six millimetres of landmark variation is a prior, not observed depth.
        errors.append(values[:point_count] / 0.006)
        errors.append(((values[point_count:].reshape(-1, 6) - pose_priors)
                       / np.array((0.15, 0.15, 0.15, 0.025, 0.025, 0.04))).ravel())
        return np.concatenate(errors)

    residual_count = 2 * n * len(rows) + point_count + 6 * len(rows)
    sparsity = lil_matrix((residual_count, len(initial)), dtype=np.int8)
    for view in range(len(rows)):
        for point in range(n):
            start = 2 * (view * n + point)
            sparsity[start:start + 2, 3 * point:3 * point + 3] = 1
        sparsity[2 * view * n:2 * (view + 1) * n, point_count + 6 * view:point_count + 6 * (view + 1)] = 1
    offset = 2 * n * len(rows)
    for index in range(point_count):
        sparsity[offset + index, index] = 1
    offset += point_count
    for index in range(6 * len(rows)):
        sparsity[offset + index, point_count + index] = 1
    lower, upper = np.full(len(initial), -np.inf), np.full(len(initial), np.inf)
    lower[:point_count], upper[:point_count] = -0.018, 0.018
    for view in range(len(rows)):
        lower[point_count + 6 * view + 5] = 0.05
    fit = least_squares(residual, initial, bounds=(lower, upper), jac_sparsity=sparsity.tocsr(),
                        loss='soft_l1', f_scale=1.5, max_nfev=100, ftol=1e-5)
    if not np.isfinite(fit.x).all():
        raise DomainError('DENSE_SURFACE_FIT_INVALID', 'Dense multiview surface fit returned nonfinite geometry', 422)
    controls = prior + fit.x[:point_count].reshape(n, 3)
    vertices, support, strength = deform_template_rbf(mesh, prior, controls, np.ones(n),
        control_vertices=indices, support=0.05, ridge=0.005, max_displacement_m=0.018, smooth_iterations=8)
    cameras = {}
    reprojection = []
    for index, (view, observed, camera, _, _, _) in enumerate(rows):
        params = fit.x[point_count + 6 * index:point_count + 6 * (index + 1)]
        rotation = cv2.Rodrigues(params[:3])[0]
        projected = cv2.projectPoints(vertices[indices].astype(np.float64) * flip,
                                      params[:3], params[3:], camera.matrix(), None)[0][:, 0]
        rms = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))
        reprojection.append({'view': int(view.index), 'rms_px': rms})
        cameras[int(view.index)] = (rotation, params[3:].copy(), rms)
    return DenseSurfaceFit(vertices.astype(np.float32), cameras,
        {'method': 'perspective_dense68_bundle_adjustment_and_compact_3d_rbf',
         'views': len(rows), 'converged': bool(fit.success), 'cost': float(fit.cost),
         'deformation_strength': float(strength), 'max_control_delta_mm': float(np.linalg.norm(controls-prior,axis=1).max()*1000),
         'per_view': reprojection, 'unconstrained_geometry': 'MakeHuman prior'})
