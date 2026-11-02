"""Multi-view, identity-specific landmark geometry (Stage 3, Phase B).

What this does, precisely
-------------------------
Given N observed 2D landmark sets of ONE person (5-point SCRFD landmarks, or 68-point sets when a FaceSet
carries them) and a coarse head pose per view, it solves a MAP bundle adjustment under a scaled-orthographic
camera (the projection family already used by ``core.pose.estimate_pose_5pt``):

    unknowns : identity landmark displacement d_j (3-vector per landmark, from the anthropometric template),
               and per view yaw / pitch / roll / scale / translation
    data     : (s_i * (R_i (T_j + d_j))_xy + t_i - x_ij) / (s_i * sigma_obs)   (Huber-reweighted)
    priors   : d_xy / sigma_xy, d_z / sigma_z, (pose - stored pose) / sigma_pose

The Gauss-Newton Hessian at the optimum gives a real posterior covariance, so the per-landmark depth
confidence is MEASURED: with one frontal view the depth rows of the Jacobian are ~0, the posterior stays at the
prior and the landmark's z confidence is ~0. Nothing is "recovered" that the views do not constrain.

What this is not
----------------
It is not FLAME and not a dense reconstruction. The fitted quantities are the identity-specific 3D positions of
the observed landmarks; the rest of the head surface is a smooth template deformation (see ``identity_asset``)
whose per-vertex confidence says how far each vertex is from an observed landmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import estimate_pose_5pt
from roop_3d.reconstruction.flame_fitter import huber_weights
from roop_3d.reconstruction.geometry_types import Mesh3D, WeakPerspectiveCamera
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh

LAYOUT_5 = "5pt"
LAYOUT_68 = "68pt"
LAYOUT_SIZES = {LAYOUT_5: 5, LAYOUT_68: 68}

# Noise / prior scales (metres, degrees). They define the MAP trade-off and therefore the confidence numbers.
SIGMA_OBS_M = 0.0020          # landmark detector noise expressed in metres at the face
SIGMA_PRIOR_XY_M = 0.012      # plausible identity variation of a landmark position in the image plane
SIGMA_PRIOR_Z_M = 0.015       # ... and in depth
SIGMA_POSE_DEG = 8.0          # accuracy of the stored (5-point derived) pose
HUBER_DELTA = 3.0             # in units of sigma_obs
YAW_SPAN_FOR_DEPTH_DEG = 25.0


# --------------------------------------------------------------------------- template
def _subdivide(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One midpoint subdivision. Original vertex indices are preserved (new vertices are appended)."""
    verts, tex = [v for v in vertices], [u for u in uvs]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            verts.append((verts[a] + verts[b]) / 2.0)
            tex.append((tex[a] + tex[b]) / 2.0)
            cache[key] = len(verts) - 1
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(int(a), int(b)), midpoint(int(b), int(c)), midpoint(int(c), int(a))
        new_faces += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.asarray(verts, np.float64), np.asarray(new_faces, np.int32), np.asarray(tex, np.float64)


@lru_cache(maxsize=4)
def _template_arrays(subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = build_canonical_head_mesh(landmark_count=68)
    verts, faces, uvs = base.vertices.astype(np.float64), base.faces.astype(np.int32), base.uvs.astype(np.float64)
    for _ in range(subdivisions):
        verts, faces, uvs = _subdivide(verts, faces, uvs)
    return verts, faces, uvs


def identity_template(subdivisions: int = 2) -> Mesh3D:
    """The anthropometric facial template (the prior), subdivided for texture resolution.

    Indices 0..67 stay the 68 canonical landmark vertices, whatever the subdivision level.
    """
    if not 0 <= subdivisions <= 3:
        raise DomainError("INVALID_SUBDIVISION", f"subdivisions must be in [0, 3], got {subdivisions}", 422)
    verts, faces, uvs = _template_arrays(subdivisions)
    return Mesh3D(verts.copy(), faces.copy(), uvs=uvs.copy())


def template_landmarks(layout: str, mesh: Mesh3D) -> np.ndarray:
    """Template positions of the landmarks of `layout` (L, 3), in mesh coordinates."""
    v = mesh.vertices.astype(np.float64)
    if layout == LAYOUT_68:
        return v[:68].copy()
    if layout == LAYOUT_5:  # image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner
        return np.vstack([v[36:42].mean(axis=0), v[42:48].mean(axis=0), v[30], v[48], v[54]])
    raise DomainError("UNSUPPORTED_LANDMARK_LAYOUT", f"Unknown landmark layout {layout!r}", 422)


def landmark_normals(layout: str, mesh: Mesh3D) -> np.ndarray:
    """Outward template normals at the landmarks (used to down-weight self-occluded landmarks)."""
    n = mesh.normals.astype(np.float64)
    if layout == LAYOUT_68:
        return n[:68].copy()
    pts = template_landmarks(layout, mesh)
    nearest = [int(np.argmin(np.linalg.norm(mesh.vertices - p, axis=1))) for p in pts]
    return n[nearest].copy()


def layout_of(points: np.ndarray) -> str:
    count = int(np.asarray(points).shape[0])
    for name, size in LAYOUT_SIZES.items():
        if size == count:
            return name
    raise DomainError("UNSUPPORTED_LANDMARK_LAYOUT",
                      f"Identity fitting supports 5-point and 68-point landmarks, got {count}", 422)


def five_from_68(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64)
    return np.vstack([p[36:42].mean(axis=0), p[42:48].mean(axis=0), p[30], p[48], p[54]])


# --------------------------------------------------------------------------- math helpers
def batch_rotations(angles_deg: np.ndarray) -> np.ndarray:
    """(V, 3) yaw/pitch/roll in degrees -> (V, 3, 3). Identical to ``core.pose.rotation_from_euler``."""
    a = np.radians(np.asarray(angles_deg, np.float64).reshape(-1, 3))
    cy, sy, cp, sp, cr, sr = np.cos(a[:, 0]), np.sin(a[:, 0]), np.cos(a[:, 1]), np.sin(a[:, 1]), np.cos(a[:, 2]), np.sin(a[:, 2])
    n = len(a)
    ry, rx, rz = (np.zeros((n, 3, 3)) for _ in range(3))
    ry[:, 0, 0], ry[:, 0, 2], ry[:, 1, 1], ry[:, 2, 0], ry[:, 2, 2] = cy, sy, 1.0, -sy, cy
    rx[:, 0, 0], rx[:, 1, 1], rx[:, 1, 2], rx[:, 2, 1], rx[:, 2, 2] = 1.0, cp, -sp, sp, cp
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = cr, -sr, sr, cr, 1.0
    return rz @ rx @ ry


@dataclass(frozen=True)
class ViewLandmarks:
    """One usable observation: 2D landmarks (pixels), a coarse standard-frame pose, and a 0..1 weight."""
    view_index: int
    points: np.ndarray                      # (L, 2) image pixels
    pose_deg: tuple[float, float, float]    # yaw, pitch (+ = up), roll
    quality: float = 1.0


@dataclass
class ViewFit:
    view_index: int
    yaw: float
    pitch: float
    roll: float
    scale: float                 # pixels per metre
    tx: float
    ty: float
    rms_mm: float                # in-sample landmark reprojection error, millimetres at the face
    holdout_mm: float | None     # error when this view is NOT used to fit the geometry
    inlier_fraction: float
    consistent: bool = True


@dataclass
class IdentityShapeFit:
    layout: str
    template_landmarks: np.ndarray
    model_center: np.ndarray             # centroid of the template landmarks; the fit rotates about this point
    landmarks_3d: np.ndarray
    displacement: np.ndarray
    sigma_post: np.ndarray               # (L, 3) posterior std-dev, metres
    confidence_xy: np.ndarray            # (L,) 0..1
    confidence_z: np.ndarray             # (L,) 0..1
    landmark_confidence: np.ndarray      # (L,) min(xy, z)
    views: list[ViewFit]
    rms_mm: float
    holdout_rms_mm: float | None
    yaw_span_deg: float
    level: str                           # single_view_prior | limited_multiview | multiview
    converged: bool
    iterations: int
    notes: list[str] = field(default_factory=list)

    @property
    def depth_observed_fraction(self) -> float:
        return float(np.mean(self.confidence_z >= 0.5))

    def camera_for(self, view: ViewFit) -> WeakPerspectiveCamera:
        """The fitted view as a camera acting on MESH coordinates (the fit works about `model_center`)."""
        rot = batch_rotations(np.array([[view.yaw, view.pitch, view.roll]]))[0]
        shift = view.scale * (rot @ self.model_center)[:2]
        return WeakPerspectiveCamera(rotation=tuple(tuple(float(x) for x in row) for row in rot), scale=view.scale,
                                     tx=float(view.tx - shift[0]), ty=float(view.ty - shift[1]))


# --------------------------------------------------------------------------- Levenberg-Marquardt
def _numeric_jacobian(fun, x: np.ndarray, steps: np.ndarray, r0: np.ndarray) -> np.ndarray:
    jac = np.empty((r0.size, x.size))
    for k in range(x.size):
        h = steps[k]
        xp = x.copy()
        xp[k] += h
        jac[:, k] = (fun(xp) - r0) / h
    return jac


def _levenberg_marquardt(fun, x0: np.ndarray, steps: np.ndarray, max_iter: int = 40, tol: float = 1e-9,
                         jac_fun=None):
    x = x0.copy()
    r = fun(x)
    cost = float(r @ r)
    lam, converged, it = 1e-3, False, 0
    for it in range(1, max_iter + 1):
        jac = jac_fun(x) if jac_fun is not None else _numeric_jacobian(fun, x, steps, r)
        jtj, grad = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(12):
            try:
                delta = np.linalg.solve(jtj + lam * np.diag(np.maximum(np.diag(jtj), 1e-12)), -grad)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            r_new = fun(x + delta)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                gain = cost - cost_new
                x, r, cost, lam, improved = x + delta, r_new, cost_new, max(lam / 3.0, 1e-9), True
                break
            lam *= 4.0
        if not improved or gain < tol * max(cost, 1.0):
            converged = True
            break
    return x, cost, converged, it


# --------------------------------------------------------------------------- problem
class _Problem:
    """Residual model shared by the full fit, the per-view hold-out fit and the Hessian."""

    def __init__(self, template: np.ndarray, normals: np.ndarray, obs: np.ndarray, quality: np.ndarray,
                 pose0: np.ndarray):
        self.template = template
        self.centroid = template.mean(axis=0)
        self.tc = template - self.centroid
        self.normals = normals
        self.obs = obs                          # (V, L, 2)
        self.quality = quality                  # (V,)
        self.pose0 = pose0                      # (V, 3) degrees
        self.v, self.l = obs.shape[0], obs.shape[1]
        self.robust = np.ones((self.v, self.l))

    # parameter layout: [d (L*3) | per view: yaw pitch roll (deg), log_scale, tx, ty]
    def split(self, x: np.ndarray):
        d = x[: self.l * 3].reshape(self.l, 3)
        views = x[self.l * 3:].reshape(self.v, 6)
        return d, views

    def visibility(self, rot: np.ndarray) -> np.ndarray:
        """Soft self-occlusion weight of each landmark in each view (0 = facing away)."""
        nz = np.einsum("vc,lc->vl", rot[:, 2, :], self.normals)
        return np.clip((nz + 0.05) / 0.35, 0.0, 1.0)

    def sqrt_weight(self, rot: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.robust * self.visibility(rot) * self.quality[:, None], 0.0))

    def predict(self, x: np.ndarray):
        d, vp = self.split(x)
        rot = batch_rotations(vp[:, :3])
        pts = np.einsum("vab,lb->vla", rot[:, :2, :], self.tc + d)
        scale = np.exp(vp[:, 3])
        return scale[:, None, None] * pts + vp[:, None, 4:6], rot, scale

    def residual(self, x: np.ndarray) -> np.ndarray:
        d, vp = self.split(x)
        pred, rot, scale = self.predict(x)
        w = self.sqrt_weight(rot)
        data = (w[:, :, None] * (pred - self.obs) / (scale * SIGMA_OBS_M)[:, None, None]).ravel()
        prior = np.concatenate([(d[:, :2] / SIGMA_PRIOR_XY_M).ravel(), d[:, 2] / SIGMA_PRIOR_Z_M])
        pose = ((vp[:, :3] - self.pose0) / SIGMA_POSE_DEG).ravel()
        return np.concatenate([data, prior, pose])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic Gauss-Newton Jacobian of `residual` (robust / visibility weights held constant)."""
        d, vp = self.split(x)
        v, l = self.v, self.l
        angles = vp[:, :3]
        rot = batch_rotations(angles)
        scale = np.exp(vp[:, 3])
        w = self.sqrt_weight(rot)
        pts = self.tc + d                                            # (L, 3)
        proj = np.einsum("vab,lb->vla", rot[:, :2, :], pts)          # (V, L, 2), unscaled
        pred = scale[:, None, None] * proj + vp[:, None, 4:6]
        resid = w[:, :, None] * (pred - self.obs) / (scale * SIGMA_OBS_M)[:, None, None]
        drot = []                                                     # dR/d(angle in degrees), central differences
        for k in range(3):
            plus, minus = angles.copy(), angles.copy()
            plus[:, k] += 1e-3
            minus[:, k] -= 1e-3
            drot.append((batch_rotations(plus) - batch_rotations(minus)) / 2e-3)
        n_data = v * l * 2
        jac = np.zeros((n_data + l * 3 + v * 3, l * 3 + v * 6))
        lidx = np.arange(l)
        for i in range(v):
            rows = (i * l + lidx)[:, None] * 2 + np.arange(2)[None, :]       # (L, 2)
            cols_d = lidx[:, None] * 3 + np.arange(3)[None, :]                # (L, 3)
            wi = w[i] / SIGMA_OBS_M                                           # (L,)
            jac[rows[:, :, None], cols_d[:, None, :]] = wi[:, None, None] * rot[i, :2, :][None, :, :]
            base = l * 3 + i * 6
            for k in range(3):
                jac[rows, base + k] = wi[:, None] * np.einsum("ab,lb->la", drot[k][i, :2, :], pts)
            jac[rows, base + 3] = wi[:, None] * proj[i] - resid[i]
            jac[rows[:, 0], base + 4] = w[i] / (scale[i] * SIGMA_OBS_M)
            jac[rows[:, 1], base + 5] = w[i] / (scale[i] * SIGMA_OBS_M)
        r1 = n_data + l * 2
        for j in range(l):
            jac[n_data + j * 2, j * 3] = 1.0 / SIGMA_PRIOR_XY_M
            jac[n_data + j * 2 + 1, j * 3 + 1] = 1.0 / SIGMA_PRIOR_XY_M
            jac[r1 + j, j * 3 + 2] = 1.0 / SIGMA_PRIOR_Z_M
        r2 = r1 + l
        for i in range(v):
            for k in range(3):
                jac[r2 + i * 3 + k, l * 3 + i * 6 + k] = 1.0 / SIGMA_POSE_DEG
        return jac

    def steps(self) -> np.ndarray:
        return np.concatenate([np.full(self.l * 3, 1e-5), np.tile([1e-4, 1e-4, 1e-4, 1e-6, 1e-3, 1e-3], self.v)])

    def initial(self) -> np.ndarray:
        """Per-view scale/translation by weighted Procrustes against the template, zero displacement."""
        rows = []
        rot = batch_rotations(self.pose0)
        vis = self.visibility(rot)
        for i in range(self.v):
            proj = self.tc @ rot[i, :2, :].T
            w = vis[i] * self.quality[i] + 1e-6
            wn = w / w.sum()
            pm, om = (wn[:, None] * proj).sum(0), (wn[:, None] * self.obs[i]).sum(0)
            a, b = proj - pm, self.obs[i] - om
            denom = float((wn[:, None] * a * a).sum())
            s = float((wn[:, None] * a * b).sum() / denom) if denom > 1e-12 else 1.0
            s = max(s, 1.0)
            rows.append([*self.pose0[i], math.log(s), *(om - s * pm)])
        return np.concatenate([np.zeros(self.l * 3), np.asarray(rows).ravel()])

    def update_robust(self, x: np.ndarray) -> None:
        pred, _, scale = self.predict(x)
        err = (pred - self.obs) / (scale * SIGMA_OBS_M)[:, None, None]
        self.robust = huber_weights(err.reshape(-1, 2), delta=HUBER_DELTA).reshape(self.v, self.l)


def _solve(problem: _Problem, x0: np.ndarray, max_iter: int, outer: int = 3):
    x, conv, its = x0.copy(), False, 0
    for _ in range(outer):  # IRLS outer loop
        problem.update_robust(x)
        x, _, conv, n = _levenberg_marquardt(problem.residual, x, problem.steps(), max_iter=max_iter,
                                             jac_fun=problem.jacobian)
        its += n
    problem.update_robust(x)
    return x, conv, its


def _fit_single_view(problem: _Problem, d: np.ndarray, view_row: np.ndarray, obs: np.ndarray,
                     pose0: np.ndarray, quality: float):
    """Fit only one view's pose/scale/translation to FIXED geometry (hold-out evaluation)."""
    sub = _Problem(problem.template, problem.normals, obs[None], np.array([quality]), pose0[None])

    def fun(vp: np.ndarray) -> np.ndarray:
        return sub.residual(np.concatenate([d.ravel(), vp]))

    steps = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-3, 1e-3])
    vp, _, _, _ = _levenberg_marquardt(fun, view_row.copy(), steps, max_iter=25)
    pred, rot, scale = sub.predict(np.concatenate([d.ravel(), vp]))
    w = sub.visibility(rot)[0]
    err_mm = np.linalg.norm(pred[0] - obs, axis=1) / scale[0] * 1000.0
    return float((err_mm * w).sum() / max(w.sum(), 1e-9))


# --------------------------------------------------------------------------- public API
def fit_identity_landmarks(
    observations: Sequence[ViewLandmarks],
    *,
    template: Mesh3D | None = None,
    holdout: bool = True,
    max_iterations: int = 40,
    outlier_holdout_mm: float = 6.0,
) -> IdentityShapeFit:
    """Estimate identity-specific 3D landmark positions from several views of one person."""
    if not observations:
        raise DomainError("NO_OBSERVATIONS", "At least one view with landmarks is required", 422)
    layout = layout_of(observations[0].points)
    for obs in observations:
        if layout_of(obs.points) != layout:
            raise DomainError("MIXED_LANDMARK_LAYOUTS", "All views must use the same landmark layout", 422)
        if not np.isfinite(np.asarray(obs.points, np.float64)).all():
            raise DomainError("NON_FINITE_LANDMARKS", f"View {obs.view_index} has non-finite landmarks", 422)
    mesh = template or identity_template(2)
    t_lm, t_n = template_landmarks(layout, mesh), landmark_normals(layout, mesh)

    obs_arr = np.stack([np.asarray(o.points, np.float64) for o in observations])
    quality = np.clip(np.array([o.quality for o in observations], np.float64), 0.05, 1.0)
    pose0 = np.array([o.pose_deg for o in observations], np.float64)
    problem = _Problem(t_lm, t_n, obs_arr, quality, pose0)

    x, converged, iterations = _solve(problem, problem.initial(), max_iterations)
    d, vp = problem.split(x)
    pred, rot, scale = problem.predict(x)
    vis = problem.visibility(rot)
    err_mm = np.linalg.norm(pred - obs_arr, axis=2) / scale[:, None] * 1000.0

    # posterior covariance from the Gauss-Newton Hessian (data + prior rows)
    jac = problem.jacobian(x)
    cov_diag = np.diag(np.linalg.inv(jac.T @ jac + 1e-9 * np.eye(x.size)))
    sigma_post = np.sqrt(np.maximum(cov_diag[: problem.l * 3].reshape(problem.l, 3), 0.0))
    c_xy = np.clip(1.0 - np.max(sigma_post[:, :2], axis=1) / SIGMA_PRIOR_XY_M, 0.0, 1.0)
    c_z = np.clip(1.0 - sigma_post[:, 2] / SIGMA_PRIOR_Z_M, 0.0, 1.0)

    holdout_mm: list[float | None] = [None] * len(observations)
    if holdout and len(observations) >= 3:
        for i in range(len(observations)):
            keep = [k for k in range(len(observations)) if k != i]
            sub = _Problem(t_lm, t_n, obs_arr[keep], quality[keep], pose0[keep])
            rows = np.concatenate([d.ravel(), vp[keep].ravel()])
            xs, _, _ = _solve(sub, rows, 12, outer=2)  # bounded: warm-started from the full solution
            ds, _ = sub.split(xs)
            holdout_mm[i] = _fit_single_view(problem, ds, vp[i], obs_arr[i], pose0[i], float(quality[i]))

    # A view is inconsistent when its hold-out error is above an absolute floor AND more than twice the median of
    # the other views' hold-out errors. (Every hold-out refit still contains the corrupted view, so the peers'
    # errors are inflated too; the corrupted view stands out relative to them, not in absolute terms.)
    def holdout_ok(i: int) -> bool:
        h = holdout_mm[i]
        peers = [x for k, x in enumerate(holdout_mm) if k != i and x is not None]
        if h is None or not peers:
            return True
        return not (h > outlier_holdout_mm and h > 2.0 * float(np.median(peers)))
    views = []
    for i, o in enumerate(observations):
        w = vis[i] > 0.2
        rms = float(np.sqrt(np.mean(err_mm[i][w] ** 2))) if w.any() else float("nan")
        inliers = float(np.mean(problem.robust[i][w] > 0.99)) if w.any() else 0.0
        yaw, pitch, roll = (float(a) for a in vp[i, :3])
        ho = holdout_mm[i]
        views.append(ViewFit(o.view_index, yaw, pitch, roll, float(scale[i]), float(vp[i, 4]), float(vp[i, 5]), rms, ho,
                             inliers, consistent=bool(holdout_ok(i) and inliers >= 0.5)))

    yaw_span = float(np.ptp(vp[:, 0])) if len(observations) > 1 else 0.0
    depth_fraction = float(np.mean(c_z >= 0.5))
    notes: list[str] = []
    if len(observations) < 2 or yaw_span < YAW_SPAN_FOR_DEPTH_DEG:
        level = "single_view_prior"
        notes.append(f"Yaw span {yaw_span:.1f} deg < {YAW_SPAN_FOR_DEPTH_DEG:.0f} deg: depth cannot be triangulated; "
                     "depth stays at the anthropometric prior.")
    elif len(observations) < 3 or depth_fraction < 0.5 or yaw_span < 60.0:
        level = "limited_multiview"
        notes.append("Depth is partially constrained; add left/right three-quarter or profile views for full coverage.")
    else:
        level = "multiview"
    holdout_vals = [h for h in holdout_mm if h is not None]
    return IdentityShapeFit(
        layout=layout, template_landmarks=t_lm, model_center=problem.centroid.copy(), landmarks_3d=t_lm + d,
        displacement=d, sigma_post=sigma_post,
        confidence_xy=c_xy, confidence_z=c_z, landmark_confidence=np.minimum(c_xy, c_z), views=views,
        rms_mm=float(np.sqrt(np.mean([v.rms_mm ** 2 for v in views if np.isfinite(v.rms_mm)]))) if views else float("nan"),
        holdout_rms_mm=float(np.sqrt(np.mean(np.square(holdout_vals)))) if holdout_vals else None,
        yaw_span_deg=yaw_span, level=level, converged=converged, iterations=iterations, notes=notes)


# --------------------------------------------------------------------------- template deformation
def default_kernel_width(n_controls: int) -> float:
    """Gaussian kernel width (m): wide for the 5 sparse controls, tight for the 68-point set."""
    return 0.045 if n_controls <= 8 else 0.022


def kernel_support(vertices: np.ndarray, control_points: np.ndarray, confidence: np.ndarray,
                   rho: float | None = None) -> np.ndarray:
    """Per-vertex 0..1 score: the best Gaussian-weighted confidence among the controls. A max, not a noisy-OR:
    nearby controls are correlated, so several weak ones must not add up to false confidence."""
    ctrl = np.asarray(control_points, np.float64)
    width = default_kernel_width(len(ctrl)) if rho is None else rho
    kv = np.exp(-(np.linalg.norm(np.asarray(vertices, np.float64)[:, None] - ctrl[None], axis=2) ** 2) / (2 * width ** 2))
    return np.clip((kv * np.clip(confidence, 0.0, 1.0)[None, :]).max(axis=1), 0.0, 1.0)


def deform_template(
    template: Mesh3D,
    control_template: np.ndarray,
    control_target: np.ndarray,
    control_confidence: np.ndarray,
    *,
    rho: float | None = None,
    ridge: float = 1e-2,
    max_displacement_m: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Smoothly carry the template onto the fitted landmarks.

    Gaussian-kernel ridge regression of the control displacements: the field decays to ZERO away from the
    controls, so unobserved regions keep the prior shape instead of being extrapolated. Returns
    (vertices, vertex_support in 0..1, applied strength). Strength < 1 means the full displacement would have
    folded triangles and was scaled back.
    """
    ctrl_t = np.asarray(control_template, np.float64)
    disp = np.asarray(control_target, np.float64) - ctrl_t
    conf = np.clip(np.asarray(control_confidence, np.float64), 0.0, 1.0)
    if rho is None:
        rho = default_kernel_width(len(ctrl_t))
    pair = np.linalg.norm(ctrl_t[:, None] - ctrl_t[None], axis=2)
    kmat = np.exp(-(pair ** 2) / (2 * rho ** 2))
    # low-confidence controls are regularised harder, i.e. they pull the surface less
    coeff = np.linalg.solve(kmat + ridge * np.diag(1.0 / np.maximum(conf, 0.02)), disp * conf[:, None])
    verts = template.vertices.astype(np.float64)
    kv = np.exp(-(np.linalg.norm(verts[:, None] - ctrl_t[None], axis=2) ** 2) / (2 * rho ** 2))
    field = kv @ coeff
    norms = np.linalg.norm(field, axis=1, keepdims=True)
    field = field * np.minimum(1.0, max_displacement_m / np.maximum(norms, 1e-12))
    support = (kv * conf[None, :]).max(axis=1)

    base_n = template.compute_face_normals()
    strength = 1.0
    for _ in range(8):
        trial = Mesh3D(verts + strength * field, template.faces, uvs=template.uvs)
        fn = trial.compute_face_normals()
        front = base_n[:, 2] > 0.3
        folded = np.any(fn[front, 2] < 0.05)
        if not folded:
            break
        strength *= 0.6
    else:
        strength = 0.0
    return verts + strength * field, np.clip(support, 0.0, 1.0), float(strength)


def estimate_pose_from_landmarks(points: np.ndarray) -> tuple[float, float, float]:
    """Standard-frame (yaw, pitch up+, roll) from a view's landmarks (5- or 68-point)."""
    pts = np.asarray(points, np.float64)
    five = five_from_68(pts) if pts.shape[0] == 68 else pts
    pose = estimate_pose_5pt(five)
    return pose.yaw, pose.pitch, pose.roll


# --------------------------------------------------------------------------- pose against KNOWN geometry
@dataclass
class ViewPoseFit:
    """Camera (weak-perspective) of ONE frame against fixed identity geometry. Standard-frame angles in degrees."""
    yaw: float
    pitch: float
    roll: float
    scale: float                 # pixels per metre
    tx: float
    ty: float
    center: np.ndarray           # the model point the rotation acts about
    rms_mm: float                # weighted landmark reprojection error at the face
    rms_iod: float               # same error in units of the 3D inter-ocular distance
    converged: bool

    def camera(self) -> WeakPerspectiveCamera:
        rot = batch_rotations(np.array([[self.yaw, self.pitch, self.roll]]))[0]
        shift = self.scale * (rot @ self.center)[:2]
        return WeakPerspectiveCamera(rotation=tuple(tuple(float(x) for x in row) for row in rot), scale=float(self.scale),
                                     tx=float(self.tx - shift[0]), ty=float(self.ty - shift[1]))


def solve_view_pose(
    points2d: np.ndarray,
    model_points: np.ndarray,
    normals: np.ndarray,
    pose0_deg: tuple[float, float, float],
    weights: np.ndarray | None = None,
    *,
    iod_points: tuple[int, int] = (0, 1),
    pose_sigma_deg: float = SIGMA_POSE_DEG,
) -> ViewPoseFit:
    """MAP pose (yaw/pitch/roll/scale/translation) of one view against FIXED 3D points.

    Same residual model as the multi-view fit (so the two are consistent), with the geometry held at the identity
    asset's landmarks. `weights` down-weights landmarks that move with expression (e.g. mouth corners).
    """
    pts = np.asarray(points2d, np.float64)
    model = np.asarray(model_points, np.float64)
    if pts.shape != (len(model), 2) or not np.isfinite(pts).all() or not np.isfinite(model).all():
        raise DomainError("INVALID_POSE_INPUT", "Pose solve needs matching finite 2D/3D point sets", 422)
    w = np.ones(len(model)) if weights is None else np.clip(np.asarray(weights, np.float64), 0.0, 1.0)
    problem = _Problem(model, np.asarray(normals, np.float64), pts[None], np.array([1.0]), np.asarray(pose0_deg, np.float64)[None])
    problem.robust = w[None].copy()
    x0 = problem.initial()
    zero_d = np.zeros(len(model) * 3)
    start = x0[len(zero_d):]

    def residual(vp: np.ndarray) -> np.ndarray:
        r = problem.residual(np.concatenate([zero_d, vp]))
        n_data = len(model) * 2
        return np.concatenate([r[:n_data], r[-3:] * (SIGMA_POSE_DEG / pose_sigma_deg)])   # data rows + pose prior rows

    steps = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-3, 1e-3])
    vp, _, converged, _ = _levenberg_marquardt(residual, start, steps, max_iter=30)
    # Huber reweighting once (a single bad landmark must not drag the pose), then refine
    pred, rot, scale = problem.predict(np.concatenate([zero_d, vp]))
    err = (pred - pts[None]) / (scale * SIGMA_OBS_M)[:, None, None]
    problem.robust = w[None] * huber_weights(err.reshape(-1, 2), delta=HUBER_DELTA).reshape(1, -1)
    vp, _, converged, _ = _levenberg_marquardt(residual, vp, steps, max_iter=30)
    pred, rot, scale = problem.predict(np.concatenate([zero_d, vp]))
    err_m = np.linalg.norm(pred[0] - pts, axis=1) / scale[0]
    ww = np.maximum(w, 1e-9)
    rms = float(np.sqrt(np.sum(ww * err_m ** 2) / np.sum(ww)))
    iod = float(np.linalg.norm(model[iod_points[0]] - model[iod_points[1]]))
    return ViewPoseFit(yaw=float(vp[0]), pitch=float(vp[1]), roll=float(vp[2]), scale=float(np.exp(vp[3])), tx=float(vp[4]),
                       ty=float(vp[5]), center=problem.centroid.copy(), rms_mm=rms * 1000.0,
                       rms_iod=rms / max(iod, 1e-9), converged=bool(converged))
