"""Open, licence-clean parametric head model built from MakeHuman CC0 data (FLAME alternative).

``MakeHumanHeadModel`` is a drop-in ``FLAMEModel`` (same ``forward`` / ``get_landmarks`` contract, so ``FLAMEFitter``
still works with it) and ``fit_multiview`` jointly fits ONE shared identity to several views of the same person, each with
its own camera pose, from the detector's 5 landmarks (eyes, nose tip, mouth corners).

Frame: right-handed OpenCV head frame in metres (x right, y down, z away from the camera the face points at), so the
identity rotation is a frontal face. Honest scope: 5 landmarks per view constrain pose and coarse proportions (face
width, eye/mouth spacing, nose depth, gender axis) -- NOT fine identity detail; unconstrained axes stay at the prior (0).
The asset is the head+neck only; ears are present, eyeballs/teeth/tongue/hair are not.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.flame_fitter import FLAMEModel
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    Mesh3D,
    Pose3D,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_euler,
)
from roop_3d.reconstruction.makehuman_build import (
    ASSET_NAME,
    DEFAULT_ASSET_DIR,
    asset_digest,
    build_asset,
)
from roop_3d.reconstruction.multiview_fit import _levenberg_marquardt

MODEL_ID = "makehuman-head-cc0"
LICENSE_ID = "CC0-1.0"
LICENSE_NOTICE = ("MakeHuman data assets (base mesh and morph targets) are released under CC0-1.0. "
                  "The AGPL MakeHuman application code is not used.")


class MakeHumanHeadModel(FLAMEModel):
    """Linear parametric head: ``V = V0 + S @ beta + E @ psi`` (beta: identity, psi: expression, ~[-1, 1] slider units)."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        base = Mesh3D(arrays["vertices"], arrays["faces"], uvs=arrays["uvs"], face_uvs=arrays["face_uvs"])
        super().__init__(base_mesh=base, landmark_indices=arrays["landmark_indices"],
                         shape_basis=arrays["shape_basis"], expr_basis=arrays["expr_basis"])
        self.shape_names = [str(n) for n in arrays["shape_names"]]
        self.expr_names = [str(n) for n in arrays["expr_names"]]
        self.landmark_names = [str(n) for n in arrays["landmark_names"]]
        self.model_id = MODEL_ID

    @classmethod
    def load(cls, path: Path | str | None = None, *, auto_build: bool = False, verify: bool = True) -> "MakeHumanHeadModel":
        target = Path(path) if path else DEFAULT_ASSET_DIR / ASSET_NAME
        if not target.exists():
            if not auto_build or path:
                raise DomainError(
                    "MAKEHUMAN_ASSET_MISSING",
                    f"{target} not found. Build it (downloads ~130 small CC0 files, hash-verified): "
                    "python -m roop_3d.reconstruction.makehuman_build", 404)
            target = build_asset()
        with np.load(target, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if verify:
            sidecar = target.with_name(target.name + ".json")
            if sidecar.exists():
                expected = json.loads(sidecar.read_text("utf-8")).get("asset_digest")
                if expected != asset_digest(arrays):
                    raise DomainError("MAKEHUMAN_ASSET_CORRUPT", f"{target} does not match its recorded digest", 422)
        return cls(arrays)

    def forward(self, shape_params: np.ndarray, expr_params: np.ndarray, jaw_pose: np.ndarray | None = None) -> np.ndarray:
        """Vertices for the given coefficients. ``jaw_pose`` is ignored: open the mouth with the ``mouth_open`` expression."""
        return super().forward(shape_params, expr_params, None)

    def landmarks_for(self, shape_params: np.ndarray, expr_params: np.ndarray | None = None) -> np.ndarray:
        """(5, 3) landmark positions, computed from the landmark rows only (cheap; exact for a linear model)."""
        lm = self.base_mesh.vertices[self.landmark_indices].astype(np.float64)
        beta = np.asarray(shape_params, np.float64).reshape(-1)
        k = min(len(beta), self.num_shape)
        lm = lm + np.einsum("lck,k->lc", self.shape_basis[self.landmark_indices][:, :, :k].astype(np.float64), beta[:k])
        if expr_params is not None:
            psi = np.asarray(expr_params, np.float64).reshape(-1)
            m = min(len(psi), self.num_expr)
            lm = lm + np.einsum("lcm,m->lc", self.expr_basis[self.landmark_indices][:, :, :m].astype(np.float64), psi[:m])
        return lm

    def mesh(self, shape_params: np.ndarray, expr_params: np.ndarray | None = None) -> Mesh3D:
        v = self.forward(shape_params, np.zeros(self.num_expr) if expr_params is None else expr_params)
        return Mesh3D(v, self.base_mesh.faces, uvs=self.base_mesh.uvs, face_uvs=self.base_mesh.face_uvs)


# --------------------------------------------------------------------------------------------------- multi-view fit
@dataclass
class ViewObservation:
    landmarks: np.ndarray            # (5, 2) pixels, detector order: image-left eye, image-right eye, nose, left/right mouth
    width: int
    height: int
    name: str = ""
    weight: float = 1.0
    reference_yaw: float | None = None   # an independent estimate (degrees), used only for reporting


@dataclass
class ViewResult:
    name: str
    pose: Pose3D
    rms_px: float                    # fitted shared identity
    neutral_rms_px: float            # same view with beta = 0 (the unfitted MakeHuman default head), pose re-solved
    holdout_rms_px: float            # beta fitted WITHOUT this view, pose re-solved on this view only
    rms_over_face_height: float
    reference_yaw: float | None = None
    rejected: bool = False           # landmarks inconsistent with the other views: excluded from the shared identity


@dataclass
class MultiViewResult:
    shape_params: np.ndarray
    views: list[ViewResult]
    converged: bool
    iterations: int
    cost: float
    notes: list[str] = field(default_factory=list)

    def named_shape(self, model: MakeHumanHeadModel, top: int = 8) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.shape_params))[:top]
        return [(model.shape_names[i], float(self.shape_params[i])) for i in order]


def intrinsics_for(view: ViewObservation, fov_degrees: float = 45.0) -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(fov_degrees, view.width, view.height)


def _face_scale_px(lm: np.ndarray) -> float:
    """Eye-line to mouth-line distance in pixels: yaw-insensitive and the same unit across views."""
    return float(max(abs(lm[3:].mean(axis=0)[1] - lm[:2].mean(axis=0)[1]), 1.0))


def _project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    out, _ = cv2.projectPoints(np.ascontiguousarray(points, np.float64), rvec.reshape(3, 1), tvec.reshape(3, 1), k, None)
    return out.reshape(-1, 2)


def _pnp(points3d: np.ndarray, pts2d: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(points3d.astype(np.float64), pts2d.astype(np.float64), k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise DomainError("PNP_SOLVER_FAILED", "PnP failed on the model landmarks", 422)
    return rvec.reshape(3), tvec.reshape(3)


def _pseudo_huber(r: np.ndarray, delta: float) -> np.ndarray:
    """Smooth robust residual transform: ~r for |r| << delta, ~sqrt(2 delta |r|) for outliers."""
    return delta * np.sqrt(2.0 * (np.sqrt(1.0 + (r / delta) ** 2) - 1.0)) * np.sign(r)


DEFAULT_PRIOR_STRENGTH = 1.0   # ridge weight on beta used for the final fit
SCREEN_PRIOR_STRENGTH = 4.0    # stronger prior for the outlier-screening pass: width trades off with yaw, so a weak prior lets one bad view bend beta
OUTLIER_RMS_UNITS = 3.0        # a view is rejected from the shared identity when its screened rms exceeds 15 % of its eye-to-mouth distance
GM_SIGMA_UNITS = 3.0  # Geman-McClure scale in 1/20-of-eye-to-mouth-distance units (15 % of that distance)


def _geman_mcclure(r: np.ndarray, sigma: float) -> np.ndarray:
    """Redescending robust transform: squared value is sigma^2 r^2 / (r^2 + sigma^2), saturating for gross outliers."""
    return sigma * r / np.sqrt(r * r + sigma * sigma)


def _solve_pose(lm3d: np.ndarray, view: ViewObservation, k: np.ndarray, init: tuple[np.ndarray, np.ndarray] | None = None,
                delta: float = 0.35):
    """Pose of one view for a fixed 3D landmark set (LM on 6 parameters), returning (rvec, tvec, rms_px)."""
    obs = np.asarray(view.landmarks, np.float64)
    scale = _face_scale_px(obs)
    candidates = []
    r0, t0 = init if init is not None else _pnp(lm3d, obs, k)
    candidates.append((r0, t0))

    def fun(x: np.ndarray) -> np.ndarray:
        return _pseudo_huber(((_project(lm3d, x[:3], x[3:], k) - obs) / scale * 20.0).ravel(), delta * 20.0)

    best = None
    for r, t in candidates:
        x0 = np.concatenate([r, t])
        x, cost, _, _ = _levenberg_marquardt(fun, x0, np.array([1e-4] * 3 + [1e-4] * 3), max_iter=60)
        if best is None or cost < best[1]:
            best = (x, cost)
    x = best[0]
    rms = float(np.sqrt(np.mean(np.sum((_project(lm3d, x[:3], x[3:], k) - obs) ** 2, axis=1))))
    return x[:3], x[3:], rms


def fit_multiview(
    model: MakeHumanHeadModel,
    views: list[ViewObservation],
    *,
    fov_degrees: float = 45.0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_iterations: int = 60,
    n_shape: int | None = None,
    holdout: bool = True,
) -> MultiViewResult:
    """Joint MAP fit of one shared identity (beta) and per-view rigid poses to 5-point landmarks.

    Residuals are in units of 1/20 of each view's eye-to-mouth distance (so views of different resolution weigh equally),
    passed through a pseudo-Huber transform (robust to one bad detection), plus ``prior_strength * beta`` (ridge prior).
    """
    if len(views) < 1:
        raise DomainError("NO_VIEWS", "At least one view is required", 422)
    n_shape = min(n_shape or model.num_shape, model.num_shape)
    ks = [intrinsics_for(v, fov_degrees).matrix() for v in views]
    obs = [np.asarray(v.landmarks, np.float64) for v in views]
    if any(o.shape != (5, 2) for o in obs):
        raise DomainError("LANDMARK_SHAPE_MISMATCH", "Every view needs exactly 5 landmarks (5, 2)", 422)
    scales = [_face_scale_px(o) for o in obs]
    weights = [math.sqrt(max(v.weight, 1e-6)) for v in views]
    nv = len(views)
    lm0 = model.landmarks_for(np.zeros(n_shape))

    poses0 = [_pnp(lm0, o, k) for o, k in zip(obs, ks)]
    x0 = np.concatenate([np.zeros(n_shape), *[np.concatenate(p) for p in poses0]])
    delta = 0.35 * 20.0
    steps = np.array([1e-3] * n_shape + [1e-4] * (6 * nv))

    def optimise(prior: float, view_w: list[float], start_x: np.ndarray):
        def make_residual(robust: str):
            def residual(x: np.ndarray) -> np.ndarray:
                beta = x[:n_shape]
                lm = model.landmarks_for(beta)
                parts = []
                for i in range(nv):
                    p = x[n_shape + 6 * i: n_shape + 6 * i + 6]
                    r = ((_project(lm, p[:3], p[3:], ks[i]) - obs[i]) / scales[i] * 20.0).ravel() * view_w[i]
                    parts.append(_pseudo_huber(r, delta) if robust == "huber" else _geman_mcclure(r, GM_SIGMA_UNITS))
                parts.append(prior * beta)
                return np.concatenate(parts)
            return residual

        # Stage 1: pseudo-Huber from the start. Stage 2: redescending Geman-McClure, so gross outliers saturate.
        xa, _, c1, i1 = _levenberg_marquardt(make_residual("huber"), start_x, steps, max_iter=max_iterations)
        xb, cst, c2, i2 = _levenberg_marquardt(make_residual("gm"), xa, steps, max_iter=max_iterations)
        return xb, cst, bool(c1 and c2), i1 + i2

    def view_rms(x: np.ndarray, i: int) -> float:
        p = x[n_shape + 6 * i: n_shape + 6 * i + 6]
        lm = model.landmarks_for(x[:n_shape])
        return float(np.sqrt(np.mean(np.sum((_project(lm, p[:3], p[3:], ks[i]) - obs[i]) ** 2, axis=1))))

    notes: list[str] = []
    rejected = [False] * nv
    x_screen, _, _, it_screen = optimise(max(SCREEN_PRIOR_STRENGTH, prior_strength), weights, x0)
    if nv >= 3:
        flagged = [view_rms(x_screen, i) / scales[i] * 20.0 > OUTLIER_RMS_UNITS for i in range(nv)]
        if 0 < sum(flagged) <= nv - 2:
            rejected = flagged
            notes.append("views excluded from the shared identity (landmarks inconsistent with the others): "
                         + ", ".join(views[i].name or str(i) for i in range(nv) if rejected[i]))
        elif sum(flagged) > nv - 2:
            notes.append("most views look inconsistent; none excluded (check detections / identity of the set)")
    final_w = [1e-3 if rejected[i] else weights[i] for i in range(nv)]
    x, cost, converged, iters = optimise(prior_strength, final_w, x_screen)
    iters += it_screen
    beta = x[:n_shape]
    lm_fit = model.landmarks_for(beta)
    for i in range(nv):
        if rejected[i]:  # pose of a rejected view is re-solved against the final identity (it did not shape that identity)
            r_i, t_i, _ = _solve_pose(lm_fit, views[i], ks[i])
            x[n_shape + 6 * i: n_shape + 6 * i + 6] = np.concatenate([r_i, t_i])

    results: list[ViewResult] = []
    for i, view in enumerate(views):
        p = x[n_shape + 6 * i: n_shape + 6 * i + 6]
        rms = float(np.sqrt(np.mean(np.sum((_project(lm_fit, p[:3], p[3:], ks[i]) - obs[i]) ** 2, axis=1))))
        _, _, neutral_rms = _solve_pose(lm0, view, ks[i])
        if holdout and nv > 2 and not rejected[i]:
            others = [j for j in range(nv) if j != i]
            sub = fit_multiview(model, [views[j] for j in others], fov_degrees=fov_degrees, prior_strength=prior_strength,
                                max_iterations=max_iterations, n_shape=n_shape, holdout=False)
            _, _, hold_rms = _solve_pose(model.landmarks_for(sub.shape_params), view, ks[i])
        else:
            hold_rms = float("nan")
        r_mat = rodrigues_to_rotation_matrix(p[:3])
        yaw, pitch, roll = rotation_matrix_to_euler(r_mat, degrees=True)
        pose = Pose3D(rvec=tuple(float(a) for a in p[:3]), tvec=tuple(float(a) for a in p[3:]),
                      yaw=yaw, pitch=pitch, roll=roll, reprojection_error=rms)
        results.append(ViewResult(view.name or str(i), pose, rms, neutral_rms, hold_rms, rms / scales[i],
                                  view.reference_yaw, rejected[i]))
    if not holdout or nv < 3:
        notes.append("hold-out validation not computed (needs holdout=True and at least 3 views)")
    return MultiViewResult(shape_params=beta, views=results, converged=converged, iterations=iters, cost=cost, notes=notes)
