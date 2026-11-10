"""Data-free skin albedo prior for atlas texels that no photo observed.

There is no third-party colour data in this module: no scans, no litsphere or texture images, no learned model. Everything is
the subject's own OBSERVED texels plus ordinary statistics, so the prior carries no licence of its own. (AlbedoMM, the obvious
statistical alternative, is academic-research-only; MakeHuman's litsphere PNGs are stamped "AGPL" although the repository says CC0.)

Method, in CIELAB (perceptual distances):

1. *Skin gate.* A skin-like set (YCrCb range, then within a Lab radius of its own median) removes eyes, hair, brows, teeth and
   specular highlights. It is a fixed support: only these texels can ever contribute to the fit (an earlier version only seeded
   with it and let dark hair back in, which visibly greyed the far field).
2. *Robust smooth fit.* A constant plus a small set of Gaussian bumps over the 3D surface position, fitted by ridge-penalised,
   Huber-weighted least squares (IRLS). Only the bumps are penalised, so far from observed texels the prediction relaxes to the
   subject's robust skin tone, and near observed texels it continues the observed gradient (forehead vs jaw, lit vs shaded side).
3. *Fill.* Only texels that are inside the atlas footprint and UNSEEN are written; observed texels are never touched.

What it is NOT: a statistical face model. It cannot know lip, brow or eyelid colour, freckles, scars, or any identity detail in a
region nobody saw; it extends a smooth tone. With no observed skin it refuses rather than inventing a colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from roop_3d.core.errors import DomainError

UNSEEN, OBSERVED = 0, 2                       # ConfidenceState values (kept local to avoid importing the pipeline)
MAX_FIT_TEXELS = 20_000
MIN_SKIN_TEXELS = 200
# Conservative sanity bounds for human skin under roughly neutral light (L*, a*, b*). Heuristic, used only to FLAG a suspicious
# tone (strong colour cast, wrong white balance); nothing is clamped to them.
GAMUT_LAB = {"L": (20.0, 92.0), "a": (3.0, 32.0), "b": (4.0, 46.0)}


# ------------------------------------------------------------------------------------------------------- colour helpers
def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR (..., 3) -> float32 Lab (L 0-100)."""
    flat = np.asarray(bgr, np.uint8).reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(np.shape(bgr))


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    flat = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    out = cv2.cvtColor(flat, cv2.COLOR_LAB2BGR).reshape(np.shape(lab))
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference."""
    return np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1)


def _skin_gate(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    return (ycc[:, 1] >= 133) & (ycc[:, 1] <= 173) & (ycc[:, 2] >= 77) & (ycc[:, 2] <= 127) & (ycc[:, 0] > 50)


# ---------------------------------------------------------------------------------------------------------- tone estimate
@dataclass
class SkinToneEstimate:
    tone_lab: np.ndarray
    tone_bgr: np.ndarray
    observed_texels: int
    skin_texels: int
    spread_de: float                 # robust (MAD-based) colour spread of the accepted skin texels
    in_gamut: bool
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"tone_bgr": [int(v) for v in self.tone_bgr], "tone_lab": [round(float(v), 2) for v in self.tone_lab],
                "observed_texels": self.observed_texels, "skin_texels": self.skin_texels,
                "spread_de": round(self.spread_de, 2), "in_gamut": self.in_gamut, "notes": list(self.notes)}


def _seed(bgr: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    gate = _skin_gate(bgr)
    if gate.sum() < MIN_SKIN_TEXELS:
        raise DomainError("ALBEDO_PRIOR_NO_SKIN",
                          f"Only {int(gate.sum())} observed texels look like skin (need {MIN_SKIN_TEXELS}); "
                          "no tone is invented when there is nothing to estimate it from", 422)
    centre = np.median(lab[gate], axis=0)
    close = gate & (delta_e(lab, centre) < 20.0)
    if close.sum() < MIN_SKIN_TEXELS:
        notes.append("skin-like texels are widely scattered in colour; the tone estimate is weak")
        close = gate
    return close, notes


def estimate_skin_tone(texture: np.ndarray, observed: np.ndarray) -> SkinToneEstimate:
    """Robust skin tone of the observed texels (``texture`` BGR uint8 (H, W, 3), ``observed`` bool (H, W))."""
    bgr = np.asarray(texture, np.uint8)[np.asarray(observed, bool)]
    if len(bgr) == 0:
        raise DomainError("ALBEDO_PRIOR_NO_SKIN", "There are no observed texels to estimate a skin tone from", 422)
    lab = bgr_to_lab(bgr)
    keep, notes = _seed(bgr, lab)
    tone = np.median(lab[keep], axis=0)
    spread = float(1.4826 * np.median(delta_e(lab[keep], tone)))
    in_gamut = all(GAMUT_LAB[k][0] <= tone[i] <= GAMUT_LAB[k][1] for i, k in enumerate("Lab"))
    if not in_gamut:
        notes.append("tone is outside the usual human-skin range: check white balance / colour cast in the source photos")
    return SkinToneEstimate(tone, lab_to_bgr(tone[None])[0], int(len(bgr)), int(keep.sum()), spread, in_gamut, notes)


# -------------------------------------------------------------------------------------------------------------- the prior
def _farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic well-spread centres (farthest-point sampling from the point nearest the centroid)."""
    first = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen = [first]
    dist = np.linalg.norm(points - points[first], axis=1)
    for _ in range(min(count, len(points)) - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return points[chosen]


class SurfaceAlbedoPrior:
    """Smooth Lab colour field over the 3D surface, relaxing to the subject's robust skin tone away from observed texels."""

    def __init__(self, centres: np.ndarray, length_m: float, coef: np.ndarray, tone: SkinToneEstimate, support: np.ndarray,
                 report: dict[str, Any]):
        self.centres, self.length_m, self.coef, self.tone, self.report = centres, length_m, coef, tone, report
        self._tree = cKDTree(support)

    @staticmethod
    def _design(points: np.ndarray, centres: np.ndarray, length_m: float) -> np.ndarray:
        d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
        return np.concatenate([np.ones((len(points), 1)), np.exp(-d2 / (2.0 * length_m ** 2))], axis=1)

    @classmethod
    def fit(cls, positions: np.ndarray, bgr: np.ndarray, *, length_m: float = 0.05, centres: int = 96, ridge: float = 0.05,
            iterations: int = 6, huber_k: float = 2.5) -> "SurfaceAlbedoPrior":
        """``positions`` (N, 3) metres, ``bgr`` (N, 3) uint8: the OBSERVED texels."""
        positions, bgr = np.asarray(positions, np.float64), np.asarray(bgr, np.uint8)
        if len(positions) != len(bgr) or len(positions) == 0:
            raise DomainError("ALBEDO_PRIOR_NO_SKIN", "positions and colours must be the same nonzero length", 422)
        if len(positions) > MAX_FIT_TEXELS:
            pick = np.linspace(0, len(positions) - 1, MAX_FIT_TEXELS).astype(np.int64)      # deterministic subsample
            positions, bgr = positions[pick], bgr[pick]
        lab = bgr_to_lab(bgr).astype(np.float64)
        seed, notes = _seed(bgr, lab.astype(np.float32))
        tone_lab = np.median(lab[seed], axis=0)
        spread = float(1.4826 * np.median(delta_e(lab[seed], tone_lab)))
        in_gamut = all(GAMUT_LAB[k][0] <= tone_lab[i] <= GAMUT_LAB[k][1] for i, k in enumerate("Lab"))
        if not in_gamut:
            notes.append("tone is outside the usual human-skin range: check white balance / colour cast in the source photos")
        weights = seed.astype(np.float64)
        centre_pts = _farthest_points(positions[seed], centres)
        phi = cls._design(positions, centre_pts, length_m)
        penalty = np.eye(phi.shape[1])
        penalty[0, 0] = 0.0                                                    # the constant is free; only the bumps shrink
        coef = np.zeros((phi.shape[1], 3))
        coef[0] = tone_lab
        for _ in range(iterations):
            nw = max(float(weights.sum()), 1.0)
            lhs = phi.T @ (phi * weights[:, None]) + ridge * nw * penalty
            coef = np.linalg.solve(lhs, phi.T @ (lab * weights[:, None]))
            res = delta_e(lab, phi @ coef)
            active = weights > 0
            scale = max(1.4826 * float(np.median(res[active])) if active.any() else 4.0, 2.0)   # robust sigma in Delta-E
            limit = huber_k * scale
            weights = np.where(res <= limit, 1.0, limit / np.maximum(res, 1e-9))
            weights = np.where(res > 5.0 * limit, 0.0, weights) * seed        # the skin gate is a fixed support: hair, lips, brows never re-enter
        final = delta_e(lab, phi @ coef)
        inliers = weights >= 0.99
        tone = SkinToneEstimate(tone_lab, lab_to_bgr(tone_lab[None].astype(np.float32))[0], int(len(bgr)), int(seed.sum()), spread,
                                in_gamut, notes)
        report = {"fit_texels": int(len(bgr)), "inlier_fraction": round(float(inliers.mean()), 4), "rejected_fraction": round(float((weights == 0).mean()), 4),
                  "residual_de_median": round(float(np.median(final[inliers])) if inliers.any() else 0.0, 3),
                  "residual_de_p90": round(float(np.percentile(final[inliers], 90)) if inliers.any() else 0.0, 3),
                  "centres": int(len(centre_pts)), "length_mm": round(length_m * 1000.0, 1), "ridge": ridge}
        return cls(centre_pts, length_m, coef, tone, positions[inliers] if inliers.any() else positions, report)

    def predict_lab(self, positions: np.ndarray) -> np.ndarray:
        return self._design(np.asarray(positions, np.float64), self.centres, self.length_m) @ self.coef

    def distance_to_observed_mm(self, positions: np.ndarray) -> np.ndarray:
        return self._tree.query(np.asarray(positions, np.float64))[0] * 1000.0


# ---------------------------------------------------------------------------------------------------------- atlas use
def prior_field(texture: np.ndarray, texel_state: np.ndarray, footprint: np.ndarray, positions: np.ndarray, **fit_options
                ) -> tuple[np.ndarray, SurfaceAlbedoPrior]:
    """The prior evaluated at every footprint texel: ``(H, W, 3)`` float32 BGR (zeros outside the footprint) and the fitted prior.

    Fitted on OBSERVED texels only. This is what :func:`complete_texture` takes as ``far_field`` so that, far from any photo, the
    harmonic extension relaxes to a smooth surface-aware skin tone instead of one flat colour.
    """
    footprint = np.asarray(footprint, bool)
    observed = footprint & (np.asarray(texel_state) == OBSERVED)
    prior = SurfaceAlbedoPrior.fit(positions[observed], np.asarray(texture, np.uint8)[observed], **fit_options)
    field_bgr = np.zeros(footprint.shape + (3,), np.float32)
    ys, xs = np.nonzero(footprint)
    field_bgr[ys, xs] = lab_to_bgr(prior.predict_lab(positions[ys, xs]).astype(np.float32)).astype(np.float32)
    return field_bgr, prior


@dataclass
class PriorFill:
    texture: np.ndarray              # uint8 BGR (H, W, 3), observed texels bit-identical to the input
    filled: np.ndarray               # bool (H, W): texels written from the prior
    report: dict[str, Any]


def fill_unseen_with_prior(texture: np.ndarray, texel_state: np.ndarray, footprint: np.ndarray, positions: np.ndarray, *,
                           feather_px: float = 3.0, **fit_options) -> PriorFill:
    """Write the prior into texels that are in ``footprint`` and UNSEEN (a standalone fill, for display and for diagnostics).

    Observed and interpolated texels are never modified; the prior fades in over ``feather_px`` texels so it never leaves a hard
    edge against what the photos show. Measured on real FaceSets this standalone fill is WORSE than the harmonic completion near
    observed texels (see docs); prefer :func:`prior_field` as the completion's far field.
    """
    texture = np.asarray(texture, np.uint8)
    footprint = np.asarray(footprint, bool)
    state = np.asarray(texel_state)
    observed = footprint & (state == OBSERVED)
    target = footprint & (state == UNSEEN)
    field_bgr, prior = prior_field(texture, state, footprint, positions, **fit_options)
    out = texture.copy()
    report: dict[str, Any] = {"method": "data_free_surface_prior", "third_party_data": False, "colour_space": "CIELAB",
                              "footprint_texels": int(footprint.sum()), "observed_texels": int(observed.sum()),
                              "unseen_texels": int(target.sum()), "skin_tone": prior.tone.public(), "fit": prior.report}
    if target.any():
        ys, xs = np.nonzero(target)
        pred = field_bgr[ys, xs]
        if feather_px > 0:
            dist = cv2.distanceTransform((~(footprint & (state != UNSEEN))).astype(np.uint8), cv2.DIST_L2, 3)[ys, xs]
            alpha = np.clip(dist / feather_px, 0.0, 1.0)[:, None]
            pred = alpha * pred + (1.0 - alpha) * out[ys, xs].astype(np.float32)
        out[ys, xs] = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
        far = prior.distance_to_observed_mm(positions[ys, xs])
        report["distance_to_observed_mm"] = {"median": round(float(np.median(far)), 1), "p90": round(float(np.percentile(far, 90)), 1),
                                              "max": round(float(far.max()), 1)}
    report["limits"] = ("A smooth skin tone only: no lip, brow, eyelid, freckle or identity detail is known for texels nobody saw. "
                        "Filled texels are priors, not observations.")
    return PriorFill(out, target, report)
