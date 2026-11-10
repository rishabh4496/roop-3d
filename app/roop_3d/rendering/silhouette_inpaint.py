# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Silhouette-delta background completion (Stage 6B).

When the swapped head has a different silhouette than the target's, the target's old cheeks / jaw / chin can stick out past the new
face (a *narrower* source). ``M_overhang = dilate(M_tgt) AND NOT M_src`` is that ring. It has to be filled with BACKGROUND before the
source face is composited, otherwise the old jaw shows around the new one.

Two tiers, tried in order, both only ever writing inside the ring:

(a) **Temporal donors** (video). Pixels the head hid in this frame were often visible in neighbouring frames (t-k .. t+k). Each donor
    is registered to the current frame with a *background* motion model - a homography fitted by RANSAC to dense optical flow sampled
    only OUTSIDE every head mask - so the non-rigid face never drives the registration and the model extrapolates cleanly into the
    hole. A donor is rejected unless it reproduces the current frame on a band surrounding the hole (photometric residual), is
    corrected for exposure drift on that band, and never supplies pixels that were head in the donor itself. Several donors are
    combined with a per-pixel median.
(b) **Spatial fill** (still images, and whatever no donor could see). A multi-scale Navier-Stokes fill: isophote propagation at the
    coarsest scale (continues background edges into the hole), then a refinement of a thin band at every finer scale so the
    boundary pixels are the real ones, then matched grain. (Navier-Stokes rather than Telea throughout: OpenCV's Telea drifts by
    up to 4 grey levels on a perfectly flat background, Navier-Stokes reproduces flat and linear backgrounds exactly.) An external neural inpainter (LaMa and friends) can be plugged in with
    ``neural_inpainter``; none is bundled, and the report says which method actually ran.

Honest limit: neither tier recovers background nobody ever saw. Tier (a) is real recovery when a donor saw it; tier (b) is plausible
continuation of the surroundings.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

NeuralInpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InpaintOptions:
    dilate_px: int = 6                  # dilation of M_tgt: covers the anti-aliased rim of the old face
    source_overlap_px: int = 2          # the hole may run this far under the new face's soft edge
    donor_margin_px: int = 4            # donors keep this distance from every head mask
    band_px: int = 7                    # surround band the donor must reproduce
    min_band_pixels: int = 40
    residual_max: float = 14.0          # median |current - donor| on the band (0..255) above which a donor is rejected
    min_votes: int = 1                  # donors that must agree on a pixel before tier (a) owns it
    flow_side: int = 256                # long side the background flow is estimated at
    min_flow_samples: int = 60
    ransac_px: float = 1.5
    scales: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
    ns_radius: int = 3
    refine_band_px: int = 3
    grain_strength: float = 0.8         # share of the surround's high-frequency std re-added to a spatial fill (0 = off)
    feather_px: float = 1.0

    def __post_init__(self) -> None:
        if min(self.dilate_px, self.source_overlap_px, self.donor_margin_px, self.refine_band_px) < 0 or self.band_px < 1:
            raise ValueError("pixel radii must be non-negative (band_px >= 1)")
        if self.min_votes < 1 or not self.scales or any(not 0 < s <= 1 for s in self.scales) or self.scales[-1] != 1.0:
            raise ValueError("min_votes >= 1 and scales within (0, 1], ending at 1.0")
        if not 0 <= self.grain_strength <= 1 or self.feather_px < 0 or self.residual_max <= 0:
            raise ValueError("grain_strength within 0..1, feather_px >= 0, residual_max > 0")


@dataclass(frozen=True)
class DonorFrame:
    """A neighbouring frame (same size as the current one) and, if known, where the head was in it (0..1, frame space)."""
    image: np.ndarray
    head_mask: np.ndarray | None = None


@dataclass(frozen=True)
class InpaintReport:
    overhang_pixels: int = 0
    filled_pixels: int = 0
    temporal_pixels: int = 0
    spatial_pixels: int = 0
    donors_offered: int = 0
    donors_used: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    spatial_method: str = "none"        # none | multiscale_ns | neural
    tier: str = "none"                  # none | temporal | spatial | temporal+spatial
    notes: tuple[str, ...] = ()


def as_mask(value, shape: tuple[int, int], name: str) -> np.ndarray:
    """Finite float32 plane of ``shape`` clipped to 0..1 (a soft mask is fine)."""
    plane = np.asarray(value, np.float32)
    if plane.shape != tuple(shape) or not np.isfinite(plane).all():
        raise DomainError("SILHOUETTE_MASK_INVALID", f"{name} must be a finite {shape[0]}x{shape[1]} plane", 422)
    return np.clip(plane, 0.0, 1.0)


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0 if radius > 0 else mask.astype(bool)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), _disk(radius)) > 0 if radius > 0 else mask.astype(bool)


def overhang_ring(target_mask, source_mask, *, dilate_px: int = 6, source_overlap_px: int = 2) -> np.ndarray:
    """``dilate(M_tgt) AND NOT M_src`` as a bool plane (the source mask is eroded by ``source_overlap_px`` so the hole runs
    slightly under the new face's soft edge and no sliver of the old jaw survives the feather)."""
    target = np.asarray(target_mask, np.float32)
    source = as_mask(source_mask, target.shape, "source mask")
    target = as_mask(target, target.shape, "target mask")
    return _dilate(target > 0.5, dilate_px) & ~_erode(source > 0.5, source_overlap_px)


def _bbox(mask: np.ndarray, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(shape[1], int(xs.max()) + 1 + pad), min(shape[0], int(ys.max()) + 1 + pad))


class SilhouetteDeltaInpainter:
    def __init__(self, options: InpaintOptions | None = None, *, neural_inpainter: NeuralInpainter | None = None) -> None:
        self.options = options or InpaintOptions()
        self.neural_inpainter = neural_inpainter

    # ------------------------------------------------------------------ public
    def overhang(self, target_mask, source_mask) -> np.ndarray:
        return overhang_ring(target_mask, source_mask, dilate_px=self.options.dilate_px,
                             source_overlap_px=self.options.source_overlap_px)

    def fill(self, frame: np.ndarray, target_mask, source_mask, *, occluder=None, donors: Sequence[DonorFrame] = (),
             extra_hole=None, extra_excluded=None, keep=None) -> tuple[np.ndarray, InpaintReport]:
        """Fill ``M_overhang`` (plus ``extra_hole``) of ``frame`` with background. ``occluder`` pixels (foreground in front of the
        face) are never touched and never donate. ``extra_excluded`` are further non-background pixels (old hair, ears, neck...); ``keep`` pixels are never written
        (they may still be excluded from donating)."""
        image = self._image(frame)
        shape = image.shape[:2]
        target = as_mask(target_mask, shape, "target mask")
        source = as_mask(source_mask, shape, "source mask")
        hole = self.overhang(target, source)
        excluded = (target > 0.01) | (source > 0.01)
        if extra_hole is not None:
            hole |= as_mask(extra_hole, shape, "extra hole") > 0.5
        if extra_excluded is not None:
            excluded |= as_mask(extra_excluded, shape, "extra excluded") > 0.01
        if occluder is not None:
            front = as_mask(occluder, shape, "occluder") > 0.01
            hole &= ~front
            excluded |= front
        if keep is not None:
            hole &= as_mask(keep, shape, "keep mask") <= 0.5
        return self.fill_region(image, hole, excluded, donors=donors)

    def fill_region(self, frame: np.ndarray, hole, excluded, *, donors: Sequence[DonorFrame] = ()) -> tuple[np.ndarray, InpaintReport]:
        """Fill ``hole`` from background only: ``excluded`` pixels (and the hole itself) are never used as donors."""
        image = self._image(frame)
        shape = image.shape[:2]
        hole = np.asarray(hole, bool)
        excluded = np.asarray(excluded, bool)
        if hole.shape != shape or excluded.shape != shape:
            raise DomainError("SILHOUETTE_MASK_INVALID", "hole and excluded must match the frame", 422)
        total = int(hole.sum())
        if total == 0:
            return image.copy(), InpaintReport(donors_offered=len(donors))
        opt = self.options
        unknown = hole | _dilate(excluded, opt.donor_margin_px)
        if unknown.all():
            raise DomainError("BACKGROUND_DONORS_UNAVAILABLE", "Silhouette completion needs surrounding background pixels", 422)
        rows, cols = np.nonzero(unknown)
        box = _bbox(unknown, max(opt.band_px + 4, int(0.25 * max(np.ptp(rows), np.ptp(cols)))), shape)
        x0, y0, x1, y1 = box
        sl = (slice(y0, y1), slice(x0, x1))
        roi, roi_hole, roi_unknown = image[sl].copy(), hole[sl], unknown[sl]

        notes: list[str] = []
        rejected: Counter = Counter()
        filled = roi.copy()
        remaining = roi_hole.copy()
        temporal_px = used = 0
        if donors:
            fill_t, covered, used, rej = self._temporal(roi, roi_hole, roi_unknown, [self._crop_donor(d, shape, sl) for d in donors])
            rejected.update(rej)
            if covered.any():
                filled[covered] = fill_t[covered]
                remaining &= ~covered
                temporal_px = int(covered.sum())
        spatial_px, method = 0, "none"
        if remaining.any():
            # Everything already filled temporally is trustworthy background: it may now donate to the spatial tier.
            known_unknown = roi_unknown & ~(roi_hole & ~remaining)
            fill_s, method, note = self._spatial(filled, remaining, known_unknown)
            if note:
                notes.append(note)
            filled[remaining] = fill_s[remaining]
            spatial_px = int(remaining.sum())
        out = image.copy()
        alpha = roi_hole.astype(np.float32)
        if opt.feather_px > 0:
            alpha = np.minimum(cv2.GaussianBlur(alpha, (0, 0), opt.feather_px) * 1.0, 1.0)
            alpha = np.maximum(alpha, roi_hole.astype(np.float32))
        blended = roi.astype(np.float32) * (1 - alpha[..., None]) + filled.astype(np.float32) * alpha[..., None]
        out[sl] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        tiers = [name for name, px in (("temporal", temporal_px), ("spatial", spatial_px)) if px]
        return out, InpaintReport(
            overhang_pixels=total, filled_pixels=temporal_px + spatial_px, temporal_pixels=temporal_px, spatial_pixels=spatial_px,
            donors_offered=len(donors), donors_used=used, rejected=dict(rejected), spatial_method=method,
            tier="+".join(tiers) or "none", notes=tuple(notes))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _image(frame) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DomainError("SILHOUETTE_FRAME_INVALID", "Frame must be a uint8 BGR image", 422)
        return image

    @staticmethod
    def _crop_donor(donor: DonorFrame, shape: tuple[int, int], sl) -> DonorFrame | None:
        image = np.asarray(donor.image)
        if image.shape[:2] != shape or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return None
        head = None
        if donor.head_mask is not None:
            head = np.asarray(donor.head_mask, np.float32)
            head = head[sl] if head.shape == shape and np.isfinite(head).all() else None
            if head is None:
                return None
        return DonorFrame(image[sl], head)

    # ------------------------------------------------------------------ tier (a)
    def _background_homography(self, current: np.ndarray, donor: np.ndarray, exclude: np.ndarray):
        """Homography current->donor fitted to dense flow sampled only on background, or None."""
        opt = self.options
        h, w = current.shape[:2]
        s = min(1.0, opt.flow_side / max(h, w))
        small = (max(8, int(round(w * s))), max(8, int(round(h * s))))
        a = cv2.cvtColor(cv2.resize(current, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(donor, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        bg = cv2.resize((~exclude).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
        bg = cv2.erode(bg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ys, xs = np.nonzero(bg)
        if len(ys) < opt.min_flow_samples:
            return None, "NO_BACKGROUND_SAMPLES"
        step = max(1, len(ys) // 1500)
        ys, xs = ys[::step], xs[::step]
        src = np.column_stack((xs, ys)).astype(np.float32)
        dst = src + flow[ys, xs]
        if float(np.median(np.linalg.norm(flow[ys, xs], axis=1))) < 0.15 and float(np.abs(flow[ys, xs]).max()) < 1.0:
            return np.eye(3), ""                                     # static camera: skip the fit
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, opt.ransac_px)
        if homography is None or inliers is None or int(inliers.sum()) < opt.min_flow_samples:
            return None, "BACKGROUND_MOTION_UNRELIABLE"
        scale = np.diag([s, s, 1.0])
        return np.linalg.inv(scale) @ homography @ scale, ""

    def _temporal(self, roi: np.ndarray, hole: np.ndarray, unknown: np.ndarray, donors: list):
        opt = self.options
        h, w = roi.shape[:2]
        rejected: Counter = Counter()
        band = _dilate(hole, opt.band_px) & ~hole & ~unknown
        votes, valids = [], []
        for donor in donors:
            if donor is None:
                rejected["DONOR_SHAPE_MISMATCH"] += 1
                continue
            head = np.zeros((h, w), bool) if donor.head_mask is None else _dilate(donor.head_mask > 0.3, opt.donor_margin_px)
            homography, why = self._background_homography(roi, donor.image, unknown | head)
            if homography is None:
                rejected[why] += 1
                continue
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            warped = cv2.warpPerspective(donor.image, homography, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
            inside = cv2.warpPerspective(np.ones((h, w), np.uint8), homography, (w, h), flags=flags) > 0
            donor_head = cv2.warpPerspective(head.astype(np.uint8), homography, (w, h), flags=flags) > 0
            valid = inside & ~donor_head
            usable_band = band & valid
            if int(usable_band.sum()) < opt.min_band_pixels:
                rejected["DONOR_BAND_TOO_SMALL"] += 1
                continue
            diff = roi.astype(np.float32) - warped.astype(np.float32)
            residual = float(np.median(np.abs(diff[usable_band])))
            if residual > opt.residual_max:
                rejected["DONOR_PHOTOMETRIC_MISMATCH"] += 1
                continue
            bias = np.median(diff[usable_band], axis=0)                 # exposure / white-balance drift between the frames
            votes.append(np.clip(warped.astype(np.float32) + bias, 0, 255))
            valids.append(valid & hole)
        if not votes:
            return roi, np.zeros((h, w), bool), 0, rejected
        stack = np.stack(votes)
        ok = np.stack(valids)
        count = ok.sum(axis=0)
        covered = count >= opt.min_votes
        masked = np.where(ok[..., None], stack, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)               # pixels no donor sees are all-NaN by construction
            median = np.nanmedian(masked, axis=0)
        fill = np.where(covered[..., None], np.nan_to_num(median), roi.astype(np.float32))
        return np.clip(np.rint(fill), 0, 255).astype(np.uint8), covered, len(votes), rejected

    # ------------------------------------------------------------------ tier (b)
    def _spatial(self, image: np.ndarray, hole: np.ndarray, unknown: np.ndarray):
        """Returns (filled ROI, method, note). ``unknown`` = hole + non-background: only ``~unknown`` pixels are ever sources."""
        if self.neural_inpainter is not None:
            try:
                result = np.asarray(self.neural_inpainter(image.copy(), (unknown.astype(np.uint8) * 255)))
                if result.shape == image.shape and result.dtype == np.uint8:
                    return result, "neural", ""
                note = "NEURAL_INPAINTER_BAD_OUTPUT: used the multi-scale fill"
            except (DomainError, RuntimeError, OSError, ValueError) as error:
                note = f"NEURAL_INPAINTER_FAILED({type(error).__name__}): used the multi-scale fill"
        else:
            note = ""
        return self._multiscale(image, hole, unknown), "multiscale_ns", note

    def _multiscale(self, image: np.ndarray, hole: np.ndarray, unknown: np.ndarray) -> np.ndarray:
        opt = self.options
        h, w = image.shape[:2]
        estimate = None
        for scale in opt.scales:
            sh, sw = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
            if estimate is None and min(sh, sw) < 12 and scale != opt.scales[-1]:
                continue
            level, unk = (image, unknown) if scale == 1.0 else self._downsample(image, unknown, sw, sh)
            if unk.all():
                continue
            unk_u8 = (unk.astype(np.uint8) * 255)
            if estimate is None:
                estimate = cv2.inpaint(level, unk_u8, opt.ns_radius, cv2.INPAINT_NS)        # isophote propagation
                continue
            seeded = level.copy()
            init = cv2.resize(estimate, (sw, sh), interpolation=cv2.INTER_CUBIC)
            seeded[unk] = init[unk]
            ring = unk & _dilate(~unk, opt.refine_band_px)
            estimate = cv2.inpaint(seeded, (ring.astype(np.uint8) * 255), opt.ns_radius, cv2.INPAINT_NS) if ring.any() else seeded
        if estimate is None or estimate.shape[:2] != (h, w):
            raise DomainError("BACKGROUND_DONORS_UNAVAILABLE", "Spatial completion found no usable background scale", 422)
        if opt.grain_strength > 0:
            estimate = self._add_grain(estimate, image, hole, unknown)
        return estimate

    @staticmethod
    def _downsample(image: np.ndarray, unknown: np.ndarray, sw: int, sh: int):
        """Area downsample that only averages KNOWN pixels (normalised convolution): a coarse pixel that mixes in any unknown
        pixel becomes unknown itself, so skin / hair colour never leaks into the background estimate through the pyramid."""
        known = (~unknown).astype(np.float32)
        weight = cv2.resize(known, (sw, sh), interpolation=cv2.INTER_AREA)
        total = cv2.resize(image.astype(np.float32) * known[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
        level = np.clip(np.rint(total / np.maximum(weight, 1e-6)[..., None]), 0, 255).astype(np.uint8)
        return level, weight < 0.999

    def _add_grain(self, filled: np.ndarray, image: np.ndarray, hole: np.ndarray, unknown: np.ndarray) -> np.ndarray:
        """Inpainting is smoother than a real sensor: re-add the surround's own high-frequency std (deterministic)."""
        opt = self.options
        # Measured only where the 1.5 px blur cannot reach an unknown pixel (an old-skin edge would read as "texture").
        surround = _dilate(hole, 28) & ~_dilate(unknown, 5)
        if int(surround.sum()) < 200:
            return filled
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        high = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
        std = float(np.clip(high[surround].std() * opt.grain_strength, 0.0, 6.0))
        if std < 0.3:
            return filled
        rng = np.random.default_rng(int(hole.sum()) & 0xFFFFFFFF)
        noise = cv2.GaussianBlur(rng.normal(0, 1, hole.shape).astype(np.float32), (0, 0), 0.6)
        noise *= std / max(float(noise[hole].std()), 1e-6)
        out = filled.astype(np.float32)
        out[hole] += noise[hole][:, None]
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
