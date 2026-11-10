# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Swap scope and the silhouette replacement step (Stage 6B).

Scope (what part of the target head the source geometry replaces):

``full_craniofacial``  100% source bone + jaw + forehead. The old silhouette is removed: overhang filled with background, hair /
                       ears / neck warped onto the new perimeter. DEFAULT.
``adaptive_skull``     the head shape is ``lock * beta_source + (1 - lock) * beta_target`` with ``lock = skull_blend``; the same
                       silhouette handling as above, applied to whatever silhouette the engine rendered for that blend.
``inner_face``         legacy: only the interior of the target's own face outline is replaced. The swap cannot leave the target's
                       silhouette, so nothing needs filling or warping.

``resolve_scope`` maps a scope to the ``source_identity_lock`` that ``ArticulatedHeadLayer.forward`` takes, whether the silhouette
step runs, and whether the swap is confined to the target outline. The compositor never rescales the silhouettes it is handed:
they are already the geometry the engine rendered for that lock.

The replacement itself (``SilhouetteReplacer.apply``), in this order and entirely BEFORE the source face is composited:

1. ``DeltaP`` perimeter correspondences (mesh vertex pairs when the engine supplies them, otherwise the two silhouettes) drive a
   ``PerimeterCageWarper`` field.
2. The structure layer (hair + ears + neck) is resampled through that field and laid back over everything outside the new head.
3. What the structure vacated, and the overhang ring ``dilate(M_tgt) & ~M_src``, are filled with background by the
   ``SilhouetteDeltaInpainter`` (temporal donors first, multi-scale spatial fill after).
4. Foreground occluders are never moved, filled or used as donors.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.rendering.perimeter_cage import CageReport, PerimeterCageWarper, perimeter_from_masks
from roop_3d.rendering.silhouette_inpaint import (DonorFrame, InpaintReport, SilhouetteDeltaInpainter, as_mask)

SWAP_SCOPES = ("full_craniofacial", "adaptive_skull", "inner_face")
DEFAULT_SWAP_SCOPE = "full_craniofacial"


@dataclass(frozen=True)
class ScopePlan:
    scope: str
    source_identity_lock: float      # ArticulatedHeadLayer.forward(source_identity_lock=...): share of beta_source in the skull
    replace_silhouette: bool         # warp hair / ears / neck and fill the overhang
    confine_to_target: bool          # legacy: the swap may not leave the target's own face outline


def resolve_scope(scope: str = DEFAULT_SWAP_SCOPE, skull_blend: float = 1.0) -> ScopePlan:
    if scope not in SWAP_SCOPES:
        raise DomainError("SWAP_SCOPE_INVALID", f"swap scope must be one of {SWAP_SCOPES}", 422)
    blend = float(skull_blend)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise DomainError("SWAP_SCOPE_INVALID", "skull_blend must be within 0..1", 422)
    if scope == "inner_face":
        return ScopePlan(scope, 0.0, False, True)
    return ScopePlan(scope, 1.0 if scope == "full_craniofacial" else blend, True, False)


@dataclass(frozen=True)
class SilhouetteReplacementReport:
    scope: str
    applied: bool
    reason: str = ""
    perimeter_source: str = "none"       # mesh | silhouettes | none
    structure_pixels: int = 0
    vacated_pixels: int = 0
    cage: CageReport | None = None
    inpaint: InpaintReport | None = None

    def as_dict(self) -> dict:
        out = {"scope": self.scope, "applied": self.applied, "reason": self.reason, "perimeter_source": self.perimeter_source,
               "structure_pixels": self.structure_pixels, "vacated_pixels": self.vacated_pixels}
        if self.cage is not None:
            out["cage"] = {"identity": self.cage.identity, "reason": self.cage.reason, "points": self.cage.points,
                           "max_displacement_px": round(self.cage.max_displacement_px, 3),
                           "fold_scale": round(self.cage.fold_scale, 4), "min_jacobian": round(self.cage.min_jacobian, 4)}
        if self.inpaint is not None:
            out["inpaint"] = {"tier": self.inpaint.tier, "overhang_pixels": self.inpaint.overhang_pixels,
                              "temporal_pixels": self.inpaint.temporal_pixels, "spatial_pixels": self.inpaint.spatial_pixels,
                              "donors_used": self.inpaint.donors_used, "donors_offered": self.inpaint.donors_offered,
                              "rejected": dict(self.inpaint.rejected), "spatial_method": self.inpaint.spatial_method,
                              "notes": list(self.inpaint.notes)}
        return out


class SilhouetteReplacer:
    def __init__(self, inpainter: SilhouetteDeltaInpainter | None = None, warper: PerimeterCageWarper | None = None) -> None:
        self.inpainter = inpainter or SilhouetteDeltaInpainter()
        self.warper = warper or PerimeterCageWarper()

    def apply(self, frame: np.ndarray, *, target_mask, source_mask, plan: ScopePlan, structure_mask=None, occluder=None,
              donors=(), perimeter: tuple[np.ndarray, np.ndarray] | None = None, invalid=None
              ) -> tuple[np.ndarray, np.ndarray, SilhouetteReplacementReport]:
        """Returns (edited frame, change alpha 0..1 for blending the edit back, report). ``perimeter`` = (source_xy, target_xy).
        ``invalid`` pixels (padding outside the real frame) are never donors."""
        image = SilhouetteDeltaInpainter._image(frame)
        shape = image.shape[:2]
        if not plan.replace_silhouette:
            return image.copy(), np.zeros(shape, np.float32), SilhouetteReplacementReport(plan.scope, False, "SCOPE_KEEPS_TARGET_SILHOUETTE")
        target = as_mask(target_mask, shape, "target mask")
        source = as_mask(source_mask, shape, "source mask")
        front = np.zeros(shape, bool) if occluder is None else as_mask(occluder, shape, "occluder") > 0.01
        structure = np.zeros(shape, np.float32)
        if structure_mask is not None:
            structure = as_mask(structure_mask, shape, "structure mask") * (~front)
        has_structure = bool((structure > 0.5).any())

        warped_structure = warped_image = None
        cage = None
        origin = "none"
        vacated = np.zeros(shape, bool)
        near_source = cv2.dilate((source > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if has_structure:
            if perimeter is not None:
                src_xy, tgt_xy = perimeter
                origin = "mesh"
            else:
                src_xy, tgt_xy = perimeter_from_masks(source, target)
                origin = "silhouettes"
            field = self.warper.estimate(src_xy, tgt_xy, shape)
            cage = field.report
            if not field.identity:
                warped_structure = self.warper.warp(field, structure) * (~near_source) * (~front)
                warped_image = self.warper.warp(field, image)
                vacated = (structure > 0.5) & (warped_structure < 0.5) & ~near_source & ~front

        keep_out = (structure > 0.01) if has_structure else np.zeros(shape, bool)
        if invalid is not None:
            keep_out = keep_out | (as_mask(invalid, shape, "invalid mask") > 0.5)
        filled, inpaint = self.inpainter.fill(
            image, target, source, occluder=front if front.any() else None, donors=donors,
            extra_hole=vacated if vacated.any() else None, extra_excluded=keep_out if keep_out.any() else None,
            keep=((structure > 0.5) & ~vacated) if has_structure else None)
        out = filled
        if warped_structure is not None:
            alpha = np.clip(warped_structure, 0, 1)[..., None]
            out = np.clip(np.rint(filled.astype(np.float32) * (1 - alpha) + warped_image.astype(np.float32) * alpha), 0, 255).astype(np.uint8)
        else:
            alpha = np.zeros((*shape, 1), np.float32)
        changed = np.any(out != image, axis=2).astype(np.float32)
        report = SilhouetteReplacementReport(plan.scope, True, perimeter_source=origin, structure_pixels=int((structure > 0.5).sum()),
                                             vacated_pixels=int(vacated.sum()), cage=cage, inpaint=inpaint)
        return out, changed, report


class DonorHistory:
    """The last ``depth`` UNSWAPPED frames of each track and where the head was in them (frame space): tier (a) donors."""

    def __init__(self, depth: int = 2) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.depth = depth
        self._frames: dict[int, deque] = {}

    def push(self, track_id: int, frame: np.ndarray, head_mask: np.ndarray | None, frame_index: int | None = None) -> None:
        if self.depth == 0:
            return
        queue = self._frames.setdefault(track_id, deque(maxlen=self.depth))
        mask = None if head_mask is None else (np.asarray(head_mask) > 0.3).astype(np.uint8)
        if frame_index is not None:
            # Re-rendering the same frame (a preview, a retry) replaces its entry instead of donating to itself.
            for entry in [e for e in queue if e[0] == frame_index]:
                queue.remove(entry)
        queue.append((frame_index, np.asarray(frame).copy(), mask))

    def donors(self, track_id: int, shape: tuple[int, int], frame_index: int | None = None,
               max_gap: int | None = None) -> list[DonorFrame]:
        """Newest first. With ``frame_index`` and ``max_gap``, frames further away in time (a seek) are not donors."""
        out = []
        for index, image, mask in reversed(self._frames.get(track_id, ())):
            if image.shape[:2] != tuple(shape):
                continue
            if frame_index is not None and index is not None:
                if index == frame_index or (max_gap is not None and abs(index - frame_index) > max_gap):
                    continue
            out.append(DonorFrame(image, mask))
        return out

    def retain(self, active) -> None:
        for key in [key for key in self._frames if key not in active]:
            del self._frames[key]

    def clear(self) -> None:
        self._frames.clear()
