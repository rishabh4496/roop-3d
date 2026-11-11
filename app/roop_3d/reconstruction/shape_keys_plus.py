# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Clean-room asymmetric morph application for craniofacial and expression bases."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


class ShapeKeyMode(str, Enum):
    SYMMETRIC = "Symmetric"
    LEFT_ONLY = "Left Only"
    RIGHT_ONLY = "Right Only"
    MIRROR_FLIP = "Mirror Flip"
    SOLO_ACTIVE_SHAPE_KEY = "Solo Active Shape Key"


class AsymmetricShapeKeyManager:
    """Apply arbitrary ``beta`` or ``psi`` linear morph bases on either side.

    Coordinates use the project head frame (+x image-right). Left/right masks
    use a cosine transition across ``midline_x``; ``blend_width`` is the full
    transition width in coordinate units. Shape bases and inputs are untouched.
    """

    def __init__(self, vertices: Any, *, midline_x: float | None = None, blend_width: float = 0.02):
        rest = np.asarray(vertices, np.float32)
        if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 2 or not np.isfinite(rest).all():
            raise ValueError("vertices must be a finite (V,3) array")
        if not np.isfinite(blend_width) or blend_width < 0:
            raise ValueError("blend_width must be finite and nonnegative")
        self.vertices = np.ascontiguousarray(rest)
        self.midline_x = float(np.median(rest[:, 0]) if midline_x is None else midline_x)
        if not np.isfinite(self.midline_x):
            raise ValueError("midline_x must be finite")
        self.blend_width = float(blend_width)
        signed = self.vertices[:, 0] - self.midline_x
        if self.blend_width == 0:
            right = (signed >= 0).astype(np.float32)
        else:
            t = np.clip((signed + self.blend_width / 2) / self.blend_width, 0.0, 1.0)
            right = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
        self._right = right
        self._left = 1.0 - right

    @property
    def left_weights(self) -> np.ndarray:
        return self._left.copy()

    @property
    def right_weights(self) -> np.ndarray:
        return self._right.copy()

    def weight_mask(self, side: str) -> np.ndarray:
        if side == "left" or side == "_L":
            return self.left_weights
        if side == "right" or side == "_R":
            return self.right_weights
        raise ValueError("side must be left/_L or right/_R")

    @staticmethod
    def _mode(value: ShapeKeyMode | str) -> ShapeKeyMode:
        if isinstance(value, ShapeKeyMode):
            return value
        try:
            return ShapeKeyMode(value)
        except ValueError:
            normalized = str(value).strip().replace("_", " ").casefold()
            for mode in ShapeKeyMode:
                if mode.value.casefold() == normalized:
                    return mode
            raise ValueError(f"unknown shape key mode: {value}") from None

    def apply(self, morph_delta: Any, mode: ShapeKeyMode | str = ShapeKeyMode.SYMMETRIC) -> np.ndarray:
        """Return a displacement array for one already-combined shape key."""
        delta = np.asarray(morph_delta, np.float32)
        if delta.shape != self.vertices.shape or not np.isfinite(delta).all():
            raise ValueError(f"morph_delta must be finite with shape {self.vertices.shape}")
        selected = self._mode(mode)
        if selected is ShapeKeyMode.SYMMETRIC or selected is ShapeKeyMode.SOLO_ACTIVE_SHAPE_KEY:
            return np.array(delta, copy=True)
        if selected is ShapeKeyMode.LEFT_ONLY:
            return delta * self._left[:, None]
        if selected is ShapeKeyMode.RIGHT_ONLY:
            return delta * self._right[:, None]
        return self._mirror_flip(delta)

    def deform(self, basis: Any, coefficients: Any, *, mode: ShapeKeyMode | str = ShapeKeyMode.SYMMETRIC,
               active_index: int | None = None, active_value: float | None = None) -> np.ndarray:
        """Evaluate a linear morph basis and return deformed vertices.

        A 60-column basis is accepted for craniofacial ``beta`` and a 100-column
        basis for ``psi``; any finite model-specific coefficient count is valid.
        Solo mode explicitly zeroes all coefficients except the active key.
        """
        directions = np.asarray(basis, np.float32)
        values = np.asarray(coefficients, np.float32).reshape(-1)
        if directions.ndim != 3 or directions.shape[:2] != self.vertices.shape or directions.shape[2] != len(values):
            raise ValueError("basis must have shape (V,3,K) matching the coefficient vector")
        if not np.isfinite(directions).all() or not np.isfinite(values).all():
            raise ValueError("basis and coefficients must be finite")
        selected = self._mode(mode)
        if selected is ShapeKeyMode.SOLO_ACTIVE_SHAPE_KEY:
            if active_index is None or not 0 <= int(active_index) < len(values):
                raise ValueError("Solo Active Shape Key requires a valid active_index")
            value = float(values[int(active_index)] if active_value is None else active_value)
            if not np.isfinite(value):
                raise ValueError("active_value must be finite")
            delta = directions[:, :, int(active_index)] * value
        else:
            if active_value is not None:
                raise ValueError("active_value applies only to Solo Active Shape Key")
            delta = np.einsum("vck,k->vc", directions, values)
        return self.vertices + self.apply(delta, selected)

    def _mirror_flip(self, delta: np.ndarray) -> np.ndarray:
        """Keep the stronger authored side and reflect its offsets onto the other side."""
        left_energy = float(np.sum((delta * self._left[:, None]) ** 2))
        right_energy = float(np.sum((delta * self._right[:, None]) ** 2))
        source_left = left_energy >= right_energy
        source_mask, destination_mask = (self._left, self._right) if source_left else (self._right, self._left)
        source_ids = np.flatnonzero(source_mask > 0.5)
        if not len(source_ids):
            return np.zeros_like(delta)
        source_points = self.vertices[source_ids].copy()
        source_points[:, 0] = 2 * self.midline_x - source_points[:, 0]
        nearest = cKDTree(source_points).query(self.vertices, k=1)[1]
        reflected = delta[source_ids[nearest]].copy()
        reflected[:, 0] *= -1.0
        own = delta * source_mask[:, None]
        mirrored = reflected * destination_mask[:, None]
        return own + mirrored


__all__ = ["ShapeKeyMode", "AsymmetricShapeKeyManager"]
