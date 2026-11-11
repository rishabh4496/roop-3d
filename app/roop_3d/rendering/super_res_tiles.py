# SPDX-License-Identifier: MIT
"""VRAM-aware, overlap-blended tiling for ultra-high-resolution head restoration."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import cv2
import numpy as np


TILE_MODEL_ALIASES = {"gpen realistic", "gpen-realistic", "gpen", "ultramax", "ultra max"}


@dataclass(frozen=True)
class ImageTile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _axis_plan(length: int, tile: int, overlap: int) -> tuple[list[int], int]:
    if length <= tile:
        return [0], length
    count = int(math.ceil((length - overlap) / (tile - overlap)))
    # Use smaller windows when possible. This avoids a very large accidental
    # overlap at the last tile of an image whose size is not a stride multiple.
    window = min(tile, int(math.ceil((length + (count - 1) * overlap) / count)))
    if count == 1:
        return [0], window
    starts = [int(round(i * (length - window) / (count - 1))) for i in range(count)]
    return starts, window


def split_image_tiles(width: int, height: int, *, tile_size: int, overlap: int) -> list[ImageTile]:
    """Cover an image with overlapping tiles, including exact coverage at the far edges."""
    if min(width, height, tile_size) <= 0 or overlap < 0 or overlap >= tile_size // 2:
        raise ValueError("image/tile dimensions must be positive and overlap must be below half a tile")
    xs, tile_width = _axis_plan(width, tile_size, overlap)
    ys, tile_height = _axis_plan(height, tile_size, overlap)
    return [ImageTile(x, y, min(width, x + tile_width), min(height, y + tile_height)) for y in ys for x in xs]


def _default_memory_probe() -> Mapping[str, Any] | None:
    try:
        from roop_3d.engine.vram_watchdog import VRAMWatchdog
        return VRAMWatchdog.get_instance().get_vram_status()
    except Exception:
        return None


def _axis_cosine_weights(length: int, *, left_overlap: int, right_overlap: int,
                         has_left: bool, has_right: bool) -> np.ndarray:
    values = np.ones(length, np.float32)
    if has_left and left_overlap > 0:
        n = min(length, left_overlap)
        t = (np.arange(n, dtype=np.float32) + 0.5) / n
        values[:n] *= 0.5 - 0.5 * np.cos(np.pi * t)
    if has_right and right_overlap > 0:
        n = min(length, right_overlap)
        t = (np.arange(n, dtype=np.float32)[::-1] + 0.5) / n
        values[-n:] *= 0.5 - 0.5 * np.cos(np.pi * t)
    return values


class SuperResTileEnhancer:
    """Run an enhancement callable whole-frame or as cosine-feathered tiles.

    ``memory_probe`` may return VRAM fields such as ``used_mb``/``total_mb``,
    ``free_mb``/``budget_mb`` or an explicit ``pressure_pct``. A probe is kept
    injectable so CPU-only and TensorRT-free installations share the same code.
    """

    def __init__(self, *, tile_size: int = 1024, overlap: int = 128,
                 pressure_threshold: float = 82.0, memory_probe: Callable[[], Mapping[str, Any] | None] | None = None):
        if tile_size < 64 or overlap < 0 or overlap >= tile_size // 2:
            raise ValueError("tile_size must be >=64 and overlap must be below half the tile size")
        if not 50 <= pressure_threshold < 100:
            raise ValueError("pressure_threshold must be in [50, 100)")
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.pressure_threshold = float(pressure_threshold)
        self.memory_probe = memory_probe or _default_memory_probe
        self.last_run: dict[str, Any] = {"tiled": False, "reason": "not_run", "tile_count": 0}

    @staticmethod
    def _pressure(snapshot: Mapping[str, Any] | None, budget_mb: float | None) -> float | None:
        if not snapshot:
            return None
        for key in ("budget_pressure_pct", "used_pct", "pressure_pct"):
            if key in snapshot and snapshot[key] is not None:
                value = float(snapshot[key])
                if value <= 1.0:
                    value *= 100.0
                return float(np.clip(value, 0, 100))
        total = snapshot.get("budget_mb", snapshot.get("total_mb", budget_mb))
        if total is None or float(total) <= 0:
            return None
        if snapshot.get("free_mb") is not None:
            used = float(total) - float(snapshot["free_mb"])
        elif snapshot.get("used_mb") is not None:
            used = float(snapshot["used_mb"])
        else:
            return None
        return float(np.clip(100.0 * used / float(total), 0.0, 100.0))

    @staticmethod
    def _oom(error: BaseException) -> bool:
        text = str(error).casefold()
        return any(token in text for token in ("out of memory", "cuda_error_out_of_memory", "cudnn_status_alloc_failed"))

    def should_tile(self, image: np.ndarray, *, model_name: str, full_head: bool = True,
                    frame_resolution: tuple[int, int] | None = None,
                    budget_mb: float | None = None,
                    memory_snapshot: Mapping[str, Any] | None = None) -> tuple[bool, str]:
        h, w = image.shape[:2]
        resolution = frame_resolution or (w, h)
        high_resolution = max(resolution) >= 2160 or min(resolution) >= 2160
        model = model_name.casefold().replace("_", " ").strip()
        supported_model = (model in TILE_MODEL_ALIASES or "ultramax" in model or "ultra max" in model
                           or ("gpen" in model and "realistic" in model))
        if not full_head or not supported_model or not high_resolution:
            return False, "not_full_head_supported_4k_model"
        snapshot = memory_snapshot
        if snapshot is None and self.memory_probe is not None:
            try:
                snapshot = self.memory_probe()
            except Exception:
                snapshot = None
        pressure = self._pressure(snapshot, budget_mb)
        if pressure is None:
            return False, "vram_pressure_unavailable"
        if pressure >= self.pressure_threshold:
            return True, f"vram_pressure_{pressure:.1f}_pct"
        return False, f"vram_pressure_{pressure:.1f}_pct_below_threshold"

    def _tiled(self, image: np.ndarray, enhancer: Callable[[np.ndarray], np.ndarray], *,
               tile_size: int | None = None, overlap: int | None = None) -> np.ndarray:
        h, w = image.shape[:2]
        active_tile_size = int(tile_size or self.tile_size)
        active_overlap = self.overlap if overlap is None else int(overlap)
        tiles = split_image_tiles(w, h, tile_size=active_tile_size, overlap=active_overlap)
        rows = sorted(set(tile.y0 for tile in tiles))
        cols = sorted(set(tile.x0 for tile in tiles))
        by_origin = {(tile.x0, tile.y0): tile for tile in tiles}
        accum = None
        weight_sum = None
        scale_x = scale_y = None
        for row_index, y0 in enumerate(rows):
            for col_index, x0 in enumerate(cols):
                tile = by_origin[(x0, y0)]
                output = np.asarray(enhancer(image[tile.y0:tile.y1, tile.x0:tile.x1].copy()))
                if output.ndim != 3 or output.shape[2] != image.shape[2] or not np.isfinite(output).all():
                    raise ValueError("enhancer must return a finite HxWxC image with the input channel count")
                local_x, local_y = output.shape[1] / tile.width, output.shape[0] / tile.height
                if scale_x is None:
                    scale_x, scale_y = local_x, local_y
                    out_w, out_h = int(round(w * scale_x)), int(round(h * scale_y))
                    accum = np.zeros((out_h, out_w, output.shape[2]), np.float32)
                    weight_sum = np.zeros((out_h, out_w), np.float32)
                elif not np.isclose(local_x, scale_x, rtol=1e-3) or not np.isclose(local_y, scale_y, rtol=1e-3):
                    raise ValueError("enhancer output scale changed between tiles")
                x0o, y0o = int(round(tile.x0 * scale_x)), int(round(tile.y0 * scale_y))
                x1o, y1o = int(round(tile.x1 * scale_x)), int(round(tile.y1 * scale_y))
                expected = (max(1, y1o - y0o), max(1, x1o - x0o))
                if output.shape[:2] != expected:
                    output = cv2.resize(output, (expected[1], expected[0]), interpolation=cv2.INTER_CUBIC)
                prev_x = cols[col_index - 1] if col_index else None
                next_x = cols[col_index + 1] if col_index + 1 < len(cols) else None
                prev_y = rows[row_index - 1] if row_index else None
                next_y = rows[row_index + 1] if row_index + 1 < len(rows) else None
                left = int(round((by_origin[(prev_x, y0)].x1 - tile.x0) * scale_x)) if prev_x is not None else 0
                right = int(round((tile.x1 - next_x) * scale_x)) if next_x is not None else 0
                top = int(round((by_origin[(x0, prev_y)].y1 - tile.y0) * scale_y)) if prev_y is not None else 0
                bottom = int(round((tile.y1 - next_y) * scale_y)) if next_y is not None else 0
                wx = _axis_cosine_weights(expected[1], left_overlap=left, right_overlap=right,
                                          has_left=prev_x is not None, has_right=next_x is not None)
                wy = _axis_cosine_weights(expected[0], left_overlap=top, right_overlap=bottom,
                                          has_left=prev_y is not None, has_right=next_y is not None)
                weight = wy[:, None] * wx[None, :]
                accum[y0o:y1o, x0o:x1o] += output.astype(np.float32) * weight[..., None]
                weight_sum[y0o:y1o, x0o:x1o] += weight
        if accum is None or weight_sum is None or np.any(weight_sum <= 0):
            raise RuntimeError("tile plan failed to cover the output image")
        result = accum / weight_sum[..., None]
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            result = np.clip(np.rint(result), info.min, info.max).astype(image.dtype)
        else:
            result = result.astype(image.dtype)
        self.last_run = {"tiled": True, "reason": "vram_pressure_or_oom", "tile_count": len(tiles),
                         "tile_size": active_tile_size, "overlap": active_overlap,
                         "scale": [float(scale_x), float(scale_y)]}
        return result

    def _tiled_with_stepdown(self, image: np.ndarray, enhancer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Retry failed tiles with smaller windows, so one large patch cannot strand an 8K job."""
        size = self.tile_size
        while True:
            overlap = min(self.overlap, max(0, size // 4))
            try:
                return self._tiled(image, enhancer, tile_size=size, overlap=overlap)
            except Exception as error:
                next_size = max(64, size // 2)
                if not self._oom(error) or next_size == size:
                    raise
                size = next_size

    def enhance(self, image: np.ndarray, enhancer: Callable[[np.ndarray], np.ndarray], *,
                model_name: str, full_head: bool = True, frame_resolution: tuple[int, int] | None = None,
                budget_mb: float | None = None, memory_snapshot: Mapping[str, Any] | None = None) -> np.ndarray:
        """Enhance once, falling back to tiles on OOM and pre-tiling under pressure."""
        value = np.asarray(image)
        if value.ndim != 3 or min(value.shape[:2]) < 1 or value.shape[2] not in (1, 3, 4):
            raise ValueError("image must be a non-empty HxWxC array")
        tile, reason = self.should_tile(value, model_name=model_name, full_head=full_head,
                                        frame_resolution=frame_resolution, budget_mb=budget_mb,
                                        memory_snapshot=memory_snapshot)
        if tile:
            return self._tiled_with_stepdown(value, enhancer)
        try:
            output = np.asarray(enhancer(value.copy()))
            if output.ndim != 3 or output.shape[2] != value.shape[2]:
                raise ValueError("enhancer must return an HxWxC image with the input channel count")
            self.last_run = {"tiled": False, "reason": reason, "tile_count": 1}
            return output
        except Exception as error:
            resolution = frame_resolution or (value.shape[1], value.shape[0])
            model = model_name.casefold().replace("_", " ").strip()
            supports_tiles = (model in TILE_MODEL_ALIASES or "ultramax" in model or "ultra max" in model
                              or ("gpen" in model and "realistic" in model))
            if self._oom(error) and full_head and supports_tiles and max(resolution) >= 2160:
                return self._tiled_with_stepdown(value, enhancer)
            raise
