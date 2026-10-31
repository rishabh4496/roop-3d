"""Deterministic overlapping ROI tiling and cosine-windowed reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class Tile:
    row: int
    column: int
    y0: int
    y1: int
    x0: int
    x1: int
    seed: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _cosine_ramp(length: int, *, rising: bool) -> np.ndarray:
    if length <= 0:
        return np.ones((0,), dtype=np.float64)
    # Endpoints are complementary across neighboring tiles: r(t)+r(1-t)=1.
    t = (np.arange(length, dtype=np.float64) + 0.5) / length
    ramp = 0.5 - 0.5 * np.cos(np.pi * t)
    return ramp if rising else ramp[::-1].copy()


class TileManager:
    def __init__(self, tile_size: int | tuple[int, int] = 512, overlap: int = 64,
                 *, job_seed: int, denoising_strength: float = 0.25):
        if isinstance(tile_size, int):
            tile_height = tile_width = tile_size
        else:
            tile_height, tile_width = tile_size
        if tile_height < 1 or tile_width < 1 or overlap < 0 or overlap >= min(tile_height, tile_width):
            raise ValueError("Tile dimensions must be positive and overlap smaller than either dimension")
        if type(job_seed) is not int or not 0 <= job_seed <= (2**64 - 1):
            raise ValueError("job_seed must be an unsigned 64-bit integer")
        if not math.isfinite(denoising_strength) or not 0.15 <= denoising_strength <= 0.45:
            raise ValueError("denoising_strength must be within 0.15..0.45")
        self.tile_height, self.tile_width = tile_height, tile_width
        self.overlap, self.job_seed = overlap, job_seed
        self.denoising_strength = float(denoising_strength)

    def split(self, image: np.ndarray) -> tuple[Tile, ...]:
        if (not isinstance(image, np.ndarray) or image.ndim not in (2, 3)
                or image.shape[0] <= 0 or image.shape[1] <= 0 or not np.isfinite(image).all()):
            raise DomainError("DIFFUSION_TILE_IMAGE_INVALID", "Tiling requires a finite nonempty HxW image", 422)
        height, width = image.shape[:2]
        ys = _starts(height, self.tile_height, self.overlap)
        xs = _starts(width, self.tile_width, self.overlap)
        result = []
        for row, y0 in enumerate(ys):
            for column, x0 in enumerate(xs):
                y1 = min(height, y0 + self.tile_height)
                x1 = min(width, x0 + self.tile_width)
                token = f"{self.job_seed}:{row}:{column}:{y0}:{x0}".encode("ascii")
                seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)
                result.append(Tile(row, column, y0, y1, x0, x1, seed))
        return tuple(result)

    @staticmethod
    def extract(image: np.ndarray, tile: Tile) -> np.ndarray:
        return image[tile.y0:tile.y1, tile.x0:tile.x1, ...].copy()

    @staticmethod
    def landmarks_for_tile(landmarks: np.ndarray, tile: Tile) -> np.ndarray:
        """Return the Stage 06/08 points visible in a tile, in tile-local XY."""
        if (not isinstance(landmarks, np.ndarray) or landmarks.ndim != 2 or landmarks.shape[1] != 2
                or not np.issubdtype(landmarks.dtype, np.number) or not np.isfinite(landmarks).all()):
            raise DomainError("DIFFUSION_LANDMARKS_INVALID", "Landmarks must be a finite Nx2 array", 422)
        inside = ((landmarks[:, 0] >= tile.x0) & (landmarks[:, 0] < tile.x1)
                  & (landmarks[:, 1] >= tile.y0) & (landmarks[:, 1] < tile.y1))
        local = landmarks[inside].astype(np.float32, copy=True)
        local[:, 0] -= tile.x0
        local[:, 1] -= tile.y0
        return local

    def _axis_weights(self, starts: list[int], index: int, tile_size: int,
                      image_length: int, start: int) -> np.ndarray:
        actual_length = min(tile_size, image_length - start)
        weights = np.ones(actual_length, dtype=np.float64)
        if index > 0:
            overlap = max(0, starts[index - 1] + tile_size - start)
            overlap = min(overlap, actual_length)
            if overlap:
                weights[:overlap] *= _cosine_ramp(overlap, rising=True)
        if index + 1 < len(starts):
            overlap = max(0, start + tile_size - starts[index + 1])
            overlap = min(overlap, actual_length)
            if overlap:
                weights[-overlap:] *= _cosine_ramp(overlap, rising=False)
        return weights

    def reassemble(self, image_shape: tuple[int, ...], tiles: tuple[Tile, ...] | list[Tile],
                   tile_images: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        if (len(image_shape) not in (2, 3) or image_shape[0] <= 0 or image_shape[1] <= 0
                or len(tiles) == 0 or len(tiles) != len(tile_images)):
            raise DomainError("DIFFUSION_TILE_SET_INVALID", "Output shape and tile collection are inconsistent", 422)
        height, width = image_shape[:2]
        ordered = sorted(tiles, key=lambda item: (item.row, item.column))
        expected_order = list(tiles)
        if ordered != expected_order:
            raise DomainError("DIFFUSION_TILE_SET_INVALID", "Tiles must be provided in split order", 422)
        ys = sorted({item.y0 for item in tiles})
        xs = sorted({item.x0 for item in tiles})
        channel_shape = image_shape[2:]
        accumulator = np.zeros(image_shape, dtype=np.float64)
        weights = np.zeros((height, width), dtype=np.float64)
        for tile, value in zip(tiles, tile_images):
            expected_shape = (tile.y1 - tile.y0, tile.x1 - tile.x0, *channel_shape)
            if not isinstance(value, np.ndarray) or value.shape != expected_shape or not np.isfinite(value).all():
                raise DomainError("DIFFUSION_TILE_OUTPUT_INVALID", "Tile output shape or finite-value check failed", 422)
            row_index, column_index = ys.index(tile.y0), xs.index(tile.x0)
            wy = self._axis_weights(ys, row_index, self.tile_height, height, tile.y0)
            wx = self._axis_weights(xs, column_index, self.tile_width, width, tile.x0)
            window = wy[:, None] * wx[None, :]
            accumulator[tile.y0:tile.y1, tile.x0:tile.x1, ...] += value.astype(np.float64) * (
                window[(...,) + (None,) * len(channel_shape)] if channel_shape else window)
            weights[tile.y0:tile.y1, tile.x0:tile.x1] += window
        if np.any(weights <= 0) or not np.isfinite(accumulator).all():
            raise DomainError("DIFFUSION_TILE_REASSEMBLY_FAILED", "Tile windows left uncovered or non-finite pixels", 422)
        divisor = weights[(...,) + (None,) * len(channel_shape)] if channel_shape else weights
        result = accumulator / divisor
        prototype = tile_images[0]
        if np.issubdtype(prototype.dtype, np.integer):
            limits = np.iinfo(prototype.dtype)
            result = np.clip(np.rint(result), limits.min, limits.max)
        else:
            limits = np.finfo(prototype.dtype)
            result = np.clip(result, limits.min, limits.max)
        return result.astype(prototype.dtype)
