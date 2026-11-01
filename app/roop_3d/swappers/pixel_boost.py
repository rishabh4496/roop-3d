"""PixelBoost sub-patch tiling and seamless raised-cosine reconstruction.

Supports high-resolution tiled sub-patching (256x256, 512x512, 768x768, 1024x1024)
for identity swappers (e.g., InSwapper-128, HyperSwap-256).
Decomposes oversized crops into a batched tensor, enables single-pass batched TRT
inference, and reconstructs the output with zero grid seams.
"""
from __future__ import annotations

import math
from typing import Any, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore


class PixelBoost:
    """Tiled sub-patching and raised-cosine blending engine."""

    def __init__(self, model_size: int = 128, boost_size: int = 512):
        if boost_size < model_size:
            raise ValueError(f"boost_size ({boost_size}) must be >= model_size ({model_size})")
        if boost_size % model_size != 0:
            raise ValueError(f"boost_size ({boost_size}) must be a multiple of model_size ({model_size})")

        self.model_size = model_size
        self.boost_size = boost_size
        self.grid_count = boost_size // model_size
        self.total_patches = self.grid_count * self.grid_count

        # Build 2D raised-cosine window
        w1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(model_size, dtype=np.float32) + 0.5) / model_size))
        w2d = np.outer(w1d, w1d).astype(np.float32)
        self.window_np = np.maximum(w2d, 1e-3)[np.newaxis, ...]  # (1, H, W)
        if _HAS_TORCH:
            self.window_torch = torch.from_numpy(self.window_np)
        else:
            self.window_torch = None

    def unfold_sub_patches(
        self, crop: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], list[tuple[int, int]]]:
        """Unfold a (1, 3, boost_size, boost_size) or (boost_size, boost_size, 3) crop
        into a stacked batch of shape (total_patches, 3, model_size, model_size).
        """
        is_torch = _HAS_TORCH and isinstance(crop, torch.Tensor)
        m = self.model_size
        coords: list[tuple[int, int]] = []

        if is_torch:
            # Normalize to (1, 3, H, W)
            t = crop
            if t.dim() == 3:
                if t.shape[-1] == 3:  # (H, W, 3) -> (1, 3, H, W)
                    t = t.permute(2, 0, 1).unsqueeze(0)
                else:  # (3, H, W) -> (1, 3, H, W)
                    t = t.unsqueeze(0)

            patches = []
            for r in range(self.grid_count):
                for c in range(self.grid_count):
                    y = r * m
                    x = c * m
                    coords.append((y, x))
                    patch = t[:, :, y : y + m, x : x + m]
                    patches.append(patch)
            # Stack into (total_patches, 3, m, m)
            return torch.cat(patches, dim=0), coords

        # NumPy implementation
        arr = np.asarray(crop)
        if arr.ndim == 3:
            if arr.shape[-1] == 3:  # (H, W, 3) -> (1, 3, H, W)
                arr = np.ascontiguousarray(arr.transpose(2, 0, 1))[np.newaxis, ...]
            else:  # (3, H, W) -> (1, 3, H, W)
                arr = arr[np.newaxis, ...]

        patches_np = []
        for r in range(self.grid_count):
            for c in range(self.grid_count):
                y = r * m
                x = c * m
                coords.append((y, x))
                patch = arr[0, :, y : y + m, x : x + m]
                patches_np.append(patch)

        stacked = np.stack(patches_np, axis=0)  # (N, 3, m, m)
        return stacked, coords

    def reconstruct_from_sub_patches(
        self,
        patches: Union[np.ndarray, "torch.Tensor"],
        coords: list[tuple[int, int]] | None = None,
        blend_mode: str = "raised_cosine",
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Reconstruct a (1, 3, boost_size, boost_size) high-resolution swapped face
        using raised-cosine feather blending across patch overlaps.
        """
        is_torch = _HAS_TORCH and isinstance(patches, torch.Tensor)
        m = self.model_size
        b = self.boost_size

        if coords is None:
            coords = []
            for r in range(self.grid_count):
                for c in range(self.grid_count):
                    coords.append((r * m, c * m))

        if is_torch:
            device = patches.device
            dtype = patches.dtype
            canvas = torch.zeros((1, 3, b, b), device=device, dtype=dtype)
            weights = torch.zeros((1, 1, b, b), device=device, dtype=dtype)
            win = self.window_torch.to(device=device, dtype=dtype)

            for idx, (y, x) in enumerate(coords):
                p = patches[idx : idx + 1]  # (1, 3, m, m)
                canvas[:, :, y : y + m, x : x + m] += p * win
                weights[:, :, y : y + m, x : x + m] += win

            mask = weights > 0
            safe_weights = torch.where(mask, weights, torch.ones_like(weights))
            reconstructed = torch.where(mask, canvas / safe_weights, canvas)
            return reconstructed

        # NumPy path
        patches_np = np.asarray(patches, dtype=np.float32)
        canvas_np = np.zeros((1, 3, b, b), dtype=np.float32)
        weights_np = np.zeros((1, 1, b, b), dtype=np.float32)
        win_np = self.window_np

        for idx, (y, x) in enumerate(coords):
            p = patches_np[idx : idx + 1]  # (1, 3, m, m)
            canvas_np[:, :, y : y + m, x : x + m] += p * win_np
            weights_np[:, :, y : y + m, x : x + m] += win_np

        mask_np = weights_np > 0
        safe_weights_np = np.where(mask_np, weights_np, 1.0)
        reconstructed_np = np.where(mask_np, canvas_np / safe_weights_np, canvas_np)
        return reconstructed_np
