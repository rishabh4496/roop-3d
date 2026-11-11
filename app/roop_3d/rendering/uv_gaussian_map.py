# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""TensorRT-friendly UV detail planes and face-size-driven continuous LoD.

This module is a clean-room tensor layout and raster-resolution policy. It
ships no Gaussian-splat data, trained weights, hair assets, or third-party code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


UV_DETAIL_CHANNELS = (
    "offset_x", "offset_y", "offset_z", "normal_x", "normal_y", "normal_z",
    "hair", "eyebrow", "eyelash", "pore", "confidence", "cavity",
)
UV_DETAIL_CHANNEL_COUNT = 12
SUPPORTED_UV_SIZES = (64, 180, 256, 320)


class UVGaussianMicroDetailMap:
    """Contiguous FP32/FP16 NCHW detail maps with exactly twelve channels.

    Offsets use model-space metres, normals are signed unit-vector components,
    and all masks/confidence/cavity values use [0,1]. The layout is directly
    consumable by ordinary 2D convolution operators in ONNX and TensorRT.
    """

    def __init__(self, tensor: Any, *, copy: bool = True):
        value = np.asarray(tensor)
        if value.ndim != 4 or value.shape[1] != UV_DETAIL_CHANNEL_COUNT:
            raise ValueError("UV detail tensor must have shape [B, 12, H_uv, W_uv]")
        if value.shape[0] < 1 or min(value.shape[2:]) < 1 or not np.isfinite(value).all():
            raise ValueError("UV detail tensor must be non-empty and finite")
        self._tensor = np.array(value, dtype=np.float32, order="C", copy=copy)

    @property
    def tensor(self) -> np.ndarray:
        """NCHW tensor; callers may pass it directly to a Conv2d/ONNX input."""
        return self._tensor

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._tensor.shape

    @property
    def uv_size(self) -> tuple[int, int]:
        return self._tensor.shape[2], self._tensor.shape[3]

    @classmethod
    def zeros(cls, batch: int = 1, size: int = 180) -> "UVGaussianMicroDetailMap":
        if batch < 1 or size < 1:
            raise ValueError("batch and size must be positive")
        return cls(np.zeros((batch, UV_DETAIL_CHANNEL_COUNT, size, size), np.float32), copy=False)

    @classmethod
    def from_channels(cls, channels: Mapping[str, Any], *, size: int | tuple[int, int] | None = None,
                       batch: int | None = None) -> "UVGaussianMicroDetailMap":
        """Assemble named HxW or BxHxW planes, resizing with linear sampling."""
        unknown = set(channels) - set(UV_DETAIL_CHANNELS)
        if unknown:
            raise ValueError(f"unknown UV detail channels: {', '.join(sorted(unknown))}")
        prepared: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        if batch is not None and batch < 1:
            raise ValueError("batch must be positive")
        for name, raw in channels.items():
            plane = np.asarray(raw, np.float32)
            if plane.ndim == 2:
                plane = plane[None]
            if plane.ndim != 3 or not np.isfinite(plane).all():
                raise ValueError(f"{name} must be a finite HxW or BxHxW array")
            if shape is None:
                shape = (plane.shape[1], plane.shape[2])
            if plane.shape[1:] != shape:
                raise ValueError("all source UV planes must share their source resolution")
            prepared[name] = plane
        if size is None:
            if shape is None:
                raise ValueError("size is required when no channels are provided")
            out_h, out_w = shape
        elif isinstance(size, int):
            out_h = out_w = size
        else:
            out_h, out_w = map(int, size)
        if min(out_h, out_w) < 1:
            raise ValueError("UV raster dimensions must be positive")
        inferred_batch = int(batch or max((plane.shape[0] for plane in prepared.values()), default=1))
        if any(plane.shape[0] not in (1, inferred_batch) for plane in prepared.values()):
            raise ValueError("channel batch dimensions must be one or the requested batch size")
        output = np.zeros((inferred_batch, UV_DETAIL_CHANNEL_COUNT, out_h, out_w), np.float32)
        for channel, name in enumerate(UV_DETAIL_CHANNELS):
            plane = prepared.get(name)
            if plane is None:
                continue
            if plane.shape[0] == 1 and inferred_batch > 1:
                plane = np.repeat(plane, inferred_batch, axis=0)
            for b in range(inferred_batch):
                source = plane[b]
                if source.shape != (out_h, out_w):
                    source = cv2.resize(source, (out_w, out_h), interpolation=cv2.INTER_AREA if out_h < source.shape[0] else cv2.INTER_LINEAR)
                output[b, channel] = source
        normal = output[:, 3:6]
        normal_length = np.linalg.norm(normal, axis=1, keepdims=True)
        output[:, 3:6] = np.divide(normal, np.maximum(normal_length, 1e-8), out=np.zeros_like(normal))
        output[:, 6:12] = np.clip(output[:, 6:12], 0.0, 1.0)
        return cls(output, copy=False)

    def resized(self, size: int | tuple[int, int]) -> "UVGaussianMicroDetailMap":
        return UVGaussianMicroDetailMap.from_channels(
            {name: self._tensor[:, i] for i, name in enumerate(UV_DETAIL_CHANNELS)}, size=size, batch=self.shape[0])

    def as_torch(self, *, device: str | None = None, fp16: bool = False):
        """Create a torch NCHW tensor lazily; torch remains an optional dependency."""
        import torch
        dtype = torch.float16 if fp16 else torch.float32
        return torch.as_tensor(self._tensor, dtype=dtype, device=device)

    @staticmethod
    def make_trt_conv(*, output_channels: int = 12, kernel_size: int = 3, bias: bool = True):
        """Return a standard ``torch.nn.Conv2d(12, C, k)`` suitable for ONNX/TRT export."""
        import torch.nn as nn
        return nn.Conv2d(UV_DETAIL_CHANNEL_COUNT, output_channels, kernel_size,
                         padding=kernel_size // 2, bias=bias)


@dataclass(frozen=True)
class UVLoDSelection:
    raster_resolution: int
    continuous_resolution: float
    base_resolution: int
    detail_resolution: int
    blend: float
    face_bbox_area_px: float

    def public(self) -> dict[str, Any]:
        return {"raster_resolution": self.raster_resolution,
                "continuous_resolution": round(self.continuous_resolution, 2),
                "base_resolution": self.base_resolution, "detail_resolution": self.detail_resolution,
                "blend": round(self.blend, 4), "face_bbox_area_px": round(self.face_bbox_area_px, 2)}


class ContinuousUVLoD:
    """Select a practical raster size while retaining continuous detail scaling.

    The target size follows the detected face's linear pixel scale and is
    smoothly clamped through 64, 180, and 320 texel anchor points. The returned
    low/high pair and blend preserve continuous LoD information for renderers
    that cross-fade adjacent precomputed UV levels.
    """

    def __init__(self, *, anchors: tuple[tuple[float, int], ...] = ((80.0, 64), (200.0, 180), (420.0, 320)),
                 hysteresis_px: float = 0.0):
        if len(anchors) < 2 or any(w <= 0 or r <= 0 for w, r in anchors):
            raise ValueError("LoD anchors require positive face widths and UV resolutions")
        if any(anchors[i][0] >= anchors[i + 1][0] or anchors[i][1] > anchors[i + 1][1] for i in range(len(anchors) - 1)):
            raise ValueError("LoD anchors must increase in face width and resolution")
        if hysteresis_px < 0:
            raise ValueError("hysteresis_px must be nonnegative")
        self.anchors = tuple((float(width), int(resolution)) for width, resolution in anchors)
        self.hysteresis_px = float(hysteresis_px)
        self._last: UVLoDSelection | None = None

    def select(self, width_px: float, height_px: float) -> UVLoDSelection:
        width, height = float(width_px), float(height_px)
        if not np.isfinite((width, height)).all() or width <= 0 or height <= 0:
            raise ValueError("face bounding box dimensions must be finite and positive")
        face_width = float(np.sqrt(width * height))
        widths = np.asarray([item[0] for item in self.anchors], np.float64)
        resolutions = np.asarray([item[1] for item in self.anchors], np.float64)
        continuous = float(np.interp(face_width, widths, resolutions))
        if face_width <= widths[0]:
            base = detail = int(resolutions[0]); blend = 0.0
        elif face_width >= widths[-1]:
            base = detail = int(resolutions[-1]); blend = 0.0
        else:
            upper = int(np.searchsorted(resolutions, continuous, side="right"))
            upper = min(max(upper, 1), len(resolutions) - 1)
            base, detail = int(resolutions[upper - 1]), int(resolutions[upper])
            blend = float(np.clip((continuous - base) / max(detail - base, 1), 0.0, 1.0))
        # Allocate at the interpolated target size rather than snapping to the
        # adjacent anchor; the integer raster itself then changes smoothly by
        # a texel as the detected face grows. ``base/detail/blend`` is also
        # returned for cross-fading separately cached coarse/high-detail maps.
        raster = int(round(continuous))
        selected = UVLoDSelection(raster, continuous, base, detail, blend, width * height)
        if self._last is not None and self.hysteresis_px:
            boundary = self._last.raster_resolution
            if raster != boundary and abs(face_width - float(np.interp(boundary, resolutions, widths))) < self.hysteresis_px:
                selected = UVLoDSelection(boundary, continuous, base, detail, blend, width * height)
        self._last = selected
        return selected

    def resolution_for_bbox(self, bbox: Any) -> int:
        """Return selected texture size from ``(x0,y0,x1,y1)`` or ``(width,height)``."""
        values = np.asarray(bbox, np.float64).reshape(-1)
        if values.size == 2:
            width, height = values
        elif values.size == 4:
            width, height = values[2] - values[0], values[3] - values[1]
        else:
            raise ValueError("bbox must contain (width,height) or (x0,y0,x1,y1)")
        return self.select(float(width), float(height)).raster_resolution

    def rasterize(self, detail_map: UVGaussianMicroDetailMap, bbox: Any) -> tuple[UVGaussianMicroDetailMap, UVLoDSelection]:
        """Choose an LoD from a target face box and resample the UV tensor to it."""
        values = np.asarray(bbox, np.float64).reshape(-1)
        if values.size == 2:
            width, height = values
        elif values.size == 4:
            width, height = values[2] - values[0], values[3] - values[1]
        else:
            raise ValueError("bbox must contain (width,height) or (x0,y0,x1,y1)")
        selection = self.select(float(width), float(height))
        return detail_map.resized(selection.raster_resolution), selection


__all__ = ["UV_DETAIL_CHANNELS", "UV_DETAIL_CHANNEL_COUNT", "SUPPORTED_UV_SIZES", "UVGaussianMicroDetailMap",
           "UVLoDSelection", "ContinuousUVLoD"]
