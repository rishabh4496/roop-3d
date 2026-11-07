"""CUDA-resident image transforms shared by the streaming inference stages.

Frames enter as pinned HWC BGR uint8 and leave as pinned HWC BGR uint8. Crops,
model normalization, affine warps and mask feathering remain device tensors.
"""
from __future__ import annotations

import math


def warp_affine_gpu(image, destination_to_source, size):
    """Warp NCHW tensors with pixel-coordinate affine matrices (CUDA in production)."""
    import torch
    import torch.nn.functional as functional

    if (image.ndim != 4 or destination_to_source.device != image.device
            or destination_to_source.shape != (image.shape[0], 2, 3)):
        raise ValueError("Affine warp requires matching batched image/matrix tensors")
    height, width = size
    if height < 1 or width < 1:
        raise ValueError("Affine output dimensions must be positive")
    y, x = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                          torch.arange(width, device=image.device, dtype=image.dtype), indexing="ij")

    def coordinate(row):
        return (x[None] * destination_to_source[:, row, 0, None, None]
                + y[None] * destination_to_source[:, row, 1, None, None]
                + destination_to_source[:, row, 2, None, None])

    grid = torch.stack((coordinate(0), coordinate(1)), -1)
    scale = image.new_tensor((max(image.shape[-1] - 1, 1), max(image.shape[-2] - 1, 1)))
    grid = grid * (2 / scale) - 1
    return functional.grid_sample(image, grid, align_corners=True, padding_mode="zeros")


def normalize_model_input(rgb01, *, channel_order="RGB", value_range="minus_one_one"):
    """Bridge an RGB [0,1] CUDA image into a model's channel/range contract."""
    if not getattr(rgb01, "is_cuda", False) or rgb01.ndim != 4 or rgb01.shape[1] != 3:
        raise ValueError("Model normalization requires a CUDA NCHW RGB tensor")
    value = rgb01.float()
    if channel_order == "BGR":
        value = value[:, [2, 1, 0]]
    elif channel_order != "RGB":
        raise ValueError("Channel order must be RGB or BGR")
    if value_range == "minus_one_one":
        return value.mul(2).sub(1)
    if value_range == "zero_one":
        return value
    raise ValueError("Value range must be zero_one or minus_one_one")


def model_output_to_rgb01(value, *, value_range="minus_one_one", channel_order="RGB"):
    """Convert model output to FP32 RGB [0,1] without leaving CUDA."""
    if not getattr(value, "is_cuda", False) or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("Model output conversion requires a CUDA NCHW image tensor")
    result = value.float()
    if value_range == "minus_one_one":
        result = (result + 1) * .5
    elif value_range != "zero_one":
        raise ValueError("Value range must be zero_one or minus_one_one")
    if channel_order == "BGR":
        result = result[:, [2, 1, 0]]
    elif channel_order != "RGB":
        raise ValueError("Channel order must be RGB or BGR")
    return result.clamp(0, 1)


def gaussian_feather_mask(mask, sigma: float = 2.0):
    """Apply a separable Gaussian feather to an NCHW mask/image tensor."""
    import torch.nn.functional as functional

    if mask.ndim != 4:
        raise ValueError("Gaussian feathering requires an NCHW tensor")
    if not math.isfinite(sigma) or not 0 <= sigma <= 64:
        raise ValueError("Gaussian sigma must be in 0..64")
    if sigma == 0:
        return mask
    radius = math.ceil(3 * sigma)
    x = mask.new_tensor(list(range(-radius, radius + 1)))
    kernel = (-x.square() / (2 * sigma * sigma)).exp()
    kernel = kernel / kernel.sum()
    channels = mask.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    value = functional.conv2d(functional.pad(mask, (radius, radius, 0, 0), mode="replicate"),
                              horizontal, groups=channels)
    return functional.conv2d(functional.pad(value, (0, 0, radius, radius), mode="replicate"),
                             vertical, groups=channels)
