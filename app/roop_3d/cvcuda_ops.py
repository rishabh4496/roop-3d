"""Optional CV-CUDA preprocessing, with a device-only Torch reference path.

Affine geometry and resize/normalize/reformat are separate operator launches;
CV-CUDA does not fuse an arbitrary face affine into its resize operator.
"""
from __future__ import annotations

import math

from roop_3d.nv_video_codec import GPUBackendUnavailable, optional_module, configure_gpu_runtime


class FaceOps:
    def __init__(self, backend="auto"):
        if backend not in ("auto", "cvcuda", "torch"):
            raise ValueError("Unknown face preprocessing backend")
        configure_gpu_runtime()
        self.cv, self.fallback_reason = None, None
        if backend != "torch":
            try:
                self.cv = optional_module("cvcuda")
            except GPUBackendUnavailable as error:
                if backend == "cvcuda":
                    raise
                self.fallback_reason = str(error)
        self.backend = "CVCUDA" if self.cv else "TORCH"

    def _stream(self, tensor):
        import torch
        return self.cv.as_stream(torch.cuda.current_stream(tensor.device).cuda_stream)

    def align(self, rgb, destination_to_source, size=(128, 128), *, scale=1 / 255., offset=0.):
        """RGB HWC uint8 -> face batch NCHW float32; transforms use pixel units."""
        import torch
        if (not rgb.is_cuda or rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
                or destination_to_source.device != rgb.device or destination_to_source.dtype != torch.float32
                or destination_to_source.ndim != 3 or destination_to_source.shape[1:] != (2, 3)
                or len(size) != 2 or min(size) < 1 or not math.isfinite(scale) or not math.isfinite(offset)):
            raise ValueError("Invalid CUDA RGB/affine preprocessing contract")
        count = destination_to_source.shape[0]
        if not count:
            return torch.empty((0, 3, *size), device=rgb.device, dtype=torch.float32)
        if self.cv:
            cv = self.cv
            stream = self._stream(rgb)
            source = cv.ImageBatchVarShape(count)
            image = cv.as_image(rgb, cv.Format.RGB8)
            source.pushback([image for _ in range(count)])
            crops = cv.ImageBatchVarShape(count)
            crops.pushback([cv.Image((size[1], size[0]), cv.Format.RGB8) for _ in range(count)])
            # Var-shape WarpAffine takes N x 6 (NC), not N x 2 x 3.
            matrices = cv.as_tensor(destination_to_source.reshape(count, 6).contiguous(), "NC")
            cv.warp_affine_into(crops, source, matrices,
                flags=cv.Interp.LINEAR | cv.Interp.WARP_INVERSE_MAP, stream=stream)
            normalized = cv.resize_crop_convert_reformat(crops, (size[1], size[0]), cv.Interp.LINEAR,
                (0, 0, size[1], size[0]), layout="NCHW", data_type=cv.Type.F32,
                scale=scale, offset=offset, srcCast=False, stream=stream)
            # Export using the same stream; Tensor owner is retained by DLPack.
            return torch.as_tensor(normalized.cuda(), device=rgb.device)
        from roop_3d.gpu_compositor import warp
        batch = rgb.permute(2, 0, 1).unsqueeze(0).float().expand(count, -1, -1, -1)
        return warp(batch, destination_to_source, size).round().clamp(0, 255) * scale + offset

    def _op(self, name, image, *args, **kwargs):
        import torch
        if not image.is_cuda or image.ndim != 4 or image.dtype != torch.float32:
            raise ValueError("Filters require CUDA NCHW float32")
        nhwc = image.permute(0, 2, 3, 1).contiguous()
        result = getattr(self.cv, name)(self.cv.as_tensor(nhwc, "NHWC"), *args,
            stream=self._stream(image), **kwargs)
        return torch.as_tensor(result.cuda(), device=image.device).permute(0, 3, 1, 2)

    def gaussian(self, image, sigma=2.):
        from roop_3d.gpu_compositor import gaussian
        if not math.isfinite(sigma) or not 0 <= sigma <= 10:
            raise ValueError("Sigma must be in 0..10")
        if not self.cv or sigma == 0:
            return gaussian(image, sigma)
        kernel = 2 * math.ceil(3 * sigma) + 1
        return self._op("gaussian", image, (kernel, kernel), (sigma, sigma), border=self.cv.Border.REPLICATE)

    def morphology(self, mask, radius=1):
        from roop_3d.gpu_compositor import morphology
        if type(radius) is not int or abs(radius) > 31:
            raise ValueError("Radius must be an integer in -31..31")
        if not self.cv or radius == 0:
            return morphology(mask, radius)
        return self._op("morphology", mask,
            self.cv.MorphologyType.ERODE if radius > 0 else self.cv.MorphologyType.DILATE,
            (2 * abs(radius) + 1,) * 2, (-1, -1), border=self.cv.Border.REPLICATE)

    def laplacian(self, image):
        if self.cv:
            return self._op("laplacian", image, 1, border=self.cv.Border.REPLICATE)
        import torch.nn.functional as F
        channels = image.shape[1]
        kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return F.conv2d(F.pad(image, (1,) * 4, mode="replicate"),
            kernel[None, None].expand(channels, 1, 3, 3), groups=channels)

    def smooth_skin(self, image, guidance, *, diameter=5, sigma_color=.1, sigma_space=2.):
        import torch
        import torch.nn.functional as F
        if (image.shape != guidance.shape or image.device != guidance.device
                or diameter < 1 or diameter > 15 or diameter % 2 != 1
                or not 0 < sigma_color <= 10 or not 0 < sigma_space <= 10):
            raise ValueError("Invalid joint bilateral filter contract")
        if self.cv:
            guide = self.cv.as_tensor(guidance.permute(0, 2, 3, 1).contiguous(), "NHWC")
            return self._op("joint_bilateral_filter", image, guide, diameter, sigma_color, sigma_space,
                border=self.cv.Border.REPLICATE)
        radius = diameter // 2
        padded = F.pad(image, (radius,) * 4, mode="replicate")
        gp = F.pad(guidance, (radius,) * 4, mode="replicate")
        numerator, denominator = torch.zeros_like(image), torch.zeros_like(image[:, :1])
        height, width = image.shape[-2:]
        for y in range(diameter):
            for x in range(diameter):
                difference = (gp[:, :, y:y + height, x:x + width] - guidance).abs().sum(1, keepdim=True)
                weight = torch.exp(-difference.square() / (2 * sigma_color**2)
                    - ((x - radius)**2 + (y - radius)**2) / (2 * sigma_space**2))
                numerator += padded[:, :, y:y + height, x:x + width] * weight
                denominator += weight
        return numerator / denominator.clamp_min(1e-12)

    def feather(self, alpha, visibility, *, radius=1):
        """Three spatial bands; reapply current visibility to preserve occluders."""
        eroded = self.morphology(alpha.clamp(0, 1), radius)
        bands = [self.gaussian(eroded, sigma) for sigma in (.8, 2., 4.)]
        return (sum(bands) / 3).clamp(0, 1) * visibility.clamp(0, 1)
