"""Target-referred optical blur, defocus and sensor grain for the final face composite.

The compositor still exchanges BGR frames with OpenCV, but all finishing kernels run on CUDA when
available. The only host transfers are the ROI upload and the finished ROI download. Dense flow from
the temporal detail stage is kept on the same device and used as a spatially varying shutter PSF.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC if image.ndim == 3 else cv2.INTER_LINEAR)


def _gaussian_kernel(torch, sigma: float, device):
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian(torch, functional, image, sigma: float):
    if sigma <= 0:
        return image
    kernel = _gaussian_kernel(torch, sigma, image.device)
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    radius = kernel.numel() // 2
    result = functional.conv2d(functional.pad(image, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return functional.conv2d(functional.pad(result, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def _normalised_gaussian(torch, functional, image, weight, sigma: float):
    denominator = _gaussian(torch, functional, weight, sigma).clamp_min(1e-5)
    return _gaussian(torch, functional, image * weight, sigma) / denominator


def _masked_energy(torch, image, weight) -> float:
    kernel = torch.tensor(((0, 1, 0), (1, -4, 1), (0, 1, 0)), device=image.device,
                          dtype=torch.float32).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect"), kernel)
    denom = weight.sum().clamp_min(1)
    return float(((lap.square() * weight).sum() / denom).item())


def _decode_transfer(torch, image, transfer: str):
    code = image.clamp(0, 1)
    if transfer == "linear":
        return code
    if transfer == "srgb":
        return torch.where(code <= .04045, code / 12.92, ((code + .055) / 1.055).pow(2.4))
    if transfer in ("bt1886", "gamma28", "gamma22"):
        gamma = {"bt1886": 2.4, "gamma28": 2.8, "gamma22": 2.2}[transfer]
        return code.pow(gamma)
    # Inverse BT.709 OETF. This keeps blur and Laplacian matching energy in light units.
    return torch.where(code < .081, code / 4.5, ((code + .099) / 1.099).pow(1 / .45))


def _encode_transfer(torch, image, transfer: str):
    linear = image.clamp(0, 1)
    if transfer == "linear":
        return linear
    if transfer == "srgb":
        return torch.where(linear <= .0031308, linear * 12.92, 1.055 * linear.pow(1 / 2.4) - .055)
    if transfer in ("bt1886", "gamma28", "gamma22"):
        gamma = {"bt1886": 2.4, "gamma28": 2.8, "gamma22": 2.2}[transfer]
        return linear.pow(1 / gamma)
    return torch.where(linear < .018, linear * 4.5, 1.099 * linear.pow(.45) - .099)


def _flow_tensor(torch, flow, *, device, height: int, width: int):
    if flow is None:
        return None
    if isinstance(flow, torch.Tensor):
        value = flow.to(device=device, dtype=torch.float32)
    else:
        value = torch.as_tensor(np.asarray(flow), device=device, dtype=torch.float32)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        return None
    if value.shape[-1] == 2:
        value = value.permute(2, 0, 1)[None]
    elif value.shape[0] == 2:
        value = value[None]
    else:
        return None
    old_h, old_w = value.shape[-2:]
    value = torch.nn.functional.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
    value[:, 0] *= width / max(old_w, 1)
    value[:, 1] *= height / max(old_h, 1)
    return value


def _bilateral_luma(torch, functional, luma, guide, mask):
    """5x5 CUDA bilateral filter used only to estimate the target's sensor residual."""
    radius, spatial_sigma, range_sigma = 2, 1.25, 0.06
    padded_luma = functional.pad(luma, (radius, radius, radius, radius), mode="reflect")
    padded_guide = functional.pad(guide, (radius, radius, radius, radius), mode="reflect")
    patches = functional.unfold(padded_luma, 5).view(1, 25, *luma.shape[-2:])
    guide_patches = patches if guide is luma else functional.unfold(padded_guide, 5).view(1, 25, *luma.shape[-2:])
    offsets = torch.arange(-radius, radius + 1, device=luma.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    spatial = torch.exp(-(xx.square() + yy.square()).reshape(1, 25, 1, 1) / (2 * spatial_sigma ** 2))
    range_weight = torch.exp(-((guide_patches - guide) ** 2) / (2 * range_sigma ** 2))
    weights = spatial * range_weight
    filtered = (patches * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1e-8)
    residual = (luma - filtered) * mask
    return residual


def _gpu_finish(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, skin: np.ndarray | None, flow,
                crop_motion: tuple[float, float], shutter_angle: float, grain_strength: float, seed: int,
                match_optics: bool, transfer: str) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as functional

    device = flow.device if isinstance(flow, torch.Tensor) and flow.is_cuda else torch.device("cuda", torch.cuda.current_device())
    height, width = source.shape[:2]
    target = _resize(target, (height, width))
    alpha = _resize(alpha.astype(np.float32), (height, width))
    if skin is None:
        skin = alpha
    else:
        skin = _resize(skin.astype(np.float32), (height, width))
    src_code = torch.as_tensor(np.ascontiguousarray(source), device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    tgt_code = torch.as_tensor(np.ascontiguousarray(target), device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    src = _decode_transfer(torch, src_code, transfer)
    tgt = _decode_transfer(torch, tgt_code, transfer)
    weight = torch.as_tensor(np.ascontiguousarray(np.clip(alpha, 0, 1)), device=device)[None, None]
    grain_weight = weight * torch.as_tensor(np.ascontiguousarray(np.clip(skin, 0, 1)), device=device)[None, None]
    report = {"backend": "cuda", "motion_magnitude_px": 0.0, "motion_length_px": 0.0,
              "motion_angle_degrees": 0.0, "frequency_energy_ratio": 1.0, "defocus_sigma_px": 0.0,
              "grain_size_px": 0.0, "grain_luma_std": 0.0, "grain_chroma_std": 0.0}
    if int((weight > .2).sum().item()) < 64:
        return source.copy(), report

    # The temporal stage exposes current-frame displacement vectors (the negated current->previous flow).
    vector = _flow_tensor(torch, flow, device=device, height=height, width=width)
    if vector is None:
        vector = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
    vector[:, 0] += float(crop_motion[0])
    vector[:, 1] += float(crop_motion[1])
    shutter = float(shutter_angle) / 360.0
    mean_x = float(((vector[:, 0:1] * weight).sum() / weight.sum().clamp_min(1)).item())
    mean_y = float(((vector[:, 1:2] * weight).sum() / weight.sum().clamp_min(1)).item())
    report["motion_magnitude_px"] = math.hypot(mean_x, mean_y)
    report["motion_length_px"] = min(math.hypot(mean_x, mean_y) * shutter, 15.0)
    report["motion_angle_degrees"] = math.degrees(math.atan2(mean_y, mean_x)) if mean_x or mean_y else 0.0

    src_luma = src[:, 0:1] * .114 + src[:, 1:2] * .587 + src[:, 2:3] * .299
    tgt_luma = tgt[:, 0:1] * .114 + tgt[:, 1:2] * .587 + tgt[:, 2:3] * .299
    target_energy = _masked_energy(torch, tgt_luma, weight)
    source_energy = _masked_energy(torch, src_luma, weight)
    report["frequency_energy_ratio"] = target_energy / max(source_energy, 1e-8)
    working = src
    report["optics_state"] = "MEASURED" if match_optics else "DISABLED"
    if match_optics and target_energy <= 1e-10:
        # A constant patch contains no evidence for blur. Its zero spectrum is
        # not a command to remove all detail restored by UltraMax/GPEN.
        match_optics = False
        report.update(optics_state="UNAVAILABLE", optics_reason="TARGET_OPTICS_UNOBSERVABLE")

    # Local directional PSF: each pixel samples along its own dense flow vector, preserving rotational flow.
    if match_optics and shutter > 0 and source_energy > target_energy * 1.02:
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        magnitude = torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-6)
        scale = (magnitude * shutter).clamp(max=15.0) / magnitude
        vector = vector * scale
        blurred = torch.zeros_like(working)
        denom = torch.zeros_like(weight)
        for offset in torch.linspace(-.5, .5, 9, device=device):
            map_x, map_y = xx + vector[:, 0] * offset, yy + vector[:, 1] * offset
            grid = torch.stack((map_x * (2.0 / max(width - 1, 1)) - 1.0,
                                map_y * (2.0 / max(height - 1, 1)) - 1.0), dim=-1)
            sample = functional.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
            sample_weight = functional.grid_sample(weight, grid, mode="bilinear", padding_mode="border", align_corners=True)
            blurred += sample * sample_weight
            denom += sample_weight
        blurred /= denom.clamp_min(1e-5)
        blurred_luma = blurred[:, 0:1] * .114 + blurred[:, 1:2] * .587 + blurred[:, 2:3] * .299
        blur_energy = _masked_energy(torch, blurred_luma, weight)
        if source_energy > blur_energy + 1e-8:
            amount = min(1.0, max(0.0, (source_energy - target_energy) / (source_energy - blur_energy)))
            working = src + (blurred - src) * amount

    # Match optical defocus by choosing the Gaussian PSF with the closest target Laplacian energy.
    if match_optics:
        current_luma = working[:, 0:1] * .114 + working[:, 1:2] * .587 + working[:, 2:3] * .299
        current_energy = _masked_energy(torch, current_luma, weight)
        if target_energy < current_energy * .98:
            best_sigma, best_error, best_blur = 0.0, abs(current_energy - target_energy), working
            for sigma in (.45, .7, 1.0, 1.35, 1.75, 2.25, 2.8):
                candidate = _normalised_gaussian(torch, functional, working, weight, sigma)
                candidate_luma = candidate[:, 0:1] * .114 + candidate[:, 1:2] * .587 + candidate[:, 2:3] * .299
                candidate_energy = _masked_energy(torch, candidate_luma, weight)
                error = abs(candidate_energy - target_energy)
                if error < best_error:
                    best_sigma, best_error, best_blur = sigma, error, candidate
            working = best_blur
            report["defocus_sigma_px"] = best_sigma

    # Estimate luminance grain with a bilateral residual; estimate chroma residuals in opponent channels.
    if grain_strength > 0:
        working_code = _encode_transfer(torch, working, transfer)
        tgt_code_luma = tgt_code[:, 0:1] * .114 + tgt_code[:, 1:2] * .587 + tgt_code[:, 2:3] * .299
        residual_y = _bilateral_luma(torch, functional, tgt_code_luma, tgt_code_luma, grain_weight)
        mask_sum = grain_weight.sum().clamp_min(1)
        variance_y = ((residual_y.square() * grain_weight).sum() / mask_sum).clamp_min(0)
        std_y = variance_y.sqrt()
        report["grain_luma_std"] = float((std_y * 255).item())
        if std_y.item() > 1e-5:
            shifted_x = torch.roll(residual_y, 1, dims=-1)
            shifted_y = torch.roll(residual_y, 1, dims=-2)
            corr_x = ((residual_y * shifted_x * grain_weight).sum() / mask_sum) / variance_y.clamp_min(1e-8)
            corr_y = ((residual_y * shifted_y * grain_weight).sum() / mask_sum) / variance_y.clamp_min(1e-8)
            grain_sigma = min(1.5, max(.5, .65 + .5 * float(torch.clamp((corr_x + corr_y) * .5, -0.25, .9).item())))
            report["grain_size_px"] = grain_sigma
            residual_code_var = functional.avg_pool2d(residual_y.square() * grain_weight, 9, stride=1, padding=4) / \
                functional.avg_pool2d(grain_weight, 9, stride=1, padding=4).clamp_min(1e-6)
            variance_code = residual_code_var * (255.0 ** 2)
            source_luma_code = (working_code[:, 0:1] * .114 + working_code[:, 1:2] * .587 + working_code[:, 2:3] * .299) * 255.0
            poisson_gain = (variance_code / source_luma_code.clamp_min(1.0)).clamp(.002, 8.0)
            rate = (source_luma_code / poisson_gain).clamp(0, 1e6)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) % (2**63 - 1))
            poisson = (torch.poisson(rate, generator=generator) * poisson_gain - source_luma_code) / 255.0
            poisson_variance = source_luma_code * poisson_gain
            gaussian_std = (variance_code - poisson_variance).clamp_min(0).sqrt() / 255.0
            gaussian = torch.randn((1, 1, height, width), device=device, dtype=torch.float32, generator=generator) * gaussian_std
            luma_noise = _gaussian(torch, functional, poisson + gaussian, grain_sigma)

            target_cb = tgt_code[:, 0:1] - tgt_code_luma
            target_cr = tgt_code[:, 2:3] - tgt_code_luma
            residual_cb = target_cb - _gaussian(torch, functional, target_cb, 1.0)
            residual_cr = target_cr - _gaussian(torch, functional, target_cr, 1.0)
            std_cb = ((residual_cb.square() * grain_weight).sum() / mask_sum).clamp_min(0).sqrt()
            std_cr = ((residual_cr.square() * grain_weight).sum() / mask_sum).clamp_min(0).sqrt()
            report["grain_chroma_std"] = float((((std_cb + std_cr) * .5) * 255).item())
            chroma_noise = torch.randn((1, 2, height, width), device=device, dtype=torch.float32, generator=generator)
            chroma_noise[:, 0:1] *= std_cb
            chroma_noise[:, 1:2] *= std_cr
            chroma_noise = _gaussian(torch, functional, chroma_noise, grain_sigma)
            b_noise = luma_noise + chroma_noise[:, 0:1]
            r_noise = luma_noise + chroma_noise[:, 1:2]
            g_noise = luma_noise - (.114 * chroma_noise[:, 0:1] + .299 * chroma_noise[:, 1:2]) / .587
            noise = torch.cat((b_noise, g_noise, r_noise), dim=1)
            source_code_luma = working_code[:, 0:1] * .114 + working_code[:, 1:2] * .587 + working_code[:, 2:3] * .299
            shadows = (1.15 - source_code_luma).clamp(.2, 1.0)
            active = grain_weight * shadows * (float(grain_strength) / 100.0)
            working_code = working_code + noise * active
        else:
            working_code = _encode_transfer(torch, working, transfer)
    else:
        working_code = _encode_transfer(torch, working, transfer)

    output = (working_code.clamp(0, 1)[0].permute(1, 2, 0).contiguous() * 255.0).round().byte().cpu().numpy()
    return output, report


def finish_target_sensor(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, skin: np.ndarray | None = None,
                         *, flow=None, crop_motion: tuple[float, float] = (0.0, 0.0), shutter_angle: float = 180.0,
                         grain_strength: float = 70.0, seed: int = 0, match_optics: bool = True,
                         transfer: str = "srgb") -> tuple[np.ndarray, dict]:
    """Match target blur, defocus and grain immediately before final alpha compositing."""
    if (source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3
            or target.dtype != np.uint8 or target.ndim != 3 or target.shape[2] != 3
            or alpha.ndim != 2 or not np.isfinite(alpha).all()
            or not 0 <= shutter_angle <= 360 or not 0 <= grain_strength <= 100):
        raise ValueError("Invalid target sensor finishing inputs")
    try:
        import torch
        if torch.cuda.is_available():
            return _gpu_finish(source, target, alpha, skin, flow, crop_motion, shutter_angle,
                               grain_strength, seed, match_optics, transfer)
    except (ImportError, RuntimeError, ValueError, OSError):
        pass

    # CPU fallback keeps the render usable on machines without CUDA. CUDA runs the full sensor model above.
    from roop_3d.compositor.quality import match_sharpness, transfer_fine_texture
    target = _resize(target, source.shape[:2])
    alpha = _resize(alpha.astype(np.float32), source.shape[:2])
    footprint = np.ones(alpha.shape, np.float32)
    image, sharpness = (match_sharpness(source, target, alpha, footprint, motion_px=crop_motion,
                                        shutter=shutter_angle / 360.0) if match_optics else (source.copy(), {}))
    skin = alpha if skin is None else _resize(skin.astype(np.float32), source.shape[:2])
    image, texture_gain = transfer_fine_texture(image, target, alpha, skin, footprint,
                                                strength=grain_strength / 100.0)
    report = {"backend": "cpu_fallback", "motion_length_px": sharpness.get("motion_length", 0.0),
              "defocus_sigma_px": sharpness.get("sigma", 0.0), "grain_luma_std": 0.0,
              "grain_chroma_std": 0.0, "texture_gain": texture_gain,
              "optics_state": "MEASURED" if match_optics else "DISABLED"}
    if sharpness.get("optics_observable") == 0.0:
        report.update(optics_state="UNAVAILABLE", optics_reason="TARGET_OPTICS_UNOBSERVABLE")
    return image, report
