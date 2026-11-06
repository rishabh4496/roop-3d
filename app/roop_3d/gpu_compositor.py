"""Optional Torch CUDA compositor. Images are NCHW RGB float32 in [0, 1].

No models are loaded at import time. Semantic skin and visibility must come from
an explicit device adapter; a geometric mask is never presented as skin parsing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def _torch():
    import torch
    return torch


def warp(image, destination_to_source, size):
    """Pixel-coordinate affine sampling, zero outside the source image."""
    torch = _torch()
    import torch.nn.functional as F
    height, width = size
    y, x = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                          torch.arange(width, device=image.device, dtype=image.dtype), indexing="ij")
    # Tiny coordinate transforms do not need GEMM. Using pointwise math avoids
    # cuBLAS retaining an otherwise unowned workspace for every worker stream.
    def coordinate(row):
        return (x[None] * destination_to_source[:, row, 0, None, None]
                + y[None] * destination_to_source[:, row, 1, None, None]
                + destination_to_source[:, row, 2, None, None])
    grid = torch.stack((coordinate(0), coordinate(1)), -1)
    scale = image.new_tensor((max(image.shape[-1] - 1, 1), max(image.shape[-2] - 1, 1)))
    grid = grid * (2 / scale) - 1
    return F.grid_sample(image, grid, align_corners=True, padding_mode="zeros")


def invert_affine(matrix):
    torch = _torch()
    # torch.linalg.inv checks solver status via a device synchronization. The
    # closed-form affine inverse stays entirely on the owning compute stream.
    a, b, c = matrix[:, 0].unbind(-1)
    d, e, f = matrix[:, 1].unbind(-1)
    det = a * e - b * d
    return torch.stack((torch.stack((e, -b, b * f - e * c), -1),
                        torch.stack((-d, a, d * c - a * f), -1)), 1) / det[:, None, None]


def gaussian(image, sigma=2.0):
    torch = _torch()
    import torch.nn.functional as F
    if not math.isfinite(sigma) or not 0 <= sigma <= 64:
        raise ValueError("Gaussian sigma must be in 0..64")
    if sigma == 0:
        return image
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = (-x.square() / (2 * sigma * sigma)).exp()
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    value = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                     kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(F.pad(value, (0, 0, radius, radius), mode="replicate"),
                    kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def morphology(mask, radius):
    """Positive radius erodes; negative radius dilates (square support)."""
    import torch.nn.functional as F
    if type(radius) is not int or abs(radius) > 64:
        raise ValueError("Morphology radius must be an integer in -64..64")
    if radius == 0:
        return mask
    value = -mask if radius > 0 else mask
    value = F.max_pool2d(value, 2 * abs(radius) + 1, stride=1, padding=abs(radius))
    return -value if radius > 0 else value


def face_feather(like, size):
    torch = _torch()
    y, x = torch.meshgrid(torch.arange(size, device=like.device, dtype=like.dtype),
                          torch.arange(size, device=like.device, dtype=like.dtype), indexing="ij")
    return ((1 - ((x - (size - 1) / 2) / (size * .46)).square()
             - ((y - (size - 1) / 2) / (size * .48)).square()) * 8).clamp(0, 1)[None, None]


def soft_matte(alpha, visibility, *, erosion=1, sigma=2., jaw_falloff=.08):
    """Feather then reapply visibility so blur cannot leak onto foreground objects."""
    torch = _torch()
    if not 0 <= jaw_falloff <= 1:
        raise ValueError("Jaw falloff must be in 0..1")
    result = gaussian(morphology(alpha.clamp(0, 1), erosion), sigma).clamp(0, 1)
    if jaw_falloff:
        y = torch.linspace(0, 1, result.shape[-2], device=result.device, dtype=result.dtype)
        falloff = ((1 - y) / jaw_falloff).clamp(0, 1)
        result = result * (falloff.square() * (3 - 2 * falloff))[None, None, :, None]
    return result * visibility.clamp(0, 1)


def _mix_channels(image, matrix):
    """Three-channel pointwise transform without a persistent cuBLAS arena."""
    return _torch().stack(tuple(image[:, 0] * row[0] + image[:, 1] * row[1] + image[:, 2] * row[2]
                                for row in matrix), 1)


def rgb_to_lab(rgb):
    torch = _torch()
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).clamp_min(0).pow(2.4), rgb / 12.92)
    matrix = ((.4124564, .3575761, .1804375),
              (.2126729, .7151522, .0721750), (.0193339, .1191920, .9503041))
    xyz = _mix_channels(linear, matrix) / rgb.new_tensor((.95047, 1., 1.08883))[None, :, None, None]
    f = torch.where(xyz > (6 / 29) ** 3, xyz.clamp_min(0).pow(1 / 3), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return torch.stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])), 1)


def lab_to_rgb(lab):
    torch = _torch()
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack((fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200), 1)
    xyz = torch.where(f > 6 / 29, f.pow(3), 3 * (6 / 29) ** 2 * (f - 4 / 29))
    xyz = xyz * lab.new_tensor((.95047, 1., 1.08883))[None, :, None, None]
    matrix = ((3.2404542, -1.5371385, -.4985314),
              (-.9692660, 1.8760108, .0415560), (.0556434, -.2040259, 1.0572252))
    linear = _mix_channels(xyz, matrix)
    return torch.where(linear > .0031308, 1.055 * linear.clamp_min(0).pow(1 / 2.4) - .055, 12.92 * linear).clamp(0, 1)


def color_transfer(source, target, inner_skin, *, strength=1., regional=False):
    """Weighted LAB Reinhard; regional=True uses spatial Gaussian moments (RCT).

    Both statistics and writes exclude eyes, teeth and occluders, as supplied by
    the semantic adapter. Empty regions are identity operations, without a host sync.
    """
    torch = _torch()
    if not 0 <= strength <= 1:
        raise ValueError("Color strength must be in 0..1")
    weight = inner_skin.clamp(0, 1)
    a, b = rgb_to_lab(source), rgb_to_lab(target)
    reduce = (lambda v: gaussian(v, 8.)) if regional else (lambda v: v.sum((-2, -1), keepdim=True))
    count = reduce(weight)
    def stats(value):
        mean = reduce(value * weight) / count.clamp_min(1e-6)
        variance = (reduce(value.square() * weight) / count.clamp_min(1e-6) - mean.square()).clamp_min(0)
        return mean, variance.clamp_min(1.).sqrt()
    am, ast = stats(a)
    bm, bst = stats(b)
    matched = lab_to_rgb((a - am) * bst / ast + bm)
    active = (weight > 0) & (count > 1e-6)
    return torch.where(active, source + (matched - source) * strength * weight, source)


def laplacian_blend(source, target, alpha, *, laplacian_blend_weight=1., levels=3):
    """Three frequency bands. The coarsest illumination remains target-owned."""
    import torch.nn.functional as F
    if levels != 3 or not 0 <= laplacian_blend_weight <= 1:
        raise ValueError("Expected three levels and a blend weight in 0..1")
    if min(source.shape[-2:]) < 4:
        raise ValueError("Pyramid requires at least 4x4 pixels")
    s, t, m = [source], [target], [alpha]
    for _ in range(levels - 1):
        for stack in (s, t, m):
            stack.append(gaussian(stack[-1], 1.)[..., ::2, ::2])
    def up(value, reference):
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    result = t[-1]
    for i in range(levels - 2, -1, -1):
        sl, tl = s[i] - up(s[i + 1], s[i]), t[i] - up(t[i + 1], t[i])
        mask = m[i] * laplacian_blend_weight
        result = up(result, s[i]) + sl * mask + tl * (1 - mask)
    # Preserve exact zero-alpha support, including protected semantic regions.
    return _torch().where(alpha > 0, result.clamp(0, 1), target)


class SemanticMaskAdapter:
    """Translate explicitly documented parser logits to hard semantic exclusions.

    There are deliberately no default class IDs or model normalization guesses.
    The supplied inference callable owns preprocessing and device I/O binding.
    """
    def __init__(self, infer, *, face_ids, skin_ids, protected_ids, occluder_ids, evidence):
        groups = tuple(tuple(v) for v in (face_ids, skin_ids, protected_ids, occluder_ids))
        if not evidence or not groups[0] or not groups[1] or not groups[2] or not groups[3]:
            raise ValueError('Semantic adapter requires explicit labels and contract evidence')
        if any(type(v) is not int or v < 0 for group in groups for v in group):
            raise ValueError('Semantic class IDs must be nonnegative integers')
        if set(groups[1]) & (set(groups[2]) | set(groups[3])):
            raise ValueError('Skin classes cannot include eyes, teeth or occluders')
        self.infer, self.groups, self.evidence = infer, groups, evidence

    def __call__(self, target, face):
        torch = _torch()
        import torch.nn.functional as F
        logits = self.infer(target, face)
        if (not logits.is_cuda or logits.device != target.device or logits.ndim != 4 or logits.shape[0] != 1
                or logits.shape[1] <= max(v for group in self.groups for v in group)):
            raise ValueError('Semantic logits do not match their declared class contract')
        labels = F.interpolate(logits.float(), (512, 512), mode='bilinear', align_corners=False).argmax(1, keepdim=True)
        def selected(ids):
            values = torch.as_tensor(ids, device=labels.device).view(1, -1, 1, 1)
            return (labels == values).any(1, keepdim=True).float()
        face_ids, skin_ids, protected_ids, occluder_ids = self.groups
        visible = 1 - selected(occluder_ids)
        inner = selected(skin_ids) * (1 - selected(protected_ids)) * visible
        return selected(face_ids) * visible, inner, visible


class TorchIOBinding:
    """Device-pointer binding. Session must be built on the worker compute stream."""
    def __init__(self, session, stream):
        providers = session.get_providers()
        if providers[0] not in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            raise ValueError("Device inference requires an activated CUDA/TRT provider")
        options = session.get_provider_options()
        for provider in providers:
            if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                if options[provider].get("user_compute_stream") != str(stream.cuda_stream):
                    raise ValueError("Inference session must use the worker compute stream")
        self.session, self.stream = session, stream

    def run(self, feeds, outputs):
        import onnxruntime as ort
        torch = _torch()
        if torch.cuda.current_stream() != self.stream:
            raise ValueError("I/O binding invoked outside its owning compute stream")
        binding = self.session.io_binding()
        for name, value in feeds.items():
            if not value.is_cuda or not value.is_contiguous():
                raise ValueError("Device inputs must be contiguous CUDA tensors")
            binding.bind_input(name, "cuda", value.device.index, np.dtype(str(value.dtype).split('.')[-1]),
                               tuple(value.shape), value.data_ptr())
        result = {}
        for name, (shape, dtype) in outputs.items():
            value = torch.empty(shape, dtype=dtype, device=self.stream.device)
            binding.bind_output(name, "cuda", value.device.index, np.dtype(str(dtype).split('.')[-1]), shape, value.data_ptr())
            result[name] = value
        options = ort.RunOptions()
        options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        self.session.run_with_iobinding(binding, options)
        # Caller keeps inputs/outputs alive until its completion event, including
        # intermediate temporary tensors allocated on the same Torch stream.
        return result


@dataclass(frozen=True)
class CompositorOptions:
    color_strength: float = 0.
    color_method: str = 'reinhard_lab'
    regional_color: bool = False
    laplacian_blend_weight: float = 0.
    erosion: int = 0
    feather_sigma: float = 0.
    jaw_falloff: float = 0.

    def __post_init__(self):
        if (self.color_method not in ('reinhard_lab', 'regional_reinhard')
                or not 0 <= self.color_strength <= 1 or not 0 <= self.laplacian_blend_weight <= 1
                or type(self.erosion) is not int or abs(self.erosion) > 64
                or not 0 <= self.feather_sigma <= 64 or not 0 <= self.jaw_falloff <= 1):
            raise ValueError("Invalid GPU compositor options")


class GpuCompositor:
    """One upload boundary, device-only per-face chain, one download boundary.

    swapper(crop128, face), restorer(crop512, face), masker(target512, face)
    execute on the current stream. Masker returns (alpha, inner_skin, visibility).
    Model-specific normalization, alignment and verified labels belong to adapters.
    """
    def __init__(self, swapper, *, restorer=None, masker=None, options=CompositorOptions()):
        if options.color_strength and masker is None:
            raise ValueError("Color transfer requires an explicit semantic inner-skin adapter")
        self.swapper, self.restorer, self.masker, self.options = swapper, restorer, masker, options

    def __call__(self, frame, faces):
        torch = _torch()
        import torch.nn.functional as F
        from roop_3d.vectorized_pipeline import crop_matrix
        if not frame.is_cuda or frame.dtype != torch.float32 or frame.ndim != 4 or frame.shape[:2] != (1, 3):
            raise ValueError('Compositor requires CUDA NCHW float32 RGB frames')
        result = frame
        for face in faces:
            crop_size = 512 if self.restorer or self.masker or self.options.laplacian_blend_weight else 128
            matrix = torch.as_tensor(crop_matrix(face.landmarks5, crop_size), device=frame.device, dtype=frame.dtype)[None]
            target = warp(frame, invert_affine(matrix), (crop_size, crop_size))
            swap_input = target if crop_size == 128 else F.interpolate(target, (128, 128), mode="bilinear", align_corners=False)
            swapped = self.swapper(swap_input.contiguous(), face)
            # Preserve the 128 reference geometry when no 512 processing is requested.
            enhanced = self.restorer(F.interpolate(swapped, (512, 512), mode="bilinear", align_corners=False), face) if self.restorer else swapped
            size = enhanced.shape[-1]
            if (enhanced.shape != (1, 3, size, size) or size not in (128, 256, 512, 1024)
                    or enhanced.device != frame.device or enhanced.dtype != torch.float32):
                raise ValueError("Restorer/swapper returned an invalid device image shape")
            local_target = target if size == crop_size else F.interpolate(target, (size, size), mode="bilinear", align_corners=False)
            alpha = face_feather(enhanced, size)
            if self.masker:
                semantic, skin, visibility = self.masker(target, face)
                for value in (semantic, skin, visibility):
                    if value.shape != (1, 1, 512, 512) or value.device != frame.device or value.dtype != torch.float32:
                        raise ValueError('Semantic adapter requires CUDA 1x1x512x512 float32 masks')
                semantic, skin, visibility = [F.interpolate(v, (size, size), mode="bilinear", align_corners=False) for v in (semantic, skin, visibility)]
                alpha = soft_matte(alpha * semantic, visibility, erosion=self.options.erosion,
                                   sigma=self.options.feather_sigma, jaw_falloff=self.options.jaw_falloff)
                inner = morphology(skin * visibility, max(1, self.options.erosion)) * visibility
                if self.options.color_strength:
                    enhanced = color_transfer(enhanced, local_target, inner, strength=self.options.color_strength,
                                              regional=self.options.regional_color or self.options.color_method == 'regional_reinhard')
            else:
                alpha = soft_matte(alpha, torch.ones_like(alpha), erosion=self.options.erosion,
                                   sigma=self.options.feather_sigma, jaw_falloff=self.options.jaw_falloff)
            if self.options.laplacian_blend_weight:
                enhanced = laplacian_blend(enhanced, local_target, alpha,
                                            laplacian_blend_weight=self.options.laplacian_blend_weight)
            # Matrix maps native frame pixels into crop pixels for inverse composite.
            forward = matrix * (size / crop_size)
            rgba = warp(torch.cat((enhanced, alpha), 1), forward, frame.shape[-2:])
            result = rgba[:, :3] * rgba[:, 3:] + result * (1 - rgba[:, 3:])
        return result.clamp(0, 1)


def stream_session(session, stream):
    """Clone an existing validated session on a worker-owned CUDA stream.

    Never changes the caller's session/provider options or model. Dynamic/TRT
    profile/cache settings are retained, and writable caches are boundary checked.
    """
    import onnxruntime as ort
    from pathlib import Path
    from roop_3d.models.owned_files import model_source, owned_path
    if not getattr(session, '_model_path', None):
        raise ValueError('Worker session cloning requires a file-backed, registry-checked model')
    path = model_source(Path(session._model_path))
    providers = []
    options = session.get_provider_options()
    for provider in session.get_providers():
        if provider not in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            continue
        values = dict(options[provider])
        values['user_compute_stream'] = str(stream.cuda_stream)
        values.pop('has_user_compute_stream', None)
        for name in ('trt_engine_cache_path', 'trt_timing_cache_path'):
            if values.get(name):
                cache = owned_path(Path(values[name]))
                for child in cache.rglob('*'):
                    owned_path(child)
        providers.append((provider, values))
    config = ort.SessionOptions()
    config.intra_op_num_threads = 1
    config.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    cloned = ort.InferenceSession(str(path), sess_options=config, providers=providers)
    cloned.disable_fallback()
    return cloned


def resident_swap_pipeline(session, embeddings, *, hardware_profile, execution_threads, restorer=None,
                           masker_factory=None, options=CompositorOptions()):
    """Construct an opt-in streaming pipeline without changing reference sessions."""
    if restorer is not None:
        # Host-only DecoupledRestorer is deliberately rejected before any frame is
        # submitted; callers must supply an explicitly implemented device factory.
        factory = getattr(restorer, 'cuda_adapter', None)
        if not callable(factory):
            raise ValueError('GPU compositor requires a CUDA restorer adapter; host bridges are forbidden')
    else:
        factory = None

    def make_compute(stream):
        torch = _torch()
        runner = TorchIOBinding(stream_session(session, stream), stream)
        with torch.cuda.stream(stream):
            sources = {key: torch.as_tensor(value, device=stream.device, dtype=torch.float32)[None].contiguous()
                       for key, value in embeddings.items()}
        def swap(crop, face):
            return runner.run({'target': crop, 'source': sources[face.track_id]},
                              {'output': ((1, 3, 128, 128), torch.float32)})['output']
        return GpuCompositor(swap, restorer=factory(stream) if factory else None,
                             masker=masker_factory(stream) if masker_factory else None, options=options)
    return TripleBufferedCudaPipeline(make_compute, hardware_profile=hardware_profile, execution_threads=execution_threads)


class TripleBufferedCudaPipeline:
    """Per-worker non-default H2D, compute, D2H streams and three owned slots.

    Ring leases may end immediately after enqueue's synchronous copy into pinned
    input. A slot and all device references survive until its D2H event completes.
    Host writer consumers receive owned arrays in submission order.
    """
    backend = "CUDA_RESIDENT_TRIPLE_BUFFER"

    def __init__(self, compute_factory, *, hardware_profile="6gb", execution_threads=1):
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required for the GPU compositor")
        if hardware_profile not in ("6gb", "12gb") or not 1 <= execution_threads <= 8:
            raise ValueError("Invalid GPU worker profile")
        self.factory, self.profile, self.workers = compute_factory, hardware_profile, execution_threads
        self.upload, self.compute, self.download = [torch.cuda.Stream() for _ in range(3)]
        self.operation = compute_factory(self.compute)
        self.slots = [dict() for _ in range(3)]
        import threading
        self.lock = threading.RLock()

    def fork_worker(self):
        return TripleBufferedCudaPipeline(self.factory, hardware_profile=self.profile, execution_threads=self.workers)

    def process(self, frame, faces):
        return self.process_batch([frame], [faces])[0]

    def process_batch(self, frames, faces_by_frame):
        torch = _torch()
        if len(frames) != len(faces_by_frame):
            raise ValueError("Frame/face batch sizes differ")
        with self.lock, torch.inference_mode():
            outputs, pending = [], deque()
            def collect():
                slot = pending.popleft()
                slot['done'].synchronize()
                if not slot['valid'].item():
                    raise RuntimeError('GPU compositor produced non-finite pixels')
                outputs.append(slot['host_out'].numpy().copy())
                slot.pop('device', None)
                slot.pop('result', None)
            try:
                for index, (frame, faces) in enumerate(zip(frames, faces_by_frame)):
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError("Expected uint8 HWC BGR frame")
                    # Bound full-frame grids/rgba, pinned slots and device intermediates
                    # independently of model arenas, for each worker/profile.
                    budget = (256 if self.profile == '6gb' else 768) * 1048576 // self.workers
                    if frame.shape[0] * frame.shape[1] * 256 > budget:
                        raise ValueError("Triple-buffer frame exceeds the worker GPU profile budget")
                    if len(pending) == 3:
                        collect()
                    slot = self.slots[index % 3]
                    if slot.get('shape') != frame.shape:
                        slot.update(shape=frame.shape,
                            host_in=torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True),
                            host_out=torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True),
                            valid=torch.empty((), dtype=torch.bool, pin_memory=True))
                    slot['host_in'].copy_(torch.from_numpy(frame))
                    with torch.cuda.stream(self.upload):
                        slot['device'] = slot['host_in'].to('cuda', non_blocking=True)
                        uploaded = torch.cuda.Event()
                        uploaded.record()
                    with torch.cuda.stream(self.compute):
                        self.compute.wait_event(uploaded)
                        slot['device'].record_stream(self.compute)
                        rgb = slot['device'].permute(2, 0, 1).flip(0)[None].float() / 255
                        result = self.operation(rgb, faces)
                        slot['device_valid'] = torch.isfinite(result).all()
                        slot['result'] = (result[0].flip(0).permute(1, 2, 0) * 255).round().clamp(0, 255).byte().contiguous()
                        computed = torch.cuda.Event()
                        computed.record()
                    with torch.cuda.stream(self.download):
                        self.download.wait_event(computed)
                        slot['result'].record_stream(self.download)
                        slot['host_out'].copy_(slot['result'], non_blocking=True)
                        slot['valid'].copy_(slot['device_valid'], non_blocking=True)
                        slot['done'] = torch.cuda.Event()
                        slot['done'].record()
                    pending.append(slot)
                while pending:
                    collect()
                return outputs
            finally:
                # Failure/cancellation must not free host or device storage in flight.
                for stream in (self.upload, self.compute, self.download):
                    stream.synchronize()
