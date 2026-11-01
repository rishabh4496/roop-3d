"""Hardware-gated synthetic compositor campaign, never neural inference FPS."""
from __future__ import annotations

import json
import time
from unittest.mock import patch

import cv2
import numpy as np


def benchmark_compositor(*, iterations=50, hardware_profile='6gb'):
    import torch
    from roop_3d.gpu_compositor import (GpuCompositor, TripleBufferedCudaPipeline,
        color_transfer, gaussian, laplacian_blend, morphology, soft_matte)
    from roop_3d.optimized_prepass import LockedFace
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.vectorized_pipeline import composite_cpu, crop_matrix
    if iterations < 3:
        raise ValueError('Compositor campaign requires at least three iterations')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA hardware is required; the gate cannot pass via CPU fallback')
    torch.manual_seed(10)
    rng = np.random.default_rng(10)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    y, x = np.mgrid[:128, :128].astype(np.float32)
    crops = np.stack([np.stack((.45 + .2 * np.sin(x / 19), .5 + .15 * np.cos(y / 17),
                                .48 + .08 * np.sin((x + y) / 4)), 0) for _ in range(2)]).astype(np.float32)
    faces = [LockedFace(0, i, np.array((0, 0, 400, 400)),
              ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 2.7 + [130 + i * 570, 100]).astype(np.float32)) for i in range(2)]
    matrices = [crop_matrix(face.landmarks5) for face in faces]
    def factory(stream):
        with torch.cuda.stream(stream):
            device_crops = torch.as_tensor(crops, device=stream.device)
        return GpuCompositor(lambda _, face: device_crops[face.track_id:face.track_id + 1])
    pipeline = TripleBufferedCudaPipeline(factory, hardware_profile=hardware_profile)
    batch = [frame] * 3
    face_batch = [faces] * 3
    cpu = composite_cpu(frame, crops, matrices)
    forbidden_calls = 0
    def forbidden(*_args, **_kwargs):
        nonlocal forbidden_calls
        forbidden_calls += 1
        raise AssertionError('CPU cv2.warpAffine called during CUDA composition')
    with patch.object(cv2, 'warpAffine', forbidden):
        for _ in range(3):
            gpu = pipeline.process_batch(batch, face_batch)[0]
        start = time.perf_counter()
        for _ in range(iterations):
            gpu = pipeline.process_batch(batch, face_batch)[0]
        gpu_seconds = time.perf_counter() - start
    for _ in range(3):
        composite_cpu(frame, crops, matrices)
    start = time.perf_counter()
    for _ in range(iterations * 3):
        composite_cpu(frame, crops, matrices)
    cpu_seconds = time.perf_counter() - start
    mse = np.mean((cpu.astype(np.float64) - gpu.astype(np.float64)) ** 2)
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else 999.

    # Exercise advanced kernels separately against their CPU mathematical path;
    # these are not claimed to match the legacy quantized OpenCV LAB algorithm.
    source = torch.from_numpy(crops[:1].copy())
    target = (source * .8 + .05).clamp(0, 1)
    skin = torch.zeros((1, 1, 128, 128))
    skin[..., 16:100, 18:110] = 1
    skin[..., 38:50, 25:105] = 0  # excluded eyes
    skin[..., 80:90, 45:85] = 0   # excluded teeth
    visible = torch.ones_like(skin)
    visible[..., 60:, :35] = 0   # foreground occluder
    def advanced(s, t, m, v):
        inner = morphology(m * v, 2) * v
        matched = color_transfer(s, t, inner, strength=.7)
        regional = color_transfer(s, t, inner, strength=.5, regional=True)
        alpha = soft_matte(m, v, erosion=2, sigma=2, jaw_falloff=.1)
        return (matched, regional, laplacian_blend(matched, t, alpha), gaussian(m, 2), morphology(m, -2), alpha)
    reference = advanced(source, target, skin, visible)
    with patch.object(cv2, 'warpAffine', forbidden), patch.object(cv2, 'GaussianBlur', forbidden):
        actual = advanced(*(v.cuda() for v in (source, target, skin, visible)))
        errors = [float((a.cpu() - b).abs().max()) for a, b in zip(actual, reference)]
    protected = (skin * visible) == 0
    protected_ok = torch.equal(actual[0].cpu().masked_select(protected), source.masked_select(protected))
    visibility_ok = torch.count_nonzero(actual[-1].cpu()[visible == 0]).item() == 0
    speedup = cpu_seconds / gpu_seconds
    passed = forbidden_calls == 0 and speedup >= 1.35 and psnr >= 42 and max(errors) < 2e-4 and protected_ok and visibility_ok
    report = {'state': 'PASS' if passed else 'FAIL', 'scope': 'synthetic_compositor_only_no_neural_inference',
        'gpu': torch.cuda.get_device_name(), 'hardware_profile': hardware_profile,
        'shape': list(frame.shape), 'faces': len(faces), 'frames': iterations * 3,
        'includes': 'pinned staging, H2D, crop warp, inverse composite, D2H, owned host output',
        'cpu_warp_calls_during_cuda': forbidden_calls, 'cpu_fps': iterations * 3 / cpu_seconds,
        'gpu_fps': iterations * 3 / gpu_seconds, 'speedup': speedup, 'psnr_db': psnr,
        'advanced_kernel_max_abs_errors': errors, 'protected_skin_pixels_unchanged': protected_ok,
        'occluded_alpha_zero': visibility_ok, 'real_model_chain_acceptance': 'NOT_RUN'}
    print(json.dumps(report, indent=2))
    return 0 if passed else 1
