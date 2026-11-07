"""Stage 9: Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization.

Provides adaptive hardware profiles for RTX 4070 Desktop (12 GB) and RTX 3060 Laptop (6 GB),
thread-safe model session caching, CUDA I/O binding, sequential memory eviction,
and strict FP32 precision contract enforcement for INSwapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Literal

import numpy as np
import onnxruntime as ort

from roop_3d.core.errors import DomainError
from roop_3d.core.hardware import Hardware, probe_hardware, resolve_hardware_profile
from roop_3d.core.settings import Settings
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, record_named
from roop_3d.models.catalog import ModelCatalog


ProfileType = Literal["12gb", "6gb", "cpu"]
ResidencyStrategy = Literal["resident", "sequential", "cpu"]
QualityTier = Literal["high", "conservative", "basic"]

# Models classified as heavy requiring sequential eviction on 6GB profiles
HEAVY_MODELS = {
    "inswapper_128.onnx",
    "inswapper_128.batch.onnx",
    "inswapper_128_dynamic.onnx",
    "gpen_bfr_512.onnx",
    "GPEN-BFR-512.onnx",
    "gpen_bfr_1024.onnx",
    "CodeFormerv0.1.onnx",
    "codeformer.fp16.onnx",
    "liveportrait_warping_spade.onnx",
}

# Lightweight models safe to keep resident on all GPU profiles (~250 MiB total)
LIGHT_MODELS = {
    "scrfd_10g_bnkps.onnx",
    "scrfd_2.5g_kps.onnx",
    "2d106det.onnx",
    "w600k_r50.onnx",
    "liveportrait_motion_extractor.onnx",
    "liveportrait_landmark.onnx",
    "liveportrait_appearance_feature_extractor.onnx",
    "liveportrait_stitching.onnx",
}


@dataclass(frozen=True)
class AdaptiveHardwareProfile:
    name: str
    profile_type: ProfileType
    total_vram_mib: int
    vram_ceiling_mib: int
    model_residency: ResidencyStrategy
    concurrency_limit: int
    quality_tier: QualityTier
    max_resolution: int | None
    allow_cpu_offload: bool
    ort_provider_options: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def validate_swap_precision(self, model_name: str, requested_precision: str | None = None) -> str:
        """Strict Precision Rule: INSwapper must retain its verified FP32 precision contract."""
        if "inswapper" in model_name.lower():
            if requested_precision and requested_precision.upper() not in ("FP32", "FLOAT32"):
                raise DomainError(
                    "PRECISION_CONTRACT_VIOLATION",
                    f"Model {model_name} is locked to FP32; precision {requested_precision} is forbidden",
                    422,
                )
            return "FP32"
        return requested_precision or ("FP16" if self.profile_type != "cpu" else "FP32")


def resolve_adaptive_profile(
    settings: Settings | None = None,
    hardware: Hardware | None = None,
) -> AdaptiveHardwareProfile:
    """Resolve adaptive profile matching host hardware capabilities and user settings."""
    settings = settings or Settings()
    hw = hardware or probe_hardware(settings)

    gpu = hw.gpus[0] if hw.gpus else None
    total_vram = int(gpu.total_vram_mib) if gpu else 0
    free_vram = int(gpu.free_vram_mib) if gpu else 0

    profile_setting = settings.hardware_profile
    resolved_type = resolve_hardware_profile(total_vram, free_vram, profile_setting)

    if resolved_type == "cpu" or not gpu or settings.preferred_provider == "CPUExecutionProvider":
        return AdaptiveHardwareProfile(
            name="CPU_Fallback",
            profile_type="cpu",
            total_vram_mib=0,
            vram_ceiling_mib=0,
            model_residency="cpu",
            concurrency_limit=1,
            quality_tier="basic",
            max_resolution=512,
            allow_cpu_offload=True,
            ort_provider_options={},
            description="CPU reference execution with sequential processing and multi-threaded CPU decode",
        )

    if resolved_type == "12gb" and total_vram >= 10240:
        # PRIMARY: RTX 4070 Desktop Profile (12 GB VRAM, 32 GB RAM)
        vram_budget = max(0, min(free_vram, 12288) - settings.memory_safety_reserve_mb)
        provider_options = {
            "gpu_mem_limit": vram_budget * 1048576,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": "1",
        }
        concurrency = 2 if free_vram > 8192 and settings.max_concurrent_jobs > 1 else 1
        return AdaptiveHardwareProfile(
            name="RTX_4070_Desktop",
            profile_type="12gb",
            total_vram_mib=total_vram,
            vram_ceiling_mib=12288,
            model_residency="resident",
            concurrency_limit=concurrency,
            quality_tier="high",
            max_resolution=None,  # Full resolution permitted
            allow_cpu_offload=False,
            ort_provider_options=provider_options,
            description="High-throughput GPU resident profile for RTX 4070 12GB Desktop",
        )

    # SECONDARY: RTX 3060 Laptop Profile (6 GB VRAM, 16 GB RAM)
    vram_budget = max(0, min(free_vram, 4600) - settings.memory_safety_reserve_mb)
    provider_options = {
        "gpu_mem_limit": vram_budget * 1048576,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
    }
    return AdaptiveHardwareProfile(
        name="RTX_3060_Laptop",
        profile_type="6gb",
        total_vram_mib=total_vram,
        vram_ceiling_mib=4600,
        model_residency="sequential",
        concurrency_limit=1,  # Strict concurrency ceiling
        quality_tier="conservative",
        max_resolution=512,  # Downscale under VRAM pressure
        allow_cpu_offload=True,
        ort_provider_options=provider_options,
        description="Sequential model execution with 4,600 MiB ceiling for RTX 3060 Laptop 6GB",
    )


class AdaptiveModelPool:
    """Thread-safe model session caching and residency lifecycle manager.

    On 12GB profiles, models remain GPU resident across frames and requests.
    On 6GB profiles, heavy models are executed sequentially with eager eviction.
    Precision contracts (such as INSwapper FP32) are strictly enforced.
    """

    _instance: "AdaptiveModelPool | None" = None
    _singleton_lock = threading.Lock()

    def __init__(self, profile: AdaptiveHardwareProfile | None = None):
        self.profile = profile or resolve_adaptive_profile()
        self._lock = threading.RLock()
        self._models: dict[str, OrtModel] = {}
        self._active_heavy: str | None = None
        self._access_order: list[str] = []

    @classmethod
    def get_instance(cls, profile: AdaptiveHardwareProfile | None = None) -> "AdaptiveModelPool":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(profile)
            elif profile is not None and cls._instance.profile.profile_type != profile.profile_type:
                cls._instance.release_all()
                cls._instance.profile = profile
            return cls._instance

    def acquire(self, catalog: ModelCatalog, name: str, config: RuntimeConfig) -> OrtModel:
        """Acquire a warm or newly initialized OrtModel session according to profile policy."""
        # Enforce precision rule before session initialization
        self.profile.validate_swap_precision(name)

        with self._lock:
            # Check if model is already loaded and valid
            if name in self._models:
                existing = self._models[name]
                if existing.session is not None and existing.provider == config.provider:
                    # Update access order
                    if name in self._access_order:
                        self._access_order.remove(name)
                    self._access_order.append(name)
                    return existing

            # 6GB sequential eviction policy: evict previous heavy model before loading new one
            if self.profile.model_residency == "sequential" and name in HEAVY_MODELS:
                if self._active_heavy and self._active_heavy != name:
                    self._evict_model(self._active_heavy)
                self._active_heavy = name

            # Initialize new OrtModel session with raw factory to prevent recursion
            raw_config = RuntimeConfig(
                provider=config.provider,
                hardware_profile=config.hardware_profile,
                reserve_mib=config.reserve_mib,
                cpu_threads=config.cpu_threads,
                model_factory=None,
                graph_optimization=config.graph_optimization,
            )
            model = OrtModel(catalog, name, raw_config)
            self._models[name] = model
            self._access_order.append(name)
            return model

    def _evict_model(self, name: str) -> None:
        """Evict a specific model and collect Python / CUDA memory."""
        if name in self._models:
            model = self._models.pop(name)
            try:
                model.release()
            except Exception:
                pass
            if name in self._access_order:
                self._access_order.remove(name)
        gc.collect()

    def release_heavy(self) -> None:
        """Release heavy models (swapper, restorer, warping generator) while keeping light models."""
        with self._lock:
            for name in list(self._models.keys()):
                if name in HEAVY_MODELS:
                    self._evict_model(name)
            self._active_heavy = None

    def release_all(self) -> None:
        """Release all cached sessions and clean up memory."""
        with self._lock:
            for name in list(self._models.keys()):
                self._evict_model(name)
            self._models.clear()
            self._access_order.clear()
            self._active_heavy = None
            gc.collect()

    def stats(self) -> dict[str, Any]:
        """Telemetry snapshot of model pool status."""
        with self._lock:
            return {
                "profile": self.profile.name,
                "profile_type": self.profile.profile_type,
                "residency_strategy": self.profile.model_residency,
                "cached_model_count": len(self._models),
                "cached_models": list(self._models.keys()),
                "active_heavy_model": self._active_heavy,
            }


class CudaIoBinder:
    """CUDA I/O binding accelerator for fixed-shape ONNX Runtime sessions."""

    def __init__(self, session: ort.InferenceSession, device_id: int = 0):
        from roop_3d.core.ort_session import GpuIoBinding
        self.session = session
        self.device_id = device_id
        self._input_nodes = {node.name: node for node in session.get_inputs()}
        self._output_nodes = {node.name: node for node in session.get_outputs()}
        outputs = {}
        for node in self._output_nodes.values():
            if all(type(dim) is int and dim > 0 for dim in node.shape):
                dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                         "tensor(double)": np.float64, "tensor(int64)": np.int64,
                         "tensor(int32)": np.int32}.get(node.type)
                if dtype is not None:
                    outputs[node.name] = (tuple(node.shape), dtype)
        self._runner = GpuIoBinding(session, device_id=device_id, output_buffers=outputs)

    def bind_and_run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute via retained CUDA buffers; callers choose the explicit host boundary."""
        device_outputs = self._runner.run({name: value for name, value in feeds.items()
                                           if name in self._input_nodes})
        return {name: device_outputs[name].numpy() for name in self._output_nodes}


class AdaptivePipelineOptimizer:
    """System-level optimization coordinator for decoding, compositing, and scheduling."""

    @staticmethod
    def get_recommended_decode_threads(hardware: Hardware | None = None) -> int:
        """Determine optimal CPU decode thread count based on physical cores (adapting from 1)."""
        hw = hardware or probe_hardware()
        cores = hw.physical_cpu_count or 4
        # Allocate up to 8 threads, preserving at least 2 cores for the operating system
        return max(2, min(8, cores - 2))

    @staticmethod
    def fast_multiband_blend(
        target: np.ndarray,
        source: np.ndarray,
        alpha: np.ndarray,
        levels: int = 4,
    ) -> np.ndarray:
        """Optimized multi-band Laplacian pyramid fusion with vectorized band recombination."""
        from roop_3d.compositor.boundary_fusion import _pyramid_blend
        blended = _pyramid_blend(target, source, alpha, levels)
        if target.dtype == np.uint8:
            return np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return blended
