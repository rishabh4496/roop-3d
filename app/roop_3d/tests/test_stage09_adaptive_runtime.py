"""Tests for Stage 9 Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.hardware import GPU, Hardware
from roop_3d.core.settings import Settings
from roop_3d.engine.adaptive_runtime import (
    AdaptiveHardwareProfile,
    AdaptiveModelPool,
    AdaptivePipelineOptimizer,
    CudaIoBinder,
    HEAVY_MODELS,
    LIGHT_MODELS,
    resolve_adaptive_profile,
)
from roop_3d.engine.runtime import RuntimeConfig


def _mock_gpu(total_mb: int, free_mb: int, name: str = "NVIDIA GeForce RTX 4070") -> GPU:
    return GPU(
        index=0,
        name=name,
        total_vram_mib=total_mb,
        free_vram_mib=free_mb,
        driver_version="621.01",
        compute_capability="8.9",
        effective_vram_budget_mib=total_mb,
    )


def _mock_hw(gpus: list[GPU], cpu_cores: int = 8, ram_mb: int = 16000) -> Hardware:
    return Hardware(
        timestamp="2026-11-04T00:00:00Z",
        os={"system": "Windows", "release": "10"},
        logical_cpu_count=cpu_cores * 2,
        physical_cpu_count=cpu_cores,
        system_ram_total_mib=ram_mb,
        gpus=gpus,
    )


def test_resolve_adaptive_profile_12gb_desktop():
    gpu = _mock_gpu(12288, 11000, name="NVIDIA GeForce RTX 4070")
    hw = _mock_hw([gpu], cpu_cores=24, ram_mb=32451)
    settings = Settings(hardware_profile="12gb")

    profile = resolve_adaptive_profile(settings, hw)
    assert profile.profile_type == "12gb"
    assert profile.name == "RTX_4070_Desktop"
    assert profile.model_residency == "resident"
    assert profile.quality_tier == "high"
    assert profile.allow_cpu_offload is False
    assert profile.vram_ceiling_mib == 12288
    assert profile.max_resolution is None
    assert profile.ort_provider_options.get("arena_extend_strategy") == "kNextPowerOfTwo"


def test_resolve_adaptive_profile_6gb_laptop():
    gpu = _mock_gpu(6144, 5500, name="NVIDIA GeForce RTX 3060 Laptop GPU")
    hw = _mock_hw([gpu], cpu_cores=8, ram_mb=16000)
    settings = Settings(hardware_profile="6gb")

    profile = resolve_adaptive_profile(settings, hw)
    assert profile.profile_type == "6gb"
    assert profile.name == "RTX_3060_Laptop"
    assert profile.model_residency == "sequential"
    assert profile.quality_tier == "conservative"
    assert profile.concurrency_limit == 1
    assert profile.allow_cpu_offload is True
    assert profile.vram_ceiling_mib == 4600
    assert profile.max_resolution == 512
    assert profile.ort_provider_options.get("arena_extend_strategy") == "kSameAsRequested"
    assert profile.ort_provider_options.get("cudnn_conv_algo_search") == "HEURISTIC"


def test_resolve_adaptive_profile_cpu_fallback():
    hw = _mock_hw([], cpu_cores=4, ram_mb=16000)
    settings = Settings(hardware_profile="cpu")

    profile = resolve_adaptive_profile(settings, hw)
    assert profile.profile_type == "cpu"
    assert profile.name == "CPU_Fallback"
    assert profile.model_residency == "cpu"
    assert profile.concurrency_limit == 1
    assert profile.quality_tier == "basic"
    assert profile.allow_cpu_offload is True


def test_strict_inswapper_fp32_precision_contract():
    profile_12gb = AdaptiveHardwareProfile(
        name="RTX_4070_Desktop",
        profile_type="12gb",
        total_vram_mib=12288,
        vram_ceiling_mib=12288,
        model_residency="resident",
        concurrency_limit=1,
        quality_tier="high",
        max_resolution=None,
        allow_cpu_offload=False,
    )

    # Valid FP32 requests
    assert profile_12gb.validate_swap_precision("inswapper_128.onnx") == "FP32"
    assert profile_12gb.validate_swap_precision("inswapper_128.onnx", "FP32") == "FP32"
    assert profile_12gb.validate_swap_precision("inswapper_128.onnx", "FLOAT32") == "FP32"

    # Forbidden low-precision requests for INSwapper must raise DomainError
    with pytest.raises(DomainError) as exc_info:
        profile_12gb.validate_swap_precision("inswapper_128.onnx", "FP16")
    assert exc_info.value.code == "PRECISION_CONTRACT_VIOLATION"

    with pytest.raises(DomainError) as exc_info:
        profile_12gb.validate_swap_precision("inswapper_128.onnx", "INT8")
    assert exc_info.value.code == "PRECISION_CONTRACT_VIOLATION"

    # Other models can use FP16
    assert profile_12gb.validate_swap_precision("gpen_bfr_512.onnx", "FP16") == "FP16"


def test_adaptive_model_pool_residency_12gb():
    profile = AdaptiveHardwareProfile(
        name="RTX_4070_Desktop",
        profile_type="12gb",
        total_vram_mib=12288,
        vram_ceiling_mib=12288,
        model_residency="resident",
        concurrency_limit=1,
        quality_tier="high",
        max_resolution=None,
        allow_cpu_offload=False,
    )
    pool = AdaptiveModelPool(profile)
    catalog = MagicMock()
    config = RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile="12gb")

    with patch("roop_3d.engine.adaptive_runtime.OrtModel") as MockOrtModel:
        mock_instance1 = MagicMock()
        mock_instance1.session = MagicMock()
        mock_instance1.provider = "CUDAExecutionProvider"

        mock_instance2 = MagicMock()
        mock_instance2.session = MagicMock()
        mock_instance2.provider = "CUDAExecutionProvider"

        MockOrtModel.side_effect = [mock_instance1, mock_instance2]

        m1 = pool.acquire(catalog, "inswapper_128.onnx", config)
        m2 = pool.acquire(catalog, "gpen_bfr_512.onnx", config)

        stats = pool.stats()
        # Both models remain resident on 12GB profile
        assert "inswapper_128.onnx" in stats["cached_models"]
        assert "gpen_bfr_512.onnx" in stats["cached_models"]
        assert stats["cached_model_count"] == 2

        # Re-acquiring inswapper returns existing session without recreation
        m1_reacquired = pool.acquire(catalog, "inswapper_128.onnx", config)
        assert m1_reacquired is m1
        assert MockOrtModel.call_count == 2


def test_adaptive_model_pool_sequential_eviction_6gb():
    profile = AdaptiveHardwareProfile(
        name="RTX_3060_Laptop",
        profile_type="6gb",
        total_vram_mib=6144,
        vram_ceiling_mib=4600,
        model_residency="sequential",
        concurrency_limit=1,
        quality_tier="conservative",
        max_resolution=512,
        allow_cpu_offload=True,
    )
    pool = AdaptiveModelPool(profile)
    catalog = MagicMock()
    config = RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile="6gb")

    with patch("roop_3d.engine.adaptive_runtime.OrtModel") as MockOrtModel:
        m_inswapper = MagicMock()
        m_inswapper.session = MagicMock()
        m_inswapper.provider = "CUDAExecutionProvider"

        m_gpen = MagicMock()
        m_gpen.session = MagicMock()
        m_gpen.provider = "CUDAExecutionProvider"

        m_scrfd = MagicMock()
        m_scrfd.session = MagicMock()
        m_scrfd.provider = "CUDAExecutionProvider"

        MockOrtModel.side_effect = [m_inswapper, m_gpen, m_scrfd]

        # 1. Acquire heavy model: inswapper
        pool.acquire(catalog, "inswapper_128.onnx", config)
        assert "inswapper_128.onnx" in pool.stats()["cached_models"]
        assert pool.stats()["active_heavy_model"] == "inswapper_128.onnx"

        # 2. Acquire another heavy model: gpen_bfr_512.
        # Under 6GB sequential policy, inswapper must be evicted before loading gpen!
        pool.acquire(catalog, "gpen_bfr_512.onnx", config)
        m_inswapper.release.assert_called_once()
        assert "inswapper_128.onnx" not in pool.stats()["cached_models"]
        assert "gpen_bfr_512.onnx" in pool.stats()["cached_models"]
        assert pool.stats()["active_heavy_model"] == "gpen_bfr_512.onnx"

        # 3. Acquire a lightweight model: scrfd_10g_bnkps.
        # Lightweight models do NOT trigger eviction of the active heavy model.
        pool.acquire(catalog, "scrfd_10g_bnkps.onnx", config)
        assert "gpen_bfr_512.onnx" in pool.stats()["cached_models"]
        assert "scrfd_10g_bnkps.onnx" in pool.stats()["cached_models"]
        assert pool.stats()["cached_model_count"] == 2


def test_adaptive_model_pool_release_heavy_and_all():
    profile = AdaptiveHardwareProfile(
        name="RTX_4070_Desktop",
        profile_type="12gb",
        total_vram_mib=12288,
        vram_ceiling_mib=12288,
        model_residency="resident",
        concurrency_limit=1,
        quality_tier="high",
        max_resolution=None,
        allow_cpu_offload=False,
    )
    pool = AdaptiveModelPool(profile)
    catalog = MagicMock()
    config = RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile="12gb")

    with patch("roop_3d.engine.adaptive_runtime.OrtModel") as MockOrtModel:
        m_inswapper = MagicMock()
        m_inswapper.session = MagicMock()
        m_inswapper.provider = "CUDAExecutionProvider"

        m_scrfd = MagicMock()
        m_scrfd.session = MagicMock()
        m_scrfd.provider = "CUDAExecutionProvider"

        MockOrtModel.side_effect = [m_inswapper, m_scrfd]

        pool.acquire(catalog, "inswapper_128.onnx", config)
        pool.acquire(catalog, "scrfd_10g_bnkps.onnx", config)
        assert len(pool.stats()["cached_models"]) == 2

        # release_heavy releases only heavy models
        pool.release_heavy()
        m_inswapper.release.assert_called_once()
        m_scrfd.release.assert_not_called()
        assert "inswapper_128.onnx" not in pool.stats()["cached_models"]
        assert "scrfd_10g_bnkps.onnx" in pool.stats()["cached_models"]

        # release_all releases all remaining models
        pool.release_all()
        m_scrfd.release.assert_called_once()
        assert len(pool.stats()["cached_models"]) == 0


def test_cuda_io_binder_surfaces_binding_failure_without_session_run():
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_tensor"
    mock_output = MagicMock()
    mock_output.name = "output_tensor"

    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]

    # Simulate IOBinding failing so it tests transparent fallback
    mock_session.io_binding.side_effect = RuntimeError("CUDA IOBinding not supported")
    with pytest.raises(RuntimeError, match="CUDA IOBinding not supported"):
        CudaIoBinder(mock_session, device_id=0)
    mock_session.run.assert_not_called()


def test_adaptive_pipeline_optimizer_decode_threads():
    # 2 cores -> max(2, min(8, 2-2)) = 2
    hw_2 = _mock_hw([], cpu_cores=2, ram_mb=8000)
    assert AdaptivePipelineOptimizer.get_recommended_decode_threads(hw_2) == 2

    # 8 cores -> max(2, min(8, 8-2)) = 6
    hw_8 = _mock_hw([], cpu_cores=8, ram_mb=16000)
    assert AdaptivePipelineOptimizer.get_recommended_decode_threads(hw_8) == 6

    # 24 cores -> max(2, min(8, 24-2)) = 8
    hw_24 = _mock_hw([], cpu_cores=24, ram_mb=32000)
    assert AdaptivePipelineOptimizer.get_recommended_decode_threads(hw_24) == 8


def test_fast_multiband_blend_numerical_bounds():
    target = np.full((64, 64, 3), 50, dtype=np.uint8)
    source = np.full((64, 64, 3), 200, dtype=np.uint8)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0

    blended = AdaptivePipelineOptimizer.fast_multiband_blend(target, source, alpha, levels=2)
    assert blended.shape == (64, 64, 3)
    assert blended.dtype == np.uint8
    assert np.all(blended >= 0)
    assert np.all(blended <= 255)
    # Inside center should be influenced by source
    assert blended[32, 32, 0] > 150
    # Outer edge should be influenced by target
    assert blended[5, 5, 0] < 80
