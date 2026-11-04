"""Unit tests for Stage 16 environment preflight and hardware profile detection."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from roop_3d.installer.preflight import (
    check_cpu,
    check_cuda_compatibility,
    check_disk_space,
    check_ffmpeg,
    check_node_version,
    check_nvidia_gpu_and_driver,
    check_onnx_runtime,
    check_python_version,
    check_system_ram,
    check_tensorrt,
    check_windows_version,
    run_environment_preflight,
)


def test_windows_version_detects_system() -> None:
    res = check_windows_version()
    assert res.check_id == "windows_version"
    assert res.status in ("PASS", "WARN")
    assert "system" in res.details
    assert res.details["64bit"] is True


def test_python_version_checks_range() -> None:
    res = check_python_version()
    assert res.check_id == "python_version"
    assert res.status in ("PASS", "WARN")
    assert "version" in res.details
    assert res.details["is_64bit"] is True


def test_node_version_checks_installed() -> None:
    res = check_node_version()
    assert res.check_id == "node_version"
    assert res.status in ("PASS", "WARN")


def test_cpu_reports_cores_and_arch() -> None:
    res = check_cpu()
    assert res.check_id == "cpu"
    assert res.status in ("PASS", "WARN")
    assert res.details["logical_cores"] >= 1
    assert res.details["architecture"] != ""


def test_system_ram_checks_capacity() -> None:
    res = check_system_ram()
    assert res.check_id == "system_ram"
    assert res.status in ("PASS", "WARN")
    assert res.details["total_mib"] > 0


def test_rtx_4070_profile_assignment() -> None:
    hw = SimpleNamespace(
        gpus=[
            SimpleNamespace(
                name="NVIDIA GeForce RTX 4070",
                total_vram_mib=12282,
                free_vram_mib=10500,
                driver_version="551.76",
                compute_capability="8.9",
            )
        ]
    )
    gpu_res, drv_res, profile = check_nvidia_gpu_and_driver(hw)
    assert profile == "12gb"
    assert gpu_res.status == "PASS"
    assert "RTX 4070 (12 GB) resident profile" in gpu_res.message
    assert drv_res.status == "PASS"


def test_rtx_3060_profile_assignment() -> None:
    hw = SimpleNamespace(
        gpus=[
            SimpleNamespace(
                name="NVIDIA GeForce RTX 3060",
                total_vram_mib=6144,
                free_vram_mib=5200,
                driver_version="535.100",
                compute_capability="8.6",
            )
        ]
    )
    gpu_res, drv_res, profile = check_nvidia_gpu_and_driver(hw)
    assert profile == "6gb"
    assert gpu_res.status == "PASS"
    assert "RTX 3060 (6 GB) Low-VRAM profile" in gpu_res.message


def test_cpu_profile_fallback_when_no_gpu() -> None:
    hw = SimpleNamespace(gpus=[])
    gpu_res, drv_res, profile = check_nvidia_gpu_and_driver(hw)
    assert profile == "cpu"
    assert gpu_res.status == "WARN"
    assert "CPU" in gpu_res.message


def test_cuda_compatibility() -> None:
    hw = SimpleNamespace(
        cuda_runtime_version="12.4",
        driver_supported_cuda_version="12.6",
    )
    res = check_cuda_compatibility(hw)
    assert res.status == "PASS"

    hw_mismatch = SimpleNamespace(
        cuda_runtime_version="12.8",
        driver_supported_cuda_version="12.2",
    )
    res_mismatch = check_cuda_compatibility(hw_mismatch)
    assert res_mismatch.status == "FAIL"


def test_onnx_runtime_checks_providers() -> None:
    hw = SimpleNamespace(
        ort_version="1.20.0",
        advertised_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    res = check_onnx_runtime(hw)
    assert res.status == "PASS"


def test_tensorrt_detection() -> None:
    res = check_tensorrt()
    assert res.check_id == "tensorrt"
    assert res.status in ("PASS", "WARN")


def test_ffmpeg_detection() -> None:
    hw = SimpleNamespace(
        ffmpeg_path="/bin/ffmpeg",
        ffmpeg_nvenc_encoders=["h264_nvenc"],
        ffmpeg_nvdec_decoders=["h264_cuvid"],
    )
    res = check_ffmpeg(hw)
    assert res.status == "PASS"
    assert "NVENC acceleration supported" in res.message


def test_disk_space(tmp_path) -> None:
    res = check_disk_space(tmp_path)
    assert res.check_id == "disk_space"
    assert res.status in ("PASS", "WARN")
    assert res.details["free_gb"] > 0


def test_full_environment_preflight() -> None:
    report = run_environment_preflight()
    assert report["status"] in ("PASS", "WARN")
    assert report["ready"] is True
    assert len(report["checks"]) >= 12
    assert report["resolved_hardware_profile"] in ("cpu", "6gb", "12gb")
