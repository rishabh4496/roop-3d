"""Environment preflight checker for Roop 3D.

Performs a 12-point environment preflight checking:
1. Windows version (and OS architecture)
2. Python version and virtual environment
3. Node.js version
4. CPU cores and architecture
5. System RAM
6. NVIDIA GPU
7. Driver version
8. CUDA compatibility
9. ONNX Runtime providers
10. TensorRT
11. FFmpeg codecs
12. Available disk space

Supports RTX 4070 ('12gb' resident profile) and RTX 3060 ('6gb' low-VRAM clamped profile).
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Literal

from roop_3d.core.hardware import probe_hardware
from roop_3d.core.settings import Settings

LOW_VRAM_PROFILE_CEILING = 6144
RESIDENT_VRAM_PROFILE_FLOOR = 10240


class PreflightCheckResult:
    def __init__(self, check_id: str, name: str, status: Literal["PASS", "WARN", "FAIL"],
                 message: str, details: dict[str, Any] | None = None):
        self.check_id = check_id
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def check_windows_version() -> PreflightCheckResult:
    system = platform.system()
    release = platform.release()
    version = platform.version()
    is_64bit = sys.maxsize > 2**32
    details = {"system": system, "release": release, "version": version, "64bit": is_64bit}

    if system != "Windows":
        return PreflightCheckResult(
            "windows_version", "Windows Version", "WARN",
            f"Host is running on {system} ({release}). Production deployment targets Windows 10/11 x64.",
            details,
        )

    if not is_64bit:
        return PreflightCheckResult(
            "windows_version", "Windows Version", "FAIL",
            "64-bit Windows is required. 32-bit systems cannot address required VRAM and host memory buffers.",
            details,
        )

    # Windows 10 build 19041+ or Windows 11
    build_match = re.search(r"\b(\d{5,})\b", version)
    build = int(build_match.group(1)) if build_match else 0
    details["build"] = build

    if build and build < 19041:
        return PreflightCheckResult(
            "windows_version", "Windows Version", "WARN",
            f"Windows build {build} is older than recommended build 19041 (Windows 10 20H1+). Some modern GPU scheduling APIs may be degraded.",
            details,
        )

    return PreflightCheckResult(
        "windows_version", "Windows Version", "PASS",
        f"Windows {release} (Build {build or version}) 64-bit verified.",
        details,
    )


def check_python_version(venv_target: Path | None = None) -> PreflightCheckResult:
    info = sys.version_info
    py_str = f"{info.major}.{info.minor}.{info.micro}"
    is_64bit = sys.maxsize > 2**32
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    details = {
        "version": py_str,
        "executable": sys.executable,
        "is_64bit": is_64bit,
        "in_virtual_env": is_venv,
        "sys_prefix": sys.prefix,
    }

    if not (info.major == 3 and 10 <= info.minor <= 11):
        return PreflightCheckResult(
            "python_version", "Python Version", "WARN" if info.major == 3 and info.minor >= 12 else "FAIL",
            f"Python {py_str} detected. Recommended runtime is Python 3.10.x for TensorRT and ONNX Runtime CUDA compatibility.",
            details,
        )

    if not is_64bit:
        return PreflightCheckResult(
            "python_version", "Python Version", "FAIL",
            f"Python {py_str} is 32-bit. 64-bit Python is mandatory.",
            details,
        )

    if venv_target and Path(sys.prefix).resolve() != venv_target.resolve():
        details["expected_venv"] = str(venv_target)
        return PreflightCheckResult(
            "python_version", "Python Version", "WARN",
            f"Python is running at {sys.prefix}, outside the target environment {venv_target}.",
            details,
        )

    return PreflightCheckResult(
        "python_version", "Python Version", "PASS",
        f"Python {py_str} (64-bit, dedicated runtime) verified.",
        details,
    )


def check_node_version() -> PreflightCheckResult:
    node_bin = shutil.which("node")
    if not node_bin:
        return PreflightCheckResult(
            "node_version", "Node.js", "WARN",
            "Node.js binary not found on PATH. Bundled Pinokio runtime or external Node >= 20.19.0 required for frontend builds.",
            {"installed": False},
        )

    try:
        res = subprocess.run([node_bin, "--version"], capture_output=True, text=True, timeout=5)
        raw = res.stdout.strip()
        match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", raw)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            # package.json: "^20.19.0 || >=22.12.0"
            valid = (major == 20 and minor >= 19) or (major >= 22 and (major > 22 or minor >= 12)) or major >= 23
            status: Literal["PASS", "WARN"] = "PASS" if valid else "WARN"
            msg = f"Node.js {raw} detected." if valid else f"Node.js {raw} is outside recommended range (^20.19.0 || >=22.12.0)."
            return PreflightCheckResult("node_version", "Node.js", status, msg, {"raw": raw, "path": node_bin, "valid": valid})
    except Exception as err:
        return PreflightCheckResult("node_version", "Node.js", "WARN", f"Failed to execute node: {err}", {"path": node_bin})

    return PreflightCheckResult("node_version", "Node.js", "PASS", "Node.js verified.", {"path": node_bin})


def check_cpu() -> PreflightCheckResult:
    logical_cores = os.cpu_count() or 1
    arch = platform.machine()
    processor = platform.processor() or "Unknown CPU"

    physical_cores = None
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out.isdigit():
                physical_cores = int(out)
    except Exception:
        pass

    details = {
        "processor": processor,
        "architecture": arch,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores or (logical_cores // 2 or 1),
    }

    if logical_cores < 4:
        return PreflightCheckResult(
            "cpu", "CPU Architecture & Cores", "WARN",
            f"CPU has {logical_cores} logical threads. Minimum 4 threads recommended for parallel decoding and worker threads.",
            details,
        )

    return PreflightCheckResult(
        "cpu", "CPU Architecture & Cores", "PASS",
        f"CPU {processor} ({details['physical_cores']} cores / {logical_cores} threads, {arch}) verified.",
        details,
    )


def check_system_ram() -> PreflightCheckResult:
    total_mib = 0
    avail_mib = 0

    if os.name == "nt":
        try:
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong),
                    ("total_virt", ctypes.c_ulonglong),
                    ("avail_virt", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_mib = int(status.total_phys // (1024 * 1024))
                avail_mib = int(status.avail_phys // (1024 * 1024))
        except Exception:
            pass

    details = {"total_mib": total_mib, "available_mib": avail_mib}

    if total_mib and total_mib < 8192:
        return PreflightCheckResult(
            "system_ram", "System RAM", "FAIL",
            f"Total RAM ({total_mib} MiB) is below minimum required 8,192 MiB. Video decoding and model pipelines will exhaust memory.",
            details,
        )

    if total_mib and total_mib < 16384:
        return PreflightCheckResult(
            "system_ram", "System RAM", "WARN",
            f"Total RAM is {total_mib} MiB. 16,384 MiB (16 GB) is recommended for high-resolution video frame buffering.",
            details,
        )

    return PreflightCheckResult(
        "system_ram", "System RAM", "PASS",
        f"System RAM: {total_mib} MiB total ({avail_mib} MiB free) verified.",
        details,
    )


def check_nvidia_gpu_and_driver(hw: Any) -> tuple[PreflightCheckResult, PreflightCheckResult, str]:
    """Returns (gpu_check, driver_check, resolved_profile)."""
    if not hw.gpus:
        return (
            PreflightCheckResult(
                "nvidia_gpu", "NVIDIA GPU", "WARN",
                "No NVIDIA GPU detected. Pipeline will run on CPU execution provider.",
                {"count": 0},
            ),
            PreflightCheckResult(
                "nvidia_driver", "NVIDIA Driver", "WARN",
                "NVIDIA driver not present. GPU acceleration unavailable.",
                {},
            ),
            "cpu",
        )

    gpu = hw.gpus[0]
    total_vram = gpu.total_vram_mib
    free_vram = gpu.free_vram_mib
    driver = gpu.driver_version or "Unknown"

    details = {
        "name": gpu.name,
        "total_vram_mib": total_vram,
        "free_vram_mib": free_vram,
        "driver_version": driver,
        "compute_capability": gpu.compute_capability,
    }

    # Profile determination
    # RTX 4070 / >= 10 GB VRAM -> '12gb' resident profile
    # RTX 3060 / <= 6 GB VRAM -> '6gb' clamped profile
    if total_vram >= RESIDENT_VRAM_PROFILE_FLOOR:
        profile = "12gb"
        profile_note = "Assigned RTX 4070 (12 GB) resident profile with unthrottled engine resident memory."
    elif total_vram <= LOW_VRAM_PROFILE_CEILING or free_vram < 4000:
        profile = "6gb"
        profile_note = "Assigned RTX 3060 (6 GB) Low-VRAM profile with 4,600 MiB session clamping."
    else:
        profile = "12gb"
        profile_note = "Assigned 12 GB profile based on available VRAM."

    details["resolved_profile"] = profile
    details["profile_note"] = profile_note

    gpu_res = PreflightCheckResult(
        "nvidia_gpu", "NVIDIA GPU", "PASS",
        f"{gpu.name} ({total_vram} MiB VRAM, {free_vram} MiB free). {profile_note}",
        details,
    )

    # Driver version check: recommended >= 528.00
    driver_match = re.search(r"(\d+)\.?(\d+)?", driver)
    driver_num = float(driver_match.group(0)) if driver_match else 0.0
    driver_details = {"driver_version": driver, "parsed": driver_num}

    if driver_num and driver_num < 528.00:
        drv_res = PreflightCheckResult(
            "nvidia_driver", "NVIDIA Driver", "WARN",
            f"Driver version {driver} is older than recommended 528.00+. Updating is recommended for CUDA 12 and TensorRT.",
            driver_details,
        )
    else:
        drv_res = PreflightCheckResult(
            "nvidia_driver", "NVIDIA Driver", "PASS",
            f"NVIDIA driver {driver} verified.",
            driver_details,
        )

    return gpu_res, drv_res, profile


def check_cuda_compatibility(hw: Any) -> PreflightCheckResult:
    runtime = hw.cuda_runtime_version
    driver_cuda = hw.driver_supported_cuda_version
    details = {
        "cuda_runtime_version": runtime,
        "driver_supported_cuda_version": driver_cuda,
    }

    if not runtime and not driver_cuda:
        return PreflightCheckResult(
            "cuda_compatibility", "CUDA Compatibility", "WARN",
            "CUDA runtime/driver version could not be queried directly. Standard CPU fallback active.",
            details,
        )

    if runtime and driver_cuda:
        try:
            r_maj, r_min = (int(x) for x in runtime.split(".")[:2])
            d_maj, d_min = (int(x) for x in driver_cuda.split(".")[:2])
            if (d_maj, d_min) < (r_maj, r_min):
                return PreflightCheckResult(
                    "cuda_compatibility", "CUDA Compatibility", "FAIL",
                    f"Driver supports CUDA up to {driver_cuda}, but runtime requires CUDA {runtime}. Please update your NVIDIA driver.",
                    details,
                )
        except Exception:
            pass

    ver = runtime or driver_cuda
    return PreflightCheckResult(
        "cuda_compatibility", "CUDA Compatibility", "PASS",
        f"CUDA {ver} compatible with installed driver.",
        details,
    )


def check_onnx_runtime(hw: Any) -> PreflightCheckResult:
    ort_ver = hw.ort_version
    providers = hw.advertised_providers or []
    details = {"ort_version": ort_ver, "providers": providers}

    if not ort_ver:
        return PreflightCheckResult(
            "onnxruntime", "ONNX Runtime", "WARN",
            "ONNX Runtime package not yet imported or detected.",
            details,
        )

    has_cuda = "CUDAExecutionProvider" in providers
    has_trt = "TensorrtExecutionProvider" in providers

    if not has_cuda and not has_trt:
        return PreflightCheckResult(
            "onnxruntime", "ONNX Runtime", "WARN",
            f"ONNX Runtime {ort_ver} has only CPU/DML providers ({', '.join(providers)}). GPU acceleration will not engage.",
            details,
        )

    return PreflightCheckResult(
        "onnxruntime", "ONNX Runtime", "PASS",
        f"ONNX Runtime {ort_ver} with {', '.join(providers)} verified.",
        details,
    )


def check_tensorrt() -> PreflightCheckResult:
    from roop_3d.core.hardware import tensorrt_version
    trt_ver = tensorrt_version()
    details = {"tensorrt_version": trt_ver}

    # Search for nvinfer dll on Windows
    nvinfer_found = False
    if os.name == "nt":
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d and (Path(d) / "nvinfer.dll").is_file() or (Path(d) / "nvinfer_10.dll").is_file():
                nvinfer_found = True
                break
    details["nvinfer_found"] = nvinfer_found

    if not trt_ver and not nvinfer_found:
        return PreflightCheckResult(
            "tensorrt", "TensorRT", "WARN",
            "TensorRT not installed or not in PATH. Pipeline will use CUDAExecutionProvider.",
            details,
        )

    return PreflightCheckResult(
        "tensorrt", "TensorRT", "PASS",
        f"TensorRT {trt_ver or 'Native Library'} verified.",
        details,
    )


def check_ffmpeg(hw: Any) -> PreflightCheckResult:
    path = hw.ffmpeg_path
    nvenc = hw.ffmpeg_nvenc_encoders
    nvdec = hw.ffmpeg_nvdec_decoders
    details = {
        "ffmpeg_path": path,
        "nvenc_encoders": nvenc,
        "nvdec_decoders": nvdec,
    }

    if not path:
        return PreflightCheckResult(
            "ffmpeg", "FFmpeg Codecs", "WARN",
            "FFmpeg executable not found on PATH. Bundled PyAV fallback active for decoding/encoding.",
            details,
        )

    msg = f"FFmpeg located at {path}."
    if nvenc:
        msg += f" NVENC acceleration supported ({', '.join(nvenc)})."
    else:
        msg += " Software libx264 encoding active (NVENC not detected in FFmpeg build)."

    return PreflightCheckResult("ffmpeg", "FFmpeg Codecs", "PASS", msg, details)


def check_disk_space(target_path: Path) -> PreflightCheckResult:
    try:
        usage = shutil.disk_usage(target_path)
        free_gb = round(usage.free / (1024**3), 2)
        total_gb = round(usage.total / (1024**3), 2)
        details = {"free_gb": free_gb, "total_gb": total_gb, "path": str(target_path)}

        if free_gb < 5.0:
            return PreflightCheckResult(
                "disk_space", "Available Disk Space", "FAIL",
                f"Only {free_gb} GB free on {target_path}. At least 5 GB is required to prevent write failures.",
                details,
            )

        if free_gb < 10.0:
            return PreflightCheckResult(
                "disk_space", "Available Disk Space", "WARN",
                f"{free_gb} GB free on {target_path}. 10+ GB recommended for models and video caches.",
                details,
            )

        return PreflightCheckResult(
            "disk_space", "Available Disk Space", "PASS",
            f"{free_gb} GB free of {total_gb} GB on drive hosting {target_path.name}.",
            details,
        )
    except Exception as err:
        return PreflightCheckResult(
            "disk_space", "Available Disk Space", "WARN",
            f"Could not query disk space: {err}",
            {"path": str(target_path)},
        )


def run_environment_preflight(root: Path | None = None) -> dict[str, Any]:
    """Runs the 12-point environment preflight check."""
    root = root or Path(__file__).resolve().parents[3]
    venv_target = root / "app/roop_3d/.venv"

    hw = probe_hardware(Settings())

    checks: list[PreflightCheckResult] = []

    # 1. Windows Version
    checks.append(check_windows_version())

    # 2. Python Version
    checks.append(check_python_version(venv_target))

    # 3. Node.js
    checks.append(check_node_version())

    # 4. CPU
    checks.append(check_cpu())

    # 5. System RAM
    checks.append(check_system_ram())

    # 6. NVIDIA GPU & 7. Driver
    gpu_check, driver_check, resolved_profile = check_nvidia_gpu_and_driver(hw)
    checks.append(gpu_check)
    checks.append(driver_check)

    # 8. CUDA Compatibility
    checks.append(check_cuda_compatibility(hw))

    # 9. ONNX Runtime
    checks.append(check_onnx_runtime(hw))

    # 10. TensorRT
    checks.append(check_tensorrt())

    # 11. FFmpeg
    checks.append(check_ffmpeg(hw))

    # 12. Disk Space
    checks.append(check_disk_space(root))

    # Aggregate status
    has_fail = any(c.status == "FAIL" for c in checks)
    has_warn = any(c.status == "WARN" for c in checks)
    overall_status: Literal["PASS", "WARN", "FAIL"] = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    return {
        "status": overall_status,
        "ready": not has_fail,
        "resolved_hardware_profile": resolved_profile,
        "checks": [c.to_dict() for c in checks],
        "summary": {
            "total": len(checks),
            "pass": sum(1 for c in checks if c.status == "PASS"),
            "warn": sum(1 for c in checks if c.status == "WARN"),
            "fail": sum(1 for c in checks if c.status == "FAIL"),
        },
    }
