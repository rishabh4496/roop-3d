"""Read-only hardware probes: provider advertisement is not execution proof."""
from __future__ import annotations

import csv
import ctypes
import ctypes.util
from datetime import datetime, timezone
from functools import lru_cache
import importlib.metadata
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel, Field

from roop_3d.core.settings import Settings


class GPU(BaseModel):
    index: int
    name: str
    total_vram_mib: int
    free_vram_mib: int
    driver_version: str
    compute_capability: str | None = None
    effective_vram_budget_mib: int = 0
    state: Literal["VERIFIED_LOCAL"] = "VERIFIED_LOCAL"


class Hardware(BaseModel):
    timestamp: str
    os: dict[str, str]
    logical_cpu_count: int | None
    physical_cpu_count: int | None = None
    physical_cpu_count_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    system_ram_total_mib: int | None = None
    system_ram_available_mib: int | None = None
    system_ram_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    gpus: list[GPU] = Field(default_factory=list)
    gpu_probe_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    advertised_providers: list[str] | None = None
    provider_probe_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    provider_execution_verified: bool = False
    ort_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_runtime_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    driver_supported_cuda_version: str | None = None
    ffmpeg_path: str | None = None
    ffmpeg_nvenc_encoders: list[str] = Field(default_factory=list)
    ffmpeg_nvdec_decoders: list[str] = Field(default_factory=list)
    ffmpeg_probe_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    ffmpeg_execution_verified: bool = False
    probe_notes: list[str] = Field(default_factory=list)
    # Flat summary of the active GPU (index 0) for dashboards and HUDs. Every value is
    # derived from the observations above; unknown stays None rather than guessed.
    gpu_name: str | None = None
    compute_capability: str | None = None
    vram_total_gb: float | None = None
    vram_available_gb: float | None = None
    cuda_version: str | None = None
    tensorrt_version: str | None = None
    onnxruntime_providers: list[str] = Field(default_factory=list)
    tensor_core_precisions: list[str] = Field(default_factory=list)
    tensor_core_state: Literal["INFERRED_FROM_COMPUTE_CAPABILITY", "UNKNOWN"] = "UNKNOWN"
    nvdec_available: bool | None = None
    nvenc_available: bool | None = None
    codec_hardware_state: Literal["FFMPEG_REGISTRATION_ONLY", "UNKNOWN"] = "UNKNOWN"
    # Stage 15: Intel QSV and OpenVINO hardware detection
    qsv_available: bool | None = None
    ffmpeg_qsv_encoders: list[str] = Field(default_factory=list)
    ffmpeg_qsv_decoders: list[str] = Field(default_factory=list)
    openvino_available: bool = False
    openvino_version: str | None = None
    openvino_devices: list[str] = Field(default_factory=list)
    heterogeneous_split_mode: bool = False


def effective_vram_budget(total: int, free: int, settings: Settings) -> int:
    if settings.hardware_profile == "cpu" or settings.preferred_provider == "CPUExecutionProvider":
        return 0
    profile = {"6gb": 6144, "12gb": 12288}.get(settings.hardware_profile, total)
    ceiling = min(total, profile, settings.vram_limit_mb or total)
    return max(0, min(free, ceiling) - settings.memory_safety_reserve_mb)


def tensor_core_precisions(compute_capability: str | None) -> list[str]:
    """Tensor Core math modes documented for each NVIDIA architecture generation.

    Inferred from the compute capability alone: it says what the silicon supports,
    not that a given model or engine runs in that precision.
    """
    if not compute_capability or not re.fullmatch(r"\d+[.]\d+", compute_capability):
        return []
    major, minor = (int(part) for part in compute_capability.split("."))
    version = (major, minor)
    modes: list[str] = []
    if version >= (7, 0):
        modes.append("fp16")
    if version >= (7, 5):
        modes.append("int8")
    if version >= (8, 0):
        modes += ["bf16", "tf32"]
    if version >= (8, 9):
        modes.append("fp8")
    return modes


@lru_cache(maxsize=1)
def tensorrt_version() -> str | None:
    """Installed TensorRT Python distribution version; no native library is loaded."""
    for name in ("tensorrt", "tensorrt-cu12", "tensorrt_cu12", "tensorrt-cu13", "tensorrt_cu13", "tensorrt-cu11", "tensorrt_cu11"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _summarize(result: Hardware) -> None:
    gpu = result.gpus[0] if result.gpus else None
    if gpu:
        result.gpu_name, result.compute_capability = gpu.name, gpu.compute_capability
        result.vram_total_gb = round(gpu.total_vram_mib / 1024, 2)
        result.vram_available_gb = round(gpu.free_vram_mib / 1024, 2)
        result.tensor_core_precisions = tensor_core_precisions(gpu.compute_capability)
        if result.tensor_core_precisions:
            result.tensor_core_state = "INFERRED_FROM_COMPUTE_CAPABILITY"
    result.cuda_version = result.cuda_runtime_version or result.driver_supported_cuda_version
    result.tensorrt_version = tensorrt_version()
    result.onnxruntime_providers = list(result.advertised_providers or [])
    if result.ffmpeg_probe_state == "VERIFIED_LOCAL":
        result.nvdec_available = bool(result.ffmpeg_nvdec_decoders)
        result.nvenc_available = bool(result.ffmpeg_nvenc_encoders)
        result.qsv_available = bool(result.ffmpeg_qsv_decoders or result.ffmpeg_qsv_encoders)
        result.codec_hardware_state = "FFMPEG_REGISTRATION_ONLY"


def _run(arguments: list[str]) -> str:
    return subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=True).stdout


def driver_cuda_version(summary: str) -> str | None:
    match = re.search(r"CUDA(?: UMD)? Version:\s*([\d.]+)", summary)
    return match.group(1) if match else None


def _system_memory(result: Hardware) -> None:
    try:
        if os.name == "nt":
            class Memory(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]]
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise OSError("Memory probe failed")
            total, available = memory.total, memory.available
            output = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                           "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"])
            result.physical_cpu_count = int(output.strip())
        elif platform.system() == "Linux":
            entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            total = int(entries["MemTotal"].split()[0]) * 1024
            available = int(entries["MemAvailable"].split()[0]) * 1024
            blocks = Path("/proc/cpuinfo").read_text().split("\n\n")
            cores = set()
            for block in blocks:
                fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                if "physical id" in fields and "core id" in fields:
                    cores.add((fields["physical id"], fields["core id"]))
            result.physical_cpu_count = len(cores) or None
        else:
            return
        result.system_ram_total_mib, result.system_ram_available_mib = total // 1048576, available // 1048576
        result.system_ram_state = "VERIFIED_LOCAL"
        if result.physical_cpu_count:
            result.physical_cpu_count_state = "VERIFIED_LOCAL"
    except (OSError, ValueError, subprocess.SubprocessError):
        result.probe_notes.append("System memory/core probe incomplete")


def probe_hardware(settings: Settings | None = None) -> Hardware:
    settings = settings or Settings()
    result = Hardware(timestamp=datetime.now(timezone.utc).isoformat(),
                      os={"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                      logical_cpu_count=os.cpu_count())
    runtime_library = (shutil.which("cudart64_12.dll") or shutil.which("cudart64_110.dll")) if os.name == "nt" else ctypes.util.find_library("cudart")
    if runtime_library:
        try:
            library = ctypes.CDLL(runtime_library)
            version = ctypes.c_int()
            function = library.cudaRuntimeGetVersion
            function.argtypes, function.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
            if function(ctypes.byref(version)) == 0 and version.value > 0:
                result.cuda_runtime_version = f"{version.value // 1000}.{(version.value % 1000) // 10}"
                result.cuda_runtime_state = "VERIFIED_LOCAL"
        except (OSError, AttributeError):
            result.probe_notes.append("CUDA runtime version query unavailable")
    command = shutil.which("nvidia-smi")
    if command:
        try:
            completed = subprocess.run([command, "--query-gpu=index,name,memory.total,memory.free,driver_version",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                       timeout=5, check=True)
            for row in csv.reader(completed.stdout.splitlines()):
                index, name, total, free, driver = [item.strip() for item in row]
                result.gpus.append(GPU(index=int(index), name=name, total_vram_mib=int(total),
                                       free_vram_mib=int(free), driver_version=driver))
            result.gpu_probe_state = "VERIFIED_LOCAL" if result.gpus else "UNKNOWN"
            try:
                capabilities = _run([command, "--query-gpu=index,compute_cap", "--format=csv,noheader,nounits"])
                by_index = {int(row[0]): row[1].strip() for row in csv.reader(capabilities.splitlines())}
                for gpu in result.gpus:
                    value = by_index.get(gpu.index)
                    gpu.compute_capability = value if value and re.fullmatch(r"\d+[.]\d+", value) else None
                summary = _run([command])
                result.driver_supported_cuda_version = driver_cuda_version(summary)
            except (OSError, ValueError, subprocess.SubprocessError):
                result.probe_notes.append("Compute capability/driver CUDA probe unavailable")
            for gpu in result.gpus:
                gpu.effective_vram_budget_mib = effective_vram_budget(gpu.total_vram_mib, gpu.free_vram_mib, settings)
        except (OSError, subprocess.SubprocessError, ValueError):
            result.gpus.clear()
            result.probe_notes.append("NVIDIA hardware probe failed; no GPU profile inferred")
    else:
        result.probe_notes.append("nvidia-smi unavailable; GPU fields are unknown")
    try:
        import onnxruntime as ort
        result.advertised_providers = ort.get_available_providers()
        result.ort_version = ort.__version__
        result.provider_probe_state = "VERIFIED_LOCAL"
    except Exception:
        result.probe_notes.append("ONNX Runtime import/probe failed; provider fields are unknown")
    _system_memory(result)
    result.ffmpeg_path = shutil.which("ffmpeg")
    if result.ffmpeg_path:
        try:
            encoders = _run([result.ffmpeg_path, "-hide_banner", "-encoders"])
            decoders = _run([result.ffmpeg_path, "-hide_banner", "-decoders"])
            result.ffmpeg_nvenc_encoders = sorted(set(re.findall(r"\b\w+_nvenc\b", encoders)))
            result.ffmpeg_nvdec_decoders = sorted(set(re.findall(r"\b\w+_cuvid\b", decoders)))
            result.ffmpeg_qsv_encoders = sorted(set(re.findall(r"\b\w+_qsv\b", encoders)))
            result.ffmpeg_qsv_decoders = sorted(set(re.findall(r"\b\w+_qsv\b", decoders)))
            result.ffmpeg_probe_state = "VERIFIED_LOCAL"
        except (OSError, subprocess.SubprocessError):
            result.probe_notes.append("FFmpeg codec inventory probe failed")
    _summarize(result)
    try:
        from roop_3d.openvino_coprocessor import get_openvino_device_info
        ov_info = get_openvino_device_info()
        result.openvino_available = ov_info.get("available", False)
        result.openvino_devices = ov_info.get("devices", [])
        result.openvino_version = ov_info.get("version")
    except Exception:
        result.probe_notes.append("OpenVINO hardware probe unavailable")
    if settings:
        result.heterogeneous_split_mode = getattr(settings, "heterogeneous_split_mode", False)
    result.probe_notes.append("Codec registration does not prove usable NVENC/NVDEC hardware")
    result.probe_notes.append("No inference sessions created; GPU execution is unverified")
    return result
