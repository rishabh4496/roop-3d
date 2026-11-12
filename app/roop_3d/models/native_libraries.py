"""Register packaged CUDA/TRT libraries before importing ONNX Runtime.

Changing LD_LIBRARY_PATH alone after Python starts is insufficient on Linux;
preload packaged shared objects with global visibility as well. No model loads.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sysconfig
import threading

_handles: dict[str, object] = {}
_lock = threading.RLock()


def packaged_library_directories() -> tuple[Path, ...]:
    sites = {Path(sysconfig.get_path(name)) for name in ("purelib", "platlib")}
    paths: set[Path] = set()
    for site in sites:
        for pattern in ("tensorrt_libs", "tensorrt*/lib", "tensorrt*/bin", "nvidia/*/lib",
                        "nvidia/*/bin", "torch/lib", "onnxruntime/capi"):
            paths.update(p for p in site.glob(pattern) if p.is_dir())
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda:
        paths.update(p for p in (Path(cuda) / "bin", Path(cuda) / "lib64") if p.is_dir())
    return tuple(sorted(paths))


def register_native_libraries() -> tuple[Path, ...]:
    with _lock:
        paths = packaged_library_directories()
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        current = os.environ.get(variable, "").split(os.pathsep)
        os.environ[variable] = os.pathsep.join(dict.fromkeys([*(str(p) for p in paths), *current]))
        if os.name == "nt":
            for path in paths:
                if str(path) not in _handles:
                    _handles[str(path)] = os.add_dll_directory(str(path))
        else:
            # Dependencies may appear in any directory; retry after loading peers.
            pending = {p for directory in paths for p in directory.glob("*.so*")
                       if p.is_file() and str(p) not in _handles
                       and not p.name.startswith("onnxruntime_pybind")}
            while pending:
                loaded = set()
                for path in sorted(pending):
                    try:
                        _handles[str(path)] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                        loaded.add(path)
                    except OSError:
                        continue  # Session construction reports unresolved native dependencies.
                if not loaded:
                    break
                pending -= loaded
        quiet_onnxruntime()
        return paths


_quiet = False


def quiet_onnxruntime() -> None:
    """Keep ONNX Runtime's *global* logger at ERROR so the Pinokio terminal stays readable.

    ``SessionOptions.log_severity_level`` does not reach the TensorRT EP, which logs through the process-wide
    default logger: a single cold start printed thousands of "Cache result detected as invalid" /
    "Detected layernorm nodes in FP16" warnings (written as wide characters, so also garbled) and buried the
    frame HUD.  ``ROOP3D_ORT_LOG_LEVEL`` (0 verbose .. 4 fatal, default 3) overrides.
    """
    global _quiet
    if _quiet:
        return
    try:
        import onnxruntime as ort
        level = int(os.environ.get("ROOP3D_ORT_LOG_LEVEL", "3"))
        ort.set_default_logger_severity(max(0, min(4, level)))
        _quiet = True
    except (ImportError, ValueError, AttributeError):
        pass
