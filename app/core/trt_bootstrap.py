"""Register packaged and system CUDA/TensorRT library directories before ORT loads.

The entrypoints call :func:`bootstrap_native_library_paths` before importing any
application module that can import ONNX Runtime. DLL directory handles are held
for the life of the process because Windows unloads a search directory when its
handle is closed.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import sysconfig
import threading

_lock = threading.RLock()
_dll_handles: dict[str, object] = {}
_so_handles: dict[str, object] = {}


def native_library_directories(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    """Find native-library folders in this Python environment and common CUDA installs."""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    app = root / "app"
    site_packages = {
        Path(value) for value in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")) if value
    }
    site_packages.update(
        Path(value)
        for value in (
            Path(sys.prefix) / "Lib" / "site-packages",
            app / "env" / "Lib" / "site-packages",
            app / "roop_3d" / ".venv" / "Lib" / "site-packages",
            root / ".venv" / "Lib" / "site-packages",
            root / "roop_3d" / ".venv" / "Lib" / "site-packages",
        )
    )
    for environment in (app / "roop_3d" / ".venv", root / ".venv", root / "roop_3d" / ".venv"):
        for lib_root in (environment / "lib", environment / "lib64"):
            if lib_root.is_dir():
                site_packages.update(lib_root.glob("python*/site-packages"))

    found: list[Path] = []
    found_keys: set[str] = set()

    def add_directory(candidate: Path) -> None:
        try:
            if candidate.is_dir():
                resolved = candidate.resolve()
                key = str(resolved).casefold() if os.name == "nt" else str(resolved)
                if key not in found_keys:
                    found.append(resolved)
                    found_keys.add(key)
        except OSError:
            return

    active_prefix = Path(sys.prefix).resolve()

    def site_order(site: Path) -> tuple[int, str]:
        try:
            active = site.resolve().is_relative_to(active_prefix)
        except OSError:
            active = False
        return (0 if active else 1, str(site).casefold())

    for site in sorted(site_packages, key=site_order):
        if not site.is_dir():
            continue
        for relative in (
            "tensorrt_libs", "tensorrt", "tensorrt/bin", "tensorrt/lib",
            "nvidia/cuda_runtime/bin", "nvidia/cuda_runtime/lib",
            "nvidia/cudnn/bin", "nvidia/cudnn/lib",
            "nvidia/cublas/bin", "nvidia/cublas/lib",
            "torch/lib", "onnxruntime/capi",
        ):
            candidate = site / relative
            add_directory(candidate)
        nvidia = site / "nvidia"
        if nvidia.is_dir():
            for component in nvidia.iterdir():
                if component.is_dir():
                    for subdir in ("bin", "lib", "lib64"):
                        candidate = component / subdir
                        add_directory(candidate)

    roots = [os.environ.get(key) for key in (
        "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "TENSORRT_HOME", "TENSORRT_PATH", "TENSORRT_ROOT", "TRT_ROOT",
        "CUDNN_PATH", "CUBLAS_PATH",
    )]
    for value in roots:
        if not value:
            continue
        library_root = Path(value).expanduser()
        for relative in ("", "bin", "lib", "lib64", "lib/x64"):
            candidate = library_root / relative
            add_directory(candidate)

    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if cuda_root.is_dir():
            for install in cuda_root.iterdir():
                candidate = install / "bin"
                add_directory(candidate)
        for base in (program_files / "NVIDIA Corporation", Path(r"C:\TensorRT")):
            if base.is_dir():
                for install in (base, *base.glob("TensorRT*")):
                    for relative in ("", "bin", "lib"):
                        candidate = install / relative
                        add_directory(candidate)
    else:
        for base in (
            Path("/usr/local/cuda"), Path("/usr/local"), Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/aarch64-linux-gnu"), Path("/usr/local/nvidia"),
            Path("/usr/lib/wsl/lib"), Path("/opt/TensorRT"), Path("/opt/tensorrt"),
        ):
            if not base.is_dir():
                continue
            for candidate in (base, base / "bin", base / "lib", base / "lib64"):
                add_directory(candidate)
            if base == Path("/usr/local"):
                for install in (*base.glob("cuda-*"), *base.glob("TensorRT*"), *base.glob("tensorrt*")):
                    for candidate in (install / "bin", install / "lib", install / "lib64"):
                        add_directory(candidate)

    return tuple(found)


def bootstrap_native_library_paths(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    """Prepend discovered library folders to PATH/LD_LIBRARY_PATH and register Windows DLL paths."""
    with _lock:
        paths = native_library_directories(repo_root)
        for variable in ("PATH", "LD_LIBRARY_PATH"):
            current = os.environ.get(variable, "").split(os.pathsep)
            seen = {value.rstrip("\\/").casefold() if os.name == "nt" else value.rstrip("/") for value in current if value}
            prefix: list[str] = []
            for path in paths:
                value = str(path)
                key = value.rstrip("\\/").casefold() if os.name == "nt" else value.rstrip("/")
                if key not in seen:
                    prefix.append(value)
                    seen.add(key)
            if prefix:
                os.environ[variable] = os.pathsep.join((*prefix, *(value for value in current if value)))

        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for path in paths:
                key = str(path).casefold()
                if key not in _dll_handles:
                    try:
                        _dll_handles[key] = os.add_dll_directory(str(path))
                    except (OSError, ValueError):
                        continue
        elif os.name != "nt":
            _preload_cuda_tensorrt_shared_libraries(paths)
        return paths


def _preload_cuda_tensorrt_shared_libraries(paths: tuple[Path, ...]) -> None:
    """Load CUDA/TensorRT dependencies globally; the ELF loader ignores late LD_LIBRARY_PATH edits."""
    prefixes = (
        "libcudart.so", "libcublas.so", "libcublasLt.so", "libcudnn", "libnvinfer",
        "libnvonnxparser", "libnvparsers", "libnvrtc", "libnvJitLink.so", "libnvToolsExt.so",
    )
    pending: list[Path] = []
    seen: set[str] = set()
    for directory in paths:
        for prefix in prefixes:
            for library in sorted(directory.glob(f"{prefix}*"), key=lambda path: path.name):
                key = str(library)
                if library.is_file() and key not in _so_handles and key not in seen:
                    pending.append(library)
                    seen.add(key)
    while pending:
        loaded: set[Path] = set()
        for library in pending:
            try:
                _so_handles[str(library)] = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                loaded.add(library)
            except OSError:
                continue
        if not loaded:
            return
        pending = [library for library in pending if library not in loaded]


def expected_nvinfer_filename(version: str | None = None) -> str:
    """Return the platform filename TensorRT's native inference library uses."""
    if version is None:
        try:
            import importlib.metadata
            for distribution in ("tensorrt", "tensorrt-cu12", "tensorrt-cu13", "tensorrt-cu11",
                                 "tensorrt-cu12-libs", "tensorrt-cu13-libs", "tensorrt-cu11-libs"):
                try:
                    version = importlib.metadata.version(distribution)
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
        except Exception:
            version = None
    major = (version or "").split(".", 1)[0]
    suffix = f"_{major}" if major.isdigit() else "_10"
    return f"nvinfer{suffix}.dll" if os.name == "nt" else f"libnvinfer.so.{major or '10'}"
