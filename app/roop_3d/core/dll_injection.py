"""Automatic CUDA / cuDNN / TensorRT DLL injection and directory bootstrap for Windows."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import List


def setup_environment(repo_root: Path | None = None) -> List[str]:
    """Create essential runtime directories and inject CUDA/TensorRT DLL paths on Windows.

    Must be called at the very top of entrypoints before `onnxruntime` is imported.
    """
    if repo_root is None:
        # Determine repo root from this file location: app/roop_3d/core/dll_injection.py -> parents[3] is root
        repo_root = Path(__file__).resolve().parents[3]

    app_root = repo_root / "app"

    # 1. Automatically ensure app/env, app/models, and app/facesets exist
    for sub in ("env", "models", "facesets"):
        target_dir = app_root / sub
        target_dir.mkdir(parents=True, exist_ok=True)

    injected: List[str] = []
    if sys.platform != "win32":
        return injected

    # 2. Search candidate site-packages directories for NVIDIA & TensorRT DLLs
    candidate_site_packages: List[Path] = [
        Path(sys.prefix) / "Lib" / "site-packages",
        app_root / "env" / "Lib" / "site-packages",
        app_root / "roop_3d" / ".venv" / "Lib" / "site-packages",
        repo_root / "roop_3d" / ".venv" / "Lib" / "site-packages",
    ]

    candidate_dll_dirs: List[Path] = []
    for sp in candidate_site_packages:
        if not sp.is_dir():
            continue

        # Check TensorRT libraries
        for trt_sub in ("tensorrt_libs", "tensorrt"):
            p = sp / trt_sub
            if p.is_dir() and p not in candidate_dll_dirs:
                candidate_dll_dirs.append(p)

        # Check nvidia/*/bin and nvidia/*/lib (e.g. cuda_runtime, cudnn, cublas)
        nvidia_root = sp / "nvidia"
        if nvidia_root.is_dir():
            for child in nvidia_root.iterdir():
                if child.is_dir():
                    for bin_name in ("bin", "lib"):
                        bin_path = child / bin_name
                        if bin_path.is_dir() and bin_path not in candidate_dll_dirs:
                            candidate_dll_dirs.append(bin_path)

    # Check system CUDA_PATH if defined
    cuda_env = os.environ.get("CUDA_PATH")
    if cuda_env:
        cuda_bin = Path(cuda_env) / "bin"
        if cuda_bin.is_dir() and cuda_bin not in candidate_dll_dirs:
            candidate_dll_dirs.append(cuda_bin)

    existing_path = os.environ.get("PATH", "")
    path_entries = set(p.strip().rstrip("\\/") for p in existing_path.split(os.pathsep) if p.strip())

    for d in candidate_dll_dirs:
        try:
            resolved_str = str(d.resolve())
            if resolved_str not in path_entries:
                os.environ["PATH"] = resolved_str + os.pathsep + os.environ["PATH"]
                path_entries.add(resolved_str)
                injected.append(resolved_str)
            if hasattr(os, "add_dll_directory") and d.is_dir():
                try:
                    os.add_dll_directory(resolved_str)
                    if resolved_str not in injected:
                        injected.append(resolved_str)
                except (OSError, ValueError):
                    pass
        except Exception:
            continue

    return injected
