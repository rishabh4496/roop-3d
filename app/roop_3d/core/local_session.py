"""Neutral local-ONNX session opener (graph safety checks + provider selection + CUDA/TRT fallback chain).

Extracted from the removed face-swap adapter. It is model-agnostic: detectors, landmark nets, backbone encoders and
audio/lip-sync graphs all open their local ONNX graphs through it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig

_LOG = logging.getLogger("roop_3d.local_session")


_VALIDATED_GRAPHS: set[tuple[str, int, int]] = set()


def open_local_session(catalog, model_id: str, path: Path, config: RuntimeConfig):
    """Validate a local graph and load it within the selected hardware profile; returns the ``SessionInfo``."""
    try:
        status = Path(path).stat()
        version = (str(path), status.st_size, status.st_mtime_ns)
    except OSError:
        version = None
    if version is None or version not in _VALIDATED_GRAPHS:
        # Parsing and checking a 500 MB graph took seconds on every preview; one pass per file version is enough.
        try:
            import onnx
            from onnx import TensorProto
            graph = onnx.load(str(path), load_external_data=False)
            if any(item.data_location == TensorProto.EXTERNAL or item.external_data
                   for item in graph.graph.initializer):
                raise ValueError("external tensors are not supported")
            if any(node.domain not in ("", "ai.onnx", "ai.onnx.ml") for node in graph.graph.node):
                raise ValueError("custom operator domains are not supported")
            onnx.checker.check_model(graph)
        except Exception as error:
            raise DomainError("LOCAL_MODEL_GRAPH_UNSUPPORTED",
                f"Local ONNX model graph failed safety/format checks ({type(error).__name__})", 422) from None
        if version is not None:
            _VALIDATED_GRAPHS.add(version)
    from roop_3d.core.ort_session import CUDA, CPU, TRT, ProviderInitializationError, SessionManager
    model_name = path.name
    if config.provider == CUDA and config.precision_policy == "fp16":
        from roop_3d.core.ort_session import _cuda_fp16_model_path
        fp16_cache = catalog.root / "roop_3d/.runtime/cache/cuda_fp16"
        try:
            path, _ = _cuda_fp16_model_path(path, fp16_cache)
        except Exception as error:
            # The mixed-precision rewrite is an optimisation; a graph it cannot convert still runs as its own dtype.
            _LOG.warning("[ORT] %s: CUDA FP16 graph conversion failed (%s); using the file as shipped", model_name, type(error).__name__)
    providers = ((TRT, CUDA) if config.provider == TRT else
                 (CUDA,) if config.provider == CUDA else (CPU,))
    if config.provider in (TRT, CUDA):
        # SessionManager disables CPU EP fallback and keeps unsupported nodes on CUDA.
        manager = SessionManager(hardware_profile=config.hardware_profile,
            fp16=config.provider == TRT and config.precision_policy != 'fp32', gpu_memory_limit=config.gpu_budget_bytes)
    else:
        manager = SessionManager(hardware_profile="6gb", fp16=False)
    try:
        # A model never goes offline because TensorRT cannot build (cache rebuild, missing EP, bad option) or CUDA cannot
        # place a node: the next provider is tried and the CPU is the logged last resort (``fallback_reason`` in capability()).
        info = manager.create(path, model_name=model_name, providers=providers,
                              intra_op_threads=config.cpu_threads, cpu_last_resort=True)
    except ProviderInitializationError as error:
        # Every provider's failure, in order (type only: ORT messages can carry local paths).
        attempts = "; ".join(": ".join(entry.split(": ", 2)[:2]) for entry in error.attempts)
        raise DomainError("MODEL_SESSION_FAILED",
            f"Local ONNX model could not initialize on the selected provider ({attempts})", 503) from None
    except Exception as error:
        raise DomainError("MODEL_SESSION_FAILED",
            f"Local ONNX model could not initialize on the selected provider ({type(error).__name__})", 503) from None
    catalog.set_vram_loaded(model_id, info.provider in (TRT, CUDA))
    return info


def load_local_session(catalog, model_id: str, path: Path, config: RuntimeConfig):
    """The ONNX Runtime session alone; ``open_local_session`` also reports the provider and any fallback taken."""
    return open_local_session(catalog, model_id, path, config).session
