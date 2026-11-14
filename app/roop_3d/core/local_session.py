"""Neutral local-ONNX session opener (graph safety checks + CUDA/DirectML/CPU selection).

Extracted from the removed face-swap adapter. It is model-agnostic: detectors, landmark nets, backbone encoders and
audio/lip-sync graphs all open their local ONNX graphs through it.
"""
from __future__ import annotations

from pathlib import Path

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig


_VALIDATED_GRAPHS: set[tuple[str, int, int]] = set()


def open_local_session(catalog, model_id: str, path: Path, config: RuntimeConfig):
    """Validate and load the real graph; retain exact diagnostics even when initialization fails."""
    from roop_3d.core.ort_session import CPU, CUDA, TRT, SessionManager, provider_chain, record_session_failure
    path = Path(path)
    model_name = path.name
    providers = provider_chain(config.provider)
    try:
        status = path.stat()
        version = (str(path.resolve()), status.st_size, status.st_mtime_ns)
        if version not in _VALIDATED_GRAPHS:
            import onnx
            from onnx import TensorProto
            graph = onnx.load(str(path), load_external_data=False)
            if any(item.data_location == TensorProto.EXTERNAL or item.external_data
                   for item in graph.graph.initializer):
                raise ValueError("external tensors are not supported")
            if any(node.domain not in ("", "ai.onnx", "ai.onnx.ml") for node in graph.graph.node):
                raise ValueError("custom operator domains are not supported")
            onnx.checker.check_model(graph)
            _VALIDATED_GRAPHS.add(version)
    except Exception as error:
        record_session_failure(model_name, path, error, requested_providers=providers)
        raise DomainError("LOCAL_MODEL_GRAPH_UNSUPPORTED",
                          f"{path}: {type(error).__name__}: {error}", 422) from error
    try:
        manager = SessionManager(hardware_profile="12gb" if config.hardware_profile == "12gb" else "6gb",
            fp16=config.precision_policy != "fp32", cuda_fp16=config.precision_policy == "fp16",
            gpu_memory_limit=config.gpu_budget_bytes if config.provider in (TRT, CUDA) else None,
            graph_optimization=config.graph_optimization)
        info = manager.create(path, model_name=model_name, providers=providers, intra_op_threads=config.cpu_threads)
    except Exception as error:
        record_session_failure(model_name, path, error, requested_providers=providers)
        raise DomainError("MODEL_SESSION_FAILED", f"{path}: {type(error).__name__}: {error}", 503) from error
    catalog.set_vram_loaded(model_id, info.provider != CPU)
    return info


def load_local_session(catalog, model_id: str, path: Path, config: RuntimeConfig):
    """The ONNX Runtime session alone; ``open_local_session`` also reports the provider and any fallback taken."""
    return open_local_session(catalog, model_id, path, config).session
