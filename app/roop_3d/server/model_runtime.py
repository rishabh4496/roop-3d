"""Build the same local TensorRT sessions used by preview and render."""
from __future__ import annotations

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig


def compile_model(catalog, model_id: str, config: RuntimeConfig) -> dict:
    from roop_3d.engine.volumetric.swapper_engine import load_local_session
    from roop_3d.core.ort_session import GpuIoBinding
    from roop_3d.models.dynamic_shapes import sample_inputs
    from roop_3d.trt_session_builder import auto_shape_profiles
    import onnx

    path = catalog.resolve_local_model(model_id)
    session = load_local_session(catalog, model_id, path, config)
    try:
        if session.get_providers()[0] != 'TensorrtExecutionProvider':
            raise DomainError('TRT_BUILD_UNAVAILABLE', 'TensorRT session initialization failed; the model used CUDA instead', 409)
        graph = onnx.load(str(path), load_external_data=False)
        inputs = sample_inputs(graph, 1, auto_shape_profiles(graph, path.name)[1])
        outputs = GpuIoBinding(session).run(inputs)
        if not outputs or not all(np.isfinite(value.numpy()).all() for value in outputs.values()):
            raise DomainError('TRT_CANARY_FAILED', 'TensorRT returned non-finite canary output', 409)
        options = session.get_provider_options().get('TensorrtExecutionProvider', {})
        return {'model_id': model_id, 'status': 'compiled', 'provider': session.get_providers()[0],
                'requested_precision': config.precision_policy,
                'fp16_enabled': str(options.get('trt_fp16_enable', '0')).lower() in ('1', 'true'),
                'canary_verified': True}
    finally:
        catalog.set_vram_loaded(model_id, False)
        del session
