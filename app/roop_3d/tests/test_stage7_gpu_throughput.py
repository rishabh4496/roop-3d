"""Stage 7 dynamic reconstruction profiles, telemetry and VRAM governance."""
from __future__ import annotations

import numpy as np
import onnx
import pytest

pytest.importorskip("torch")

from roop_3d.core.settings import Settings
from roop_3d.async_pipeline.cuda_video import CUDA_STAGE_ORDER, CudaFramePipeline
from roop_3d.engine.vram_watchdog import VRAMWatchdog
from roop_3d.reconstruction.monocular_regressors import (
    DisentangledFaceEncoder,
    LipPhonemeRegressor,
)
from roop_3d.rendering.hybrid_renderer import ReducedBlendshapeBank
from roop_3d.trt_session_builder import BatchProfile, TensorRTBuildError, TensorRTSessionBuilder


def _bank():
    expression = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    atlas = np.zeros((3, 8, 8, 3), np.uint8)
    atlas[1, 2:6, 2:6] = 80
    atlas[2, 3:5, 1:7] = 120
    return ReducedBlendshapeBank.fit(expression, atlas)


def test_reconstruction_models_export_dynamic_batch_and_compile_profile(tmp_path, monkeypatch):
    encoder = DisentangledFaceEncoder(hidden_dim=16)
    lips = LipPhonemeRegressor(2, hidden_dim=16)
    bank = _bank()
    builder = TensorRTSessionBuilder(hardware_profile="12gb")
    observed = []

    def capture(path, *, profile):
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        assert profile == BatchProfile(tuple(range(1, 9)), 4)
        assert all(value.type.tensor_type.shape.dim[0].dim_param == "batch"
                   for value in graph.graph.input)
        observed.append(path.name)
        return path.name

    monkeypatch.setattr(builder, "build", capture)
    result = builder.build_reconstruction_models(disentangled_encoder=encoder, lip_regressor=lips,
        blendshape_bank=bank, artifact_directory=tmp_path, face_size=32, mouth_size=32)
    assert set(result) == {"disentangled_face_encoder", "lip_phoneme_regressor", "reduced_blendshape_bank"}
    assert len(observed) == 3


def test_reconstruction_trt_compiler_keeps_6gb_cpu_cuda_fallback():
    builder = TensorRTSessionBuilder(hardware_profile="6gb")
    with pytest.raises(TensorRTBuildError, match="12gb profile"):
        builder.build_reconstruction_models(disentangled_encoder=DisentangledFaceEncoder(hidden_dim=16),
            lip_regressor=LipPhonemeRegressor(2, hidden_dim=16), blendshape_bank=_bank())


def test_budget_caps_effective_gpu_memory_and_profile_startup():
    six_gb = Settings(hardware_profile="6gb", vram_budget_mb=3072)
    assert six_gb.vram_budget_mb == 3072
    watchdog = VRAMWatchdog()
    watchdog.configure_budget(3072, reserve_mb=512, profile="6gb")
    assert watchdog.current_micro_batch == 2
    assert watchdog.current_roi_resolution == 512
    adapted = watchdog.stepdown("simulated GPU pressure")
    assert adapted["micro_batch"] == 1
    assert adapted["roi_resolution"] == 384


def test_builder_observer_reports_all_supported_batches():
    events = []
    builder = TensorRTSessionBuilder(telemetry=events.append, hardware_profile="12gb")
    builder._emit("COMPILING_TRT", "lip_phoneme_regressor_dynamic.onnx", "FP16", tuple(range(1, 9)))
    assert events[0]["engine_precision"] == "FP16"
    assert events[0]["engine_batch_profile"] == list(range(1, 9))


def test_cuda_frame_ring_keeps_pipeline_pixels_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    stages = {name: (lambda item: item.payload + 1) for name in CUDA_STAGE_ORDER}
    source = [np.zeros((16, 16, 3), np.uint8)]
    with CudaFramePipeline(source, stages, shape=(16, 16, 3), ring_slots=2,
                           queue_size=2, vram_budget_mb=1024) as pipeline:
        output = next(pipeline.results())
        assert output.payload.is_cuda
        assert torch.all(output.payload == len(CUDA_STAGE_ORDER))
