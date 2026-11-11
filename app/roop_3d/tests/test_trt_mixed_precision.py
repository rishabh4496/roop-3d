"""TensorRT mixed precision policy, fallback chain, and opt-in GPU qualification."""
from __future__ import annotations

from pathlib import Path
import os
import time
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest

from roop_3d.execution.trt_mixed_precision import (
    CPU,
    CUDA,
    TRT,
    FP16_TENSOR_OPS,
    FP32_PINNED_OPS,
    TensorRTMixedPrecisionSessionManager,
    derive_optimization_profiles,
    prepare_mixed_precision_model,
)


def _dynamic_model(path: Path, *, shape=(None, 3, 16, 16), op="Identity") -> Path:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))
    graph = helper.make_graph([helper.make_node(op, ["input"], ["output"])], "fixture",
                              [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


class _FakeSessionOptions:
    def __init__(self):
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class _FakeBinding:
    def __init__(self, session):
        self.session = session
        self.inputs = {}
        self.output_names = []
        self.results = []

    def clear_binding_inputs(self):
        self.inputs.clear()

    def clear_binding_outputs(self):
        self.output_names.clear()

    def bind_cpu_input(self, name, value):
        self.inputs[name] = value

    def bind_ortvalue_input(self, name, value):
        failure = _FakeInferenceSession.failures.get((self.session.provider, "bind_input"))
        if failure:
            raise failure
        self.inputs[name] = value

    def bind_input(self, name, *value):
        self.inputs[name] = value

    def bind_ortvalue_output(self, name, value):
        self.output_names.append(name)

    def bind_output(self, name, *value):
        self.output_names.append((name, value[0] if value else "cpu"))

    def synchronize_outputs(self):
        pass

    def synchronize_inputs(self):
        pass

    def get_outputs(self):
        return self.results


class _FakeInferenceSession:
    created = []
    failures = {}

    def __init__(self, model_path, *, sess_options, providers):
        self.model_path, self.options = model_path, sess_options
        self.requested_providers = providers
        first = providers[0]
        self.provider = first[0] if isinstance(first, tuple) else first
        self.provider_options = first[1] if isinstance(first, tuple) else {}
        self.providers = [self.provider]
        if self.provider == TRT:
            self.providers.append(CUDA)
        self.runs = 0
        self.created.append(self)
        failure = self.failures.get((self.provider, "create"))
        if failure:
            raise failure

    def get_providers(self):
        return list(self.providers)

    def disable_fallback(self):
        pass

    def run(self, output_names, input_feed, **kwargs):
        self.runs += 1
        failure = self.failures.get((self.provider, "run"))
        if failure:
            raise failure
        return [input_feed.get("input", np.array([1], dtype=np.float32))]

    def io_binding(self):
        return _FakeBinding(self)

    def run_with_iobinding(self, binding, **kwargs):
        failure = self.failures.get((self.provider, "binding"))
        if failure:
            raise failure
        binding.results = [_FakeOrtValue(np.array([1], dtype=np.float32), device)
                           for _name, device in binding.output_names]


class _FakeOrtValue:
    def __init__(self, array, device="cpu"):
        self._array, self._device = np.asarray(array), device

    def device_name(self):
        return self._device

    def numpy(self):
        return self._array

    def update_inplace(self, array):
        self._array = np.array(array, copy=True)


class _FakeOrt:
    SessionOptions = _FakeSessionOptions
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=99)
    InferenceSession = _FakeInferenceSession
    OrtValue = SimpleNamespace(ortvalue_from_numpy=staticmethod(
        lambda value, device, device_id=0: _FakeOrtValue(value, device)))

    @staticmethod
    def get_available_providers():
        return [TRT, CUDA, CPU]


@pytest.fixture(autouse=True)
def reset_fake_sessions():
    _FakeInferenceSession.created = []
    _FakeInferenceSession.failures = {}


def _manager(tmp_path, **kwargs):
    return TensorRTMixedPrecisionSessionManager(
        cache_path=tmp_path / "models" / ".trt_cache",
        ort_module=_FakeOrt,
        session_factory=_FakeInferenceSession,
        **kwargs,
    )


def test_profile_shapes_bind_batch_1_4_8_for_supported_dynamic_model(tmp_path):
    model = _dynamic_model(tmp_path / "detector.onnx", shape=(None, 3, 640, 640))

    profiles = derive_optimization_profiles(model, "Detector")

    assert profiles.dynamic_batch
    assert profiles.minimum["input"] == (1, 3, 640, 640)
    assert profiles.optimal["input"] == (4, 3, 640, 640)
    assert profiles.maximum["input"] == (8, 3, 640, 640)
    assert profiles.as_options() == {
        "trt_profile_min_shapes": "input:1x3x640x640",
        "trt_profile_opt_shapes": "input:4x3x640x640",
        "trt_profile_max_shapes": "input:8x3x640x640",
    }


@pytest.mark.parametrize(("name", "shape"), [
    ("ArcFace", (None, 3, 112, 112)),
    ("SMIRK Encoder", (None, 3, 224, 224)),
    ("Inswapper", (None, 3, 128, 128)),
    ("3D Texture Prior", (None, 3, 256, 256)),
    ("GPEN Realistic", (None, 3, 512, 512)),
    ("UltraMax", (None, 3, 512, 512)),
    ("Background Inpainter", (None, 3, 512, 512)),
])
def test_family_profile_preserves_model_spatial_contract(tmp_path, name, shape):
    path = _dynamic_model(tmp_path / "model.onnx", shape=shape)
    profile = derive_optimization_profiles(path, name)
    assert profile.minimum["input"][1:] == shape[1:]
    assert profile.optimal["input"][0] == 4 and profile.maximum["input"][0] == 8


def test_static_batch_model_does_not_get_an_invalid_wide_profile(tmp_path):
    path = _dynamic_model(tmp_path / "static.onnx", shape=(1, 3, 128, 128))

    profile = derive_optimization_profiles(path, "Inswapper")

    assert not profile.dynamic_batch
    assert profile.as_options() == {}


def test_static_batch_with_dynamic_spatial_axes_gets_a_fixed_shape_profile(tmp_path):
    path = _dynamic_model(tmp_path / "dynamic-spatial.onnx", shape=(1, 3, None, None))

    profile = derive_optimization_profiles(path, "Inswapper")

    assert not profile.dynamic_batch
    assert profile.minimum == profile.optimal == profile.maximum == {"input": (1, 3, 128, 128)}


def test_fp32_sensitive_ops_are_islands_and_tensor_ops_convert_to_fp16(tmp_path):
    vertices = helper.make_tensor_value_info("vertices", TensorProto.FLOAT, [None, 3])
    projection = helper.make_tensor_value_info("projection", TensorProto.FLOAT, [None, 3])
    weights = helper.make_tensor("weights", TensorProto.FLOAT, [3, 3],
                                 [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    exponent = helper.make_tensor("exponent", TensorProto.FLOAT, [], [1.0])
    graph = helper.make_graph([
        helper.make_node("MatMul", ["vertices", "weights"], ["linear"]),
        helper.make_node("Pow", ["linear", "exponent"], ["projection"]),
    ], "3d_projection", [vertices], [projection], [weights, exponent])
    source = tmp_path / "projection.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), source)

    mixed, counts = prepare_mixed_precision_model(source, tmp_path / "cache")
    model = onnx.load(mixed)
    value_types = {value.name: value.type.tensor_type.elem_type
                   for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    node_types = {node.op_type for node in model.graph.node}
    matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
    pow_node = next(node for node in model.graph.node if node.op_type == "Pow")

    assert FP16_TENSOR_OPS == {"Conv", "ConvTranspose", "Gemm", "MatMul"}
    assert FP32_PINNED_OPS == {"RodriguesRotation", "SphericalHarmonics", "LayerNormalization",
                               "InstanceNormalization", "Pow", "ReduceMean", "Softmax"}
    assert counts == {"Pow": 1}
    assert "MatMul" in node_types and "Cast" in node_types
    assert value_types[matmul.output[0]] == TensorProto.FLOAT16
    assert all(value_types.get(name) != TensorProto.FLOAT16 for name in pow_node.input if name in value_types)
    assert all(value_types.get(name) != TensorProto.FLOAT16 for name in pow_node.output if name in value_types)


def test_all_tensor_core_operator_families_convert_to_fp16(tmp_path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2, 8, 8])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3])
    conv_weights = helper.make_tensor("conv_weights", TensorProto.FLOAT, [2, 2, 3, 3],
                                      np.full((2, 2, 3, 3), 0.05, np.float32).ravel().tolist())
    conv_bias = helper.make_tensor("conv_bias", TensorProto.FLOAT, [2], [0.01, -0.02])
    trans_weights = helper.make_tensor("trans_weights", TensorProto.FLOAT, [2, 2, 3, 3],
                                       np.full((2, 2, 3, 3), 0.02, np.float32).ravel().tolist())
    trans_bias = helper.make_tensor("trans_bias", TensorProto.FLOAT, [2], [0.0, 0.0])
    gemm_weights = helper.make_tensor("gemm_weights", TensorProto.FLOAT, [128, 16],
                                      np.full((128, 16), 0.01, np.float32).ravel().tolist())
    gemm_bias = helper.make_tensor("gemm_bias", TensorProto.FLOAT, [16], np.zeros(16, np.float32).tolist())
    matmul_weights = helper.make_tensor("matmul_weights", TensorProto.FLOAT, [16, 3],
                                        np.full((16, 3), 0.03, np.float32).ravel().tolist())
    graph = helper.make_graph([
        helper.make_node("Conv", ["input", "conv_weights", "conv_bias"], ["conv"], kernel_shape=[3, 3]),
        helper.make_node("ConvTranspose", ["conv", "trans_weights", "trans_bias"], ["trans"],
                         kernel_shape=[3, 3]),
        helper.make_node("Flatten", ["trans"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "gemm_weights", "gemm_bias"], ["dense"]),
        helper.make_node("MatMul", ["dense", "matmul_weights"], ["output"]),
    ], "tensor_core_ops", [input_info], [output_info],
        [conv_weights, conv_bias, trans_weights, trans_bias, gemm_weights, gemm_bias, matmul_weights])
    source = tmp_path / "tensor_ops.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), source)

    mixed, _ = prepare_mixed_precision_model(source, tmp_path / "cache")
    model = onnx.load(mixed)

    found = {node.op_type for node in model.graph.node} & FP16_TENSOR_OPS
    assert found == FP16_TENSOR_OPS


def test_tensor_rt_options_and_cache_path_are_explicit(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "detector.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))

    session = _manager(tmp_path).create(path, model_name="Detector")

    assert session.provider == TRT
    opts = _FakeInferenceSession.created[0].provider_options
    assert opts["trt_fp16_enable"] == "True"
    assert opts["trt_engine_cache_enable"] == "True"
    assert Path(opts["trt_engine_cache_path"]) == tmp_path / "models" / ".trt_cache"
    assert opts["trt_timing_cache_enable"] == "True"
    assert opts["trt_builder_optimization_level"] == "3"
    assert opts["trt_profile_min_shapes"] == "input:1x3x16x16"
    assert opts["trt_profile_opt_shapes"] == "input:4x3x16x16"
    assert opts["trt_profile_max_shapes"] == "input:8x3x16x16"


def test_provider_chain_can_share_the_torch_compute_stream(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "model.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))

    _manager(tmp_path, user_compute_stream=123456).create(path)

    trt_options = _FakeInferenceSession.created[0].provider_options
    cuda_options = _FakeInferenceSession.created[0].requested_providers[1][1]
    assert trt_options["user_compute_stream"] == "123456"
    assert cuda_options["user_compute_stream"] == "123456"


def test_initialization_failure_falls_back_to_cuda_with_exact_reason(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "model.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))
    _FakeInferenceSession.failures[(TRT, "create")] = RuntimeError("TensorRT engine parser rejected node Pow")

    session = _manager(tmp_path).create(path)

    assert session.provider == CUDA
    assert any("TensorRT engine parser rejected node Pow" in reason for reason in session.fallback_history)


def test_runtime_failure_cascades_from_trt_through_cuda_to_cpu(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "model.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))
    _FakeInferenceSession.failures[(TRT, "run")] = RuntimeError("TRT kernel launch failed: driver error 700")
    _FakeInferenceSession.failures[(CUDA, "run")] = RuntimeError("CUDA EP cuDNN initialization failed")
    manager = _manager(tmp_path)

    session = manager.create(path)
    result = session.run(None, {"input": np.ones((1, 3, 16, 16), dtype=np.float32)})

    assert session.provider == CPU
    assert result[0].shape == (1, 3, 16, 16)
    assert any("TRT kernel launch failed: driver error 700" in item for item in session.fallback_history)
    assert any("CUDA EP cuDNN initialization failed" in item for item in session.fallback_history)


def test_iobinding_runtime_failure_replays_inputs_after_cpu_fallback(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "model.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))
    _FakeInferenceSession.failures[(TRT, "binding")] = RuntimeError("TensorRT kernel launch failed")
    _FakeInferenceSession.failures[(CUDA, "binding")] = RuntimeError("CUDA driver context was lost")
    session = _manager(tmp_path).create(path)
    binding = session.io_binding()
    binding.bind_cpu_input("input", np.ones((1, 3, 16, 16), dtype=np.float32))
    binding.bind_output("output", "cuda", 0)

    session.run_with_iobinding(binding)

    assert session.provider == CPU
    assert binding.get_outputs()[0].device_name() == "cpu"
    assert any("TensorRT kernel launch failed" in item for item in session.fallback_history)
    assert any("CUDA driver context was lost" in item for item in session.fallback_history)


def test_iobinding_bind_failure_falls_back_before_execution(tmp_path, monkeypatch):
    path = _dynamic_model(tmp_path / "model.onnx")
    monkeypatch.setattr("roop_3d.execution.trt_mixed_precision.prepare_mixed_precision_model",
                        lambda source, cache: (Path(source), {}))
    _FakeInferenceSession.failures[(TRT, "bind_input")] = RuntimeError("TRT device pointer bind failed")
    _FakeInferenceSession.failures[(CUDA, "bind_input")] = RuntimeError("CUDA context is gone")
    session = _manager(tmp_path).create(path)
    binding = session.io_binding()
    binding.bind_cpu_input("input", np.ones((1, 3, 16, 16), dtype=np.float32))
    binding.bind_output("output", "cuda", 0)

    session.run_with_iobinding(binding)

    assert session.provider == CPU
    assert binding.get_outputs()[0].device_name() == "cpu"
    assert any("TRT device pointer bind failed" in item for item in session.fallback_history)
    assert any("CUDA context is gone" in item for item in session.fallback_history)


def test_onnx_vertex_projection_drift_stays_under_point_one_percent(tmp_path):
    # Run the actual converted ONNX graph on CPU so this remains a useful
    # numerical gate when the optional CUDA/TensorRT hardware gate is skipped.
    axis = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
    vertices = np.stack(np.meshgrid(axis, axis, axis[::8], indexing="ij"), axis=-1).reshape(-1, 3)
    matrix = np.array([[0.91, -0.12, 0.04], [0.07, 1.03, -0.09], [-0.03, 0.11, 0.97]], dtype=np.float32)
    input_info = helper.make_tensor_value_info("vertices", TensorProto.FLOAT, [None, 3])
    output_info = helper.make_tensor_value_info("projected", TensorProto.FLOAT, [None, 3])
    matrix_tensor = helper.make_tensor("projection", TensorProto.FLOAT, [3, 3], matrix.ravel().tolist())
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["vertices", "projection"], ["projected"])],
        "3d_vertex_projection", [input_info], [output_info], [matrix_tensor])
    source = tmp_path / "vertex_projection.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), source)
    mixed_path, _ = prepare_mixed_precision_model(source, tmp_path / "mixed_cache")
    fp32_session = ort.InferenceSession(str(source), providers=[CPU])
    mixed_session = ort.InferenceSession(str(mixed_path), providers=[CPU])
    fp32 = fp32_session.run(None, {"vertices": vertices})[0]
    mixed = mixed_session.run(None, {"vertices": vertices})[0]
    relative = np.max(np.abs(fp32 - mixed)) / max(float(np.max(np.abs(fp32))), 1e-12)

    assert relative < 0.001


def _gpu_runtime_available() -> bool:
    if os.environ.get("ROOP_RUN_GPU_BENCHMARKS") != "1":
        return False
    try:
        import torch
        import onnxruntime as ort
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and TRT in ort.get_available_providers()
                and CUDA in ort.get_available_providers())


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_runtime_available(), reason="set ROOP_RUN_GPU_BENCHMARKS=1 with CUDA, TensorRT, and PyTorch")
def test_real_trt_projection_drift_and_tensor_core_speedup(tmp_path):
    """Hardware gate: mixed TensorRT should keep vertex drift <0.1% and exceed 2.5x throughput."""
    import onnxruntime as ort

    input_info = helper.make_tensor_value_info("vertices", TensorProto.FLOAT, [None, 224, 3])
    output_info = helper.make_tensor_value_info("projected", TensorProto.FLOAT, [None, 224, 3])
    weights = helper.make_tensor("projection", TensorProto.FLOAT, [3, 3],
                                 [0.91, -0.12, 0.04, 0.07, 1.03, -0.09, -0.03, 0.11, 0.97])
    graph = helper.make_graph([helper.make_node("MatMul", ["vertices", "projection"], ["projected"])],
                              "vertex_projection", [input_info], [output_info], [weights])
    source = tmp_path / "smirk_projection.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), source)
    mixed_session = TensorRTMixedPrecisionSessionManager(
        cache_path=tmp_path / "trt_cache", hardware_profile="12gb").create(source, model_name="SMIRK Encoder")
    fp32_session = ort.InferenceSession(str(source), providers=[(CUDA, {"device_id": 0})])
    rng = np.random.default_rng(14)
    vertices = rng.uniform(-1.0, 1.0, (4, 224, 3)).astype(np.float32)
    reference = fp32_session.run(None, {"vertices": vertices})[0]
    actual = mixed_session.run(None, {"vertices": vertices})[0]
    drift = float(np.max(np.abs(reference - actual)) / max(float(np.max(np.abs(reference))), 1e-12))
    assert drift < 0.001

    heavy_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 256, 256])
    heavy_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 64, 254, 254])
    filters = np.random.default_rng(3).normal(0, 0.02, (64, 3, 3, 3)).astype(np.float32)
    filter_tensor = helper.make_tensor("filters", TensorProto.FLOAT, filters.shape, filters.ravel().tolist())
    heavy_graph = helper.make_graph([
        helper.make_node("Conv", ["input", "filters"], ["conv"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["conv"], ["output"]),
    ], "throughput", [heavy_input], [heavy_output], [filter_tensor])
    heavy_model = tmp_path / "gpen_realistic_benchmark.onnx"
    onnx.save(helper.make_model(heavy_graph, opset_imports=[helper.make_opsetid("", 13)]), heavy_model)
    trt = TensorRTMixedPrecisionSessionManager(
        cache_path=tmp_path / "trt_cache", hardware_profile="12gb").create(heavy_model, model_name="GPEN Realistic")
    cuda = ort.InferenceSession(str(heavy_model), providers=[(CUDA, {"device_id": 0})])
    sample = np.random.default_rng(4).normal(size=(4, 3, 256, 256)).astype(np.float32)

    def latency(session, iterations=30):
        for _ in range(8):
            session.run(None, {"input": sample})
        measurements = []
        for _ in range(iterations):
            started = time.perf_counter()
            session.run(None, {"input": sample})
            measurements.append(time.perf_counter() - started)
        return float(np.median(measurements))

    speedup = latency(cuda) / latency(trt)
    assert speedup > 2.5, f"Expected >2.5x TensorRT throughput, measured {speedup:.2f}x"
