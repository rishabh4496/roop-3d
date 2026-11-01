"""Stage 2: actual CPU graph validation and simulated native TRT failure paths.

Fake sessions exercise orchestration only; they are not GPU compilation evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

from roop_3d.models.dynamic_shapes import (DynamicShapeError, StaticBatchRequired, convert, input_shapes,
                                          static_batch_reason, surgery, validate_conversion)
from roop_3d.models.owned_files import owned_path
from roop_3d.trt_session_builder import (BatchProfile, StaticBatchError, TensorRTBuildError,
                                        TensorRTSessionBuilder, precision_for)


def graph(*, dynamic=False, codeformer=False, fixed_reshape=True):
    h, t = onnx.helper, onnx.TensorProto
    inputs = [h.make_tensor_value_info("image", t.FLOAT, ["batch" if dynamic else 1, 3, 2, 2])]
    initializers = [onnx.numpy_helper.from_array(np.array([1, 12] if fixed_reshape else [0, 12], np.int64), "shape")]
    nodes = [h.make_node("Reshape", ["image", "shape"], ["flat"])]
    if codeformer:
        inputs.append(h.make_tensor_value_info("w", t.FLOAT, [1]))
        nodes.append(h.make_node("Mul", ["flat", "w"], ["output"]))
    else:
        nodes.append(h.make_node("Identity", ["flat"], ["output"]))
    return h.make_model(h.make_graph(nodes, "codeformer" if codeformer else "test", inputs,
                        [h.make_tensor_value_info("output", t.FLOAT, ["batch" if dynamic else 1, 12])], initializers),
                        opset_imports=[h.make_opsetid("", 17)])


@pytest.mark.parametrize("name", ["inswapper_128.onnx", "inswapper_128_dynamic.onnx", "INSWAPPER_128_DYNAMIC.ONNX"])
def test_swapper_fp32_overflow_guard(name):
    assert precision_for(name, {}) == "FP32"
    assert precision_for(name, {"ROOP_SWAP_FP16": "1"}) == "FP16"
    assert precision_for(name, {"ROOP_SWAP_FP16": "1", "ROOP_TRT_FP16": "0"}) == "FP32"


@pytest.mark.parametrize("name", ["scrfd_2.5g_kps.onnx", "det_10g_dynamic.onnx", "GPEN-BFR-512.onnx", "CodeFormer.onnx"])
def test_detector_enhancer_precision_defaults(name):
    assert precision_for(name, {}) == "FP16"
    assert precision_for(name, {"ROOP_TRT_FP16": "0"}) == "FP32"


def test_actual_reshape_surgery_and_finite_parity(tmp_path):
    source = tmp_path / "inswapper_128.onnx"
    onnx.save(graph(), source)
    original = source.read_bytes()
    destination = convert(source)
    assert source.read_bytes() == original
    report = json.loads(destination.with_suffix(".json").read_text())
    assert report["checked_batches"] == [1, 4, 8]
    assert report["finite"] and report["single_sample_parity"]
    assert onnx.load(destination).graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"


def test_codeformer_keeps_w_runtime_second_input():
    original = graph(codeformer=True)
    dynamic = surgery(original, "CodeFormer.onnx")
    assert [v.name for v in dynamic.graph.input] == ["image", "w"]
    assert input_shapes(dynamic, 8)["w"] == (1,)
    validate_conversion(original, dynamic)
    session = ort.InferenceSession(dynamic.SerializeToString(), providers=["CPUExecutionProvider"])
    image = np.ones((4, 3, 2, 2), np.float32)
    assert np.all(session.run(None, {"image": image, "w": np.array([0], np.float32)})[0] == 0)
    assert np.all(session.run(None, {"image": image, "w": np.array([1], np.float32)})[0] == 1)


def test_codeformer_refuses_already_frozen_export():
    with pytest.raises(DynamicShapeError, match="second runtime input"):
        surgery(graph(), "CodeFormer.onnx")


@pytest.mark.parametrize("name", ["GPEN-BFR-512.onnx", "GPEN-256.onnx", "gpen_512.onnx"])
def test_gpen_classification_and_static_receipt(tmp_path, name):
    source = tmp_path / name
    onnx.save(graph(), source)
    assert static_batch_reason(name)
    with pytest.raises(StaticBatchRequired, match="weight reshaping"):
        convert(source)
    assert not source.with_stem(source.stem + "_dynamic").exists()
    receipt = source.with_name(source.stem + "_dynamic.json")
    assert json.loads(receipt.read_text())["state"] == "STATIC_BATCH_1"


def test_conversion_cannot_publish_failed_batch_or_nonfinite(tmp_path):
    source = tmp_path / "broken.onnx"
    model = graph()
    # Deliberately invalidate a constant target that surgery cannot repair.
    model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(np.array([2, 6], np.int64), "shape"))
    onnx.save(model, source)
    with pytest.raises(Exception):
        convert(source)
    assert not source.with_stem("broken_dynamic").exists()


@pytest.fixture
def fake_trt(tmp_path, monkeypatch):
    import roop_3d.trt_session_builder as module
    import roop_3d.models.trt_builder as parity_builder
    cache = tmp_path / "engines"
    cache.mkdir()
    state = types.SimpleNamespace(write_engine=True, finite=True, cpu=False, activate=True, batches=[], options=None)
    original_version = module.importlib.metadata.version
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "10.9.0.34" if name == "tensorrt-cu12-libs" else original_version(name))
    monkeypatch.setattr(module, "register_native_libraries", lambda: ())
    # Simulated builds must not change process-wide native temp/cache policy for
    # unrelated tests (the real builder intentionally applies it to its process).
    monkeypatch.setattr(parity_builder, "configure_native_paths", lambda _: None)
    monkeypatch.setattr(module, "cache_directory", lambda: cache)
    monkeypatch.setattr(module, "cuda_memory_budget", lambda *_: 3 * 1024**3)
    monkeypatch.setattr(ort, "get_available_providers", lambda: [module.TRT, "CUDAExecutionProvider"])

    class Session:
        def __init__(self, path, sess_options, providers):
            state.options = sess_options
            state.providers = providers
            self.config = providers[0][1]
            self.trace = Path(sess_options.profile_file_prefix + ".json")

        def disable_fallback(self):
            state.fallback_disabled = True

        def get_providers(self):
            return [module.TRT] if state.activate else ["CPUExecutionProvider"]

        def run(self, names, inputs):
            state.batches.append(inputs["image"].shape[0])
            if state.write_engine:
                (cache / (self.config["trt_engine_cache_prefix"] + "_test.engine")).write_bytes(b"fake-test-engine")
            return [inputs["image"] * (1 if state.finite else np.nan)]

        def end_profiling(self):
            self.trace.write_text(json.dumps([{"cat": "Node", "args": {"provider": module.TRT}}] +
                                            ([{"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}] if state.cpu else [])))
            return str(self.trace)

    monkeypatch.setattr(ort, "InferenceSession", Session)
    path = tmp_path / "inswapper_128_dynamic.onnx"
    onnx.save(graph(dynamic=True, fixed_reshape=False), path)
    return state, path


def test_strict_builder_profiles_precision_cache_and_canary(fake_trt):
    state, path = fake_trt
    events = []
    builder = TensorRTSessionBuilder(telemetry=events.append)
    builder.build(path)
    assert state.batches == [1, 4, 8]
    assert state.options.get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    assert state.fallback_disabled
    provider = state.providers[0][1]
    assert provider["trt_fp16_enable"] is False
    assert provider["trt_profile_min_shapes"] == "image:1x3x2x2"
    assert provider["trt_profile_opt_shapes"] == "image:4x3x2x2"
    assert provider["trt_profile_max_shapes"] == "image:8x3x2x2"
    assert provider["trt_engine_cache_enable"] and provider["trt_timing_cache_enable"]
    assert [e["engine_state"] for e in events] == ["COMPILING_TRT", "RUNNING_CANARY", "IDLE"]
    assert all(e["engine_label"] == "Compiling TensorRT Engine..." for e in events[:-1])
    assert events[0]["engine_precision"] == "FP32" and isinstance(events[0]["engine_since"], float)
    events.clear()
    builder.build(path)
    assert [e["engine_state"] for e in events] == ["RUNNING_CANARY", "IDLE"]
    assert builder.last_build["cache_hit"]


@pytest.mark.parametrize("failure,match", [("write_engine", "No model/profile"), ("finite", "non-finite"),
                                           ("cpu", "node placement"), ("activate", "activation failed")])
def test_strict_tripwires_and_idle_on_failure(fake_trt, failure, match):
    state, path = fake_trt
    setattr(state, failure, True if failure == "cpu" else False)
    events = []
    with pytest.raises(TensorRTBuildError, match=match):
        TensorRTSessionBuilder(telemetry=events.append).build(path)
    assert events[-1]["engine_state"] == "IDLE"


def test_unrelated_engine_does_not_satisfy_cache_gate(fake_trt):
    state, path = fake_trt
    state.write_engine = False
    (path.parent / "engines/unrelated.engine").write_bytes(b"unrelated")
    with pytest.raises(TensorRTBuildError, match="No model/profile"):
        TensorRTSessionBuilder().build(path)


def test_static_batch_rejected_before_session_creation(fake_trt):
    state, path = fake_trt
    onnx.save(graph(), path)
    with pytest.raises(StaticBatchError, match="_dynamic.onnx"):
        TensorRTSessionBuilder().build(path)
    assert state.options is None


def test_gpen_default_profile_is_batch_one(fake_trt):
    state, path = fake_trt
    gpen = path.with_name("GPEN-BFR-512.onnx")
    onnx.save(graph(), gpen)
    TensorRTSessionBuilder().build(gpen)
    assert state.batches == [1]
    assert state.providers[0][1]["trt_fp16_enable"] is True
    with pytest.raises(StaticBatchError):
        TensorRTSessionBuilder().build(gpen, profile=BatchProfile())


def test_strict_environment_cannot_select_cpu(monkeypatch):
    monkeypatch.setenv("ROOP_OPT_STRICT_TRT", "1")
    with pytest.raises(TensorRTBuildError, match="ROOP_OPT_STRICT_TRT"):
        TensorRTSessionBuilder().build("unused.onnx", selected_provider="CPUExecutionProvider")


def test_precision_and_profile_cache_are_separate(fake_trt, monkeypatch):
    state, path = fake_trt
    builder = TensorRTSessionBuilder()
    builder.build(path)
    prefix = builder.last_build["engine_prefix"]
    monkeypatch.setenv("ROOP_SWAP_FP16", "1")
    builder.build(path)
    assert builder.last_build["engine_prefix"] != prefix
    assert not builder.last_build["cache_hit"]
    fp16_prefix = builder.last_build["engine_prefix"]
    builder.build(path, profile=BatchProfile((1,), 1))
    assert builder.last_build["engine_prefix"] != fp16_prefix


def test_compilation_broadcasts_to_both_live_websockets(fake_trt, tmp_path):
    from roop_3d.server.app import create_app
    state, path = fake_trt
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app, base_url="http://127.0.0.1:8030")
    with client.websocket_connect("ws://127.0.0.1:8030/ws/telemetry") as telemetry, \
            client.websocket_connect("ws://127.0.0.1:8030/ws/v1/pipeline/status") as status:
        assert telemetry.receive_json()["type"] == "stream.ready"
        assert status.receive_json()["type"] == "stream.ready"
        assert status.receive_json()["engine_state"] == "IDLE"
        app.state.trt_builder.build(path)
        for expected in ("COMPILING_TRT", "RUNNING_CANARY", "IDLE"):
            for stream in (telemetry, status):
                frame = stream.receive_json()
                assert frame["type"] == "pipeline.status" and frame["engine_state"] == expected
                if expected != "IDLE":
                    assert frame["engine_model"] == path.name and frame["engine_precision"] == "FP32"
                    assert isinstance(frame["engine_since"], float)


def test_output_boundary_and_hardlink_rejection(tmp_path):
    import os
    with pytest.raises(ValueError, match="checkout"):
        owned_path(Path("G:/not-this-checkout/engine"))
    source = tmp_path / "source"
    source.write_bytes(b"reference")
    linked = tmp_path / "link"
    os.link(source, linked)
    with pytest.raises(ValueError, match="Hardlinked"):
        owned_path(linked)


def test_memory_budgets_respect_profiles_and_live_free_memory(monkeypatch):
    from roop_3d.core import hardware
    from roop_3d.trt_session_builder import cuda_memory_budget
    gpu = types.SimpleNamespace(free_vram_mib=24000, total_vram_mib=24000)
    monkeypatch.setattr(hardware, "probe_hardware", lambda _: types.SimpleNamespace(gpus=[gpu]))
    assert cuda_memory_budget("6gb", 1024 * 1048576) == (6144 - 768 - 1024) * 1048576
    assert cuda_memory_budget("12gb", 2048 * 1048576) == (12288 - 768 - 2048) * 1048576
    gpu.free_vram_mib = 1000
    with pytest.raises(TensorRTBuildError, match="Insufficient free VRAM"):
        cuda_memory_budget("6gb", 1024 * 1048576)


def test_detector_squeeze_export_converts_without_losing_samples():
    h, t = onnx.helper, onnx.TensorProto
    original = h.make_model(h.make_graph([
        h.make_node("Squeeze", ["image", "axes"], ["output"])
    ], "detector_export", [h.make_tensor_value_info("image", t.FLOAT, [1, 12, 4])],
        [h.make_tensor_value_info("output", t.FLOAT, [12, 4])],
        [onnx.numpy_helper.from_array(np.array([0], np.int64), "axes")]),
        opset_imports=[h.make_opsetid("", 17)])
    candidate = surgery(original, "scrfd_2.5g_kps.onnx")
    assert validate_conversion(original, candidate)["checked_batches"] == [1, 4, 8]
