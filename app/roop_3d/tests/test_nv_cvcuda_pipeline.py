"""Stage 14 contracts and real device checks, with a fail-closed acceptance mode.

ROOP_STAGE14_ACCEPTANCE=1 makes missing native hardware prerequisites failures.
Synthetic native inference is transport evidence, not production model parity.
"""
from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np
import onnx
import pytest

from roop_3d.nv_video_codec import (GPUBackendUnavailable, NvVideoDecoder, NvVideoEncoder,
    cuda_tensor, motion_summary)
from roop_3d.optical_flow_tracker import OpticalFlowTracker, RAFTSmallFlow, select_flow
from roop_3d.trt_fp8_builder import HEAVY_RESTORERS, validate_fp8_graph
from roop_3d.trt_session_builder import precision_for


@pytest.fixture(autouse=True)
def restore_runtime_environment():
    """Native cache/temp policy is process-wide; GPU tests must not leak it."""
    keys = ("TEMP", "TMP", "TMPDIR", "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "NVIDIA_TF32_OVERRIDE")
    previous = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def cuda():
    torch = native("torch")
    if not torch.cuda.is_available():
        if os.getenv("ROOP_STAGE14_ACCEPTANCE") == "1":
            pytest.fail("Stage 14 acceptance needs CUDA")
        pytest.skip("CUDA unavailable")
    return torch


def native(name):
    import importlib
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        if os.getenv("ROOP_STAGE14_ACCEPTANCE") == "1":
            pytest.fail(f"Stage 14 acceptance prerequisite missing: {name}: {error}")
        pytest.skip(f"Optional native backend unavailable: {name}")


def test_cpu_surface_is_never_accepted_as_zero_copy():
    class Host:
        def __dlpack__(self):
            raise AssertionError("must reject before importing a host capsule")
        def __dlpack_device__(self):
            return (1, 0)
    with pytest.raises(ValueError, match="Host surfaces"):
        cuda_tensor(Host())


def test_decode_statistics_quarter_pixel_units_and_unavailable():
    assert motion_summary(None)["high_motion"] is None
    assert motion_summary({"qp_luma": [20, 30], "mv0_x": [12], "mv0_y": [16]}) == {
        "available": True, "mean_motion_pixels": 5., "high_motion": True, "mean_qp": 25.}
    assert not motion_summary({"mv0_x": [0], "mv0_y": [0]})["high_motion"]


@pytest.mark.parametrize("name", ["inswapper_128.onnx", "INSWAPPER_128_dynamic.ONNX", "inswapper_128.batch.onnx",
    "inswapper_128_dynamic.batch.onnx", "inswapper_128.batch_dynamic.onnx"])
def test_swapper_has_no_low_precision_override(name):
    assert precision_for(name, {"ROOP_SWAP_FP16": "1", "ROOP_TRT_FP8": "1"}) == "FP32"


@pytest.mark.parametrize("name", sorted(HEAVY_RESTORERS))
def test_fp8_is_opt_in_and_allowlisted(name):
    stable_precision = "FP32" if "gpen" in name else "FP16"
    assert precision_for(name + ".onnx", {}) == stable_precision
    assert precision_for(name + ".onnx", {"ROOP_TRT_FP8": "1"}) == ("FP32" if stable_precision == "FP32" else "FP8")
    assert precision_for(name + ".onnx", {"ROOP_TRT_FP8": "1", "ROOP_TRT_FP16": "0"}) == "FP32"
    assert precision_for("CodeFormer.onnx", {"ROOP_TRT_FP8": "1"}) == "FP16"


def qdq_model():
    h, t = onnx.helper, onnx.TensorProto
    return h.make_model(h.make_graph([
        h.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
        h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"])], "synthetic_fp8_contract",
        [h.make_tensor_value_info("x", t.FLOAT, [1, 3, 4, 4])],
        [h.make_tensor_value_info("y", t.FLOAT, [1, 3, 4, 4])], [
            onnx.numpy_helper.from_array(np.array(.1, np.float32), "scale"),
            h.make_tensor("zero", t.FLOAT8E4M3FN, [], bytes([0]), raw=True)]),
        opset_imports=[h.make_opsetid("", 19)])


def test_fp8_requires_explicit_e4m3fn_and_observed_hardware():
    model = qdq_model()
    assert validate_fp8_graph(model, "GPEN-BFR-1024.onnx", (8, 9), "10.9.0")["qdq_pairs"] == 1
    for name, capability, version in [("inswapper_128.onnx", (8, 9), "10.9"),
            ("GPEN-BFR-1024.onnx", (8, 6), "10.9"), ("GPEN-BFR-1024.onnx", (8, 9), "9.2")]:
        with pytest.raises(ValueError):
            validate_fp8_graph(model, name, capability, version)
    model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(np.array(0., np.float32), "scale"))
    with pytest.raises(ValueError, match="positive"):
        validate_fp8_graph(model, "GPEN-BFR-1024.onnx", (8, 9), "10.9")


def test_swapper_rejects_quantized_graph():
    from roop_3d.trt_fp8_builder import validate_swapper_fp32
    with pytest.raises(ValueError, match="FP32"):
        validate_swapper_fp32(qdq_model())


def test_dlpack_aliases_device_storage_and_torch_affine_stays_cuda():
    torch = cuda()
    from roop_3d.cvcuda_ops import FaceOps
    frame = torch.arange(32 * 32 * 3, device="cuda", dtype=torch.int32).to(torch.uint8).reshape(32, 32, 3)
    alias = cuda_tensor(frame)
    assert alias.data_ptr() == frame.data_ptr()
    matrix = torch.tensor([[[1., 0., 0.], [0., 1., 0.]], [[1., 0., 4.], [0., 1., 5.]]], device="cuda")
    output = FaceOps("torch").align(alias, matrix, (8, 8))
    assert output.is_cuda and output.shape == (2, 3, 8, 8)
    torch.testing.assert_close(output[0], frame[:8, :8].permute(2, 0, 1).float() / 255)
    torch.testing.assert_close(output[1], frame[5:13, 4:12].permute(2, 0, 1).float() / 255)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optical_flow_smooths_30_moving_frames_and_preserves_occluders(device):
    torch = cuda() if device == "cuda" else native("torch")
    tracker = OpticalFlowTracker(.75)
    raw_error, filtered_error = [], []
    forward = torch.zeros((1, 64, 96, 2), device=device)
    forward[..., 0] = 1
    vertices = torch.tensor([[[30., 30., 7.]]], device=device)
    yy, xx = torch.meshgrid(torch.arange(64, device=device), torch.arange(96, device=device), indexing="ij")
    for index in range(30):
        truth = (((xx - (24 + index))**2 + (yy - 32)**2) < 14**2).float()[None, None]
        noisy = truth * (1. if index % 2 else .6)
        visible = torch.ones_like(truth)
        visible[:, :, :10] = 0  # Current hand/hair visibility cannot be blurred away.
        result, mesh = tracker.propagate(noisy, -forward, vertices=vertices,
            forward_flow=forward, visibility=visible)
        torch.testing.assert_close(mesh[..., 0], vertices[..., 0] + 1)
        torch.testing.assert_close(mesh[..., 2], vertices[..., 2])
        assert not result[:, :, :10].count_nonzero()
        if index > 3:
            raw_error.append((noisy - truth).square().mean())
            filtered_error.append((result - truth).square().mean())
    assert torch.stack(filtered_error).mean() < torch.stack(raw_error).mean() * .7
    reset, _ = tracker.propagate(torch.zeros_like(truth), scene_cut=True)
    assert not reset.count_nonzero()


def test_disocclusion_confidence_and_missing_flow():
    torch = native("torch")
    alpha = torch.ones((1, 1, 8, 8))
    tracker = OpticalFlowTracker()
    tracker.propagate(alpha)
    with pytest.raises(ValueError, match="requires"):
        tracker.propagate(alpha)
    result, _ = tracker.propagate(alpha * 0, torch.zeros((1, 8, 8, 2)), confidence=alpha * 0)
    assert not result.count_nonzero()


def test_flow_fallback_requires_explicit_local_runner(monkeypatch):
    import roop_3d.optical_flow_tracker as flow
    def unavailable(*args, **kwargs):
        raise GPUBackendUnavailable("synthetic missing hardware")
    monkeypatch.setattr(flow, "NvidiaHardwareFlow", unavailable)
    with pytest.raises(GPUBackendUnavailable, match="local RAFT runner"):
        select_flow((32, 32))
    torch = cuda()
    # A synthetic runner verifies the fallback's device/shape contract only.
    backend = select_flow((32, 32), raft_runner=lambda a, b: a.new_zeros((1, 2, 32, 32)))
    assert isinstance(backend, RAFTSmallFlow)
    image = torch.zeros((1, 3, 32, 32), device="cuda")
    forward, backward = backend.pair(image, image)
    assert forward.shape == backward.shape == (1, 32, 32, 2)
    with pytest.raises(ValueError, match="full-resolution"):
        RAFTSmallFlow(lambda a, b: a[:, :2, ::2, ::2]).pair(image, image)


def test_cuda_io_binding_with_real_ort_session(tmp_path):
    """Actual CUDA EP transport; explicitly not evidence of a TensorRT build."""
    torch = cuda()
    import onnxruntime as ort
    from roop_3d.models.native_libraries import register_native_libraries
    from roop_3d.optimized_trt_engine import OptimizedTRTEngine
    register_native_libraries()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA EP unavailable")
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(h.make_graph([h.make_node("Relu", ["x"], ["y"])], "cuda_binding_fixture",
        [h.make_tensor_value_info("x", t.FLOAT, [1, 3, 8, 8])],
        [h.make_tensor_value_info("y", t.FLOAT, [1, 3, 8, 8])]), opset_imports=[h.make_opsetid("", 17)])
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    engine = OptimizedTRTEngine(tmp_path / "unused.onnx")
    engine._session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
        providers=["CUDAExecutionProvider"])
    engine._build = {"checked_batches": [1], "profiles": {"min": {"x": (1, 3, 8, 8)},
        "max": {"x": (1, 3, 8, 8)}}}
    source = torch.randn((1, 3, 8, 8), device="cuda")
    output = torch.empty_like(source)
    try:
        pointer = output.data_ptr()
        assert engine.run_cuda({"x": source}, {"y": output})["y"].data_ptr() == pointer
        torch.testing.assert_close(output, source.relu())
        with pytest.raises(ValueError, match="contiguous CUDA"):
            engine.run_cuda({"x": torch.zeros((1, 3, 8, 8))}, {"y": output})
    finally:
        engine.close()


def test_torch_mask_filters_preserve_edges_and_visibility():
    torch = cuda()
    from roop_3d.cvcuda_ops import FaceOps
    ops = FaceOps("torch")
    image = torch.zeros((1, 3, 16, 16), device="cuda")
    image[..., 8:] = 1
    result = ops.smooth_skin(image, image)
    torch.testing.assert_close(result, image, atol=1e-5, rtol=0)
    assert ops.laplacian(image).abs().sum() > 0
    visible = torch.ones_like(image[:, :1]); visible[..., :8] = 0
    feathered = ops.feather(image[:, :1], visible)
    assert not feathered[..., :8].count_nonzero()
    assert ops.morphology(image[:, :1], -1).sum() >= image[:, :1].sum()


def test_real_nvenc_decode_dlpack_roundtrip(tmp_path):
    torch = cuda()
    native("PyNvVideoCodec")
    path = tmp_path / "native.mp4"
    encoder = NvVideoEncoder(path, (256, 256, 4), Fraction(30))
    rgba = torch.zeros((256, 256, 4), dtype=torch.uint8, device="cuda")
    rgba[..., 0] = 220; rgba[..., 3] = 255
    try:
        for _ in range(3):
            encoder.write(rgba)
        encoder.close()
    except BaseException:
        encoder.abort()
        raise
    frames = 0
    for frame in NvVideoDecoder(path).frames():
        assert frame.tensor.is_cuda and tuple(frame.tensor.shape) == (256, 256, 3)
        assert frame.tensor[..., 0].float().mean() > 180
        assert cuda_tensor(frame.tensor).data_ptr() == frame.tensor.data_ptr()
        frames += 1
    assert frames == 3


def test_native_decoder_cvcuda_trt_zero_host_pixels(tmp_path):
    """Real native stack + CUDA profiler; never replace missing libraries with fakes."""
    torch = cuda()
    native("cvcuda"); native("PyNvVideoCodec"); native("tensorrt")
    import onnxruntime as ort
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        pytest.fail("Native acceptance requires TensorRT EP")
    from roop_3d.cvcuda_ops import FaceOps
    from roop_3d.optimized_trt_engine import OptimizedTRTEngine
    from roop_3d.trt_session_builder import BatchProfile
    path = tmp_path / "input.mp4"
    writer = NvVideoEncoder(path, (256, 256, 4), Fraction(30))
    writer.write(torch.full((256, 256, 4), 128, dtype=torch.uint8, device="cuda"))
    writer.close()
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(h.make_graph([h.make_node("Relu", ["x"], ["y"])], "native_transport_fixture",
        [h.make_tensor_value_info("x", t.FLOAT, [1, 3, 32, 32])],
        [h.make_tensor_value_info("y", t.FLOAT, [1, 3, 32, 32])]), opset_imports=[h.make_opsetid("", 17)])
    model_path = tmp_path / "transport.onnx"; onnx.save(model, model_path)
    engine = OptimizedTRTEngine(model_path, profile=BatchProfile((1,), 1))
    try:
        engine.prepare()  # Host canaries and compilation excluded from frame transport.
        ops = FaceOps("cvcuda")
        matrix = torch.tensor([[[1., 0., 4.], [0., 1., 5.]]], device="cuda")
        output = torch.empty((1, 3, 32, 32), device="cuda")
        for frame in NvVideoDecoder(path).frames():
            wrapped = ops.cv.as_tensor(frame.tensor, "HWC")
            borrowed = torch.as_tensor(wrapped.cuda(), device=frame.tensor.device)
            assert borrowed.data_ptr() == frame.tensor.data_ptr(), "CV-CUDA import must alias decoded storage"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                crops = ops.align(frame.tensor, matrix, (32, 32))
                engine.run_cuda({"x": crops.contiguous()}, {"y": output})
                torch.cuda.synchronize()
            trace = tmp_path / "cuda-trace.json"; profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            assert any(e.get("cat") == "kernel" for e in events), "CUPTI kernel tracing required"
            transfers = [e for e in events if any(s in e.get("name", "").lower()
                for s in ("dtoh", "device to host", "device -> host"))]
            assert transfers == [], f"Device-to-host transfers during swap: {transfers}"
            torch.testing.assert_close(output, crops)
    finally:
        engine.close()
