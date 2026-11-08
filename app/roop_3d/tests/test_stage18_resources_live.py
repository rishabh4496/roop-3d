"""Stage 18: resources that need the real thing - ONNX Runtime on the real GPU, real allocation failures, server WebSockets,
GPU admission leases after failures, a corrupt model through the whole job path, and a high-resolution render."""
from __future__ import annotations

import gc
import time

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from roop_3d.engine.vram_governor import is_cuda_oom
from roop_3d.jobs.schema import JobState
from roop_3d.reliability import probe, soak
from roop_3d.reliability.classify import classify
from roop_3d.reliability.growth import DEFAULT_PROFILE, analyse
from roop_3d.reliability.harness import build_rig

CUDA = "CUDAExecutionProvider" in ort.get_available_providers() and probe.gpu_memory(max_age=0).available


def _matmul(size: int = 256) -> bytes:
    weight = helper.make_tensor("w", TensorProto.FLOAT, [size, size], [0.001] * (size * size))
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, size])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, size])], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _expand_model() -> bytes:
    graph = helper.make_graph([helper.make_node("Expand", ["x", "shape"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                               helper.make_tensor_value_info("shape", TensorProto.INT64, [3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


# --------------------------------------------------------------------------------- real allocation failures
@pytest.mark.parametrize("provider", ["CPUExecutionProvider",
                                      pytest.param("CUDAExecutionProvider", marks=pytest.mark.skipif(not CUDA, reason="no CUDA GPU"))])
def test_real_onnxruntime_allocation_failures_are_recognised_as_out_of_memory(provider):
    """16 PiB (2^20 x 2^20 x 2^12 floats) can never be allocated, whatever the page file does (a 64 GiB request is
    satisfied when Windows grows the page file under memory pressure): whatever ORT raises must take the OOM path."""
    session = ort.InferenceSession(_expand_model(), providers=[provider])
    with pytest.raises(Exception) as caught:
        session.run(None, {"x": np.ones(1, np.float32), "shape": np.array([1 << 20, 1 << 20, 1 << 12], np.int64)})
    assert is_cuda_oom(caught.value), f"unrecognised {provider} allocation failure: {str(caught.value)[:200]}"
    # ORT's arena words CPU and CUDA failures identically, so the neutral code is the honest one; it must stay resumable.
    assert classify(caught.value).code in {"MEMORY_EXHAUSTED", "GPU_MEMORY_EXHAUSTED", "HOST_MEMORY_EXHAUSTED"}
    assert classify(caught.value).recoverable
    assert classify(RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation")).code == "HOST_MEMORY_EXHAUSTED"
    del session


# ------------------------------------------------------------------------------------------ GPU sessions
@pytest.mark.skipif(not CUDA, reason="requires an NVIDIA GPU with the CUDA execution provider")
def test_creating_and_destroying_cuda_sessions_does_not_grow_gpu_memory_or_leak_sessions():
    """Real VRAM, via nvidia-smi (device total: Windows WDDM does not report per-process memory)."""
    data = _matmul(512)
    warm = ort.InferenceSession(data, providers=["CUDAExecutionProvider"])
    warm.run(None, {"x": np.ones((1, 512), np.float32)})
    del warm
    gc.collect()
    series, baseline = [], probe.gpu_memory(max_age=0).used_mib
    for cycle in range(1, 41):
        session = ort.InferenceSession(data, providers=["CUDAExecutionProvider"])
        for _ in range(5):
            session.run(None, {"x": np.ones((1, 512), np.float32)})
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        del session
        gc.collect()
        series.append((float(cycle), probe.gpu_memory(max_age=0).used_mib))
    assert probe.count_ort_sessions() == 0
    report = analyse(DEFAULT_PROFILE, {"gpu_used_mib": series})
    assert report.ok, report.as_dict()["metrics"]["gpu_used_mib"]
    assert series[-1][1] - baseline < 160, f"{baseline} -> {series[-1][1]} MiB after 40 create/destroy cycles"


# ------------------------------------------------------------------------------------- server websockets
def test_server_websocket_subscribers_are_released_when_clients_disconnect(tmp_path):
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    with TestClient(rig.app, base_url="http://127.0.0.1:8030") as client:
        events = rig.app.state.events
        gc.collect()
        before = probe.process_memory().handles
        for _ in range(150):
            with client.websocket_connect("ws://127.0.0.1:8030/api/v1/ws/events") as socket:
                assert socket.receive_json()["type"] == "stream.ready"
                assert events.subscriber_count == 1
        deadline = time.monotonic() + 5
        while events.subscriber_count and time.monotonic() < deadline:
            time.sleep(0.02)
        assert events.subscriber_count == 0
        gc.collect()
        assert probe.process_memory().handles - before <= 8


# --------------------------------------------------------------------------------- GPU admission leases
def test_gpu_admission_is_released_after_success_failure_cancel_and_interrupt(tmp_path):
    def boom(index):
        if index == 4:
            raise KeyError("bug")
    rig = build_rig(tmp_path, frames=60, tracks=1, delay=0.005, on_frame=lambda i: boom(i) if rig_state["fail"] else None)
    rig_state = {"fail": False}
    try:
        ok = rig.wait(rig.submit("a").job_id)
        rig_state["fail"] = True
        bad = rig.wait(rig.submit("b").job_id)
        rig_state["fail"] = False
        cancelled = rig.submit("c")
        rig.wait_frames(cancelled.job_id, 5)
        rig.store.control(cancelled.job_id, "cancel")
        rig.wait(cancelled.job_id)
        assert (ok.state, bad.state) == (JobState.succeeded, JobState.failed)
        deadline = time.monotonic() + 10
        while (rig.worker.scheduler.active_count or rig.worker._futures) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rig.worker.scheduler.active_count == 0
        assert rig.worker.scheduler.governor.active_count == 0, "a leaked lease would permanently shrink the admission budget"
        running = rig.submit("d")
        rig.wait_frames(running.job_id, 5)
    finally:
        rig.worker.stop()                                   # clean stop mid-job: interrupted, lease released
    assert rig.job(running.job_id).state == JobState.interrupted
    assert rig.worker.scheduler.active_count == 0 and rig.worker.scheduler.governor.active_count == 0


# ------------------------------------------------------------------------------------------ corrupt model
def test_corrupt_model_fails_the_job_with_a_named_code_and_publishes_nothing(tmp_path):
    def factory(catalog, config):
        ort.InferenceSession(b"definitely not an onnx protobuf", providers=["CPUExecutionProvider"])

    rig = build_rig(tmp_path, frames=20, tracks=1, session_factory=factory)
    try:
        failed = rig.wait(rig.submit("m").job_id)
        assert failed.state == JobState.failed and failed.failure.code == "MODEL_CORRUPT"
        assert "re-verify" in failed.failure.message.lower() or "re-import" in failed.failure.message.lower()
        assert rig.output_files() == []
        assert rig.worker.scheduler.governor.active_count == 0
    finally:
        rig.worker.stop()


# --------------------------------------------------------------------------------------- high resolution
def test_high_resolution_render_is_bounded_and_complete():
    """1080p, noisy content (worst case for PNG checkpoint size), 100 frames through checkpoint -> mux."""
    result = soak.scenario_hires(width=1920, height=1080, frames=100, tracks=2, interval=0.5)
    assert result["passed"], {"growth": result["growth"]["violations"], "invariants": result["invariants"],
                              "state": result["state"], "failure": result["failure"]}
    assert result["frames"] == 100 and result["workdir_released"]
    rss_samples = result["series"].get("rss_mib", [])
    assert rss_samples, "the soak must sample process RSS before it starts rendering"
    baseline_rss = rss_samples[0][1]
    peak_growth = result["peak_rss_mib"] - baseline_rss
    assert peak_growth < 1024, f"frame buffers added {peak_growth:.1f} MiB above the process baseline"
