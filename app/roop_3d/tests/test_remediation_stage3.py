"""Real host/codec/ORT tests; production GPU fidelity is a separate campaign."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import threading
import time

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

from roop_3d.compositor.stream_restoration import (MaskSeamWatchdog, DecoupledRestorer, RestorerSpec,
    CODEFORMER_REFERENCE, GPEN_REFERENCE, codeformer_spec, gpen_spec, load_ultramax_spec)
from roop_3d.hardware_streamer import SharedMemoryFrameRing, FrameMetadata, NvdecFrameSource, NVHardwareVideoWriter
from roop_3d.models.emap_cache import cached_emap
from roop_3d.optimized_prepass import SparsePrepass, LockedFace
from roop_3d.optimized_processor import StreamingProcessor
from roop_3d.render_policy import RenderPolicy, CudaWorkerContext
from roop_3d.tools.verify_stream_integration import run_verification, create_clip, verify_geometry
from roop_3d.vectorized_pipeline import CpuBatchPipeline, OrtValueRunner, affine_graph, normalized_matrix
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.media.video_reader import DecodedFrame
from roop_3d.media.timestamps import Timestamp


def test_complete_stream_integration_gate():
    assert run_verification()["state"] == "PASS"


def test_ring_rejects_shapes_and_enforces_backpressure():
    with SharedMemoryFrameRing((4, 4, 3), 2) as ring:
        metadata = FrameMetadata(1, 1, 1, 25)
        with pytest.raises(ValueError):
            ring.put(np.zeros((4, 4, 3), np.float32), metadata)
        ring.put(np.zeros((4, 4, 3), np.uint8), metadata)
        ring.put(np.ones((4, 4, 3), np.uint8), metadata)
        with pytest.raises(TimeoutError):
            ring.put(np.ones((4, 4, 3), np.uint8), metadata, timeout=0.01)
        with ring.lease() as leased:
            assert leased[1].time_base_den == 25
        del leased
        ring.cancel()
        with pytest.raises(RuntimeError):
            ring.put(np.ones((4, 4, 3), np.uint8), metadata)


def test_cached_projection_binds_to_model_hash_and_refuses_objects(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"fixture1")
    destination = tmp_path / "projection.npy"
    calls = []
    def extract(_):
        calls.append(1)
        return np.eye(512, dtype=np.float32)
    cached_emap(source, extract, destination=destination)
    cached_emap(source, extract, destination=destination)
    assert len(calls) == 1
    source.write_bytes(b"fixture2")
    cached_emap(source, extract, destination=destination)
    assert len(calls) == 2


def test_prepass_checkpoint_identity_and_tamper_checks(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"analytic input identity")
    points = (ARCFACE_REFERENCE_5PTS * 0.3).astype(np.float32)
    def reader():
        for index in range(5):
            yield DecodedFrame(index, Timestamp(index, Fraction(1, 25)), 1, index == 0,
                               np.full((32, 32, 3), index, np.uint8))
    events = []
    prepass = SparsePrepass(tmp_path, "fixture", source, detect=lambda _: [1],
        track=lambda _: [(0, np.array([-4, -3, 30, 31]), points)],
        reader_factory=reader, policy=RenderPolicy(sparse_interval=2, range_padding=2), emit=events.append)
    first = prepass.run()
    assert first.total_frames == 5 and first.verified_target_frames == 5
    assert prepass.run().restored
    assert any(e["state"] == "RESTORED" for e in events)
    source.write_bytes(b"changed identity")
    with pytest.raises(ValueError, match="identity differs"):
        prepass.run()


def test_sparse_absence_is_explicitly_an_estimate(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic")
    events = []
    reader = lambda: iter([DecodedFrame(index, Timestamp(index, Fraction(1, 25)), 1, index == 0,
                                        np.zeros((16, 16, 3), np.uint8)) for index in range(5)])
    result = SparsePrepass(tmp_path, "estimate", source, detect=lambda _: [], track=lambda _: [],
        reader_factory=reader, emit=events.append).run()
    assert not result.faces
    stage1 = next(e for e in events if e["state"] == "DONE")
    assert stage1["absence_evidence"] == "SPARSE_ESTIMATE"


def test_ortvalue_buffers_are_reused_across_calls():
    session = ort.InferenceSession(affine_graph(32, 32, 1).SerializeToString(), providers=["CPUExecutionProvider"])
    runner = OrtValueRunner(session, {"output": (1, 3, 128, 128)}, device="cpu")
    matrix = normalized_matrix(np.array([[0.25, 0, 0], [0, 0.25, 0]], np.float32), 32, 32)[None]
    feeds = {"image": np.ones((1, 3, 32, 32), np.float32), "matrix": matrix}
    first = runner.run(feeds)["output"]
    pointers = {n: v.data_ptr() for n, v in runner.inputs.items()}
    second = runner.run(feeds)["output"]
    assert first.data_ptr() == second.data_ptr()
    assert pointers == {n: v.data_ptr() for n, v in runner.inputs.items()}


def test_out_of_frame_graph_geometry():
    verify_geometry()


def test_worker_context_serializes_shared_buffer_owners():
    context = CudaWorkerContext(enabled=False)
    active, peak = 0, 0
    def invoke():
        nonlocal active, peak
        with context.activate():
            active += 1
            peak = max(peak, active)
            time.sleep(0.005)
            active -= 1
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: invoke(), range(8)))
    assert peak == 1


def test_native_ultramax_bypass_before_model_construction():
    calls = []
    class Restorer:
        def restore(self, image):
            calls.append(1)
            return image
    spec = RestorerSpec("ultramax", "explicit-fixture", CODEFORMER_REFERENCE, 128)
    pool = DecoupledRestorer(Restorer, spec, workers=2)
    image = np.full((64, 64, 3), 80, np.uint8)
    points = (CODEFORMER_REFERENCE / 8).astype(np.float32)
    try:
        for size in (20, 319, 320):
            np.testing.assert_array_equal(pool.submit(image, points, np.array([0, 0, size, size]), 1).result(), image)
        assert not calls and not pool.instances
        assert pool.submit(image, points, np.array([0, 0, 321, 321]), 1).result().shape == image.shape
        assert len(calls) == 1
    finally:
        pool.close()


def test_mask_baselines_do_not_cross_faces_threads_or_mask_types():
    watchdog = MaskSeamWatchdog()
    ones, zeros = np.ones((16, 16), np.float32), np.zeros((16, 16), np.float32)
    assert watchdog.observe(1, ones, kind="geometric") == 0
    assert watchdog.observe(1, zeros, kind="occluder") == 0
    assert watchdog.observe(1, ones, kind="geometric") == 0
    assert watchdog.observe(2, zeros, kind="geometric") == 0
    output = []
    thread = threading.Thread(target=lambda: output.append(watchdog.observe(1, zeros, kind="geometric")))
    thread.start()
    thread.join()
    assert output == [0]
    assert watchdog.observe(1, zeros, kind="geometric") == 1


def test_restorer_training_templates_remain_architecture_specific(tmp_path):
    assert not np.allclose(GPEN_REFERENCE, CODEFORMER_REFERENCE)
    assert gpen_spec().name == "gpen_realistic" and codeformer_spec().name == "codeformer"
    ultra = load_ultramax_spec(tmp_path)
    assert ultra.name == "ultramax" and ultra.model_name == "codeformer.fp16.onnx"
    np.testing.assert_array_equal(ultra.template, CODEFORMER_REFERENCE)


def test_parallel_renderer_preserves_frame_order_and_segment_identity(tmp_path):
    source = tmp_path / "input.mp4"
    create_clip(source)
    seen = set()
    lock = threading.Lock()
    def inference(value, _):
        with lock:
            seen.add(threading.get_ident())
        time.sleep(0.01)
        return value
    points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4).astype(np.float32)
    pipeline = CpuBatchPipeline(inference)
    events = []
    processor = StreamingProcessor(tmp_path, "parallel", source, detect=lambda _: [1],
        track=lambda _: [(0, np.array([0, 0, 64, 64]), points)], pipeline=pipeline,
        policy=RenderPolicy(hardware_profile="cpu", execution_threads=3, ring_slots=3, segment_frames=4),
        emit=events.append)
    result = processor.run(tmp_path / "output.mp4", audio=False)
    assert result["frames"] == 10 and len(seen) > 1
    assert len(result["segments"]) == len(set(result["segments"])) == 3


def test_cancelled_renderer_stops_producer_and_does_not_publish(tmp_path):
    source = tmp_path / "input.mp4"
    create_clip(source)
    class Stop(Exception):
        pass
    def check():
        if (tmp_path / "cancel").exists():
            raise Stop()
    def inference(value, _):
        (tmp_path / "cancel").touch()
        return value
    points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4).astype(np.float32)
    processor = StreamingProcessor(tmp_path, "cancel", source, detect=lambda _: [1],
        track=lambda _: [(0, np.array([0, 0, 64, 64]), points)], pipeline=CpuBatchPipeline(inference),
        policy=RenderPolicy(hardware_profile="cpu", segment_frames=3), check=check)
    with pytest.raises(Stop):
        processor.run(tmp_path / "cancelled.mp4", audio=False)
    assert not (tmp_path / "cancelled.mp4").exists()
    assert not any(t.name == "roop3d-stream-decode" and t.is_alive() for t in threading.enumerate())


def test_thread_benchmark_api_and_real_cpu_workload(tmp_path, monkeypatch):
    from roop_3d.server.app import create_app
    from roop_3d.core import hardware
    from types import SimpleNamespace
    monkeypatch.setattr(hardware, "probe_hardware", lambda _: SimpleNamespace(gpus=[]))
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app, base_url="http://127.0.0.1:8030")
    assert client.get("/api/settings/benchmark_status").json()["state"] == "IDLE"
    assert client.post("/api/settings/benchmark_threads", json={"profile": "unsupported"}).status_code == 422
    response = client.post("/api/settings/benchmark_threads", json={"profile": "quick"})
    assert response.status_code == 202
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = client.get("/api/settings/benchmark_status").json()
        if value["state"] != "RUNNING":
            break
        time.sleep(0.01)
    assert value["state"] == "DONE", value
    assert value["provider"] == "CPUExecutionProvider"
    assert len(value["results"]) == 6 and value["recommendation"]["samples_per_second"] > 0
    app.state.thread_benchmark.close()


def test_stage3_pipeline_status_fields_are_measured():
    from roop_3d.server.pipeline_status import PipelineTracker
    tracker = PipelineTracker(lambda _: None)
    tracker.render_event("job", {"stage": "STAGE_1_SPARSE_PRESCAN", "state": "DONE", "active_ranges_count": 2})
    tracker.render_event("job", {"stage": "STAGE_2_ACTIVE_PREPASS", "state": "RESTORED",
                                  "verified_target_frames": 10, "skipped_frames_pct": 50})
    value = tracker.snapshot()
    assert value.stages["STAGE_2_ACTIVE_PREPASS"] == "RESTORED"
    assert value.active_ranges_count == 2 and value.verified_target_frames == 10


def test_batch_swap_model_retains_fp32_overflow_guard():
    from roop_3d.trt_session_builder import precision_for
    assert precision_for("inswapper_128.batch.onnx", {}) == "FP32"


def test_codeformer_fidelity_scalar_double_contract():
    from roop_3d.engine.runtime import contract_array
    value = np.array(0.5, np.float64)
    assert contract_array(value, [], dtype=np.dtype("float64"), batch_one=True).shape == ()


def test_codeformer_adapter_runs_a_real_runtime_fidelity_tensor(tmp_path):
    import hashlib
    import onnx
    from roop_3d.models.registry import ModelRecord, ModelRegistry, Manifest, Locator
    from roop_3d.models.catalog import ModelCatalog
    from roop_3d.models.local_provider import LocalWeightProvider
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.compositor.stream_restoration import OnnxFaceRestorer
    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_model(h.make_graph([h.make_node("Cast", ["w"], ["weight"], to=t.FLOAT),
        h.make_node("Mul", ["x", "weight"], ["output"])], "codeformer_contract_fixture",
        [h.make_tensor_value_info("x", t.FLOAT, [1, 3, 512, 512]), h.make_tensor_value_info("w", t.DOUBLE, [])],
        [h.make_tensor_value_info("output", t.FLOAT, ["batch", 3, "height", "width"])]), opset_imports=[h.make_opsetid("", 17)])
    path = tmp_path / "CodeFormerv0.1.onnx"
    onnx.save(graph, path)
    data = path.read_bytes()
    record = ModelRecord(model_id="stage3-codeformer-fixture", name=path.name, role="face_restorer",
        role_evidence="Generated scalar fidelity contract fixture", filenames=(path.name,),
        expected_sha256=hashlib.sha256(data).hexdigest(), byte_size=len(data), status="verified_local",
        locator=Locator(scope="fixture", path=path.name), evidence="Synthetic ONNX graph",
        license_notice="Generated test graph; no production model rights asserted",
        license_id="LicenseRef-Local-Test", license_url="https://roop3d.invalid/test-fixture-terms",
        commercial_use="allow", redistribution_permission="allow", attribution_requirements="No third-party attribution",
        original_repository="LOCAL_TEST_FIXTURE", source_revision="LOCAL_GENERATED_FIXTURE",
        architecture="FixtureNet", required_base_model="NONE", hardware_requirements={"state": "VERIFIED"},
        precision_requirements=("FP32",), dependency_compatibility={"state": "COMPATIBLE", "requirements": []},
        security_status="VERIFIED", serialization_format="ONNX",
        input_output_contract={"inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1, 3, 512, 512]},
            {"name": "w", "dtype": "DOUBLE", "shape": []}],
            "outputs": [{"name": "output", "dtype": "FLOAT", "shape": ["batch", 3, "height", "width"]}]})
    registry = ModelRegistry(manifest=Manifest(evidence_scope="synthetic fixture", models=(record,)))
    catalog = ModelCatalog(registry=registry)
    catalog.provider = LocalWeightProvider(registry, root=tmp_path, scopes={"fixture": tmp_path})
    restorer = OnnxFaceRestorer(catalog, RuntimeConfig(), codeformer_spec(), fidelity_weight=0)
    image = np.full((512, 512, 3), 220, np.uint8)
    try:
        first = restorer.restore(image)
        restorer.fidelity_weight = 1
        second = restorer.restore(image)
        assert np.all(first == 128) and np.all(second == 220)
    finally:
        restorer.close()
    ultra = OnnxFaceRestorer(catalog, RuntimeConfig(), RestorerSpec("ultramax", path.name, CODEFORMER_REFERENCE), fidelity_weight=1)
    try:
        np.testing.assert_array_equal(ultra.restore(image), image)
        pointer = ultra._output.data_ptr()
        np.testing.assert_array_equal(ultra.restore(image), image)
        assert ultra._output.data_ptr() == pointer
    finally:
        ultra.close()
