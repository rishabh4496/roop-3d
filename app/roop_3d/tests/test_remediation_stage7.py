"""Stage 7: Stage 27 pre-flight lint checks, measured stage timings and NPZ project checkpoints."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.hardware import GPU, Hardware
from roop_3d.core.stage_timing import STAGES, StageTimings, validate_timings
from roop_3d.engine.pipeline_runner import (MasterPipelineRunner, PipelineCheckpoint, checkpoint_npz_bytes,
                                            read_checkpoint_npz)
from roop_3d.jobs.schema import JobProgress
from roop_3d.server import diagnostics
from roop_3d.tests.test_stage14_workstation import project_fixture


def fake_hardware(*, free_mib: int = 8192, providers=("TensorrtExecutionProvider", "CUDAExecutionProvider")) -> Hardware:
    gpus = [GPU(index=0, name="RTX Test", total_vram_mib=12288, free_vram_mib=free_mib, driver_version="1")] if free_mib >= 0 else []
    return Hardware(timestamp="2026-01-01T00:00:00Z", os={"system": "test"}, logical_cpu_count=8, gpus=gpus,
                    advertised_providers=list(providers))


def lint(client, payload, **parameters):
    response = client.post("/api/v1/system/preflight-lint", json={
        "target_path": payload["target_path"], "inspection_id": payload["inspection_id"],
        "face_mappings": payload["face_mappings"], "workstation_parameters": parameters or None})
    assert response.status_code == 200, response.text
    return {check["id"]: check for check in response.json()["checks"]}


def configure(app, monkeypatch, *, hardware=None, **settings):
    app.state.settings = app.state.settings.model_copy(update=settings)
    monkeypatch.setattr(diagnostics, "probe_hardware", lambda _settings: hardware or fake_hardware())


# -- Stage 27 checks a-d ------------------------------------------------------------------------------------
def test_lint_warns_when_tensorrt_selected_without_dynamic_models_and_offers_conversion(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, preferred_provider="TensorrtExecutionProvider")
    monkeypatch.setattr(diagnostics, "missing_dynamic_models",
                        lambda _state: [{"model": "inswapper_128.onnx", "expected": "inswapper_128_dynamic.onnx"}])
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        check = lint(client, payload)["trt_dynamic_models"]
    assert check["status"] == "WARN" and "inswapper_128_dynamic.onnx" in check["details"][0]
    assert check["action"]["label"] == "Convert & Optimize Now" and check["action"]["endpoint"] == "/system/optimize-models"


def test_lint_trt_check_is_silent_without_nvidia_gpu_or_when_provider_is_cpu(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    monkeypatch.setattr(diagnostics, "missing_dynamic_models", lambda _state: [{"model": "m", "expected": "m_dynamic.onnx"}])
    configure(app, monkeypatch, preferred_provider="TensorrtExecutionProvider", hardware=fake_hardware(free_mib=-1))
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert lint(client, payload)["trt_dynamic_models"]["status"] == "PASS"
    configure(app, monkeypatch, preferred_provider="CPUExecutionProvider")
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert lint(client, payload)["trt_dynamic_models"]["status"] == "PASS"


def test_optimize_endpoint_converts_each_missing_model_and_reports_failures(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    missing = [{"model": "a.onnx", "expected": "a_dynamic.onnx"}, {"model": "b.onnx", "expected": "b_dynamic.onnx"}]
    state = {"left": list(missing)}
    monkeypatch.setattr(diagnostics, "missing_dynamic_models", lambda _s: list(state["left"]))
    monkeypatch.setattr(type(app.state.models), "records", lambda self: [{"name": "a.onnx"}, {"name": "b.onnx"}])
    monkeypatch.setattr(type(app.state.models), "_path", lambda self, record: record["name"])
    from roop_3d.models import dynamic_shapes

    def fake_convert(source):
        if source == "b.onnx":
            raise dynamic_shapes.StaticBatchRequired("static batch only")
        state["left"] = [m for m in state["left"] if m["model"] != source]
        return type("P", (), {"name": "a_dynamic.onnx"})()
    monkeypatch.setattr(dynamic_shapes, "convert", fake_convert)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/system/optimize-models", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    states = {item["model"]: item["state"] for item in body["results"]}
    assert states == {"a.onnx": "CONVERTED", "b.onnx": "STATIC_BATCH_1"} and body["converted"] == 1
    assert [m["model"] for m in body["remaining"]] == ["b.onnx"]


@pytest.mark.parametrize("strength,expected", [(0.8, "WARN"), (0.5, "PASS"), (0.2, "PASS")])
def test_lint_frontalization_above_half_with_populated_pose_matrix(tmp_path, monkeypatch, strength, expected):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch)
    monkeypatch.setattr(diagnostics, "_populated_pose_matrices",
                        lambda _state, _maps: [{"faceset_id": "src", "populated": 7, "filled": 2}])
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        check = lint(client, payload, pose_frontalization=strength)["frontalization_vs_pose_matrix"]
    assert check["status"] == expected
    if expected == "WARN":
        assert "7/9" in check["details"][0] and check["action"]["value"] == 0.5


def test_lint_frontalization_passes_when_matrix_not_populated(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch)
    monkeypatch.setattr(diagnostics, "_populated_pose_matrices", lambda _state, _maps: [])
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert lint(client, payload, pose_frontalization=1.0)["frontalization_vs_pose_matrix"]["status"] == "PASS"


@pytest.mark.parametrize("bypass,pixels,geometry,expected", [
    (False, 320, (2160, 3), "PASS"),    # default: UltraMax restores every face, even on 4K multi-face video
    (True, 320, (2160, 3), "WARN"),     # explicit opt-in bypass skips small faces: say so
    (True, 640, (2160, 3), "WARN"),
    (False, 320, (720, 3), "PASS"),
    (False, 320, (2160, 1), "PASS"),
])
def test_lint_ultramax_warns_only_when_the_small_face_bypass_is_enabled(tmp_path, monkeypatch, bypass, pixels, geometry, expected):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, face_restoration_policy="ultramax", ultramax_native_bypass=bypass, ultramax_min_pixels=pixels)
    monkeypatch.setattr(diagnostics, "_target_geometry", lambda _state, _payload: geometry)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        check = lint(client, payload)["ultramax_crop_bypass"]
    assert check["status"] == expected
    if expected == "WARN":
        assert check["action"]["settings"] == {"ultramax_native_bypass": False}


def test_lint_ultramax_check_ignores_other_restorers(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, face_restoration_policy="gpen", ultramax_native_bypass=False)
    monkeypatch.setattr(diagnostics, "_target_geometry", lambda _state, _payload: (2160, 4))
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert lint(client, payload)["ultramax_crop_bypass"]["status"] == "PASS"


def test_lint_low_vram_with_many_threads_recommends_measured_benchmark_value(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, hardware=fake_hardware(free_mib=2048), execution_threads=6)
    benchmark = app.state.thread_benchmark
    monkeypatch.setattr(benchmark, "snapshot", lambda: {"state": "DONE", "recommendation": {"execution_threads": 2}})
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        check = lint(client, payload)["vram_threads"]
    assert check["status"] == "WARN" and check["recommended_threads"] == 2
    assert check["action"]["settings"] == {"execution_threads": 2} and "measured by benchmark_threads" in check["message"]


def test_lint_low_vram_without_benchmark_uses_conservative_bound(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, hardware=fake_hardware(free_mib=2048), execution_threads=8)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        check = lint(client, payload)["vram_threads"]
    assert check["status"] == "WARN" and check["recommended_threads"] == 1 and "benchmark_threads" in check["message"]


@pytest.mark.parametrize("free,threads", [(8192, 8), (2048, 2), (2048, 3)])
def test_lint_threads_pass_when_vram_or_threads_are_modest(tmp_path, monkeypatch, free, threads):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, hardware=fake_hardware(free_mib=free), execution_threads=threads)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert lint(client, payload)["vram_threads"]["status"] == "PASS"


def test_existing_four_checks_still_present_alongside_the_stage27_checks(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        ids = set(lint(client, payload))
    assert {"3d_pose_coverage", "vram_headroom", "out_of_frame_seams", "output_encoder_storage",
            "trt_dynamic_models", "frontalization_vs_pose_matrix", "ultramax_crop_bypass", "vram_threads"} <= ids


# -- measured stage timings ----------------------------------------------------------------------------------
def test_stage_timings_report_only_measured_stages_as_means():
    timings = StageTimings()
    timings.add("inswapper", 40.0, count=4)
    timings.add("inswapper", 20.0)
    timings.add("decode", 3.0)
    assert timings.take() == {"decode": 3.0, "inswapper": 12.0}
    assert timings.take() == {}                                    # window resets; nothing is invented
    assert set(STAGES) == {"decode", "detection", "pose_tps", "inswapper", "masking", "restorer", "encode"}


@pytest.mark.parametrize("bad", [{"warp": 1.0}, {"decode": -1.0}, {"decode": float("nan")}, {"decode": float("inf")}, {"decode": True}])
def test_job_progress_rejects_unknown_or_invalid_stage_timings(bad):
    with pytest.raises(ValueError):
        JobProgress(frame=1, total_frames=2, stage_timings_ms=bad)
    with pytest.raises(ValueError):
        validate_timings(bad)


def test_reference_runner_maps_engine_latency_to_the_seven_flamegraph_stages():
    result = type("R", (), {"latency": type("L", (), {"detection_ms": 6.0, "swap_ms": 9.0, "retarget_ms": 3.0, "projection_ms": 1.5})()})()
    session = type("S", (), {"compositor": type("C", (), {"last_timing": {"masking": 4.0, "restorer": 14.0}})()})()
    timings = MasterPipelineRunner._stage_timings(result, session, decode_ms=2.0, encode_ms=3.0)
    assert timings == {"decode": 2.0, "detection": 6.0, "inswapper": 9.0, "masking": 4.0, "restorer": 14.0,
                       "encode": 3.0, "pose_tps": 4.5}
    assert MasterPipelineRunner._stage_timings(object(), object(), 1.0, 2.0) == {"decode": 1.0, "encode": 2.0}


def test_stream_processor_measures_decode_swap_and_encode(tmp_path):
    from roop_3d.optimized_processor import StreamingProcessor
    from roop_3d.render_policy import RenderPolicy
    from roop_3d.tools.verify_stream_integration import create_clip
    from roop_3d.vectorized_pipeline import CpuBatchPipeline
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    source = tmp_path / "input.mp4"
    create_clip(source)
    points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4).astype(np.float32)
    processor = StreamingProcessor(tmp_path, "timed", source, detect=lambda _: [1],
        track=lambda _: [(0, np.array([0, 0, 64, 64]), points)], pipeline=CpuBatchPipeline(lambda value, _: value),
        policy=RenderPolicy(hardware_profile="cpu", execution_threads=1, ring_slots=3, segment_frames=4))
    processor.run(tmp_path / "output.mp4", audio=False)
    timings = processor.timings.take()
    assert {"decode", "inswapper", "encode"} <= set(timings) and all(v >= 0 for v in timings.values())
    assert "restorer" not in timings and "masking" not in timings          # not separately measurable on this backend


# -- NPZ project checkpoint ----------------------------------------------------------------------------------
def test_npz_checkpoint_round_trips_inputs_pose_provider_and_hardware():
    checkpoint = PipelineCheckpoint(
        job_id="a" * 32, input_sha256="b" * 64, request_sha256="c" * 64, completed_frame=12, total_frames=40,
        execution_provider="TensorrtExecutionProvider", pose_cache={"track_0000": [0.1, -2.5, 7.0]},
        inputs={"target_path": "roop_3d/.runtime/media/t.mp4", "inspection_id": "d" * 32, "face_mappings": []},
        pose_settings={"pose_frontalization": 0.4, "pose_matrices": {"src": {"fingerprint": "f" * 64, "populated": 6, "filled": 3}}},
        hardware_state={"probe": "VERIFIED_LOCAL", "gpus": [{"name": "RTX Test", "total_vram_mib": 12288, "free_vram_mib": 9000}]})
    restored = read_checkpoint_npz(checkpoint_npz_bytes(checkpoint))
    assert restored.model_dump(exclude={"updated_at"}) == checkpoint.model_dump(exclude={"updated_at"})
    assert restored.pose_cache["track_0000"] == pytest.approx([0.1, -2.5, 7.0])
    assert restored.pose_settings["pose_matrices"]["src"]["populated"] == 6


@pytest.mark.parametrize("data", [b"", b"not an npz", b"PK\x03\x04garbage"])
def test_npz_checkpoint_rejects_corrupt_data(data):
    with pytest.raises(ValueError):
        read_checkpoint_npz(data)


def test_npz_checkpoint_refuses_pickled_object_arrays():
    import io
    buffer = io.BytesIO()
    np.savez(buffer, manifest=np.array([{"x": 1}], dtype=object), pose_cache_keys=np.array([], dtype=object))
    with pytest.raises(ValueError):
        read_checkpoint_npz(buffer.getvalue())


def test_runner_writes_json_bin_and_npz_checkpoints_that_agree(tmp_path):
    runner = object.__new__(MasterPipelineRunner)
    runner.root = tmp_path
    checkpoint = PipelineCheckpoint(job_id="e" * 32, input_sha256="1" * 64, request_sha256="2" * 64, completed_frame=3,
                                    execution_provider="CUDAExecutionProvider", pose_cache={"track_0001": [4.0, 5.0]},
                                    hardware_state={"probe": "VERIFIED_LOCAL"}, pose_settings={"pose_matrices": {}})
    runner._write_checkpoint(checkpoint)
    folder = tmp_path / f"roop_3d/.runtime/jobs/{checkpoint.job_id}"
    assert {path.name for path in folder.iterdir()} == {"checkpoint.json", "checkpoint.bin", "checkpoint.npz"}
    from_json = PipelineCheckpoint.model_validate_json((folder / "checkpoint.json").read_text("utf-8"))
    from_npz = read_checkpoint_npz((folder / "checkpoint.npz").read_bytes())
    assert from_npz == from_json
    runner.cleanup(checkpoint.job_id)
    assert not folder.exists()                      # the NPZ twin is removed with the rest of the job work directory
