"""Final workstation stage: parameter curves for relight / occlusion, clean relit pass, transfer ledger, flamegraph lanes, linter.

Every assertion here is about behaviour the code measures or enforces; nothing mocks a result the product claims to measure.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from roop_3d.async_pipeline import AsyncPipeline
from roop_3d.core.animation import BOUNDS, FIELDS, ParameterCurves
from roop_3d.core.transfer_ledger import TransferLedger
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.media.finishing import ExportOptions
from roop_3d.media.vfx import PassCollector
from roop_3d.server import diagnostics
from roop_3d.tests.test_remediation_stage7 import configure, fake_hardware, lint
from roop_3d.tests.test_stage14_workstation import project_fixture


# -- curves ---------------------------------------------------------------------------------------------------------

def curves(name: str, *keys: tuple[int, float]) -> ParameterCurves:
    return ParameterCurves.model_validate({"tracks": {name: [{"frame": f, "value": v} for f, v in keys]}})


@pytest.mark.parametrize("name,field", [("relight_strength", "relight_strength"), ("occlusion_threshold", "occlusion_threshold")])
def test_new_curves_drive_real_workstation_fields_per_frame(name, field):
    track = curves(name, (0, 0.0), (100, 1.0))
    base = WorkstationParameters(relight=True)
    assert getattr(track.evaluate(0, base), field) == pytest.approx(0.0)
    assert getattr(track.evaluate(50, base), field) == pytest.approx(0.5)
    assert getattr(track.evaluate(100, base), field) == pytest.approx(1.0)
    assert FIELDS[name] == field and BOUNDS[name] == (0, 1)


@pytest.mark.parametrize("name", ["relight_strength", "occlusion_threshold"])
def test_curve_values_outside_the_parameter_bounds_are_rejected(name):
    with pytest.raises(ValidationError):
        curves(name, (0, 1.5))


def test_occlusion_threshold_reaches_the_compositor_and_is_opt_in_on_the_wire():
    assert WorkstationParameters().composite_options().semantic_confidence_threshold == pytest.approx(0.55)
    assert WorkstationParameters(occlusion_threshold=0.8).composite_options().semantic_confidence_threshold == pytest.approx(0.8)
    # Earlier requests never set it, so their serialisation (and idempotency fingerprint) must not change.
    assert "occlusion_threshold" not in WorkstationParameters().model_dump()
    assert WorkstationParameters(occlusion_threshold=0.8).model_dump()["occlusion_threshold"] == pytest.approx(0.8)
    with pytest.raises(ValidationError):
        WorkstationParameters(occlusion_threshold=1.2)


# -- clean relit face pass -------------------------------------------------------------------------------------------

def test_relit_is_a_supported_pass_and_the_pass_set_is_still_unique():
    assert ExportOptions(passes=("alpha", "relit")).passes == ("alpha", "relit")
    with pytest.raises(ValidationError):
        ExportOptions(passes=("relit", "relit"))
    with pytest.raises(ValidationError):
        ExportOptions(passes=("not_a_pass",))
    assert len(ExportOptions(passes=("alpha", "occluder", "uv", "normal", "depth", "relit")).passes) == 6


def test_job_request_refuses_the_relit_pass_when_relighting_is_off(tmp_path):
    from roop_3d.jobs.schema import JobRequest
    _, payload, _ = project_fixture(tmp_path)
    export = {"baked_video": False, "passes": ["relit"]}
    with pytest.raises(ValidationError, match="requires SH relighting"):
        JobRequest.model_validate({**payload, "export": export, "workstation_parameters": {"relight": False}})
    ok = JobRequest.model_validate({**payload, "export": export, "workstation_parameters": {"relight": True}})
    assert ok.export.passes == ("relit",)


def test_pass_collector_records_the_relit_face_from_the_colour_stage_only():
    shape = (8, 8)
    collector = PassCollector(shape, ("relit",))
    face = np.full((8, 8, 3), 200, np.uint8)
    alpha = np.zeros((8, 8), np.float32)
    alpha[2:6, 2:6] = 1.0
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    collector.observe("swapper", face, identity, alpha)            # earlier stages must not write the pass
    assert collector.buffers["relit"].max() == 0
    collector.observe("color", face, identity, alpha)
    out = collector.buffers["relit"]
    assert out.shape == (8, 8, 3) and out[3, 3, 0] == pytest.approx(200 / 255)
    assert out[0, 0].max() == 0, "pixels outside the face coverage stay black (no background in a clean pass)"


# -- transfer ledger ---------------------------------------------------------------------------------------------------

class FakeOrt:
    """Minimal stand-in exposing the exact ONNX Runtime attributes the ledger wraps."""
    __version__ = "fake"

    class OrtValue:
        def __init__(self, array, device="cpu"):
            self.array, self.device = array, device

        @classmethod
        def ortvalue_from_numpy(cls, numpy_obj, device_type="cpu", device_id=0, *rest):
            return cls(numpy_obj, device_type)

        def update_inplace(self, np_arr):
            self.array = np_arr

        def numpy(self):
            return self.array

        def device_name(self):
            return self.device

    class InferenceSession:
        def __init__(self, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

        def run(self, names, feed, *rest, **keywords):
            return [np.zeros((2, 2), np.float32)]

        def run_with_iobinding(self, binding):
            return None


def make_ledger() -> tuple[TransferLedger, type]:
    ort = type("Ort", (), {"__version__": "fake", "OrtValue": type("OrtValue", (FakeOrt.OrtValue,), {}),
                           "InferenceSession": type("InferenceSession", (FakeOrt.InferenceSession,), {})})
    ledger = TransferLedger()
    assert ledger.install(ort)
    return ledger, ort


def test_ledger_counts_device_copies_exactly_and_ignores_cpu_traffic():
    ledger, ort = make_ledger()
    frame = np.zeros((4, 4, 3), np.float32)
    ort.OrtValue.ortvalue_from_numpy(frame, "cpu").numpy()                  # CPU only: no PCIe traffic
    assert ledger.snapshot()["total"]["h2d_count"] == 0
    value = ort.OrtValue.ortvalue_from_numpy(frame, "cuda", 0)             # host -> device
    value.update_inplace(frame)                                             # host -> device again
    value.numpy()                                                           # device -> host
    total = ledger.snapshot()["total"]
    assert (total["h2d_count"], total["h2d_bytes"]) == (2, 2 * frame.nbytes)
    assert (total["d2h_count"], total["d2h_bytes"]) == (1, frame.nbytes)


def test_ledger_counts_session_run_on_cuda_but_not_on_cpu():
    ledger, ort = make_ledger()
    feed = {"x": np.zeros((1, 3, 8, 8), np.float32)}
    ort.InferenceSession(["CPUExecutionProvider"]).run(None, feed)
    assert ledger.snapshot()["total"]["h2d_count"] == 0
    ort.InferenceSession(["CUDAExecutionProvider", "CPUExecutionProvider"]).run(None, feed)
    total = ledger.snapshot()["total"]
    assert total["h2d_count"] == 1 and total["h2d_bytes"] == feed["x"].nbytes and total["d2h_count"] == 1


def test_zero_intermediate_copies_is_unknown_until_a_gpu_run_is_observed_then_exact():
    ledger, ort = make_ledger()
    assert ledger.snapshot()["zero_intermediate_copies"] is None, "nothing observed must never read as 'zero copies'"
    ort.InferenceSession(["CUDAExecutionProvider"]).run_with_iobinding(None)
    assert ledger.snapshot()["zero_intermediate_copies"] is True          # a GPU run with every tensor device-resident
    with ledger.scope("swap"):
        ort.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)
    snapshot = ledger.snapshot()
    assert snapshot["zero_intermediate_copies"] is False and snapshot["by_stage"]["swap"]["h2d_count"] == 1


def test_boundary_copies_are_the_allowed_upload_and_download_not_intermediate():
    ledger, ort = make_ledger()
    with ledger.boundary():
        value = ort.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)
        value.numpy()
    snapshot = ledger.snapshot()
    assert snapshot["total"]["boundary_copies"] == 2 and snapshot["total"]["intermediate_copies"] == 0
    assert snapshot["zero_intermediate_copies"] is True


def test_take_returns_a_window_and_resets_it_while_totals_accumulate():
    ledger, ort = make_ledger()
    ort.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)
    assert ledger.take()["h2d_count"] == 1 and ledger.take()["h2d_count"] == 0
    assert ledger.snapshot()["total"]["h2d_count"] == 1


def test_uninstall_restores_the_original_onnxruntime_behaviour():
    ledger, ort = make_ledger()
    patched = ort.OrtValue.numpy
    ledger.uninstall()
    assert ort.OrtValue.numpy is not patched and ledger.installed_on is None
    ort.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0).numpy()
    assert ledger.snapshot()["total"]["d2h_count"] == 0


def test_ledger_installs_on_real_onnxruntime_and_counts_a_real_device_copy():
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("no CUDA execution provider in this environment")
    ledger = TransferLedger()
    assert ledger.install(ort)
    try:
        value = ort.OrtValue.ortvalue_from_numpy(np.zeros((4, 4), np.float32), "cuda", 0)
        value.numpy()
        total = ledger.snapshot()["total"]
        assert (total["h2d_count"], total["d2h_count"], total["h2d_bytes"]) == (1, 1, 64)
    finally:
        ledger.uninstall()


# -- flamegraph lanes --------------------------------------------------------------------------------------------------

def _sleep(ms):
    def run(item):
        time.sleep(ms / 1000)
        return item.payload
    return run


def test_pipeline_snapshot_exposes_measured_lanes_with_overlap_and_honest_gpu_stream_note():
    pipeline = AsyncPipeline("lanes", range(12), queue_size=3)
    pipeline.add_stage("pose", _sleep(8)).add_stage("swap", _sleep(8)).add_stage("write", _sleep(8))
    with pipeline:
        list(pipeline.results())
    snap = pipeline.snapshot(lane_frames=6)
    lanes = snap["lanes"]
    assert set(lanes["stages"]) == {"decode", "pose", "swap", "write"}
    assert all(len(rows) <= 6 for rows in lanes["stages"].values())
    for rows in lanes["stages"].values():
        assert all(end >= start >= 0 for _, start, end in rows)
    frame_rows = [row for row in lanes["stages"]["swap"] if row[0] == 11]
    assert frame_rows, "the lane keeps the frame index so the UI can line stages up per frame"
    assert snap["overlap_wall"]["ge2"] > 0.2, "pipelined stages must be measured as overlapping"
    # torch is not part of this runtime: the snapshot must say CUDA stream windows were NOT measured instead of inventing them.
    if snap["overlap_gpu_streams"] is None:
        assert "PyTorch" in snap["gpu_stream_note"]
    assert "transfers" in snap and snap["transfers"]["scope"] == "onnxruntime-boundary"


def test_stage_workers_attribute_ledger_copies_to_their_stage():
    from roop_3d.core.transfer_ledger import TRANSFERS
    ledger = TransferLedger()
    ort = type("Ort", (), {"__version__": "fake", "OrtValue": type("OrtValue", (FakeOrt.OrtValue,), {}),
                           "InferenceSession": type("InferenceSession", (FakeOrt.InferenceSession,), {})})
    TRANSFERS.reset()
    assert ledger.install(ort)
    TRANSFERS.install(ort)
    try:
        def upload(item):
            ort.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)
            return item.payload
        pipeline = AsyncPipeline("attr", range(5), queue_size=2)
        pipeline.add_stage("swap", upload)
        with pipeline:
            list(pipeline.results())
        assert TRANSFERS.snapshot()["by_stage"]["swap"]["h2d_count"] == 5
    finally:
        TRANSFERS.uninstall()
        TRANSFERS.reset()
        ledger.uninstall()


# -- pre-flight linter -------------------------------------------------------------------------------------------------

def run_lint(tmp_path, monkeypatch, parameters=None, *, hardware=None, mutate=None, export_passes=None, **settings):
    app, payload, _ = project_fixture(tmp_path)
    configure(app, monkeypatch, hardware=hardware, **settings)
    if mutate:
        mutate(app, payload)
    body = {"target_path": payload["target_path"], "inspection_id": payload["inspection_id"],
            "face_mappings": payload["face_mappings"], "workstation_parameters": parameters or None}
    if export_passes:
        body["export_passes"] = export_passes
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/system/preflight-lint", json=body)
    assert response.status_code == 200, response.text
    return {check["id"]: check for check in response.json()["checks"]}, payload


def test_linter_reports_the_four_render_readiness_checks(tmp_path, monkeypatch):
    checks, _ = run_lint(tmp_path, monkeypatch)
    assert {"fsz_extreme_yaw", "trt_engines_built", "ultramax_on_4k", "storage_disk_space"} <= set(checks)


def mapping_ids(payload):
    mapping = payload["face_mappings"][0]
    return mapping["track_id"], mapping["faceset_id"]


@pytest.mark.parametrize("faceset_yaws,expected,needle", [
    ([0.0, 8.0, 30.0], "WARN", "no left-facing view"),                       # target reaches -62, FaceSet only has frontal + right views
    ([-5.0, 0.0, 8.0, 30.0], "WARN", "only 5 deg"),                          # a barely-left view does not cover a -62 turn
    ([-5.0, -55.0, 0.0, 30.0], "PASS", None),                                # -55 is within one pose cell of -62
    (None, "PASS", "not readable"),                                          # unreadable archive = unknown, never a false alarm
])
def test_linter_flags_missing_extreme_yaw_views_against_the_targets_real_range(tmp_path, monkeypatch, faceset_yaws, expected, needle):
    def mutate(_app, payload):
        track_id, _ = mapping_ids(payload)
        monkeypatch.setattr(diagnostics, "_target_yaw_ranges", lambda _root, _id: {track_id: (-62.0, 20.0)})
        monkeypatch.setattr(diagnostics, "_faceset_view_yaws", lambda _state, _fid: faceset_yaws)

    checks, _ = run_lint(tmp_path, monkeypatch, mutate=mutate)
    check = checks["fsz_extreme_yaw"]
    assert check["status"] == expected
    if needle:
        assert needle in " ".join(check["details"])


def test_linter_yaw_check_is_a_pass_when_the_target_never_turns_far(tmp_path, monkeypatch):
    def mutate(_app, payload):
        track_id, _ = mapping_ids(payload)
        monkeypatch.setattr(diagnostics, "_target_yaw_ranges", lambda _root, _id: {track_id: (-20.0, 25.0)})
    checks, _ = run_lint(tmp_path, monkeypatch, mutate=mutate)
    assert checks["fsz_extreme_yaw"]["status"] == "PASS"


@pytest.mark.parametrize("status,expected", [({"built": ["inswapper_128"], "unbuilt": []}, "PASS"),
                                             ({"built": [], "unbuilt": ["inswapper_128", "ultramax"]}, "WARN")])
def test_linter_flags_unbuilt_tensorrt_engines_only_when_tensorrt_is_selected(tmp_path, monkeypatch, status, expected):
    monkeypatch.setattr(diagnostics, "_trt_engine_status", lambda _s, _p, _h: status)
    checks, _ = run_lint(tmp_path, monkeypatch, preferred_provider="TensorrtExecutionProvider")
    assert checks["trt_engines_built"]["status"] == expected
    if expected == "WARN":
        assert "ultramax" in checks["trt_engines_built"]["message"]
    (tmp_path / "cpu").mkdir()
    cpu_checks, _ = run_lint(tmp_path / "cpu", monkeypatch, hardware=fake_hardware(free_mib=-1, providers=("CPUExecutionProvider",)))
    assert cpu_checks["trt_engines_built"]["status"] == "PASS" and "not the selected provider" in cpu_checks["trt_engines_built"]["message"]


@pytest.mark.parametrize("edge,parameters,policy,expected", [
    (2160, None, "off", "WARN"),                                             # 4K, no restorer
    (2160, {"restorer_model": "gpen", "restoration_blend": 0.5}, "off", "WARN"),
    (2160, {"restorer_model": "ultramax", "restoration_blend": 0.0}, "off", "WARN"),   # chosen but blended to nothing
    (2160, {"restorer_model": "ultramax", "restoration_blend": 0.6}, "off", "PASS"),
    (2160, None, "ultramax", "PASS"),                                        # policy supplies UltraMax
    (1080, None, "off", "PASS"),                                             # not 4K
    (0, None, "off", "PASS"),                                                # unprobeable: skip, do not alarm
])
def test_linter_flags_disabled_ultramax_on_4k_targets(tmp_path, monkeypatch, edge, parameters, policy, expected):
    monkeypatch.setattr(diagnostics, "_target_geometry", lambda _s, _p: (edge, 1))
    checks, _ = run_lint(tmp_path, monkeypatch, parameters, face_restoration_policy=policy)
    check = checks["ultramax_on_4k"]
    assert check["status"] == expected
    if expected == "WARN":
        assert check["action"]["parameter"] == "restorer_model" and check["action"]["value"] == "ultramax"


GIB = 1024 ** 3


@pytest.mark.parametrize("free_gib,need_gib,expected", [(50, 10, "PASS"), (12, 10, "WARN"), (3, 10, "FAIL")])
def test_linter_disk_check_compares_free_space_with_the_estimated_render_size(tmp_path, monkeypatch, free_gib, need_gib, expected):
    estimate = {"frames": 1000, "width": 3840, "height": 2160, "checkpoint": need_gib * GIB, "passes": 0, "baked": 0, "total": need_gib * GIB}
    monkeypatch.setattr(diagnostics, "_output_space_estimate", lambda _s, _p: estimate)
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda _p: type("U", (), {"free": free_gib * GIB, "total": 500 * GIB, "used": 0})())
    checks, _ = run_lint(tmp_path, monkeypatch)
    check = checks["storage_disk_space"]
    assert check["status"] == expected
    assert check["estimated_gb"] == pytest.approx(need_gib, abs=0.01)


def test_disk_estimate_scales_with_frames_resolution_and_exported_passes(tmp_path, monkeypatch):
    app, payload, _ = project_fixture(tmp_path)
    fake = type("Probe", (), {"width": 1920, "height": 1080, "fps": 30.0, "declared_frame_count": 300, "duration_seconds": 10.0})()
    monkeypatch.setattr(type(app.state.media), "probe", lambda self, value: fake)
    base = diagnostics.PreFlightLintRequest(target_path=payload["target_path"])
    with_passes = diagnostics.PreFlightLintRequest(target_path=payload["target_path"], export_passes=["alpha", "relit"])
    first, second = diagnostics._output_space_estimate(app.state, base), diagnostics._output_space_estimate(app.state, with_passes)
    assert first["frames"] == 300 and first["checkpoint"] == int(300 * 1920 * 1080 * 3 * diagnostics.PNG_FRACTION)
    assert first["passes"] == 0 and second["passes"] > 0 and second["total"] > first["total"]
    # a 3-channel relit pass costs three times a 1-channel alpha pass
    only_relit = diagnostics._output_space_estimate(app.state, diagnostics.PreFlightLintRequest(target_path=payload["target_path"], export_passes=["relit"]))
    only_alpha = diagnostics._output_space_estimate(app.state, diagnostics.PreFlightLintRequest(target_path=payload["target_path"], export_passes=["alpha"]))
    assert only_relit["passes"] == 3 * only_alpha["passes"]


# -- telemetry endpoint -------------------------------------------------------------------------------------------------

def test_system_pipeline_endpoint_reports_the_transfer_ledger_with_its_scope_and_blind_spots(tmp_path):
    app, _, _ = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        body = client.get("/api/v1/system/pipeline").json()
    transfers = body["transfers"]
    assert transfers["scope"] == "onnxruntime-boundary"
    assert any("NVENC" in item for item in transfers["not_counted"]), "the response must name what it does not count"
    assert transfers["zero_intermediate_copies"] in (None, True, False)
    assert set(transfers["total"]) >= {"h2d_count", "d2h_count", "intermediate_copies", "boundary_copies"}


# -- the ledger only observes: it must never be able to break the code that creates sessions -------------------------------

def test_ledger_declines_an_onnxruntime_shape_it_does_not_understand_without_raising():
    class Session:                                       # a test double with no .run, as test_remediation_stage2 installs
        pass
    ort = type("Ort", (), {"__version__": "double", "OrtValue": FakeOrt.OrtValue, "InferenceSession": Session})
    ledger = TransferLedger()
    assert ledger.install(ort) is False
    assert ledger.installed_on is None and not hasattr(Session, "run")      # nothing was half-patched


def test_ledger_retargets_when_a_different_onnxruntime_module_appears():
    first, second = make_ledger()[1], None
    ledger = TransferLedger()
    assert ledger.install(first)
    second = type("Ort", (), {"__version__": "second", "OrtValue": type("OrtValue", (FakeOrt.OrtValue,), {}),
                              "InferenceSession": type("InferenceSession", (FakeOrt.InferenceSession,), {})})
    assert ledger.install(second) and ledger.installed_on == "second"
    second.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)
    assert ledger.snapshot()["total"]["h2d_count"] == 1
    first.OrtValue.ortvalue_from_numpy(np.zeros(4, np.float32), "cuda", 0)   # the first module was restored: no longer counted
    assert ledger.snapshot()["total"]["h2d_count"] == 1


def test_ledger_restores_everything_and_reports_failure_when_patching_raises_midway():
    class Hostile(type):
        def __setattr__(cls, name, value):
            if name == "run":
                raise RuntimeError("read-only")
            super().__setattr__(name, value)
    ort = type("Ort", (), {"__version__": "hostile", "OrtValue": type("OrtValue", (FakeOrt.OrtValue,), {}),
                           "InferenceSession": Hostile("InferenceSession", (FakeOrt.InferenceSession,), {})})
    original = ort.OrtValue.numpy
    ledger = TransferLedger()
    assert ledger.install(ort) is False
    assert "read-only" in ledger.install_error and ledger.installed_on is None
    assert ort.OrtValue.numpy == original, "the OrtValue patches applied before the failure must be undone"


def test_session_creation_survives_a_test_double_onnxruntime(monkeypatch):
    """The regression this guards: SessionManager.create installs the ledger, and an embedded onnxruntime double must not break it."""
    import sys
    from roop_3d.core.transfer_ledger import TRANSFERS
    double = type(sys)("onnxruntime")
    double.InferenceSession = type("Session", (), {})
    double.OrtValue = FakeOrt.OrtValue
    monkeypatch.setitem(sys.modules, "onnxruntime", double)
    TRANSFERS.uninstall()
    try:
        assert TRANSFERS.install() is False
    finally:
        TRANSFERS.uninstall()


# -- job progress is per stage -------------------------------------------------------------------------------------------

def test_progress_may_restart_after_the_angle_scan_but_never_move_backwards_otherwise(tmp_path):
    from roop_3d.core.errors import DomainError
    from roop_3d.jobs.schema import JobProgress, JobRequest, JobState
    app, payload, _ = project_fixture(tmp_path)
    jobs = app.state.jobs
    job, _ = jobs.create(JobRequest.model_validate(payload), "progress-stage")
    for state in (JobState.preparing, JobState.running):
        job = jobs.transition(job.job_id, state)
    jobs.progress(job.job_id, JobProgress(frame=120, total_frames=120, stage="angle_scan"))        # the scan finishes at N ...
    record = jobs.progress(job.job_id, JobProgress(frame=1, total_frames=120, stage="render"))    # ... and the render starts counting again
    assert record.progress.frame == 1 and record.progress.stage == "render"
    jobs.progress(job.job_id, JobProgress(frame=5, total_frames=120, stage="render"))
    with pytest.raises(DomainError) as error:
        jobs.progress(job.job_id, JobProgress(frame=4, total_frames=120, stage="render"))        # backwards inside a stage is still a defect
    assert error.value.code == "PROGRESS_REGRESSION"
    jobs.progress(job.job_id, JobProgress(frame=9, total_frames=120, stage="detect"))
    with pytest.raises(DomainError):                                                             # a change of stage is not an exemption
        jobs.progress(job.job_id, JobProgress(frame=2, total_frames=120, stage="swap"))


# -- tracker box filtering across a jump ---------------------------------------------------------------------------------

def _tracked(track_id, box):
    from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
    pts = np.zeros((5, 2), np.float32)
    observation = FaceObservation(box=np.asarray(box, np.float32), landmarks5=pts, landmarks106=np.zeros((106, 2), np.float32),
                                  embedding=np.zeros(512, np.float32), pose=(0.0, 0.0, 0.0))
    return TrackedFace(track_id, observation, 1, 0)


def test_tracker_box_filter_does_not_overshoot_after_a_non_overlapping_jump():
    """Regression: a face moving left at 30 px/frame then re-appearing 170 px to the right made the filtered box swing ~110 px past it."""
    from roop_3d.engine.detection.face_tracker import FaceTracker
    from roop_3d.temporal import TrackTemporalStabilizer
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.stabilizer, tracker.stabilize, tracker.frame_index, tracker._last_raw_box = TrackTemporalStabilizer(), True, 0, {}
    x = 640.0
    for _ in range(10):                                           # steady leftward motion builds up velocity
        x -= 30.0
        tracker._stabilize([_tracked(0, (x, 200, x + 90, 330))], {0})
    errors = []
    for _ in range(8):                                            # the face is now stationary 170 px to the right of where it was heading
        out = tracker._stabilize([_tracked(0, (x + 170, 200, x + 260, 330))], {0})[0].observation.box
        errors.append(abs(float(out[0]) - (x + 170)))
    assert max(errors) < 1.0, errors


def test_tracker_box_filter_still_smooths_ordinary_motion():
    from roop_3d.engine.detection.face_tracker import FaceTracker
    from roop_3d.temporal import TrackTemporalStabilizer
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.stabilizer, tracker.stabilize, tracker.frame_index, tracker._last_raw_box = TrackTemporalStabilizer(), True, 0, {}
    rng = np.random.default_rng(0)
    raw, filtered = [], []
    for step in range(40):
        x = 300 + step * 3 + float(rng.normal(0, 2.0))             # overlapping boxes with detector jitter: no reset may happen
        raw.append(x)
        filtered.append(float(tracker._stabilize([_tracked(0, (x, 200, x + 90, 330))], {0})[0].observation.box[0]))
    assert np.std(np.diff(filtered[5:])) < np.std(np.diff(raw[5:])), "jitter must still be smoothed"


def test_app_shutdown_helper_releases_detector_and_recognizer_and_tolerates_partial_services():
    from types import SimpleNamespace
    from roop_3d.server.app import release_media_models
    released = []
    part = lambda name: SimpleNamespace(release_for_build=lambda: released.append(name))
    release_media_models(SimpleNamespace(detector=part("detector"), analyzer=SimpleNamespace(recognizer=part("recognizer"))))
    assert released == ["detector", "recognizer"]
    release_media_models(SimpleNamespace(detector=None))              # services that never loaded: nothing to release, nothing raised
    release_media_models(SimpleNamespace())


def test_reenactment_smoothing_never_overshoots_a_step():
    """Regression: the Kalman predictor behind the engine's smoothing took a 70 degree yaw step to 88 degrees."""
    from roop_3d.reconstruction.reenactment import ReenactmentSettings, _EuroSmoother
    s = ReenactmentSettings()
    smoother = _EuroSmoother(s.pose_min_cutoff, s.pose_beta, s.pose_d_cutoff)
    values = [float(smoother.filter(0, "head_pose", np.array([0.0 if i < 3 else 70.0]), i / 30.0)[0]) for i in range(40)]
    assert max(values) <= 70.0 + 1e-9, max(values)
    assert values[-1] > 68.0, "and it still converges on the target"


def test_app_shutdown_also_releases_the_render_runners_faceset_reader_models():
    from types import SimpleNamespace
    from roop_3d.faceset.faceset_reader import FaceSetReader
    from roop_3d.server.app import release_media_models
    dropped = []
    reader = FaceSetReader.__new__(FaceSetReader)
    reader._detector = SimpleNamespace(release_for_build=lambda: dropped.append("detector"))
    reader._recognizer = SimpleNamespace(release_for_build=lambda: dropped.append("recognizer"))
    release_media_models(SimpleNamespace(), SimpleNamespace(runner=SimpleNamespace(faceset_reader=reader)))
    assert dropped == ["detector", "recognizer"] and reader._detector is None and reader._recognizer is None
