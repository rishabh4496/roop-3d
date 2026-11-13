"""W2 ops G15 (shared TRT level, INT8 opt-in refusal) and G16 (state mapping totality, atomic checkpoint, resume)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from roop_3d.core import ort_session
from roop_3d.core.ort_session import TRT_BUILDER_OPTIMIZATION_LEVEL, TensorRTInt8Refused, trt_int8_options
from roop_3d.core.storage import atomic_json, owned


# ------------------------------------------------------------------------------------------------------- G15
def test_both_trt_paths_use_the_one_level_constant():
    import inspect
    import roop_3d.trt_session_builder as builder
    assert builder.TRT_BUILDER_OPTIMIZATION_LEVEL is TRT_BUILDER_OPTIMIZATION_LEVEL == 5
    assert ort_session._trt_mixed_options(Path("c"))["trt_builder_optimization_level"] == str(TRT_BUILDER_OPTIMIZATION_LEVEL)
    source = inspect.getsource(builder) + inspect.getsource(ort_session)
    assert '"trt_builder_optimization_level": 5' not in source and '"optimization_level": 3' not in source


def test_int8_is_off_by_default_and_fp16_default_is_untouched(tmp_path):
    assert trt_int8_options("codeformer", tmp_path, {}) == {}
    assert trt_int8_options("codeformer", tmp_path, {"ROOP_TRT_INT8": "0"}) == {}
    assert ort_session._trt_mixed_options(tmp_path)["trt_fp16_enable"] == "True"


def test_int8_without_calibration_cache_is_a_named_refusal(tmp_path):
    with pytest.raises(TensorRTInt8Refused) as raised:
        trt_int8_options("codeformer", tmp_path, {"ROOP_TRT_INT8": "1"})
    assert raised.value.code == "INT8_CALIBRATION_MISSING"
    with pytest.raises(TensorRTInt8Refused) as raised:
        trt_int8_options("codeformer", tmp_path, {"ROOP_TRT_INT8": "1", "ROOP_TRT_INT8_CALIBRATION_CACHE": str(tmp_path / "absent.cache")})
    assert raised.value.code == "INT8_CALIBRATION_MISSING"


def test_int8_with_calibration_in_cache_enables_it_and_outside_cache_is_refused(tmp_path):
    cache = tmp_path / "trt"
    cache.mkdir()
    table = cache / "calib.cache"
    table.write_bytes(b"table")
    options = trt_int8_options("codeformer", cache, {"ROOP_TRT_INT8": "1", "ROOP_TRT_INT8_CALIBRATION_CACHE": str(table)})
    assert options["trt_int8_enable"] == "True" and options["trt_int8_calibration_table_name"] == "calib.cache"
    stray = tmp_path / "stray.cache"
    stray.write_bytes(b"x")
    with pytest.raises(TensorRTInt8Refused) as raised:
        trt_int8_options("codeformer", cache, {"ROOP_TRT_INT8": "1", "ROOP_TRT_INT8_CALIBRATION_CACHE": str(stray)})
    assert raised.value.code == "INT8_CALIBRATION_NOT_IN_CACHE"


@pytest.mark.parametrize("name", ["inswapper_128", "GPEN-BFR-512", "motion_extractor"])
def test_fp32_locked_models_refuse_int8_even_with_a_cache(tmp_path, name):
    table = tmp_path / "c.cache"
    table.write_bytes(b"t")
    with pytest.raises(TensorRTInt8Refused) as raised:
        trt_int8_options(name, tmp_path, {"ROOP_TRT_INT8": "1", "ROOP_TRT_INT8_CALIBRATION_CACHE": str(table)})
    assert raised.value.code == "INT8_FP32_LOCKED_MODEL"


# ------------------------------------------------------------------------------------------------------- G16
def test_spec_mapping_is_total_over_every_enum_member():
    from roop_3d.jobs.schema import JobState
    from roop_3d.queue.lifecycle import SPEC_SLOTS, LifecycleState, SpecState, spec_slot, spec_state_of
    from roop_3d.queue_manager import CanonicalJobState
    members = [*CanonicalJobState, *JobState, *LifecycleState]
    assert len(members) == 12 + 10 + 10
    for member in members:
        assert isinstance(spec_state_of(member), SpecState), member
        assert spec_state_of(member.value) is spec_state_of(member)
        assert spec_slot(spec_state_of(member)) in SPEC_SLOTS
    assert len(SPEC_SLOTS) == 10 and len(SpecState) == 11
    # every spec state is reachable and the pair shares the 10th slot
    reachable = {spec_state_of(m) for m in members} | {spec_state_of("RENDERING", hint) for hint in ("slicing", "restoring", "merging")}
    assert reachable == set(SpecState)
    assert spec_slot(SpecState.failed) == spec_slot(SpecState.cancelled) == "failed/cancelled"


def test_spec_mapping_values_and_stage_refinement_and_unknown_fails_closed():
    from roop_3d.queue.lifecycle import SpecState, spec_state_of
    assert spec_state_of("PREFLIGHT") is SpecState.validating and spec_state_of("PRESCANNING") is SpecState.tracking
    assert spec_state_of("RENDERING") is SpecState.swapping and spec_state_of("MUXING") is SpecState.remuxing
    assert spec_state_of("RENDERING", "restoring faces") is SpecState.restoring
    assert spec_state_of("running", "gop slicing") is SpecState.slicing
    assert spec_state_of("RENDERING", "concat segments") is SpecState.merging
    assert spec_state_of("QUEUED", "restoring") is SpecState.queued          # stage text only refines frame-work states
    assert spec_state_of("succeeded") is SpecState.completed and spec_state_of("paused") is SpecState.queued
    with pytest.raises(ValueError):
        spec_state_of("exploded")


def test_queue_manager_docstrings_no_longer_claim_ten_canonical_states():
    from roop_3d import queue_manager
    from roop_3d.queue_manager import CanonicalJobState, normalize_canonical_state
    assert len(CanonicalJobState) == 12
    assert "12" in normalize_canonical_state.__doc__ and "10 canonical" not in normalize_canonical_state.__doc__
    assert "10-state persistent" not in queue_manager.__doc__


def _checkpoint(job_id: str, completed: int):
    from roop_3d.engine.pipeline_runner import PipelineCheckpoint
    return PipelineCheckpoint(job_id=job_id, input_sha256="a" * 64, request_sha256="b" * 64, total_frames=10, completed_frame=completed)


def _runner(root: Path):
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner
    return MasterPipelineRunner(root, None, None, lambda: None, faceset_reader=object())


def test_resume_continues_after_the_last_checkpointed_frame_without_reemitting(tmp_path):
    from roop_3d.core.storage import atomic_bytes
    job = "c" * 32
    first = _runner(tmp_path)
    emitted: list[int] = []
    for frame in range(10):                       # first process: renders frames, checkpoints every 4th, is "killed" after frame 6
        if frame == 7:
            break
        atomic_bytes(tmp_path, first._frame_relative(job, frame), b"png")
        emitted.append(frame)
        if frame % 4 == 3:
            first._write_checkpoint(_checkpoint(job, frame))
    # truncation: the unfinished frame 6 is on disk but only frames <= 3 were checkpointed
    resumed = _runner(tmp_path)._read_checkpoint(job, "a" * 64, "b" * 64, 10, "cpu")
    assert resumed.completed_frame == 3
    again = [f for f in range(10) if f > resumed.completed_frame]
    assert again[0] == 4 and not set(again) & set(range(0, 4))
    assert emitted == list(range(7))


def test_resume_refuses_a_checkpoint_whose_finished_frames_were_truncated(tmp_path):
    from roop_3d.core.errors import DomainError
    from roop_3d.core.storage import atomic_bytes
    job = "d" * 32
    runner = _runner(tmp_path)
    for frame in range(4):
        atomic_bytes(tmp_path, runner._frame_relative(job, frame), b"png")
    runner._write_checkpoint(_checkpoint(job, 3))
    owned(tmp_path, runner._frame_relative(job, 2)).unlink()
    with pytest.raises(DomainError) as raised:
        _runner(tmp_path)._read_checkpoint(job, "a" * 64, "b" * 64, 10, "cpu")
    assert raised.value.code == "JOB_CHECKPOINT_FRAMES_MISSING"


def test_crash_between_tmp_write_and_rename_leaves_the_previous_checkpoint_readable(tmp_path, monkeypatch):
    from roop_3d.core import storage
    job = "e" * 32
    runner = _runner(tmp_path)
    runner._write_checkpoint(_checkpoint(job, 1))
    previous = owned(tmp_path, runner._checkpoint_relative(job)).read_bytes()
    real_replace = os.replace

    def crash(src, dst):
        raise OSError("simulated power loss before rename")

    monkeypatch.setattr(storage.os, "replace", crash)
    with pytest.raises(OSError):
        atomic_json(tmp_path, runner._checkpoint_relative(job), {"completed_frame": 99})
    monkeypatch.setattr(storage.os, "replace", real_replace)
    path = owned(tmp_path, runner._checkpoint_relative(job))
    assert path.read_bytes() == previous                                   # untouched
    assert not list(path.parent.glob("*.part"))                            # temp file cleaned up
    from roop_3d.engine.pipeline_runner import PipelineCheckpoint
    assert PipelineCheckpoint.model_validate_json(path.read_text("utf-8")).completed_frame == 1


def test_queue_pause_writes_its_npz_checkpoint_atomically(tmp_path, monkeypatch):
    from roop_3d.core import storage
    from roop_3d.engine.pipeline_runner import read_checkpoint_npz
    from roop_3d.queue_manager import CanonicalJobState, QueueManager
    manager = QueueManager(tmp_path)
    job, _created = manager.submit({"target_path": "x.mp4"})
    manager.transition(job.job_id, CanonicalJobState.PREFLIGHT)
    manager.transition(job.job_id, CanonicalJobState.RENDERING)
    checkpoint = _checkpoint(job.job_id, 2)
    manager.pause(job.job_id, checkpoint)
    path = owned(tmp_path, f"roop_3d/.runtime/jobs/{job.job_id}/checkpoint.npz")
    good = path.read_bytes()
    assert read_checkpoint_npz(good).completed_frame == 2
    manager.resume(job.job_id)
    manager.transition(job.job_id, CanonicalJobState.PREFLIGHT)
    manager.transition(job.job_id, CanonicalJobState.RENDERING)
    monkeypatch.setattr(storage.os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        manager.pause(job.job_id, _checkpoint(job.job_id, 7))
    assert path.read_bytes() == good and read_checkpoint_npz(path.read_bytes()).completed_frame == 2
