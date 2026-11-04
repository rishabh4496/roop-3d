"""Stage 18: scaled-down soak runs, judged against the bounded profile. The 30/60-minute runs use the same code via
``python -m roop_3d.reliability.soak`` (results in docs/reliability/)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from roop_3d.jobs.schema import JobState
from roop_3d.reliability import probe
from roop_3d.reliability.harness import StubSession, build_rig
from roop_3d.reliability.ledger import LEDGER
from roop_3d.reliability import soak

RECORD = Path(__file__).resolve().parents[3] / "docs" / "reliability"


def _record(name: str, result: dict) -> None:
    """``ROOP3D_RECORD_SOAK=1`` keeps the measured result next to the docs (the figures quoted in RELIABILITY_STAGE18.md)."""
    if os.environ.get("ROOP3D_RECORD_SOAK") == "1":
        RECORD.mkdir(parents=True, exist_ok=True)
        (RECORD / f"{name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")


def test_the_detector_catches_a_deliberate_leak():
    """A leak detector that has never fired proves nothing: leak ~2 MiB per job and require the verdict to flag it."""
    result = soak.scenario_selftest()
    _record("soak_detector_selftest", result)
    assert result["selftest_leak_detected"], result["growth"]["metrics"]
    assert result["growth"]["violations"]


def test_successive_mixed_jobs_stay_within_the_bounded_profile():
    result = soak.scenario_successive(jobs=30, frames=10, tracks=2)
    _record("soak_successive_30jobs", result)
    assert result["passed"], {"growth": result["growth"]["violations"], "invariants": result["invariants"],
                              "outcomes": result["outcomes"]}
    assert result["outcomes"]["failed"] == 0 and result["leftover_workdirs"] == 0
    assert result["outcomes"]["cancelled"] >= 3 and result["outcomes"]["succeeded"] >= 20


def test_repeated_model_switching_has_no_onnx_session_growth():
    result = soak.scenario_model_switch(cycles=300)
    _record("soak_model_switch_300", result)
    assert result["passed"], result["growth"]["violations"]
    assert result["final_ort_sessions"] == 0


def test_many_facesets_and_target_identities_in_one_job(tmp_path):
    ids = [f"identity-{i}" for i in range(12)]
    rig = build_rig(tmp_path, frames=24, tracks=12, faceset_ids=ids)
    try:
        for index in range(3):
            done = rig.wait(rig.submit(f"multi-{index}").job_id, timeout=90)
            assert done.state == JobState.succeeded and done.output.frames == 24
            manifest = (tmp_path / done.output.manifest_path).read_text("utf-8")
            assert all(identity in manifest for identity in ids[:3])
    finally:
        rig.worker.stop()
    assert LEDGER.live("video_decoder") == 0 and StubSession.live_instances == 0 and not probe.child_pids()


@pytest.mark.skipif(os.environ.get("ROOP3D_SKIP_LONG") == "1", reason="explicitly skipped")
def test_scaled_long_video_is_bounded():
    """A 100-second clip (2,500 frames) through the full worker/checkpoint/decode/encode path."""
    frames = int(os.environ.get("ROOP3D_SOAK_FRAMES", "2500"))
    result = soak.scenario_long_video(minutes=frames / 25 / 60, fps=25, tracks=2, interval=1.0)
    _record("soak_long_video_scaled", result)
    assert result["passed"], {"growth": result["growth"]["violations"], "invariants": result["invariants"],
                              "state": result["state"], "failure": result["failure"]}
    assert result["frames"] == frames
