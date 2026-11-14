"""Stage 0.5: every frame must pass the ordered pure-3D pipeline; a failed 3D fit skips cleanly and never falls back."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.true3d_guard import FRAME_PIPELINE_STAGES, True3DFramePipelineError, require_true3d_frame_pipeline
from roop_3d.pipeline.session_adapter import SessionFrameProcessor


def test_the_full_ordered_stage_sequence_is_accepted():
    assert require_true3d_frame_pipeline(FRAME_PIPELINE_STAGES) == FRAME_PIPELINE_STAGES
    assert require_true3d_frame_pipeline(FRAME_PIPELINE_STAGES[:3], through=FRAME_PIPELINE_STAGES[2]) == FRAME_PIPELINE_STAGES[:3]


@pytest.mark.parametrize("stages", [
    (),                                                              # nothing was fitted
    FRAME_PIPELINE_STAGES[1:],                                       # a texture/raster layer with no 3D mesh fit
    (FRAME_PIPELINE_STAGES[1], FRAME_PIPELINE_STAGES[0]),            # out of order
    ("2D Affine Crop Paste",),                                       # a foreign stage is never accepted
])
def test_a_frame_missing_or_reordering_a_stage_is_rejected_with_the_next_expected_stage(stages):
    with pytest.raises(True3DFramePipelineError) as error:
        require_true3d_frame_pipeline(stages)
    assert error.value.code == "TRUE3D_FRAME_PIPELINE_INVALID"
    assert error.value.details["observed"] == list(stages)
    assert error.value.details["next_stage"] in FRAME_PIPELINE_STAGES


class _Session:
    catalog = None
    config = None

    def __init__(self, error: DomainError):
        self.error = error

    def track_frame(self, frame):
        return None

    def infer_frame(self, frame, tracked=None):
        raise self.error


def test_a_failed_3d_fit_logs_a_structured_diagnostic_and_returns_the_frame_untouched(caplog):
    frame = np.zeros((16, 16, 3), np.uint8)
    processor = SessionFrameProcessor(_Session(DomainError("HEAD_POSE_FAILED", "occluded", 422)), SimpleNamespace())
    with caplog.at_level(logging.WARNING, logger="roop3d.pipeline"):
        output, report = processor.process(frame, 0)
    assert output is frame
    assert report.skipped == [{"event": "true3d_frame_skipped", "code": "HEAD_POSE_FAILED", "message": "occluded"}]
    assert any("true3d_frame_skipped" in record.getMessage() and "HEAD_POSE_FAILED" in record.getMessage() for record in caplog.records)


def test_errors_that_are_not_3d_fit_failures_are_never_swallowed():
    processor = SessionFrameProcessor(_Session(DomainError("RENDERER_NOT_WIRED", "3D renderer not wired", 503)), SimpleNamespace())
    with pytest.raises(DomainError) as error:
        processor.process(np.zeros((16, 16, 3), np.uint8), 0)
    assert error.value.code == "RENDERER_NOT_WIRED"
