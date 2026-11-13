"""Stage 6: UltraMax is a real, selectable restorer in the preview and render paths (it used to be rejected outright)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
from roop_3d.compositor.stream_restoration import CODEFORMER_REFERENCE
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.server.live_preview import LivePreviewPayload
from roop_3d.tests.quality_world import restorer_catalog


def _payload(**extra) -> LivePreviewPayload:
    return LivePreviewPayload(inspection_id="a" * 32, target_path="roop_3d/.runtime/media/t.mp4", frame_index=3,
                              source_face_ids=("alice",), target_face_id="track_0000", **extra)


def test_workstation_parameters_accept_ultramax_and_keep_the_wire_format():
    parameters = WorkstationParameters.model_validate({"restorer_model": "ultramax", "restoration_blend": 0.6})
    assert parameters.restorer_model == "ultramax"
    assert WorkstationParameters.model_validate(parameters.model_dump()).restorer_model == "ultramax"
    with pytest.raises(ValueError):
        WorkstationParameters.model_validate({"restorer_model": "nonsense"})


def test_live_preview_accepts_every_restorer_with_its_own_adapter():
    for name in ("gpen", "codeformer", "gfpgan", "ultramax"):
        payload = _payload(restorer=name)
        payload.validate_restorer()
        assert payload.restorer == name and "restorer" not in payload.unsupported()
    assert "fidelity_weight" in _payload(restorer="gfpgan", fidelity_weight=0.2).unsupported()
    assert "fidelity_weight" not in _payload(restorer="ultramax", fidelity_weight=0.2).unsupported()


def test_ultramax_keeps_the_swap_colour_and_takes_only_restorer_detail(tmp_path):
    rng = np.random.default_rng(7)
    crop = cv2.GaussianBlur(rng.integers(60, 200, (512, 512, 3), dtype=np.uint8), (0, 0), 3)
    restorer = UltraMaxPipelineRestorer(restorer_catalog(tmp_path), RuntimeConfig())
    try:
        fused = restorer.restore_aligned(crop, CODEFORMER_REFERENCE)
        assert fused.shape == crop.shape and fused.dtype == np.uint8
        assert np.max(np.abs(fused.mean((0, 1)) - crop.mean((0, 1)))) < 4
        sharpness = lambda image: float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        assert sharpness(fused) > sharpness(crop), "the observed graph's fine detail must reach the output"
        assert restorer.capability()["telemetry"]["applied"] == 1
    finally:
        restorer.release()
