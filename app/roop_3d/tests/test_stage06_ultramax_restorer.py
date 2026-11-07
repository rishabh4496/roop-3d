"""Stage 6: UltraMax is a real, selectable restorer in the preview and render paths (it used to be rejected outright)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.compositor.restoration import CodeFormerRestorer, UltraMaxPipelineRestorer
from roop_3d.core.errors import DomainError
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.server.live_preview import LivePreviewPayload


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
        _payload(restorer=name).validate_restorer()          # must not raise
    assert "fidelity_weight" in _payload(restorer="gfpgan", fidelity_weight=0.2).unsupported()
    assert "fidelity_weight" not in _payload(restorer="ultramax", fidelity_weight=0.2).unsupported()


def test_ultramax_keeps_the_swap_colour_and_takes_only_restorer_detail(monkeypatch):
    rng = np.random.default_rng(7)
    crop = np.clip(cv2.GaussianBlur(rng.integers(60, 200, (128, 128, 3), dtype=np.uint8), (0, 0), 3), 0, 255).astype(np.uint8)
    # A "restorer" that shifts the colour a lot and adds high-frequency detail.
    detail = cv2.addWeighted(crop, 1.0, cv2.Laplacian(crop, cv2.CV_8U), 0.8, 0)
    shifted = np.clip(detail.astype(np.int16) + np.array([40, -30, 25]), 0, 255).astype(np.uint8)
    monkeypatch.setattr(CodeFormerRestorer, "__init__", lambda self, catalog, config: setattr(self, "fidelity_weight", 0.5) or setattr(self, "inference_verified", False))
    monkeypatch.setattr(CodeFormerRestorer, "restore_aligned", lambda self, image, landmarks5: shifted)
    restorer = UltraMaxPipelineRestorer(None, None)
    fused = restorer.restore_aligned(crop, np.zeros((5, 2)))
    assert fused.shape == crop.shape and fused.dtype == np.uint8
    colour_error_fused = float(np.abs(fused.mean((0, 1)) - crop.mean((0, 1))).max())
    colour_error_raw = float(np.abs(shifted.mean((0, 1)) - crop.mean((0, 1))).max())
    assert colour_error_fused < colour_error_raw / 2, "low frequencies / colour must stay the swap crop's"
    sharpness = lambda image: float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    assert sharpness(fused) > sharpness(crop), "the restorer's fine detail must reach the output"
    assert restorer.fidelity_weight == 0.7 and restorer.capability.__name__ == "capability"
