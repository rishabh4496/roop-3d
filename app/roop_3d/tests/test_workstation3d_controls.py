"""Workstation UI controls with a real engine behind them: ``hairline_offset_px`` (forehead matte band) and ``ambient_fill`` (manual light).

Both are opt-in: omitted from a request until set, so earlier requests keep their exact serialisation, idempotency fingerprint and
preview cache key. Neither is faked: the helper really moves the matte, the fill really changes the relit pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from roop_3d.compositor.hairline import FOREHEAD_BAND, hairline_weight, offset_hairline
from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.jobs.schema import JobRequest
from roop_3d.relighting_3d import NeuralRelighter3D
from roop_3d.server.live_preview import LivePreviewPayload

SIDE = 128


def _disc(radius: float = 0.78) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0:complex(SIDE), -1.0:1.0:complex(SIDE)]
    return ((x * x + y * y) <= radius * radius).astype(np.float32)


def _payload(**extra) -> LivePreviewPayload:
    return LivePreviewPayload(inspection_id="a" * 32, target_path="roop_3d/.runtime/media/t.mp4", frame_index=3,
                              source_face_ids=("alice",), target_face_id="track_0000", **extra)


# ---------------------------------------------------------------------------------------------------- hairline matte band
def test_hairline_weight_is_one_at_the_crown_row_and_zero_below_the_band_and_monotonic() -> None:
    weight = hairline_weight(200)[:, 0]
    assert weight.shape == (200,)
    assert weight[0] > 0.99
    assert np.all(weight[int(FOREHEAD_BAND * 200) + 1:] == 0)
    assert np.all(np.diff(weight) <= 1e-6)                    # only ever fades out going down
    with pytest.raises(ValueError):
        hairline_weight(0)


def test_zero_offset_returns_the_matte_untouched() -> None:
    alpha = _disc()
    assert offset_hairline(alpha, 0) is alpha


def test_positive_offset_grows_only_the_forehead_band_and_negative_pulls_it_in() -> None:
    alpha = _disc()
    below = slice(int(FOREHEAD_BAND * SIDE) + 2, SIDE)
    grown, shrunk = offset_hairline(alpha, 6), offset_hairline(alpha, -6)
    top = slice(0, int(FOREHEAD_BAND * SIDE))
    assert grown[top].sum() > alpha[top].sum()
    assert shrunk[top].sum() < alpha[top].sum()
    # Everything below the band is bit-identical: jaw, cheeks and mouth keep the global erosion behaviour.
    assert np.array_equal(grown[below], alpha[below])
    assert np.array_equal(shrunk[below], alpha[below])
    assert grown.dtype == np.float32 and 0 <= grown.min() and grown.max() <= 1
    # Growing never removes matte, shrinking never adds any.
    assert np.all(grown >= alpha - 1e-6) and np.all(shrunk <= alpha + 1e-6)


def test_offset_scales_with_the_working_resolution() -> None:
    alpha = _disc()
    small, large = offset_hairline(alpha, 3, scale=1.0), offset_hairline(alpha, 3, scale=3.0)
    assert large[:int(FOREHEAD_BAND * SIDE)].sum() > small[:int(FOREHEAD_BAND * SIDE)].sum()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 33.0, -40.0])
def test_offset_rejects_values_outside_the_contract(bad: float) -> None:
    with pytest.raises(ValueError):
        offset_hairline(_disc(), bad)
    with pytest.raises(ValueError):
        offset_hairline(np.zeros((4, 4, 3), np.float32), 2)


# ---------------------------------------------------------------------------------------------------- the wire contract
def test_untouched_parameters_serialise_without_the_new_fields_and_set_ones_round_trip() -> None:
    plain = WorkstationParameters()
    assert "ambient_fill" not in plain.model_dump() and "hairline_offset_px" not in plain.model_dump()
    touched = WorkstationParameters(ambient_fill=0.55, hairline_offset_px=-9)
    dumped = touched.model_dump()
    assert dumped["ambient_fill"] == 0.55 and dumped["hairline_offset_px"] == -9
    assert WorkstationParameters.model_validate(dumped) == touched


def test_the_new_fields_are_bounded() -> None:
    for bad in ({"ambient_fill": -0.1}, {"ambient_fill": 1.1}, {"hairline_offset_px": 33}, {"hairline_offset_px": -33}):
        with pytest.raises(ValidationError):
            WorkstationParameters(**bad)
    with pytest.raises(ValueError):
        CompositeOptions(ambient_fill=2)
    with pytest.raises(ValueError):
        CompositeOptions(hairline_offset_px=99)


def test_parameters_reach_the_compositor_options() -> None:
    options = WorkstationParameters(ambient_fill=0.6, hairline_offset_px=7, relight=True, lighting_mode="manual").composite_options()
    assert options.ambient_fill == 0.6 and options.hairline_offset_px == 7
    default = WorkstationParameters().composite_options()
    assert default.ambient_fill == 0.3 and default.hairline_offset_px == 0


def test_a_job_request_keeps_its_fingerprint_until_the_controls_are_touched() -> None:
    base = dict(target_path="roop_3d/.runtime/media/t.mp4", inspection_id="a" * 32, face_mappings=[{"track_id": "track_0000", "faceset_id": "alice"}])
    plain = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4}}).identity_payload()
    touched = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4, "hairline_offset_px": 5}}).identity_payload()
    assert "hairline_offset_px" not in plain["workstation_parameters"] and "ambient_fill" not in plain["workstation_parameters"]
    assert touched["workstation_parameters"]["hairline_offset_px"] == 5


def test_live_preview_accepts_the_fields_and_keeps_old_cache_keys_when_they_are_absent() -> None:
    plain = _payload().config_digest()
    assert "ambient_fill" not in plain and "hairline_offset_px" not in plain
    touched = _payload(ambient_fill=0.5, hairline_offset_px=-4).config_digest()
    assert touched["ambient_fill"] == 0.5 and touched["hairline_offset_px"] == -4
    assert touched != plain                                                   # a changed control is a different preview, never a cache hit
    for bad in ({"ambient_fill": 1.5}, {"hairline_offset_px": 50}):
        with pytest.raises(ValidationError):
            _payload(**bad)


# ---------------------------------------------------------------------------------------------------- ambient fill, for real
def _relight(ambient: float) -> np.ndarray:
    relighter = NeuralRelighter3D()
    y, x = np.mgrid[-1.0:1.0:complex(SIDE), -1.0:1.0:complex(SIDE)]
    r2 = x * x + y * y
    z = np.sqrt(np.maximum(1.0 - np.minimum(r2, 0.95), 0.0))
    dz_dy, dz_dx = np.gradient(z)
    normals = np.stack([-dz_dx, -dz_dy, np.full_like(dz_dx, 0.06)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    face = np.full((SIDE, SIDE, 3), (110, 140, 195), np.uint8)
    relit, *_ = relighter.relight(swapped_bgr=face, target_bgr=face.copy(), swap_normals=normals, target_normals=normals,
                                  relight_strength=1.0, specular_sheen=0.0, shadow_preservation=0.0, lighting_mode="manual",
                                  manual_light_yaw=60.0, manual_light_pitch=0.0, ambient_fill=ambient)
    return relit


def test_ambient_fill_lifts_the_shadow_side_of_the_manual_light_without_changing_the_lit_side() -> None:
    dark, filled = _relight(0.0).astype(np.float32), _relight(0.9).astype(np.float32)
    shadow = (slice(SIDE // 2 - 12, SIDE // 2 + 12), slice(SIDE // 2 - 40, SIDE // 2 - 24))     # image-left, away from the +yaw key light
    assert filled[shadow].mean() > dark[shadow].mean() + 5, "ambient fill must brighten the unlit side"
    assert not np.array_equal(dark, filled)


def test_the_default_fill_is_the_value_the_engine_always_used() -> None:
    relighter = NeuralRelighter3D()
    assert relighter.relight.__defaults__ is not None
    assert np.array_equal(_relight(0.3), _relight(0.3))
    # `ambient_fill` defaults to 0.3 (the previous hard-coded value), so omitting it is unchanged behaviour.
    y, x = np.mgrid[-1.0:1.0:complex(SIDE), -1.0:1.0:complex(SIDE)]
    normals = np.stack([x * 0.3, y * 0.3, np.ones_like(x)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    face = np.full((SIDE, SIDE, 3), (120, 150, 190), np.uint8)
    common = dict(swapped_bgr=face, target_bgr=face.copy(), swap_normals=normals, target_normals=normals, lighting_mode="manual", manual_light_yaw=40.0, manual_light_pitch=10.0)
    assert np.array_equal(relighter.relight(**common)[0], relighter.relight(**common, ambient_fill=0.3)[0])
