"""Hair / neck / ear protection toggles and the skin micro-texture amount: real engine behind each control, opt-in on the wire.

The defaults must equal the compositor's previous hard-coded behaviour (so nothing changes until a control is touched), and each toggle
must really change the class set or matte it names.
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from roop_3d.compositor.masks import OCCLUDER_CLASS_IDS, STRICT_FORBIDDEN_CLASSES, apply_submental_jaw_cutoff
from roop_3d.compositor.hairline import refine_hairline_matte
from roop_3d.compositor.pipeline import CompositeOptions, protect_matte
from roop_3d.compositor.quality import inject_target_skin_micro_texture
from roop_3d.core.workstation import PROTECTION_FIELDS, WorkstationParameters
from roop_3d.jobs.schema import JobRequest
from roop_3d.server.live_preview import LivePreviewPayload

NECK, EARS_GLASSES, HAIR_HAT = {14, 15, 16}, {6, 7, 8, 9}, {17, 18}


def _jaw_landmarks() -> np.ndarray:
    points = np.zeros((68, 2), np.float32)
    for i in range(17):
        t = i / 16.0
        points[i] = [20.0 + t * 88.0, 50.0 + 30.0 * (1.0 - 4.0 * (t - 0.5) ** 2)]
    return points


def _payload(**extra) -> LivePreviewPayload:
    return LivePreviewPayload(inspection_id="a" * 32, target_path="roop_3d/.runtime/media/t.mp4", frame_index=3,
                              source_face_ids=("alice",), target_face_id="track_0000", **extra)


# ------------------------------------------------------------------------------------------------ defaults are the old behaviour
def test_defaults_reproduce_the_hard_coded_stage3_behaviour() -> None:
    options = CompositeOptions()
    assert options.forbidden_classes() == STRICT_FORBIDDEN_CLASSES
    assert options.front_classes() == OCCLUDER_CLASS_IDS
    assert options.jaw_falloff() == 6.0 and options.skin_texture_retention == 0.25
    assert options.neck_protection and options.hair_matting and options.preserve_ears_glasses


def test_each_toggle_drops_exactly_its_classes_and_hair_is_never_dropped() -> None:
    no_neck = set(CompositeOptions(neck_protection=False).forbidden_classes())
    assert no_neck == set(STRICT_FORBIDDEN_CLASSES) - NECK
    no_ears = CompositeOptions(preserve_ears_glasses=False)
    assert set(no_ears.forbidden_classes()) == set(STRICT_FORBIDDEN_CLASSES) - EARS_GLASSES
    assert set(no_ears.front_classes()) == set(OCCLUDER_CLASS_IDS) - {6}          # glasses stop being a semantic occluder, hair/hat stay
    both = set(CompositeOptions(neck_protection=False, preserve_ears_glasses=False).forbidden_classes())
    assert HAIR_HAT <= both and not (both & (NECK | EARS_GLASSES))


@pytest.mark.parametrize("bad", [{"jaw_feather_px": 1.9}, {"jaw_feather_px": 24.1}, {"jaw_feather_px": float("nan")},
                                 {"skin_texture_retention": -0.01}, {"skin_texture_retention": 1.01}])
def test_compositor_options_reject_out_of_range_values(bad: dict) -> None:
    with pytest.raises(ValueError):
        CompositeOptions(**bad)


def test_jaw_falloff_is_clamped_to_what_the_cutoff_accepts_at_any_working_scale() -> None:
    assert CompositeOptions(jaw_feather_px=24).jaw_falloff(scale=4.0) == 32.0
    assert CompositeOptions(jaw_feather_px=2).jaw_falloff(scale=0.25) == 2.0


# ------------------------------------------------------------------------------------------------ the toggles change the matte
def test_neck_protection_clamps_the_matte_along_the_jaw_only_when_on() -> None:
    mask = np.ones((128, 128), np.float32)
    guide = np.full((128, 128, 3), 180, np.uint8)
    on = protect_matte(mask, guide, _jaw_landmarks(), CompositeOptions(feather_px=0))
    off = protect_matte(mask, guide, _jaw_landmarks(), CompositeOptions(feather_px=0, neck_protection=False))
    assert np.all(on[95:] == 0.0), "the throat must be cut off"
    assert np.all(off == 1.0), "with protection off the matte is left alone"


def test_jaw_feather_falloff_widens_the_transition_band() -> None:
    mask = np.ones((128, 128), np.float32)
    guide = np.full((128, 128, 3), 180, np.uint8)
    # Matting off and no feather isolate the jaw clamp (the guided filter would also touch the forehead band).
    tight = protect_matte(mask, guide, _jaw_landmarks(), CompositeOptions(feather_px=0, hair_matting=False, jaw_feather_px=2))
    wide = protect_matte(mask, guide, _jaw_landmarks(), CompositeOptions(feather_px=0, hair_matting=False, jaw_feather_px=24))
    band = lambda matte: int(np.count_nonzero((matte[:, 64] > 0.02) & (matte[:, 64] < 0.98)))
    assert band(wide) > band(tight)
    assert np.array_equal(tight, apply_submental_jaw_cutoff(mask, _jaw_landmarks(), falloff_px=2.0))


def test_hair_matting_uses_the_guided_filter_only_when_on() -> None:
    guide = np.full((128, 128, 3), 180, np.uint8)
    guide[:50, 62:66] = 20                                   # a dark hair strand across the forehead
    mask = np.ones((128, 128), np.float32)
    mask[:50, 62:66] = 0.3
    options = CompositeOptions(feather_px=4, neck_protection=False)
    on = protect_matte(mask, guide, None, options)
    off = protect_matte(mask, guide, None, CompositeOptions(feather_px=4, neck_protection=False, hair_matting=False))
    assert np.array_equal(on, refine_hairline_matte(mask, guide, radius=8, eps=1e-4, smooth_feather_px=4.0))
    assert on[20, 64] < off[20, 64], "the guided filter keeps the strand sharper than a Gaussian feather"
    assert off.dtype == np.float32 and 0 <= off.min() and off.max() <= 1


def test_zero_feather_without_matting_leaves_the_matte_unblurred() -> None:
    mask = np.zeros((32, 32), np.float32)
    mask[8:24, 8:24] = 1
    result = protect_matte(mask, np.zeros((32, 32, 3), np.uint8), None, CompositeOptions(feather_px=0, neck_protection=False, hair_matting=False))
    assert np.array_equal(result, mask)


def test_texture_retention_scales_the_reinjected_pore_residual() -> None:
    rng = np.random.default_rng(3)
    target = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
    enhanced = np.full((64, 64, 3), 128, np.uint8)
    skin = np.ones((64, 64), np.float32)
    deviation = lambda rate: float(np.abs(inject_target_skin_micro_texture(enhanced, target, skin, skin, injection_rate=rate).astype(np.float32) - 128).mean())
    assert deviation(0.0) == 0.0
    assert 0 < deviation(0.25) < deviation(1.0)


# ------------------------------------------------------------------------------------------------ the wire contract
def test_untouched_parameters_carry_none_of_the_protection_fields() -> None:
    dumped = WorkstationParameters().model_dump()
    assert not (set(PROTECTION_FIELDS) & set(dumped))
    touched = WorkstationParameters(neck_protection=False, jaw_feather_px=12, skin_texture_retention=0.6)
    assert {"neck_protection", "jaw_feather_px", "skin_texture_retention"} <= set(touched.model_dump())
    assert "hair_matting" not in touched.model_dump()
    assert WorkstationParameters.model_validate(touched.model_dump()) == touched


@pytest.mark.parametrize("bad", [{"jaw_feather_px": 1}, {"jaw_feather_px": 25}, {"skin_texture_retention": -0.1}, {"skin_texture_retention": 1.1}])
def test_the_new_parameters_are_bounded(bad: dict) -> None:
    with pytest.raises(ValidationError):
        WorkstationParameters(**bad)


def test_parameters_reach_the_compositor_options() -> None:
    options = WorkstationParameters(neck_protection=False, hair_matting=False, preserve_ears_glasses=False,
                                    jaw_feather_px=14, skin_texture_retention=0.8).composite_options()
    assert (options.neck_protection, options.hair_matting, options.preserve_ears_glasses) == (False, False, False)
    assert options.jaw_feather_px == 14 and options.skin_texture_retention == 0.8
    default = WorkstationParameters().composite_options()
    assert default.forbidden_classes() == STRICT_FORBIDDEN_CLASSES and default.jaw_feather_px == 6.0


def test_a_job_request_keeps_its_fingerprint_until_a_control_is_touched() -> None:
    base = dict(target_path="roop_3d/.runtime/media/t.mp4", inspection_id="a" * 32, face_mappings=[{"track_id": "track_0000", "faceset_id": "alice"}])
    plain = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4}}).identity_payload()["workstation_parameters"]
    touched = JobRequest.model_validate({**base, "workstation_parameters": {"feather_px": 4, "neck_protection": False}}).identity_payload()["workstation_parameters"]
    assert not (set(PROTECTION_FIELDS) & set(plain))
    assert touched["neck_protection"] is False


def test_live_preview_keeps_old_cache_keys_when_absent_and_separates_them_when_set() -> None:
    plain = _payload().config_digest()
    assert not (set(PROTECTION_FIELDS) & set(plain))
    touched = _payload(hair_matting=False, jaw_feather_px=10, skin_texture_retention=0.5).config_digest()
    assert touched["hair_matting"] is False and touched["jaw_feather_px"] == 10 and touched["skin_texture_retention"] == 0.5
    assert touched != plain
    for bad in ({"jaw_feather_px": 30}, {"skin_texture_retention": 2}):
        with pytest.raises(ValidationError):
            _payload(**bad)
