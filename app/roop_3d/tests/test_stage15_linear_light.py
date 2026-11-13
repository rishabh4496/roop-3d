"""Stage 15 linear-light compositing: analytic expectations, exact identity at the extremes, every blend mode."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.color import transfer
from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop, fuse_warped
from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.compositor.quality import blend_linear_light, linear_lut
from roop_3d.core.workstation import WorkstationParameters

NAMES = transfer.BLEND_TRANSFERS


def flat(value, shape=(16, 16)):
    return np.full((*shape, 3), value, np.uint8)


@pytest.mark.parametrize("name,expected", [("srgb", 188), ("bt1886", 191), ("gamma22", 186), ("linear", 128)])
def test_half_coverage_of_black_and_white_follows_the_named_transfer(name, expected):
    # Analytic: encode(0.5 * decode(0) + 0.5 * decode(1)) = encode(0.5). sRGB 0.7354, 1/2.4 -> 0.7491, 1/2.2 -> 0.7297.
    mixed = blend_linear_light(flat(0), flat(255), np.full((16, 16), 0.5, np.float32), name)
    assert np.allclose(mixed, expected, atol=0.6)
    assert not np.allclose(mixed, 127.5, atol=2) or name == "linear"  # gamma-space mixing would give 127.5


@pytest.mark.parametrize("name", NAMES)
def test_alpha_zero_and_one_return_the_inputs_exactly_for_every_code_value(name):
    codes = np.arange(256, dtype=np.uint8)
    target = np.broadcast_to(codes[:, None, None], (256, 4, 3)).copy()
    source = target[::-1].copy()
    for alpha, want in ((0.0, target), (1.0, source)):
        got = blend_linear_light(target, source, np.full((256, 4), alpha, np.float32), name)
        assert np.array_equal(np.rint(got).astype(np.uint8), want), (name, alpha)


@pytest.mark.parametrize("name", NAMES)
def test_blend_is_monotonic_in_alpha_and_bounded_by_its_inputs(name):
    t, s = flat(40), flat(200)
    previous = -1.0
    for a in np.linspace(0, 1, 21):
        value = float(blend_linear_light(t, s, np.full((16, 16), a, np.float32), name)[0, 0, 0])
        assert value >= previous - 1e-6 and 40 - 0.6 <= value <= 200 + 0.6
        previous = value


def test_linear_mix_equals_the_encoded_average_of_linear_values():
    rng = np.random.default_rng(2)
    t = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    s = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    a = rng.random((8, 8)).astype(np.float32)
    got = blend_linear_light(t, s, a, "bt1886")
    lin = (transfer.to_linear("bt1886", t / 255.0) * (1 - a[..., None]) + transfer.to_linear("bt1886", s / 255.0) * a[..., None])
    assert np.allclose(got, transfer.from_linear("bt1886", lin) * 255, atol=0.02)
    assert np.allclose(linear_lut("srgb"), transfer.to_linear("srgb", np.arange(256) / 255), atol=1e-7)


def square_alpha(size=128, low=48, high=80, soft=7):
    """Compact-support alpha (a box blur reaches only ``soft // 2`` px), so 'outside the support' is well defined."""
    import cv2
    alpha = np.zeros((size, size), np.float32)
    alpha[low:high, low:high] = 1
    return cv2.blur(alpha, (soft, soft))


def ramp_source(size=128):
    """Bright textured source: gradient-domain (Poisson) cloning has gradients to transfer, unlike a flat image."""
    ramp = np.linspace(110, 230, size, dtype=np.float32)[None, :, None]
    return np.broadcast_to(ramp, (size, size, 3)).astype(np.uint8).copy()


@pytest.mark.parametrize("mode", [BlendMode.FEATHER, BlendMode.MULTI_BAND])
def test_blend_modes_honour_linear_light_and_only_touch_their_support(mode):
    target, source = flat(30, (128, 128)), ramp_source()
    alpha = square_alpha()
    gamma = fuse_warped(target, source, alpha, mode=mode, linear_light=False)
    linear = fuse_warped(target, source, alpha, mode=mode, linear_light=True, transfer="srgb")
    edge = (alpha > 0.05) & (alpha < 0.95)
    assert edge.sum() > 20
    # Dark target + bright source: mixing in linear light brightens the transition relative to gamma mixing.
    assert linear[edge][:, 0].astype(int).mean() > gamma[edge][:, 0].astype(int).mean() + 2
    if mode == BlendMode.FEATHER:  # multi-band smooths the mask at its coarsest level, in either domain
        assert np.abs(linear[64, 64].astype(int) - source[64, 64].astype(int)).max() <= 1
    # Beyond the dilated support (<= 3 px of alpha tail + 16 px multi-band margin) nothing may change.
    far = np.zeros_like(alpha, bool)
    far[:20], far[-20:], far[:, :20], far[:, -20:] = True, True, True, True
    assert np.array_equal(linear[far], target[far]) and np.array_equal(gamma[far], target[far])


@pytest.mark.parametrize("linear_light", [False, True])
def test_poisson_applies_linear_light_in_its_final_alpha_mix(monkeypatch, linear_light):
    # Gradient-domain cloning is perceptual and equals the target at the mask border by construction, so with
    # the solver made the identity, Poisson must reduce exactly to the feather mix of the same domain.
    import cv2
    monkeypatch.setattr(cv2, "seamlessClone", lambda src, dst, mask, centre, flags: src)
    target, source, alpha = flat(30, (128, 128)), ramp_source(), square_alpha()
    poisson = fuse_warped(target, source, alpha, mode=BlendMode.POISSON, linear_light=linear_light, transfer="bt1886")
    feather = fuse_warped(target, source, alpha, mode=BlendMode.FEATHER, linear_light=linear_light, transfer="bt1886")
    assert np.array_equal(poisson, feather)


def test_poisson_still_clones_when_the_solver_runs():
    target, source, alpha = flat(30, (128, 128)), ramp_source(), square_alpha()
    result = fuse_warped(target, source, alpha, mode=BlendMode.POISSON, linear_light=True)
    assert result.shape == target.shape and (result != target).any()


def test_default_behaviour_is_unchanged_when_linear_light_is_off():
    target, source = flat(60, (32, 32)), flat(180, (32, 32))
    alpha = square_alpha(32, 8, 24, 5)
    expected = np.clip(np.rint(target * (1 - alpha[..., None]) + source * alpha[..., None]), 0, 255).astype(np.uint8)
    got = fuse_warped(target, source, alpha, mode=BlendMode.FEATHER)
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1


def test_fuse_crop_passes_the_transfer_through():
    crop = flat(255, (16, 16))
    inverse = np.array([[1, 0, 24.5], [0, 1, 24.5]], np.float32)  # half-pixel shift: border pixels get partial coverage
    mask, frame = np.ones((16, 16), np.float32), flat(0, (64, 64))
    # Pointwise transfer ordering is a feather contract, not a Laplacian-band invariant.
    gamma = fuse_crop(frame, crop, inverse, mask, mode=BlendMode.FEATHER)
    srgb = fuse_crop(frame, crop, inverse, mask, mode=BlendMode.FEATHER, linear_light=True, transfer="srgb")
    bt1886 = fuse_crop(frame, crop, inverse, mask, mode=BlendMode.FEATHER, linear_light=True, transfer="bt1886")
    border = (gamma != srgb).any(axis=2)
    assert border.sum() >= 20
    g, s, b = (x[border].astype(int) for x in (gamma, srgb, bt1886))
    assert (s >= g).all() and (b >= s).all() and (b > g).any()      # gamma < sRGB < BT.1886 for partial coverage
    assert np.array_equal(gamma[~border], srgb[~border]) and np.array_equal(srgb[~border], bt1886[~border])


# --- options and auto resolution -----------------------------------------------------------------------------
def test_composite_options_validate_the_transfer_and_default_to_off():
    assert CompositeOptions().linear_light is False and CompositeOptions().blend_transfer == "srgb"
    assert CompositeOptions(linear_light=True, blend_transfer="bt1886").blend_transfer == "bt1886"
    with pytest.raises(ValueError):
        CompositeOptions(blend_transfer="pq")


@pytest.mark.parametrize("code,expected", [(13, "srgb"), (1, "bt1886"), (6, "bt1886"), (14, "bt1886"), (4, "gamma22"), (8, "linear")])
def test_auto_blend_transfer_follows_the_source_transfer(code, expected):
    parameters = WorkstationParameters(linear_light=True)
    assert parameters.blend_transfer == "auto"
    resolved = parameters.with_source_transfer(code)
    assert resolved.blend_transfer == expected
    options = resolved.composite_options()
    assert options.linear_light and options.blend_transfer == expected


def test_explicit_transfer_is_never_overridden_and_unresolved_auto_falls_back_to_the_old_curve():
    explicit = WorkstationParameters(linear_light=True, blend_transfer="gamma28")
    assert explicit.with_source_transfer(1).blend_transfer == "gamma28"
    assert WorkstationParameters(linear_light=True).composite_options().blend_transfer == "srgb"
    assert WorkstationParameters().composite_options().linear_light is False


def test_new_fields_do_not_change_the_fingerprint_of_requests_saved_before_stage15():
    from roop_3d.jobs.schema import JobRequest
    base = dict(target_path="roop_3d/.runtime/media/x.mp4", inspection_id="0" * 32,
                face_mappings=[{"track_id": "track_0001", "faceset_id": "a"}])
    old = JobRequest(**base, workstation_parameters=WorkstationParameters(feather_px=6))
    payload = old.identity_payload()
    assert "linear_light" not in payload["workstation_parameters"] and "blend_transfer" not in payload["workstation_parameters"]
    new = JobRequest(**base, workstation_parameters=WorkstationParameters(feather_px=6, linear_light=True))
    assert new.identity_payload()["workstation_parameters"]["linear_light"] is True
