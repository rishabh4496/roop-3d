"""Data-free albedo prior: robustness, ground-truth extrapolation, refusal without skin, and the completion far-field hook.

A synthetic ellipsoid atlas carries a known smooth skin-colour field (lighter at the brow, darker at the jaw), so each claim is
checked against ground truth. Real-photo evidence is in docs/ALBEDO_PRIOR_STAGE1.md (measured on FaceSets, not asserted here).
"""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import albedo_prior as ap
from roop_3d.reconstruction.uv_completion import complete_texture

SIZE = 96
UNSEEN, INTERP, OBSERVED = 0, 1, 2


def atlas():
    """(positions (H, W, 3), truth BGR uint8 (H, W, 3)) on an ellipsoid; the colour field is smooth in 3D position."""
    v, u = np.mgrid[0:SIZE, 0:SIZE]
    lon, lat = (u + 0.5) / SIZE * 2 * np.pi - np.pi, (v + 0.5) / SIZE * np.pi
    pos = np.stack([0.085 * np.sin(lat) * np.sin(lon), -0.11 * np.cos(lat), 0.095 * np.sin(lat) * np.cos(lon)], axis=-1)
    height = (pos[..., 1] / 0.11 + 1) / 2                      # 0 at the crown, 1 at the chin (+y is down)
    lab = np.stack([76 - 34 * height, 17 + 6 * height + 3 * pos[..., 0] / 0.085, 21 - 6 * height], axis=-1).astype(np.float32)
    return pos, ap.lab_to_bgr(lab)


def whole_footprint():
    return np.ones((SIZE, SIZE), bool)


# ----------------------------------------------------------------------------------------------------- colour + tone
def test_lab_round_trip_is_close():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (200, 3)).astype(np.uint8)
    back = ap.lab_to_bgr(ap.bgr_to_lab(bgr))
    assert np.abs(back.astype(int) - bgr.astype(int)).max() <= 1


def test_tone_estimate_rejects_non_skin_outliers():
    pos, truth = atlas()
    clean = ap.estimate_skin_tone(truth, whole_footprint())
    dirty = truth.copy()
    rng = np.random.default_rng(1)
    bad = rng.random((SIZE, SIZE)) < 0.12
    dirty[bad & (rng.random((SIZE, SIZE)) < 0.5)] = (20, 20, 25)        # hair / shadow
    dirty[bad & (rng.random((SIZE, SIZE)) < 0.5)] = (60, 40, 170)       # lips (saturated red)
    est = ap.estimate_skin_tone(dirty, whole_footprint())
    assert ap.delta_e(est.tone_lab, clean.tone_lab) < 3.0, "a median over skin-like texels is not dragged by hair/lips"
    assert est.in_gamut and est.skin_texels <= est.observed_texels
    assert abs(est.spread_de - clean.spread_de) < 4.0, "outliers do not inflate the reported colour spread"


def test_out_of_gamut_tone_is_flagged_not_clamped():
    washed = np.full((SIZE, SIZE, 3), (232, 236, 255), np.uint8)
    est = ap.estimate_skin_tone(washed, whole_footprint())
    assert est.in_gamut is False and any("colour cast" in n for n in est.notes)
    assert np.array_equal(est.tone_bgr, ap.lab_to_bgr(est.tone_lab[None])[0]), "reported as measured"


@pytest.mark.parametrize("texture", [np.full((SIZE, SIZE, 3), (200, 40, 40), np.uint8),       # blue wall
                                     np.full((SIZE, SIZE, 3), (30, 200, 30), np.uint8)])       # green
def test_no_skin_means_no_prior(texture):
    with pytest.raises(DomainError) as error:
        ap.estimate_skin_tone(texture, whole_footprint())
    assert error.value.code == "ALBEDO_PRIOR_NO_SKIN" and error.value.status == 422
    pos, _ = atlas()
    state = np.full((SIZE, SIZE), OBSERVED, np.uint8)
    with pytest.raises(DomainError):
        ap.fill_unseen_with_prior(texture, state, whole_footprint(), pos)


def test_nothing_observed_is_refused_rather_than_invented():
    pos, truth = atlas()
    with pytest.raises(DomainError) as error:
        ap.fill_unseen_with_prior(truth, np.zeros((SIZE, SIZE), np.uint8), whole_footprint(), pos)
    assert error.value.code == "ALBEDO_PRIOR_NO_SKIN"


# ----------------------------------------------------------------------------------------------------- the surface fit
def hole_state(radius=0.03, row=0.5):
    pos, truth = atlas()
    centre = pos[int(SIZE * row), SIZE // 2 + 5]
    hole = np.linalg.norm(pos - centre, axis=-1) < radius
    state = np.full((SIZE, SIZE), OBSERVED, np.uint8)
    state[hole] = UNSEEN
    return pos, truth, state, hole


def test_prior_reconstructs_a_hidden_patch_far_better_than_a_flat_tone():
    pos, truth, state, hole = hole_state(0.035, row=0.8)          # near the chin: far from the atlas-wide median tone
    fill = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos, feather_px=0)
    truth_lab = ap.bgr_to_lab(truth[hole])
    err_prior = ap.delta_e(ap.bgr_to_lab(fill.texture[hole]), truth_lab).mean()
    flat = ap.estimate_skin_tone(truth, state == OBSERVED).tone_lab
    err_flat = ap.delta_e(flat[None, :], truth_lab).mean()
    assert fill.filled.sum() == hole.sum()
    assert err_flat > 4.0, "the test field must be steep enough that a flat tone is clearly wrong"
    assert err_prior < 3.0, f"smooth ground truth must be recovered closely (got dE {err_prior:.2f})"
    assert err_prior < 0.4 * err_flat


def test_fit_reports_its_own_quality():
    pos, truth, state, _ = hole_state()
    fill = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos)
    fit = fill.report["fit"]
    assert fit["residual_de_median"] < 2.0 and fit["inlier_fraction"] > 0.9
    assert fill.report["third_party_data"] is False and fill.report["method"] == "data_free_surface_prior"
    assert "no lip, brow" in fill.report["limits"] and fill.report["skin_tone"]["in_gamut"] is True
    assert fill.report["unseen_texels"] == int((state == UNSEEN).sum())


def test_outliers_in_the_observations_do_not_bend_the_field():
    pos, truth, state, hole = hole_state(0.035)
    noisy = truth.copy()
    rng = np.random.default_rng(3)
    bad = (rng.random((SIZE, SIZE)) < 0.1) & ~hole
    noisy[bad] = (25, 25, 30)
    clean_fill = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos, feather_px=0).texture[hole]
    noisy_fill = ap.fill_unseen_with_prior(noisy, state, whole_footprint(), pos, feather_px=0).texture[hole]
    assert ap.delta_e(ap.bgr_to_lab(clean_fill), ap.bgr_to_lab(noisy_fill)).mean() < 3.0


def test_prediction_relaxes_to_the_skin_tone_far_from_data():
    pos, truth = atlas()
    state = np.full((SIZE, SIZE), UNSEEN, np.uint8)
    state[: SIZE // 3] = OBSERVED                                   # only the crown was seen
    prior = ap.SurfaceAlbedoPrior.fit(pos[state == OBSERVED], truth[state == OBSERVED])
    far = pos[-1, SIZE // 2][None, :]
    assert prior.distance_to_observed_mm(far)[0] > 80
    seen = ap.bgr_to_lab(truth[state == OBSERVED])
    envelope = ap.delta_e(seen, prior.tone.tone_lab).max()
    assert ap.delta_e(prior.predict_lab(far)[0], prior.tone.tone_lab) <= 1.5 * envelope, "no runaway extrapolation: stays near the observed colour range"
    assert ap.delta_e(prior.predict_lab(far)[0], ap.bgr_to_lab(truth[-1, SIZE // 2])) > 15, "and it does not pretend to know the unseen chin"


def test_fit_is_deterministic():
    pos, truth, state, _ = hole_state()
    a = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos).texture
    b = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos).texture
    assert np.array_equal(a, b)


# -------------------------------------------------------------------------------------------------- what is (not) written
def test_only_unseen_footprint_texels_are_written():
    pos, truth, state, hole = hole_state()
    state[:4] = INTERP                                              # interpolated texels are not the prior's to touch
    footprint = whole_footprint()
    footprint[:, -10:] = False                                      # outside the atlas footprint
    garbage = truth.copy()
    garbage[hole] = (0, 0, 0)
    fill = ap.fill_unseen_with_prior(garbage, state, footprint, pos, feather_px=3)
    untouched = ~(hole & footprint)
    assert np.array_equal(fill.texture[untouched], garbage[untouched])
    assert (fill.filled == ((state == UNSEEN) & footprint)).all()
    assert not np.array_equal(fill.texture[hole & footprint], garbage[hole & footprint])


def test_feathering_blends_into_the_neighbouring_observation():
    pos, truth, state, hole = hole_state(0.04)
    hard = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos, feather_px=0).texture
    soft = ap.fill_unseen_with_prior(truth, state, whole_footprint(), pos, feather_px=6).texture
    edge = hole & (np.asarray(__import__("cv2").erode(hole.astype(np.uint8), np.ones((3, 3), np.uint8))) == 0)
    assert edge.any() and not np.array_equal(hard[edge], soft[edge])
    assert np.array_equal(hard[~hole], soft[~hole])


def test_prior_field_covers_the_footprint_only():
    pos, truth, state, _ = hole_state()
    footprint = whole_footprint()
    footprint[:10] = False
    field, prior = ap.prior_field(truth, state, footprint, pos)
    assert field.shape == (SIZE, SIZE, 3) and field.dtype == np.float32
    assert (field[~footprint] == 0).all() and (field[footprint] > 0).any() and prior.report["centres"] > 0


# -------------------------------------------------------------------------------------------- completion far-field hook
def test_far_field_none_leaves_completion_unchanged():
    pos, truth, state, hole = hole_state(0.04)
    observed = state == OBSERVED
    a = complete_texture(truth, observed, whole_footprint(), pos, mirror_x=0.0)
    b = complete_texture(truth, observed, whole_footprint(), pos, mirror_x=0.0, far_field=None)
    assert np.array_equal(a.texture, b.texture) and np.array_equal(a.source, b.source)


def test_far_field_replaces_the_flat_tone_far_from_texture():
    pos, truth = atlas()
    observed = np.zeros((SIZE, SIZE), bool)
    observed[:20] = True
    marker = np.zeros((SIZE, SIZE, 3), np.float32)
    marker[...] = (30.0, 90.0, 200.0)                                # an unmistakable far-field colour
    plain = complete_texture(truth, observed, whole_footprint(), pos, mirror_x=1e3, far_field=None, tone_decay_px=5.0)
    steered = complete_texture(truth, observed, whole_footprint(), pos, mirror_x=1e3, far_field=marker, tone_decay_px=5.0)
    far = np.zeros((SIZE, SIZE), bool)
    far[-15:] = True
    assert np.abs(steered.texture[..., :3][far].astype(float) - marker[far]).mean() < 6.0
    assert np.abs(plain.texture[..., :3][far].astype(float) - marker[far]).mean() > 40.0
    assert np.array_equal(steered.texture[:10], plain.texture[:10]), "texels with data nearby are unaffected"
