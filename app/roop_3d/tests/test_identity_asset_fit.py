"""Stage 3 Phase B: multi-view landmark geometry. Assertions compare against a KNOWN synthetic ground truth,
which the code under test never sees: the recovered geometry must follow the observations, not a generic head."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.reconstruction import multiview_fit as mv  # noqa: E402
from roop_3d.reconstruction.multiview_fit import (  # noqa: E402
    LAYOUT_5,
    LAYOUT_68,
    ViewLandmarks,
    batch_rotations,
    deform_template,
    fit_identity_landmarks,
    identity_template,
)


def _observations(subject, yaws, *, pitches=None, noise=0.7, pose_noise=2.5, expressions=None, seed=0):
    rng = np.random.default_rng(seed)
    pitches = pitches or [0.0] * len(yaws)
    expressions = expressions or [None] * len(yaws)
    out = []
    for i, (yaw, pitch, expr) in enumerate(zip(yaws, pitches, expressions)):
        pts = W.project_landmarks(subject, yaw, pitch, 0.0, noise_px=noise, seed=seed + i, expression=expr)
        out.append(ViewLandmarks(i, pts, (yaw + rng.normal(0, pose_noise), pitch + rng.normal(0, pose_noise), 0.0), 0.9))
    return out


def _err_mm(estimate, truth):
    return float(np.linalg.norm(W.sim_align(estimate, truth) - truth, axis=1).mean() * 1000)


@pytest.fixture(scope="module", params=[LAYOUT_5, LAYOUT_68])
def subject(request):
    return W.make_subject("fit", request.param, 21)


def test_multiview_recovers_the_actual_identity_not_the_template(subject):
    prior_err = _err_mm(subject.template_lm, subject.true_lm)
    fit = fit_identity_landmarks(_observations(subject, (0, -40, 40, -72, 72)))
    assert fit.layout == subject.layout and fit.converged
    assert _err_mm(fit.landmarks_3d, subject.true_lm) < 0.35 * prior_err
    assert fit.rms_mm < 1.5                                   # in-sample reprojection
    assert fit.holdout_rms_mm is not None and fit.holdout_rms_mm < 2.0   # leave-one-view-out: true multi-view consistency
    assert all(v.consistent for v in fit.views)


def test_front_only_cannot_triangulate_and_says_so(subject):
    fit = fit_identity_landmarks(_observations(subject, (0,)))
    assert fit.level == "single_view_prior"
    assert fit.depth_observed_fraction == 0.0 and float(fit.confidence_z.max()) < 0.15
    z_shift = np.abs(fit.landmarks_3d[:, 2] - fit.template_landmarks[:, 2]).max()
    assert z_shift < 0.003, "depth must stay at the prior when no view constrains it"
    assert any("cannot be triangulated" in n for n in fit.notes)
    # a near-frontal pair does not triangulate depth either
    near = fit_identity_landmarks(_observations(subject, (0, 6, -6)))
    assert near.level == "single_view_prior"


def test_more_views_monotonically_increase_depth_confidence(subject):
    conf = [float(fit_identity_landmarks(_observations(subject, yaws), holdout=False).confidence_z.mean())
            for yaws in ((0,), (0, 40), (0, -40, 40), (0, -40, 40, -72, 72))]
    assert conf[0] < 0.05 < conf[1] < conf[3] + 1e-6
    assert conf[3] >= conf[2] - 0.02


def test_reconstruction_follows_each_identity():
    a = W.make_subject("idA", LAYOUT_68, 31, width=0.16, nose_depth=0.020, jaw_drop=0.014)
    b = W.make_subject("idB", LAYOUT_68, 32, width=-0.08, nose_depth=-0.006, jaw_drop=-0.006)
    fit_a = fit_identity_landmarks(_observations(a, (0, -40, 40)), holdout=False)
    fit_b = fit_identity_landmarks(_observations(b, (0, -40, 40)), holdout=False)
    for fit, own, foreign in ((fit_a, a, b), (fit_b, b, a)):
        assert _err_mm(fit.landmarks_3d, own.true_lm) < 0.4 * _err_mm(fit.landmarks_3d, foreign.true_lm)
        assert _err_mm(fit.landmarks_3d, own.true_lm) < 0.4 * _err_mm(own.template_lm, own.true_lm)
    # absolute size is a prior (not observable without a reference length); the aspect ratio is observable
    aspect = lambda lm: float(np.ptp(lm[:, 0]) / np.ptp(lm[:, 1]))
    for fit, own in ((fit_a, a), (fit_b, b)):
        assert aspect(fit.landmarks_3d) == pytest.approx(aspect(own.true_lm), rel=0.02)
        assert abs(aspect(fit.landmarks_3d) - aspect(own.true_lm)) < 0.4 * abs(aspect(own.template_lm) - aspect(own.true_lm)) + 1e-9
    assert aspect(fit_a.landmarks_3d) > aspect(fit_b.landmarks_3d) + 0.05      # the wider face is reconstructed wider


def test_expression_variation_does_not_corrupt_identity():
    subj = W.make_subject("expr", LAYOUT_68, 41)
    expressions = [None, None, None, "open", None]
    neutral = fit_identity_landmarks(_observations(subj, (0, -40, 40, 10, -72)), holdout=False)
    varied = fit_identity_landmarks(_observations(subj, (0, -40, 40, 10, -72), expressions=expressions), holdout=False)
    assert _err_mm(varied.landmarks_3d, subj.true_lm) < 3.0 * max(_err_mm(neutral.landmarks_3d, subj.true_lm), 0.3)
    assert _err_mm(varied.landmarks_3d, subj.true_lm) < 0.7 * _err_mm(subj.template_lm, subj.true_lm)


def test_gross_outlier_view_is_detected_and_dropping_it_restores_the_fit(subject):
    """Detection + drop-and-refit is the pipeline's policy (identity_asset.build_identity_asset does the same)."""
    obs = _observations(subject, (0, -40, 40, -72, 72))
    bad = obs[1]
    rng = np.random.default_rng(5)
    corrupted = ViewLandmarks(bad.view_index, bad.points + rng.normal(0, 28.0, bad.points.shape), bad.pose_deg, bad.quality)
    fit = fit_identity_landmarks([obs[0], corrupted, *obs[2:]])
    flagged = [v.view_index for v in fit.views if not v.consistent]
    assert flagged == [1], "only the corrupted view may be flagged"
    refit = fit_identity_landmarks([o for o in obs if o.view_index not in flagged])
    assert all(v.consistent for v in refit.views)
    assert _err_mm(refit.landmarks_3d, subject.true_lm) < 0.4 * _err_mm(subject.template_lm, subject.true_lm)


def test_clean_views_are_never_flagged(subject):
    fit = fit_identity_landmarks(_observations(subject, (0, -40, 40, -72, 72), noise=1.5, pose_noise=4.0, seed=9))
    assert all(v.consistent for v in fit.views)


def test_noisy_stored_pose_is_refined(subject):
    obs = _observations(subject, (0, -40, 40, -72, 72), pose_noise=5.0, seed=3)
    fit = fit_identity_landmarks(obs, holdout=False)
    truth = {i: y for i, y in enumerate((0, -40, 40, -72, 72))}
    refined = np.mean([abs(v.yaw - truth[v.view_index]) for v in fit.views])
    stored = np.mean([abs(o.pose_deg[0] - truth[o.view_index]) for o in obs])
    assert refined < stored


def test_input_validation():
    subj = W.make_subject("val", LAYOUT_5, 51)
    good = _observations(subj, (0, 30))
    with pytest.raises(DomainError) as info:
        fit_identity_landmarks([])
    assert info.value.code == "NO_OBSERVATIONS"
    with pytest.raises(DomainError) as info:
        fit_identity_landmarks([ViewLandmarks(0, np.zeros((106, 2)), (0, 0, 0))])   # 106-point layouts are not trusted
    assert info.value.code == "UNSUPPORTED_LANDMARK_LAYOUT"
    dense = W.make_subject("val68", LAYOUT_68, 52)
    with pytest.raises(DomainError) as info:
        fit_identity_landmarks([good[0], _observations(dense, (0,))[0]])
    assert info.value.code == "MIXED_LANDMARK_LAYOUTS"
    bad = ViewLandmarks(0, np.full((5, 2), np.nan), (0, 0, 0))
    with pytest.raises(DomainError) as info:
        fit_identity_landmarks([bad])
    assert info.value.code == "NON_FINITE_LANDMARKS"


def test_batch_rotations_match_core_pose_and_jacobian_is_exact():
    from roop_3d.core.pose import rotation_from_euler
    angles = np.array([[0, 0, 0], [30, -10, 5], [-70, 15, -20.0]])
    for row, rot in zip(angles, batch_rotations(angles)):
        assert np.allclose(rot, rotation_from_euler(*row), atol=1e-12)
    template = identity_template(1)
    lm = mv.template_landmarks(LAYOUT_5, template)
    normals = np.tile([0.0, 0.0, 1.0], (5, 1))                  # fully visible: weights are constants
    rng = np.random.default_rng(0)
    problem = mv._Problem(lm, normals, rng.normal(300, 40, (3, 5, 2)), np.array([0.9, 0.8, 1.0]),
                          np.array([[0, 0, 0], [5, 2, 2], [-4, 0, 0.0]]))
    x = problem.initial()
    analytic = problem.jacobian(x)
    numeric = mv._numeric_jacobian(problem.residual, x, problem.steps(), problem.residual(x))
    assert np.abs(analytic - numeric).max() / np.abs(numeric).max() < 1e-4


def test_template_deformation_is_local_and_never_folds_triangles():
    template = identity_template(2)
    lm = mv.template_landmarks(LAYOUT_5, template)
    target = lm.copy()
    target[2, 2] += 0.02                                       # push the nose tip forward 20 mm
    verts, support, strength = deform_template(template, lm, target, np.ones(5))
    moved = np.linalg.norm(verts - template.vertices, axis=1)
    nose = int(np.argmin(np.linalg.norm(template.vertices - lm[2], axis=1)))
    assert moved[nose] > 0.01
    far = np.linalg.norm(template.vertices[:, None] - lm[None], axis=2).min(axis=1) > 0.08   # far from EVERY control
    assert far.any() and moved[far].max() < 0.1 * moved[nose] and support[far].max() < 0.25   # Gaussian taper, not extrapolation
    folded = identity_template(2).compute_face_normals()[:, 2] > 0.3
    from roop_3d.reconstruction.geometry_types import Mesh3D
    new_n = Mesh3D(verts, template.faces).compute_face_normals()
    assert (new_n[folded, 2] > 0.0).all() and 0.0 < strength <= 1.0
    # zero confidence => the template is returned unchanged
    same, _, _ = deform_template(template, lm, target, np.zeros(5))
    assert np.abs(same - template.vertices).max() < 1e-6
