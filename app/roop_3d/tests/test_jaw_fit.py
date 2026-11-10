"""Opt-in jaw-width fit of the canonical proxy head in the mesh pose-blend path (``PoseBlendProjector``, ``ReferencePipeline``).

Ground truth is synthetic and exact: a target outline made from the proxy's own jaw with a KNOWN width ratio. What that does
and does not show: the measurement and the geometry follow the ratio to within a few percent, the five PnP anchors never move,
and the default path is untouched. It does not show that the detector's 106 contour and the iBUG jaw agree on a real face; that
needs measured footage and is why the fit has a dead band, a clamp and a pose gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.reconstruction import jaw_fit
from roop_3d.reconstruction.jaw_fit import DEAD_BAND, JawFitReport, ProxyJawAdapter, decide, measure_jaw_ratio
from roop_3d.reconstruction.pose_blend import PoseBlendProjector
from roop_3d.rendering.swap_region_3d import SwapRegionConfig, adapt_jaw, jaw_target_from_ratios

SIDE = 256
OBSERVED = ARCFACE_REFERENCE_5PTS.astype(np.float64) * (SIDE / 112.0)
ANCHOR_VERTICES = [*range(36, 48), 30, 48, 54]


def u_shape(n: int, half_width: float, depth: float, cx: float = 128.0, top: float = 100.0) -> np.ndarray:
    """A jaw-like polyline from one ear down to the chin and up to the other, ``n`` points."""
    t = np.linspace(-np.pi / 2, np.pi / 2, n)
    return np.column_stack((cx + half_width * np.sin(t), top + depth * np.cos(t)))      # ears at the top, chin lowest


def resample(poly: np.ndarray, n: int) -> np.ndarray:
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    s = np.linspace(0, arc[-1], n)
    return np.column_stack([np.interp(s, arc, poly[:, k]) for k in range(2)])


def widened(poly: np.ndarray, ratio: float) -> np.ndarray:
    mid = 0.5 * (poly[:, 0].min() + poly[:, 0].max())
    return np.column_stack((mid + (poly[:, 0] - mid) * ratio, poly[:, 1]))


@pytest.fixture(scope="module")
def projector():
    return PoseBlendProjector()


@pytest.fixture(scope="module")
def crop():
    return np.random.default_rng(0).integers(60, 200, (SIDE, SIDE, 3)).astype(np.uint8)


@pytest.fixture(scope="module")
def baseline(projector, crop):
    return projector.render(crop, OBSERVED, 1.0)


def target_contour(baseline, ratio: float) -> np.ndarray:
    """The target's 33-point outline: the jaw the proxy RENDERS (mesh projection), resampled, with a known width ratio."""
    return widened(resample(baseline.mesh_pixels[:17], 33), ratio)


# =========================================================================== measurement
@pytest.mark.parametrize("ratio", [0.88, 1.0, 1.1, 1.17])
def test_measurement_recovers_a_known_width_ratio(ratio):
    proxy = u_shape(17, 70.0, 90.0)
    contour = widened(resample(proxy, 33), ratio)
    measured, levels, reason = measure_jaw_ratio(proxy, contour)
    assert reason == "ok" and levels >= 6 and measured == pytest.approx(ratio, abs=0.01)


def test_measurement_is_invariant_to_point_order_and_translation_and_tolerates_noise():
    proxy = u_shape(17, 70.0, 90.0)
    contour = widened(resample(proxy, 33), 1.12)
    ref, _, _ = measure_jaw_ratio(proxy, contour)
    assert measure_jaw_ratio(proxy, contour[::-1])[0] == pytest.approx(ref, abs=1e-6)                 # either direction
    assert measure_jaw_ratio(proxy + (31.0, -17.0), contour + (31.0, -17.0))[0] == pytest.approx(ref, abs=1e-6)
    noisy = contour + np.random.default_rng(1).normal(0, 0.6, contour.shape)
    assert measure_jaw_ratio(proxy, noisy)[0] == pytest.approx(ref, abs=0.03)


def test_measurement_refuses_what_it_cannot_measure():
    proxy = u_shape(17, 70.0, 90.0)
    good = resample(proxy, 33)
    for proxy_bad, contour_bad, reason in (
            (proxy[:16], good, "contour_invalid"), (proxy, good[:30], "contour_invalid"),
            (proxy, np.full((33, 2), np.nan), "contour_invalid"), (proxy, np.zeros((33, 2)), "contour_invalid"),
            (proxy, good + np.array([0.0, 400.0]), "too_few_levels")):          # an outline far below the lower face
        ratio, levels, why = measure_jaw_ratio(proxy_bad, contour_bad)
        assert ratio is None and why == reason


def test_decision_applies_the_pose_gate_dead_band_and_clamp():
    assert decide(1.10, 8, "ok", 0.95)[0] == pytest.approx(1.10)
    ratio, report = decide(1.10, 8, "ok", 0.6)                       # ~53 degrees: one side is not the jaw outline
    assert ratio is None and report.reason == "pose_too_oblique" and not report.applied
    ratio, report = decide(1.0 + DEAD_BAND * 0.9, 8, "ok", 0.95)     # inside the dead band: leave the proxy alone
    assert ratio is None and report.reason == "within_dead_band" and report.ratio_used == 1.0
    assert decide(1.6, 8, "ok", 0.95)[0] == pytest.approx(jaw_fit.RATIO_LIMITS[1])        # clamped, never trusted blindly
    assert decide(0.5, 8, "ok", 0.95)[0] == pytest.approx(jaw_fit.RATIO_LIMITS[0])
    ratio, report = decide(None, 0, "contour_invalid", 0.95)
    assert ratio is None and report.reason == "contour_invalid" and report.measured_ratio is None


# =========================================================================== proxy geometry
def test_adapted_proxy_follows_the_ratio_never_folds_and_never_moves_the_pnp_anchors(projector):
    adapter = ProxyJawAdapter(projector._proxy_vertices, projector._proxy_faces)
    base = projector._proxy_vertices
    for ratio in (0.86, 0.95, 1.05, 1.12, 1.18):
        vertices, folded = adapter.adapted(ratio)
        assert folded == 0
        assert np.ptp(vertices[:17, 0]) == pytest.approx(np.ptp(base[:17, 0]) * ratio, rel=0.011)
        shift = np.linalg.norm(vertices - base, axis=1)
        assert shift[ANCHOR_VERTICES].max() == 0 and shift[17:68].max() == 0        # the whole inner face is held still
        assert shift[:17].max() <= 0.02
    assert adapter.adapted(1.101)[0] is adapter.adapted(1.10)[0]                    # cached per 1 %


def test_pinned_vertices_hold_still_where_an_unpinned_field_would_move_them(projector):
    v, f = projector._proxy_vertices, projector._proxy_faces
    config = SwapRegionConfig(include_jawline=True)
    target = jaw_target_from_ratios(v, np.arange(68), width_ratio=1.15, config=config)
    free = adapt_jaw(v, f, np.arange(68), target, config)
    pinned = adapt_jaw(v, f, np.arange(68), target, config, pinned=range(17, 68))
    assert np.linalg.norm(free.field[ANCHOR_VERTICES], axis=1).max() > 1e-3          # the coarse mesh leaks into the eyes ...
    assert np.linalg.norm(pinned.field[17:68], axis=1).max() == 0                    # ... unless they are pinned
    assert pinned.report.jaw_width_after_m == pytest.approx(pinned.report.jaw_width_target_m, rel=0.011)
    with pytest.raises(DomainError):
        adapt_jaw(v, f, np.arange(68), target, config, pinned=[999])
    with pytest.raises(DomainError):
        adapt_jaw(v, f, np.arange(68), target, config, pinned=[3])                   # a handle cannot also be pinned


# =========================================================================== projector
def extent(coverage: np.ndarray, rows: slice) -> float:
    widths = [np.ptp(np.nonzero(r > 0.5)[0]) for r in coverage[rows] if (r > 0.5).any()]
    return float(np.mean(widths))


def test_default_render_is_unchanged_and_carries_no_jaw_report(projector, crop, baseline):
    assert baseline.jaw is None
    again = projector.render(crop, OBSERVED, 1.0)
    np.testing.assert_array_equal(again.image, baseline.image)
    np.testing.assert_array_equal(again.coverage, baseline.coverage)


@pytest.mark.parametrize("ratio", [0.88, 1.12])
def test_footprint_follows_the_targets_jaw_width(projector, crop, baseline, ratio):
    contour = target_contour(baseline, ratio)
    result = projector.render(crop, OBSERVED, 1.0, jaw_contour=contour)
    assert isinstance(result.jaw, JawFitReport) and result.jaw.applied and result.jaw.reason == "adapted"
    assert result.jaw.measured_ratio == pytest.approx(ratio, abs=0.03) and result.jaw.folded_triangles == 0
    jaw_y = baseline.geometry.flat_pixels[:17, 1]
    rows = slice(int(jaw_y.min() + 0.45 * np.ptp(jaw_y)), int(jaw_y.min() + 0.75 * np.ptp(jaw_y)))      # mid lower face
    before, after = extent(baseline.coverage, rows), extent(result.coverage, rows)
    assert after / before == pytest.approx(ratio, abs=0.05)                                          # the footprint moved with the jaw
    assert abs(after - extent(_fill(contour), rows)) < 0.06 * before                                  # ... onto the target outline
    # the camera fit is untouched: the projected anchors are exactly where the unadapted proxy puts them
    np.testing.assert_allclose(result.mesh_pixels[ANCHOR_VERTICES], baseline.mesh_pixels[ANCHOR_VERTICES], atol=1e-3)
    assert result.reprojection_error == pytest.approx(baseline.reprojection_error)


def _fill(contour: np.ndarray) -> np.ndarray:
    """Coverage image (H, W) of the filled target outline, as the reference for the footprint."""
    mask = np.zeros((SIDE, SIDE), np.float32)
    cv2.fillPoly(mask, [np.round(contour).astype(np.int32)], 1.0)
    return mask


def test_blended_weights_use_the_adapted_geometry_too(projector, crop, baseline):
    contour = target_contour(baseline, 1.12)
    half = projector.render(crop, OBSERVED, 0.5, jaw_contour=contour)
    plain = projector.render(crop, OBSERVED, 0.5)
    assert half.jaw.applied and plain.jaw is None
    assert (half.coverage > 0.5).sum() > (plain.coverage > 0.5).sum()               # a wider jaw covers more, at any blend weight


def test_no_change_when_the_jaw_already_matches_or_the_fit_cannot_be_trusted(projector, crop, baseline):
    same = projector.render(crop, OBSERVED, 1.0, jaw_contour=target_contour(baseline, 1.0))
    assert not same.jaw.applied and same.jaw.reason == "within_dead_band"
    np.testing.assert_array_equal(same.image, baseline.image)                         # bit-identical: nothing was touched
    broken = projector.render(crop, OBSERVED, 1.0, jaw_contour=np.full((33, 2), np.nan))
    assert broken.jaw.reason == "contour_invalid" and np.array_equal(broken.coverage, baseline.coverage)


def test_oblique_faces_are_left_alone_and_say_why(projector, crop, baseline):
    rvec = np.array([0.0, np.radians(55.0), 0.0])
    camera = np.array(((1.5 * SIDE, 0, SIDE / 2), (0, 1.5 * SIDE, SIDE / 2), (0, 0, 1)))
    anchors, _ = cv2.projectPoints(projector.anchors.astype(np.float64), rvec, np.array([0.0, 0.0, 0.55]), camera, None)
    observed = anchors.reshape(-1, 2)
    contour = target_contour(baseline, 1.12)
    result = projector.render(crop, observed, 1.0, jaw_contour=contour)
    assert not result.jaw.applied and result.jaw.reason == "pose_too_oblique"
    np.testing.assert_array_equal(result.image, projector.render(crop, observed, 1.0).image)


def test_a_fold_or_failure_in_the_adaptation_falls_back_to_the_generic_proxy(projector, crop, baseline, monkeypatch):
    contour = target_contour(baseline, 1.12)
    adapter = ProxyJawAdapter(projector._proxy_vertices, projector._proxy_faces)
    monkeypatch.setattr(adapter, "adapted", lambda ratio: (projector._proxy_vertices.copy(), 3))
    monkeypatch.setattr(projector, "_jaw_adapter", adapter)
    result = projector.render(crop, OBSERVED, 1.0, jaw_contour=contour)
    assert not result.jaw.applied and result.jaw.reason == "adaptation_failed" and result.jaw.folded_triangles == 3
    np.testing.assert_array_equal(result.image, baseline.image)


# =========================================================================== pipeline + wire format
def plausible_106(landmarks5: np.ndarray, contour33: np.ndarray) -> np.ndarray:
    """A 106-point layout (contour 0-32, eye 33-42, brow 43-51, mouth 52-71, nose 72-86, eye 87-96, brow 97-105) around 5 points."""
    eye_a, eye_b, nose, mouth_a, mouth_b = landmarks5
    unit = float(np.linalg.norm(eye_b - eye_a))
    ring = lambda c, n, rx, ry: np.column_stack((c[0] + rx * np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
                                                 c[1] + ry * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))))
    arc = lambda c, n: np.column_stack((c[0] + np.linspace(-0.25, 0.25, n) * unit, np.full(n, c[1] - 0.4 * unit)))
    mouth_c = 0.5 * (mouth_a + mouth_b)
    nose_pts = np.column_stack((np.full(15, nose[0]), np.linspace(eye_a[1] + 0.3 * unit, eye_a[1] + 0.7 * unit, 15)))
    return np.vstack([contour33, ring(eye_a, 10, 0.2 * unit, 0.08 * unit), arc(eye_a, 9),
                      ring(mouth_c, 20, 0.4 * unit, 0.1 * unit), nose_pts, ring(eye_b, 10, 0.2 * unit, 0.08 * unit), arc(eye_b, 9)])


def make_session(landmarks106_contour: np.ndarray | None, *, landmarks106: np.ndarray | None = None):
    from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
    from roop_3d.engine.volumetric.session import ReferencePipeline
    marks106 = (landmarks106 if landmarks106 is not None else np.zeros((106, 2)) if landmarks106_contour is None
                else plausible_106(OBSERVED, landmarks106_contour))
    face = FaceObservation(np.array([20.0, 20.0, 236.0, 236.0]), OBSERVED.copy(), marks106, np.ones(512), (0.0, 0.0, 0.0))
    tracked = TrackedFace(0, face, 1, 0)
    session = ReferencePipeline()
    session.tracker = SimpleNamespace(process=lambda *a, **k: (tracked,), active_ids={0}, scene_cut=False,
                                      timestamp_seconds=0.0, release=lambda: None)
    session.prepared = True
    swapped = np.random.default_rng(0).integers(60, 200, (SIDE, SIDE, 3)).astype(np.uint8)
    session.swapper.infer_frame = lambda *a, **k: SimpleNamespace(
        image=swapped, alignment=SimpleNamespace(inverse=np.array([[1.0, 0, 0], [0, 1.0, 0]])), reference_index=0)
    return session


def test_pipeline_fits_the_jaw_only_when_asked_and_reports_the_outcome(baseline):
    frame = np.zeros((SIDE, SIDE, 3), np.uint8)
    session = make_session(target_contour(baseline, 1.12))
    try:
        plain, = session.infer_frame(frame, pose_blend=1.0).replacements
        assert plain.projection.jaw is None and plain.method == "identity_swap_tps_mesh_blend"      # default: untouched
        asked, = session.infer_frame(frame, pose_blend=1.0, jaw_adaptation=True).replacements
        assert asked.projection.jaw.applied and asked.projection.jaw.measured_ratio == pytest.approx(1.12, abs=0.04)
        assert (asked.coverage > 0.5).sum() > (plain.coverage > 0.5).sum()
        ignored, = session.infer_frame(frame, jaw_adaptation=True, strength=0.0).replacements        # no pose_blend: no mesh path
        assert ignored.projection is None
        assert session.capability()["pose_projection"]["jaw_adaptation"].startswith("OPT_IN")
    finally:
        session.release_session()


def test_pipeline_reports_an_untrustworthy_106_layout_instead_of_silently_skipping():
    session = make_session(None)                                    # all-zero 106 points: not a face layout
    try:
        item, = session.infer_frame(np.zeros((SIDE, SIDE, 3), np.uint8), pose_blend=1.0, jaw_adaptation=True).replacements
        assert not item.projection.jaw.applied and item.projection.jaw.reason == "contour_invalid"
    finally:
        session.release_session()


def test_wire_format_is_opt_in_and_reaches_the_inference_options():
    assert "jaw_adaptation" not in WorkstationParameters().model_dump()                # earlier requests keep their fingerprint
    assert "jaw_adaptation" not in WorkstationParameters().inference_options()
    on = WorkstationParameters(jaw_adaptation=True, pose_mode="mesh_blend")
    assert on.model_dump()["jaw_adaptation"] is True and on.inference_options()["jaw_adaptation"] is True


def test_texture_coordinates_follow_the_moved_outline_and_nothing_else(projector, crop, baseline):
    """The moved jaw vertices take the reference-space coordinates of their NEW canonical position; every other UV is untouched,
    so the crop is not compressed into a wider or narrower outline and the inner face renders exactly as before."""
    adapted = projector.render(crop, OBSERVED, 1.0, jaw_contour=target_contour(baseline, 1.12))
    new_uv, old_uv = adapted.geometry.uv, baseline.geometry.uv
    assert adapted.jaw.applied
    np.testing.assert_array_equal(new_uv[17:68], old_uv[17:68])                           # brows, nose, eyes, mouth: identical
    assert np.ptp(new_uv[:17, 0]) / np.ptp(old_uv[:17, 0]) == pytest.approx(1.12, rel=0.03)  # the jaw's UVs widened with it


def test_flat_and_blended_weights_use_the_adapted_outline_too(projector, crop, baseline):
    contour = target_contour(baseline, 1.12)
    for weight in (0.0, 0.5):                                                              # 0 = flat TPS path, 0.5 = morph
        adapted = projector.render(crop, OBSERVED, weight, jaw_contour=contour)
        plain = projector.render(crop, OBSERVED, weight)
        jaw_y = plain.geometry.flat_pixels[:17, 1]
        rows = slice(int(jaw_y.min() + 0.45 * np.ptp(jaw_y)), int(jaw_y.min() + 0.75 * np.ptp(jaw_y)))
        assert extent(adapted.coverage, rows) > 1.05 * extent(plain.coverage, rows)


def test_the_measurement_heights_are_the_lower_face(baseline):
    """A taper (12 % wider under the ears, equal at the chin) makes the answer depend on WHICH heights are compared."""
    proxy = u_shape(17, 70.0, 90.0)
    outline = resample(proxy, 33)
    top, chin = 0.5 * (proxy[0, 1] + proxy[-1, 1]), proxy[:, 1].max()
    fraction = np.clip((outline[:, 1] - top) / (chin - top), 0, 1)
    mid = outline[:, 0].mean()
    tapered = np.column_stack((mid + (outline[:, 0] - mid) * (1.2 - 0.2 * fraction), outline[:, 1]))     # 1.2 at the ears -> 1.0 at the chin
    measured, _, _ = measure_jaw_ratio(proxy, tapered)
    assert measured == pytest.approx(1.2 - 0.2 * 0.525, abs=0.02)                    # the median over 20-85 % of the way down


def test_a_scrambled_106_layout_is_not_trusted_even_with_a_usable_looking_contour(baseline):
    scrambled = plausible_106(OBSERVED, target_contour(baseline, 1.12))
    scrambled[[*range(43, 52), *range(52, 61)]] = scrambled[[*range(52, 61), *range(43, 52)]]       # brow block <-> mouth block
    session = make_session(None, landmarks106=scrambled)
    try:
        item, = session.infer_frame(np.zeros((SIDE, SIDE, 3), np.uint8), pose_blend=1.0, jaw_adaptation=True).replacements
        assert not item.projection.jaw.applied and item.projection.jaw.reason == "contour_invalid"
    finally:
        session.release_session()


def test_a_moderate_yaw_is_not_gated_but_a_large_one_is(projector, crop, baseline):
    camera = np.array(((1.5 * SIDE, 0, SIDE / 2), (0, 1.5 * SIDE, SIDE / 2), (0, 0, 1)))
    contour = target_contour(baseline, 1.12)
    reasons = {}
    for yaw in (20.0, 55.0):
        anchors, _ = cv2.projectPoints(projector.anchors.astype(np.float64), np.array([0.0, np.radians(yaw), 0.0]),
                                       np.array([0.0, 0.0, 0.55]), camera, None)
        reasons[yaw] = projector.render(crop, anchors.reshape(-1, 2), 1.0, jaw_contour=contour).jaw.reason
    assert reasons[20.0] != "pose_too_oblique" and reasons[55.0] == "pose_too_oblique"
