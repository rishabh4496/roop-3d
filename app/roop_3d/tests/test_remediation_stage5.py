"""Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression and "Use Face From Frame".

Neural inference is replaced by deterministic fixtures (as in the earlier remediation tests): these verify the
geometry, the contracts and the plumbing, not face-swap quality on real weights.
"""
import base64
import json
import logging
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import (ALL_CELLS, Pose3D, PoseCell, as_legacy_triplet, as_pose3d, estimate_pose_5pt,
                               legacy_pose_5pt, pose_cell, project_model5)
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.engine.detection.dense_landmarks import geometric_pose
from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper
from roop_3d.faceset import pose_matrix as stage0
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset.identity_blend import IdentityBlendSpec
from roop_3d.faceset.pose_selector import PoseSelector
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, PoseBinSummary, QualityMetrics
from roop_3d.jobs.schema import JobRequest
from roop_3d.reconstruction.expression_tps import (BLOCKS, apply_expression_to_crop, layout_is_plausible, warp_expression)
from roop_3d.reconstruction.frontalization import frontalize_crop
from roop_3d.server import workstation_preview as preview_module
from roop_3d.tests.test_remediation_stage4 import LandmarkSession, live_payload
from roop_3d.tests.test_stage04_faceset_reader import _create_synthetic_face_image, _create_v2_zip
from roop_3d.tests.test_stage12_worker import _fixture
from roop_3d.tests.test_workstation_preview import FixtureSession, preview_fixture

BASE = "http://127.0.0.1:8030"


@pytest.fixture(autouse=True)
def fresh_stage0(monkeypatch):
    monkeypatch.delenv(stage0.ENV_FLAG, raising=False)
    stage0.invalidate()
    yield
    stage0.invalidate()


# --------------------------------------------------------------------------- helpers
def unit(values) -> np.ndarray:
    vector = np.asarray(values, np.float64)
    return vector / np.linalg.norm(vector)


def angle_vector(degrees: float, base: int = 0) -> np.ndarray:
    """A unit 512-D vector `degrees` away from basis vector `base` (in the base/base+1 plane)."""
    vector = np.zeros(512)
    vector[base], vector[base + 1] = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return vector


def make_view(index: int, yaw: float, pitch: float, embedding: np.ndarray, *, roll: float = 0.0, score: float = 0.9) -> FaceView:
    """`yaw/pitch` here are FaceView values, i.e. the LEGACY frame (pitch positive = looking down)."""
    return FaceView(index=index, image_bytes=b"png", embedding=tuple(float(x) for x in unit(embedding)),
                    landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)), bbox=(0.0, 0.0, 32.0, 32.0),
                    yaw=yaw, pitch=pitch, roll=roll, expression={}, quality=QualityMetrics(score=score, detector_confidence=1.0),
                    pose_bin="center_center")


def make_document(faceset_id: str, views: list[FaceView], canonical: np.ndarray | None = None, **metadata) -> FaceSetDocument:
    mean = unit(canonical if canonical is not None else np.mean([v.embedding for v in views], axis=0))
    summary = PoseBinSummary(yaw_bin="center", pitch_bin="center", members=(views[0].index,), support=1,
                             representative_index=views[0].index, embedding=tuple(float(x) for x in unit(views[0].embedding)))
    return FaceSetDocument(faceset_id=faceset_id, name=faceset_id, source_sha256="b" * 64, archive_format="v2_standard",
                           canonical_embedding=tuple(float(x) for x in mean), medoid_embedding=tuple(float(x) for x in mean),
                           views=tuple(views), pose_bins={"center_center": summary}, metadata=metadata)


# One legacy-frame (yaw, pitch) per cell: pitch + = down, so "down" cells use positive legacy pitch.
CELL_POSES = {(yi, pi): ((-30.0, 0.0, 30.0)[yi], (20.0, 0.0, -20.0)[pi]) for yi in range(3) for pi in range(3)}


def full_coverage_document(faceset_id="alice", base=0) -> FaceSetDocument:
    identity = angle_vector(0, base)
    views = []
    for cell_number, ((yi, pi), (yaw, pitch)) in enumerate(CELL_POSES.items()):
        views.append(make_view(cell_number, yaw, pitch, identity + 0.08 * angle_vector(90, base + 4 + 2 * cell_number)))
    return make_document(faceset_id, views, canonical=identity)


# --------------------------------------------------------------------------- (1) pose estimation + sanitizer
def test_estimator_recovers_known_rotations_and_signs():
    worst = 0.0
    for yaw in (-60, -30, 0, 15, 45):
        for pitch in (-30, 0, 20):
            for roll in (-20, 0, 25):
                truth = Pose3D(yaw=yaw, pitch=pitch, roll=roll)
                found = estimate_pose_5pt(project_model5(truth, scale=2400, center=(300, 200)))
                worst = max(worst, abs(found.yaw - yaw), abs(found.pitch - pitch), abs(found.roll - roll))
    assert worst < 1e-6
    # noisy detections still land close (a 0.5 px jitter on a ~150 px face)
    rng = np.random.default_rng(3)
    points = project_model5(Pose3D(yaw=25, pitch=10, roll=0), scale=2400, center=(300, 200)) + rng.normal(0, .5, (5, 2))
    noisy = estimate_pose_5pt(points)
    assert abs(noisy.yaw - 25) < 5 and abs(noisy.pitch - 10) < 6
    # looking up is positive pitch; the legacy frame (stored data) is the mirror image
    up = estimate_pose_5pt(project_model5(Pose3D(yaw=0, pitch=20, roll=0), scale=2400))
    assert up.pitch > 0 and up.legacy_tuple()[1] < 0
    assert legacy_pose_5pt(project_model5(Pose3D(yaw=0, pitch=20, roll=0), scale=2400)) == up.legacy_tuple()
    assert geometric_pose(project_model5(Pose3D(yaw=0, pitch=20, roll=0), scale=2400)) == up.legacy_tuple()
    # degenerate input never raises
    assert estimate_pose_5pt(np.zeros((5, 2))) == Pose3D(yaw=0, pitch=0, roll=0)
    assert estimate_pose_5pt(np.full((5, 2), np.nan)) == Pose3D(yaw=0, pitch=0, roll=0)


def test_sanitizer_normalizes_every_pose_spelling():
    expected = Pose3D(yaw=10.0, pitch=-5.0, roll=3.0)  # legacy (10, 5, 3): pitch + = down -> standard -5
    spellings = [(10, 5, 3), [10.0, 5.0, 3.0], np.array([10, 5, 3.0]), {"yaw": 10, "pitch": 5, "roll": 3},
                 SimpleNamespace(yaw=10.0, pitch=-5.0, roll=3.0), expected]
    assert all(as_pose3d(s) == expected for s in spellings)
    assert as_pose3d((10, 5)) == Pose3D(yaw=10, pitch=-5, roll=0)                      # legacy 2-tuple
    assert as_pose3d({"yaw": 1, "pitch": 2}) == Pose3D(yaw=1, pitch=-2, roll=0)         # dict without roll
    assert as_pose3d((10, 5), frame="standard") == Pose3D(yaw=10, pitch=5, roll=0)
    assert as_pose3d(None, default=expected) is expected
    assert as_legacy_triplet((10, 5)) == (10.0, 5.0, 0.0)
    assert as_pose3d((400, 400, 540)).yaw == 90 and as_pose3d((0, 0, 540)).roll == pytest.approx(-180)  # clamp / wrap, never crash
    for bad in ((1, 2, 3, 4), (1,), (math.nan, 0), (0, math.inf, 0), "abc", None, {"yaw": 1}, object(), [[1, 2], [3, 4]]):
        with pytest.raises(DomainError) as error:
            as_pose3d(bad)
        assert error.value.code == "POSE_INVALID"


def test_mixed_two_and_three_tuple_poses_do_not_crash_the_matcher():
    document = full_coverage_document()
    selector = PoseSelector(document)
    mixed = [(0, 0), (0, 0, 0), [30, -20], (30.0, -20.0, 5.0), np.array([-30, 20]), np.array([-30, 20, 0.0]),
             {"yaw": 0, "pitch": 0}, {"yaw": 30, "pitch": -20, "roll": 0}, Pose3D(yaw=30, pitch=20, roll=0)]
    picks = [selector.select_best_reference(pose).view_index for pose in mixed]
    assert picks[0] == picks[1] and picks[2] == picks[3] and picks[4] == picks[5] and picks[6] == picks[0]
    assert picks[2] == picks[7] == picks[8]  # legacy (30,-20) == standard Pose3D(30, +20): the same head attitude
    assert len({picks[0], picks[2], picks[4]}) == 3
    assert selector.select_best_reference(0.0, 0.0).view_index == picks[0]          # historical positional call still works
    assert selector.select_best_reference(0.0, 0.0, 0.0, 0.5).view_index == picks[0]

    swapper = bare_swapper(document)
    for pose in mixed:
        vector, index, weight = swapper.reference_embedding(None, pose)
        assert vector.shape == (512,) and abs(np.linalg.norm(vector) - 1) < 1e-6 and 0 <= weight <= 0.15
    with pytest.raises(DomainError):
        swapper.reference_embedding(None, (1, 2, 3, 4))


def test_archive_poses_in_any_shape_load_into_views(tmp_path):
    identity = [float(x) for x in angle_vector(0)]
    images = [_create_synthetic_face_image(float(k)) for k in range(4)]
    poses = [[10, 5], [10, 5, 2], None, [5]]  # 2-list, 3-list, absent (geometry fallback), invalid -> neutral
    sources = [{"source_index": k, "filename": f"{k}.png", "identity": {"embedding": identity},
                **({"pose": p} if p is not None else {"geometry": {"yaw": 7.0, "pitch": 3.0}}),
                "quality": {"score": 0.9, "detector_confidence": 1.0}} for k, p in enumerate(poses)]
    path = tmp_path / "mixed.fsz"
    path.write_bytes(_create_v2_zip(images, {"version": 2, "schema": "roop.fsz", "source_name": "mixed", "sources": sources}))
    reader = FaceSetReader(SimpleNamespace(), cache_root=tmp_path / "cache")
    views = {v.index: (v.yaw, v.pitch, v.roll) for v in reader.read(path, faceset_id="mixed").views}
    assert views == {0: (10.0, 5.0, 0.0), 1: (10.0, 5.0, 2.0), 2: (7.0, 3.0, 0.0), 3: (0.0, 0.0, 0.0)}


def bare_swapper(document: FaceSetDocument, by_track: dict | None = None) -> IdentitySwapper:
    """An IdentitySwapper with sources bound but no ONNX model (reference selection needs none)."""
    swapper = IdentitySwapper(SimpleNamespace())
    swapper.source, swapper.canonical = document, unit(document.canonical_embedding)
    swapper.sources_by_track = dict(by_track or {})
    swapper.canonical_by_track = {t: unit(d.canonical_embedding) for t, d in swapper.sources_by_track.items()}
    swapper._selectors = {t: PoseSelector(d) for t, d in swapper.sources_by_track.items()}
    swapper._matrices = {None: stage0.preflight(document)}
    swapper._matrices.update({t: stage0.preflight(d) for t, d in swapper.sources_by_track.items()})
    return swapper


# --------------------------------------------------------------------------- (2) Stage 0 pose matrix + identity guard
def test_stage0_populates_all_nine_cells_and_freezes_them(caplog):
    caplog.set_level(logging.INFO, logger="roop3d.stage0")
    document = full_coverage_document()
    matrix = stage0.preflight(document)
    assert matrix is not None and len(matrix.cells) == 9
    assert (matrix.populated, matrix.filled) == (9, 0)
    assert {entry.cell for entry in matrix.cells} == set(ALL_CELLS)
    for (yi, pi), (yaw, pitch) in CELL_POSES.items():
        entry = matrix.lookup((yaw, pitch))                                  # legacy tuple
        assert entry.cell == PoseCell(yi, pi) and entry.support == 1 and not entry.filled
        assert entry.view_index == list(CELL_POSES).index((yi, pi))          # the cell's own view, not a neighbour's
        assert matrix.lookup(Pose3D(yaw=yaw, pitch=-pitch, roll=0)).cell == entry.cell   # same head, standard frame
        assert abs(np.linalg.norm(entry.vector) - 1) < 1e-6
        with pytest.raises(ValueError):
            entry.vector[0] = 0.0                                            # frozen
    assert len({tuple(np.round(e.vector, 5)) for e in matrix.cells}) == 9             # nine distinct identity vectors
    assert matrix.verify() and matrix.guard["max_distance"] <= stage0.GUARD_MAX_COSINE_DISTANCE
    assert stage0.PoseMatrix.from_json(matrix.to_json()).fingerprint == matrix.fingerprint
    tampered = matrix.to_json()
    tampered["cells"][0]["vector"][0] += 0.1
    with pytest.raises(DomainError):
        stage0.PoseMatrix.from_json(tampered)
    messages = [r.getMessage() for r in caplog.records if r.name == "roop3d.stage0"]
    assert any(m.startswith("[Stage0] pre-flight") and "cells=9" in m and "populated=9" in m for m in messages)
    assert stage0.preflight(document) is matrix                                # memoised: no rebuild per preview session


def test_cell_edges_follow_the_spec_thresholds():
    edges = {(-15.01, 0): (0, 1), (-15.0, 0): (1, 1), (15.0, 0): (1, 1), (15.01, 0): (2, 1),
             (0, -10.01): (1, 0), (0, -10.0): (1, 1), (0, 10.0): (1, 1), (0, 10.01): (1, 2)}
    for (yaw, pitch), (yi, pi) in edges.items():
        assert pose_cell(Pose3D(yaw=yaw, pitch=pitch, roll=0)) == PoseCell(yi, pi), (yaw, pitch)
    assert PoseCell(0, 0).name == "left_down" and PoseCell(2, 2).name == "right_up" and PoseCell(1, 1).name == "center_level"
    assert [c.index for c in ALL_CELLS] == list(range(9))


def test_sparse_portfolio_fills_missing_cells_from_nearest_real_view():
    identity = angle_vector(0)
    document = make_document("sparse", [make_view(0, 0, 0, identity), make_view(1, 30, 0, identity + .1 * angle_vector(90, 2)),
                                        make_view(2, -30, -20, identity + .1 * angle_vector(90, 4))])
    matrix = stage0.preflight(document)
    assert (matrix.populated, matrix.filled) == (3, 6)
    for entry in matrix.cells:
        assert entry.filled == (entry.support == 0)
        if entry.filled:
            assert entry.filled_from is not None and matrix.entry(entry.filled_from).support > 0
            assert entry.confidence < matrix.entry(entry.filled_from).confidence   # filled cells are less trusted
    # every pose on a fine grid resolves; nothing raises, nothing is searched
    for yaw in range(-90, 91, 6):
        for pitch in range(-60, 61, 6):
            assert matrix.lookup((yaw, pitch)).vector.shape == (512,)


def test_identity_guard_excludes_off_identity_views_and_rejects_hopeless_sets():
    identity = angle_vector(0)
    stranger = angle_vector(90, 0)   # cosine distance 1.0 from the identity: beyond the 0.80 boundary
    document = make_document("mixed", [make_view(0, 0, 0, identity), make_view(1, 30, 0, stranger)], canonical=identity)
    matrix = stage0.preflight(document)
    assert matrix.guard["rejected_views"] == 1 and matrix.entry(PoseCell(2, 1)).filled  # the stranger never entered a cell
    assert all(stage0.cosine_distance(e.vector, identity) <= stage0.GUARD_MAX_COSINE_DISTANCE for e in matrix.cells)
    hopeless = make_document("hopeless", [make_view(0, 0, 0, stranger)], canonical=identity)
    with pytest.raises(DomainError) as error:
        stage0.preflight(hopeless)
    assert error.value.code == "IDENTITY_GUARD_VIOLATION"


def blend_pair(angle: float):
    a = make_document("a", [make_view(i, *CELL_POSES[k], angle_vector(0)) for i, k in enumerate(CELL_POSES)], canonical=angle_vector(0))
    b = make_document("b", [make_view(i, *CELL_POSES[k], angle_vector(angle)) for i, k in enumerate(CELL_POSES)], canonical=angle_vector(angle))
    return a, b


def test_identity_mixer_is_blended_per_cell_inside_the_guard():
    a, b = blend_pair(60)
    spec = IdentityBlendSpec(method="slerp", source_face_ids=("a", "b"), weights=(0.6, 0.4))
    matrix = stage0.build_blended_matrix([a, b], spec)
    assert matrix.mode == "blend:slerp" and matrix.populated == 9
    expected = unit(angle_vector(0) * math.sin(math.radians(36)) + angle_vector(60) * math.sin(math.radians(24)))  # slerp t=0.4 at 60 deg
    for entry in matrix.cells:
        assert np.allclose(entry.vector, expected, atol=1e-5)
        assert stage0.cosine_distance(entry.vector, angle_vector(0)) <= stage0.GUARD_MAX_COSINE_DISTANCE
    linear = stage0.build_blended_matrix([a, b], IdentityBlendSpec(method="linear", source_face_ids=("a", "b"), weights=(0.6, 0.4)))
    assert not np.allclose(linear.entry(PoseCell(1, 1)).vector, matrix.entry(PoseCell(1, 1)).vector, atol=1e-4)


def test_tangent_dials_are_clamped_to_the_guard_and_hopeless_blends_rejected():
    a, b = blend_pair(82)   # 82 deg apart: distance(a, b) = 0.861 > 0.80
    # a full dial toward b would land exactly on b (distance 0.861 from the dominant source a): must be scaled back
    spec = IdentityBlendSpec(source_face_ids=("a", "b"), weights=(0.51, 0.49), tangent_dials=(0.0, 1.0))
    matrix = stage0.build_blended_matrix([a, b], spec)
    assert matrix.guard["clamped_cells"] and all(c["dial_scale"] < 1 for c in matrix.guard["clamped_cells"])
    assert matrix.guard["max_distance"] <= stage0.GUARD_MAX_COSINE_DISTANCE
    undialed = stage0.build_blended_matrix([a, b], IdentityBlendSpec(source_face_ids=("a", "b"), weights=(0.51, 0.49)))
    assert not undialed.guard["clamped_cells"]
    # near-antipodal sources: even the plain midpoint is beyond the boundary
    far_a, far_b = blend_pair(170)
    with pytest.raises(DomainError) as error:
        stage0.build_blended_matrix([far_a, far_b], IdentityBlendSpec(source_face_ids=("a", "b")))
    assert error.value.code == "IDENTITY_GUARD_VIOLATION"


def test_blended_document_carries_its_stage0_matrix_into_the_swapper():
    from roop_3d.faceset.identity_blend import blend_documents
    a, b = blend_pair(60)
    blended = blend_documents([a, b], IdentityBlendSpec(source_face_ids=("a", "b"), weights=(0.5, 0.5)))
    assert blended.metadata["stage0"]["mode"] == "blend:slerp" and len(blended.metadata["stage0"]["cells"]) == 9
    swapper = bare_swapper(blended)
    vector, index, _ = swapper.reference_embedding(None, (0, 0))
    assert index in range(9) and abs(np.linalg.norm(vector) - 1) < 1e-6


def test_stage0_lookup_is_o1_and_never_runs_the_per_frame_search(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="roop3d.stage0")
    document = full_coverage_document()
    swapper = bare_swapper(document)

    def boom(*args, **kwargs):
        raise AssertionError("per-frame pose search must not run while Stage 0 is on")

    monkeypatch.setattr(PoseSelector, "select_best_reference", boom)
    for yaw, pitch in [(0, 0), (30, 20), (-30, -20), (0, 0)]:
        swapper.reference_embedding(None, (yaw, pitch))
    lookups = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[Stage0] lookup")]
    assert len(lookups) == 3 and "cell=center_level" in lookups[0]   # first hit per cell is INFO, repeats are DEBUG


def test_roop_stage0_zero_and_unindexed_portfolios_fall_back_per_frame(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="roop3d.stage0")
    document = full_coverage_document()
    monkeypatch.setenv(stage0.ENV_FLAG, "0")
    swapper = bare_swapper(document)
    assert swapper._matrices == {None: None}
    expected = PoseSelector(document).select_best_reference((30, 20)).view_index
    assert swapper.reference_embedding(None, (30, 20))[1] == expected        # per-frame PoseSelector result
    assert any("[Stage0] disabled" in r.getMessage() for r in caplog.records)
    monkeypatch.delenv(stage0.ENV_FLAG)
    stage0.invalidate()
    identity = angle_vector(0)
    custom = make_document("custom", [make_view(i, 0, 0, identity + .05 * angle_vector(90, 2 + i)) for i in range(3)])
    assert stage0.preflight(custom) is None                                  # every view claims dead-centre: no usable spread
    assert stage0.preflight(make_document("flagged", [make_view(0, 0, 0, identity)], custom_angles=True)) is None
    assert bare_swapper(custom).reference_embedding(None, (5, 5))[0].shape == (512,)
    assert any("custom angles" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- (3) TPS expression warp + frontalization
def synthetic_face106(cx=128.0, cy=128.0, w=100.0, mouth_open=0.0, brow_raise=0.0) -> np.ndarray:
    """A plausible 106-point face in the standard block layout (contour 0-32, eyes 33-42/87-96, brows 43-51/97-105, mouth 52-71, nose 72-86)."""
    points = np.zeros((106, 2))
    t = np.linspace(0, np.pi, 33)
    points[0:33] = np.column_stack((cx + w * np.cos(np.pi - t), cy + 0.2 * w + 0.9 * w * np.sin(t)))

    def ring(centre, rx, ry, count):
        a = np.linspace(0, 2 * np.pi, count, endpoint=False)
        return np.column_stack((centre[0] + rx * np.cos(a), centre[1] + ry * np.sin(a)))

    points[33:43] = ring((cx - 0.4 * w, cy - 0.1 * w), 0.18 * w, 0.07 * w, 10)
    points[87:97] = ring((cx + 0.4 * w, cy - 0.1 * w), 0.18 * w, 0.07 * w, 10)
    points[43:52] = np.column_stack((np.linspace(cx - 0.6 * w, cx - 0.2 * w, 9), np.full(9, cy - 0.3 * w - brow_raise * w)))
    points[97:106] = np.column_stack((np.linspace(cx + 0.2 * w, cx + 0.6 * w, 9), np.full(9, cy - 0.3 * w - brow_raise * w)))
    points[72:87] = ring((cx, cy + 0.15 * w), 0.1 * w, 0.1 * w, 15)
    points[52:72] = ring((cx, cy + 0.55 * w), 0.3 * w, 0.05 * w + mouth_open * w, 20)
    return points


def smooth_image(size=256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.dstack([xx * 0.9, yy * 0.9, (xx + yy) * 0.45]).astype(np.uint8)


def test_tps_moves_inner_expression_landmarks_and_pins_the_silhouette():
    neutral, expressive = synthetic_face106(), synthetic_face106(mouth_open=0.12, brow_raise=0.05)
    assert layout_is_plausible(neutral) and layout_is_plausible(expressive)
    image = smooth_image()
    result = warp_expression(image, neutral, expressive, 1.0)
    assert result.applied and result.reason == "OK"
    moved = np.linalg.norm(result.warped_landmarks - neutral, axis=1)
    wanted = np.linalg.norm(expressive - neutral, axis=1)
    contour, nose = list(BLOCKS["contour"]), list(BLOCKS["nose"])
    # (a) outer boundary + nose: unchanged to numerical precision
    assert moved[contour].max() < 1e-3 and moved[nose].max() < 1e-3 and result.boundary_drift_px < 1e-3
    # (b) expression landmarks: shifted by what was asked for
    for name in ("mouth", "brow_a", "brow_b"):
        idx = list(BLOCKS[name])
        assert wanted[idx].max() > 3
        assert np.allclose(moved[idx], wanted[idx], rtol=0.02, atol=0.05), name
    # half weight moves half as far
    half = warp_expression(image, neutral, expressive, 0.5)
    assert np.allclose(np.linalg.norm(half.warped_landmarks - neutral, axis=1)[list(BLOCKS["mouth"])], wanted[list(BLOCKS["mouth"])] / 2, atol=0.05)
    # the pixels follow the landmarks: a bright dot on the lower lip travels with it; one on the jaw stays put
    marked = image.copy()
    lip = neutral[list(BLOCKS["mouth"])][np.argmax(neutral[list(BLOCKS["mouth"])][:, 1])]
    jaw = neutral[16]
    for centre in (lip, jaw):
        cv2.circle(marked, (int(round(centre[0])), int(round(centre[1]))), 3, (255, 255, 255), -1)
    out = warp_expression(marked, neutral, expressive, 1.0).image
    lip_after = expressive[list(BLOCKS["mouth"])][np.argmax(neutral[list(BLOCKS["mouth"])][:, 1])]

    def dot_centre(img, near, radius=10):
        y0, x0 = int(near[1]), int(near[0])
        window = img[max(0, y0 - radius):y0 + radius, max(0, x0 - radius):x0 + radius].min(axis=2) > 240
        ys, xs = np.nonzero(window)
        return None if len(xs) == 0 else (xs.mean() + max(0, x0 - radius), ys.mean() + max(0, y0 - radius))

    seen = dot_centre(out, lip_after)
    assert seen is not None and np.hypot(seen[0] - lip_after[0], seen[1] - lip_after[1]) < 2.5
    still = dot_centre(out, jaw)
    assert still is not None and np.hypot(still[0] - jaw[0], still[1] - jaw[1]) < 1.0
    # the image border is pinned too
    assert np.abs(out[0].astype(int) - image[0]).max() <= 2 and np.abs(out[:, 0].astype(int) - image[:, 0]).max() <= 2


def test_tps_refuses_unverified_layouts_and_failed_detections():
    neutral = synthetic_face106()
    shuffled = neutral[np.random.default_rng(0).permutation(106)]
    assert not layout_is_plausible(shuffled)
    image = smooth_image()
    assert warp_expression(image, shuffled, neutral, 1.0).reason == "LAYOUT_UNVERIFIED"
    assert warp_expression(image, neutral, neutral, 0.0).reason == "WEIGHT_ZERO"
    assert warp_expression(image, neutral[:50], neutral, 1.0).reason == "LANDMARKS_INVALID"
    with pytest.raises(ValueError):
        warp_expression(image, neutral, neutral, 1.5)
    identity = apply_expression_to_crop(image, np.array([[1.0, 0, 0], [0, 1.0, 0]]), neutral, 0.7, lambda crop: None)
    assert not identity.applied and identity.reason == "DETECTION_FAILED" and identity.image is image
    assert apply_expression_to_crop(image, np.eye(2, 3), neutral, 0.7, lambda crop: 1 / 0).reason == "DETECTION_FAILED"


def test_pipeline_hook_maps_target_landmarks_into_the_generated_crop():
    neutral, expressive = synthetic_face106(), synthetic_face106(mouth_open=0.12)
    frame_to_crop = np.array([[0.5, 0.0, 10.0], [0.0, 0.5, 6.0]])               # frame -> crop
    inverse = cv2.invertAffineTransform(frame_to_crop)                         # crop -> frame (what the pipeline stores)
    crop = smooth_image(128)
    detected = neutral * 0.5 + np.array([10.0, 6.0])                           # landmarks of the generated face, in crop space
    seen = []

    def detect(image):
        seen.append(image.shape[:2])
        return detected * (image.shape[0] / 128)                               # the hook upsamples before detecting

    result = apply_expression_to_crop(crop, inverse, expressive, 1.0, detect)
    assert result.applied and seen == [(256, 256)]
    mouth = list(BLOCKS["mouth"])
    expected = (expressive * 0.5 + np.array([10.0, 6.0]))[mouth] - detected[mouth]
    assert np.allclose(result.warped_landmarks[mouth] - detected[mouth], expected, atol=0.05)
    assert np.linalg.norm(result.warped_landmarks[list(BLOCKS["contour"])] - detected[list(BLOCKS["contour"])], axis=1).max() < 1e-3


def test_frontalization_reduces_yaw_and_restores_geometry():
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    crop = np.dstack([xx * 1.9, yy * 1.9, (xx + yy) * 0.95]).astype(np.uint8)
    points = project_model5(Pose3D(yaw=40, pitch=10, roll=0), scale=1500, center=(64, 64))
    full = frontalize_crop(crop, points, 1.0)
    assert full.applied and full.pose.yaw == pytest.approx(40, abs=1e-6) and full.target_pose.yaw == pytest.approx(0, abs=1e-9)
    assert full.target_pose.pitch == pytest.approx(0, abs=1e-9)
    half = frontalize_crop(crop, points, 0.5)
    assert half.target_pose.yaw == pytest.approx(20, abs=1e-6) and half.target_pose.pitch == pytest.approx(5, abs=1e-6)
    assert not np.array_equal(full.image, crop) and not np.array_equal(full.image, half.image)
    # the warped frontal landmarks really read as less turned
    assert abs(estimate_pose_5pt(full.frontal_points[:5]).yaw) < 8 and abs(estimate_pose_5pt(half.frontal_points[:5]).yaw) < 28
    # swapper output comes back to the observed geometry
    restored = full.restore(full.image)
    assert np.abs(restored.astype(int) - crop.astype(int)).mean() < 6
    assert np.array_equal(full.image[0, :], crop[0, :]) or np.abs(full.image[0].astype(int) - crop[0]).max() <= 3  # border pinned
    # nothing to do at weight 0 or for an already frontal face
    zero = frontalize_crop(crop, points, 0.0)
    assert not zero.applied and zero.image is crop and zero.restore(crop) is crop
    assert not frontalize_crop(crop, project_model5(Pose3D(yaw=0, pitch=0, roll=0), scale=1500, center=(64, 64)), 1.0).applied
    with pytest.raises(ValueError):
        frontalize_crop(crop, points, 2.0)


def test_swapper_frontalizes_before_inference_and_unwarps_after():
    class Recorder:
        inputs = []

        def run(self, feeds):
            Recorder.inputs.append(feeds["target"].copy())
            return {"output": np.full((1, 3, 128, 128), 0.5, np.float32)}

    document = full_coverage_document()
    swapper = bare_swapper(document)
    swapper.model, swapper.projection = Recorder(), np.eye(512, dtype=np.float32)
    image = np.dstack([np.tile(np.arange(256, dtype=np.uint8), (256, 1))] * 3)
    landmarks = project_model5(Pose3D(yaw=40, pitch=0, roll=0), scale=2200, center=(128, 128))
    Recorder.inputs.clear()
    swapper.infer_frame(image, landmarks, pose=(40, 0), frontalization=0.0)
    swapper.infer_frame(image, landmarks, pose=(40, 0), frontalization=1.0)
    plain, flat = Recorder.inputs
    assert plain.shape == flat.shape == (1, 3, 128, 128) and not np.array_equal(plain, flat)
    with pytest.raises(ValueError):
        swapper.infer_frame(image, landmarks, frontalization=1.5)


def test_workstation_parameters_carry_the_new_controls_without_disturbing_old_requests():
    old = WorkstationParameters(feather_px=7)
    assert "pose_frontalization" not in old.model_dump() and "tps_expression_weight" not in old.model_dump()
    assert old.inference_options() == {"strength": 0.35}
    new = WorkstationParameters(pose_frontalization=0.6, tps_expression_weight=0.4)
    assert new.model_dump()["pose_frontalization"] == 0.6
    assert new.inference_options() == {"strength": 0.35, "frontalization": 0.6, "tps_expression": 0.4}
    request = {"target_path": "roop_3d/.runtime/media/t.mp4", "inspection_id": "a" * 32,
               "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}]}
    before = JobRequest.model_validate({**request, "workstation_parameters": {"feather_px": 7}}).identity_payload()
    assert "pose_frontalization" not in before["workstation_parameters"]
    after = JobRequest.model_validate({**request, "workstation_parameters": {"feather_px": 7, "pose_frontalization": 0.6}}).identity_payload()
    assert after["workstation_parameters"]["pose_frontalization"] == 0.6
    with pytest.raises(Exception):
        WorkstationParameters(pose_frontalization=1.5)


class RecordingSession(LandmarkSession):
    calls = []

    def infer_frame(self, image, **kwargs):
        RecordingSession.calls.append(kwargs)
        return super().infer_frame(image, **kwargs)


def test_live_preview_applies_frontalization_and_tps_controls(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(preview_module, "ReferencePipeline", RecordingSession)
    RecordingSession.calls.clear()
    payload = live_payload(fixture, pose_frontalization=0.7, tps_expression_weight=0.4)
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=payload)
        assert response.status_code == 200, response.text
        meta = json.loads(response.headers["x-preview-meta"])
        assert {"pose_frontalization", "tps_expression_weight"} <= set(meta["applied"]) and meta["unsupported"] == []
        assert RecordingSession.calls[-1]["frontalization"] == 0.7 and RecordingSession.calls[-1]["tps_expression"] == 0.4
        assert client.post("/api/preview", json={**payload, "pose_frontalization": 1.5}).status_code == 422


# --------------------------------------------------------------------------- (4) Use Face From Frame
class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def detect(self, image, threshold):
        self.calls += 1
        return list(self.faces), {}


class FakeRecognizer:
    """Identity comes from `identity` (a unit vector) plus a tiny deterministic per-call wobble."""
    def __init__(self, identity):
        self.identity, self.count = unit(identity), 0

    def embed(self, image, landmarks=None):
        self.count += 1
        return unit(self.identity + 0.01 * angle_vector(90, 6 + self.count % 8)).astype(np.float32)


def detection(center_x: float, pose: Pose3D):
    points = project_model5(pose, scale=230, center=(center_x, 28.0)).astype(np.float32)
    box = np.array([center_x - 14, 10, center_x + 14, 48], np.float32)
    return box, points


def extraction_app(tmp_path, monkeypatch):
    """App + fixture video with a fake detector (two faces) and recognizer, and a REAL FaceSetReader for previews."""
    app, worker, request = _fixture(tmp_path)
    service = app.state.face_extract
    service.detector = FakeDetector([detection(68.0, Pose3D(yaw=0, pitch=0, roll=0)),          # right-hand face (listed first on purpose)
                                     detection(28.0, Pose3D(yaw=28, pitch=-16, roll=0))])        # left-hand face: yaw +28, looking down
    service.recognizer = FakeRecognizer(angle_vector(0))
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner as Real
    reader = FaceSetReader(app.state.models, cache_root=tmp_path / "reader-cache")
    monkeypatch.setattr(preview_module, "MasterPipelineRunner", lambda *a, **k: Real(*a, **k, faceset_reader=reader))
    monkeypatch.setattr(preview_module, "ReferencePipeline", FixtureSession)
    return app, service, request


def extract(client, **body):
    payload = {"target_media_path": "roop_3d/.runtime/media/target.mp4", "frame_index": 3, "face_index": 0,
               "destination": "source_gallery", **body}
    return client.post("/api/faces/extract_from_frame", json=payload)


def test_extract_face_then_swap_it_onto_another_frame(tmp_path, monkeypatch):
    app, service, request = extraction_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        response = extract(client, face_index=1, faceset_name="Frame Person")   # face_index 1 = the frontal, right-hand face
        assert response.status_code == 201, response.text
        body = response.json()
        face = body["face"]
        assert body["destination"] == "source_gallery" and body["active_source"]["name"] == "Frame Person"
        faceset_id = body["active_source"]["faceset_id"]
        assert faceset_id.startswith("Frame_Person_")
        # a valid extraction: unit 512-D embedding, Pose3D, thumbnail + metadata stored
        assert face["embedding_dim"] == 512 and len(face["embedding"]) == 512 and abs(np.linalg.norm(face["embedding"]) - 1) < 1e-5
        assert face["pose"]["yaw"] == pytest.approx(0, abs=1e-3) and face["pose_cell"] == "center_level"
        assert base64.b64decode(face["thumbnail_base64"])[:2] == b"\xff\xd8"
        assert (tmp_path / face["thumbnail_path"]).is_file() and (tmp_path / face["metadata_path"]).is_file()
        stored = json.loads((tmp_path / face["metadata_path"]).read_text("utf-8"))
        assert stored["face_id"] == face["face_id"] and stored["target_media_path"].endswith("target.mp4") and "thumbnail_base64" not in stored
        assert body["pose_matrix"]["cells"] == 9
        # it shows up in the library and reads back (no recognizer: the archive carries its embedding)
        assert faceset_id in {item["faceset_id"] for item in client.get("/api/facesets").json()}
        document = FaceSetReader(app.state.models, cache_root=tmp_path / "rc").read(tmp_path / f"roop_3d/outputs/facesets/{faceset_id}.fsz", faceset_id=faceset_id)
        assert len(document.views) == 1 and np.allclose(document.views[0].embedding, face["embedding"], atol=1e-6)
        assert document.views[0].image_bytes[:4] == b"\x89PNG" and cv2.imdecode(np.frombuffer(document.views[0].image_bytes, np.uint8), 1).shape[:2] == (512, 512)
        # ...and is swapped onto a different frame through POST /api/preview
        swap = client.post("/api/preview", json={"target_path": request["target_path"], "inspection_id": request["inspection_id"],
                                                 "frame_index": 6, "source_face_ids": [faceset_id], "target_face_id": "track_0000"})
        assert swap.status_code == 200, swap.text
        swapped = cv2.imdecode(np.frombuffer(swap.content, np.uint8), cv2.IMREAD_COLOR)
        original = app.state.media.frame(request["target_path"], None, 6)
        assert swapped.shape == original.shape and not np.array_equal(swapped, original)
        assert np.abs(swapped.astype(int) - original.astype(int))[18:42, 18:42].mean() > 5
        assert json.loads(swap.headers["x-preview-meta"])["frame_index"] == 6


def test_face_index_is_left_to_right_and_bad_requests_are_rejected(tmp_path, monkeypatch):
    app, service, _ = extraction_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        left = extract(client, face_index=0).json()["face"]      # x=28, yaw +28, pitch -16 (looking down in the standard frame)
        right = extract(client, face_index=1).json()["face"]     # x=68, frontal
        assert left["pose"]["yaw"] == pytest.approx(28, abs=0.5) and left["pose"]["pitch"] == pytest.approx(-16, abs=0.5)
        assert left["pose_cell"] == "right_down" and right["pose_cell"] == "center_level"
        assert left["bbox"][0] < right["bbox"][0]
        assert extract(client, face_index=2).status_code == 422
        assert extract(client, destination="active_faceset").status_code == 422           # needs faceset_name
        assert extract(client, destination="active_faceset", faceset_name="nobody").status_code == 404
        assert extract(client, target_media_path="../outside.mp4").status_code == 422
        assert extract(client, destination="elsewhere").status_code == 422
        assert extract(client, frame_index=-1).status_code == 422
        service.detector = FakeDetector([])
        assert extract(client).status_code == 422


def test_face_can_be_extracted_from_a_still_image(tmp_path, monkeypatch):
    app, service, _ = extraction_app(tmp_path, monkeypatch)
    image = np.full((64, 96, 3), 120, np.uint8)
    path = tmp_path / "roop_3d/.runtime/media/still.png"
    cv2.imwrite(str(path), image)
    with TestClient(app, base_url=BASE) as client:
        response = extract(client, target_media_path="roop_3d/.runtime/media/still.png", frame_index=0, face_index=1)
        assert response.status_code == 201, response.text
        assert response.json()["face"]["pose_cell"] == "center_level"
        assert extract(client, target_media_path="roop_3d/.runtime/media/missing.png").status_code == 404


def test_extracted_face_is_inserted_into_the_matching_cell_and_stage0_refreshes(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="roop3d.stage0")
    app, service, _ = extraction_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        created = extract(client, face_index=1, faceset_name="Person").json()     # frontal face seeds the FaceSet
        faceset_id = created["active_source"]["faceset_id"]
        first = created["pose_matrix"]
        assert (first["populated"], first["filled"]) == (1, 8)
        inserted = extract(client, face_index=0, destination="active_faceset", faceset_name=faceset_id)   # yaw +28, down
        assert inserted.status_code == 201, inserted.text
        body = inserted.json()
        assert body["inserted"] == {"view_index": 1, "pose_cell": "right_down", "cell_index": PoseCell(2, 0).index}
        assert body["faceset"]["item_count"] == 2
        matrix = body["pose_matrix"]
        assert (matrix["populated"], matrix["filled"], matrix["cells"]) == (2, 7, 9)
        assert matrix["matrix"]["right_down"] == {"view_index": 1, "support": 1, "filled_from": None, "confidence": matrix["matrix"]["right_down"]["confidence"]}
        assert matrix["fingerprint"] != first["fingerprint"]                       # Stage 0 was refreshed, not reused
        # by display name too
        by_name = extract(client, face_index=0, destination="active_faceset", faceset_name="person")
        assert by_name.status_code == 201 and by_name.json()["faceset"]["item_count"] == 3
        # the archive on disk really holds the views, with poses in the legacy frame the rest of the code reads
        document = FaceSetReader(app.state.models, cache_root=tmp_path / "rc").read(tmp_path / f"roop_3d/outputs/facesets/{faceset_id}.fsz", faceset_id=faceset_id)
        assert [v.index for v in document.views] == [0, 1, 2]
        assert (document.views[1].yaw, document.views[1].pitch) == (pytest.approx(28, abs=0.5), pytest.approx(16, abs=0.5))
        assert json.loads((tmp_path / f"roop_3d/.runtime/facesets/{faceset_id}.json").read_text())["item_count"] == 3
        assert stage0.preflight(document).lookup(Pose3D(yaw=28, pitch=-16, roll=0)).view_index in (1, 2)
        assert not list((tmp_path / "roop_3d/outputs/facesets").glob(".*.new.fsz"))   # no temp files left behind
    assert any("[Stage0] pre-flight" in r.getMessage() for r in caplog.records)


def test_a_different_person_is_not_added_to_a_faceset(tmp_path, monkeypatch):
    app, service, _ = extraction_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        faceset_id = extract(client, face_index=1).json()["active_source"]["faceset_id"]
        archive = tmp_path / f"roop_3d/outputs/facesets/{faceset_id}.fsz"
        before = archive.read_bytes()
        service.recognizer = FakeRecognizer(angle_vector(90))                     # orthogonal identity: cosine distance 1.0
        rejected = extract(client, face_index=0, destination="active_faceset", faceset_name=faceset_id)
        assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "IDENTITY_GUARD_VIOLATION"
        assert archive.read_bytes() == before                                      # the archive is untouched


def test_target_clusters_are_pinned_with_stable_ranks(tmp_path, monkeypatch):
    app, service, _ = extraction_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        first = extract(client, face_index=1, destination="target_cluster").json()["cluster"]
        assert first["rank"] == 0 and first["created"] and first["count"] == 1
        merged = extract(client, face_index=0, destination="target_cluster").json()["cluster"]   # same person again
        assert merged["rank"] == 0 and not merged["created"] and merged["count"] == 2 and merged["similarity"] > 0.9
        service.recognizer = FakeRecognizer(angle_vector(90))
        other = extract(client, face_index=0, destination="target_cluster").json()["cluster"]
        assert other["rank"] == 1 and other["created"]
        service.recognizer = FakeRecognizer(angle_vector(0))
        again = extract(client, face_index=1, destination="target_cluster").json()["cluster"]
        assert again["rank"] == 0 and again["count"] == 3                          # earlier pins keep their rank
        listed = client.get("/api/faces/target_clusters", params={"target_media_path": "roop_3d/.runtime/media/target.mp4"}).json()
        assert [c["rank"] for c in listed["clusters"]] == [0, 1] and [c["count"] for c in listed["clusters"]] == [3, 1]
        assert abs(np.linalg.norm(listed["clusters"][0]["centroid"]) - 1) < 1e-5
        assert client.get("/api/faces/target_clusters", params={"target_media_path": "../x.mp4"}).status_code == 422


def test_exported_archive_round_trips_many_degenerate_embeddings_through_the_validator(tmp_path):
    """Per-view embeddings live in their own entries, so metadata.json cannot grow into the zip-bomb ratio check."""
    import zipfile
    from roop_3d.faceset.faceset_reader import export_v2
    views = [make_view(i, ((i % 7) - 3) * 12.0, ((i % 5) - 2) * 8.0, np.eye(512)[i]) for i in range(40)]   # one-hot: maximally compressible
    document = make_document("many", views, canonical=np.ones(512))
    archive = export_v2(document, tmp_path / "many.fsz", app_root=tmp_path)
    with zipfile.ZipFile(archive) as zf:
        manifest = zf.getinfo("metadata.json")
        assert manifest.file_size < 64 * 1024 and f"{len(views) - 1}.embedding.json" in zf.namelist()
        index = json.loads(zf.read("metadata.json"))["pose_matrix"]["cells"]
        assert sorted(i for members in index.values() for i in members) == list(range(40)) and len(index) == 9
    back = FaceSetReader(SimpleNamespace(), cache_root=tmp_path / "cache").read(archive, faceset_id="many")
    assert len(back.views) == 40
    assert all(np.allclose(a.embedding, b.embedding, atol=1e-6) and (a.yaw, a.pitch) == (b.yaw, b.pitch) for a, b in zip(back.views, views))
