"""Explicit identity-asset sessions on real CPU math and an arithmetic ONNX swapper.

The scripted landmarks/FaceSets are synthetic: these tests prove routing, source
selection and truthful fallback, not photographic identity or 80-degree accuracy.
No detector, learned swapper or dense reconstruction bundle is claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402
import target_world as TW  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace  # noqa: E402
from roop_3d.engine.volumetric.session import ReferencePipeline, TrackedFrame  # noqa: E402
from roop_3d.reconstruction.identity_asset import IdentityAssetSettings, IdentityAssetStore  # noqa: E402
from roop_3d.tests.quality_world import swap_catalog  # noqa: E402


@pytest.fixture(scope="module")
def source_document():
    subject = W.make_subject("session_source", W.LAYOUT_68, 816, width=0.18, nose_depth=0.02)
    return W.full_set(subject, (0, -40, 40))


@pytest.fixture(scope="module")
def stored_identity(tmp_path_factory, source_document):
    store = IdentityAssetStore(tmp_path_factory.mktemp("session_assets"))
    asset, _ = store.get_or_build(source_document, IdentityAssetSettings())
    return store, asset


@pytest.fixture(scope="module")
def analytic_catalog(tmp_path_factory):
    return swap_catalog(tmp_path_factory.mktemp("session_models"))


@pytest.fixture(scope="module")
def performer():
    return TW.Performer(W.make_subject("session_target", W.LAYOUT_68, 817, width=-0.06))


@pytest.fixture
def session_factory(analytic_catalog, source_document, tmp_path, monkeypatch):
    import roop_3d.reconstruction.identity_asset as assets
    import roop_3d.reconstruction.morphable_backbone as dense

    # Isolate the two filesystem discovery boundaries, never the routing or renderer.
    monkeypatch.setattr(assets, "application_root", lambda: tmp_path)
    monkeypatch.setattr(dense, "application_root", lambda: tmp_path)
    monkeypatch.setenv("ROOP3D_AUTO_3D_YAW", "30")
    monkeypatch.setenv("ROOP3D_AUTO_3D_PITCH", "25")
    sessions = []

    def prepare(mode="standard", *, source=None, **kwargs):
        session = ReferencePipeline(analytic_catalog)
        sessions.append(session)
        session.select_swapper("hyperswap_1a_256")
        capability = session.prepare_session(source or source_document, retarget="off", render_mode=mode, **kwargs)
        return session, capability

    yield prepare
    for session in sessions:
        session.release_session()


def tracked_frame(performer, index=0, *, yaw=0.0, mouth_open=0.0, track_ids=(7,), scene_cut=False):
    observation, _ = TW.frame(performer, yaw=yaw, mouth_open=mouth_open, t=index / TW.FPS,
                              noise_px=0, scene_cut=scene_cut)
    face = FaceObservation(
        box=np.array([60, 40, 340, 370], np.float32), landmarks5=observation["landmarks5"].astype(np.float32),
        landmarks106=observation["landmarks106"].astype(np.float32), embedding=performer.subject.embedding,
        pose=(yaw, 0.0, 0.0),
    )
    faces = tuple(TrackedFace(track_id, face, index + 1, 0) for track_id in track_ids)
    return TrackedFrame(faces, scene_cut, frozenset(track_ids), index / TW.FPS, 0.0)


@pytest.fixture(scope="module")
def target_image():
    return np.random.default_rng(818).integers(45, 205, (*TW.CANVAS, 3), dtype=np.uint8)


def test_standard_preparation_does_not_build_or_activate_identity_assets(session_factory, tmp_path):
    store = IdentityAssetStore(tmp_path / "not_built")
    session, capability = session_factory(identity_store=store)
    assert capability["render_modes"]["identity_assets"] == {}
    assert capability["pose_projection"]["active"] is False
    assert not store.base.exists()
    assert session.prepared and not capability["full_head"]["active"]


def test_explicit_store_builds_then_reuses_the_selected_asset(session_factory, source_document, tmp_path):
    store = IdentityAssetStore(tmp_path / "explicit")
    session, first = session_factory("pose_3d", identity_store=store)
    first_status = first["render_modes"]["identity_assets"][source_document.faceset_id]
    assert first_status["available"] and first_status["identity_asset"]["reused"] is False
    key = first_status["identity_asset"]["build_key"]
    assert store.load(source_document.faceset_id, key).build_key == key
    again = session.prepare_session(source_document, retarget="off", render_mode="reenactment_3d", identity_store=store)
    status = again["render_modes"]["identity_assets"][source_document.faceset_id]
    assert status["available"] and status["identity_asset"]["reused"] is True
    assert status["identity_asset"]["build_key"] == key
    assert again["pose_projection"]["geometry"] == "SOURCE_IDENTITY_ASSET"
    assert not again["dense_face"]["available"] and not again["full_head"]["active"]


def test_pose_mode_renders_the_selected_identity_geometry(session_factory, stored_identity, performer, target_image):
    store, asset = stored_identity
    session, capability = session_factory("pose_3d", identity_store=store)
    item, = session.infer_frame(target_image, tracked=tracked_frame(performer)).replacements
    assert item.method == "identity_swap_pose_3d" and item.render_mode == "pose_3d"
    assert item.fallback_reason is None and item.reenactment is None and not item.full_head
    assert capability["pose_projection"]["geometry"] == "SOURCE_IDENTITY_ASSET"
    assert np.count_nonzero(item.coverage > 0.5) > 100
    # A rigid camera transform preserves pairwise metric distances. This catches
    # silently substituting the canonical head while claiming an asset was used.
    ids = np.array([0, 16, 30, 36, 45, 48, 54])
    camera_points = item.projection.geometry.mesh_camera_vertices[ids]
    source_points = asset.vertices[ids].astype(np.float64)
    camera_distances = np.linalg.norm(camera_points[:, None] - camera_points[None], axis=-1)
    source_distances = np.linalg.norm(source_points[:, None] - source_points[None], axis=-1)
    np.testing.assert_allclose(camera_distances, source_distances, atol=1e-7)
    assert not session.capability()["dense_face"]["available"]


@pytest.mark.parametrize("mode", ["pose_3d", "reenactment_3d"])
def test_cache_only_sessions_load_exact_prebuilt_assets(session_factory, stored_identity, source_document, mode):
    _, asset = stored_identity
    cache = IdentityAssetStore()
    cache.save(asset)
    _, capability = session_factory(mode)
    status = capability["render_modes"]["identity_assets"][source_document.faceset_id]
    assert status["available"] and status["identity_asset"]["reused"] is True
    assert status["identity_asset"]["build_key"] == asset.build_key


@pytest.mark.parametrize("mode", ["pose_3d", "reenactment_3d"])
def test_missing_asset_fallback_is_reported_per_replacement(session_factory, source_document, performer, target_image, mode):
    session, capability = session_factory(mode)
    status = capability["render_modes"]["identity_assets"][source_document.faceset_id]
    assert status["available"] is False and "IDENTITY_ASSET_NOT_BUILT" in status["reason"]
    item, = session.infer_frame(target_image, tracked=tracked_frame(performer)).replacements
    assert item.render_mode == "standard" and "IDENTITY_ASSET_UNAVAILABLE" in item.fallback_reason
    assert "IDENTITY_ASSET_NOT_BUILT" in item.fallback_reason
    assert item.method == ("identity_swap_pose_3d" if mode == "pose_3d" else "identity_swap")
    if mode == "pose_3d":
        assert np.count_nonzero(item.coverage > 0.5) > 100
        assert session.capability()["pose_projection"]["geometry"] == "ESTIMATED_CANONICAL_PROXY_68"
    else:
        baseline, _ = session_factory()
        expected, = baseline.infer_frame(target_image, tracked=tracked_frame(performer)).replacements
        np.testing.assert_array_equal(item.image, expected.image)
        assert item.reenactment is None


@pytest.mark.parametrize("field,value,reason", [
    ("faceset_id", "another_source", "IDENTITY_ASSET_MISMATCH"),
    ("faceset_source_sha256", "0" * 64, "IDENTITY_ASSET_MISMATCH"),
    ("build_key", "0" * 64, "IDENTITY_ASSET_NOT_BUILT"),
])
def test_mismatched_cached_manifest_never_activates_a_rig(
        session_factory, stored_identity, source_document, performer, target_image, field, value, reason):
    _, asset = stored_identity
    cache = IdentityAssetStore()
    directory = cache.save(asset)
    manifest = directory / "manifest.json"
    info = json.loads(manifest.read_text("utf-8"))
    info[field] = value
    manifest.write_text(json.dumps(info), encoding="utf-8")
    session, capability = session_factory("reenactment_3d")
    status = capability["render_modes"]["identity_assets"][source_document.faceset_id]
    assert not status["available"] and reason in status["reason"]
    item, = session.infer_frame(target_image, tracked=tracked_frame(performer)).replacements
    assert item.method == "identity_swap" and item.render_mode == "standard"
    assert reason in item.fallback_reason and item.reenactment is None


def test_track_source_mapping_does_not_borrow_another_identity_rig(
        session_factory, stored_identity, source_document, performer, target_image):
    _, asset = stored_identity
    IdentityAssetStore().save(asset)
    missing = source_document.model_copy(update={"faceset_id": "not_cached"})
    session, capability = session_factory("pose_3d", source=missing, sources_by_track={7: source_document})
    statuses = capability["render_modes"]["identity_assets"]
    assert statuses[source_document.faceset_id]["available"] and not statuses[missing.faceset_id]["available"]
    items = {item.track_id: item for item in session.infer_frame(
        target_image, tracked=tracked_frame(performer, track_ids=(3, 7))).replacements}
    assert items[7].render_mode == "pose_3d" and items[7].fallback_reason is None
    assert items[3].render_mode == "standard" and "IDENTITY_ASSET_NOT_BUILT" in items[3].fallback_reason


def test_reenactment_calibrates_then_transfers_expression_from_real_landmark_math(
        session_factory, stored_identity, performer, target_image):
    store, _ = stored_identity
    session, _ = session_factory("reenactment_3d", identity_store=store)
    items = [session.infer_frame(target_image, tracked=tracked_frame(
        performer, index, mouth_open=0.0 if index < 16 else 0.25)).replacements[0] for index in range(24)]
    assert items[0].method == "identity_swap" and items[0].render_mode == "standard"
    assert items[0].fallback_reason == "EXPRESSION_NOT_CALIBRATED"
    last = items[-1]
    assert last.method == "identity_reenactment_3d" and last.render_mode == "reenactment_3d"
    assert last.fallback_reason is None and last.reenactment.engaged and last.reenactment.weight == 1.0
    assert last.reenactment.params.mouth_open > 0.15
    covered = last.coverage > 0.5
    assert covered.sum() > 100
    np.testing.assert_allclose(last.image[covered], last.reenactment.image[covered], atol=1)
    cut = session.infer_frame(target_image, tracked=tracked_frame(performer, 24, scene_cut=True)).replacements[0]
    assert cut.fallback_reason == "EXPRESSION_NOT_CALIBRATED" and cut.render_mode == "standard"
    assert not session.capability()["full_head"]["active"] and not session.capability()["dense_face"]["available"]


def test_automatic_dense_route_still_fails_closed_without_a_bundle(session_factory, performer, target_image):
    session, _ = session_factory()
    with pytest.raises(DomainError) as failure:
        session.infer_frame(target_image, tracked=tracked_frame(performer, yaw=35))
    assert failure.value.code == "DENSE_FACE_ASSETS_MISSING"
    assert not session.capability()["dense_face"]["available"] and not session.capability()["full_head"]["active"]


def test_optional_dense_preview_reports_the_standard_swap_fallback(session_factory, performer, target_image):
    session, _ = session_factory()
    session.dense_face_optional = True
    item, = session.infer_frame(target_image, tracked=tracked_frame(performer, yaw=35)).replacements
    assert item.method == "identity_swap" and item.render_mode == "standard"
    assert "DENSE_FACE_ASSETS_MISSING" in item.fallback_reason
    assert "DENSE_FACE_ASSETS_MISSING" in session.capability()["dense_face"]["reason"]
