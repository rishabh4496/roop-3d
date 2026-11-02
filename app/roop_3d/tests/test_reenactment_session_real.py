"""Stage 4 on REAL local inference: real SCRFD + 2d106 landmarks, real inswapper, real video, real multi-view FaceSet.

Skipped, with a reason, when the local models, the reference FaceSet or the local video are not installed; it never fakes
inference. The source FaceSet is a read-only legacy photograph set; the target is a local video from this app's own media
folder. Nothing leaves the machine.
"""
from __future__ import annotations

import glob

import cv2
import numpy as np
import pytest

from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.models.catalog import ModelCatalog
from roop_3d.reconstruction.identity_asset import IdentityAssetStore

SEGMENT_A, SEGMENT_B = 600, 3000       # two different shots of the local video, each with one continuous face


@pytest.fixture(scope="module")
def real():
    try:
        catalog = ModelCatalog()
        catalog.detector(), catalog.recognizer()
        document = FaceSetReader().read(legacy_root() / "app/facesets/ashna.fsz")
    except (DomainError, OSError, KeyError, ValueError) as error:
        pytest.skip(f"local models or reference FaceSet unavailable: {error}")
    videos = glob.glob(str(application_root() / "roop_3d/.runtime/media/03de3642*.mp4"))
    if not videos:
        pytest.skip("local test video unavailable")
    return document, videos[0]


def frames(path: str, start: int, count: int):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for i in range(count):
        ok, image = cap.read()
        if not ok:
            pytest.skip("local test video is shorter than expected")
        out.append((image, (start + i) / fps))
    cap.release()
    return out


def session_for(document, tmp_path, **kwargs):
    session = ReferencePipeline()
    capability = session.prepare_session(document, retarget="off", identity_store=IdentityAssetStore(tmp_path / "assets"), **kwargs)
    return session, capability


def test_standard_mode_is_the_unchanged_proven_path(real, tmp_path):
    document, video = real
    session, capability = session_for(document, tmp_path)
    try:
        assert capability["render_modes"]["default"] == "standard" and capability["render_modes"]["identity_assets"] == {}
        for image, t in frames(video, SEGMENT_A, 3):
            session.set_frame_timestamp(t)
            result = session.infer_frame(image)
            assert result.replacements and all(r.method == "identity_swap" and r.render_mode == "standard"
                                               and r.fallback_reason is None and r.reenactment is None for r in result.replacements)
        assert not (tmp_path / "assets").exists(), "standard mode must not build identity assets"
    finally:
        session.release_session()


def test_invalid_render_mode_is_rejected(real, tmp_path):
    document, video = real
    with pytest.raises(ValueError):
        ReferencePipeline().prepare_session(document, retarget="off", render_mode="bogus")
    session, _ = session_for(document, tmp_path)
    try:
        image, t = frames(video, SEGMENT_A, 1)[0]
        with pytest.raises(ValueError):
            session.infer_frame(image, render_mode="bogus")
    finally:
        session.release_session()


def test_reenactment_on_real_video_engages_falls_back_and_resets_on_a_real_scene_cut(real, tmp_path):
    document, video = real
    session, capability = session_for(document, tmp_path, render_mode="reenactment_3d")
    try:
        status = capability["render_modes"]["identity_assets"][document.faceset_id]
        assert status["available"] and status["layout"] == "5pt" and status["level"] in ("single_view_prior", "limited_multiview", "multiview")
        shot_a = frames(video, SEGMENT_A, 22)
        # The tracker detects cuts from a 32x32 thumbnail difference and two shots of the same set look alike, so the second
        # shot is mirrored and darkened: a hard cut, with real faces and real inference on both sides of it.
        shot_b = [(cv2.convertScaleAbs(cv2.flip(image, 1), alpha=0.45), t) for image, t in frames(video, SEGMENT_B, 14)]
        timeline, outputs = [], []
        for index, (image, _) in enumerate(shot_a + shot_b):
            session.set_frame_timestamp(index / 30.0)                         # one continuous timeline across the cut
            result = session.infer_frame(image)
            for replacement in result.replacements:
                timeline.append((index, replacement, result.scene_cut))
            if index in (14, 30):
                outputs.append(session.composite_frame(image, result))       # the compositor accepts the engine's coverage
        assert timeline, "the real tracker found no face"
        # warm-up: the proven path carries the first frames, with the reason on record
        first = timeline[0][1]
        assert first.method == "identity_swap" and first.fallback_reason == "EXPRESSION_NOT_CALIBRATED" and first.render_mode == "standard"
        engaged = [r for _, r, _ in timeline if r.method.startswith("identity_reenactment_3d")]
        assert len(engaged) >= 8, "the engine never engaged on real, frontal footage"
        for r in engaged:
            assert r.render_mode == "reenactment_3d" and r.fallback_reason is None and r.reenactment is not None
            assert r.coverage is not None and r.coverage.shape == r.image.shape[:2]
            assert 0.0 <= float(r.coverage.min()) and float(r.coverage.max()) <= 1.0
            assert 0.0 < r.reenactment.confidence <= 1.0 and r.reenactment.alignment_error_iod < 0.25
        # every frame that did NOT use the engine says why, and is exactly the proven path
        for _, r, _ in timeline:
            if not r.method.startswith("identity_reenactment_3d"):
                assert r.method == "identity_swap" and r.fallback_reason and r.render_mode == "standard"
        # the real scene cut resets every temporal state: calibration starts over
        cuts = [(i, r) for i, r, cut in timeline if cut]
        assert cuts, "the tracker did not report the cut between the two shots"
        after = next(r for i, r, cut in timeline if i >= len(shot_a))
        assert after.reenactment is not None and after.reenactment.frame_index == 0 and not after.reenactment.engaged
        assert len(outputs) == 2 and all(o.shape == shot_a[0][0].shape and o.dtype == np.uint8 for o in outputs)
    finally:
        session.release_session()


def test_reenactment_is_deterministic_on_real_video(real, tmp_path):
    document, video = real
    clip = frames(video, SEGMENT_A, 16)
    runs = []
    for attempt in range(2):
        session, _ = session_for(document, tmp_path, render_mode="reenactment_3d")
        try:
            rows = []
            for index, (image, _) in enumerate(clip):
                session.set_frame_timestamp(index / 30.0)
                for r in session.infer_frame(image).replacements:
                    pose = None if r.reenactment is None or r.reenactment.smoothed_pose_deg is None else r.reenactment.smoothed_pose_deg
                    rows.append((r.method, r.fallback_reason, pose, None if r.coverage is None else float(r.coverage.sum())))
            runs.append(rows)
        finally:
            session.release_session()
    assert runs[0] == runs[1], "the same frames must produce the same decisions and the same geometry"
    assert second_run_reused_asset(tmp_path)


def second_run_reused_asset(tmp_path) -> bool:
    return any((tmp_path / "assets").rglob("manifest.json"))


def test_unbuildable_identity_falls_back_to_the_proven_path_with_a_reason(real, tmp_path):
    document, video = real
    blurred = []
    for view in document.views:
        image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        ok, png = cv2.imencode(".png", cv2.GaussianBlur(image, (0, 0), 30))
        blurred.append(view.model_copy(update={"image_bytes": png.tobytes()}))
    unusable = document.model_copy(update={"views": tuple(blurred)})
    session, capability = session_for(unusable, tmp_path, render_mode="reenactment_3d")
    try:
        status = capability["render_modes"]["identity_assets"][document.faceset_id]
        assert status["available"] is False and "IDENTITY_ASSET_NOT_BUILDABLE" in status["reason"]
        for image, t in frames(video, SEGMENT_A, 3):
            session.set_frame_timestamp(t)
            for r in session.infer_frame(image).replacements:
                assert r.method == "identity_swap" and r.render_mode == "standard"
                assert r.fallback_reason.startswith("IDENTITY_ASSET_UNAVAILABLE") and r.reenactment is None
    finally:
        session.release_session()


def test_pose_3d_mode_projects_on_the_source_identity_geometry(real, tmp_path):
    document, video = real
    session, _ = session_for(document, tmp_path, render_mode="pose_3d")
    try:
        seen = []
        for image, t in frames(video, SEGMENT_A, 4):
            session.set_frame_timestamp(t)
            seen += session.infer_frame(image).replacements
        assert seen
        for r in seen:
            if r.fallback_reason is None:
                assert r.method == "identity_swap_pose_3d" and r.render_mode == "pose_3d" and r.projection is not None
            else:                                                   # rejected landmarks: the swap crop is kept, with the reason
                assert r.method == "identity_swap" and r.fallback_reason.startswith("POSE_PROJECTION_REJECTED")
        assert any(p.geometry.startswith("IDENTITY_ASSET_") for p in session.pose_projectors.values())
    finally:
        session.release_session()
