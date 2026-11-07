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
from roop_3d.tests.model_governance_support import require_reference_models

SEGMENT_A, SEGMENT_B = 600, 3000       # two different shots of the local video, each with one continuous face


@pytest.fixture(scope="module")
def real():
    try:
        catalog = ModelCatalog()
        require_reference_models(catalog)
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
    capability = session.prepare_session(document, retarget=kwargs.pop("retarget", "off"),
                                         identity_store=IdentityAssetStore(tmp_path / "assets"), **kwargs)
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


def test_full_head_3d_modes_on_real_video(real, tmp_path):
    document, video = real
    for mode in ("pose_3d", "reenactment_3d"):
        session, capability = session_for(document, tmp_path, render_mode=mode)
        try:
            assert capability["full_head"]["active"]
            status = capability["render_modes"]["identity_assets"][document.faceset_id]
            assert status["available"] and status["layout"] in ("5pt", "68pt")
            seen = []
            for image, timestamp in frames(video, SEGMENT_A, 4):
                session.set_frame_timestamp(timestamp)
                result = session.infer_frame(image)
                seen.extend(result.replacements)
                output = session.composite_frame(image, result)
                assert output.shape == image.shape and output.dtype == np.uint8
            assert seen
            for replacement in seen:
                assert replacement.full_head and replacement.render_mode == mode
                assert replacement.method == "full_head_gpu_uv_only"
                assert replacement.fallback_reason is None
                assert replacement.image.shape[0] >= 512
                assert replacement.coverage.shape == replacement.image.shape[:2]
                assert 0 <= replacement.coverage.min() <= replacement.coverage.max() <= 1
            assert session.full_head_engines
        finally:
            session.release_session()


def test_unbuildable_identity_never_falls_back_to_an_inner_crop(real, tmp_path):
    document, _ = real
    blurred = []
    for view in document.views:
        image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        ok, png = cv2.imencode(".png", cv2.GaussianBlur(image, (0, 0), 30))
        assert ok
        blurred.append(view.model_copy(update={"image_bytes": png.tobytes()}))
    unusable = document.model_copy(update={"views": tuple(blurred)})
    with pytest.raises(DomainError) as error:
        session_for(unusable, tmp_path, render_mode="reenactment_3d")
    assert error.value.code == "IDENTITY_ASSET_NOT_BUILDABLE"


def test_full_head_neural_refinement_on_real_frame(real, tmp_path):
    document, video = real
    session, capability = session_for(document, tmp_path, render_mode="pose_3d", retarget="required")
    try:
        assert capability["full_head"]["neural_refinement"]
        image, timestamp = frames(video, SEGMENT_A, 1)[0]
        session.set_frame_timestamp(timestamp)
        replacements = session.infer_frame(image).replacements
        assert replacements
        assert all(item.method == "full_head_gpu_liveportrait" and item.full_head for item in replacements)
    finally:
        session.release_session()
