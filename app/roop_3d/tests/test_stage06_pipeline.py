"""Real local reference inference plus deterministic tracking/alignment contracts."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.audit.project_paths import legacy_root
from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker, optimal_assignment
from roop_3d.engine.runtime import RuntimeConfig, cuda_fallback_reason, tensor
from roop_3d.engine.volumetric.preprocess import aligned_crop, transform_points
from roop_3d.engine.volumetric.reference_adapter import LivePortraitAdapter, angle, rotation
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import ModelRegistry


@pytest.fixture(scope="module")
def real_faces():
    reader = FaceSetReader()
    source = reader.read(legacy_root() / "app/facesets/akansha.fsz")
    target = reader.read(legacy_root() / "app/facesets/anushree.fsz")
    images = []
    for document in (source, target):
        view = min(document.views, key=lambda item: abs(item.yaw))
        image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        images.append(image)
    return source, target, images[0], images[1]


@pytest.mark.parametrize("size", [128, 256, 512])
def test_exact_similarity_inverse_at_every_model_size(size):
    points = ARCFACE_REFERENCE_5PTS * 2 + np.array((45, 19), np.float32)
    crop = aligned_crop(np.zeros((512, 512, 3), np.uint8), points, size)
    assert crop.image.shape == (size, size, 3)
    assert crop.inverse.shape == (2, 3) and crop.inverse.dtype == np.float64
    np.testing.assert_allclose(transform_points(transform_points(points, crop.matrix), crop.inverse), points, atol=1e-10)
    with pytest.raises(DomainError):
        aligned_crop(crop.image, np.zeros((5, 2)), size)


def observation(identity: int, left: float) -> FaceObservation:
    vector = np.zeros(512, np.float32)
    vector[identity] = 1
    return FaceObservation(np.array((left, 0, left + 10, 10), np.float32), np.zeros((5, 2), np.float32),
                           np.zeros((106, 2), np.float32), vector, (0, 0, 0))


def test_global_assignment_beats_greedy_and_survives_crossing_occlusion():
    assert set(optimal_assignment(np.array(((0.9, 0.8), (0.85, 0.1))))) == {(0, 1), (1, 0)}
    assert optimal_assignment(np.array(((1.0, 0.49), (0.49, -1)))) == [(0, 0)]
    tracker = FaceTracker(max_missed=2)
    first = tracker.assign([observation(0, 0), observation(1, 30)])
    crossed = tracker.assign([observation(1, 0), observation(0, 30)])
    assert [face.track_id for face in first] == [0, 1]
    assert [face.track_id for face in crossed] == [1, 0]
    tracker.assign([])
    assert [face.track_id for face in tracker.assign([observation(0, 20)])] == [0]
    for _ in range(3):
        tracker.assign([])
    assert not tracker.active_ids
    assert tracker.assign([observation(0, 20)])[0].track_id == 2
    tracker.release()
    with pytest.raises(DomainError):
        tracker.assign([])


def test_runtime_contracts_fail_on_wrong_shape_dtype_nan_or_unprepared_session():
    for value in (np.zeros((1, 512), np.float64), np.zeros(511, np.float32), np.full(512, np.nan, np.float32)):
        with pytest.raises(DomainError) as error:
            tensor(value, (512,))
        assert error.value.code == "INFERENCE_TENSOR_INVALID"
    session = ReferencePipeline()
    with pytest.raises(DomainError):
        session.infer_frame(np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ValueError):
        RuntimeConfig(provider="TensorrtExecutionProvider")
    with pytest.raises(ValueError):
        RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile="cpu")
    for profile in ("6gb", "12gb"):
        assert RuntimeConfig(hardware_profile=profile).provider == "CPUExecutionProvider"


def test_motion_math_is_normalized_and_has_proper_rotations():
    assert angle(np.zeros((1, 66), np.float32)) == pytest.approx(0)
    for angles in ((0, 0, 0), (20, -30, 45)):
        matrix = rotation(angles)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(matrix) == pytest.approx(1)


def test_only_recognized_cuda_resource_or_operator_errors_allow_cpu_retry():
    assert cuda_fallback_reason("Only 4-D tensor is supported")
    assert cuda_fallback_reason("BFCArena::AllocateRawInternal Available memory of 30 is smaller than requested bytes of 32")
    assert cuda_fallback_reason("CUDNN_STATUS_ALLOC_FAILED")
    assert cuda_fallback_reason("Invalid tensor shape") is None
    assert cuda_fallback_reason("NaN output") is None


def test_real_multi_face_detection_dense_landmarks_and_identity_crossing(real_faces):
    _, _, source_image, target_image = real_faces
    tracker = FaceTracker()
    left = np.hstack((source_image, target_image))
    right = np.hstack((target_image, source_image))
    try:
        first = tracker.process(left, include203=True)
        second = tracker.process(right, include203=True)
        assert len(first) == len(second) == 2
        assert not tracker.scene_cut
        for face in first:
            assert face.observation.landmarks106.shape == (106, 2)
            assert face.observation.landmarks203.shape == (203, 2)
            assert np.linalg.norm(face.observation.embedding) == pytest.approx(1, abs=1e-5)
            match = max(second, key=lambda other: np.dot(other.observation.embedding, face.observation.embedding))
            assert match.track_id == face.track_id
            assert np.dot(match.observation.embedding, face.observation.embedding) > 0.9
        tracker.process(np.zeros_like(left))
        tracker.process(np.full_like(left, 255))
        assert tracker.scene_cut and not tracker.active_ids
    finally:
        tracker.release()


def test_real_swapper_identity_improvement_and_session_release(real_faces):
    source, _, _, image = real_faces
    session = ReferencePipeline()
    try:
        prepared = session.prepare_session(source, retarget="off", track_ids={0})
        assert prepared["identity_swapper"]["inference_state"] == "UNKNOWN"
        result = session.infer_frame(image)
        assert len(result.replacements) == 1
        output, observation = result.replacements[0], result.observations[0].observation
        assert output.image.shape == (128, 128, 3) and output.method == "identity_swap"
        before = session.tracker.recognizer.embed(image, observation.landmarks5)
        after = session.tracker.recognizer.embed(output.image, ARCFACE_REFERENCE_5PTS + np.array((8, 0), np.float32))
        source_vector = np.asarray(source.canonical_embedding)
        assert np.dot(after, source_vector) > np.dot(before, source_vector) + 0.2
        assert result.latency.swap_ms > 0 and result.latency.detection_ms > 0 and result.latency.retarget_ms == 0
        assert session.capability()["identity_swapper"]["inference_state"] == "VERIFIED_LOCAL"
        with pytest.raises(ValueError):
            session.infer_frame(image, strength=float("nan"))
        with pytest.raises(DomainError):
            session.swapper.model.run({"target": np.zeros((1, 3, 127, 128), np.float32), "source": np.zeros((1, 512), np.float32)})
    finally:
        session.release_session()
    assert session.tracker is None and not session.portrait.models and session.swapper.model is None
    with pytest.raises(DomainError):
        session.infer_frame(image)


def test_real_liveportrait_all_present_parts_and_identity_retention(real_faces):
    source, _, _, image = real_faces
    session = ReferencePipeline()
    try:
        session.prepare_session(source, retarget="required")
        result = session.infer_frame(image)
        output, observation = result.replacements[0], result.observations[0].observation
        assert output.image.shape == (512, 512, 3) and output.method == "identity_swap_liveportrait"
        landmarks = transform_points(observation.landmarks5, cv2.invertAffineTransform(output.inverse)).astype(np.float32)
        before = session.tracker.recognizer.embed(image, observation.landmarks5)
        after = session.tracker.recognizer.embed(output.image, landmarks)
        identity = np.asarray(source.canonical_embedding)
        assert np.dot(after, identity) > np.dot(before, identity) + 0.2
        capability = session.capability()["portrait"]
        assert capability["liveportrait"] == "VERIFIED_LOCAL"
        assert capability["learned_eye_retargeting"] == "VERIFIED_LOCAL"
        assert capability["learned_lip_retargeting"] == "UNKNOWN"
        assert set(capability["models"]) == {"appearance", "motion", "warp_spade", "stitch", "eye"}
        assert all(model["inference_state"] == "VERIFIED_LOCAL" for model in capability["models"].values())
        # Motion must be recomputed; no cached frame output substitutes for inference.
        moved = cv2.warpAffine(image, np.array(((1, 0, 2), (0, 1, 0)), np.float64), (512, 512))
        later = session.infer_frame(moved)
        assert later.replacements[0].track_id == output.track_id
        assert result.latency.retarget_ms > 0
        assert session.portrait.models["motion"].executions >= 4
        assert len(session.portrait.tracks) == 1
        for cached in session.portrait.tracks.values():
            assert cached.feature.nbytes == 8388608
    finally:
        session.release_session()


def test_missing_portrait_weights_use_real_identity_only_path(real_faces):
    source, _, _, image = real_faces
    catalog = ModelCatalog()
    manifest = catalog.registry.manifest.model_copy(update={"models": tuple(r for r in catalog.registry.records()
                                                if not r.name.startswith("liveportrait_"))})
    # This is an actual registry without the optional artifacts, not a fake inference model.
    reduced = ModelCatalog(registry=ModelRegistry(manifest=manifest))
    session = ReferencePipeline(reduced)
    try:
        capability = session.prepare_session(source, retarget="auto")
        assert not capability["portrait"]["available"] and capability["portrait"]["fallback_reason"]
        result = session.infer_frame(image)
        assert result.replacements[0].method == "identity_swap"
    finally:
        session.release_session()
    with pytest.raises(DomainError):
        session.prepare_session(source, retarget="required")
    assert not session.prepared and session.swapper.model is None


def test_inspection_endpoint_backend_has_selectable_integer_tracks(tmp_path, real_faces):
    import av
    from fastapi.testclient import TestClient
    from roop_3d.core.settings import Settings
    from roop_3d.server.app import create_app
    _, _, left, right = real_faces
    path = tmp_path / "roop_3d/.runtime/media/people.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.codec_context.max_b_frames = 0
        for image in (np.hstack((left, right)), np.hstack((right, left))):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    app = create_app(Settings(), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/media/inspect-faces", json={"path": path.relative_to(tmp_path).as_posix(), "max_samples": 8})
        assert response.status_code == 200, response.text
        payload = response.json()
    assert payload["tracking_method"] == "iou_arcface_hungarian"
    assert payload["decoded_frames"] == 2 and len(payload["tracks"]) == 2
    assert {item["persistent_id"] for item in payload["tracks"]} == {0, 1}
    assert all(item["last_frame"] == 1 and item["landmark_count"] == 106 for item in payload["tracks"])
    assert payload["recognizer_sha256"] and not payload["identity_continuity_verified"]
