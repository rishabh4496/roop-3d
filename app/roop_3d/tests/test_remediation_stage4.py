"""Stage 4: non-blocking live preview, cache-first serving and model-lifecycle mutex.

Inference is the deterministic fixture used by test_workstation_preview (no neural fidelity or GPU
claims); what is verified here is the HTTP contract, the concurrency contract and the plumbing.
"""
import base64
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.engine.volumetric.session import Replacement, StageLatency
from roop_3d.faceset.identity_blend import IdentityBlendSpec, blend_embeddings
from roop_3d.server import workstation_preview as module
from roop_3d.server.engine_gate import EngineBusy, EngineGate
from roop_3d.server.live_preview import CacheEntry, MaskStroke, PreviewCache, rasterize_strokes
from roop_3d.tests.test_stage12_worker import _wait_state
from roop_3d.tests.test_workstation_preview import FixtureSession, preview_fixture

STAGE_1 = "STAGE_1_SPARSE_PRESCAN"
BASE = "http://127.0.0.1:8030"


def live_payload(fixture_payload: dict, **extra) -> dict:
    request = fixture_payload["request"]
    return {"target_path": request["target_path"], "inspection_id": request["inspection_id"],
            "face_mappings": request["face_mappings"], "frame_index": 2, **extra}


def decode(response) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, "response body is not a decodable image"
    return image


def lock_gpu(app, stage=STAGE_1):
    """What a running Stage 1 job looks like to the preview service: a gate owner plus a RUNNING stage."""
    app.state.pipeline.render_event("job-x", {"stage": stage, "state": "RUNNING"})
    return app.state.engine_gate.acquire_owner("RENDER_JOB")


class LandmarkSession(FixtureSession):
    """Same fixture inference, but with 5-point landmarks and a non-zero pose on the observation."""
    def infer_frame(self, image, **kwargs):
        result = super().infer_frame(image, **kwargs)
        face = result.observations[0].observation
        face.landmarks5 = np.array([[20, 22], [38, 22], [29, 30], [22, 38], [36, 38]], np.float32)
        face.pose = (12.5, -3.0, 1.5)
        return result


# --------------------------------------------------------------------------- gate verification (1)(2)(3)
def test_post_preview_returns_swapped_pixels_with_face_metadata(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ReferencePipeline", LandmarkSession)
    payload = live_payload(fixture)
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=payload)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-preview-cache"] == "MISS"
        swapped = decode(response)
        target = app.state.media.frame(payload["target_path"], None, payload["frame_index"])
        assert swapped.shape == target.shape
        # (1) a real swapped buffer: pixels differ from the unswapped target, in the face region
        assert not np.array_equal(swapped, target)
        assert np.abs(swapped.astype(int) - target.astype(int))[18:42, 18:42].mean() > 5
        meta = json.loads(response.headers["x-preview-meta"])
        face = meta["faces"][0]
        assert face["box"] == [18, 18, 42, 42] and len(face["landmarks5"]) == 5
        # raw tracker tuples are legacy-frame (pitch + = down); the API reports the standard frame (pitch + = up)
        assert (face["yaw"], face["pitch"], face["roll"]) == (12.5, 3.0, 1.5)
        assert meta["media_id"] == payload["inspection_id"] and meta["frame_index"] == 2
        assert "swap_weight" in meta["applied"] and meta["unsupported"] == []
        # JSON envelope carries the same pixels
        as_json = client.post("/api/preview?response=json", json=payload).json()
        assert as_json["faces"] == meta["faces"] and as_json["cached"] is True
        assert base64.b64decode(as_json["image"].split(",", 1)[1]) == response.content
    assert FixtureSession.instances[-1].released and app.state.engine_gate.previews_in_flight == 0


def test_gpu_contention_returns_503_busy_fast_without_inference(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture)
    sessions_before = len(FixtureSession.instances)
    token = lock_gpu(app)
    try:
        with TestClient(app, base_url=BASE) as client:
            client.get("/api/v1/health")  # warm the stack so the measurement is the endpoint, not first-use imports
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                response = client.post("/api/preview", json=payload)
                timings.append((time.perf_counter() - started) * 1000)
                assert response.status_code == 503
                assert response.json() == {"busy": True, "stage": STAGE_1, "retry_after_ms": 1000}
                assert response.headers["retry-after"] == "1"
            # (2) 503 with retry_after_ms 1000 well inside 50 ms (best-of-5 to absorb scheduler noise)
            assert min(timings) < 50, timings
            # the other endpoints obey the same rule
            assert client.get("/api/target/preview", params={"config": "0" * 16, "frame_index": 2}).status_code == 404
    finally:
        app.state.engine_gate.release_owner(token)
    assert len(FixtureSession.instances) == sessions_before, "no session may be created while the GPU is owned"


def test_cached_frames_are_served_while_the_gpu_is_locked(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture)
    with TestClient(app, base_url=BASE) as client:
        first = client.post("/api/preview", json=payload)
        assert first.status_code == 200 and first.headers["x-preview-cache"] == "MISS"
        sessions = len(FixtureSession.instances)
        token = lock_gpu(app)
        try:
            second = client.post("/api/preview", json=payload)
            # (3) cache-first: HTTP 200 although Stage 1 owns the GPU
            assert second.status_code == 200 and second.headers["x-preview-cache"] == "HIT"
            assert second.content == first.content
            assert client.post("/api/preview", json={**payload, "frame_index": 3}).status_code == 503  # uncached
            assert client.post("/api/preview", json={**payload, "color_strength": .1}).status_code == 503  # new config
        finally:
            app.state.engine_gate.release_owner(token)
        assert len(FixtureSession.instances) == sessions
        # once the GPU is free the previously-refused frame renders
        assert client.post("/api/preview", json={**payload, "frame_index": 3}).status_code == 200


def test_timestamp_selector_maps_to_the_same_cache_slot_as_frame_index(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture)
    with TestClient(app, base_url=BASE) as client:
        by_time = client.post("/api/preview", json={**{k: v for k, v in payload.items() if k != "frame_index"}, "timestamp": 0.08})
        assert by_time.status_code == 200 and by_time.headers["x-preview-frame-index"] == "2"  # 25 fps fixture
        token = lock_gpu(app)
        try:  # same configuration addressed by frame number is already cached
            by_index = client.post("/api/preview", json=payload)
            assert by_index.status_code == 200 and by_index.headers["x-preview-cache"] == "HIT"
        finally:
            app.state.engine_gate.release_owner(token)


def test_tensorrt_compilation_counts_as_gpu_owner(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    app.state.pipeline.set_engine("COMPILING_TRT", label="inswapper fp16")
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=live_payload(fixture))
        assert response.status_code == 503 and response.json()["stage"] == "TENSORRT_COMPILATION"
        app.state.pipeline.set_engine("IDLE")
        assert client.post("/api/preview", json=live_payload(fixture)).status_code == 200


# --------------------------------------------------------------------------- mutex vs Start button
class BlockingSession(FixtureSession):
    entered, proceed = threading.Event(), threading.Event()

    def infer_frame(self, image, **kwargs):
        type(self).entered.set()
        assert type(self).proceed.wait(15)
        return super().infer_frame(image, **kwargs)


def test_start_waits_for_in_flight_preview_and_blocks_new_previews(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ReferencePipeline", BlockingSession)
    BlockingSession.entered.clear(); BlockingSession.proceed.clear()
    gate, payload, outcome = app.state.engine_gate, live_payload(fixture), {}
    with TestClient(app, base_url=BASE) as client:
        flight = threading.Thread(target=lambda: outcome.update(preview=client.post("/api/preview", json=payload)))
        flight.start()
        assert BlockingSession.entered.wait(10), "preview inference never started"
        assert gate.previews_in_flight == 1

        acquired, seen = threading.Event(), {}

        def start_button():
            token = gate.acquire_owner("RENDER_JOB")
            seen["torn_down_first"] = FixtureSession.instances[-1].released  # model teardown already done by the preview
            seen["previews"] = gate.previews_in_flight
            acquired.set()
            gate.release_owner(token)

        start = threading.Thread(target=start_button)
        start.start()
        time.sleep(0.3)
        assert not acquired.is_set(), "Start must wait for the in-flight preview"
        # while Start is waiting, new previews are refused instead of starving it
        refused = client.post("/api/preview", json={**payload, "frame_index": 3})
        assert refused.status_code == 503 and refused.json()["stage"] == "MODEL_REBIND"

        BlockingSession.proceed.set()
        flight.join(15); start.join(15)
        assert outcome["preview"].status_code == 200, "the in-flight preview completes normally"
        assert acquired.is_set() and seen == {"torn_down_first": True, "previews": 0}
        assert client.post("/api/preview", json={**payload, "frame_index": 3}).status_code == 200


def test_job_execution_waits_for_in_flight_preview(tmp_path, monkeypatch):
    from roop_3d.tests.test_stage12_worker import _fixture
    app, worker, request = _fixture(tmp_path)
    worker.gate = app.state.engine_gate
    assert app.state.engine_gate.try_enter_preview()  # a preview is mid-inference
    with TestClient(app, base_url=BASE) as client:
        worker.start()
        try:
            job = client.post("/api/v1/jobs", json=request, headers={"Idempotency-Key": "k" * 8}).json()
            time.sleep(0.4)
            # the worker thread is parked on the gate, before the job even enters `preparing`
            assert client.get(f"/api/v1/jobs/{job['job_id']}").json()["state"] == "queued"
            assert app.state.engine_gate.busy_reason() == "MODEL_REBIND"
            app.state.engine_gate.leave_preview()
            done = _wait_state(client, job["job_id"], "succeeded")
            assert done["state"] == "succeeded"
        finally:
            worker.stop()
    assert app.state.engine_gate.owner_count == 0


def test_old_workstation_preview_is_gated_too(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    token = lock_gpu(app)
    try:
        with TestClient(app, base_url=BASE) as client:
            response = client.post("/api/v1/workstation/preview", json=fixture)
            assert response.status_code == 503 and response.json()["busy"] is True
    finally:
        app.state.engine_gate.release_owner(token)


def test_gate_unit_semantics():
    gate = EngineGate()
    with gate.preview_section():
        with pytest.raises(TimeoutError):
            gate.acquire_owner("job", timeout=0.05)
        assert gate.busy_reason() is None  # a timed-out waiter leaves no residue
    a, b = gate.acquire_owner("job-a"), gate.acquire_owner("job-b")  # owners are shared with each other
    assert gate.owner_count == 2
    with pytest.raises(EngineBusy) as busy:
        with gate.preview_section():
            pass
    assert busy.value.stage == "job-a"
    gate.release_owner(a); gate.release_owner(b)
    assert gate.try_enter_preview()
    gate.leave_preview()
    with pytest.raises(RuntimeError):
        gate.leave_preview()


def test_async_owner_does_not_block_the_event_loop():
    import asyncio
    gate = EngineGate()
    assert gate.try_enter_preview()
    ticks = 0

    async def scenario():
        nonlocal ticks

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        async def hold():
            async with gate.aowner("x"):
                return gate.owner_count

        task = asyncio.create_task(ticker())
        waiter = asyncio.create_task(hold())
        await asyncio.sleep(0.2)
        assert not waiter.done() and ticks > 5  # loop stayed responsive while the owner waited
        gate.leave_preview()
        held = await asyncio.wait_for(waiter, 5)
        task.cancel()
        return held

    assert asyncio.run(scenario()) == 1 and gate.owner_count == 0


# --------------------------------------------------------------------------- GET endpoints
def test_get_preview_grid_and_sequence_are_cache_addressable(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture)
    with TestClient(app, base_url=BASE) as client:
        posted = client.post("/api/preview", json=payload)
        digest = posted.headers["x-preview-config-hash"]
        media = posted.headers["x-preview-media-id"]
        got = client.get("/api/target/preview", params={"config": digest, "frame_index": 2, "media_id": media})
        assert got.status_code == 200 and got.headers["x-preview-cache"] == "HIT" and got.content == posted.content
        assert client.get("/api/target/preview", params={"config": digest, "frame_index": 2, "media_id": "f" * 32}).status_code == 409
        assert client.get("/api/target/preview", params={"config": "unknown-config"}).status_code == 422

        seq = client.get("/api/target/preview_seq", params={"config": digest, "start": 1, "count": 3}).json()
        assert [f["frame_index"] for f in seq["frames"]] == [1, 2, 3]
        assert [f["status"] for f in seq["frames"]] == ["ready"] * 3
        assert seq["frames"][1]["cached"] is True and seq["frames"][0]["cached"] is False
        assert client.get(seq["frames"][0]["url"].replace("/api/", "/api/v1/")).status_code == 200

        grid = client.get("/api/target/preview_grid", params={"config": digest, "rows": 2, "cols": 2})
        assert grid.status_code == 200 and decode(grid).shape[1] >= 2 * 96
        assert grid.headers["x-preview-pending"] == ""

        token = lock_gpu(app)
        try:  # fully cached manifest is still served; an uncached window is refused
            assert client.get("/api/target/preview_seq", params={"config": digest, "start": 1, "count": 3}).status_code == 200
            # cached so far: 0,1,2,3 (post/seq) and 5,7 (grid). Frame 4 and 6 are not.
            busy = client.get("/api/target/preview_seq", params={"config": digest, "start": 4, "count": 1})
            assert busy.status_code == 503 and busy.json()["retry_after_ms"] == 1000
            partial = client.get("/api/target/preview_seq", params={"config": digest, "start": 3, "count": 3}).json()
            assert [f["status"] for f in partial["frames"]] == ["ready", "busy", "ready"]
        finally:
            app.state.engine_gate.release_owner(token)


# --------------------------------------------------------------------------- payload contract
@pytest.mark.parametrize("extra,status", [
    ({"frame_index": None}, 422),                                   # neither frame selector
    ({"timestamp": .1}, 422),                                       # both selectors
    ({"restorer": "codeformer"}, 422),                              # no verified adapter, rejected up front
    ({"source_face_ids": ["alice"]}, 422),                          # explicit mappings + selectors
    ({"mask_strokes": [{"radius": .1, "points": [[2, 2]]}]}, 422),  # strokes are normalised 0..1
    ({"frontalization_weight": 1.5}, 422),
    ({"solo_node": "nonexistent"}, 422),
    ({"unknown_field": 1}, 422),
])
def test_invalid_payloads_are_rejected(tmp_path, monkeypatch, extra, status):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        assert client.post("/api/preview", json={**live_payload(fixture), **extra}).status_code == status


def test_unsupported_controls_are_reported_not_silently_dropped(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture, tps_expression_weight=.4, fidelity_weight=.9, frontalization_weight=.6,
                           occlusion={"erosion_px": 2, "feather_px": 8}, solo_node="matte")
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=payload)
        assert response.status_code == 200, response.text
        meta = json.loads(response.headers["x-preview-meta"])
        assert set(meta["unsupported"]) == {"fidelity_weight"}  # tps_expression_weight became a real control in Stage 5
        assert {"frontalization_weight", "tps_expression_weight"} <= set(meta["applied"]) and meta["solo_node"] == "matte"
        assert decode(response).ndim == 3  # the matte buffer, not the composite
        final = client.post("/api/preview", json={**payload, "solo_node": "final"})
        assert final.content != response.content


def test_target_selectors_resolve_against_the_inspection(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    base = {k: v for k, v in live_payload(fixture).items() if k != "face_mappings"}
    with TestClient(app, base_url=BASE) as client:
        ok = [{"source_face_ids": ["alice"], "target_face_id": "track_0000"},
              {"fsz_faceset_id": "alice", "all_faces": True},
              {"source_face_ids": ["alice"], "all_faces": True, "cluster_index": 0}]
        for selector in ok:
            assert client.post("/api/preview", json={**base, **selector}).status_code == 200, selector
        assert client.post("/api/preview", json={**base, "source_face_ids": ["alice"], "all_faces": True, "cluster_index": 5}).status_code == 422
        assert client.post("/api/preview", json={**base, "source_face_ids": ["alice"], "target_face_id": "track_0009"}).status_code == 422
        assert client.post("/api/preview", json={**base, "source_face_ids": ["alice"]}).status_code == 422  # no target chosen


def test_custom_mask_strokes_change_the_result(tmp_path, monkeypatch):
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    payload = live_payload(fixture)
    stroke = {"mode": "paint", "radius": .3, "points": [[.2, .5], [.5, .5], [.8, .5]]}
    with TestClient(app, base_url=BASE) as client:
        plain = client.post("/api/preview", json=payload)
        masked = client.post("/api/preview", json={**payload, "mask_strokes": [stroke]})
        assert plain.status_code == masked.status_code == 200, masked.text
        assert masked.headers["x-preview-config-hash"] != plain.headers["x-preview-config-hash"]
        assert "custom_mask" in json.loads(masked.headers["x-preview-meta"])["applied"]


def test_rasterize_strokes_paint_then_erase():
    paint = MaskStroke(mode="paint", radius=.2, points=((.5, .5),))
    erase = MaskStroke(mode="erase", radius=.05, points=((.5, .5),))
    mask = rasterize_strokes([paint], 100, 50, None)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(mask.split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == (50, 100) and decoded[25, 50] == 255 and decoded[0, 0] == 0
    carved = rasterize_strokes([erase], 100, 50, mask)
    after = cv2.imdecode(np.frombuffer(base64.b64decode(carved.split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert after[25, 50] == 0 and after[25, 57] == 255
    with pytest.raises(Exception):
        rasterize_strokes([paint], 64, 64, mask)  # base mask of a different size


# --------------------------------------------------------------------------- cache + blend maths
def test_lru_cache_evicts_by_entries_and_bytes():
    cache = PreviewCache(max_entries=2, max_bytes=100)
    entry = lambda n: CacheEntry(b"x" * n, "image/png", {})
    cache.put(("m", 1, "c"), entry(10)); cache.put(("m", 2, "c"), entry(10))
    assert cache.get(("m", 1, "c"))  # refresh 1, so 2 becomes the eviction candidate
    cache.put(("m", 3, "c"), entry(10))
    assert cache.get(("m", 2, "c")) is None and cache.get(("m", 1, "c")) and cache.get(("m", 3, "c"))
    cache.put(("m", 4, "c"), entry(95))
    assert len(cache) == 1 and cache.size_bytes == 95
    cache.put(("m", 4, "c"), entry(20))  # replacing an entry re-counts its bytes
    assert cache.size_bytes == 20


def test_identity_blend_math():
    e1, e2 = np.eye(512)[0], np.eye(512)[1]
    mid = blend_embeddings([e1, e2], [1, 1], method="slerp")
    assert np.allclose(mid, (e1 + e2) / np.sqrt(2), atol=1e-6)
    quarter = blend_embeddings([e1, e2], [3, 1], method="slerp")  # t = 0.25 along the geodesic
    assert np.allclose(quarter[:2], [np.cos(np.pi / 8), np.sin(np.pi / 8)], atol=1e-6)
    linear = blend_embeddings([e1, e2], [3, 1], method="linear")
    assert np.allclose(linear[:2], np.array([3, 1]) / np.sqrt(10), atol=1e-6)
    assert not np.allclose(linear, quarter)  # slerp and linear differ off-centre
    toward_2 = blend_embeddings([e1, e2], [1, 1], dials=[0, 1])  # full dial slides along the tangent plane to e2
    assert toward_2[1] > mid[1] and abs(np.linalg.norm(toward_2) - 1) < 1e-6
    away_from_2 = blend_embeddings([e1, e2], [1, 1], dials=[0, -1])
    assert away_from_2[1] < mid[1]
    with pytest.raises(ValueError):
        IdentityBlendSpec(source_face_ids=("a", "b"), weights=(1,))
    with pytest.raises(ValueError):
        IdentityBlendSpec(source_face_ids=("a", "a"))
    with pytest.raises(ValueError):
        IdentityBlendSpec(source_face_ids=("a", "b"), tangent_dials=(0, 2))


def test_multi_source_payload_builds_a_blended_identity(tmp_path, monkeypatch):
    import shutil
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    root = tmp_path / "roop_3d"
    shutil.copy(root / "outputs/facesets/alice.fsz", root / "outputs/facesets/bob.fsz")
    seen = {}

    class Recording(FixtureSession):
        def prepare_session(self, source, **kwargs):
            seen["canonical"] = np.asarray(source.canonical_embedding)
            seen["metadata"] = source.metadata

    monkeypatch.setattr(module, "ReferencePipeline", Recording)

    class Reader:
        def read(self, path, *, faceset_id):
            from roop_3d.tests.test_stage12_worker import _faceset
            document = _faceset(faceset_id)
            if faceset_id == "bob":
                vector = tuple([0.0, 1.0] + [0.0] * 510)
                view = document.views[0].model_copy(update={"embedding": vector})  # views must agree with the identity (Stage 0 guard)
                document = document.model_copy(update={"canonical_embedding": vector, "medoid_embedding": vector, "views": (view,)})
            return document

    from roop_3d.engine.pipeline_runner import MasterPipelineRunner as real_runner
    monkeypatch.setattr(module, "MasterPipelineRunner", lambda *a, **k: real_runner(*a, **k, faceset_reader=Reader()))
    payload = {k: v for k, v in live_payload(fixture).items() if k != "face_mappings"}
    payload.update(source_face_ids=["alice", "bob"], target_face_id="track_0000",
                   identity_blend={"method": "slerp", "source_face_ids": ["alice", "bob"], "weights": [1, 1]})
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=payload)
        assert response.status_code == 200, response.text
    assert np.allclose(seen["canonical"][:2], [2 ** -.5, 2 ** -.5], atol=1e-6)
    assert seen["metadata"]["identity_blend"]["method"] == "slerp"
    assert "identity_blend:slerp" in json.loads(response.headers["x-preview-meta"])["applied"]
