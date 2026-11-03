"""Stage 7 identity-panel API: view thumbnails and the Stage 3 identity asset (status / explicit build / viewer mesh).

The synthetic FaceSets come from the ground-truth world in ``identity_world`` (real rendered images, real landmarks,
real reconstruction code: nothing about the asset is mocked). The asset store is redirected to ``tmp_path`` so the
tests never touch the checkout's runtime state. A second test runs the same routes against a real local FaceSet when
one is present.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.settings import Settings
from roop_3d.reconstruction.identity_asset import IdentityAssetStore
from roop_3d.server import identity_assets
from roop_3d.server.app import create_app
from roop_3d.tests import identity_world as W

BASE = "http://127.0.0.1:8030"


def _register(tmp_path: Path, document) -> None:
    runtime = tmp_path / "roop_3d/.runtime/facesets"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / f"{document.faceset_id}.json").write_text(json.dumps({"faceset_id": document.faceset_id, "name": document.name,
                                                                      "item_count": len(document.views), "schema_version": 1}))
    (runtime / f"{document.faceset_id}.fsz").write_bytes(b"archive placeholder: the reader is redirected below")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    documents = {}

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, archive, faceset_id):
            return documents[faceset_id]

    store_root = tmp_path / "identity_store"
    monkeypatch.setattr(identity_assets, "FaceSetReader", Reader)
    monkeypatch.setattr(identity_assets, "_store", lambda: IdentityAssetStore(root=store_root))
    app = create_app(Settings(), repo_root=tmp_path)

    def add(document):
        documents[document.faceset_id] = document
        _register(tmp_path, document)
        return document.faceset_id

    return app, add, store_root


def test_view_image_is_bounded_jpeg_without_embeddings(world):
    app, add, _ = world
    faceset_id = add(W.full_set(W.make_subject("thumbs", W.LAYOUT_68, 81), (0, -40, 40)))
    with TestClient(app, base_url=BASE) as client:
        response = client.get(f"/api/v1/facesets/{faceset_id}/views/1/image?size=96")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/jpeg"
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and max(image.shape[:2]) == 96
        assert client.get(f"/api/v1/facesets/{faceset_id}/views/99/image").status_code == 404
        assert client.get(f"/api/v1/facesets/{faceset_id}/views/0/image?size=4096").status_code == 422
        assert client.get("/api/v1/facesets/bad.id/views/0/image").status_code in (404, 422)
        assert client.get("/api/v1/facesets/missing/views/0/image").status_code == 404


def test_status_never_builds_then_explicit_build_is_reused_and_meshed(world):
    app, add, store_root = world
    faceset_id = add(W.full_set(W.make_subject("panel", W.LAYOUT_68, 82), (0, -40, 40, -72, 72)))
    with TestClient(app, base_url=BASE) as client:
        status = client.get(f"/api/v1/facesets/{faceset_id}/identity-asset").json()
        assert status["state"] == "not_built" and status["asset"] is None
        assert status["coverage"]["view_count"] == 5 and status["coverage"]["readiness"] in {"partial_coverage", "full_coverage"}
        assert not store_root.exists() or not any(store_root.rglob("manifest.json")), "GET must never build"
        assert client.get(f"/api/v1/facesets/{faceset_id}/identity-asset/mesh").status_code == 404

        built = client.post(f"/api/v1/facesets/{faceset_id}/identity-asset")
        assert built.status_code == 200, built.text
        built = built.json()
        assert built["state"] == "built" and built["reused"] is False
        asset = built["asset"]
        assert asset["build_key"] == built["build_key"] == status["build_key"]
        assert asset["landmark_layout"] and asset["reconstruction_level"]
        assert asset["vertex_count"] > 100 and asset["face_count"] > 100 and asset["sources_used"] >= 3
        text = json.dumps(built)  # capability prose may mention embeddings; no embedding VALUES may be returned
        assert '"embedding":' not in text and "canonical_embedding" not in text and "medoid_embedding" not in text

        again = client.post(f"/api/v1/facesets/{faceset_id}/identity-asset").json()
        assert again["reused"] is True and again["asset"]["build_key"] == asset["build_key"]
        assert client.get(f"/api/v1/facesets/{faceset_id}/identity-asset").json()["state"] == "built"

        mesh = client.get(f"/api/v1/facesets/{faceset_id}/identity-asset/mesh").json()
        assert mesh["method"] == "identity_asset" and mesh["coordinate_system"] == "opengl"
        vertices, faces = np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
        assert vertices.shape == (asset["vertex_count"], 3) and faces.shape == (asset["face_count"], 3)
        assert faces.min() >= 0 and faces.max() < len(vertices)
        assert len(mesh["uvs"]) == len(vertices)
        png = base64.b64decode(mesh["texture"].split(",", 1)[1])
        atlas = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert list(atlas.shape[1::-1]) == asset["texture_size"]
        # glTF frame is y-up: the forehead (largest y) is above the chin, the opposite of the stored y-down mesh.
        stored = IdentityAssetStore(root=store_root).load(faceset_id, asset["build_key"])
        assert np.allclose(vertices[:, 1], -stored.vertices[:, 1], atol=1e-5)


def test_build_conflict_and_unknown_faceset(world):
    app, add, _ = world
    faceset_id = add(W.full_set(W.make_subject("lock", W.LAYOUT_5, 83), (0, -40, 40)))
    with TestClient(app, base_url=BASE) as client:
        lock = identity_assets._BUILD_LOCKS.setdefault(faceset_id, identity_assets.threading.Lock())
        assert lock.acquire(blocking=False)
        try:
            response = client.post(f"/api/v1/facesets/{faceset_id}/identity-asset")
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "IDENTITY_ASSET_BUILD_IN_PROGRESS"
        finally:
            lock.release()
        assert client.get("/api/v1/facesets/nope/identity-asset").status_code == 404
        assert client.post("/api/v1/facesets/nope/identity-asset").status_code == 404


REPO = Path(__file__).resolve().parents[2]
REAL_FACESETS = [path.stem for path in sorted((REPO / "roop_3d/.runtime/facesets").glob("*.json"))
                 if any((REPO / f"roop_3d/{base}/facesets/{path.stem}.fsz").is_file() for base in ("outputs", ".runtime"))]


@pytest.mark.skipif(not REAL_FACESETS, reason="no local FaceSet archive in this checkout")
def test_real_local_faceset_views_status_and_build(tmp_path, monkeypatch):
    """Real archive read by the real reader; the asset is built into tmp_path, never into the checkout's store."""
    monkeypatch.setattr(identity_assets, "_store", lambda: IdentityAssetStore(root=tmp_path / "store"))
    app = create_app(Settings(), repo_root=REPO)
    with TestClient(app, base_url=BASE) as client:
        faceset_id = max(REAL_FACESETS, key=lambda name: json.loads((REPO / f"roop_3d/.runtime/facesets/{name}.json").read_text("utf-8")).get("item_count", 0))
        views = client.get(f"/api/v1/facesets/{faceset_id}/views").json()
        assert views, "real FaceSet has views"
        image = client.get(f"/api/v1/facesets/{faceset_id}/views/{views[0]['index']}/image")
        assert image.status_code == 200 and image.content[:2] == b"\xff\xd8"
        status = client.get(f"/api/v1/facesets/{faceset_id}/identity-asset").json()
        assert status["coverage"]["view_count"] == len(views)
        built = client.post(f"/api/v1/facesets/{faceset_id}/identity-asset")
        if built.status_code == 200:
            asset = built.json()["asset"]
            mesh = client.get(f"/api/v1/facesets/{faceset_id}/identity-asset/mesh").json()
            assert len(mesh["vertices"]) == asset["vertex_count"]
        else:  # a real set that cannot support an asset must say why (e.g. inconsistent identity), never fake one
            assert built.status_code in (409, 422), built.text
            assert built.json()["error"]["code"]


# --------------------------------------------------------------------------- live preview carries the 3D / occlusion modes
def test_live_preview_carries_render_and_occlusion_modes(tmp_path, monkeypatch):
    """Stage 7 'preview without a full render': the 3D render mode reaches session preparation and inference, and the
    occlusion mode reaches the compositor options, exactly as a queued job's workstation_parameters would."""
    from roop_3d.server import workstation_preview as preview_module
    from roop_3d.tests.test_remediation_stage4 import LandmarkSession, live_payload
    from roop_3d.tests.test_workstation_preview import preview_fixture

    class Recording(LandmarkSession):
        prepared, inferred = [], []

        def prepare_session(self, *args, **kwargs):
            Recording.prepared.append(kwargs)
            return super().prepare_session(*args, **kwargs)

        def infer_frame(self, image, **kwargs):
            Recording.inferred.append(kwargs)
            return super().infer_frame(image, **kwargs)

    app, fixture = preview_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(preview_module, "ReferencePipeline", Recording)
    with TestClient(app, base_url=BASE) as client:
        base = live_payload(fixture)
        response = client.post("/api/preview", json=base)
        assert response.status_code == 200, response.text
        assert "render_mode" not in Recording.prepared[-1] and "render_mode" not in Recording.inferred[-1]
        response = client.post("/api/preview", json={**base, "render_mode": "pose_3d", "occlusion_mode": "robust"})
        assert response.status_code == 200, response.text
        assert Recording.prepared[-1]["render_mode"] == "pose_3d"
        assert Recording.inferred[-1]["render_mode"] == "pose_3d"
        meta_old = json.loads(client.post("/api/preview", json=base).headers["x-preview-meta"])
        meta_new = json.loads(client.post("/api/preview", json={**base, "render_mode": "pose_3d", "occlusion_mode": "robust"}).headers["x-preview-meta"])
        assert meta_old["swap_config_hash"] != meta_new["swap_config_hash"], "different modes must not share a cache entry"
        assert client.post("/api/preview", json={**base, "render_mode": "flame_magic"}).status_code == 422
        assert client.post("/api/preview", json={**base, "occlusion_mode": "maybe"}).status_code == 422


def test_live_preview_cancelled_returns_204(tmp_path, monkeypatch):
    from roop_3d.server import live_preview as lp_module
    from roop_3d.tests.test_remediation_stage4 import live_payload
    from roop_3d.tests.test_workstation_preview import preview_fixture

    app, fixture = preview_fixture(tmp_path, monkeypatch)
    base = live_payload(fixture)

    async def cancelling_frames(*args, **kwargs):
        raise lp_module.PreviewCancelled("Client disconnected")

    monkeypatch.setattr(app.state.live_previews, "frames", cancelling_frames)
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/api/preview", json=base)
        assert response.status_code == 204
        assert response.headers.get("X-Preview-Status") == "cancelled"
