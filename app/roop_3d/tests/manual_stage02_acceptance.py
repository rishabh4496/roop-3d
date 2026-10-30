"""Explicit real-model/media acceptance campaign, separate from offline fixtures.

Run from app/: python -B -m roop_3d.tests.manual_stage02_acceptance
Reads the downloaded OpenCV sample only from this app's ignored runtime tree.
It does not fetch media, import legacy code, or execute a rendering worker.
"""
from __future__ import annotations

import hashlib
import json

import av
import cv2
import numpy as np
from fastapi.testclient import TestClient

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json, owned
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app


def main() -> None:
    root = application_root()
    fixture = owned(root, "roop_3d/.runtime/acceptance/opencv-lena.jpg")
    image = cv2.imread(str(fixture))
    assert image is not None, "Download the documented official OpenCV sample first"
    campaign_root = owned(root, "roop_3d/.runtime/acceptance/campaign")
    campaign_root.mkdir(parents=True, exist_ok=True)
    video = owned(campaign_root, "roop_3d/.runtime/media/two-targets.mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video), "w") as output:
        stream = output.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.codec_context.gop_size = 1
        for index in range(5):
            canvas = np.hstack((image, image))
            canvas = np.roll(canvas, index * 2, axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    for name in ("alice", "bob"):
        atomic_json(campaign_root, f"roop_3d/.runtime/facesets/{name}.json",
                    {"faceset_id": name, "name": name.title(), "item_count": 1})
    app = create_app(Settings(), repo_root=campaign_root)
    target = video.relative_to(campaign_root).as_posix()
    # Prior explicit campaign runs may be replayed; use a digest-bound key.
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        probe = client.post("/api/v1/media/probe", json={"path": target})
        assert probe.status_code == 200, probe.text
        assert probe.json()["width"] == 1024
        inspection = client.post("/api/v1/media/inspect-faces", json={"path": target})
        assert inspection.status_code == 200, inspection.text
        data = inspection.json()
        assert len(data["tracks"]) == 2, data
        assert all(item["sampled_detections"] == 5 for item in data["tracks"])
        assert all(len(item["thumbnail_base64"]) > 100 for item in data["tracks"])
        preview = client.get("/api/v1/media/preview-frame", params={"path": target, "frame_index": 2,
                                                                   "comparison_path": target})
        assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg"
        decoded = cv2.imdecode(np.frombuffer(preview.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (512, 2048, 3)
        payload = {"target_path": target, "inspection_id": data["inspection_id"],
                   "face_mappings": [{"track_id": item["track_id"], "faceset_id": name}
                                     for item, name in zip(data["tracks"], ("alice", "bob"))]}
        response = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": data["inspection_id"]})
        assert response.status_code == 201, response.text
        job_id = response.json()["job_id"]
        assert client.post(f"/api/v1/jobs/{job_id}/pause").json()["state"] == "paused"
        assert client.post(f"/api/v1/jobs/{job_id}/resume").json()["state"] == "queued"
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").json()["state"] == "cancelled"
        assert client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "cancelled"
    report = {"status": "passed", "assertions": "Real SCRFD -> 2 spatial tracks -> 2 FaceSet mappings -> durable job controls",
              "fixture_source": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "detector_sha256": data["detector_sha256"], "track_count": len(data["tracks"]),
              "sampled_frames": data["sampled_frames"], "preview_shape": list(decoded.shape),
              "rendering_worker_available": False, "gpu_performance_qualified": False}
    atomic_json(root, "roop_3d/.runtime/acceptance/STAGE02_REAL_MEDIA.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
