"""Exercise single-view and multiview faceset reconstruction without external weights."""
from __future__ import annotations

import hashlib
import base64
from pathlib import Path
import sys

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.faceset.schemas import FaceSetDocument, FaceView, QualityMetrics
from roop_3d.reconstruction.faceset_3d import TOTAL_BINS, reconstruct_faceset
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, euler_to_rotation_matrix, rotation_matrix_to_rodrigues
from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks


def _document(poses: list[tuple[float, float, float]], faceset_id: str) -> FaceSetDocument:
    size = 512
    camera = CameraIntrinsics.from_fov(55, size, size)
    model_points = get_canonical_3d_landmarks(5).astype(np.float64)
    embedding = tuple([1.0] + [0.0] * 511)
    views = []
    for index, (yaw, pitch, roll) in enumerate(poses):
        rng = np.random.default_rng(7300 + index)
        image = np.clip(rng.normal(92, 3, (size, size, 3)), 0, 255).astype(np.uint8)
        shift_x, shift_y = int(round(np.sin(np.radians(yaw)) * 22)), int(round(-np.sin(np.radians(pitch)) * 15))
        center = (size // 2 + shift_x, size // 2 + shift_y)
        cv2.ellipse(image, center, (132, 176), roll, 0, 360, (188, 205, 229), -1, cv2.LINE_AA)
        for eye_x in (-29, 29):
            eye = (center[0] + eye_x, center[1] - 15)
            cv2.ellipse(image, eye, (13, 5), roll, 0, 360, (45, 46, 54), -1, cv2.LINE_AA)
            cv2.line(image, (eye[0] - 15, eye[1] - 13), (eye[0] + 15, eye[1] - 15), (75, 72, 77), 4, cv2.LINE_AA)
        cv2.line(image, (center[0], center[1] - 3), (center[0] - 7, center[1] + 28), (137, 159, 190), 5, cv2.LINE_AA)
        cv2.ellipse(image, (center[0], center[1] + 48), (27, 7), roll, 0, 360, (75, 67, 123), -1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        image_bytes = encoded.tobytes()
        matrix = euler_to_rotation_matrix(yaw, pitch, roll)
        rvec = rotation_matrix_to_rodrigues(matrix)
        projected, _ = cv2.projectPoints(model_points, rvec, np.asarray([0.0, 0.0, .58]), camera.matrix(), None)
        landmarks = tuple((float(x), float(y)) for x, y in projected.reshape(-1, 2))
        views.append(FaceView(index=index, image_bytes=image_bytes, embedding=embedding, landmarks=landmarks,
            bbox=(96.0, 48.0, 416.0, 464.0), yaw=max(-90, min(90, yaw)), pitch=max(-60, min(60, -pitch)),
            roll=roll, expression={}, quality=QualityMetrics(sharpness=.9, detector_confidence=.99, score=.9,
            laplacian_variance=450, blur=.1, face_pixels=160000), pose_bin=f"sample_{index}", landmarks_trusted=True))
    digest = hashlib.sha256(b"".join(view.image_bytes for view in views)).hexdigest()
    return FaceSetDocument(faceset_id=faceset_id, name="Synthetic reconstruction sample", source_sha256=digest,
        archive_format="v2_standard", canonical_embedding=embedding, medoid_embedding=embedding,
        views=tuple(views), pose_bins={}, metadata={"verification": "synthetic"})


def _assert_payload(payload: dict, expected_views: int):
    assert payload["status"] == "ready"
    assert payload["observed_views"] == expected_views and payload["observed_views"] >= 1
    assert payload["sphere_coverage_total"] == TOTAL_BINS == 96
    assert payload["sphere_coverage_count"] >= 1
    assert len(payload["observed_pose_points"]) == expected_views
    assert payload["uv_texture_data_url"].startswith("data:image/png;base64,")
    for key in ("mesh_2_5d", "mesh_full_3d"):
        mesh = payload[key]
        assert len(mesh["vertices"]) > 0
        assert len(mesh["uvs"]) == len(mesh["vertices"])
        assert len(mesh["indices"]) > 0
        assert len(mesh["normals"]) == len(mesh["vertices"])
        png = base64.b64decode(mesh["texture_base64"])
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        atlas = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        assert atlas is not None and atlas.shape == (512, 512, 4)


def _verify_http_contract(document: FaceSetDocument):
    from roop_3d.server import faceset_3d as routes

    class FakeFaceExtract:
        def __init__(self):
            self.requests = []

        def _media_file(self, relative):
            return APP / relative, True

        def extract(self, value):
            self.requests.append(value)
            if value.destination == "source_gallery":
                return {"faceset": {"faceset_id": document.faceset_id}}
            return {"active_source": {"faceset_id": document.faceset_id}}

    app = FastAPI()
    app.include_router(routes.router)
    app.state.root = APP
    app.state.face_extract = FakeFaceExtract()
    with patch.object(routes, "_read_archive", return_value=document):
        with TestClient(app) as client:
            response = client.post("/api/v1/faceset/reconstruct-3d",
                                   json={"source_image_paths": ["roop_3d/.runtime/media/front.png",
                                                                 "roop_3d/.runtime/media/angle.png"]})
            assert response.status_code == 200, response.text
            _assert_payload(response.json(), len(document.views))
            assert [request.destination for request in app.state.face_extract.requests] == ["source_gallery", "active_faceset"]
            state = client.get(f"/api/v1/faceset/3d-state?faceset_id={document.faceset_id}")
            assert state.status_code == 200, state.text
            assert state.json()["sphere_coverage_count"] == response.json()["sphere_coverage_count"]


def main() -> int:
    single = reconstruct_faceset(_document([(0, 0, 0)], "stage3_single"))
    _assert_payload(single, 1)
    multi = reconstruct_faceset(_document([(0, 0, 0), (-35, 10, -3), (38, -8, 4), (10, 25, 1)], "stage3_multi"))
    _assert_payload(multi, 4)
    _verify_http_contract(_document([(0, 0, 0)], "stage3_http"))
    assert multi["sphere_coverage_count"] >= single["sphere_coverage_count"]
    print({"state": "PASS", "single": {"views": single["observed_views"], "bins": single["sphere_coverage_count"]},
           "multi": {"views": multi["observed_views"], "bins": multi["sphere_coverage_count"]},
           "single_mesh_vertices": len(single["mesh_full_3d"]["vertices"]),
           "multi_mesh_vertices": len(multi["mesh_full_3d"]["vertices"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
