"""Stage 11 verification: MakeHuman (CC0) head deformed onto a subject and textured from a single photo / a .fsz faceset.

Loads (1) a single photo and (2) a multi-angle ``.fsz`` archive through ``roop_3d.faceset_loader`` and through the real
``POST /api/v1/faceset/reconstruct-3d`` + ``GET /api/v1/faceset/3d-state`` routes, then checks that

* every sample reports embedding / landmarks / Euler pose / aligned 512x512 crop / 96-bin and 3x3(+roll) placement;
* the MakeHuman vertices moved away from the neutral template (measured, in millimetres);
* a 512x512 RGBA UV texture data URL with a hole-free footprint is returned and ``sphere_coverage_count`` matches the
  distinct populated bins.

The subject is a real multi-angle archive (``ROOP3D_VERIFY_FSZ`` or roop-ultimate's ``facesets/harjot.fsz``). Exit 0 = pass.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.faceset_loader import LandmarkEngine, load_faceset  # noqa: E402
from roop_3d.reconstruction.makehuman_template import makehuman_template  # noqa: E402
from roop_3d.server import faceset_3d as routes  # noqa: E402


def find_fsz() -> Path:
    candidates = [Path(p) for p in [os.environ.get("ROOP3D_VERIFY_FSZ", "")] if p]
    candidates += [ROOT.parent / "roop-ultimate/app/facesets/harjot.fsz"]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("FAIL: no multi-angle .fsz found (set ROOP3D_VERIFY_FSZ)")


def check(condition: bool, message: str) -> None:
    if not condition:
        print("FAIL:", message)
        raise SystemExit(1)
    print("ok  :", message)


def decode_texture(data_url: str) -> np.ndarray:
    check(data_url.startswith("data:image/png;base64,"), "uv_texture_data_url is a PNG data URL")
    image = cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    check(image is not None and image.shape == (512, 512, 4), f"UV texture decodes as 512x512 RGBA (got {None if image is None else image.shape})")
    return image


def verify_payload(label: str, payload: dict, expected_views: int, min_mean_mm: float) -> None:
    mesh = payload["mesh_full_3d"]
    base = makehuman_template().vertices.astype(np.float64)
    vertices = np.asarray(mesh["vertices"], np.float64)
    check(vertices.shape == base.shape, f"{label}: MakeHuman topology kept ({len(vertices)} vertices)")
    deviation = np.linalg.norm(vertices - base, axis=1) * 1000.0
    check(deviation.mean() > min_mean_mm and deviation.max() > 3 * min_mean_mm,
          f"{label}: vertices deformed away from the neutral baseline (mean {deviation.mean():.2f} mm, p95 "
          f"{np.percentile(deviation, 95):.2f} mm, max {deviation.max():.2f} mm)")
    normals = np.asarray(mesh["normals"], np.float64)
    check(normals.shape == vertices.shape and np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3),
          f"{label}: per-vertex normals recomputed and unit length")
    report = mesh["deformation"]
    check(report["template"] == "makehuman", f"{label}: template is MakeHuman (layout {report['landmark_layout']}, "
                                              f"{report['reconstruction_level']})")
    texture = decode_texture(payload["uv_texture_data_url"])
    footprint = texture[..., 3] == 255
    black = footprint & (texture[..., :3].max(axis=2) < 4)
    check(footprint.mean() > 0.2, f"{label}: texture footprint covers {footprint.mean():.0%} of the atlas")
    check(black.sum() / footprint.sum() < 0.02, f"{label}: no black holes on the mesh footprint ({black.sum() / footprint.sum():.2%} near-black)")
    completion = mesh["texture_completion"]
    print(f"      texture sources: observed {completion['observed_ratio']:.0%}, symmetry {completion['symmetry_ratio']:.0%}, "
          f"harmonic skin-tone {completion['harmonic_ratio']:.0%}")
    check(payload["observed_views"] == expected_views, f"{label}: observed_views == {expected_views}")
    check(payload["sphere_coverage_total"] == 96, f"{label}: 96-bin sphere")
    check(1 <= payload["sphere_coverage_count"] <= expected_views, f"{label}: sphere_coverage_count = {payload['sphere_coverage_count']}")
    check(len(payload["observed_pose_points"]) == expected_views, f"{label}: one glowing sphere marker per observed angle")
    check(len({p["bin_index"] for p in payload["observed_pose_points"]}) == payload["sphere_coverage_count"],
          f"{label}: coverage count equals the number of distinct populated bins")
    check(len(payload["mesh_2_5d"]["vertices"]) > 0, f"{label}: 2.5D relief mesh present")


def verify_samples(loaded, label: str) -> None:
    for sample in loaded.samples:
        check(abs(np.linalg.norm(sample.normed_embedding) - 1.0) < 1e-4 and sample.normed_embedding.shape == (512,),
              f"{label} view {sample.view_index}: 512-D normed embedding")
        check(sample.crop_rgb_512.shape == (512, 512, 3), f"{label} view {sample.view_index}: aligned 512x512 RGB crop")
        check(0 <= sample.sphere_bin < 96 and len(sample.matrix_index) == 3,
              f"{label} view {sample.view_index}: bin {sample.sphere_bin}, 3x3 cell {sample.matrix_cell}, roll axis {sample.matrix_index[2]}"
              f" (yaw {sample.yaw:.1f}, pitch {sample.pitch:.1f}, roll {sample.roll:.1f}; {sample.landmark_source})")
    check(loaded.sphere_coverage_count == len(loaded.sphere_bins), f"{label}: loader coverage matches its bin map")


def main() -> int:
    fsz = find_fsz()
    work = Path(tempfile.mkdtemp(prefix="stage11_"))
    try:
        (work / "facesets").mkdir()
        (work / "roop_3d/.runtime/cache").mkdir(parents=True)
        shutil.copy(fsz, work / "facesets/multi.fsz")
        engine = LandmarkEngine(APP)
        print(f"landmark models: 1k3d68={engine.has_68} 2d106det={engine.has_106}")

        multi = load_faceset(work / "facesets/multi.fsz", root=APP, cache_root=work / "roop_3d/.runtime/cache", engine=engine)
        verify_samples(multi, "multi")
        check(multi.sphere_coverage_count >= 3, f"multi-angle archive populates {multi.sphere_coverage_count} sphere bins")

        # single photo: the most frontal view, as a loose image file and as a one-image archive
        frontal = min(multi.samples, key=lambda s: abs(s.yaw) + abs(s.pitch)).view_index
        view = next(v for v in multi.document.views if v.index == frontal)
        photo = work / "photo.png"
        photo.write_bytes(view.image_bytes)
        single = load_faceset([photo], root=APP, cache_root=work / "roop_3d/.runtime/cache", engine=engine)
        verify_samples(single, "single")
        check(len(single.samples) == 1, "single photo yields one sample")
        with zipfile.ZipFile(work / "facesets/single.fsz", "w") as archive:
            archive.writestr("0.png", view.image_bytes)

        app = FastAPI()
        app.include_router(routes.router)
        app.state.root = work
        app.state.models = None
        app.state.landmark_engine = engine     # the temp root cannot locate the models itself
        with TestClient(app) as client:
            r = client.post("/api/v1/faceset/reconstruct-3d", json={"faceset_path": "facesets/multi.fsz"})
            check(r.status_code == 200, f"POST reconstruct-3d (multi-angle .fsz) -> {r.status_code} {r.text[:200] if r.status_code != 200 else ''}")
            payload = r.json()
            verify_payload("multi .fsz", payload, len(multi.samples), min_mean_mm=0.5)
            check("faceset_samples" in payload and len(payload["faceset_samples"]["samples"]) == len(multi.samples),
                  "payload reports per-sample landmark/pose/bin data")
            state = client.get("/api/v1/faceset/3d-state", params={"faceset_id": "multi"})
            check(state.status_code == 200, "GET /api/v1/faceset/3d-state -> 200")
            body = state.json()
            decode_texture(body["uv_texture_data_url"])
            check(body["sphere_coverage_count"] == payload["sphere_coverage_count"], "3d-state returns the same sphere_coverage_count")

            r = client.post("/api/v1/faceset/reconstruct-3d", json={"faceset_path": "facesets/single.fsz"})
            check(r.status_code == 200, f"POST reconstruct-3d (single image) -> {r.status_code} {r.text[:200] if r.status_code != 200 else ''}")
            verify_payload("single image", r.json(), 1, min_mean_mm=0.05)
            state = client.get("/api/v1/faceset/3d-state", params={"faceset_id": "single"})
            check(state.status_code == 200 and state.json()["sphere_coverage_count"] == 1,
                  "3d-state for the single image reports sphere_coverage_count == 1")
        print("PASS")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
