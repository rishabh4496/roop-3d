"""Stage 12 verification: accurate face selection from any frame and rank-preserved target locking.

Builds a test video from REAL photos of two different people (roop-ultimate facesets): frames 0-11 person A | person B,
12-23 B | A (left-to-right order swapped), 24-35 A in profile | B, 36-47 B at 3/4 | A at 3/4. Then, through the real FastAPI app:

* inspects a multi-face frame and a profile frame (embedding, kps_5, 68-pt 3D, pose, thumbnail, ids);
* checks the returned boxes are the exact native-pixel detector output (sub-pixel) and lie on the pasted photos;
* locks person A from frame 0 via ``POST /api/v1/target/lock-from-frame`` (reference bank grows across angles);
* confirms the cluster id of each person persists across frames where the left-to-right order swaps and on the profile;
* adds the profile as an angle sample via ``POST /api/v1/target/add-angle-sample``.

The temporary video lives in ``app/roop_3d/.runtime/media`` and is deleted afterwards. Exit 0 = pass.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import sys
import zipfile

import av
import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.core.settings import Settings  # noqa: E402
from roop_3d.server.app import create_app  # noqa: E402

FACESETS = ROOT.parent / "roop-ultimate/app/facesets"
VIDEO = "roop_3d/.runtime/media/_stage12_verify.mp4"
VIDEO_B = "roop_3d/.runtime/media/_stage12_verify_b.mp4"      # byte copy: a fresh cluster registry for the anchor-only phase
W, H, FPS, SEG = 1280, 720, 25, 12
PHOTO_H = 560


def check(condition: bool, message: str) -> None:
    if not condition:
        print("FAIL:", message)
        raise SystemExit(1)
    print("ok  :", message)


def fsz_images(name: str) -> list[np.ndarray]:
    path = Path(os.environ.get("ROOP3D_VERIFY_FACESETS", FACESETS)) / f"{name}.fsz"
    if not path.is_file():
        raise SystemExit(f"FAIL: faceset {path} not found (set ROOP3D_VERIFY_FACESETS)")
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))),
                       key=lambda s: [int(t) if t.isdigit() else t for t in __import__("re").split(r"(\d+)", s)])
        return [cv2.imdecode(np.frombuffer(archive.read(n), np.uint8), cv2.IMREAD_COLOR) for n in names]


def paste(canvas: np.ndarray, photo: np.ndarray, x0: int) -> list[int]:
    scale = PHOTO_H / photo.shape[0]
    resized = cv2.resize(photo, (int(round(photo.shape[1] * scale)), PHOTO_H), interpolation=cv2.INTER_AREA)
    y0 = (H - PHOTO_H) // 2
    canvas[y0:y0 + PHOTO_H, x0:x0 + resized.shape[1]] = resized
    return [x0, y0, x0 + resized.shape[1], y0 + PHOTO_H]


def build_video(path: Path) -> dict:
    a, b = fsz_images("harjot"), fsz_images("akansha")          # views are ordered profile, 3/4, frontal, 3/4, profile
    layouts = [("A_front", a[2], "B_front", b[2]), ("B_front", b[2], "A_front", a[2]),
               ("A_profile", a[0], "B_front", b[2]), ("B_34", b[1], "A_34", a[1])]
    regions = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.codec_context.gop_size = 6
        stream.codec_context.options = {"qscale": "2"}
        for left_name, left, right_name, right in layouts:
            canvas = np.full((H, W, 3), 128, np.uint8)
            lb = paste(canvas, left, 60)
            rb = paste(canvas, right, 700)
            regions.append({left_name: lb, right_name: rb})
            for _ in range(SEG):
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                    output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return {"layouts": regions}


def inside(box, region, slack=0) -> bool:
    return box[0] >= region[0] - slack and box[1] >= region[1] - slack and box[2] <= region[2] + slack and box[3] <= region[3] + slack


def main() -> int:
    video = APP / VIDEO
    info = build_video(video)
    try:
        app = create_app(Settings(), repo_root=APP, worker_enabled=False)
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            def faces(frame: int, detail: str = "full") -> dict:
                r = client.get("/api/v1/media/faces", params={"path": VIDEO, "frame_index": frame, "detail": detail})
                check(r.status_code == 200, f"GET /media/faces frame {frame} -> {r.status_code} {r.text[:160] if r.status_code != 200 else ''}")
                return r.json()

            # ---- 1. multi-face frame: fields + sub-pixel bbox alignment
            f0 = faces(0)
            check(len(f0["faces"]) == 2, f"frame 0 has two faces (got {len(f0['faces'])})")
            for face in f0["faces"]:
                check(all(k in face for k in ("face_id", "track_id", "cluster_id", "bbox", "bbox_norm", "kps_5", "kps_68_3d", "pose",
                                              "embedding", "crop_thumbnail_base64")), f"{face['face_id']}: every requested field is present")
                check(len(face["kps_5"]) == 5, f"{face['face_id']}: 5-point ArcFace landmarks")
                check(face["embedding"] is not None and len(face["embedding"]) == 512 and abs(np.linalg.norm(face["embedding"]) - 1) < 1e-3,
                      f"{face['face_id']}: 512-D L2-normalised embedding")
                thumb = cv2.imdecode(np.frombuffer(base64.b64decode(face["crop_thumbnail_base64"]), np.uint8), cv2.IMREAD_COLOR)
                check(thumb is not None and max(thumb.shape[:2]) <= 96, f"{face['face_id']}: crop thumbnail decodes")
                check(face["kps_68_3d"] is None or len(face["kps_68_3d"]) == 68, f"{face['face_id']}: kps_68_3d is 68 points or null ({face['kps_68_3d_status']})")
                check(set(face["pose"]) == {"yaw", "pitch", "roll"}, f"{face['face_id']}: pose {face['pose']}")
            frames = app.state.media.frames
            native = frames.read(VIDEO, frame_index=0)
            direct, _ = app.state.media.detector.detect_scored(native.image, 0.5)
            direct = sorted(direct, key=lambda item: float(item[0][0]))
            for face, (box, _, _) in zip(f0["faces"], direct):
                err = float(np.abs(np.asarray(face["bbox"]) - np.asarray(box)).max())
                check(err < 1e-2, f"{face['face_id']}: bbox equals the native-frame detector box (max error {err:.4f} px)")
                norm_err = float(np.abs(np.asarray(face["bbox_norm"]) * [W, H, W, H] - np.asarray(face["bbox"])).max())
                check(norm_err < 0.01, f"{face['face_id']}: normalised and pixel boxes agree to {norm_err:.4f} px")
            small = client.get("/api/v1/media/frame", params={"path": VIDEO, "frame_index": 0, "detect_faces": "true", "max_size": 320,
                                                              "format": "json"}).json()
            check(small["faces"][0]["bbox"] == f0["faces"][0]["bbox"], "boxes stay in native pixels when the preview image is downscaled to 320 px")
            regions = info["layouts"][0]
            left_face, right_face = f0["faces"]
            check(inside(left_face["bbox"], regions["A_front"]) and inside(right_face["bbox"], regions["B_front"]),
                  "frame 0 boxes lie on the pasted photos (left = A, right = B)")
            cluster_a, cluster_b = left_face["cluster_id"], right_face["cluster_id"]
            check(cluster_a is not None and cluster_b is not None and cluster_a != cluster_b,
                  f"two different people get two clusters (A={cluster_a}, B={cluster_b})")

            # ---- 2. profile frame
            f2 = faces(2 * SEG + 1)
            check(len(f2["faces"]) == 2, f"profile frame has both faces (got {len(f2['faces'])})")
            profile = f2["faces"][0]
            check(inside(profile["bbox"], info["layouts"][2]["A_profile"]), "profile face is boxed on the pasted profile photo")
            print(f"      profile face: yaw {profile['pose3d']['yaw']}, tier {profile['detection_tier']}, confidence {profile['confidence']}, "
                  f"68pt {profile['kps_68_3d_status']}")

            # ---- 3. lock person A from the frame-0 click
            cx = (left_face["bbox"][0] + left_face["bbox"][2]) / 2
            cy = (left_face["bbox"][1] + left_face["bbox"][3]) / 2
            r = client.post("/api/v1/target/lock-from-frame", json={"path": VIDEO, "frame_index": 0, "point": [cx, cy], "sample_count": 16, "wait": True})
            check(r.status_code == 201, f"POST lock-from-frame -> {r.status_code} {r.text[:200] if r.status_code != 201 else ''}")
            lock = r.json()
            check(lock["selected_face_index"] == 0 and lock["label"] == "Target #1", "clicking the left face locks it as Target #1")
            check(lock["scan"]["state"] == "done" and lock["scan"]["sampled"] >= 8, f"keyframe scan finished ({lock['scan']})")
            yaws = [e["yaw"] for e in lock["reference_bank"]]
            check(lock["bank_size"] >= 3 and max(yaws) - min(yaws) >= 40,
                  f"reference bank holds several angles: yaw {min(yaws):.0f}..{max(yaws):.0f} over {lock['bank_size']} entries")
            check(lock["cluster_id"] == cluster_a, "the lock is the cluster frame 0 already assigned to person A")

            # ---- 4. rank-preserved cluster ids when the left/right order swaps
            f1 = faces(SEG + 1)
            ids_1 = {f["face_index"]: f["cluster_id"] for f in f1["faces"]}
            check(ids_1 == {0: cluster_b, 1: cluster_a}, f"order swapped (B left, A right) but ids follow the person: {ids_1}")
            m1 = client.get(f"/api/v1/target/{lock['lock_id']}/match", params={"frame_index": SEG + 1}).json()
            m0 = client.get(f"/api/v1/target/{lock['lock_id']}/match", params={"frame_index": 1}).json()
            check(m0["target_face_index"] == 0 and m1["target_face_index"] == 1,
                  f"locked target is face #{m0['target_face_index']} in frame 1 and face #{m1['target_face_index']} after the swap")
            m2 = client.get(f"/api/v1/target/{lock['lock_id']}/match", params={"frame_index": 2 * SEG + 1}).json()
            check(m2["target_face_index"] == 0, f"locked target is still found when turned to profile (face #{m2['target_face_index']})")
            reason = next(f for f in m2["faces"] if f["face_index"] == 0)["cluster_match"]
            print(f"      profile association: {reason}")
            m3 = client.get(f"/api/v1/target/{lock['lock_id']}/match", params={"frame_index": 3 * SEG + 1}).json()
            check(m3["target_face_index"] == 1, f"locked target found at 3/4 view on the right (face #{m3['target_face_index']})")
            other = [f for f in m3["faces"] if f["face_index"] == 0][0]
            check(other["cluster_id"] == cluster_b and not other["is_target"], "the other person is never mistaken for the target")

            # ---- 5. add an angle sample
            r = client.post("/api/v1/target/add-angle-sample", json={"lock_id": lock["lock_id"], "frame_index": 2 * SEG + 1, "face_index": 0})
            check(r.status_code == 200, f"POST add-angle-sample -> {r.status_code} {r.text[:200] if r.status_code != 200 else ''}")
            added = r.json()
            check(any(e["source"] == "user" for e in added["reference_bank"]), "the profile is in the bank as a user sample")
            bad = client.post("/api/v1/target/add-angle-sample", json={"lock_id": lock["lock_id"], "frame_index": 0, "face_index": 1})
            check(bad.status_code == 409 and bad.json()["error"]["code"] == "ANGLE_SAMPLE_DISSIMILAR", "adding the OTHER person as an angle of Target #1 is refused (409)")
            miss = client.post("/api/v1/target/lock-from-frame", json={"path": VIDEO, "frame_index": 0, "point": [5.0, 5.0], "sample_count": 0})
            check(miss.status_code == 404, "a click that misses every face is rejected (404)")
            # ---- 6. anchor-only lock: NOTHING of the profile / 3/4 views has been seen, so this is real pose bridging
            (APP / VIDEO_B).write_bytes(video.read_bytes())
            f0b = client.get("/api/v1/media/faces", params={"path": VIDEO_B, "frame_index": 0, "detail": "basic"}).json()
            lb = f0b["faces"][0]
            r = client.post("/api/v1/target/lock-from-frame", json={
                "path": VIDEO_B, "frame_index": 0, "point": [(lb["bbox"][0] + lb["bbox"][2]) / 2, (lb["bbox"][1] + lb["bbox"][3]) / 2],
                "sample_count": 0})
            check(r.status_code == 201 and r.json()["bank_size"] == 1, "anchor-only lock has a single frontal reference")
            anchor_lock = r.json()["lock_id"]
            for label, frame, expected in (("profile (yaw ~79)", 2 * SEG + 1, 0), ("3/4 view, right side", 3 * SEG + 1, 1)):
                m = client.get(f"/api/v1/target/{anchor_lock}/match", params={"frame_index": frame}).json()
                hit = next((f for f in m["faces"] if f["face_index"] == expected), None)
                print(f"      anchor-only, {label}: target face #{m['target_face_index']}, association {hit['cluster_match'] if hit else None}")
                check(m["target_face_index"] == expected,
                      f"anchor-only lock (frontal reference only) still finds the person at {label}")
        print("PASS")
        return 0
    finally:
        video.unlink(missing_ok=True)
        (APP / VIDEO_B).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
