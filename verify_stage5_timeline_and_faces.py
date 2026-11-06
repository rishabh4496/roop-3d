"""Stage 5 verification: /media/info, /media/filmstrip and /media/frame?detect_faces=true on a REAL video with the REAL detector.

Uses a local test clip from `app/roop_3d/.runtime/media` (copied into a scratch root, so nothing in the runtime is touched)
and the production `MediaService` + `VideoFrameService` + router, over HTTP (FastAPI TestClient). Checks:

* info: width, height, fps, total_frames, duration, against PyAV's own reading of the file;
* filmstrip: 16 ascending thumbnails across the whole clip, decodable JPEGs, fast;
* frame: the frame returned for index N *is* frame N (compared with the older frame-replay path), including indices >= 10 000
  that the replay path refuses; the second request is a cache hit;
* faces: boxes inside the frame, five keypoints inside (or at the edge of) their box, confidence in (0, 1], both pose frames,
  left-to-right ranks, and the boxes agree (IoU) with an independent run of the detector on the replay path's frame;
* scrubbing speed: sequential and random access, one container opened for the whole run;
* errors: out of range -> 416, no selector / bad path -> 422.

Run:  app/roop_3d/.venv/Scripts/python.exe verify_stage5_timeline_and_faces.py [clip.mp4]
An annotated frame is written next to the scratch root so the boxes can be looked at.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

import av
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.core.errors import DomainError
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog
from roop_3d.server.media_frames import router

DEFAULT_CLIP = APP / "roop_3d/.runtime/media/03de3642d8d64b22bee7ffe543fa2966.mp4"
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
    if not clip.is_file():
        print(f"clip not found: {clip}")
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="stage5_verify_"))
    target = "roop_3d/.runtime/media/clip.mp4"
    (scratch / target).parent.mkdir(parents=True)
    shutil.copyfile(clip, scratch / target)

    media = MediaService(scratch, ModelCatalog())
    app = FastAPI()
    app.include_router(router)
    app.state.media = media

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, error: DomainError):
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status)

    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        truth = {"width": stream.width, "height": stream.height, "fps": float(stream.average_rate), "frames": stream.frames}

    with TestClient(app) as client:
        print("== GET /media/info")
        began = time.perf_counter()
        info = client.get("/api/v1/media/info", params={"path": target}).json()
        print(f"   {info}  ({(time.perf_counter() - began) * 1000:.0f} ms)")
        check((info["width"], info["height"]) == (truth["width"], truth["height"]), f"width x height = {truth['width']}x{truth['height']}")
        check(abs(info["fps"] - truth["fps"]) < 1e-6, f"fps = {truth['fps']}")
        check(info["total_frames"] == truth["frames"], f"total_frames = {truth['frames']}")
        check(abs(info["duration_sec"] - truth["frames"] / truth["fps"]) < 0.5, f"duration_sec ~ {truth['frames'] / truth['fps']:.1f}")
        total = info["total_frames"]

        print("== GET /media/filmstrip?count=16")
        began = time.perf_counter()
        strip = client.get("/api/v1/media/filmstrip", params={"path": target, "count": 16}).json()
        took = time.perf_counter() - began
        indices = [item["frame_index"] for item in strip["frames"]]
        print(f"   {len(indices)} thumbnails {strip['thumb_width']}x{strip['thumb_height']} in {took * 1000:.0f} ms: {indices}")
        check(len(indices) == 16 and indices == sorted(indices) and len(set(indices)) == 16, "16 distinct ascending frames")
        check(indices[0] < total / 16 and indices[-1] > total * 15 / 16, "spread across the whole clip")
        decoded = [cv2.imdecode(np.frombuffer(__import__("base64").b64decode(item["data_url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR) for item in strip["frames"]]
        check(all(image is not None and image.shape[:2] == (strip["thumb_height"], strip["thumb_width"]) for image in decoded), "every thumbnail decodes at the advertised size")
        check(len({image.tobytes() for image in decoded}) > 8, "thumbnails are different pictures")
        check(took < 5.0, f"filmstrip took {took:.2f} s (< 5 s)")

        print("== GET /media/frame (no faces)")
        response = client.get("/api/v1/media/frame", params={"path": target, "frame_index": 1200})
        meta = json.loads(response.headers["x-frame-meta"])
        jpeg = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        check(response.status_code == 200 and response.headers["content-type"] == "image/jpeg", "200 image/jpeg")
        check(meta["frame_index"] == 1200 and meta["width"] == truth["width"] and "faces" not in meta, f"frame_index 1200, native {meta['width']}x{meta['height']}, no faces key")
        replay = media.frame(target, None, 1200)            # the older replay-from-frame-0 path, as ground truth
        difference = float(np.abs(jpeg.astype(np.int16) - replay.astype(np.int16)).mean())
        check(difference < 4.0, f"frame 1200 matches the replay path (mean abs diff {difference:.2f} after JPEG)")
        again = json.loads(client.get("/api/v1/media/frame", params={"path": target, "frame_index": 1200}).headers["x-frame-meta"])
        check(again["cache"] == "hit", "the same frame again is a cache hit")

        print("== frames beyond 9 999 (refused by the replay path)")
        refused = False
        try:
            media.frame(target, None, 12000)
        except DomainError:
            refused = True
        far = client.get("/api/v1/media/frame", params={"path": target, "frame_index": 12000})
        check(refused and far.status_code == 200 and json.loads(far.headers["x-frame-meta"])["frame_index"] == 12000, "frame 12000 is served (the replay path refuses it)")

        print("== GET /media/frame?detect_faces=true: scanning for frames with faces")
        found = []
        began = time.perf_counter()
        for index in range(0, total, 150):
            meta = json.loads(client.get("/api/v1/media/frame", params={"path": target, "frame_index": index, "detect_faces": True}).headers["x-frame-meta"])
            if meta["faces"]:
                found.append((index, meta))
            if len(found) >= 4:
                break
        print(f"   scanned to frame {index}, {len(found)} frames with faces, {time.perf_counter() - began:.1f} s")
        check(len(found) >= 1, "at least one frame with a detected face")
        annotated_path = None
        for index, meta in found:
            width, height = meta["width"], meta["height"]
            faces = meta["faces"]
            print(f"   frame {index}: {len(faces)} face(s)")
            check([face["face_index"] for face in faces] == list(range(len(faces))), f"frame {index}: face_index ranks are 0..{len(faces) - 1}")
            check([face["bbox"][0] for face in faces] == sorted(face["bbox"][0] for face in faces), f"frame {index}: faces ordered left to right (as face extraction ranks them)")
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                check(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"frame {index} #{face['face_index'] + 1}: box {face['bbox']} inside {width}x{height}")
                inside = sum(x1 - 0.25 * (x2 - x1) <= x <= x2 + 0.25 * (x2 - x1) and y1 - 0.25 * (y2 - y1) <= y <= y2 + 0.25 * (y2 - y1) for x, y in face["kps_5"])
                check(len(face["kps_5"]) == 5 and inside == 5, f"frame {index} #{face['face_index'] + 1}: 5 keypoints on the face")
                eyes_ok = face["kps_5"][0][0] < face["kps_5"][1][0] or face["pose"]["yaw"] != 0
                check(eyes_ok and 0 < face["confidence"] <= 1, f"frame {index} #{face['face_index'] + 1}: confidence {face['confidence']}, yaw {face['pose']['yaw']}, pitch {face['pose']['pitch']}")
                check(abs(face["pose3d"]["pitch"] + face["pose"]["pitch"]) < 0.02 and face["track_id"] is None, f"frame {index} #{face['face_index'] + 1}: pose3d is the pitch-flipped pose; track_id null until locked")
                check(abs(face["bbox_norm"][0] - x1 / width) < 1e-4, f"frame {index} #{face['face_index'] + 1}: bbox_norm consistent with the pixel box")
            check("embedding_norm" in meta["unsupported"], f"frame {index}: embedding_norm reported as unsupported, not invented")
            # Independent check: the detector run straight on the replay path's frame must find the same boxes.
            reference = media.frame(target, None, index) if index < 10000 else None
            if reference is not None:
                detections, _ = media.detector.detect(reference, 0.5)
                reference_boxes = sorted((box.tolist() for box, _ in detections), key=lambda box: box[0])
                agree = len(reference_boxes) == len(faces) and all(iou(face["bbox"], box) > 0.85 for face, box in zip(faces, reference_boxes))
                check(agree, f"frame {index}: boxes agree with an independent detector run (IoU > 0.85)")
            if annotated_path is None:
                image = cv2.imdecode(np.frombuffer(client.get("/api/v1/media/frame", params={"path": target, "frame_index": index}).content, np.uint8), cv2.IMREAD_COLOR)
                for face in faces:
                    x1, y1, x2, y2 = (int(v) for v in face["bbox"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (211, 211, 34), 2)
                    for x, y in face["kps_5"]:
                        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
                    cv2.putText(image, f"#{face['face_index'] + 1} Y{face['pose']['yaw']:+.0f} P{face['pose3d']['pitch']:+.0f}", (x1, max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (211, 211, 34), 1, cv2.LINE_AA)
                annotated_path = scratch / f"annotated_frame_{index}.jpg"
                cv2.imwrite(str(annotated_path), image)
                print(f"   annotated frame: {annotated_path}")

        print("== scrubbing speed over HTTP")
        began = time.perf_counter()
        sequential = []
        for index in range(2000, 2060):
            t = time.perf_counter()
            client.get("/api/v1/media/frame", params={"path": target, "frame_index": index})
            sequential.append((time.perf_counter() - t) * 1000)
        random_ms = []
        rng = np.random.default_rng(5)
        for index in rng.integers(0, total, 40):
            t = time.perf_counter()
            body = client.get("/api/v1/media/frame", params={"path": target, "frame_index": int(index)})
            random_ms.append((time.perf_counter() - t) * 1000)
            check_index = json.loads(body.headers["x-frame-meta"])["frame_index"]
            if check_index != int(index):
                failures.append(f"random frame {index} returned {check_index}")
        print(f"   60 sequential frames: median {statistics.median(sequential):.1f} ms ({1000 / statistics.median(sequential):.0f} fps), "
              f"40 random seeks: median {statistics.median(random_ms):.0f} ms, p95 {sorted(random_ms)[37]:.0f} ms")
        check(statistics.median(sequential) < 33, "sequential scrubbing faster than 30 fps")
        check(statistics.median(random_ms) < 150, "random seeks under 150 ms")
        check(media.frames.stats["containers_opened"] == 1, f"one container opened for the whole run (opened {media.frames.stats['containers_opened']})")

        print("== errors")
        check(client.get("/api/v1/media/frame", params={"path": target, "frame_index": total}).status_code == 416, f"frame {total} -> 416")
        check(client.get("/api/v1/media/frame", params={"path": target}).status_code == 422, "no selector -> 422")
        check(client.get("/api/v1/media/frame", params={"path": "../outside.mp4", "frame_index": 0}).status_code == 422, "path outside the runtime -> 422")
        check(client.get("/api/v1/media/info", params={"path": "roop_3d/.runtime/media/missing.mp4"}).status_code == 404, "missing file -> 404")

    media.frames.close()
    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed")
        for message in failures:
            print("  -", message)
        return 1
    print("PASS: all Stage 5 media endpoint checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
