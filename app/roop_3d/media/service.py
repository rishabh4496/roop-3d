"""Bounded local PyAV decoding; remote protocols and external references disabled."""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterator
import uuid

import av
import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file, safe_path
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_legacy_triplet, as_pose3d, pose_cell
from roop_3d.core.settings import Settings, runtime_path
from roop_3d.core.storage import atomic_json
from roop_3d.media.detector import Detector
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.identity.report import IdentityReportCollector, write_report
from roop_3d.media.probe import open_media, probe
from roop_3d.media.timestamps import Timestamp
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.schema import AudioTrack, BoundingBox, FaceInspection, FaceTrack, InspectRequest, MediaProbe, PoseRange
from roop_3d.models.catalog import ModelCatalog

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}


class MediaService:
    def __init__(self, root: Path, catalog: ModelCatalog, settings: Callable[[], Settings] = Settings):
        self.root = root
        self.detector = Detector(catalog)
        self.settings = settings

    def path(self, value: str) -> Path:
        try:
            path = runtime_path(value, self.root)
        except ValueError:
            raise DomainError("MEDIA_PATH_DENIED", "Media must be a local file inside the owned runtime", 422) from None
        permitted = (self.root / "roop_3d/.runtime/media", runtime_path(self.settings().output_root, self.root))
        if not any(path.is_relative_to(parent) for parent in permitted) or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise DomainError("MEDIA_PATH_DENIED", "Only video files in runtime/media or runtime/outputs are accepted", 422)
        if not path.is_file():
            raise DomainError("MEDIA_NOT_FOUND", "Local video file does not exist", 404)
        return path

    @contextmanager
    def open(self, value: str) -> Iterator[av.container.InputContainer]:
        path = self.path(value)
        with open_media(path) as container:
            yield container

    def probe(self, value: str) -> MediaProbe:
        info = probe(self.path(value))
        video = info.videos[0]
        rate = video.average_rate
        duration = video.duration * video.time_base if video.duration is not None else info.duration
        audio = tuple(AudioTrack(stream_index=item.index, codec=item.codec, sample_rate=item.sample_rate,
                                 channels=item.channels, language=item.language, time_base=str(item.time_base),
                                 channel_layout=item.layout) for item in info.audio)
        return MediaProbe(container=info.container, duration_seconds=float(duration) if duration is not None else None,
                          fps=float(rate) if rate else None, fps_rational=str(rate) if rate else None,
                          width=video.width, height=video.height, video_stream_index=video.index,
                          video_codec=video.codec, declared_frame_count=video.frame_count,
                          vfr=video.vfr, vfr_state="VERIFIED_LOCAL" if video.vfr is not None else "UNKNOWN",
                          timing_sample_frames=video.sampled_frames, timing_sample_complete=video.sample_complete,
                          audio_tracks=audio, time_base=str(video.time_base), nominal_frame_rate=str(video.nominal_rate) if video.nominal_rate else None,
                          pixel_format=video.pixel_format, color_space=video.color_space_name,
                          color_range=video.color_range, rotation_degrees=video.rotation)

    def frame(self, value: str, timestamp: float | None, frame_index: int | None) -> np.ndarray:
        if (timestamp is None) == (frame_index is None):
            raise DomainError("FRAME_SELECTOR_INVALID", "Specify exactly one timestamp or frame_index", 422)
        if timestamp is not None and (not math.isfinite(timestamp) or not 0 <= timestamp <= 86400):
            raise DomainError("FRAME_SELECTOR_INVALID", "Timestamp must be finite and within one day", 422)
        if frame_index is not None and not 0 <= frame_index < 10000:
            raise DomainError("FRAME_SELECTOR_INVALID", "Frame index must be between 0 and 9999", 422)
        path = self.path(value)
        reader = VideoReader(path)
        if timestamp is not None:
            with self.open(value) as container:
                stream = container.streams.video[0]
                at = Fraction(str(timestamp)) + (stream.start_time or 0) * Fraction(stream.time_base)
            return reader.at(Timestamp(at.numerator, Fraction(1, at.denominator))).image
        began = time.monotonic()
        iterator = reader.frames()
        try:
            for frame in iterator:
                if time.monotonic() - began > 15:
                    raise DomainError("DECODE_BUDGET_EXCEEDED", "Requested frame exceeds the decode budget", 422)
                if frame.index == frame_index:
                    return frame.image
        finally:
            iterator.close()
        raise DomainError("FRAME_OUT_OF_RANGE", "Requested frame is beyond the video", 416)

    def preview(self, value: str, timestamp: float | None, frame_index: int | None,
                comparison_path: str | None = None) -> bytes:
        images = [self.frame(value, timestamp, frame_index)]
        if comparison_path is not None:
            images.append(self.frame(comparison_path, timestamp, frame_index))
        images = [cv2.resize(image, (max(1, int(image.shape[1] * min(720 / image.shape[0], 1280 / image.shape[1], 1))),
                                    max(1, int(image.shape[0] * min(720 / image.shape[0], 1280 / image.shape[1], 1)))))
                  for image in images]
        if len(images) == 2:
            height = min(image.shape[0] for image in images)
            images = [cv2.resize(image, (max(1, int(image.shape[1] * height / image.shape[0])), height)) for image in images]
        success, jpeg = cv2.imencode(".jpg", np.hstack(images), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise DomainError("PREVIEW_ENCODE_FAILED", "Preview JPEG could not be encoded", 422)
        return jpeg.tobytes()

    def inspect(self, request: InspectRequest) -> FaceInspection:
        path = self.path(request.path)
        _, digest = sha256_file(path)
        records: dict[int, dict] = {}
        samples = decoded = 0
        complete = True
        began = time.monotonic()
        detector_record = self.detector.catalog.detector()
        recognizer_record = self.detector.catalog.recognizer()
        policy = TrackPolicy() if request.identity_lock else None
        tracker = FaceTracker(self.detector.catalog, threshold=request.detection_threshold, detector=self.detector,
                              policy=policy)
        collector = (IdentityReportCollector(policy, detection_threshold=request.detection_threshold)
                     if policy is not None else None)
        inspection_id = uuid.uuid4().hex
        cuts: list[int] = []
        with self.open(request.path) as container:
            stream = container.streams.video[0]
            origin = (stream.start_time or 0) * Fraction(stream.time_base)
        try:
            with closing(VideoReader(path).frames()) as frames:
                for index, frame in enumerate(frames):
                    decoded = index + 1
                    image = frame.image
                    tracked = tracker.process(image)
                    if tracker.scene_cut and len(cuts) < 64:
                        cuts.append(index)
                    at = max(0.0, float(frame.timestamp.seconds - origin))
                    if collector is not None:
                        collector.observe(index, at, image, tracked)
                    for face in tracked:
                        observation = face.observation
                        box = observation.box
                        yaw, pitch, roll = pose = as_legacy_triplet(observation.pose)
                        if face.track_id not in records:
                            if len(records) >= 64:
                                complete = False
                                break
                            x1, y1, x2, y2 = box.astype(int)
                            crop = cv2.resize(image[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)], (128, 128))
                            success, jpg = cv2.imencode(".jpg", crop)
                            if not success:
                                raise DomainError("PREVIEW_ENCODE_FAILED", "Face thumbnail could not be encoded", 422)
                            records[face.track_id] = dict(first=index, last=index, first_at=at, last_at=at,
                                box=box, pose_min=[yaw, pitch, roll], pose_max=[yaw, pitch, roll], count=1,
                                scene_id=face.scene_id, thumbnail=base64.b64encode(jpg).decode("ascii"))
                        else:
                            item = records[face.track_id]
                            item.update(last=index, last_at=at, count=item["count"] + 1)
                            item["pose_min"] = np.minimum(item["pose_min"], pose).tolist()
                            item["pose_max"] = np.maximum(item["pose_max"], pose).tolist()
                    if frame.key_frame or index == 0:
                        samples += 1
                    if (not complete or (not request.full_video and
                            (decoded >= request.max_decoded_frames or samples >= request.max_samples))
                            or time.monotonic() - began > request.time_budget_seconds):
                        complete = False
                        break
        finally:
            tracker.release()
        if sha256_file(path)[1] != digest:
            raise DomainError("MEDIA_CHANGED", "Video changed during inspection")
        identity_extra: dict = {}
        identity_warnings: tuple[str, ...] = ()
        if collector is not None:
            document = collector.finalize(inspection_id=inspection_id, target_sha256=digest, events=tracker.events,
                                          events_dropped=tracker.events_dropped, scene_cut_frames=cuts,
                                          decoded_frames=decoded, truncated=not complete)
            identity_extra = dict(tracking_policy_fingerprint=policy.fingerprint(),
                                  detection_threshold=request.detection_threshold,
                                  identity_report_sha256=write_report(self.root, document))
            identity_warnings = ("Identity lock: faces whose identity is uncertain or ambiguous are never swapped.",
                                 "Identity thresholds are uncalibrated heuristics; review the contact sheet before locking.")
            if not complete:
                identity_warnings += ("Inspection stopped early; later appearances of a person may be unreviewed.",)
        result = FaceInspection(inspection_id=inspection_id, target_sha256=digest,
            detector_model_id=detector_record["model_id"], detector_sha256=detector_record["sha256"],
            recognizer_model_id=recognizer_record["model_id"], recognizer_sha256=recognizer_record["sha256"],
            decoded_frames=decoded, sampled_frames=samples, truncated=not complete,
            tracking_method="identity_lock_v1" if collector is not None else "iou_arcface_hungarian",
            scene_cut_frames=tuple(cuts), **identity_extra,
            tracks=tuple(FaceTrack(track_id=f"track_{identity:04d}", persistent_id=identity,
                scene_id=item["scene_id"], landmark_count=106, first_frame=item["first"], last_frame=item["last"],
                first_timestamp_seconds=item["first_at"], last_timestamp_seconds=item["last_at"],
                bounding_box=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), item["box"]))),
                pose_range=PoseRange(yaw_degrees=(item["pose_min"][0], item["pose_max"][0]),
                    pitch_degrees=(item["pose_min"][1], item["pose_max"][1]),
                    roll_degrees=(item["pose_min"][2], item["pose_max"][2]),
                    method="five_point_geometric_estimate", yaw_pitch_state="CANDIDATE"),
                sampled_detections=item["count"],
                thumbnail_base64=(collector.best_thumbnail(identity) if collector is not None else None) or item["thumbnail"])
                for identity, item in records.items()),
            warnings=("IoU and ArcFace assignment can still split or merge tracks; review mappings.",
                      "Yaw and pitch are uncalibrated geometric estimates, not reconstructed 3D pose.",
                      "persistent_id is the integer engine track ID; track_id retains the job mapping label.",
                      *identity_warnings))
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{result.inspection_id}.json", result.model_dump(mode="json"))
        return result

    def detect_frame_faces(self, value: str, timestamp: float | None = None,
                           frame_index: int | None = None, threshold: float = 0.5) -> dict:
        path = self.path(value)
        _, digest = sha256_file(path)

        info = self.probe(value)
        fps = info.fps or 30.0
        width = info.width
        height = info.height

        if timestamp is None and frame_index is None:
            timestamp = 0.0
            frame_index = 0
        elif timestamp is not None and frame_index is None:
            frame_index = int(round(timestamp * fps))
        elif frame_index is not None and timestamp is None:
            timestamp = float(frame_index / fps)

        image = self.frame(value, timestamp, None)
        retained, detector_record = self.detector.detect(image, threshold=threshold)
        recognizer_record = self.detector.catalog.recognizer()

        from roop_3d.engine.detection.dense_landmarks import geometric_pose

        inspections_dir = safe_path(self.root, "roop_3d/.runtime/inspections")
        inspections_dir.mkdir(parents=True, exist_ok=True)

        existing_inspection = None
        for insp_path in inspections_dir.glob("*.json"):
            try:
                insp_data = json.loads(insp_path.read_text("utf-8"))
                if insp_data.get("target_sha256") == digest:
                    existing_inspection = FaceInspection.model_validate(insp_data)
                    break
            except Exception:
                continue

        if not retained:
            return {
                "inspection_id": existing_inspection.inspection_id if existing_inspection else "",
                "faces": [],
                "inspection": existing_inspection.model_dump(mode="json") if existing_inspection else None,
            }

        faces_output = []
        new_tracks = list(existing_inspection.tracks) if existing_inspection else []
        track_start_idx = len(new_tracks)

        for idx, (box, points) in enumerate(retained):
            x1, y1, x2, y2 = box.astype(int)
            x1_cl = max(0, min(x1, width - 1))
            y1_cl = max(0, min(y1, height - 1))
            x2_cl = max(x1_cl + 1, min(x2, width))
            y2_cl = max(y1_cl + 1, min(y2, height))

            crop = cv2.resize(image[y1_cl:y2_cl, x1_cl:x2_cl], (128, 128))
            success, jpg = cv2.imencode(".jpg", crop)
            thumb_b64 = base64.b64encode(jpg).decode("ascii") if success else ""

            yaw, pitch, roll = geometric_pose(points)
            pose3d = as_pose3d((yaw, pitch, roll))

            track_id = f"track_{track_start_idx + idx:04d}"
            bbox_norm = [float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)]
            landmarks5_norm = [[float(p[0] / width), float(p[1] / height)] for p in points]

            face_item = {
                "id": f"{track_id}_{frame_index}",
                "track_id": track_id,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "bbox": bbox_norm,
                "landmarks5": landmarks5_norm,
                "confidence": 0.98,
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},   # legacy frame (pitch + = down), kept for existing clients
                "pose3d": pose3d.model_dump(), "pose_cell": pose_cell(pose3d).name,   # standard frame (pitch + = up)
                "thumbnail_base64": thumb_b64,
            }
            faces_output.append(face_item)

            track_obj = FaceTrack(
                track_id=track_id,
                persistent_id=track_start_idx + idx,
                scene_id=0,
                landmark_count=5,
                first_frame=frame_index,
                last_frame=frame_index,
                first_timestamp_seconds=timestamp,
                last_timestamp_seconds=timestamp,
                bounding_box=BoundingBox(x1=float(x1_cl), y1=float(y1_cl), x2=float(x2_cl), y2=float(y2_cl)),
                pose_range=PoseRange(
                    yaw_degrees=(yaw, yaw),
                    pitch_degrees=(pitch, pitch),
                    roll_degrees=(roll, roll),
                    method="five_point_geometric_estimate",
                    yaw_pitch_state="CANDIDATE",
                ),
                thumbnail_base64=thumb_b64,
                sampled_detections=1,
            )
            new_tracks.append(track_obj)

        inspection_id = existing_inspection.inspection_id if existing_inspection else uuid.uuid4().hex
        inspection_obj = FaceInspection(
            inspection_id=inspection_id,
            target_sha256=digest,
            detector_model_id=detector_record["model_id"],
            detector_sha256=detector_record["sha256"],
            recognizer_model_id=recognizer_record["model_id"],
            recognizer_sha256=recognizer_record["sha256"],
            decoded_frames=max(1, frame_index + 1),
            sampled_frames=1,
            truncated=False,
            tracking_method="iou_arcface_hungarian",
            scene_cut_frames=(),
            tracks=tuple(new_tracks),
            warnings=("Target face selected directly from frame preview.",),
        )
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{inspection_id}.json",
                    inspection_obj.model_dump(mode="json"))

        return {
            "inspection_id": inspection_id,
            "faces": faces_output,
            "inspection": inspection_obj.model_dump(mode="json"),
        }

    def segment_boundaries(self, path_str: str, max_segment_duration: float = 15.0) -> list[dict]:
        target = self.path(path_str)
        probe_res = self.probe(path_str)
        total_duration = probe_res.duration_seconds or 0.0
        if total_duration <= 0.0:
            return [{"segment_index": 0, "start_seconds": 0.0, "end_seconds": None, "duration": 0.0}]

        if total_duration <= max_segment_duration:
            return [{"segment_index": 0, "start_seconds": 0.0, "end_seconds": round(total_duration, 3), "duration": round(total_duration, 3)}]

        keyframes: list[float] = [0.0]
        try:
            with self.open(path_str) as container:
                stream = container.streams.video[0]
                time_base = Fraction(stream.time_base)
                start_pts = stream.start_time or 0
                for packet in container.demux(stream):
                    if packet.is_keyframe and packet.pts is not None:
                        sec = float((packet.pts - start_pts) * time_base)
                        if sec > 0.05 and (not keyframes or sec - keyframes[-1] >= 0.5):
                            keyframes.append(sec)
        except Exception:
            pass

        if len(keyframes) <= 1:
            step = max_segment_duration
            cuts = [i * step for i in range(int(math.ceil(total_duration / step)) + 1)]
            cuts[-1] = total_duration
        else:
            cuts = [0.0]
            current_target = max_segment_duration
            while current_target < total_duration - 1.0:
                candidates = [k for k in keyframes if k > cuts[-1] + 1.0 and k <= current_target + max_segment_duration * 0.5]
                if candidates:
                    best = min(candidates, key=lambda k: abs(k - current_target))
                    cuts.append(best)
                    current_target = best + max_segment_duration
                else:
                    after = [k for k in keyframes if k > cuts[-1] + 1.0]
                    if after:
                        cuts.append(after[0])
                        current_target = after[0] + max_segment_duration
                    else:
                        break
            if cuts[-1] < total_duration:
                cuts.append(total_duration)

        segments = []
        for idx in range(len(cuts) - 1):
            start_t = round(cuts[idx], 3)
            end_t = round(cuts[idx + 1], 3)
            segments.append({
                "segment_index": idx,
                "start_seconds": start_t,
                "end_seconds": end_t,
                "duration": round(end_t - start_t, 3)
            })
        return segments

    def concat_segments(self, segment_paths: list[str], output_filename: str | None = None) -> dict:
        import subprocess
        import shutil
        if not segment_paths:
            raise DomainError("SEGMENTS_EMPTY", "No segment paths provided", 422)

        resolved_paths = [self.path(p) for p in segment_paths]
        for p in resolved_paths:
            if not p.is_file():
                raise DomainError("SEGMENT_FILE_MISSING", f"Segment file {p.name} does not exist", 404)

        out_name = output_filename or f"roop3d_concat_{uuid.uuid4().hex[:12]}.mp4"
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"

        output_dir = runtime_path(self.settings().output_root, self.root)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / out_name

        token = uuid.uuid4().hex
        temp_list = self.root / f"roop_3d/outputs/{token}_list.txt"
        temp_list.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"file '{p.resolve().as_posix()}'\n" for p in resolved_paths]
        temp_list.write_text("".join(lines), encoding="utf-8")

        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
        try:
            cmd = [
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                "-i", str(temp_list), "-c", "copy", str(out_file)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as err:
            temp_list.unlink(missing_ok=True)
            raise DomainError("CONCAT_FAILED", f"FFmpeg stream copy failed: {err.stderr[-200:] if err.stderr else 'unknown'}", 500)
        finally:
            temp_list.unlink(missing_ok=True)

        try:
            out_rel = out_file.relative_to(self.root).as_posix()
            out_probe = self.probe(out_rel)
            total_frames = out_probe.declared_frame_count or 0
        except Exception:
            out_rel = out_file.relative_to(self.root).as_posix()
            total_frames = 0

        return {
            "output_path": out_rel,
            "frames": total_frames,
            "segments_concatenated": len(segment_paths)
        }
