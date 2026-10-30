"""Bounded local PyAV decoding; remote protocols and external references disabled."""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from fractions import Fraction
import math
from pathlib import Path
import time
from typing import Callable, Iterator
import uuid

import av
import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings, runtime_path
from roop_3d.core.storage import atomic_json
from roop_3d.media.detector import Detector
from roop_3d.engine.detection.face_tracker import FaceTracker
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
        tracker = FaceTracker(self.detector.catalog, threshold=request.detection_threshold, detector=self.detector)
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
                    for face in tracked:
                        observation = face.observation
                        box = observation.box
                        yaw, pitch, roll = observation.pose
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
                            item["pose_min"] = np.minimum(item["pose_min"], observation.pose).tolist()
                            item["pose_max"] = np.maximum(item["pose_max"], observation.pose).tolist()
                    if frame.key_frame or index == 0:
                        samples += 1
                    if (not complete or decoded >= request.max_decoded_frames or samples >= request.max_samples
                            or time.monotonic() - began > 30):
                        complete = False
                        break
        finally:
            tracker.release()
        if sha256_file(path)[1] != digest:
            raise DomainError("MEDIA_CHANGED", "Video changed during inspection")
        result = FaceInspection(inspection_id=uuid.uuid4().hex, target_sha256=digest,
            detector_model_id=detector_record["model_id"], detector_sha256=detector_record["sha256"],
            recognizer_model_id=recognizer_record["model_id"], recognizer_sha256=recognizer_record["sha256"],
            decoded_frames=decoded, sampled_frames=samples, truncated=not complete,
            tracking_method="iou_arcface_hungarian", scene_cut_frames=tuple(cuts),
            tracks=tuple(FaceTrack(track_id=f"track_{identity:04d}", persistent_id=identity,
                scene_id=item["scene_id"], landmark_count=106, first_frame=item["first"], last_frame=item["last"],
                first_timestamp_seconds=item["first_at"], last_timestamp_seconds=item["last_at"],
                bounding_box=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), item["box"]))),
                pose_range=PoseRange(yaw_degrees=(item["pose_min"][0], item["pose_max"][0]),
                    pitch_degrees=(item["pose_min"][1], item["pose_max"][1]),
                    roll_degrees=(item["pose_min"][2], item["pose_max"][2]),
                    method="five_point_geometric_estimate", yaw_pitch_state="CANDIDATE"),
                sampled_detections=item["count"], thumbnail_base64=item["thumbnail"])
                for identity, item in records.items()),
            warnings=("IoU and ArcFace assignment can still split or merge tracks; review mappings.",
                      "Yaw and pitch are uncalibrated geometric estimates, not reconstructed 3D pose.",
                      "persistent_id is the integer engine track ID; track_id retains the job mapping label."))
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{result.inspection_id}.json", result.model_dump(mode="json"))
        return result
