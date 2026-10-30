"""Bounded local PyAV decoding; remote protocols and external references disabled."""
from __future__ import annotations

import base64
from contextlib import contextmanager
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
from roop_3d.media.detector import Detector, iou
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
        try:
            with path.open("rb") as handle, av.open(handle, mode="r", options={
                "protocol_whitelist": "", "enable_drefs": "0", "use_absolute_path": "0",
                "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts",
            }) as container:
                if not container.streams.video:
                    raise DomainError("VIDEO_STREAM_MISSING", "Container has no video stream", 422)
                stream = container.streams.video[0]
                if stream.width <= 0 or stream.height <= 0 or stream.width * stream.height > 16777216:
                    raise DomainError("VIDEO_DIMENSIONS_DENIED", "Video dimensions exceed the decode budget", 422)
                yield container
        except (av.FFmpegError, OSError):
            raise DomainError("MEDIA_DECODE_FAILED", "Local video could not be decoded", 422) from None

    def probe(self, value: str) -> MediaProbe:
        with self.open(value) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            duration = float(stream.duration * stream.time_base) if stream.duration is not None else (
                container.duration / av.time_base if container.duration is not None else None)
            audio = tuple(AudioTrack(stream_index=item.index, codec=item.codec_context.name,
                                     sample_rate=item.codec_context.sample_rate or None,
                                     channels=len(item.codec_context.layout.channels),
                                     language=item.metadata.get("language")) for item in container.streams.audio)
            timestamps = []
            complete = True
            started = time.monotonic()
            for frame in container.decode(stream):
                if frame.pts is not None:
                    timestamps.append(float(frame.pts * frame.time_base))
                if len(timestamps) >= 300 or time.monotonic() - started > 10:
                    complete = False
                    break
            differences = np.diff(timestamps)
            # Constant rates can alternate adjacent ticks in coarse time bases.
            tolerance = max(0.0005, 1.5 * float(stream.time_base))
            varying = len(differences) > 1 and float(np.ptp(differences)) > tolerance
            vfr = bool(varying) if varying or (complete and len(differences) > 1) else None
            return MediaProbe(container=container.format.name, duration_seconds=duration,
                              fps=float(rate) if rate else None, fps_rational=str(rate) if rate else None,
                              width=stream.width, height=stream.height, video_stream_index=stream.index,
                              video_codec=stream.codec_context.name, declared_frame_count=stream.frames or None,
                              vfr=vfr, vfr_state="VERIFIED_LOCAL" if vfr is not None else "UNKNOWN",
                              timing_sample_frames=len(timestamps), timing_sample_complete=complete, audio_tracks=audio)

    def frame(self, value: str, timestamp: float | None, frame_index: int | None) -> np.ndarray:
        if (timestamp is None) == (frame_index is None):
            raise DomainError("FRAME_SELECTOR_INVALID", "Specify exactly one timestamp or frame_index", 422)
        if timestamp is not None and (not math.isfinite(timestamp) or not 0 <= timestamp <= 86400):
            raise DomainError("FRAME_SELECTOR_INVALID", "Timestamp must be finite and within one day", 422)
        if frame_index is not None and not 0 <= frame_index < 10000:
            raise DomainError("FRAME_SELECTOR_INVALID", "Frame index must be between 0 and 9999", 422)
        with self.open(value) as container:
            stream = container.streams.video[0]
            start = float((stream.start_time or 0) * stream.time_base)
            if timestamp is not None:
                container.seek(int((timestamp + start) / stream.time_base), stream=stream, backward=True)
            began = time.monotonic()
            for index, frame in enumerate(container.decode(stream)):
                if index >= 10000 or time.monotonic() - began > 15:
                    raise DomainError("DECODE_BUDGET_EXCEEDED", "Requested frame exceeds the decode budget", 422)
                at = float(frame.pts * frame.time_base) - start if frame.pts is not None else None
                if (frame_index is not None and index == frame_index) or (
                    timestamp is not None and at is not None and at + 1e-6 >= timestamp
                ):
                    return frame.to_ndarray(format="bgr24")
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
        tracks: list[dict] = []
        samples = decoded = 0
        complete = True
        began = time.monotonic()
        record = self.detector.catalog.detector()
        with self.open(request.path) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base)
            for index, frame in enumerate(container.decode(stream)):
                decoded = index + 1
                if frame.key_frame or index == 0:
                    image = frame.to_ndarray(format="bgr24")
                    detections, record = self.detector.detect(image, request.detection_threshold)
                    at = max(0.0, float(frame.pts * frame.time_base) - origin) if frame.pts is not None else 0.0
                    unused = {i for i, track in enumerate(tracks) if samples - track["last_sample"] <= 2}
                    for box, landmarks in detections:
                        match = max(unused, key=lambda i: iou(tracks[i]["last_box"], box), default=None)
                        if match is not None and iou(tracks[match]["last_box"], box) < 0.2:
                            match = None
                        roll = math.degrees(math.atan2(*(landmarks[1] - landmarks[0])[::-1]))
                        if match is None:
                            if len(tracks) >= 64:
                                continue
                            x1, y1, x2, y2 = box.astype(int)
                            crop = image[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
                            crop = cv2.resize(crop, (128, 128))
                            success, jpg = cv2.imencode(".jpg", crop)
                            if not success:
                                continue
                            tracks.append(dict(first=index, last=index, first_at=at, last_at=at,
                                               box=box, last_box=box, rolls=[roll], count=1, last_sample=samples,
                                               thumbnail=base64.b64encode(jpg).decode("ascii")))
                        else:
                            unused.remove(match)
                            track = tracks[match]
                            track.update(last=index, last_at=at, last_box=box, last_sample=samples, count=track["count"] + 1)
                            track["rolls"].append(roll)
                    samples += 1
                if decoded >= request.max_decoded_frames or samples >= request.max_samples or time.monotonic() - began > 30:
                    complete = False
                    break
        # Refuse to bind observations to a file which changed during inspection.
        if sha256_file(path)[1] != digest:
            raise DomainError("MEDIA_CHANGED", "Video changed during inspection")
        result = FaceInspection(inspection_id=uuid.uuid4().hex, target_sha256=digest,
                                detector_model_id=record["model_id"], detector_sha256=record["sha256"],
                                decoded_frames=decoded, sampled_frames=samples, truncated=not complete,
                                tracks=tuple(FaceTrack(track_id=f"track_{index:04d}", first_frame=item["first"],
                                                       last_frame=item["last"], first_timestamp_seconds=item["first_at"],
                                                       last_timestamp_seconds=item["last_at"],
                                                       bounding_box=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), item["box"]))),
                                                       pose_range=PoseRange(roll_degrees=(min(item["rolls"]), max(item["rolls"]))),
                                                       sampled_detections=item["count"], thumbnail_base64=item["thumbnail"])
                                             for index, item in enumerate(tracks)),
                                warnings=("Spatial tracks can split or merge at motion, crossings or cuts; review mappings.",
                                          "Yaw and pitch are unknown; only eye-line roll is measured."))
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{result.inspection_id}.json", result.model_dump(mode="json"))
        return result
