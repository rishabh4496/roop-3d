"""Scrub-speed frame access: persistent decoders, a decoded-frame cache and per-frame face detection.

`MediaService.frame` replays the video from its first frame (bounded to 10 000 frames) and re-opens the
container on every call, which is right for exact batch work and wrong for a timeline scrubber. This service keeps
a small LRU of *open* containers instead:

* a request ahead of the last decoded frame (within ``FORWARD_WINDOW_S``) just keeps decoding: no seek, no reopen;
* any other request seeks (backward, to the previous keyframe) and decodes forward to the target;
* decoded frames, encoded JPEGs and face detections are cached, so going back and forth is instant.

Frame numbers map to time as ``frame_index / fps`` with the stream's average frame rate (exact for constant frame
rate). The response always reports the timestamp of the frame that was really decoded, so a variable-frame-rate file
is visible as a difference between the requested and the reported time rather than a silent substitution.
All colour handling (matrix, range, HDR tone mapping, rotation) is the production `VideoReader`'s.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass, field
from fractions import Fraction
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator

import av
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_pose3d, pose_cell
from roop_3d.media.probe import open_media
from roop_3d.media.video_reader import VideoReader

MAX_OPEN_CONTAINERS = 4
FORWARD_WINDOW_S = 1.0           # keep decoding instead of seeking when the target is at most this far ahead
DECODE_BUDGET_S = 15.0
MAX_DECODE_FRAMES = 4000         # one GOP is far below this; it only stops a pathological stream
IMAGE_CACHE_BYTES = 192 * 1024**2
JPEG_CACHE_BYTES = 64 * 1024**2
DETECTION_CACHE_ENTRIES = 256
FILMSTRIP_CACHE_ENTRIES = 8
FILMSTRIP_APPROX_FRAMES = 30     # thumbnails accept a nearby frame after this many decodes past a keyframe
MAX_FACES = 32


def _lru_put(cache: OrderedDict, key, value, *, limit_entries: int | None = None, limit_bytes: int | None = None,
             size: Callable[[Any], int] = lambda _v: 0) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while cache and ((limit_entries is not None and len(cache) > limit_entries)
                     or (limit_bytes is not None and len(cache) > 1 and sum(size(v) for v in cache.values()) > limit_bytes)):
        cache.popitem(last=False)


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    fps_rational: str
    total_frames: int
    duration_sec: float
    rotation_degrees: int
    frame_count_declared: bool

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps, "fps_rational": self.fps_rational,
                "total_frames": self.total_frames, "duration_sec": self.duration_sec,
                "rotation_degrees": self.rotation_degrees, "frame_count_declared": self.frame_count_declared}


@dataclass(frozen=True)
class FrameResult:
    image: np.ndarray            # upright BGR, native resolution
    frame_index: int             # index of the frame that was decoded (nominal fps)
    requested_index: int | None
    timestamp_sec: float         # presentation time of the decoded frame, relative to the stream start
    cache: str                   # "hit" or "miss"
    decode_ms: float


@dataclass
class _Entry:
    key: tuple
    stack: ExitStack
    container: Any
    stream: Any
    reader: VideoReader
    info: MediaInfo
    rotation: int
    start_pts: int
    time_base: Fraction
    fps: Fraction
    lock: threading.Lock = field(default_factory=threading.Lock)
    decoder: Iterator | None = None
    position: Fraction | None = None     # presentation time (s, relative to start) of the last frame the decoder yielded
    opens: int = 1

    def close(self) -> None:
        self.decoder = None
        try:
            self.stack.close()
        except Exception:  # a failing close must never take a request down
            pass


class VideoFrameService:
    """Thread-safe: one decode at a time per file, different files decode in parallel."""

    def __init__(self, resolve: Callable[[str], Path], detector: Any = None, *, max_open: int = MAX_OPEN_CONTAINERS,
                 analyzer: Any = None):
        self._resolve = resolve
        self.detector = detector
        self.analyzer = analyzer      # media.face_analysis.FaceAnalyzer: embeddings, dense landmarks, adaptive thresholds
        self._max_open = max(1, max_open)
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._images: OrderedDict[tuple, FrameResult] = OrderedDict()
        self._jpegs: OrderedDict[tuple, bytes] = OrderedDict()
        self._detections: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._filmstrips: OrderedDict[tuple, dict] = OrderedDict()
        self.stats = {"containers_opened": 0, "seeks": 0, "forward_decodes": 0, "frames_decoded": 0}

    # ------------------------------------------------------------------ containers
    def _entry(self, value: str) -> _Entry:
        path = self._resolve(value)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            # A changed file replaces its stale entries; the least recently used entry makes room.
            for stale in [k for k in self._entries if k[0] == key[0]]:
                self._entries.pop(stale).close()
            while len(self._entries) >= self._max_open:
                self._entries.popitem(last=False)[1].close()
            entry = self._open(path, key)
            self._entries[key] = entry
            return entry

    def _open(self, path: Path, key: tuple) -> _Entry:
        stack = ExitStack()
        try:
            container = stack.enter_context(open_media(path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 4
            reader = VideoReader(path, hardware="software")
            rotation = 0
            width, height = stream.width, stream.height
            for frame in container.decode(stream):    # the first frame carries the rotation and the colour description
                rotation = frame.rotation
                reader.convert(frame, stream, rotation)
                width, height = (frame.height, frame.width) if rotation % 180 else (frame.width, frame.height)
                break
            else:
                raise DomainError("MEDIA_DECODE_FAILED", "Video has no decodable frames", 422)
            rate = stream.average_rate or stream.guessed_rate or Fraction(30)
            fps = Fraction(rate)
            time_base = Fraction(stream.time_base)
            if container.duration:
                duration = container.duration / av.time_base
            elif stream.duration:
                duration = float(stream.duration * time_base)
            else:
                duration = 0.0
            declared = bool(stream.frames)
            total = int(stream.frames) if declared else int(round(duration * float(fps)))
            info = MediaInfo(width=int(width), height=int(height), fps=float(fps), fps_rational=f"{fps.numerator}/{fps.denominator}",
                             total_frames=max(1, total), duration_sec=float(duration), rotation_degrees=int(rotation) % 360,
                             frame_count_declared=declared)
            container.seek(0, stream=stream, backward=True)
            self.stats["containers_opened"] += 1
            return _Entry(key=key, stack=stack, container=container, stream=stream, reader=reader, info=info, rotation=rotation,
                          start_pts=int(stream.start_time or 0), time_base=time_base, fps=fps)
        except BaseException:
            stack.close()
            raise

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.close()
            self._entries.clear()
        with self._cache_lock:
            self._images.clear(); self._jpegs.clear(); self._detections.clear(); self._filmstrips.clear()

    # ------------------------------------------------------------------ decoding
    def _decode_to(self, entry: _Entry, target: Fraction, tolerance: Fraction, *, soft_after: int | None = None):
        """The first frame at or after ``target - tolerance`` seconds; decoder state is kept for the next request.

        ``soft_after``: accept the nearest frame once this many frames were decoded past a seek (thumbnails).
        """
        began = time.monotonic()
        # Keep decoding when the wanted frame is still to come and close; otherwise (earlier, or far ahead) seek.
        ahead = (entry.decoder is not None and entry.position is not None
                 and target - tolerance > entry.position and target <= entry.position + FORWARD_WINDOW_S)
        if not ahead:
            pts = int(target / entry.time_base) + entry.start_pts
            entry.container.seek(max(entry.start_pts, pts), stream=entry.stream, backward=True)
            entry.decoder = entry.container.decode(entry.stream)
            entry.position = None
            self.stats["seeks"] += 1
        else:
            self.stats["forward_decodes"] += 1
        decoded = 0
        last = None
        for frame in entry.decoder:
            if frame.pts is None:
                continue
            decoded += 1
            self.stats["frames_decoded"] += 1
            seconds = (frame.pts - entry.start_pts) * entry.time_base
            entry.position = seconds
            last = (frame, seconds)
            if seconds >= target - tolerance:
                return frame, seconds
            if soft_after is not None and decoded >= soft_after:
                return frame, seconds
            if decoded >= MAX_DECODE_FRAMES or time.monotonic() - began > DECODE_BUDGET_S:
                raise DomainError("DECODE_BUDGET_EXCEEDED", "Requested frame exceeds the decode budget", 422)
        entry.decoder = None
        if last is not None and soft_after is not None:
            return last
        raise DomainError("FRAME_OUT_OF_RANGE", "Requested frame is beyond the video", 416)

    def info(self, value: str) -> MediaInfo:
        entry = self._entry(value)
        return entry.info

    def read(self, value: str, *, frame_index: int | None = None, timestamp: float | None = None) -> FrameResult:
        """One full-resolution frame, selected by frame number or by time (exactly one)."""
        if (frame_index is None) == (timestamp is None):
            raise DomainError("FRAME_SELECTOR_INVALID", "Specify exactly one timestamp or frame_index", 422)
        if frame_index is not None and frame_index < 0:
            raise DomainError("FRAME_SELECTOR_INVALID", "Frame index must not be negative", 422)
        if timestamp is not None and (not math.isfinite(timestamp) or not 0 <= timestamp <= 86400):
            raise DomainError("FRAME_SELECTOR_INVALID", "Timestamp must be finite and within one day", 422)
        entry = self._entry(value)
        if frame_index is not None:
            if frame_index >= entry.info.total_frames + (0 if entry.info.frame_count_declared else 2):
                raise DomainError("FRAME_OUT_OF_RANGE", f"Frame {frame_index} is beyond the video ({entry.info.total_frames} frames)", 416)
            cache_key = (entry.key, "i", frame_index)
        else:
            cache_key = (entry.key, "t", int(round(timestamp * 1e6)))
        with self._cache_lock:
            hit = self._images.get(cache_key)
            if hit is not None:
                self._images.move_to_end(cache_key)
        if hit is not None:
            return FrameResult(hit.image, hit.frame_index, hit.requested_index, hit.timestamp_sec, "hit", 0.0)
        began = time.perf_counter()
        with entry.lock:
            if frame_index is not None:
                target = Fraction(frame_index) / entry.fps
                tolerance = Fraction(1, 2) / entry.fps    # presentation times jitter around the nominal grid
            else:
                target = Fraction(timestamp).limit_denominator(1_000_000)
                tolerance = Fraction(1, 1_000_000)
            frame, seconds = self._decode_to(entry, target, tolerance)
            image = entry.reader.convert(frame, entry.stream, entry.rotation)
        actual_index = int(round(float(seconds * entry.fps)))
        result = FrameResult(image, actual_index, frame_index, float(seconds), "miss", (time.perf_counter() - began) * 1000.0)
        with self._cache_lock:
            _lru_put(self._images, cache_key, result, limit_bytes=IMAGE_CACHE_BYTES, size=lambda r: r.image.nbytes)
        return result

    # ------------------------------------------------------------------ encoding
    def jpeg(self, value: str, result: FrameResult, *, max_size: int = 1280, quality: int = 82) -> tuple[bytes, int, int]:
        """(JPEG bytes, width, height) of the frame, shrunk so its long edge is at most ``max_size``."""
        image = result.image
        height, width = image.shape[:2]
        scale = min(1.0, max_size / max(width, height))
        out_w, out_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        key = (self._resolve(value).as_posix(), result.frame_index, result.timestamp_sec, out_w, quality)
        with self._cache_lock:
            cached = self._jpegs.get(key)
            if cached is not None:
                self._jpegs.move_to_end(key)
                return cached, out_w, out_h
        scaled = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
        ok, encoded = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise DomainError("PREVIEW_ENCODE_FAILED", "Frame JPEG could not be encoded", 422)
        data = encoded.tobytes()
        with self._cache_lock:
            _lru_put(self._jpegs, key, data, limit_bytes=JPEG_CACHE_BYTES, size=len)
        return data, out_w, out_h

    # ------------------------------------------------------------------ faces
    def analyzed_faces(self, value: str, result: FrameResult, threshold: float = 0.5):
        """Cached ``FaceAnalyzer`` output (left to right) for one decoded frame; None when no analyzer is configured."""
        if self.analyzer is None:
            return None
        key = ("analysis", self._resolve(value).as_posix(), result.frame_index, result.timestamp_sec, round(float(threshold), 3),
               self.analyzer.signature())
        with self._cache_lock:
            cached = self._detections.get(key)
            if cached is not None:
                self._detections.move_to_end(key)
                return cached
        faces = self.analyzer.analyze(result.image, float(threshold), detector=self.detector)
        with self._cache_lock:
            _lru_put(self._detections, key, faces, limit_entries=DETECTION_CACHE_ENTRIES)
        return faces

    def faces(self, value: str, result: FrameResult, threshold: float = 0.5, detail: str = "basic") -> list[dict]:
        """Every face of the frame, left to right: native-pixel box, 5-point landmarks, pose and confidence.

        With a ``FaceAnalyzer`` (production) each face also carries ``detection_tier`` / ``landmark_validity`` and, with
        ``detail="full"``, the embedding, thumbnail and dense landmarks. Read-only: nothing is written.
        """
        if self.detector is None:
            raise DomainError("DETECTOR_UNAVAILABLE", "Face detection is not configured", 503)
        analyzed = self.analyzed_faces(value, result, threshold)
        height, width = result.image.shape[:2]
        if analyzed is not None:
            from roop_3d.media.face_analysis import face_to_dict
            return [face_to_dict(face, rank, result.frame_index, width, height, detail=detail) for rank, face in enumerate(analyzed)]
        key = (self._resolve(value).as_posix(), result.frame_index, result.timestamp_sec, round(float(threshold), 3))
        with self._cache_lock:
            cached = self._detections.get(key)
            if cached is not None:
                self._detections.move_to_end(key)
                return cached
        from roop_3d.engine.detection.dense_landmarks import geometric_pose
        scored_detect = getattr(self.detector, "detect_scored", None)
        if scored_detect is not None:
            detections, _ = scored_detect(result.image, threshold)
        else:
            detections = [(box, points, None) for box, points in self.detector.detect(result.image, threshold)[0]]
        # Same order as face extraction (`face_extract`: by left edge), so face #n here is face #n there.
        ordered = sorted(detections, key=lambda item: float(item[0][0]))[:MAX_FACES]
        faces = []
        for rank, (box, points, score) in enumerate(ordered):
            yaw, pitch, roll = geometric_pose(points)
            pose3d = as_pose3d((yaw, pitch, roll))
            x1, y1, x2, y2 = (float(v) for v in box)
            faces.append({
                "face_id": f"f{result.frame_index}_{rank}",
                "face_index": rank,                       # left edge, left to right: the rank `Use as source` / face extraction takes
                "track_id": None,                         # a per-frame detection has no tracker; locking a face creates its track
                "cluster_id": None,
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_norm": [round(x1 / width, 5), round(y1 / height, 5), round(x2 / width, 5), round(y2 / height, 5)],
                "kps_5": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in points],
                "pose": {"yaw": round(float(yaw), 2), "pitch": round(float(pitch), 2), "roll": round(float(roll), 2)},   # legacy frame, pitch + = down
                "pose3d": {"yaw": round(float(pose3d.yaw), 2), "pitch": round(float(pose3d.pitch), 2), "roll": round(float(pose3d.roll), 2)},   # standard frame, pitch + = up
                "pose_cell": pose_cell(pose3d).name,
                "confidence": round(float(score), 4) if score is not None else None,
            })
        with self._cache_lock:
            _lru_put(self._detections, key, faces, limit_entries=DETECTION_CACHE_ENTRIES)
        return faces

    # ------------------------------------------------------------------ filmstrip
    def filmstrip(self, value: str, count: int = 16, thumb_height: int = 90, quality: int = 72) -> dict:
        """``count`` evenly spaced thumbnails in one pass. Each carries the timestamp of the frame it really is.

        Thumbnails accept a frame near the target (within ``FILMSTRIP_APPROX_FRAMES`` frames after a keyframe) instead
        of decoding a whole GOP per thumbnail, which is what makes sixteen of them fast.
        """
        if not 1 <= count <= 64 or not 24 <= thumb_height <= 360:
            raise DomainError("FILMSTRIP_REQUEST_INVALID", "count must be 1-64 and thumb_height 24-360", 422)
        entry = self._entry(value)
        key = (entry.key, count, thumb_height, quality)
        with self._cache_lock:
            cached = self._filmstrips.get(key)
            if cached is not None:
                self._filmstrips.move_to_end(key)
                return cached
        info = entry.info
        thumb_width = max(1, int(round(info.width * thumb_height / info.height)))
        frames = []
        with entry.lock:
            for slot in range(count):
                index = min(info.total_frames - 1, int((slot + 0.5) / count * info.total_frames))
                frame, seconds = self._decode_to(entry, Fraction(index) / entry.fps, Fraction(1, 2) / entry.fps,
                                                 soft_after=FILMSTRIP_APPROX_FRAMES)
                image = entry.reader.convert(frame, entry.stream, entry.rotation)
                thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    raise DomainError("PREVIEW_ENCODE_FAILED", "Thumbnail could not be encoded", 422)
                frames.append({"slot": slot, "frame_index": int(round(float(seconds * entry.fps))), "requested_frame_index": index,
                               "timestamp_sec": round(float(seconds), 4),
                               "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")})
        payload = {"count": count, "thumb_width": thumb_width, "thumb_height": thumb_height, "total_frames": info.total_frames,
                   "duration_sec": info.duration_sec, "frames": frames}
        with self._cache_lock:
            _lru_put(self._filmstrips, key, payload, limit_entries=FILMSTRIP_CACHE_ENTRIES)
        return payload
