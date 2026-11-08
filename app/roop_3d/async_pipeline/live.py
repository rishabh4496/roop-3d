"""Zero-copy-ish live egress: GPU JPEG encoding on a background thread and a 16-byte binary frame header.

Nothing here runs on an inference thread except ``LiveFrameHub.publish`` -- a memcpy of the frame into a staging slot
and a notify.  JPEG encoding (nvJPEG when available) and the fan-out to WebSocket subscribers happen on the hub's own
thread, so a slow browser or a slow encoder can never stall a render; the hub keeps only the newest frame
("latest wins") and counts what it dropped.

Wire format of one WebSocket binary message (all little-endian)::

     0        4        8   10   12   14   16
     +--------+--------+----+----+----+----+----------------------+
     | frame  | media  | s1 | s2 | s3 |flag| JPEG bytes ...       |
     | index  | ts (ms)|    |    |    |    |                      |
     +--------+--------+----+----+----+----+----------------------+

``s1/s2/s3`` are the wall latencies of pipeline stages 1..3 for *this* frame in units of 0.1 ms (saturating at
6553.5 ms); ``flag`` bits 0-3 are the protocol version (1), bit 4 = JPEG payload, bit 5 = frames were dropped just
before this one, bit 6 = last frame of the stream, bit 7 = served from cache, bits 8-9 = source (0 job egress,
1 interactive preview).
"""
from __future__ import annotations

import asyncio
from collections import deque
import logging
import struct
import threading
import time
from typing import Any

import numpy as np

HEADER = struct.Struct("<IIHHHH")
HEADER_SIZE = HEADER.size
assert HEADER_SIZE == 16

VERSION = 1
FLAG_JPEG = 1 << 4
FLAG_DROPPED = 1 << 5
FLAG_LAST = 1 << 6
FLAG_CACHED = 1 << 7
SOURCE_JOB, SOURCE_PREVIEW = 0, 1
logger = logging.getLogger(__name__)


def _units(milliseconds: float) -> int:
    return max(0, min(0xFFFF, int(round(milliseconds * 10))))


def pack_header(frame_index: int, timestamp_s: float, stage_ms: tuple[float, float, float] = (0.0, 0.0, 0.0), *,
                flags: int = FLAG_JPEG, source: int = SOURCE_JOB) -> bytes:
    s1, s2, s3 = (_units(v) for v in stage_ms)
    return HEADER.pack(frame_index & 0xFFFFFFFF, max(0, min(0xFFFFFFFF, int(round(timestamp_s * 1000)))), s1, s2, s3,
                       (VERSION | flags | (source & 3) << 8) & 0xFFFF)


def unpack_header(message: bytes | memoryview) -> dict[str, Any]:
    if len(message) < HEADER_SIZE:
        raise ValueError("Frame message shorter than its 16-byte header")
    index, stamp_ms, s1, s2, s3, flags = HEADER.unpack_from(message, 0)
    if flags & 0xF != VERSION:
        raise ValueError(f"Unsupported live-frame protocol version {flags & 0xF}")
    return {"frame_index": index, "timestamp_s": stamp_ms / 1000, "stage_ms": (s1 / 10, s2 / 10, s3 / 10),
            "jpeg": bool(flags & FLAG_JPEG), "dropped": bool(flags & FLAG_DROPPED), "last": bool(flags & FLAG_LAST),
            "cached": bool(flags & FLAG_CACHED), "source": (flags >> 8) & 3}


class JpegEncoder:
    """JPEG encoder with a measured backend: nvJPEG (GPU) > PyTurboJPEG > OpenCV's bundled libjpeg-turbo.

    ``encode`` takes an HWC BGR ``uint8`` array.  With nvJPEG the frame is uploaded on a private non-blocking stream,
    optionally reduced on the GPU to ``max_edge``, entropy-coded by the GPU and only the compressed bytes cross back.
    """

    def __init__(self, quality: int = 82, *, backend: str = "auto", max_edge: int = 1920):
        if not 1 <= quality <= 100 or max_edge < 64:
            raise ValueError("Invalid JPEG settings")
        self.quality, self.max_edge = quality, max_edge
        self.backend, self.fallback_reason = "", None
        self._torch = self._encode_jpeg = self._turbo = self._stream = None
        order = ("nvjpeg", "turbojpeg", "opencv") if backend == "auto" else (backend,)
        reasons = []
        for name in order:
            try:
                # Backend-specific initialization may perform a test encode. Set
                # the candidate first so that test exercises that backend rather
                # than silently probing OpenCV and failing on the first frame.
                self.backend = name
                getattr(self, f"_init_{name}")()
                break
            except Exception as error:
                reasons.append(f"{name}: {type(error).__name__}: {error}")
        else:
            raise RuntimeError("No JPEG backend available: " + "; ".join(reasons))
        self.fallback_reason = "; ".join(reasons) or None

    def _init_nvjpeg(self) -> None:
        import torch
        from torchvision.io import encode_jpeg
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self._torch, self._encode_jpeg = torch, encode_jpeg
        self._stream = torch.cuda.Stream()
        self.encode(np.zeros((64, 64, 3), np.uint8))          # fail now, not on the first live frame

    def _init_turbojpeg(self) -> None:
        from turbojpeg import TurboJPEG
        self._turbo = TurboJPEG()
        self.encode(np.zeros((64, 64, 3), np.uint8))

    def _init_opencv(self) -> None:
        pass

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        edge = max(height, width)
        if edge <= self.max_edge:
            return height, width
        scale = self.max_edge / edge
        return max(2, int(round(height * scale)) & ~1), max(2, int(round(width * scale)) & ~1)

    def encode(self, frame: np.ndarray) -> bytes:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an HWC BGR uint8 frame")
        height, width = self._target_size(*frame.shape[:2])
        if self.backend == "nvjpeg":
            torch = self._torch
            with torch.cuda.stream(self._stream):
                device = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda", non_blocking=True)
                chw = device.flip(-1).permute(2, 0, 1)                          # BGR -> RGB, CHW view
                if (height, width) != tuple(frame.shape[:2]):
                    chw = torch.nn.functional.interpolate(chw[None].float(), size=(height, width), mode="area")[0]
                    chw = chw.round().clamp_(0, 255).to(torch.uint8)
                data = self._encode_jpeg(chw.contiguous(), quality=self.quality)
                payload = data.cpu().numpy().tobytes()                          # compressed bytes only
            self._stream.synchronize()
            return payload
        import cv2
        if (height, width) != tuple(frame.shape[:2]):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if self.backend == "turbojpeg":
            return self._turbo.encode(frame, quality=self.quality)             # BGR native in PyTurboJPEG
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()


class Subscription:
    """One WebSocket's view of the hub: the newest ``depth`` messages, awaitable from the event loop."""

    def __init__(self, hub: "LiveFrameHub", loop: asyncio.AbstractEventLoop, depth: int = 2):
        self.hub, self.loop = hub, loop
        self._messages: deque[bytes] = deque(maxlen=depth)
        self._event = asyncio.Event()
        self.dropped = 0
        self.closed = False

    def _push(self, message: bytes) -> None:          # called from the hub thread
        if len(self._messages) == self._messages.maxlen:
            self.dropped += 1                         # slow browser: oldest frame is discarded, never the render
        self._messages.append(message)
        try:
            self.loop.call_soon_threadsafe(self._event.set)
        except RuntimeError:                          # loop already closed
            self.closed = True

    async def next(self) -> bytes:
        while not self._messages:
            self._event.clear()
            if self._messages:
                break
            await self._event.wait()
        return self._messages.popleft()

    def close(self) -> None:
        self.closed = True
        self.hub._unsubscribe(self)


class LiveFrameHub:
    """Publish frames from any thread; encode and fan out on one background thread."""

    def __init__(self, encoder: JpegEncoder | None = None, *, max_fps: float = 60.0):
        self.encoder = encoder
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._condition = threading.Condition(threading.Lock())
        self._publish_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._slots: list[np.ndarray | None] = [None, None]
        self._pending_slot: int | None = None
        self._pending_meta: dict[str, Any] = {}
        self._encoding_slot: int | None = None
        self._subscribers: list[Subscription] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._last_published = 0.0
        self._dropped_since_send = 0
        self._last_message: bytes | None = None
        self.encoder_error: str | None = None
        # telemetry
        self.published = self.skipped = self.dropped = self.encoded = self.bytes_out = 0
        self.encode_ms_total = 0.0
        self.last_encode_ms = 0.0

    # -- producer side (inference threads): O(frame memcpy), never blocks on encode/network -------------------
    def publish(self, frame: np.ndarray, *, frame_index: int, timestamp_s: float,
                stage_ms: tuple[float, float, float] = (0.0, 0.0, 0.0), last: bool = False,
                cached: bool = False, source: int = SOURCE_JOB, target: "Subscription | None" = None,
                force: bool = False) -> bool:
        """``target`` delivers to one subscriber only (an interactive reply); ``force`` skips the rate limit."""
        if self.encoder_error is not None:
            return False
        if not self._subscribers and not last and target is None:
            return False                                        # nobody is watching: spend nothing
        now = time.perf_counter()
        if not (last or force) and now - self._last_published < self.min_interval:
            self.skipped += 1
            return False
        self._ensure_thread()
        with self._publish_lock:                                # one writer at a time; the encoder never takes this
            with self._condition:
                free = [s for s in (0, 1) if s != self._encoding_slot and s != self._pending_slot]
                if free:
                    slot = free[0]
                else:                                           # both slots busy: newest frame replaces the waiting one
                    slot = self._pending_slot
                    self._pending_slot = None
                    self.dropped += 1
                    self._dropped_since_send += 1
            buffer = self._slots[slot]
            if buffer is None or buffer.shape != frame.shape:
                buffer = self._slots[slot] = np.empty(frame.shape, np.uint8)
            np.copyto(buffer, frame)                            # outside the condition lock; the slot is ours
            with self._condition:
                if self._pending_slot is not None:              # an older unencoded frame is superseded
                    self.dropped += 1
                    self._dropped_since_send += 1
                self._pending_slot = slot
                self._pending_meta = {"frame_index": frame_index, "timestamp_s": timestamp_s, "stage_ms": stage_ms,
                                      "last": last, "cached": cached, "source": source, "target": target}
                self.published += 1
                self._last_published = now
                self._condition.notify()
        return True

    # -- encoder thread ---------------------------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        # Several frame workers can publish at once. Starting the encoder must
        # be atomic, and its imports/GPU probe must never run on those workers.
        with self._start_lock:
            if self.encoder_error is not None or (self._thread is not None and self._thread.is_alive()):
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._loop, name="r3d-live-encode", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        if self.encoder is None:
            try:
                self.encoder = JpegEncoder()
            except Exception as error:
                self.encoder_error = f"{type(error).__name__}: {error}"
                logger.exception("Live preview JPEG encoder initialization failed")
                with self._condition:
                    if self._pending_slot is not None:
                        self.dropped += 1
                        self._pending_slot = None
                    self._stopping = True
                    self._condition.notify_all()
                return
        while True:
            with self._condition:
                while self._pending_slot is None and not self._stopping:
                    self._condition.wait(0.25)
                if self._stopping and self._pending_slot is None:
                    return
                self._encoding_slot, self._pending_slot = self._pending_slot, None
                frame, meta = self._slots[self._encoding_slot], self._pending_meta
                dropped, self._dropped_since_send = self._dropped_since_send, 0
            began = time.perf_counter()
            try:
                jpeg = self.encoder.encode(frame)
            except Exception:                                   # a bad frame must not kill the stream
                with self._condition:
                    self._encoding_slot = None
                continue
            with self._condition:
                self._encoding_slot = None                      # the slot may be overwritten from here on
            self.last_encode_ms = (time.perf_counter() - began) * 1000
            self.encode_ms_total += self.last_encode_ms
            self.encoded += 1
            flags = (FLAG_JPEG | (FLAG_DROPPED if dropped else 0) | (FLAG_LAST if meta["last"] else 0)
                     | (FLAG_CACHED if meta["cached"] else 0))
            message = pack_header(meta["frame_index"], meta["timestamp_s"], meta["stage_ms"], flags=flags,
                                  source=meta["source"]) + jpeg
            self.bytes_out += len(message)
            target = meta.get("target")
            if target is not None:
                if not target.closed:
                    target._push(message)
                continue
            self._last_message = message                        # only broadcast frames are replayed to late joiners
            for subscriber in list(self._subscribers):
                if subscriber.closed:
                    self._unsubscribe(subscriber)
                else:
                    subscriber._push(message)

    # -- consumer side ----------------------------------------------------------------------------------------
    def subscribe(self, loop: asyncio.AbstractEventLoop | None = None, depth: int = 2) -> Subscription:
        subscription = Subscription(self, loop or asyncio.get_running_loop(), depth)
        self._subscribers.append(subscription)
        if self._last_message is not None:
            subscription._push(self._last_message)             # a late joiner sees the current picture at once
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        try:
            self._subscribers.remove(subscription)
        except ValueError:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(2)

    def stats(self) -> dict[str, Any]:
        return {"backend": self.encoder.backend if self.encoder else None,
                "backend_note": self.encoder.fallback_reason if self.encoder else None,
                "encoder_error": self.encoder_error,
                "subscribers": len(self._subscribers), "published": self.published, "skipped": self.skipped,
                "dropped": self.dropped, "encoded": self.encoded, "bytes_out": self.bytes_out,
                "mean_encode_ms": round(self.encode_ms_total / self.encoded, 3) if self.encoded else None,
                "last_encode_ms": round(self.last_encode_ms, 3)}
