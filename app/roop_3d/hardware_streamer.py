"""Bounded host-frame transport and explicit FFmpeg NVDEC/NVENC boundaries."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from roop_3d.models.owned_files import owned_path
from roop_3d.media.probe import probe
from roop_3d.reliability.processes import REGISTRY, spawn_kwargs, terminate


@dataclass(frozen=True)
class FrameMetadata:
    index: int
    pts: int
    time_base_num: int
    time_base_den: int
    keyframe: bool = False


class SharedMemoryFrameRing:
    """Single-producer/single-consumer ring; one ring per worker for fanout.

    Only handles/synchronizers are serialized once during process spawn. Frames
    and fixed-width metadata never enter a Queue, Pipe, pickle, or disk file.
    A lease keeps the slot unavailable to the producer until the consumer exits.
    """
    META = np.dtype([("index", "<i8"), ("pts", "<i8"), ("num", "<i8"), ("den", "<i8"), ("key", "u1")])

    def __init__(self, shape: tuple[int, int, int], capacity: int = 4, *, context=None):
        if len(shape) != 3 or shape[2] != 3 or min(shape) < 1 or not 2 <= capacity <= 32:
            raise ValueError("Ring expects bounded HWC BGR frames")
        size = int(np.prod(shape)) * capacity
        if size > 512 * 1024**2:
            raise ValueError("Frame ring exceeds 512 MiB host budget")
        ctx = context or mp.get_context("spawn")
        self.shape, self.capacity = tuple(shape), capacity
        self.frames = SharedMemory(create=True, size=size)
        try:
            self.meta = SharedMemory(create=True, size=self.META.itemsize * capacity)
        except BaseException:
            self.frames.close()
            self.frames.unlink()
            raise
        self.empty, self.full = ctx.Semaphore(capacity), ctx.Semaphore(0)
        self.write_index, self.read_index = ctx.Value("Q", 0), ctx.Value("Q", 0)
        self.finished, self.cancelled = ctx.Event(), ctx.Event()
        self.owner = True
        self.closed = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state["frames"], state["meta"] = self.frames.name, self.meta.name
        state["owner"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.frames = SharedMemory(name=state["frames"])
        self.meta = SharedMemory(name=state["meta"])

    def _wait(self, semaphore, timeout: float, *, reader=False):
        deadline = time.monotonic() + timeout
        while not semaphore.acquire(timeout=min(0.1, max(0, deadline - time.monotonic()))):
            if self.cancelled.is_set():
                raise RuntimeError("Frame ring cancelled")
            if reader and self.finished.is_set():
                return False
            if time.monotonic() >= deadline:
                raise TimeoutError("Frame ring stalled")
        return True

    def put(self, frame: np.ndarray, metadata: FrameMetadata, timeout: float = 30):
        if self.closed or self.finished.is_set() or self.cancelled.is_set():
            raise RuntimeError("Frame ring is closed")
        numbers = (metadata.index, metadata.pts, metadata.time_base_num, metadata.time_base_den)
        if (frame.shape != self.shape or frame.dtype != np.uint8 or metadata.time_base_den <= 0
                or metadata.index < 0 or metadata.time_base_num <= 0
                or any(type(v) is not int or not -(2**63) <= v < 2**63 for v in numbers)):
            raise ValueError("Frame/metadata contract mismatch")
        self._wait(self.empty, timeout)
        index = self.write_index.value % self.capacity
        np.copyto(np.ndarray((self.capacity, *self.shape), np.uint8, buffer=self.frames.buf)[index], frame)
        np.ndarray((self.capacity,), self.META, buffer=self.meta.buf)[index] = (
            metadata.index, metadata.pts, metadata.time_base_num, metadata.time_base_den, metadata.keyframe)
        self.write_index.value += 1
        self.full.release()

    @contextmanager
    def lease(self, timeout: float = 30):
        if self.closed:
            raise RuntimeError("Frame ring handle is closed")
        if not self._wait(self.full, timeout, reader=True):
            yield None
            return
        index = self.read_index.value % self.capacity
        row = np.ndarray((self.capacity,), self.META, buffer=self.meta.buf)[index]
        metadata = FrameMetadata(int(row["index"]), int(row["pts"]), int(row["num"]), int(row["den"]), bool(row["key"]))
        frame = np.ndarray((self.capacity, *self.shape), np.uint8, buffer=self.frames.buf)[index]
        frame.setflags(write=False)
        try:
            yield frame, metadata
        finally:
            del frame, row
            self.read_index.value += 1
            self.empty.release()

    def finish(self):
        self.finished.set()

    def cancel(self):
        self.cancelled.set()

    def close(self):
        if not self.closed:
            self.frames.close()
            self.meta.close()
            if self.owner:
                self.frames.unlink()
                self.meta.unlink()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cancel()
        self.close()


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is unavailable in this app's environment")
    return path


def _process_environment():
    from roop_3d.audit.project_paths import application_root
    from roop_3d.core.storage import owned
    root = application_root()
    cache = owned(root, "roop_3d/cache/cuda")
    temporary = owned(root, "roop_3d/.runtime/native-tmp")
    cache.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    return {**os.environ, "CUDA_CACHE_PATH": str(cache), "TMP": str(temporary),
            "TEMP": str(temporary), "TMPDIR": str(temporary)}


#: FFmpeg command-line names for the H.273 codes this chain has been validated for (8-bit 4:2:0 SDR).
_FF_MATRIX = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020nc"}
_FF_PRIMARIES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020"}
_FF_TRC = {1: "bt709", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 8: "linear", 13: "iec61966-2-1", 14: "bt2020-10"}


def stream_color_supported(color) -> str | None:
    """None when the raw-stream chain reproduces this material's colour; otherwise why the reference path is needed.

    The chain decodes through FFmpeg's scaler (which honours the file's matrix and range tags) and encodes 8-bit
    4:2:0. It cannot resolve an untagged matrix or range the way the reference reader does, and it has no
    path for >8-bit or non-4:2:0 material, so those require the colour-managed reference path.
    """
    if color.is_hdr:
        return "HDR transfer functions require the preserved color-managed reference path"
    if color.bit_depth != 8 or color.chroma != "420":
        return f"{color.bit_depth}-bit {color.chroma} material requires the color-managed reference path"
    if color.matrix not in _FF_MATRIX or color.primaries not in _FF_PRIMARIES or color.transfer not in _FF_TRC:
        return "This colour description has no validated raw-stream conversion"
    untagged = [name for name in ("matrix", "range") if color.provenance.get(name) != "tagged"]
    if untagged:
        return f"Untagged {'/'.join(untagged)} must be resolved by the color-managed reference path"
    return None


class NvdecFrameSource:
    def __init__(self, path: Path, *, strict: bool | None = None, hardware: bool = True):
        import os
        self.path = owned_path(Path(path))
        self.strict = os.environ.get("ROOP_STRICT_NVDEC") == "1" if strict is None else strict
        self.hardware = hardware
        self._cancelled = threading.Event()
        self._process_lock = threading.Lock()
        self._process = None
        if self.strict and not hardware:
            raise ValueError("Strict NVDEC cannot select software decode")
        self.info = probe(self.path).videos[0]
        self.backend, self.fallback_reason = "UNINITIALIZED", None
        self.rate = self.info.average_rate or self.info.nominal_rate
        if (self.rate is None or self.rate <= 0 or self.info.vfr is not False or self.info.rotation % 360
                or self.info.start_time not in (None, 0)):
            raise ValueError("Raw streaming requires verified CFR, unrotated input; use the timing-preserving reference reader")
        self.shape = (self.info.height, self.info.width, 3)
        if self.info.color_trc in (16, 18):
            raise ValueError("HDR transfer functions require the preserved color-managed reference path")
        self.color = self.info.color
        reason = stream_color_supported(self.color)
        if reason:
            raise ValueError(reason)
        from roop_3d.media.probe import open_media
        self.keyframes = set()
        with open_media(self.path) as container:
            stream = container.streams.video[0]
            if stream.sample_aspect_ratio not in (None, Fraction(1)):
                raise ValueError("Non-square pixels require the reference media path")
            seen, count, maximum = bytearray(), 0, -1
            for packet in container.demux(stream):
                if packet.size and packet.pts is not None:
                    position = Fraction(packet.pts) * stream.time_base * self.rate
                    if position.denominator != 1 or not 0 <= position < 100000000:
                        raise ValueError("Packet timing is not a bounded exact CFR frame grid")
                    index = int(position)
                    byte, bit = divmod(index, 8)
                    if len(seen) <= byte:
                        seen.extend(bytes(byte + 1 - len(seen)))
                    if seen[byte] & (1 << bit):
                        raise ValueError("Duplicate presentation timestamp in raw-stream input")
                    seen[byte] |= 1 << bit
                    count += 1
                    maximum = max(maximum, index)
                    if packet.is_keyframe:
                        self.keyframes.add(index)
            if count == 0 or count != maximum + 1:
                raise ValueError("Packet grid contains timing gaps; use the timing-preserving reference path")
            self.expected_frames = count

    def _ffmpeg_frames(self, hardware):
        if self._cancelled.is_set():
            return
        command = [_binary("ffmpeg"), "-nostdin", "-v", "error"]
        if hardware:
            command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        command += ["-protocol_whitelist", "file,pipe", "-i", str(self.path), "-map", "0:v:0", "-an", "-sn", "-dn"]
        if hardware:
            command += ["-vf", "hwdownload,format=nv12,format=bgr24"]
        command += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        process = REGISTRY.track(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                                  env=_process_environment(), **spawn_kwargs()), "ffmpeg_decode")
        with self._process_lock:
            self._process = process
            if self._cancelled.is_set():
                process.terminate()
        errors = bytearray()
        def drain():
            for chunk in iter(lambda: process.stderr.read(4096), b""):
                if len(errors) < 16384:
                    errors.extend(chunk[:16384 - len(errors)])
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        count = 0
        try:
            size = int(np.prod(self.shape))
            while True:
                data = bytearray()
                while len(data) < size:
                    chunk = process.stdout.read(size - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if not data:
                    break
                if len(data) != size:
                    raise RuntimeError("Truncated FFmpeg frame")
                self.backend = "NVDEC" if hardware else "FFMPEG_SOFTWARE"
                yield np.frombuffer(data, np.uint8).reshape(self.shape), FrameMetadata(count, count,
                    self.rate.denominator, self.rate.numerator, count in self.keyframes)
                count += 1
            code = process.wait(timeout=30)
            if code or count != self.expected_frames:
                raise RuntimeError(f"FFmpeg decode failed ({code}): {errors.decode(errors='replace')}")
        finally:
            terminate(process, grace=10)  # Only the subprocess created by this source; terminate -> wait -> kill.
            process.stdout.close()
            thread.join(timeout=5)
            process.stderr.close()
            REGISTRY.untrack(process)
            with self._process_lock:
                if self._process is process:
                    self._process = None

    def cancel(self):
        """Unblock a stalled read by stopping only this instance's FFmpeg child."""
        self._cancelled.set()
        with self._process_lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    def gpu_frames(self, *, device=0, statistics=False):
        """Explicit CUDA RGB leases; host-oriented frames() remains the fallback.

        A CUDA caller must explicitly choose the existing host pipeline after an
        unavailable backend. Never silently change a yielded frame's residence.
        """
        from roop_3d.nv_video_codec import NvVideoDecoder
        if not self.hardware:
            raise ValueError("CUDA frames require hardware decode")
        count = 0
        decoder = NvVideoDecoder(self.path, device=device, statistics=statistics)
        for frame in decoder.frames():
            if self._cancelled.is_set():
                return
            if count >= self.expected_frames or tuple(frame.tensor.shape) != self.shape:
                raise RuntimeError("GPU decoder frame count/shape mismatch")
            self.backend = decoder.backend
            yield frame, FrameMetadata(count, count, self.rate.denominator, self.rate.numerator,
                                        count in self.keyframes)
            count += 1
        if count != self.expected_frames:
            raise RuntimeError("GPU decoder returned a truncated stream")

    def frames(self):
        emitted = 0
        attempts = [True] if self.strict else ([True, False] if self.hardware else [False])
        for hardware in attempts:
            try:
                for item in self._ffmpeg_frames(hardware):
                    emitted += 1
                    yield item
                return
            except (RuntimeError, OSError) as error:
                if emitted or self.strict or self._cancelled.is_set():
                    raise  # Never restart halfway and duplicate frames.
                self.fallback_reason = str(error)
        cap = cv2.VideoCapture(str(self.path))
        try:
            index = 0
            while cap.isOpened():
                if self._cancelled.is_set():
                    break
                success, frame = cap.read()
                if not success:
                    break
                self.backend = "OPENCV_SOFTWARE"
                yield frame, FrameMetadata(index, index, self.rate.denominator, self.rate.numerator)
                index += 1
            if index != self.expected_frames:
                raise RuntimeError("All local video decoders failed")
        finally:
            cap.release()


class NVHardwareVideoWriter:
    @staticmethod
    def from_cuda(destination, shape, rate, **kwargs):
        from roop_3d.nv_video_codec import NvVideoEncoder
        return NvVideoEncoder(destination, shape, rate, **kwargs)

    def __init__(self, destination: Path, shape: tuple[int, int, int], rate: Fraction, *,
                 codec: str = "h264_nvenc", hardware: bool = True, strict: bool = False,
                 lut_path: Path | None = None, lut_bake: bool = False, color=None):
        """``color`` (a resolved ``ColorDescription``) makes the RGB->YCbCr matrix and range explicit and tags the file
        with them. Without it the encoder uses FFmpeg's untagged BT.601 default, which shifts every pixel of
        BT.709 material by ~40/255; callers that know the source colour must pass it."""
        self.destination = owned_path(destination)
        if self.destination.exists() or self.destination.suffix.lower() != ".mp4":
            raise ValueError("Writer requires a fresh owned MP4 path")
        if codec not in ("h264_nvenc", "hevc_nvenc") or len(shape) != 3 or shape[2] != 3 or min(shape) < 1 or rate <= 0:
            raise ValueError("Invalid video output contract")
        self.shape, self.rate, self.frames = shape, rate, 0
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.codec = codec if hardware else "libx264"
        if hardware:
            # Real encoder smoke test, not advertisement. Do not retry a partial stream.
            result = subprocess.run([_binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"color=size={shape[1]}x{shape[0]}:rate=1",
                "-frames:v", "1", "-c:v", codec, "-f", "null", "-"], capture_output=True, timeout=30, env=_process_environment())
            if result.returncode:
                if strict:
                    raise RuntimeError("Strict NVENC encoder smoke test failed: " + result.stderr.decode(errors="replace")[:2048])
                self.codec = "libx264"
        elif strict:
            raise ValueError("Strict NVENC cannot select a software encoder")
        command = [_binary("ffmpeg"), "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s:v", f"{shape[1]}x{shape[0]}", "-r", str(rate), "-i", "pipe:0", "-an"]
        filters: list[str] = []
        tags: list[str] = []
        if color is not None:
            if color.matrix not in _FF_MATRIX or color.primaries not in _FF_PRIMARIES or color.transfer not in _FF_TRC:
                raise ValueError("No validated raw-stream conversion for this colour description")
            span = "pc" if color.range == 2 else "tv"
            matrix = _FF_MATRIX[color.matrix]
            tags = ["-colorspace", matrix, "-color_primaries", _FF_PRIMARIES[color.primaries], "-color_trc", _FF_TRC[color.transfer],
                    "-color_range", span]
        if lut_bake:
            if lut_path is None:
                raise ValueError("LUT baking requires a local .cube file")
            lut = owned_path(lut_path)
            if lut.suffix.lower() != ".cube" or not lut.is_file() or lut.stat().st_size > 16 * 1024**2:
                raise ValueError("LUT must be a bounded local .cube file")
            # FFmpeg filter escaping is independent of shell escaping; no shell is used.
            escaped = lut.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
            filters.append(f"lut3d=file='{escaped}'")
        if color is not None:
            # ``scale`` performs the conversion with the explicit matrix/range; ``setparams`` stamps all four colour
            # properties on the frames, which is where FFmpeg 8's encoder reads them from (the -color_* options alone
            # leave primaries and transfer unspecified).
            filters += [f"scale=out_color_matrix={matrix}:out_range={span}:flags=accurate_rnd+full_chroma_int", "format=yuv420p",
                        f"setparams=colorspace={matrix}:color_primaries={_FF_PRIMARIES[color.primaries]}:"
                        f"color_trc={_FF_TRC[color.transfer]}:range={span}"]
        if filters:
            command += ["-vf", ",".join(filters)]
        command += ["-c:v", self.codec, "-pix_fmt", "yuv420p", *tags, "-bf", "0", "-movflags", "+faststart", str(self.destination)]
        self.process = REGISTRY.track(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                                       stderr=subprocess.PIPE, env=_process_environment(), **spawn_kwargs()),
                                      "ffmpeg_encode")
        self.errors = bytearray()
        def drain():
            for chunk in iter(lambda: self.process.stderr.read(4096), b""):
                if len(self.errors) < 16384:
                    self.errors.extend(chunk[:16384 - len(self.errors)])
        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()
        self.closed = False

    def write(self, image: np.ndarray):
        if self.closed or image.shape != self.shape or image.dtype != np.uint8:
            raise ValueError("Writer frame/stream contract failed")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frames += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                terminate(self.process, grace=10)
                raise
            finally:
                self.thread.join(timeout=5)
                self.process.stderr.close()
                REGISTRY.untrack(self.process)
            if code or not self.frames:
                raise RuntimeError(f"Video encode failed ({code}): {self.errors.decode(errors='replace')}")

    def abort(self):
        if not self.closed:
            self.closed = True
            try:
                self.process.stdin.close()
            except OSError:
                pass  # the encoder already died (BrokenPipe on flush); we are aborting anyway
            terminate(self.process, grace=10)
            self.thread.join(timeout=5)
            self.process.stderr.close()
            REGISTRY.untrack(self.process)
        self.destination.unlink(missing_ok=True)

    @staticmethod
    def remux_audio(video: Path, source: Path, destination: Path):
        destination = owned_path(destination)
        if destination.exists():
            raise ValueError("Remux output already exists")
        result = subprocess.run([_binary("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", str(owned_path(video)),
            "-i", str(owned_path(source)), "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-map_metadata", "1",
            "-movflags", "+faststart", str(destination)], capture_output=True, timeout=120, env=_process_environment())
        if result.returncode:
            destination.unlink(missing_ok=True)
            raise RuntimeError("Lossless audio/video remux failed; incompatible audio is never silently transcoded")


class SegmentWorkspace:
    def __init__(self, directory: Path):
        self.directory = owned_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{time.time_ns()}_{uuid.uuid4().hex}"
        self.segments: list[Path] = []

    def next_path(self):
        path = owned_path(self.directory / f"segment_{self.run_id}_{len(self.segments):06d}.mp4")
        self.segments.append(path)
        return path

    def concat(self, destination: Path):
        destination = owned_path(destination)
        if not self.segments or destination.exists():
            raise ValueError("Concat requires nonempty segments and a fresh output")
        entries = []
        for index, segment in enumerate(self.segments):
            owned_path(segment)
            if not segment.is_file() or segment.stat().st_size == 0:
                raise ValueError(f"Missing segment {index}")
            entries.append(f"file '{segment.name}'\n")
        listing = owned_path(self.directory / f"segments_{self.run_id}.ffconcat")
        listing.write_text("ffconcat version 1.0\n" + "".join(entries), encoding="utf-8")
        result = subprocess.run([_binary("ffmpeg"), "-nostdin", "-v", "error", "-n", "-f", "concat", "-safe", "1",
            "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(destination)], capture_output=True, timeout=120, env=_process_environment())
        if result.returncode:
            destination.unlink(missing_ok=True)
            raise RuntimeError("Segment stream-copy concatenation failed")
        owned_path(self.directory / f"segments_{self.run_id}.json").write_text(json.dumps(
            {"segments": [p.name for p in self.segments], "output": destination.name}, indent=2))
        return destination
