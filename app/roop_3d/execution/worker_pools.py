"""CPU decode/preprocess and asynchronous post-encode worker pools.

Decode workers place prepared batches in pinned host slots and enqueue a
nonblocking H2D copy on a dedicated CUDA stream. The separate encode pool owns
FFmpeg/NVENC chunk writes and GOP-aligned stream-copy stitching, keeping file
and mux work off the inference thread.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def _cpu_workers(workers: int | None) -> int:
    cores = os.cpu_count() or 1
    count = min(16, max(1, cores - 1)) if workers is None else workers
    if not 1 <= count <= 32:
        raise ValueError("Worker count must be within 1..32")
    return count


class DeviceFrameLease:
    """One pinned-input/device-output slot held until its downstream CUDA event completes."""

    def __init__(self, owner: "DecodePreprocessPool", slot: int, tensor: Any, ready_event: Any):
        self.owner, self.slot = owner, slot
        self.tensor, self.ready_event = tensor, ready_event
        self._released = False

    def wait_on(self, stream):
        if self.ready_event is not None:
            stream.wait_event(self.ready_event)
        return self.tensor

    def release_after(self, event):
        if self._released:
            raise ValueError("Device frame lease released twice")
        self._released = True
        self.owner._release(self.slot, event)
        self.tensor = None

    def release(self):
        if self._released:
            raise ValueError("Device frame lease released twice")
        event = self.owner.torch.cuda.Event()
        event.record(self.owner.stream)
        self.release_after(event)


@dataclass(frozen=True)
class _TransferSlot:
    host: Any
    device: Any


class DecodePreprocessPool:
    """Multi-core CPU preparation pool with pinned buffers and async H2D transfer."""

    def __init__(self, shape: Sequence[int], *, dtype=np.float32, workers: int | None = None,
                 slots: int = 4, device_id: int = 0, torch_module=None):
        if not shape or any(int(value) <= 0 for value in shape) or not 2 <= slots <= 32 or device_id < 0:
            raise ValueError("Invalid decode/preprocess buffer configuration")
        import torch as default_torch
        self.torch = torch_module or default_torch
        if not self.torch.cuda.is_available():
            raise RuntimeError("DecodePreprocessPool requires CUDA for pinned H2D staging")
        self.shape, self.numpy_dtype = tuple(int(value) for value in shape), np.dtype(dtype)
        self.device_id, self.device = device_id, f"cuda:{device_id}"
        self.worker_count, self.slot_count = _cpu_workers(workers), slots
        self.stream = self.torch.cuda.Stream(device=device_id)
        torch_dtype = self._torch_dtype(self.numpy_dtype)
        with self.torch.cuda.device(device_id):
            self._slots = [
                _TransferSlot(
                    self.torch.empty(self.shape, dtype=torch_dtype, pin_memory=True),
                    self.torch.empty(self.shape, dtype=torch_dtype, device=self.device),
                )
                for _ in range(slots)
            ]
        if any(not item.host.is_pinned() for item in self._slots):
            raise RuntimeError("Pinned host allocation failed")
        self._free = deque(range(slots))
        self._retired: dict[int, Any] = {}
        self._condition = threading.Condition(threading.Lock())
        self._closed = False
        self._executor_shutdown = False
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count, thread_name_prefix="roop3d-decode")
        self.submitted = self.completed = 0

    def _torch_dtype(self, dtype):
        names = {
            np.dtype(np.float16): self.torch.float16, np.dtype(np.float32): self.torch.float32,
            np.dtype(np.float64): self.torch.float64, np.dtype(np.uint8): self.torch.uint8,
            np.dtype(np.int8): self.torch.int8, np.dtype(np.int16): self.torch.int16,
            np.dtype(np.int32): self.torch.int32, np.dtype(np.int64): self.torch.int64,
            np.dtype(np.bool_): self.torch.bool,
        }
        try:
            return names[dtype]
        except KeyError:
            raise TypeError(f"Unsupported transfer dtype: {dtype}") from None

    def submit(self, source: Any, decoder: Callable[[Any], Any],
               preprocess: Callable[[Any], Any] | None = None) -> Future:
        if self._closed:
            raise RuntimeError("DecodePreprocessPool is closed")
        self.submitted += 1
        return self.executor.submit(self._decode_and_transfer, source, decoder, preprocess)

    def _reclaim(self):
        for slot, event in tuple(self._retired.items()):
            if event.query():
                del self._retired[slot]
                self._free.append(slot)

    def _acquire(self) -> int:
        with self._condition:
            while True:
                self._reclaim()
                if self._closed:
                    raise RuntimeError("DecodePreprocessPool is closed")
                if self._free:
                    return self._free.popleft()
                self._condition.wait(0.01)

    def _decode_and_transfer(self, source, decoder, preprocess):
        decoded = decoder(source)
        prepared = preprocess(decoded) if preprocess is not None else decoded
        if getattr(prepared, "is_cuda", False):
            raise ValueError("Decode/preprocess workers must return host arrays before H2D transfer")
        array = np.ascontiguousarray(prepared, dtype=self.numpy_dtype)
        if tuple(array.shape) != self.shape:
            raise ValueError(f"Prepared frame shape {array.shape} does not match {self.shape}")
        slot_index = self._acquire()
        slot = self._slots[slot_index]
        try:
            np.copyto(slot.host.numpy(), array)
            with self.torch.cuda.stream(self.stream):
                slot.device.copy_(slot.host, non_blocking=True)
                ready = self.torch.cuda.Event()
                ready.record(self.stream)
            self.completed += 1
            return DeviceFrameLease(self, slot_index, slot.device, ready)
        except BaseException:
            with self._condition:
                self._free.append(slot_index)
                self._condition.notify()
            raise

    def _release(self, slot: int, event):
        with self._condition:
            if slot in self._free or slot in self._retired:
                raise ValueError("Transfer slot released more than once")
            self._retired[slot] = event
            self._reclaim()
            self._condition.notify_all()

    def close(self, *, wait: bool = True):
        if self._closed and self._executor_shutdown and not self._slots:
            return
        self._closed = True
        if not self._executor_shutdown:
            self.executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor_shutdown = True
        if not wait:
            return
        if wait:
            with self._condition:
                held = self.slot_count - len(self._free) - len(self._retired)
                if held:
                    raise RuntimeError("Release active device frame leases before closing DecodePreprocessPool")
                while self._retired:
                    self._reclaim()
                    if self._retired:
                        self._condition.wait(0.01)
            self.stream.synchronize()
        with self._condition:
            if len(self._free) != self.slot_count:
                # Leases are caller-owned; refusing teardown prevents reuse-after-free.
                raise RuntimeError("Cannot close DecodePreprocessPool while device frame leases are active")
            self._slots.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class GopChunk:
    """One encoded frame range beginning on an IDR/GOP boundary."""

    start_frame: int
    end_frame: int  # exclusive
    payload: Any
    gop_size: int = 1
    final: bool = False

    def __post_init__(self):
        if self.start_frame < 0 or self.end_frame <= self.start_frame or self.gop_size < 1:
            raise ValueError("Invalid GOP chunk frame range")
        if self.start_frame % self.gop_size != 0:
            raise ValueError("Chunk start must be GOP aligned")
        if self.end_frame % self.gop_size != 0 and not self.final:
            raise ValueError("Non-final chunk end must be GOP aligned")


def _concat_quote(path: Path) -> str:
    # FFmpeg concat files use single quoted paths; embedded apostrophes are
    # escaped using the concat demuxer's documented close/escape/reopen form.
    value = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
    return f"file '{value}'"


def stitch_gop_chunks(chunks: Iterable[str | Path], destination: str | Path, *,
                      audio_source: str | Path | None = None, ffmpeg: str = "ffmpeg",
                      timeout: float = 600.0) -> Path:
    """Losslessly concat already-encoded GOP chunks and optionally remux source audio."""
    paths = tuple(Path(path).resolve(strict=True) for path in chunks)
    target = Path(destination).resolve()
    if not paths or target.exists() or any(not path.is_file() for path in paths):
        raise ValueError("Stitching requires existing chunks and a fresh destination")
    binary = shutil.which(ffmpeg) if not Path(ffmpeg).is_file() else str(Path(ffmpeg).resolve())
    if not binary:
        raise FileNotFoundError(f"FFmpeg binary not found: {ffmpeg}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roop3d-gop-") as temp_dir:
        list_path = Path(temp_dir) / "chunks.txt"
        list_path.write_text("\n".join(_concat_quote(path) for path in paths) + "\n", encoding="utf-8")
        command = [binary, "-nostdin", "-v", "error", "-n", "-f", "concat", "-safe", "0",
                   "-i", str(list_path)]
        if audio_source is not None:
            audio = Path(audio_source).resolve(strict=True)
            if not audio.is_file():
                raise ValueError("Audio remux source must be a local file")
            command.extend(["-i", str(audio), "-map", "0:v:0", "-map", "1:a?"])
        else:
            command.extend(["-map", "0:v:0"])
        command.extend(["-c", "copy", "-movflags", "+faststart", str(target)])
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        if result.returncode:
            target.unlink(missing_ok=True)
            message = result.stderr.decode(errors="replace")[:2048]
            raise RuntimeError(f"FFmpeg GOP stitch failed ({result.returncode}): {message}")
    return target


def encode_nvenc_chunk(source: str | Path, destination: str | Path, *, codec: str = "h264",
                       gop_size: int = 60, ffmpeg: str = "ffmpeg", timeout: float = 600.0) -> Path:
    """Encode one prepared video chunk with NVENC in lossless mode on a pool thread."""
    source_path = Path(source).resolve(strict=True)
    target = Path(destination).resolve()
    if not source_path.is_file() or target.exists() or codec not in ("h264", "hevc") or gop_size < 1:
        raise ValueError("NVENC chunk encoding requires a source file, fresh output, codec, and GOP size")
    binary = shutil.which(ffmpeg) if not Path(ffmpeg).is_file() else str(Path(ffmpeg).resolve())
    if not binary:
        raise FileNotFoundError(f"FFmpeg binary not found: {ffmpeg}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, "-nostdin", "-v", "error", "-n", "-i", str(source_path), "-an",
               "-c:v", "h264_nvenc" if codec == "h264" else "hevc_nvenc",
               "-tune", "lossless", "-rc", "constqp", "-qp", "0",
               "-g", str(gop_size), "-keyint_min", str(gop_size), "-sc_threshold", "0",
               "-forced-idr", "1", "-movflags", "+faststart", str(target)]
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        target.unlink(missing_ok=True)
        message = result.stderr.decode(errors="replace")[:2048]
        raise RuntimeError(f"FFmpeg NVENC chunk encode failed ({result.returncode}): {message}")
    return target


class PostEncodeWorkerPool:
    """Dedicated encoder/mux executor; submitted chunks are stitched in frame order."""

    def __init__(self, *, workers: int = 1, thread_name_prefix: str = "roop3d-encode"):
        if not 1 <= workers <= 8:
            raise ValueError("Post-encode worker count must be within 1..8")
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix)
        self._lock = threading.Lock()
        self._chunks: list[tuple[GopChunk, Future]] = []
        self._closed = False

    def submit_chunk(self, chunk: GopChunk, encoder: Callable[[GopChunk], str | Path] | None = None) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("PostEncodeWorkerPool is closed")
            if encoder is None:
                if not isinstance(chunk.payload, (str, Path)):
                    raise TypeError("An encoder callback is required when chunk payload is not a file path")
                future = self.executor.submit(lambda: Path(chunk.payload).resolve(strict=True))
            else:
                future = self.executor.submit(encoder, chunk)
            self._chunks.append((chunk, future))
        return future

    def submit_nvenc_chunk(self, chunk: GopChunk, destination: str | Path, *,
                           codec: str = "h264", ffmpeg: str = "ffmpeg") -> Future:
        """Encode this chunk on the post-encode workers, then include it in final stream-copy stitch."""
        return self.submit_chunk(
            chunk,
            lambda item: encode_nvenc_chunk(item.payload, destination, codec=codec,
                                            gop_size=item.gop_size, ffmpeg=ffmpeg),
        )

    def finalize(self, destination: str | Path, *, audio_source: str | Path | None = None,
                 stitcher: Callable[..., Path] = stitch_gop_chunks, ffmpeg: str = "ffmpeg") -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("PostEncodeWorkerPool is closed")
            self._closed = True

        def finish():
            with self._lock:
                snapshot = tuple(self._chunks)
            ordered = sorted(snapshot, key=lambda item: item[0].start_frame)
            chunks = []
            previous_end = 0
            for chunk, future in ordered:
                path = Path(future.result()).resolve(strict=True)
                if chunk.start_frame != previous_end:
                    raise ValueError(f"GOP chunks are not contiguous at frame {previous_end}")
                previous_end = chunk.end_frame
                chunks.append(path)
            return stitcher(chunks, destination, audio_source=audio_source, ffmpeg=ffmpeg)

        return self.executor.submit(finish)

    def close(self, *, wait: bool = True):
        with self._lock:
            self._closed = True
        self.executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["DecodePreprocessPool", "DeviceFrameLease", "GopChunk", "PostEncodeWorkerPool",
           "encode_nvenc_chunk", "stitch_gop_chunks"]
