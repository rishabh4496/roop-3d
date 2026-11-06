"""Stage 9 hardware video writer: NVENC with automatic CPU fallback, optional .cube LUT bake, trimmed audio mux.

Frames (8-bit BGR ``ndarray``) are piped to one ``ffmpeg`` process. ``h264_nvenc`` / ``hevc_nvenc`` is tried first;
if NVENC is missing, busy, or the process dies while the first frames are written, the writer restarts on
``libx264`` / ``libx265`` and replays the frames it buffered, so a caller never sees the difference except through
``encoder`` and ``fallback_reason``.

This is a segment/intermediate writer. It is deliberately separate from ``media.video_writer.encode_video``, which
owns the validated, colour-managed, whole-file export path.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.segment_dispatcher import find_ffmpeg

Codec = Literal["h264", "hevc"]
_NVENC = {"h264": "h264_nvenc", "hevc": "hevc_nvenc"}
_CPU = {"h264": "libx264", "hevc": "libx265"}
_REPLAY_FRAMES = 8
_probe_cache: dict[tuple[str, str], tuple[bool, str]] = {}


def probe_nvenc(codec: Codec = "h264", ffmpeg: str | None = None) -> tuple[bool, str]:
    """Whether a tiny real NVENC encode succeeds right now. Cached per process (and per ffmpeg binary)."""
    exe = ffmpeg or find_ffmpeg()
    key = (exe, codec)
    if key not in _probe_cache:
        cmd = [exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=10", "-frames:v", "3",
               "-c:v", _NVENC[codec], "-f", "null", "-"]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            ok, reason = run.returncode == 0, (run.stderr.strip().splitlines() or ["unknown"])[-1]
        except (OSError, subprocess.SubprocessError) as err:
            ok, reason = False, str(err)
        _probe_cache[key] = (ok, "" if ok else reason)
    return _probe_cache[key]


class NVHardwareVideoWriter:
    """``with NVHardwareVideoWriter(path, w, h, fps) as writer: writer.write(frame_bgr)``.

    ``lut_path`` + ``lut_bake=True`` applies the ``.cube`` 3D LUT in the ffmpeg filtergraph (``lut3d``) before the
    encode. ``lut_bake=False`` leaves pixels untouched and copies the LUT next to the output as ``<name>.cube`` so
    the grade travels with the file without being burned in.
    """

    def __init__(self, path: Path | str, width: int, height: int, fps: float, *, codec: Codec = "h264",
                 prefer_hardware: bool = True, quality: int = 19, lut_path: Path | str | None = None,
                 lut_bake: bool = True, gop: int | None = None, ffmpeg: str | None = None):
        if codec not in _NVENC:
            raise ValueError("codec must be 'h264' or 'hevc'")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise DomainError("WRITER_BAD_SIZE", f"yuv420p output needs positive even dimensions, got {width}x{height}", 422)
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = Path(path)
        self.width, self.height, self.fps, self.codec, self.quality = int(width), int(height), float(fps), codec, int(quality)
        self.gop = int(gop or max(1, round(fps)))
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.lut_path = Path(lut_path).resolve() if lut_path else None
        self.lut_bake = bool(lut_bake)
        if self.lut_path is not None and (not self.lut_path.is_file() or self.lut_path.suffix.lower() != ".cube"):
            raise DomainError("LUT_INVALID", f"LUT must be an existing .cube file: {self.lut_path}", 422)
        self.frames_written = 0
        self.encoder = ""
        self.fallback_reason: str | None = None
        self.lut_applied = False
        self.lut_sidecar: Path | None = None
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self._replay: list[bytes] = []
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if prefer_hardware:
            ok, reason = probe_nvenc(codec, self.ffmpeg)
            if ok:
                self._start(_NVENC[codec])
            else:
                self.fallback_reason = f"NVENC unavailable: {reason}"
                self._start(_CPU[codec])
        else:
            self.fallback_reason = "hardware encoding not requested"
            self._start(_CPU[codec])

    # -- process management -------------------------------------------------------------------------------------

    def _command(self, encoder: str, lut_name: str | None) -> list[str]:
        chain = ([f"lut3d=file={lut_name}"] if lut_name else []) + ["format=yuv420p"]
        cmd = [self.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.width}x{self.height}", "-r", repr(self.fps), "-i", "-", "-an",
               "-vf", ",".join(chain), "-c:v", encoder, "-g", str(self.gop)]
        if encoder.endswith("_nvenc"):
            cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(self.quality), "-b:v", "0"]
        else:
            cmd += ["-preset", "medium", "-crf", str(self.quality)]
        if encoder == "libx265":
            cmd += ["-tag:v", "hvc1"]
        elif encoder == "hevc_nvenc":
            cmd += ["-tag:v", "hvc1"]
        return cmd + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path.resolve())]

    def _start(self, encoder: str) -> None:
        cwd = None
        lut_name = None
        if self.lut_path is not None and self.lut_bake:
            # ffmpeg filter-option escaping of Windows drive paths is a minefield; run beside the LUT instead.
            cwd, lut_name = str(self.lut_path.parent), self.lut_path.name
            self.lut_applied = True
        elif self.lut_path is not None:
            self.lut_sidecar = self.path.with_suffix(".cube")
            shutil.copyfile(self.lut_path, self.lut_sidecar)
        self.encoder = encoder
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(self._command(encoder, lut_name), stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=self._stderr, cwd=cwd)

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", "replace").strip().splitlines()
        return lines[-1] if lines else ""

    def _abort_process(self) -> str:
        proc = self._process
        tail = ""
        if proc is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.wait(timeout=30)
            tail = self._stderr_tail()
        if self._stderr is not None:
            self._stderr.close()
        self._process = self._stderr = None
        return tail

    # -- public API ---------------------------------------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(f"expected uint8 {self.height}x{self.width}x3 BGR frame, got {frame.dtype} {frame.shape}")
        data = np.ascontiguousarray(frame).tobytes()
        if self.frames_written < _REPLAY_FRAMES:
            self._replay.append(data)
        try:
            self._process.stdin.write(data)
        except (BrokenPipeError, OSError):
            self._fall_back_after_failure()
            return
        self.frames_written += 1

    def _fall_back_after_failure(self) -> None:
        """NVENC died mid-start (busy / session limit). Restart on the CPU encoder and replay what was buffered."""
        failed = self.encoder
        tail = self._abort_process()
        if failed.endswith("_nvenc") and self.frames_written < _REPLAY_FRAMES:
            self.fallback_reason = f"{failed} failed while starting: {tail or 'pipe closed'}"
            self._start(_CPU[self.codec])
            try:
                for chunk in self._replay:
                    self._process.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                raise DomainError("WRITER_FAILED", f"CPU fallback encoder also failed: {self._abort_process()}", 500) from None
            self.frames_written = len(self._replay)
            return
        raise DomainError("WRITER_FAILED", f"{failed} stopped accepting frames: {tail or 'pipe closed'}", 500)

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if self.frames_written == 0:
            self._abort_process()
            self.path.unlink(missing_ok=True)
            raise DomainError("WRITER_EMPTY", "No frames were written", 422)
        proc = self._process
        proc.stdin.close()
        code = proc.wait()
        tail = self._stderr_tail()
        self._stderr.close()
        self._process = self._stderr = None
        if code != 0 and self.encoder.endswith("_nvenc") and self.frames_written <= _REPLAY_FRAMES:
            # Encoder only failed at flush (short clip): redo the whole thing on the CPU from the replay buffer.
            self.fallback_reason = f"{self.encoder} failed at flush: {tail}"
            self._start(_CPU[self.codec])
            for chunk in self._replay:
                self._process.stdin.write(chunk)
            self._process.stdin.close()
            code = self._process.wait()
            tail = self._stderr_tail()
            self._stderr.close()
            self._process = self._stderr = None
        if code != 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            raise DomainError("WRITER_FAILED", f"{self.encoder} exited {code}: {tail}", 500)
        self._replay.clear()
        return self.path

    def __enter__(self) -> "NVHardwareVideoWriter":
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is not None:
            self._closed = True
            self._abort_process()
            self.path.unlink(missing_ok=True)
        else:
            self.close()


def mux_source_audio(video: Path | str, source: Path | str | None, output: Path | str, *,
                     start_seconds: float | None = None, end_seconds: float | None = None,
                     ffmpeg: str | None = None) -> bool:
    """Copy ``video`` (stream copy, no re-encode) and add ``source``'s first audio track, trimmed to
    ``[start_seconds, end_seconds)`` so it lines up with a timeline In/Out render. Returns whether audio was added.

    An untrimmed audio track is stream-copied when the container allows it; a trimmed one is re-encoded to AAC
    because a copy can only cut on packet boundaries and would drift against the frame-exact video.
    """
    exe = ffmpeg or find_ffmpeg()
    video, output = Path(video), Path(output)
    if source is None or not Path(source).is_file() or not _has_audio(exe, Path(source)):
        shutil.copyfile(video, output)
        return False
    trim = (start_seconds or 0.0) > 0 or end_seconds is not None
    seek = ["-ss", f"{start_seconds:.6f}"] if (start_seconds or 0.0) > 0 else []
    length = ["-t", f"{(end_seconds - (start_seconds or 0.0)):.6f}"] if end_seconds is not None else []
    base = [exe, "-y", "-v", "error", "-i", str(video), *seek, *length, "-i", str(source)]
    maps = ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    for audio in (["-c:a", "aac", "-b:a", "192k"],) if trim else (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):
        run = subprocess.run(base + maps + audio + ["-shortest", "-movflags", "+faststart", str(output)],
                             capture_output=True, text=True)
        if run.returncode == 0:
            return True
    raise DomainError("AUDIO_MUX_FAILED", f"Audio mux failed: {run.stderr.strip()[-300:]}", 500)


def _has_audio(exe: str, path: Path) -> bool:
    import av
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False
