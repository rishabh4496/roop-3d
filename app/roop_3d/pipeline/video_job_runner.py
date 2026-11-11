"""End-to-end video job runner: FFmpeg decode pipe -> photorealistic swap per frame -> FFmpeg encode pipe -> audio remux.

``stream_video`` is the model-free, reusable core of that loop (it knows nothing about faces). ``EndToEndVideoJobRunner`` is the
job-engine adapter (``engine_id = "photoreal_3d_ffmpeg"``): it inherits input validation (inspection, FaceSet mapping, identity
lock, model digests) from ``MasterPipelineRunner.prepare`` and replaces only the render loop, so jobs keep their durable
lifecycle, idempotency, HUD (``report_progress`` -> ``PinokioTerminalHUD``) and ``/ws/v1/pipeline/status`` progress events.

Honest limits (also written into every manifest):

* The runner is NOT resumable: a paused / interrupted job restarts from frame 0 (no PNG frame store is written).
* Output is constant-frame-rate at the source's average rate; a variable-frame-rate source is flagged ``vfr_source`` in the manifest.
* HDR (PQ / HLG) sources are refused (``HDR_INPUT_UNSUPPORTED``): a raw BGR pipe cannot tone-map them correctly. Use the
  reference engine for HDR.
* Output carries no source metadata (``-map_metadata -1``) and no subtitle tracks; audio is stream-copied when the MP4
  container accepts the codec, otherwise re-encoded to AAC and reported as ``transcode``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Iterator

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.pipeline_runner import (MasterPipelineRunner, PipelineCancelled, PipelineInterrupted, PipelinePaused,
                                            RecoverablePipelineOOM, PreparedPipeline)
from roop_3d.engine.vram_governor import is_cuda_oom
from roop_3d.jobs.schema import JobOutput, JobProgress, JobRecord
from roop_3d.media.nv_writer import NVHardwareVideoWriter, probe_nvenc
from roop_3d.reliability.processes import REGISTRY, spawn_kwargs, terminate
from roop_3d.segment_dispatcher import find_ffmpeg, find_ffprobe
from roop_3d.telemetry.pinokio_terminal import AUDIO_POLICY_TEXT, HUD

log = logging.getLogger("roop3d.pipeline")

ENGINE_ID = "photoreal_3d_ffmpeg"
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
_MATRIX = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020", "bt2020c": "bt2020"}


# ================================================================================================================ probing
@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int                    # as DISPLAYED (rotation applied)
    height: int
    rate: Fraction                # average frame rate
    frames: int | None            # declared frame count (None when the container does not say)
    duration: float | None
    color_space: str | None
    color_transfer: str | None
    color_range: str | None
    vfr: bool
    audio_codec: str | None
    audio_streams: int
    rotation: int = 0

    @property
    def decode_matrix(self) -> str:
        return _MATRIX.get(self.color_space or "", "bt709" if self.height >= 720 else "bt601")


def _rational(text: str | None) -> Fraction | None:
    try:
        value = Fraction(text or "0/1")
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def probe_video(path: Path | str, *, ffprobe: str | None = None) -> VideoInfo:
    """Container facts from ``ffprobe`` (the same tool the verification uses, so they cannot disagree)."""
    path = Path(path)
    command = [ffprobe or find_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        document = json.loads(run.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise DomainError("MEDIA_PROBE_FAILED", f"ffprobe could not read the target video: {type(error).__name__}", 422) from None
    if run.returncode != 0 or not document.get("streams"):
        raise DomainError("MEDIA_PROBE_FAILED", "ffprobe could not read the target video", 422)
    video = next((s for s in document["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise DomainError("MEDIA_NO_VIDEO", "The target has no video stream", 422)
    audio = [s for s in document["streams"] if s.get("codec_type") == "audio"]
    rotation = 0
    for item in video.get("side_data_list", []):
        if "rotation" in item:
            rotation = int(round(float(item["rotation"]))) % 360
    if "rotate" in video.get("tags", {}):
        rotation = int(float(video["tags"]["rotate"])) % 360
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        width, height = height, width
    average, real = _rational(video.get("avg_frame_rate")), _rational(video.get("r_frame_rate"))
    rate = average or real
    if rate is None:
        raise DomainError("VIDEO_RATE_UNKNOWN", "The target video has no usable frame rate", 422)
    frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() and int(video["nb_frames"]) > 0 else None
    try:
        duration = float(video.get("duration") or document.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return VideoInfo(path, width, height, rate, frames, duration, video.get("color_space"), video.get("color_transfer"),
                     video.get("color_range"), bool(average and real and abs(average - real) > Fraction(1, 1000)),
                     audio[0].get("codec_name") if audio else None, len(audio), rotation)


# ================================================================================================================ decode
class FFmpegDecoder:
    """Frames from one ``ffmpeg`` child through its stdout (``rawvideo`` BGR24), with the source's own colour matrix."""

    def __init__(self, info: VideoInfo, *, ffmpeg: str | None = None):
        if info.color_transfer in HDR_TRANSFERS:
            raise DomainError("HDR_INPUT_UNSUPPORTED", "PQ / HLG sources cannot be tone-mapped through a raw BGR pipe; "
                              "use the reference engine for HDR input", 422)
        self.info, self.ffmpeg = info, ffmpeg or find_ffmpeg()
        self._process: subprocess.Popen | None = None
        self._stderr = None

    def __enter__(self) -> "FFmpegDecoder":
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.info.path), "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-fps_mode", "passthrough", "-vf", f"scale=in_color_matrix={self.info.decode_matrix}",
                   "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        self._stderr = tempfile.TemporaryFile()
        self._process = REGISTRY.track(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self._stderr,
                                                        bufsize=0, **spawn_kwargs()), "decode")
        return self

    def frames(self) -> Iterator[np.ndarray]:
        width, height = self.info.width, self.info.height
        size = width * height * 3
        stream = self._process.stdout
        while True:
            buffer = bytearray(size)
            view, filled = memoryview(buffer), 0
            while filled < size:
                count = stream.readinto(view[filled:])
                if not count:
                    break
                filled += count
            if filled == 0:
                break
            if filled < size:
                raise DomainError("MEDIA_DECODE_FAILED", f"The decoder stopped inside a frame: {self._tail()}", 422)
            yield np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
        code = self._process.wait()
        if code != 0:
            raise DomainError("MEDIA_DECODE_FAILED", f"ffmpeg decode exited {code}: {self._tail()}", 422)

    def _tail(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", "replace").strip().splitlines()
        return lines[-1][:200] if lines else ""

    def __exit__(self, *_exc) -> None:
        process = self._process
        if process is not None:
            try:
                if process.stdout:
                    process.stdout.close()
            except OSError:
                pass
            if process.poll() is None:
                terminate(process)
            REGISTRY.untrack(process)
        if self._stderr is not None:
            self._stderr.close()
        self._process = self._stderr = None


# ================================================================================================================ encode
class ColourManagedWriter(NVHardwareVideoWriter):
    """``NVHardwareVideoWriter`` (NVENC first, automatic libx264 fallback with frame replay) with the parts a final export needs:
    BT.709 limited-range conversion with the matching tags, a rational frame rate, and orphan-safe process tracking."""

    def __init__(self, path: Path | str, width: int, height: int, rate: Fraction, **kwargs):
        self.rate = Fraction(rate)
        super().__init__(path, width, height, float(self.rate), **kwargs)

    def _command(self, encoder: str, lut_name: str | None) -> list[str]:
        command = [self.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
                   "-framerate", f"{self.rate.numerator}/{self.rate.denominator}", "-i", "-", "-an",
                   "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", encoder, "-g", str(self.gop)]
        if encoder.endswith("_nvenc"):
            command += ["-preset", "p5", "-rc", "vbr", "-cq", str(self.quality), "-b:v", "0"]
        else:
            command += ["-preset", "medium", "-crf", str(self.quality)]
        return command + ["-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                          "-color_range", "tv", "-movflags", "+faststart", str(self.path.resolve())]

    def _start(self, encoder: str) -> None:
        super()._start(encoder)
        REGISTRY.track(self._process, "encode")


def _remux(video: Path, source: Path, output: Path, *, seconds: float, policy: str, has_audio: bool, ffmpeg: str) -> str:
    """Video stream-copied from ``video``; the source's first audio stream stream-copied (or AAC when the MP4 refuses the codec)."""
    if policy == "drop" or not has_audio:
        shutil.copyfile(video, output)
        return "drop"
    base = [ffmpeg, "-y", "-v", "error", "-i", str(video), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-map_metadata", "-1", "-t", f"{seconds:.6f}", "-movflags", "+faststart"]
    if policy != "transcode":
        run = subprocess.run(base + ["-c:a", "copy", str(output)], capture_output=True, text=True)
        if run.returncode == 0:
            return "copy"
        log.info("audio stream copy refused by the MP4 container (%s); re-encoding to AAC", run.stderr.strip()[-120:])
    run = subprocess.run(base + ["-c:a", "aac", "-b:a", "256k", str(output)], capture_output=True, text=True)
    if run.returncode != 0:
        raise DomainError("AUDIO_REMUX_FAILED", f"Audio remux failed: {run.stderr.strip()[-200:]}", 500)
    return "transcode"


# ================================================================================================================ the loop
@dataclass
class StreamResult:
    output: Path
    frames: int
    verified_frames: int
    rate: Fraction
    width: int
    height: int
    encoder: str
    encoder_fallback: str | None
    audio_policy: str
    seconds: float
    padded_to_even: bool
    decode_ms: float
    encode_ms: float
    info: VideoInfo = field(repr=False, default=None)


def verify_output(path: Path, *, ffprobe: str | None = None) -> dict:
    """What the finished file really contains (ffprobe, packet-exact): frames, rate, size, audio."""
    command = [ffprobe or find_ffprobe(), "-v", "error", "-print_format", "json", "-count_packets", "-show_streams", str(path)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    document = json.loads(run.stdout or "{}")
    video = next((s for s in document.get("streams", []) if s.get("codec_type") == "video"), None)
    if run.returncode != 0 or video is None:
        raise DomainError("OUTPUT_UNREADABLE", "The finished file could not be read back", 500)
    audio = [s for s in document["streams"] if s.get("codec_type") == "audio"]
    return {"frames": int(video.get("nb_read_packets") or 0), "rate": _rational(video.get("avg_frame_rate")),
            "width": int(video["width"]), "height": int(video["height"]), "codec": video.get("codec_name"),
            "audio_streams": len(audio), "audio_codec": audio[0].get("codec_name") if audio else None,
            "colorspace": video.get("color_space"), "pix_fmt": video.get("pix_fmt")}


def stream_video(target: Path | str, output: Path | str, frame_fn: Callable[[np.ndarray, int], np.ndarray], *,
                 info: VideoInfo | None = None, prefer_nvenc: bool = True, audio_policy: str = "copy", quality: int = 16,
                 control: Callable[[], None] | None = None, progress: Callable[[int, float, float, float], None] | None = None,
                 ffmpeg: str | None = None) -> StreamResult:
    """Decode ``target`` through a pipe, run ``frame_fn(frame_bgr, index) -> frame_bgr`` on every frame, encode through a pipe
    (``h264_nvenc`` with automatic ``libx264`` CRF ``quality`` fallback) and remux the source audio. Partial files are removed on
    any failure. ``control()`` may raise to stop the loop; ``progress(done, fps, decode_ms, encode_ms)`` is called per frame."""
    target, output = Path(target), Path(output)
    ffmpeg = ffmpeg or find_ffmpeg()
    info = info or probe_video(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(f"{output.stem}.video.mp4")
    staged = output.with_name(f"{output.stem}.staged.mp4")
    pad_w, pad_h = info.width % 2, info.height % 2
    writer = ColourManagedWriter(silent, info.width + pad_w, info.height + pad_h, info.rate, prefer_hardware=prefer_nvenc,
                                 quality=quality, ffmpeg=ffmpeg)
    began = time.perf_counter()
    count, decode_s, encode_s = 0, 0.0, 0.0
    try:
        with FFmpegDecoder(info, ffmpeg=ffmpeg) as decoder:
            frames = decoder.frames()
            while True:
                mark = time.perf_counter()
                frame = next(frames, None)
                decode_s += time.perf_counter() - mark
                if frame is None:
                    break
                if control is not None:
                    control()
                processed = frame_fn(frame, count)
                if (not isinstance(processed, np.ndarray) or processed.dtype != np.uint8
                        or processed.shape != (info.height, info.width, 3)):
                    raise DomainError("FRAME_PROCESSOR_INVALID", "The frame processor must return a uint8 BGR frame of the input size", 500)
                if pad_w or pad_h:
                    processed = np.pad(processed, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                mark = time.perf_counter()
                writer.write(processed)
                encode_s += time.perf_counter() - mark
                count += 1
                if progress is not None:
                    progress(count, count / max(time.perf_counter() - began, 1e-6), decode_s * 1000 / count, encode_s * 1000 / count)
        if count == 0:
            raise DomainError("MEDIA_EMPTY", "Target video contains no decodable frames", 422)
        writer.close()
        seconds = count * info.rate.denominator / info.rate.numerator
        actual = _remux(silent, target, staged, seconds=seconds, policy=audio_policy, has_audio=info.audio_streams > 0, ffmpeg=ffmpeg)
        check = verify_output(staged)
        if check["frames"] != count:
            raise DomainError("OUTPUT_FRAME_MISMATCH", f"Encoded {count} frames but the file holds {check['frames']}", 500)
        os.replace(staged, output)
        return StreamResult(output, count, check["frames"], info.rate, info.width, info.height, writer.encoder,
                            writer.fallback_reason if writer.encoder == "libx264" else None, actual, seconds,
                            bool(pad_w or pad_h), decode_s * 1000 / count, encode_s * 1000 / count, info)
    except BaseException:
        writer._closed = True
        try:
            writer._abort_process()
        except Exception:
            pass
        raise
    finally:
        silent.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


# ============================================================================================================ job adapter
class EndToEndVideoJobRunner(MasterPipelineRunner):
    """Job engine ``photoreal_3d_ffmpeg``: real detection + inswapper + the photorealistic swap core, FFmpeg in and out."""

    ENGINE_ID = ENGINE_ID

    def _require_disk_space(self, record: JobRecord, target: Path, total, checkpoint) -> None:
        """No PNG frame store here: the output (a fraction of the source) plus the reserve is all this runner writes."""
        from roop_3d.reliability import disk
        disk.require(self.root, disk.estimate(0, 0, 0, target.stat().st_size))

    # ---- restorer / options -----------------------------------------------------------------------------------------
    def _restorer(self, record: JobRecord, settings, runtime_config):
        """(callable | None, display name, engine | None).

        Restoration is part of this engine's contract (step 4), so ``auto`` means UltraMax. The saved global policy only overrides
        that when it names a concrete architecture: its default ``off`` is the reference engine's opt-in switch, not a request to
        skip this engine's restorer. Switching it off here is explicit: put ``restorer`` in the parameters' ``bypass`` list."""
        parameters = record.request.parameters_at(0)
        architecture = parameters.restorer_model
        if architecture == "auto":
            policy = settings.face_restoration_policy
            architecture = policy if policy in ("gpen", "codeformer", "gfpgan", "ultramax") else "ultramax"
        if "restorer" in parameters.bypass:
            return None, "off", None
        from roop_3d.compositor import restoration as r
        if architecture == "ultramax":
            engine = r.UltraMaxPipelineRestorer(self.catalog, runtime_config)
        elif architecture == "codeformer":
            engine = r.CodeFormerRestorer(self.catalog, runtime_config)
        elif architecture == "gfpgan":
            engine = r.GfpganRestorer(self.catalog, runtime_config)
        elif architecture == "gpen":
            engine = r.GpenRestorer(self.catalog, runtime_config)
        else:
            raise DomainError("RESTORER_UNAVAILABLE", "The selected restoration architecture has no verified local adapter", 503)
        return (lambda crop, landmarks5: engine.restore_hires(crop, landmarks5)), architecture, engine

    def _core_options(self, record: JobRecord):
        from roop_3d.pipeline.photorealistic_swap_core import SwapCoreOptions
        parameters = record.request.parameters_at(0)
        composite = parameters.composite_options()
        return SwapCoreOptions(swap_scope=parameters.swap_scope, skull_blend=parameters.skull_blend,
                               relight_strength=parameters.relight_strength,
                               restoration_blend=parameters.restoration_blend if parameters.restoration_blend > 0 else 0.85,
                               pyramid_levels=max(4, min(6, composite.pyramid_levels)))

    # ---- the render loop --------------------------------------------------------------------------------------------
    def run_prepared(self, record: JobRecord, admission, *, prepared: PreparedPipeline, control, mark_running, report_progress) -> JobOutput:
        from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
        from roop_3d.pipeline.session_adapter import SessionFrameProcessor
        settings, request = prepared.settings, record.request
        checkpoint = self._checkpoint(prepared.checkpoint, completed_frame=-1, profile=admission.profile, provider=admission.provider)
        info = probe_video(prepared.target)
        total = prepared.total_frames or info.frames
        runtime_config = self._runtime_config(admission, self.settings_for(request, settings))
        session = self.pipeline_factory(self.catalog, runtime_config)
        if callable(getattr(session, "select_swapper", None)):
            session.select_swapper(request.swapper_model_id or settings.swapper_model_id)
        numeric_ids = set(prepared.labels_to_ids.values())
        base_source = next(iter(prepared.sources_by_track.values()))
        parameters = request.parameters_at(0)
        relative = f"{settings.output_root}/{record.job_id}/result.mp4"
        manifest_relative = f"{settings.output_root}/{record.job_id}/manifest.json"
        final = owned(self.root, relative)                  # ownership is checked here, before anything is written
        last: dict = {}
        processor: SessionFrameProcessor | None = None
        restorer_engine = None
        try:
            signal = self._control_signal(control)
            if signal is not None:
                raise signal()
            session.prepare_session(base_source, track_ids=numeric_ids, sources_by_track=prepared.sources_by_track,
                                    retarget="off" if parameters.uses_mesh_projection() else "auto",
                                    **parameters.session_options(), **(prepared.tracking or {}))
            if hasattr(session, "set_target_registry") and hasattr(self.media, "target_locks"):
                session.set_target_registry(self.media.target_locks.registry(request.target_path))
            if prepared.identity_guard is not None:
                session.set_identity_guard(prepared.identity_guard)
            if hasattr(session, "configure_temporal"):
                session.configure_temporal(settings)
            restore_fn, restorer_name, restorer_engine = self._restorer(record, settings, runtime_config)
            core = Photorealistic3DSwapCore(self._core_options(record), restorer=restore_fn, restorer_name=restorer_name)
            processor = SessionFrameProcessor(session, core)
            mark_running()
            HUD.milestone("PIPE", "ffmpeg stream started", job=record.job_id[:8], size=f"{info.width}x{info.height}",
                          fps=f"{float(info.rate):.3f}", restorer=restorer_name, scope=core.options.swap_scope)

            def frame_fn(frame: np.ndarray, index: int) -> np.ndarray:
                swapped, report = processor.process(frame, index)
                last["report"] = report
                return swapped

            def on_progress(done: int, fps: float, decode_ms: float, encode_ms: float) -> None:
                report = last.get("report")
                timings = {name: value for name, value in (report.stage_ms if report else {}).items()
                           if name in ("detection", "inswapper", "pose_tps", "masking", "restorer")}
                timings["decode"], timings["encode"] = round(decode_ms, 3), round(encode_ms, 3)
                advisory = None
                if processor.faces_skipped or processor.faces_withheld:
                    advisory = f"{processor.faces_skipped} face(s) not swapped, {processor.faces_withheld} withheld by the identity gate"
                shown_total = None if total is None else max(total, done)
                report_progress(JobProgress(frame=done, total_frames=shown_total, fps=fps, stage="render",
                                            stage_latency_ms=report.stage_ms.get("total", 0.0) if report else 0.0,
                                            stage_timings_ms=timings, advisory=advisory))

            def check() -> None:
                signal = self._control_signal(control)
                if signal is not None:
                    raise signal()

            report_progress(JobProgress(frame=0, total_frames=total, stage="start"))
            result = stream_video(prepared.target, final, frame_fn, info=info, prefer_nvenc=admission.profile != "cpu",
                                  audio_policy=settings.audio_policy, control=check, progress=on_progress)
            if processor.faces_swapped == 0:
                final.unlink(missing_ok=True)
                raise DomainError("TRACK_NOT_DETECTED", "No mapped face could be swapped in any frame", 422)
            report_progress(JobProgress(frame=result.frames, total_frames=max(total or 0, result.frames), stage="mux"))
            capability = session.capability()
            # The restorer THIS engine used (the session's compositor restorer is a different, unused object here).
            restorer_capability = restorer_engine.capability() if restorer_engine is not None and hasattr(restorer_engine, "capability") else {}
        except BaseException as error:
            final.unlink(missing_ok=True)
            if is_cuda_oom(error):
                if checkpoint.oom_retries == 0 and admission.profile != "cpu":
                    self._checkpoint(checkpoint, completed_frame=-1, profile=admission.profile, retry_profile="cpu", oom_retries=1)
                    raise RecoverablePipelineOOM("CUDA allocation failed; one CPU retry is queued") from None
                raise DomainError("PIPELINE_OOM", "Pipeline exhausted the available memory profiles", 503) from None
            raise
        finally:
            if restorer_engine is not None and hasattr(restorer_engine, "release"):
                try:
                    restorer_engine.release()
                except Exception:
                    log.debug("restorer release failed", exc_info=True)
            session.release_session()

        digest = sha256_file(final)[1]
        manifest = {
            "stage": "17", "engine_id": ENGINE_ID, "job_id": record.job_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "output_path": relative, "output_sha256": digest, "input_sha256": prepared.input_sha256,
            "verified_frames": result.verified_frames, "source_frames_declared": info.frames,
            "fps": f"{result.rate.numerator}/{result.rate.denominator}", "width": result.width, "height": result.height,
            "encoder": result.encoder, "encoder_fallback_reason": result.encoder_fallback,
            "color": {"matrix": "bt709", "range": "tv", "decoded_with": info.decode_matrix, "source_space": info.color_space},
            "audio_policy": result.audio_policy, "audio_policy_text": AUDIO_POLICY_TEXT.get(result.audio_policy),
            "source_audio_codec": info.audio_codec, "metadata_stripped": True,
            "restorer": restorer_name, "restorer_capability": restorer_capability,
            "swap_scope": core.options.swap_scope, "skull_blend": core.options.skull_blend,
            "per_frame_mean_ms": processor.mean_stage_ms(), "decode_ms": round(result.decode_ms, 3), "encode_ms": round(result.encode_ms, 3),
            "faces": {"swapped": processor.faces_swapped, "skipped": processor.faces_skipped, "withheld": processor.faces_withheld},
            "fallbacks": dict(sorted(processor.fallbacks.items())),
            "identity_swapper": capability.get("identity_swapper", {}), "face_mappings": [
                {"track_id": label, "faceset_id": source} for label, source in zip(prepared.labels_to_ids, prepared.faceset_ids)],
            "limits": {"resumable": False, "vfr_source": info.vfr, "padded_to_even": result.padded_to_even,
                       "skull": "faceset_view_106_similarity_aligned", "sh_normals": "silhouette_ellipsoid unless mesh normals supplied"},
            "request_sha256": self._request_digest(record),
        }
        atomic_json(self.root, manifest_relative, manifest)
        HUD.milestone("REMUX", "output written and verified", job=record.job_id[:8], frames=result.verified_frames,
                      encoder=result.encoder, audio=AUDIO_POLICY_TEXT.get(result.audio_policy, result.audio_policy), sha256=digest[:16])
        output = JobOutput(output_path=relative, manifest_path=manifest_relative, output_sha256=digest,
                           frames=result.verified_frames, audio_policy=result.audio_policy)
        signal = self._control_signal(control)
        if signal is not None:
            self.discard_output(output)
            raise signal()
        return output
