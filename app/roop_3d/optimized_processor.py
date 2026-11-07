"""Three-stage streaming coordinator and production job-runner dispatch."""
from __future__ import annotations

from dataclasses import replace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.stage_timing import StageTimings
from roop_3d.core.storage import atomic_json, owned
from roop_3d.hardware_streamer import (FrameMetadata, NvdecFrameSource, SharedMemoryFrameRing,
                                      NVHardwareVideoWriter, SegmentWorkspace)
from roop_3d.optimized_prepass import SparsePrepass, MAIN
from roop_3d.render_policy import RenderPolicy


class StreamingProcessor:
    def __init__(self, root: Path, project_id: str, source: Path, *, detect, track, pipeline,
                 policy: RenderPolicy = RenderPolicy(), identity=None, emit=None,
                 check=lambda: None, progress=lambda *_: None, timings: StageTimings | None = None,
                 execution_provider: str = "CPUExecutionProvider", settings_3d: dict | None = None):
        self.timings = timings if timings is not None else StageTimings()
        self.root, self.project_id, self.source = root, project_id, source
        self.detect, self.track, self.pipeline, self.policy = detect, track, pipeline, policy
        self.identity, self.emit, self.check, self.progress = identity or {}, emit, check, progress
        self.execution_provider, self.settings_3d = execution_provider, settings_3d or {}

    def run(self, destination: Path, *, audio=True, lut_path=None, lut_bake=False, prepass_reader=None):
        from roop_3d.models.owned_files import owned_path
        destination = owned_path(destination)
        if destination.exists():
            raise ValueError("Streaming output already exists")
        published = False
        prepass = SparsePrepass(self.root, self.project_id, self.source, detect=self.detect, track=self.track,
            policy=self.policy, emit=self.emit, identity=self.identity, check=self.check, reader_factory=prepass_reader)
        locked = prepass.run()
        self.check()
        source = NvdecFrameSource(self.source, strict=self.policy.strict_nvdec,
                                  hardware=self.policy.hardware_profile != "cpu")
        workspace = SegmentWorkspace(owned(self.root, f"roop_3d/.runtime/stream/{self.project_id}"))
        by_frame = locked.by_frame()
        # Stage 9: resume from the last verified finished segment of an earlier attempt of this job.
        from roop_3d.stream_checkpoint import StreamSegmentCheckpoint
        fingerprint = hashlib.sha256(json.dumps(
            {"identity": self.identity, "segment_frames": self.policy.segment_frames, "lut": [str(lut_path), lut_bake],
             "audio": audio, "source": [self.source.name, self.source.stat().st_size, source.expected_frames]},
            sort_keys=True, default=str).encode()).hexdigest()
        checkpoint = StreamSegmentCheckpoint(
            self.root, self.project_id, fingerprint,
            inputs={"target": self.source.name, "segment_frames": self.policy.segment_frames,
                    "lut_path": str(lut_path) if lut_path else None, "lut_bake": lut_bake, "audio": audio},
            settings_3d=self.settings_3d, execution_provider=self.execution_provider, fps=float(source.rate),
            width=source.shape[1], height=source.shape[0], total_frames=locked.total_frames)
        reused, resume_from = checkpoint.resume()
        if resume_from >= locked.total_frames or resume_from % self.policy.segment_frames:
            reused, resume_from = [], 0  # nothing left to resume, or not on this run's segment grid: render afresh
            checkpoint.reset()
        workspace.segments.extend(reused)
        if self.emit:
            self.emit({"stage": MAIN, "state": "RUNNING", "active_ranges_count": len(locked.ranges),
                       "verified_target_frames": locked.verified_target_frames, "skipped_frames_pct": locked.skipped_frames_pct})
        writer = segment_path = None
        segment_start = 0
        decoded = resume_from
        failures = []
        started = time.perf_counter()
        ring = SharedMemoryFrameRing(source.shape, self.policy.ring_slots)
        def produce():
            iterator = source.frames()
            try:
                for frame, metadata in iterator:
                    if ring.cancelled.is_set():
                        break
                    if metadata.index < resume_from:
                        continue  # already inside a finished segment
                    ring.put(frame, metadata)
            except BaseException as error:
                failures.append(error)
                ring.cancel()
            finally:
                iterator.close()
                ring.finish()
        producer = threading.Thread(target=produce, name="roop3d-stream-decode", daemon=True)
        producer.start()
        executor = ThreadPoolExecutor(max_workers=self.policy.execution_threads, thread_name_prefix="roop3d-render")
        local = threading.local()
        pending = deque()
        submitted = resume_from
        bundle_frames, bundle_faces, bundle_metadata = [], [], []
        batch_limit = min(self.policy.batch_size, self.policy.ring_slots)
        max_pending = max(1, min(self.policy.execution_threads,
            (512 * 1024**2) // (int(np.prod(source.shape)) * batch_limit)))
        def infer(frames, faces_by_frame):
            if not hasattr(local, "pipeline"):
                factory = getattr(self.pipeline, "fork_worker", None)
                if not callable(factory) and self.policy.execution_threads > 1:
                    raise ValueError("Concurrent rendering requires independent worker pipeline/buffer ownership")
                local.pipeline = factory() if callable(factory) else self.pipeline
            batch = getattr(local.pipeline, "process_batch", None)
            # Swap, optional decoupled restore and host bridges are awaited together here, so this stage
            # is reported as the whole batch time per frame rather than split into guessed shares.
            with self.timings.measure("inswapper", count=max(1, len(frames))):
                if callable(batch) and all(len(v) == 1 for v in faces_by_frame):
                    return batch(frames, faces_by_frame)
                return [local.pipeline.process(frame, faces) if faces else frame for frame, faces in zip(frames, faces_by_frame)]
        def submit_bundle():
            nonlocal bundle_frames, bundle_faces, bundle_metadata
            if bundle_frames:
                pending.append((executor.submit(infer, bundle_frames, bundle_faces), bundle_metadata))
                bundle_frames, bundle_faces, bundle_metadata = [], [], []
        def write_next():
            nonlocal writer, decoded, segment_start, segment_path
            future, metadata_batch = pending.popleft()
            outputs = future.result()
            if len(outputs) != len(metadata_batch):
                raise RuntimeError("Batch output count differs from submitted source frames")
            for output, metadata in zip(outputs, metadata_batch):
                if metadata.index != decoded:
                    raise RuntimeError("Renderer reordered source frames")
                if writer is None:
                    segment_start, segment_path = decoded, workspace.next_path()
                    writer = NVHardwareVideoWriter(segment_path, source.shape, source.rate,
                        hardware=self.policy.hardware_profile != "cpu", strict=self.policy.strict_nvenc,
                        lut_path=lut_path, lut_bake=lut_bake, color=source.color)
                with self.timings.measure("encode"):
                    writer.write(output)
                decoded += 1
                self.progress(decoded, locked.total_frames, (decoded - resume_from) / max(1e-6, time.perf_counter() - started))
                if decoded % self.policy.segment_frames == 0:
                    writer.close()
                    checkpoint.record(segment_path, segment_start, decoded - segment_start, writer.codec)
                    writer = None
        try:
            while True:
                self.check()
                lease_began = time.perf_counter()
                with ring.lease() as item:
                    if item is None:
                        break
                    self.timings.add("decode", (time.perf_counter() - lease_began) * 1000)
                    frame, metadata = item
                    if metadata.index != submitted:
                        raise RuntimeError("Frame transport reordered or dropped source frames")
                    faces = by_frame.get(metadata.index, ())
                    # A bounded owned host copy lets the ring slot be released before
                    # asynchronous work completes. No frame serialization takes place.
                    if len(faces) != 1:
                        submit_bundle()
                        pending.append((executor.submit(infer, [frame.copy()], [faces]), [metadata]))
                    else:
                        bundle_frames.append(frame.copy())
                        bundle_faces.append(faces)
                        bundle_metadata.append(metadata)
                        if len(bundle_frames) == batch_limit:
                            submit_bundle()
                    submitted += 1
                del item, frame
                if len(pending) >= max_pending:
                    self.check()
                    write_next()
            submit_bundle()
            while pending:
                self.check()
                write_next()
            if failures:
                raise failures[0]
            if decoded != locked.total_frames:
                raise RuntimeError("Rendered frame count differs from the verified prepass source")
            if writer:
                writer.close()
                checkpoint.record(segment_path, segment_start, decoded - segment_start, writer.codec)
                writer = None
            self.check()
            final_video = owned(self.root, f"roop_3d/.runtime/stream/{self.project_id}/video_{workspace.run_id}.mp4")
            workspace.concat(final_video)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if audio:
                NVHardwareVideoWriter.remux_audio(final_video, self.source, destination)
                published = True
            else:
                from roop_3d.models.owned_files import owned_path
                owned_path(destination)
                if destination.exists():
                    raise ValueError("Final output exists")
                final_video.replace(destination)
                published = True
            self.check()
            from roop_3d.media.video_reader import VideoReader
            verified = (sum(1 for _ in NvdecFrameSource(destination, strict=True).frames())
                        if self.policy.strict_nvdec else sum(1 for _ in VideoReader(destination, hardware="software").frames()))
            if verified != decoded:
                raise RuntimeError("Final stream-copy output failed frame-count verification")
            if self.emit:
                self.emit({"stage": MAIN, "state": "DONE"})
            checkpoint.finish({"output": destination.name, "frames": decoded, "verified": True})
            return {"frames": decoded, "resumed_from_frame": resume_from, "prepass_restored": locked.restored,
                    "active_ranges_count": len(locked.ranges), "verified_target_frames": locked.verified_target_frames,
                    "skipped_frames_pct": locked.skipped_frames_pct, "decoder": source.backend,
                    "pipeline": self.pipeline.backend, "segments": [p.name for p in workspace.segments],
                    "source_sha256": sha256_file(self.source)[1], "output_sha256": sha256_file(destination)[1]}
        except BaseException:
            if writer:
                writer.abort()
            if self.emit:
                self.emit({"stage": MAIN, "state": "FAILED"})
            if published:
                owned_path(destination).unlink(missing_ok=True)
            raise
        finally:
            ring.cancel()
            source.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            producer.join(timeout=35)
            if producer.is_alive():
                raise RuntimeError("Decoder did not stop; shared-memory handles are preserved until it exits")
            item = frame = None
            ring.close()


def make_stream_runner(root, catalog, media, settings_provider, *, pipeline_tracker=None):
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner

    class StreamDispatchRunner(MasterPipelineRunner):
        def run_prepared(self, record, admission, *, prepared, control, mark_running, report_progress):
            settings = prepared.settings
            requested = settings.render_backend
            # The fused batch kernel currently encodes InSwapper's fixed 128px
            # target/source contract. Alternate native-size architectures share
            # the reference direct pipeline's dynamic crop compositor instead.
            if (record.request.swapper_model_id or settings.swapper_model_id) != "inswapper_128":
                return super().run_prepared(record, admission, prepared=prepared, control=control,
                    mark_running=mark_running, report_progress=report_progress)
            basic = (settings.execution_mode == "swap" and settings.lip_sync_mode == "off"
                and record.request.export is None and record.request.trim_start_seconds is None
                and record.request.trim_end_seconds is None and record.request.face_segments is None
                and record.request.workstation_parameters is None and record.request.parameter_curves is None
                and record.request.mask_base64 is None and settings.audio_policy in ("copy", "drop")
                and settings.face_restoration_policy in ("off", "auto", "gpen", "codeformer", "ultramax"))
            if requested == "reference" or requested == "auto" and (admission.profile == "cpu" or not basic):
                return super().run_prepared(record, admission, prepared=prepared, control=control,
                    mark_running=mark_running, report_progress=report_progress)
            if not basic:
                raise DomainError("STREAM_FEATURE_UNAVAILABLE", "This job requires the preserved reference timeline/finishing path", 422)
            from roop_3d.media.probe import probe
            info = probe(prepared.target).videos[0]
            if (info.vfr is not False or info.rotation % 360 or not (info.average_rate or info.nominal_rate)
                    or info.start_time not in (None, 0)):
                if requested == "auto":
                    return super().run_prepared(record, admission, prepared=prepared, control=control,
                        mark_running=mark_running, report_progress=report_progress)
                raise DomainError("STREAM_TIMING_UNSUPPORTED", "Streaming requires verified CFR and upright frames", 422)
            try:
                # Full packet-grid/SAR/HDR validation happens before model activation.
                NvdecFrameSource(prepared.target, hardware=admission.profile != "cpu", strict=False)
            except ValueError:
                if requested == "auto":
                    return super().run_prepared(record, admission, prepared=prepared, control=control,
                        mark_running=mark_running, report_progress=report_progress)
                raise DomainError("STREAM_TIMING_UNSUPPORTED", "Input requires the reference timing/color pipeline", 422) from None
            from roop_3d.engine.runtime import OrtModel, record_named, session_factory
            from roop_3d.engine.detection.face_tracker import FaceTracker
            from roop_3d.media.detector import Detector
            from roop_3d.models.emap_cache import cached_emap
            from roop_3d.engine.volumetric.identity_swapper import embedding_map, unit_embedding
            from roop_3d.vectorized_pipeline import CpuBatchPipeline, GpuBatchPipeline
            from roop_3d.render_policy import CudaWorkerContext
            from roop_3d.core.hardware import probe_hardware, resolve_accelerator
            hardware = probe_hardware(settings)
            accelerator = resolve_accelerator(settings.preferred_provider, hardware.advertised_providers, bool(hardware.gpus))
            if admission.profile == "cpu":
                accelerator = "CPUExecutionProvider"
            config = replace(self._runtime_config(admission, settings), provider=accelerator)
            context = CudaWorkerContext(enabled=admission.profile != "cpu")
            tracker = detector = model = restorer = None
            def check():
                signal = self._control_signal(control)
                if signal:
                    raise signal()
            def emit(value):
                if pipeline_tracker:
                    pipeline_tracker.render_event(record.job_id, value)
            try:
                with context.activate():
                    detector = Detector(catalog, session_factory(catalog, config))
                    tracker = FaceTracker(catalog, config=config, stabilize=False)
                    static = record_named(catalog, "inswapper_128.onnx")
                    projection = cached_emap(catalog._path(static), embedding_map, source_sha256=static["sha256"])
                    embeddings = {key: unit_embedding(unit_embedding(np.asarray(value.canonical_embedding)) @ projection)
                                  for key, value in prepared.sources_by_track.items()}
                    model = OrtModel(catalog, "inswapper_128.batch.onnx", config)
                    if settings.face_restoration_policy != "off":
                        from roop_3d.compositor.stream_restoration import DecoupledRestorer, OnnxFaceRestorer, gpen_spec, codeformer_spec, load_ultramax_spec
                        if settings.face_restoration_policy == "ultramax":
                            spec = load_ultramax_spec(self.root)
                        else:
                            spec = codeformer_spec() if settings.face_restoration_policy == "codeformer" else gpen_spec()
                        restorer = DecoupledRestorer(lambda: OnnxFaceRestorer(catalog, config, spec), spec,
                            workers=settings.restorer_threads, pool_size=settings.restorer_pool_size,
                            cuda=admission.profile != "cpu", ultramax_native_bypass=settings.ultramax_native_bypass,
                            ultramax_min_pixels=settings.ultramax_min_pixels,
                            cuda_provider=accelerator if accelerator != 'CPUExecutionProvider' else 'CUDAExecutionProvider',
                            hardware_profile=admission.profile if admission.profile != 'cpu' else '6gb')
                    selected_session = model.session
                    if admission.profile == "cpu":
                        swap_output_name = model.record["graph"]["outputs"][0]["name"]
                        pipeline = CpuBatchPipeline(lambda target, source: model.run({"target": target,
                            "source": source})[swap_output_name], embeddings=embeddings, restorer=restorer)
                    else:
                        compositor_options = None
                        try:
                            import torch
                            if torch.cuda.is_available():
                                from roop_3d.gpu_compositor import CompositorOptions
                                compositor_options = CompositorOptions()
                            elif restorer is not None:
                                raise DomainError("GPU_RESTORER_UNAVAILABLE",
                                    "GPU face restoration requires a working PyTorch CUDA device", 503)
                        except (ImportError, OSError):
                            if restorer is not None:
                                raise DomainError("GPU_RESTORER_UNAVAILABLE",
                                    "GPU face restoration requires the installed CUDA compositor extra", 503) from None
                        pipeline = GpuBatchPipeline(selected_session, embeddings, max_batch=settings.stream_batch_size,
                            hardware_profile=admission.profile, restorer=restorer,
                            execution_threads=settings.execution_threads, compositor_options=compositor_options)
                timings = StageTimings()
                def detect(image):
                    with context.activate(), timings.measure("detection"):
                        return detector.detect(image, 0.5)[0]
                def track(image):
                    with context.activate(), timings.measure("detection"):
                        return [(v.track_id, v.observation.box, v.observation.landmarks5) for v in tracker.process(image)
                                if v.track_id in embeddings]
                mark_running()
                policy = RenderPolicy.from_env(temporal_min_cutoff=settings.temporal_min_cutoff,
                    temporal_beta=settings.temporal_beta, temporal_d_cutoff=settings.temporal_d_cutoff,
                    batch_size=settings.stream_batch_size,
                    execution_threads=settings.execution_threads, ring_slots=settings.stream_ring_slots,
                    hardware_profile=admission.profile, sparse_interval=settings.sparse_interval,
                    range_padding=settings.sparse_interval, segment_frames=settings.stream_segment_frames)
                destination = owned(self.root, f"{settings.output_root}/roop3d_{record.job_id}_{time.time_ns()}.mp4")
                processor = StreamingProcessor(self.root, record.job_id, prepared.target, detect=detect, track=track,
                    pipeline=pipeline, policy=policy, emit=emit, check=check, timings=timings, execution_provider=accelerator,
                    identity={"request": self._request_digest(record), "settings": settings.model_dump(),
                              "model_sha256": model.record["sha256"]},
                    progress=lambda frame, total, fps: report_progress(__import__("roop_3d.jobs.schema", fromlist=["JobProgress"]).JobProgress(
                        frame=frame, total_frames=total, fps=fps, stage="render", stage_timings_ms=timings.take())))
                metadata = processor.run(destination, audio=settings.audio_policy == "copy")
                relative = destination.relative_to(self.root).as_posix()
                manifest_path = relative + ".manifest.json"
                execution_backend = getattr(pipeline, "backend", "unknown")
                gpu_resident_compositor = execution_backend == "CUDA_RESIDENT_TRIPLE_BUFFER"
                metadata.update(output_path=relative, manifest_path=manifest_path, audio_policy=settings.audio_policy,
                    model_hashes={"identity_swapper": model.record["sha256"]}, verified_frames=metadata["frames"],
                    sparse_absence_evidence="SPARSE_ESTIMATE", render_backend="stream",
                    render_execution_backend=execution_backend,
                    gpu_resident_compositor=gpu_resident_compositor,
                    restorer=settings.face_restoration_policy,
                    restorer_host_bridge=restorer is not None and not gpu_resident_compositor)
                if restorer:
                    metadata["restorer_providers"] = sorted({v.model.provider for v in restorer.instances if v.model})
                    metadata["restorer_model_hashes"] = sorted({v.model.record["sha256"] for v in restorer.instances if v.model})
                atomic_json(self.root, manifest_path, metadata)
                from roop_3d.jobs.schema import JobOutput
                return JobOutput(output_path=relative, manifest_path=manifest_path,
                    output_sha256=metadata["output_sha256"], frames=metadata["frames"], audio_policy=settings.audio_policy)
            finally:
                if restorer:
                    restorer.close()
                with context.activate():
                    if tracker:
                        tracker.release()
                    if detector:
                        detector.release_for_build()
                    if model:
                        model.release()
    return StreamDispatchRunner(root, catalog, media, settings_provider)
