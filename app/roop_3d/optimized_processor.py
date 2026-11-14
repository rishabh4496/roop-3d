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
                 execution_provider: str = "CPUExecutionProvider", settings_3d: dict | None = None, live_hub=None):
        self.timings = timings if timings is not None else StageTimings()
        self.root, self.project_id, self.source = root, project_id, source
        self.detect, self.track, self.pipeline, self.policy = detect, track, pipeline, policy
        self.identity, self.emit, self.check, self.progress = identity or {}, emit, check, progress
        self.execution_provider, self.settings_3d = execution_provider, settings_3d or {}
        self.live_hub = live_hub     # a LiveFrameHub: finished frames are offered to the binary live stream (never blocks)

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
            # Head render and host bridges are awaited together here, so this stage
            # is reported as the whole batch time per frame rather than split into guessed shares.
            with self.timings.measure("head_render", count=max(1, len(frames))):
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
                if self.live_hub is not None:
                    self.live_hub.publish(output, frame_index=metadata.index,
                                          timestamp_s=metadata.pts * metadata.time_base_num / metadata.time_base_den)
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
            NVHardwareVideoWriter.remux_audio(final_video, self.source, destination, include_audio=audio)
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
    """The default ``reference_onnx`` job runner.

    The fused 2D crop-swap batch kernel (and its restorer / paste compositor) was removed. Until the true-3D full-head
    renderer is wired (``roop_3d.pipeline.true3d_renderer.is_wired()``), every render job fails LOUDLY through
    :func:`roop_3d.core.true3d_guard.require_true3d_renderer`; nothing falls back to a 2D path.
    """
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner

    class StreamDispatchRunner(MasterPipelineRunner):
        def run_prepared(self, record, admission, *, prepared, control, mark_running, report_progress):
            from roop_3d.core.true3d_guard import require_true3d_renderer
            require_true3d_renderer("render job")
            return super().run_prepared(record, admission, prepared=prepared, control=control,
                                        mark_running=mark_running, report_progress=report_progress)

    return StreamDispatchRunner(root, catalog, media, settings_provider)
