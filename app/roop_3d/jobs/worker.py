"""Persistent threaded job worker with pause, cancellation and OOM retry."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
import time
import re
from typing import Callable

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.engine.pipeline_runner import (MasterPipelineRunner, PipelineCancelled,
    PipelineInterrupted, PipelinePaused, RecoverablePipelineOOM)
from roop_3d.engine.scheduler import RuntimeScheduler, ScheduledJob
from roop_3d.engine.vram_governor import RuntimeProfile, VRAMGovernor
from roop_3d.jobs.schema import JobFailure, JobProgress, JobState
from roop_3d.jobs.store import JobStore
from roop_3d.models.catalog import ModelCatalog


RENDER_JOB = "RENDER_JOB"


class JobWorker:
    """Poll durable queued jobs and run them under live scheduler admission."""

    def __init__(self, root: Path, store: JobStore, settings_provider: Callable[[], Settings],
                 catalog: ModelCatalog, media, *, pipeline_runner=None, scheduler: RuntimeScheduler | None = None,
                 poll_interval: float = 0.15, max_threads: int = 8, engine_gate=None):
        if poll_interval <= 0 or max_threads < 1:
            raise ValueError("Worker polling and thread limits must be positive")
        self.root, self.store, self.settings_provider = root, store, settings_provider
        self.catalog, self.media = catalog, media
        self.runner = pipeline_runner or MasterPipelineRunner(root, catalog, media, settings_provider)
        self.scheduler = scheduler or RuntimeScheduler(VRAMGovernor())
        self.poll_interval, self.max_threads = poll_interval, max_threads
        # Shared with live preview: model teardown/re-binding waits for in-flight preview inference.
        self.gate = engine_gate
        self._stop = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[Future, tuple[str, ScheduledJob | None]] = {}
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._wake_event.clear()
            self._executor = ThreadPoolExecutor(max_workers=self.max_threads, thread_name_prefix="roop3d-job")
            self._thread = threading.Thread(target=self._serve, name="roop3d-worker", daemon=True)
            self._thread.start()

    def wake(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()

    def _serve(self) -> None:
        try:
            while not self._stop.is_set():
                self._reap()
                settings = self.settings_provider()
                with self._lock:
                    active_ids = {job_id for job_id, _ in self._futures.values()}
                for record in self.store.list():
                    if self._stop.is_set():
                        break
                    if record.state == JobState.cancelled and record.job_id not in active_ids:
                        cleanup = getattr(self.runner, "cleanup", None)
                        if callable(cleanup):
                            cleanup(record.job_id)
                        continue
                    if record.state != JobState.queued or record.job_id in active_ids:
                        continue
                    if len(active_ids) >= settings.max_concurrent_jobs:
                        break
                    executor = self._executor
                    if executor is None:
                        break
                    future = executor.submit(self._execute, record.job_id)
                    with self._lock:
                        self._futures[future] = (record.job_id, None)
                        active_ids.add(record.job_id)
                self._wake_event.wait(self.poll_interval)
                self._wake_event.clear()
        finally:
            executor = self._executor
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
            self._reap()

    def _reap(self) -> None:
        with self._lock:
            completed = [future for future in self._futures if future.done()]
            for future in completed:
                job_id, _ = self._futures.pop(future)
                self.scheduler.release(job_id)
                try:
                    future.result()
                except BaseException:
                    # _execute converts failures to durable job state. The
                    # future is intentionally not logged with raw exception text.
                    pass

    def _control(self, job_id: str):
        if self._stop.is_set():
            return "shutdown"
        try:
            return self.store.get(job_id).state
        except DomainError:
            return JobState.cancelled

    def _mark_running(self, job_id: str) -> None:
        state = self._control(job_id)
        if state == JobState.pausing:
            raise PipelinePaused()
        if state == JobState.cancelling:
            raise PipelineCancelled()
        if state == "shutdown":
            raise PipelineInterrupted()
        current = self.store.get(job_id)
        if current.state != JobState.preparing:
            raise DomainError("JOB_STATE_CHANGED", "Job left preparation before inference started", 409)
        self.store.transition(job_id, JobState.running, expected_revision=current.revision)

    def _report_progress(self, job_id: str, progress: JobProgress) -> None:
        current = self.store.get(job_id)
        if current.state == JobState.running:
            self.store.progress(job_id, progress)

    def _acknowledge_control(self, job_id: str, action: str) -> None:
        current = self.store.get(job_id)
        if action == "pause" and current.state == JobState.pausing:
            self.store.transition(job_id, JobState.paused, expected_revision=current.revision)
        elif action == "cancel" and current.state == JobState.cancelling:
            self.store.transition(job_id, JobState.cancelled, expected_revision=current.revision)

    def _interrupt(self, job_id: str) -> None:
        current = self.store.get(job_id)
        if current.state in {JobState.preparing, JobState.running, JobState.pausing, JobState.cancelling}:
            self.store.transition(job_id, JobState.interrupted, expected_revision=current.revision)

    def _fail(self, job_id: str, code: str, message: str) -> None:
        current = self.store.get(job_id)
        if current.state in {JobState.queued, JobState.preparing, JobState.running,
                             JobState.pausing, JobState.cancelling, JobState.interrupted}:
            self.store.fail(job_id, JobFailure(code=code, message=message))

    def _execute(self, job_id: str) -> None:
        if self.gate is None:
            self._execute_job(job_id)
            return
        # Start-button path: wait (off the event loop) for in-flight preview inference to drain; while
        # waiting the gate refuses new previews, so slider scrubbing cannot starve the job.
        with self.gate.owner(RENDER_JOB):
            self._execute_job(job_id)

    def _execute_job(self, job_id: str) -> None:
        admission: ScheduledJob | None = None
        try:
            record = self.store.get(job_id)
            if record.state != JobState.queued:
                return
            self.store.transition(job_id, JobState.preparing, expected_revision=record.revision)
            settings = self.settings_provider()
            prepared = self.runner.prepare(record, settings)
            while admission is None:
                state = self._control(job_id)
                if state == JobState.pausing:
                    self._acknowledge_control(job_id, "pause")
                    return
                if state == JobState.cancelling:
                    self._acknowledge_control(job_id, "cancel")
                    cleanup = getattr(self.runner, "cleanup", None)
                    if callable(cleanup):
                        cleanup(job_id)
                    return
                if state == "shutdown":
                    self._interrupt(job_id)
                    return
                force_profile = self.runner.retry_profile(job_id)
                admission = self.scheduler.try_admit(job_id, settings, force_profile=force_profile)
                if admission is None:
                    self._wake_event.wait(self.poll_interval)
                    self._wake_event.clear()
            output = self.runner.run_prepared(record, admission.lease.admission, prepared=prepared,
                control=lambda: self._control(job_id),
                mark_running=lambda: self._mark_running(job_id),
                report_progress=lambda value: self._report_progress(job_id, value))
            current = self.store.get(job_id)
            if current.state != JobState.running:
                discard = getattr(self.runner, "discard_output", None)
                if callable(discard):
                    discard(output)
                if current.state == JobState.pausing:
                    self._acknowledge_control(job_id, "pause")
                    return
                if current.state == JobState.cancelling:
                    self._acknowledge_control(job_id, "cancel")
                    cleanup = getattr(self.runner, "cleanup", None)
                    if callable(cleanup):
                        cleanup(job_id)
                    return
                if current.state == JobState.interrupted:
                    return
                raise DomainError("JOB_STATE_CHANGED", "Job left rendering before output publication", 409)
            self.store.succeed(job_id, output)
            cleanup = getattr(self.runner, "cleanup", None)
            if callable(cleanup):
                cleanup(job_id)
        except PipelinePaused:
            self._acknowledge_control(job_id, "pause")
        except PipelineCancelled:
            self._acknowledge_control(job_id, "cancel")
            cleanup = getattr(self.runner, "cleanup", None)
            if callable(cleanup):
                cleanup(job_id)
        except PipelineInterrupted:
            self._interrupt(job_id)
        except RecoverablePipelineOOM:
            current = self.store.get(job_id)
            if current.state == JobState.cancelling:
                self._acknowledge_control(job_id, "cancel")
            elif current.state == JobState.pausing:
                self._acknowledge_control(job_id, "pause")
            else:
                VRAMGovernor.evict_disposable()
                if current.state in {JobState.preparing, JobState.running}:
                    current = self.store.transition(job_id, JobState.interrupted,
                                                    expected_revision=current.revision)
                if current.state == JobState.interrupted:
                    self.store.transition(job_id, JobState.queued, expected_revision=current.revision)
                    self.wake()
        except BaseException as error:
            if self._stop.is_set():
                self._interrupt(job_id)
            elif isinstance(error, DomainError):
                self._fail(job_id, error.code if re.fullmatch(r"[A-Z0-9_]+", error.code) else "PIPELINE_FAILED",
                           "Job processing failed; the local input or configured capability is unavailable.")
            else:
                self._fail(job_id, "WORKER_FAILED", "Job processing failed due to a local runtime error.")
        finally:
            if admission is not None:
                self.scheduler.release(job_id)

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._reap()
            if not self._futures and not any(job.state == JobState.queued for job in self.store.list()):
                return True
            self.wake()
            time.sleep(min(self.poll_interval, 0.05))
        return False
