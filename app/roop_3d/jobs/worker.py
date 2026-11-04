"""Persistent threaded job worker with pause, cancellation and OOM retry."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
import threading
import time
import re
import shutil
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
from roop_3d.engine.registry import EngineRegistry
from roop_3d.reliability.classify import classify, classify_domain
from roop_3d.reliability.monitor import MONITOR


RENDER_JOB = "RENDER_JOB"
log = logging.getLogger("roop3d.worker")


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
        self.engines = EngineRegistry()
        self.engines.bind_reference(self.runner)
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
        # Cancelled jobs whose work directory has been released. Without this the poll loop re-ran cleanup for every
        # cancelled job ever recorded, on every 150 ms tick, for the life of the process.
        self._cleaned: set[str] = set()

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
                        if record.job_id in self._cleaned:
                            continue
                        try:
                            cancelled_runner = self.engines.resolve(record.request.engine_id)
                        except DomainError:
                            continue
                        cleanup = getattr(cancelled_runner, "cleanup", None)
                        try:
                            if callable(cleanup):
                                cleanup(record.job_id)
                            self._cleaned.add(record.job_id)
                        except (OSError, DomainError):
                            pass  # retried on the next tick: a file may be briefly locked
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
        if progress.stage_timings_ms:
            MONITOR.observe_stages(job_id, progress.stage_timings_ms)
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

    def _release_failed_frames(self, job_id: str) -> None:
        """A failed job can never resume, so its checkpoint frames (tens of GB for a long video) are garbage."""
        try:
            from roop_3d.core.storage import owned
            frames = owned(self.root, f"roop_3d/.runtime/jobs/{job_id}/frames")
            if frames.is_dir() and not frames.is_symlink():
                shutil.rmtree(frames, ignore_errors=True)
        except Exception:
            pass  # the startup reconciliation removes whatever is left

    def _stop_resumable(self, job_id: str, code: str, message: str) -> None:
        """Environmental stop (disk, RAM, locked file): keep the checkpoint, make the job resumable, say why."""
        current = self.store.get(job_id)
        if current.state == JobState.cancelling:  # the user already asked to stop: honour that, not the error
            self._acknowledge_control(job_id, "cancel")
        elif current.state == JobState.pausing:
            self._acknowledge_control(job_id, "pause")
        elif current.state in {JobState.preparing, JobState.running}:
            self.store.interrupt(job_id, JobFailure(code=code, message=message, recoverable=True))
        elif current.state == JobState.queued:
            self.store.fail(job_id, JobFailure(code=code, message=message))

    def _handle_failure(self, job_id: str, error: BaseException) -> None:
        if isinstance(error, DomainError):
            resumable = classify_domain(error)
            if resumable is None:
                code = error.code if re.fullmatch(r"[A-Z0-9_]+", error.code) else "PIPELINE_FAILED"
                kind, message, warning = ("fail", "Job processing failed; the local input or configured capability is unavailable.",
                                          "domain error " + code)
            else:
                code, message, kind, warning = resumable.code, resumable.message, "interrupt", resumable.warning
        else:
            result = classify(error)
            code, message, warning = result.code, result.message, result.warning
            kind = "interrupt" if result.recoverable else "fail"
        # Exception *text* can carry paths and media names, so only the class-level summary is logged.
        log.warning("Job %s %s: %s", job_id[:8], "interrupted" if kind == "interrupt" else "failed", warning)
        try:
            if kind == "interrupt":
                self._stop_resumable(job_id, code, message)
            else:
                self._fail(job_id, code, message)
                self._release_failed_frames(job_id)
        except DomainError:
            pass  # the job moved on (cancelled / paused) while the error propagated

    def _execute_job(self, job_id: str) -> None:
        MONITOR.job_started(job_id)
        try:
            self._run_job(job_id)
        finally:
            MONITOR.job_finished(job_id)

    def _run_job(self, job_id: str) -> None:
        admission: ScheduledJob | None = None
        runner = None
        try:
            record = self.store.get(job_id)
            if record.state != JobState.queued:
                return
            self.store.transition(job_id, JobState.preparing, expected_revision=record.revision)
            runner = self.engines.resolve(record.request.engine_id)
            settings = record.effective_settings or self.settings_provider()
            prepared = runner.prepare(record, settings)
            while admission is None:
                state = self._control(job_id)
                if state == JobState.pausing:
                    self._acknowledge_control(job_id, "pause")
                    return
                if state == JobState.cancelling:
                    self._acknowledge_control(job_id, "cancel")
                    cleanup = getattr(runner, "cleanup", None)
                    if callable(cleanup):
                        cleanup(job_id)
                    return
                if state == "shutdown":
                    self._interrupt(job_id)
                    return
                force_profile = runner.retry_profile(job_id)
                admission = self.scheduler.try_admit(job_id, settings, force_profile=force_profile)
                if admission is None:
                    self._wake_event.wait(self.poll_interval)
                    self._wake_event.clear()
            output = runner.run_prepared(record, admission.lease.admission, prepared=prepared,
                control=lambda: self._control(job_id),
                mark_running=lambda: self._mark_running(job_id),
                report_progress=lambda value: self._report_progress(job_id, value))
            current = self.store.get(job_id)
            if current.state != JobState.running:
                discard = getattr(runner, "discard_output", None)
                if callable(discard):
                    discard(output)
                if current.state == JobState.pausing:
                    self._acknowledge_control(job_id, "pause")
                    return
                if current.state == JobState.cancelling:
                    self._acknowledge_control(job_id, "cancel")
                    cleanup = getattr(runner, "cleanup", None)
                    if callable(cleanup):
                        cleanup(job_id)
                    return
                if current.state == JobState.interrupted:
                    return
                raise DomainError("JOB_STATE_CHANGED", "Job left rendering before output publication", 409)
            self.store.succeed(job_id, output)
            cleanup = getattr(runner, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup(job_id)
                except (OSError, DomainError):
                    # The job already succeeded and its output is published. A locked scratch file (antivirus, an
                    # indexer) must neither be reported as a job failure nor block the worker; the audit removes it.
                    log.warning("Job %s succeeded; its work directory could not be released yet", job_id[:8])
        except PipelinePaused:
            self._acknowledge_control(job_id, "pause")
        except PipelineCancelled:
            self._acknowledge_control(job_id, "cancel")
            cleanup = getattr(runner, "cleanup", None)
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
            else:
                self._handle_failure(job_id, error)
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
