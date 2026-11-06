"""Crash-consistency audit and repair for jobs, checkpoints, outputs and the queue.

Everything the application publishes is written with write-to-temp + ``os.replace``, so a crash can never leave a
torn *file*. What it can leave is a torn *transaction*: an output renamed into place whose manifest was never written,
an encoder temp nobody will ever finish, a job work directory whose job is already terminal, a queue mirror that
disagrees with its database. This module finds those states and repairs the ones with an unambiguous safe fix.

Rules the repair path follows:

* It never touches the work directory or outputs of a job that can still run (queued, preparing, running, pausing,
  paused, interrupted, cancelling) unless a checkpoint is provably inconsistent, and then it only rolls the
  checkpoint back to the last contiguous frame - frames are never deleted.
* A published output that no *succeeded* job owns is **quarantined**, not deleted, because it may be the only copy of
  hours of work; the quarantine is purged by age.
* ``FaceSet`` archives, models, projects and anything outside the known temp patterns are never touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Any, Callable

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.jobs.schema import JobRecord, JobState

log = logging.getLogger("roop3d.reliability")

JOBS_RELATIVE = "roop_3d/.runtime/jobs"
QUARANTINE_RELATIVE = "roop_3d/.runtime/quarantine"
WRITER_TEMP_DIR = "roop_3d/outputs"

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_ENCODER_TEMP_FILE = re.compile(r"^[0-9a-f]{32}\.(video|mux|signed)\.(mp4|mov|mkv)$")
_ENCODER_TEMP_DIR = re.compile(r"^[0-9a-f]{32}\.sequence$")
# `roop3d_<job>[.ext]` (reference path) and `roop3d_<job>_<time_ns>.mp4` (streaming path: one unique name per attempt).
_JOB_OUTPUT = re.compile(r"^roop3d_([0-9a-f]{32})(?:_\d{10,})?(\.[A-Za-z0-9]+)?$")
STREAM_WORKSPACE_RELATIVE = "roop_3d/.runtime/stream"
_PART_FILE = re.compile(r"\.[0-9a-f]{32}\.part$")

CAN_STILL_RUN = frozenset({JobState.queued, JobState.preparing, JobState.running, JobState.pausing,
                           JobState.paused, JobState.interrupted, JobState.cancelling})
ACTIVE_STATES = frozenset({JobState.preparing, JobState.running, JobState.pausing, JobState.cancelling})
QUARANTINE_RETENTION_S = 7 * 24 * 3600


@dataclass
class Finding:
    code: str
    severity: str                       # "info" | "warning" | "error"
    detail: str
    path: str | None = None
    job_id: str | None = None
    action: str | None = None           # repair action that resolves it, if one exists
    repaired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)
    scanned: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    repaired: int = 0

    @property
    def unresolved(self) -> list[Finding]:
        return [item for item in self.findings if not item.repaired and item.severity in ("warning", "error")]

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" and not item.repaired for item in self.findings)

    def by_code(self, code: str) -> list[Finding]:
        return [item for item in self.findings if item.code == code]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "repaired": self.repaired, "unresolved": len(self.unresolved),
                "seconds": round(self.seconds, 3), "scanned": self.scanned,
                "findings": [item.as_dict() for item in self.findings]}


class ReliabilityAuditor:
    def __init__(self, root: Path, store, settings_provider: Callable[[], Any], *,
                 active_job_ids: Callable[[], set[str]] | None = None, runner: Any = None,
                 stall_after_s: float = 180.0, clock: Callable[[], datetime] | None = None):
        self.root, self.store, self.settings_provider = Path(root), store, settings_provider
        self.active_job_ids = active_job_ids or (lambda: set())
        self.runner = runner
        self.stall_after_s = stall_after_s
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------------------------------------ helpers
    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _records(self) -> tuple[dict[str, JobRecord] | None, Finding | None]:
        try:
            return {record.job_id: record for record in self.store.list()}, None
        except DomainError as error:
            return None, Finding(error.code, "error", "The persisted job database is unreadable. "
                                 "Use the explicit job-store reset repair to quarantine it and start empty.",
                                 path=JobStore_relative(self.store), action="reset_job_store")

    def _output_dirs(self) -> list[Path]:
        configured = self.root / self.settings_provider().output_root
        return [configured] + ([self.root / WRITER_TEMP_DIR] if (self.root / WRITER_TEMP_DIR) != configured else [])

    # ------------------------------------------------------------------------------------------------- audit
    def audit(self, *, deep: bool = False, min_age_s: float = 0.0) -> AuditReport:
        """Read-only. ``min_age_s`` ignores temp artefacts younger than this (use >0 while jobs are running)."""
        started = time.monotonic()
        report = AuditReport()
        now = time.time()
        records, problem = self._records()
        if problem is not None:
            report.findings.append(problem)
            records = {}
        report.scanned["jobs"] = len(records)
        # The persistent queue (queue_manager.py) is a second job system that keeps its own checkpoints in the same
        # `.runtime/jobs/<id>` layout. Anything it knows about - in any state - is never ours to judge or delete.
        protected = ({job_id for job_id, record in records.items() if record.state in CAN_STILL_RUN}
                     | self.active_job_ids() | self._queue_job_ids())

        self._audit_part_files(report, now, min_age_s)
        self._audit_encoder_temp(report, now, min_age_s)
        self._audit_outputs(report, records, deep, now, min_age_s)
        if "*" not in protected:
            self._audit_workdirs(report, records, protected, now, min_age_s)
        self._audit_checkpoints(report, records, deep)
        self._audit_active_without_worker(report, records)
        self._audit_queue(report)
        report.seconds = time.monotonic() - started
        return report

    def _old_enough(self, path: Path, now: float, min_age_s: float) -> bool:
        try:
            modified = path.stat().st_mtime
            # Zero disables the age filter, including future mtimes after clock drift.
            return min_age_s <= 0 or now - modified >= min_age_s
        except OSError:
            return False

    def _audit_part_files(self, report: AuditReport, now: float, min_age_s: float) -> None:
        count = 0
        for base in (self.root / "roop_3d/.runtime", self.root / WRITER_TEMP_DIR):
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [name for name in dirnames if name not in {"pytest", "quarantine", ".venv"}]
                for name in filenames:
                    if _PART_FILE.search(name):
                        path = Path(dirpath) / name
                        count += 1
                        if self._old_enough(path, now, min_age_s):
                            report.findings.append(Finding(
                                "STALE_PART_FILE", "warning", "Interrupted atomic write left a temporary file",
                                path=self._rel(path), action="delete"))
        report.scanned["part_files"] = count

    def _audit_encoder_temp(self, report: AuditReport, now: float, min_age_s: float) -> None:
        base = self.root / WRITER_TEMP_DIR
        count = 0
        if base.is_dir():
            for entry in base.iterdir():
                if _ENCODER_TEMP_FILE.match(entry.name) or (entry.is_dir() and _ENCODER_TEMP_DIR.match(entry.name)):
                    count += 1
                    if self._old_enough(entry, now, min_age_s):
                        report.findings.append(Finding(
                            "ORPHAN_ENCODER_TEMP", "warning", "Encoder/mux temporary from a render that did not finish",
                            path=self._rel(entry), action="delete"))
        report.scanned["encoder_temp"] = count

    def _audit_outputs(self, report: AuditReport, records: dict[str, JobRecord], deep: bool,
                       now: float, min_age_s: float) -> None:
        seen = 0
        live = self.active_job_ids() | self._queue_job_ids()
        if "*" in live:
            return
        for directory in self._output_dirs():
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                match = _JOB_OUTPUT.match(entry.name)
                if match is None or entry.name.endswith(".manifest.json") or entry.name.endswith(".lock"):
                    continue
                seen += 1
                job_id = match.group(1)
                manifest = entry.with_name(entry.name + ".manifest.json")
                record = records.get(job_id)
                owned_by_success = (record is not None and record.state == JobState.succeeded and record.output is not None
                                    and (self.root / record.output.output_path) == entry)
                if owned_by_success:
                    if not manifest.exists():
                        report.findings.append(Finding("SUCCEEDED_MANIFEST_MISSING", "error",
                            "A succeeded job's output has no manifest", path=self._rel(entry), job_id=job_id))
                    elif deep and entry.is_file():
                        from roop_3d.audit.isolation import sha256_file
                        if sha256_file(entry)[1] != record.output.output_sha256:
                            report.findings.append(Finding("OUTPUT_HASH_MISMATCH", "error",
                                "Output bytes no longer match the recorded SHA-256", path=self._rel(entry), job_id=job_id))
                    continue
                if job_id in live or not self._old_enough(entry, now, min_age_s):
                    continue  # a running worker may be between publishing and acknowledging success
                if record is not None and record.state == JobState.succeeded:
                    continue  # a different, current output owns this job
                reason = ("has no manifest (publish was interrupted)" if not manifest.exists()
                          else "belongs to a job that never reached succeeded")
                report.findings.append(Finding(
                    "UNOWNED_OUTPUT", "warning", f"Published output {reason}; a re-run would fail with OUTPUT_EXISTS",
                    path=self._rel(entry), job_id=job_id, action="quarantine"))
        for record in records.values():
            if record.state == JobState.succeeded and record.output is not None:
                target = self.root / record.output.output_path
                if not target.exists():
                    report.findings.append(Finding("SUCCEEDED_OUTPUT_MISSING", "error",
                        "A succeeded job's output file is gone", path=record.output.output_path, job_id=record.job_id))
        report.scanned["outputs"] = seen

    def _audit_workdirs(self, report: AuditReport, records: dict[str, JobRecord], protected: set[str],
                        now: float, min_age_s: float) -> None:
        base = self.root / JOBS_RELATIVE
        count = 0
        if base.is_dir():
            for entry in base.iterdir():
                if not entry.is_dir() or not _HEX32.match(entry.name):
                    continue
                count += 1
                if entry.name in protected or not self._old_enough(entry, now, min_age_s):
                    continue
                record = records.get(entry.name)
                if record is not None and record.state == JobState.failed:
                    # Terminal, so the frames can never be used again; checkpoint.json etc. stay for post-mortems.
                    frames = entry / "frames"
                    if frames.is_dir():
                        report.findings.append(Finding("ORPHAN_JOB_FRAMES", "warning",
                            "A failed job's checkpoint frames can never be resumed", path=self._rel(frames),
                            job_id=entry.name, action="delete"))
                    continue
                why = ("no job record references it" if record is None
                       else f"its job is {record.state.value} and can never resume")
                report.findings.append(Finding("ORPHAN_JOB_WORKDIR", "warning",
                    f"Job work directory (checkpoint frames) is garbage: {why}", path=self._rel(entry),
                    job_id=entry.name, action="delete"))
        # The streaming render path keeps its resumable segment cache in a second work area, keyed by job id. It is only
        # useful to a job that can still run; for a finished/failed/cancelled/unknown job it is the largest garbage there is.
        stream_base = self.root / STREAM_WORKSPACE_RELATIVE
        if stream_base.is_dir():
            for entry in stream_base.iterdir():
                if not entry.is_dir() or not _HEX32.match(entry.name):
                    continue
                count += 1
                if entry.name in protected or not self._old_enough(entry, now, min_age_s):
                    continue
                record = records.get(entry.name)
                why = "no job record references it" if record is None else f"its job is {record.state.value}"
                report.findings.append(Finding("ORPHAN_JOB_WORKDIR", "warning",
                    f"Streaming segment cache is garbage: {why}", path=self._rel(entry), job_id=entry.name, action="delete"))
        report.scanned["job_workdirs"] = count

    def _audit_checkpoints(self, report: AuditReport, records: dict[str, JobRecord], deep: bool) -> None:
        from roop_3d.engine.pipeline_runner import PipelineCheckpoint
        checked = 0
        for job_id, record in records.items():
            if record.state not in {JobState.queued, JobState.paused, JobState.interrupted}:
                continue
            directory = self.root / JOBS_RELATIVE / job_id
            checkpoint_path = directory / "checkpoint.json"
            if not checkpoint_path.is_file():
                continue
            checked += 1
            try:
                checkpoint = PipelineCheckpoint.model_validate_json(checkpoint_path.read_text("utf-8"))
            except (OSError, ValueError):
                report.findings.append(Finding("CHECKPOINT_UNREADABLE", "error",
                    "Checkpoint cannot be parsed; the job cannot resume", path=self._rel(checkpoint_path), job_id=job_id))
                continue
            good = self._last_contiguous_frame(directory / "frames", checkpoint.completed_frame, deep)
            if good < checkpoint.completed_frame:
                report.findings.append(Finding("CHECKPOINT_FRAMES_MISSING", "error",
                    f"Checkpoint claims frame {checkpoint.completed_frame} but only frames 0..{good} are intact",
                    path=self._rel(directory), job_id=job_id, action="rollback_checkpoint"))
        report.scanned["checkpoints"] = checked

    @staticmethod
    def _last_contiguous_frame(frames_dir: Path, claimed: int, deep: bool) -> int:
        if claimed < 0:
            return claimed
        sizes: dict[int, int] = {}
        if frames_dir.is_dir():
            with os.scandir(frames_dir) as entries:
                for entry in entries:
                    name = entry.name
                    if name.endswith(".png") and name[:-4].isdigit():
                        try:
                            sizes[int(name[:-4])] = entry.stat().st_size
                        except OSError:
                            pass
        good = -1
        for index in range(claimed + 1):
            if sizes.get(index, 0) <= 0:
                break
            good = index
        if deep and good >= 0:  # a PNG must end with IEND: catches a truncated final write that survived
            path = frames_dir / f"{good:08d}.png"
            try:
                with path.open("rb") as stream:
                    head = stream.read(8)
                    stream.seek(-12, os.SEEK_END)
                    tail = stream.read(12)
                if head != b"\x89PNG\r\n\x1a\n" or tail[4:8] != b"IEND":
                    good -= 1
            except OSError:
                good -= 1
        return good

    def _audit_active_without_worker(self, report: AuditReport, records: dict[str, JobRecord]) -> None:
        live = self.active_job_ids()
        now = self.clock()
        for job_id, record in records.items():
            if record.state in ACTIVE_STATES and job_id not in live:
                report.findings.append(Finding("ACTIVE_JOB_WITHOUT_WORKER", "error",
                    f"Job is {record.state.value} but no worker owns it (it was orphaned by a crash)",
                    job_id=job_id, action="mark_interrupted"))
            elif record.state == JobState.running and job_id in live:
                # A running job reports progress several times a second. Silence means the worker is wedged (a hung
                # FFmpeg pipe, a driver stall) or the whole process was frozen - never a healthy state to leave unreported.
                silent = (now - record.updated_at).total_seconds()
                if silent > self.stall_after_s:
                    report.findings.append(Finding("JOB_STALLED", "warning",
                        f"Job has been 'running' but silent for {int(silent)} s (frame {record.progress.frame}, "
                        f"stage {record.progress.stage}); pause it and resume to restart from the last checkpoint",
                        job_id=job_id))

    def _queue_job_ids(self) -> set[str]:
        database = self.root / "roop_3d/.runtime/queue/queue.db"
        if not database.is_file():
            return set()
        try:
            connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=5)
            try:
                return {row[0] for row in connection.execute("SELECT job_id FROM queue_jobs")}
            finally:
                connection.close()
        except sqlite3.Error:
            return {"*"}   # cannot tell what the queue owns: fail closed (see _audit_workdirs / _audit_outputs)

    def _audit_queue(self, report: AuditReport) -> None:
        database = self.root / "roop_3d/.runtime/queue/queue.db"
        mirror = self.root / "roop_3d/.runtime/queue/queue.json"
        if not database.is_file():
            return
        try:
            connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error as error:
            report.findings.append(Finding("QUEUE_DB_UNREADABLE", "error", f"Queue database cannot be opened ({type(error).__name__})",
                                           path=self._rel(database)))
            return
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                report.findings.append(Finding("QUEUE_DB_CORRUPT", "error", "SQLite integrity_check failed",
                                               path=self._rel(database)))
                return
            count = connection.execute("SELECT COUNT(*) FROM queue_jobs").fetchone()[0]
            orders = [row[0] for row in connection.execute("SELECT queue_order FROM queue_jobs")]
            report.scanned["queue_jobs"] = count
            if len(orders) != len(set(orders)):
                report.findings.append(Finding("QUEUE_ORDER_DUPLICATE", "warning",
                    "Two queue entries share one position", path=self._rel(database)))
            if mirror.is_file():
                try:
                    mirrored = json.loads(mirror.read_text("utf-8")).get("job_count")
                except (OSError, ValueError):
                    mirrored = None
                if mirrored != count:
                    report.findings.append(Finding("QUEUE_MIRROR_DRIFT", "warning",
                        f"queue.json lists {mirrored} jobs but the database holds {count}", path=self._rel(mirror),
                        action="resync_queue_mirror"))
        except sqlite3.Error as error:
            report.findings.append(Finding("QUEUE_DB_CORRUPT", "error", f"Queue database query failed ({type(error).__name__})",
                                           path=self._rel(database)))
        finally:
            connection.close()

    # ------------------------------------------------------------------------------------------------ repair
    def repair(self, report: AuditReport | None = None, *, deep: bool = False, min_age_s: float = 0.0,
               queue: Any = None, reset_job_store: bool = False) -> AuditReport:
        """Apply every safe fix. ``reset_job_store`` is the only destructive action and is never implicit."""
        report = report or self.audit(deep=deep, min_age_s=min_age_s)
        for finding in report.findings:
            if finding.action is None or finding.repaired:
                continue
            try:
                done = self._apply(finding, queue, reset_job_store)
            except (OSError, DomainError, ValueError) as error:
                log.warning("Repair %s failed (%s)", finding.action, type(error).__name__)
                done = False
            if done:
                finding.repaired = True
                report.repaired += 1
        self.purge_quarantine()
        return report

    def _apply(self, finding: Finding, queue: Any, reset_job_store: bool) -> bool:
        action = finding.action
        if action == "delete" and finding.path:
            target = owned(self.root, finding.path)
            if target.is_symlink():
                return False  # never follow or remove a link someone planted in an owned directory
            if target.is_dir():
                self._rmtree_checked(target)
            else:
                target.unlink(missing_ok=True)
            return True
        if action == "quarantine" and finding.path:
            return self._quarantine(finding.path)
        if action == "rollback_checkpoint" and finding.job_id:
            return self._rollback_checkpoint(finding.job_id)
        if action == "mark_interrupted" and finding.job_id:
            record = self.store.get(finding.job_id)
            if record.state in ACTIVE_STATES:
                self.store.transition(finding.job_id, JobState.interrupted, expected_revision=record.revision)
            return True
        if action == "resync_queue_mirror" and queue is not None:
            queue._sync_json()
            return True
        if action == "reset_job_store" and reset_job_store:
            return self._quarantine(finding.path, rename_only=True) if finding.path else False
        return False

    def _rmtree_checked(self, directory: Path) -> None:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise DomainError("RECOVERY_LINK_FOUND", "Refusing to delete a directory that contains a link", 409)
        shutil.rmtree(directory)

    def _quarantine(self, relative: str, *, rename_only: bool = False) -> bool:
        source = owned(self.root, relative)
        if source.is_symlink() or not source.exists():
            return False
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        destination = owned(self.root, f"{QUARANTINE_RELATIVE}/{stamp}-{source.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        manifest = source.with_name(source.name + ".manifest.json")
        if not rename_only and manifest.is_file() and not manifest.is_symlink():
            os.replace(manifest, owned(self.root, f"{QUARANTINE_RELATIVE}/{stamp}-{manifest.name}"))
        return True

    def _rollback_checkpoint(self, job_id: str) -> bool:
        if self.runner is None:
            return False
        from roop_3d.engine.pipeline_runner import PipelineCheckpoint
        directory = self.root / JOBS_RELATIVE / job_id
        checkpoint = PipelineCheckpoint.model_validate_json((directory / "checkpoint.json").read_text("utf-8"))
        good = self._last_contiguous_frame(directory / "frames", checkpoint.completed_frame, True)
        self.runner._write_checkpoint(checkpoint.model_copy(update={"completed_frame": good}))
        return True

    def purge_quarantine(self, older_than_s: float = QUARANTINE_RETENTION_S) -> int:
        base = self.root / QUARANTINE_RELATIVE
        removed = 0
        if base.is_dir():
            cutoff = time.time() - older_than_s
            for entry in base.iterdir():
                try:
                    if entry.stat().st_mtime < cutoff and not entry.is_symlink():
                        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                        removed += 1
                except OSError:
                    continue
        return removed


def JobStore_relative(store: Any) -> str:
    return getattr(store, "relative", "roop_3d/.runtime/jobs/state.json")


def reconcile_on_startup(root: Path, store, settings_provider, *, runner: Any = None, queue: Any = None) -> AuditReport:
    """Run once, after ``recover_interrupted`` and before the worker starts: nothing is active, so every
    leftover temp/orphan is stale by definition (``min_age_s=0``)."""
    auditor = ReliabilityAuditor(root, store, settings_provider, runner=runner)
    report = auditor.repair(deep=False, min_age_s=0.0, queue=queue)
    if report.findings:
        log.warning("Startup reconciliation: %d finding(s), %d repaired, %d unresolved",
                    len(report.findings), report.repaired, len(report.unresolved))
    return report
