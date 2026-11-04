"""Backend-owned 10-state persistent job queue with ACID-safe SQLite/JSON storage.

Supported Canonical Job States (exactly 10):
  1. QUEUED
  2. PREFLIGHT
  3. PRESCANNING
  4. PREPASSING
  5. RENDERING
  6. MUXING
  7. PAUSED
  8. COMPLETED
  9. FAILED
  10. CANCELLED

Controls:
  - Per-job progress tracking
  - Per-job cancel
  - Pause/resume from .stage2.npz / checkpoint.npz
  - Drag-and-drop queue reordering
  - Duplicate job
  - Retry failed slice
  - Automatic clip joining with lossless stream copy and original audio remux
  - Zero loss of QUEUED and PAUSED state across server restarts/crashes
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable
import uuid

from pydantic import ConfigDict, Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.pipeline_runner import PipelineCheckpoint, checkpoint_npz_bytes, read_checkpoint_npz
from roop_3d.segment_dispatcher import SegmentSlice, concat_lossless


class CanonicalJobState(str, Enum):
    QUEUED = "QUEUED"
    PREFLIGHT = "PREFLIGHT"
    PRESCANNING = "PRESCANNING"
    PREPASSING = "PREPASSING"
    RENDERING = "RENDERING"
    MUXING = "MUXING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


CANONICAL_TRANSITIONS: dict[CanonicalJobState, frozenset[CanonicalJobState]] = {
    CanonicalJobState.QUEUED: frozenset({
        CanonicalJobState.PREFLIGHT,
        CanonicalJobState.PRESCANNING,
        CanonicalJobState.PREPASSING,
        CanonicalJobState.RENDERING,
        CanonicalJobState.MUXING,
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.PREFLIGHT: frozenset({
        CanonicalJobState.PRESCANNING,
        CanonicalJobState.PREPASSING,
        CanonicalJobState.RENDERING,
        CanonicalJobState.MUXING,
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.PRESCANNING: frozenset({
        CanonicalJobState.PREPASSING,
        CanonicalJobState.RENDERING,
        CanonicalJobState.MUXING,
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.PREPASSING: frozenset({
        CanonicalJobState.RENDERING,
        CanonicalJobState.MUXING,
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.RENDERING: frozenset({
        CanonicalJobState.MUXING,
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.MUXING: frozenset({
        CanonicalJobState.COMPLETED,
        CanonicalJobState.PAUSED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.PAUSED: frozenset({
        CanonicalJobState.QUEUED,
        CanonicalJobState.CANCELLED,
        CanonicalJobState.FAILED,
    }),
    CanonicalJobState.FAILED: frozenset({
        CanonicalJobState.QUEUED,  # Retry
        CanonicalJobState.CANCELLED,
    }),
    CanonicalJobState.COMPLETED: frozenset(),
    CanonicalJobState.CANCELLED: frozenset(),
}


def normalize_canonical_state(state: Any) -> CanonicalJobState:
    """Normalize input state to one of the 10 canonical states."""
    if isinstance(state, CanonicalJobState):
        return state
    s = str(state).strip().upper()
    if hasattr(state, "value"):
        s = str(state.value).strip().upper()

    aliases = {
        "QUEUED": CanonicalJobState.QUEUED,
        "PREFLIGHT": CanonicalJobState.PREFLIGHT,
        "PREPARING": CanonicalJobState.PREFLIGHT,
        "PRESCANNING": CanonicalJobState.PRESCANNING,
        "INSPECTING": CanonicalJobState.PRESCANNING,
        "PREPASSING": CanonicalJobState.PREPASSING,
        "PREPASS": CanonicalJobState.PREPASSING,
        "RENDERING": CanonicalJobState.RENDERING,
        "RUNNING": CanonicalJobState.RENDERING,
        "MUXING": CanonicalJobState.MUXING,
        "PAUSED": CanonicalJobState.PAUSED,
        "PAUSING": CanonicalJobState.PAUSED,
        "INTERRUPTED": CanonicalJobState.PAUSED,
        "COMPLETED": CanonicalJobState.COMPLETED,
        "SUCCEEDED": CanonicalJobState.COMPLETED,
        "SUCCESS": CanonicalJobState.COMPLETED,
        "FAILED": CanonicalJobState.FAILED,
        "ERROR": CanonicalJobState.FAILED,
        "CANCELLED": CanonicalJobState.CANCELLED,
        "CANCELED": CanonicalJobState.CANCELLED,
        "CANCELLING": CanonicalJobState.CANCELLED,
    }
    if s in aliases:
        return aliases[s]
    raise DomainError("INVALID_JOB_STATE", f"Unrecognized job state '{state}'", 422)


class QueueSlice(Contract):
    model_config = ConfigDict(frozen=False)
    slice_id: str
    slice_index: int
    chunk_path: str
    rendered_chunk_path: str | None = None
    start_seconds: float = 0.0
    end_seconds: float | None = None
    duration: float = 0.0
    status: str = "PENDING"  # PENDING, RENDERING, COMPLETED, FAILED
    error: str | None = None
    retry_count: int = 0
    frame_count: int = 0


class QueueJobProgress(Contract):
    frame: int = 0
    total_frames: int | None = None
    fps: float = 0.0
    stage: str = "QUEUED"
    stage_latency_ms: float = 0.0
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)


class QueueJob(Contract):
    job_id: str
    name: str = ""
    state: CanonicalJobState = CanonicalJobState.QUEUED
    queue_order: int = 0
    request: dict[str, Any] = Field(default_factory=dict)
    progress: QueueJobProgress = Field(default_factory=QueueJobProgress)
    slices: list[QueueSlice] = Field(default_factory=list)
    checkpoint_path: str | None = None
    output: dict[str, Any] | None = None
    error_message: str | None = None
    revision: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class _ClosingConnection(sqlite3.Connection):
    """``with connection:`` on a stock sqlite3 connection commits but never closes it, so every queue call relied on
    refcounting to release the database/WAL/SHM handles. Closing deterministically keeps that true on any interpreter
    and while an exception (and its traceback) is still being held by a logger or a caller."""

    def __exit__(self, *exc_info):
        try:
            return super().__exit__(*exc_info)
        finally:
            self.close()


class QueueManager:
    """ACID-safe SQLite + JSON persistent queue manager."""

    relative_db = "roop_3d/.runtime/queue/queue.db"
    relative_json = "roop_3d/.runtime/queue/queue.json"

    def __init__(self, root: Path, notify: Callable[[dict[str, Any]], None] | None = None):
        self.root = root
        self.notify = notify or (lambda event: None)
        self._lock = threading.RLock()
        self._db_path = owned(self.root, self.relative_db)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.recover_on_startup()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0, isolation_level=None, factory=_ClosingConnection)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS queue_jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                queue_order INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                progress_json TEXT NOT NULL,
                slices_json TEXT NOT NULL,
                checkpoint_path TEXT,
                output_json TEXT,
                error_message TEXT,
                revision INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_q_order ON queue_jobs(queue_order);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_q_state ON queue_jobs(state);")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS queue_idempotency (
                key TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

    def _sync_json(self) -> None:
        """Atomically persist snapshot of all queue jobs to queue.json for portable inspection."""
        jobs = self.list_jobs()
        data = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "job_count": len(jobs),
            "jobs": [j.model_dump(mode="json") for j in jobs],
        }
        atomic_json(self.root, self.relative_json, data)

    def _row_to_job(self, row: sqlite3.Row) -> QueueJob:
        return QueueJob(
            job_id=row["job_id"],
            name=row["name"],
            state=CanonicalJobState(row["state"]),
            queue_order=row["queue_order"],
            request=json.loads(row["request_json"]),
            progress=QueueJobProgress.model_validate_json(row["progress_json"]),
            slices=[QueueSlice.model_validate(s) for s in json.loads(row["slices_json"])],
            checkpoint_path=row["checkpoint_path"],
            output=json.loads(row["output_json"]) if row["output_json"] else None,
            error_message=row["error_message"],
            revision=row["revision"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def submit(
        self,
        request: dict[str, Any] | Contract,
        idempotency_key: str | None = None,
        name: str | None = None,
        slices: list[dict[str, Any] | SegmentSlice | QueueSlice] | None = None,
    ) -> tuple[QueueJob, bool]:
        """Submit a new job to the persistent queue."""
        raw_req = request.model_dump(mode="json") if hasattr(request, "model_dump") else dict(request)
        fingerprint = hashlib.sha256(json.dumps(raw_req, sort_keys=True).encode()).hexdigest()

        with self._lock, self._get_connection() as conn:
            # Check idempotency
            if idempotency_key:
                hashed_key = hashlib.sha256(idempotency_key.encode()).hexdigest()
                cur = conn.execute("SELECT job_id, fingerprint FROM queue_idempotency WHERE key = ?", (hashed_key,))
                existing = cur.fetchone()
                if existing:
                    if existing["fingerprint"] != fingerprint:
                        raise DomainError("IDEMPOTENCY_CONFLICT", "Idempotency key was used with different parameters", 409)
                    return self.get(existing["job_id"]), False

            # Compute next queue_order
            cur = conn.execute("SELECT COALESCE(MAX(queue_order), -1) + 1 AS next_order FROM queue_jobs")
            next_order = cur.fetchone()["next_order"]

            job_id = uuid.uuid4().hex
            job_name = name or f"Job {job_id[:8]}"
            now_iso = datetime.now(timezone.utc).isoformat()

            # Process slices
            formatted_slices: list[dict[str, Any]] = []
            if slices:
                for idx, s in enumerate(slices, start=1):
                    s_data = s.model_dump(mode="json") if hasattr(s, "model_dump") else dict(s)
                    if "slice_id" not in s_data:
                        s_data["slice_id"] = uuid.uuid4().hex[:12]
                    if "slice_index" not in s_data:
                        s_data["slice_index"] = idx
                    formatted_slices.append(s_data)

            initial_progress = QueueJobProgress(stage="QUEUED").model_dump_json()

            conn.execute(
                """
                INSERT INTO queue_jobs (
                    job_id, name, state, queue_order, request_json, progress_json,
                    slices_json, checkpoint_path, output_json, error_message, revision,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_name,
                    CanonicalJobState.QUEUED.value,
                    next_order,
                    json.dumps(raw_req),
                    initial_progress,
                    json.dumps(formatted_slices),
                    None,
                    None,
                    None,
                    0,
                    now_iso,
                    now_iso,
                ),
            )

            if idempotency_key:
                conn.execute(
                    "INSERT INTO queue_idempotency (key, job_id, fingerprint, created_at) VALUES (?, ?, ?, ?)",
                    (hashed_key, job_id, fingerprint, now_iso),
                )

        self._sync_json()
        job = self.get(job_id)
        self.notify({"type": "queue.event", "action": "job_added", "job_id": job_id, "state": job.state.value, "data": {"queue_order": job.queue_order}})
        return job, True

    def get(self, job_id: str) -> QueueJob:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                raise DomainError("JOB_NOT_FOUND", f"Queue job {job_id} not found", 404)
            return self._row_to_job(row)

    def list_jobs(self) -> list[QueueJob]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM queue_jobs ORDER BY queue_order ASC, created_at ASC")
            return [self._row_to_job(row) for row in cur.fetchall()]

    def transition(
        self,
        job_id: str,
        new_state: CanonicalJobState | str,
        error_message: str | None = None,
        output: dict[str, Any] | None = None,
    ) -> QueueJob:
        """Enforce canonical transitions between the 10 states."""
        target_state = normalize_canonical_state(new_state)

        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                raise DomainError("JOB_NOT_FOUND", f"Queue job {job_id} not found", 404)

            current_state = CanonicalJobState(row["state"])
            if target_state != current_state:
                allowed = CANONICAL_TRANSITIONS[current_state]
                if target_state not in allowed:
                    raise DomainError(
                        "ILLEGAL_JOB_TRANSITION",
                        f"Cannot transition queue job {job_id} from {current_state.value} to {target_state.value}",
                        422,
                    )

            now_iso = datetime.now(timezone.utc).isoformat()
            out_json = json.dumps(output) if output else row["output_json"]
            err_msg = error_message if error_message is not None else row["error_message"]

            conn.execute(
                """
                UPDATE queue_jobs
                SET state = ?, output_json = ?, error_message = ?, revision = revision + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (target_state.value, out_json, err_msg, now_iso, job_id),
            )

        self._sync_json()
        updated = self.get(job_id)
        self.notify({
            "type": "queue.event",
            "action": "state_change",
            "job_id": job_id,
            "state": target_state.value,
            "data": {
                "previous_state": current_state.value,
                "revision": updated.revision,
            },
        })
        return updated

    def update_progress(self, job_id: str, progress: QueueJobProgress | dict[str, Any]) -> QueueJob:
        prog_obj = progress if isinstance(progress, QueueJobProgress) else QueueJobProgress.model_validate(progress)

        with self._lock, self._get_connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE queue_jobs
                SET progress_json = ?, revision = revision + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (prog_obj.model_dump_json(), now_iso, job_id),
            )

        self._sync_json()
        job = self.get(job_id)
        self.notify({
            "type": "queue.event",
            "action": "progress",
            "job_id": job_id,
            "state": job.state.value,
            "data": prog_obj.model_dump(mode="json"),
        })
        return job

    def cancel(self, job_id: str) -> QueueJob:
        """Cancel a job in the queue."""
        return self.transition(job_id, CanonicalJobState.CANCELLED)

    def pause(self, job_id: str, checkpoint: PipelineCheckpoint | None = None) -> QueueJob:
        """Pause a job and record its .stage2.npz checkpoint."""
        job = self.get(job_id)
        checkpoint_rel = None

        if checkpoint:
            checkpoint_rel = f"roop_3d/.runtime/jobs/{job_id}/checkpoint.npz"
            cp_path = owned(self.root, checkpoint_rel)
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            cp_path.write_bytes(checkpoint_npz_bytes(checkpoint))
        else:
            default_cp = owned(self.root, f"roop_3d/.runtime/jobs/{job_id}/checkpoint.npz")
            if default_cp.is_file():
                checkpoint_rel = f"roop_3d/.runtime/jobs/{job_id}/checkpoint.npz"

        with self._lock, self._get_connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE queue_jobs SET checkpoint_path = ?, updated_at = ? WHERE job_id = ?",
                (checkpoint_rel, now_iso, job_id),
            )

        return self.transition(job_id, CanonicalJobState.PAUSED)

    def resume(self, job_id: str) -> QueueJob:
        """Resume a PAUSED job from its .stage2.npz / checkpoint.npz."""
        job = self.get(job_id)
        if job.state != CanonicalJobState.PAUSED:
            raise DomainError("ILLEGAL_JOB_TRANSITION", f"Only PAUSED jobs can be resumed (current: {job.state.value})", 422)

        # Verify checkpoint validity if present
        if job.checkpoint_path:
            cp_path = owned(self.root, job.checkpoint_path)
            if cp_path.is_file():
                try:
                    read_checkpoint_npz(cp_path.read_bytes())
                except Exception as err:
                    raise DomainError("CHECKPOINT_INVALID", f"Checkpoint at {job.checkpoint_path} is corrupt: {err}", 422)

        return self.transition(job_id, CanonicalJobState.QUEUED)

    def reorder(self, ordered_job_ids: list[str]) -> list[QueueJob]:
        """Drag-and-drop reordering: atomically update queue_order for all supplied job IDs."""
        with self._lock, self._get_connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            for order, j_id in enumerate(ordered_job_ids):
                conn.execute(
                    "UPDATE queue_jobs SET queue_order = ?, updated_at = ? WHERE job_id = ?",
                    (order, now_iso, j_id),
                )

        self._sync_json()
        updated_list = self.list_jobs()
        self.notify({"type": "queue.event", "action": "reordered", "data": {"order": ordered_job_ids}})
        return updated_list

    def duplicate(self, job_id: str) -> QueueJob:
        """Duplicate an existing queue job."""
        job = self.get(job_id)
        new_name = f"{job.name} (Copy)"
        new_job, _ = self.submit(
            request=job.request,
            name=new_name,
            slices=job.slices,
        )
        return new_job

    def retry_failed_slice(self, job_id: str, slice_index_or_id: int | str) -> QueueJob:
        """Reset a failed slice to PENDING and requeue the job."""
        job = self.get(job_id)
        target_idx = None
        for idx, sl in enumerate(job.slices):
            if sl.slice_index == slice_index_or_id or sl.slice_id == str(slice_index_or_id):
                target_idx = idx
                break

        if target_idx is None:
            raise DomainError("SLICE_NOT_FOUND", f"Slice {slice_index_or_id} not found in job {job_id}", 404)

        updated_slices = list(job.slices)
        sl = updated_slices[target_idx]
        updated_slices[target_idx] = QueueSlice(
            slice_id=sl.slice_id,
            slice_index=sl.slice_index,
            chunk_path=sl.chunk_path,
            rendered_chunk_path=None,
            start_seconds=sl.start_seconds,
            end_seconds=sl.end_seconds,
            duration=sl.duration,
            status="PENDING",
            error=None,
            retry_count=sl.retry_count + 1,
            frame_count=sl.frame_count,
        )

        with self._lock, self._get_connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            raw_slices = [s.model_dump(mode="json") for s in updated_slices]
            conn.execute(
                """
                UPDATE queue_jobs
                SET slices_json = ?, error_message = NULL, revision = revision + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(raw_slices), now_iso, job_id),
            )

        if job.state == CanonicalJobState.FAILED:
            self.transition(job_id, CanonicalJobState.QUEUED)

        self._sync_json()
        updated_job = self.get(job_id)
        self.notify({
            "type": "queue.event",
            "action": "slice_retried",
            "job_id": job_id,
            "data": {"slice_index": sl.slice_index},
        })
        return updated_job

    def automatic_clip_joining(self, job_id: str, output_path: str | None = None) -> QueueJob:
        """Lossless stream-copy concat demuxer pass remuxing original source audio."""
        job = self.get(job_id)
        if not job.slices:
            raise DomainError("NO_SLICES", f"Job {job_id} has no video slices to join", 422)

        # Check all slices are completed
        rendered_chunks: list[str] = []
        for sl in job.slices:
            chunk = sl.rendered_chunk_path or sl.chunk_path
            if not chunk or not Path(chunk).is_file():
                raise DomainError("SLICE_INCOMPLETE", f"Slice {sl.slice_index} chunk file is missing", 422)
            rendered_chunks.append(chunk)

        self.transition(job_id, CanonicalJobState.MUXING)

        source_video = job.request.get("target_path")
        resolved_source = owned(self.root, source_video) if source_video else None

        out_rel = output_path or f"roop_3d/.runtime/outputs/roop3d_{job_id}_joined.mp4"
        out_abs = owned(self.root, out_rel)

        concat_result = concat_lossless(
            slice_paths=rendered_chunks,
            source_audio_video=resolved_source,
            output_path=out_abs,
            preserve_intermediates=True,
        )

        output_payload = {
            "output_path": out_rel,
            "frames": concat_result.get("total_frames", 0),
            "segments_joined": concat_result.get("segments_concatenated", len(rendered_chunks)),
            "preserved_chunks": concat_result.get("preserved_chunks", []),
            "audio_sync_remuxed": concat_result.get("audio_sync_remuxed", False),
        }

        return self.transition(job_id, CanonicalJobState.COMPLETED, output=output_payload)

    def recover_on_startup(self) -> dict[str, int]:
        """ACID recovery gate: preserve QUEUED and PAUSED intact across crashes/restarts.

        Transient active states (PREFLIGHT, PRESCANNING, PREPASSING, RENDERING, MUXING)
        are restored to PAUSED (or QUEUED) so no jobs are ever lost.
        """
        counts = {"queued": 0, "paused": 0, "active_recovered": 0, "total": 0}
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT job_id, state FROM queue_jobs")
            rows = cur.fetchall()
            counts["total"] = len(rows)

            now_iso = datetime.now(timezone.utc).isoformat()
            for r in rows:
                st = r["state"]
                if st == CanonicalJobState.QUEUED.value:
                    counts["queued"] += 1
                elif st == CanonicalJobState.PAUSED.value:
                    counts["paused"] += 1
                elif st in (
                    CanonicalJobState.PREFLIGHT.value,
                    CanonicalJobState.PRESCANNING.value,
                    CanonicalJobState.PREPASSING.value,
                    CanonicalJobState.RENDERING.value,
                    CanonicalJobState.MUXING.value,
                ):
                    # Active job was interrupted when server terminated; restore to PAUSED intact
                    conn.execute(
                        "UPDATE queue_jobs SET state = ?, updated_at = ? WHERE job_id = ?",
                        (CanonicalJobState.PAUSED.value, now_iso, r["job_id"]),
                    )
                    counts["active_recovered"] += 1
                    counts["paused"] += 1

        self._sync_json()
        return counts
