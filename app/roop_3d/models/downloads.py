"""Explicit, cancellable downloads for immutable registered model artifacts."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import uuid
from typing import Any

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog


@dataclass
class DownloadJob:
    download_id: str
    model_id: str
    total_bytes: int
    state: str = "queued"
    bytes_downloaded: int = 0
    error_code: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    future: Future | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        percent = (100.0 * self.bytes_downloaded / self.total_bytes) if self.total_bytes else 0.0
        return {"download_id": self.download_id, "model_id": self.model_id, "state": self.state,
                "bytes_downloaded": self.bytes_downloaded, "total_bytes": self.total_bytes,
                "percent": min(100.0, percent), "error_code": self.error_code}


class ModelDownloadManager:
    def __init__(self, catalog: ModelCatalog, *, max_concurrent: int = 2):
        self.catalog = catalog
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="roop3d-model-download")
        self._lock = threading.RLock()
        self._jobs: dict[str, DownloadJob] = {}
        self._closed = False
        self._max_concurrent = max_concurrent

    def start(self, model_id: str, *, accepted_license: bool = False, commercial: bool = False) -> dict[str, Any]:
        record = self.catalog.registry.get(model_id)
        from roop_3d.models.registry import authorize
        authorize(record, action="download", accepted_license=accepted_license, commercial=commercial)
        with self._lock:
            if self._closed:
                raise DomainError("DOWNLOADS_CLOSED", "Model downloads are shutting down", 503)
            for active_job in self._jobs.values():
                if active_job.model_id == model_id and active_job.state in {"queued", "downloading", "cancelling"}:
                    return active_job.public()
            active = sum(job.state in {"queued", "downloading", "cancelling"} for job in self._jobs.values())
            if active >= self._max_concurrent:
                raise DomainError("DOWNLOAD_LIMIT", "The local model download limit has been reached", 429)
            job = DownloadJob(uuid.uuid4().hex, model_id, record.byte_size or 0)
            self._jobs[job.download_id] = job
            job.future = self._executor.submit(self._run, job, accepted_license, commercial)
            return job.public()

    def _progress(self, job: DownloadJob, downloaded: int, total: int) -> None:
        with self._lock:
            job.bytes_downloaded = min(downloaded, total)

    def _run(self, job: DownloadJob, accepted_license: bool, commercial: bool) -> None:
        with self._lock:
            if job.cancel_event.is_set():
                job.state = "cancelled"
                return
            job.state = "downloading"
        try:
            record = self.catalog.registry.get(job.model_id)
            self.catalog.downloader.download(
                record, accepted_license=accepted_license, commercial=commercial,
                cancel_event=job.cancel_event,
                progress_callback=lambda downloaded, total: self._progress(job, downloaded, total))
        except DomainError as error:
            with self._lock:
                job.state = "cancelled" if error.code == "DOWNLOAD_CANCELLED" or job.cancel_event.is_set() else "failed"
                job.error_code = None if job.state == "cancelled" else error.code
        except Exception:
            with self._lock:
                job.state = "cancelled" if job.cancel_event.is_set() else "failed"
                job.error_code = None if job.state == "cancelled" else "DOWNLOAD_FAILED"
        else:
            with self._lock:
                job.bytes_downloaded = job.total_bytes
                job.state = "completed"

    def get(self, download_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(download_id)
            if job is None:
                raise DomainError("DOWNLOAD_NOT_FOUND", "Model download was not found", 404)
            return job.public()

    def active(self) -> list[dict[str, Any]]:
        """Expose running jobs so a remounted UI can resume progress and cancellation."""
        with self._lock:
            return [job.public() for job in self._jobs.values()
                    if job.state in {"queued", "downloading", "cancelling"}]

    def cancel(self, download_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(download_id)
            if job is None:
                raise DomainError("DOWNLOAD_NOT_FOUND", "Model download was not found", 404)
            if job.state in {"queued", "downloading"}:
                job.cancel_event.set()
                if job.state == "queued" and job.future is not None and job.future.cancel():
                    job.state = "cancelled"
                elif job.state == "downloading":
                    job.state = "cancelling"
            return job.public()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.state in {"queued", "downloading", "cancelling"}:
                    job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
