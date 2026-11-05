"""HTTPS-only, bounded, resumable downloads of reviewed immutable artifacts.

Redirects are deliberately rejected, including redirects to another allowed host.
No archives are opened and no model code is imported or executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import re
import ssl
import stat
import threading
from urllib.parse import unquote, urljoin, urlsplit
import uuid
from contextlib import ExitStack
from typing import Callable

import httpx

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.models.registry import ModelRecord, authorize
from roop_3d.models.verifier import verify_file

# Exact hosts only. Model-specific URLs and provenance still require manifest review.
OFFICIAL_HOSTS = frozenset({"github.com", "raw.githubusercontent.com", "huggingface.co"})
HF_STORAGE_HOSTS = frozenset({"us.aws.cdn.hf.co"})
MAX_ARTIFACT_BYTES = 64 * 1024**3
BLOCKED_SUFFIXES = frozenset({".zip", ".tar", ".tgz", ".gz", ".7z", ".rar", ".exe", ".dll",
                              ".so", ".dylib", ".py", ".ps1", ".bat", ".cmd", ".sh",
                              ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".bin"})


def tls_context() -> ssl.SSLContext:
    """Verify servers against the operating-system trust store plus certifi.

    HTTPS-inspecting security software (for example antivirus web scanning) re-signs
    traffic with a root that is installed in the OS store but absent from certifi, so
    certifi alone fails intermittently with CERTIFICATE_VERIFY_FAILED. Verification and
    hostname checking stay on, and the artifact SHA-256 is verified independently.
    """
    context = ssl.create_default_context()  # loads the OS root/CA stores
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return context


def validate_url(value: str) -> str:
    try:
        url = urlsplit(value)
        decoded = unquote(unquote(url.path))
        if (url.scheme != "https" or url.hostname not in OFFICIAL_HOSTS or url.port not in (None, 443)
                or url.username or url.password or url.fragment or url.query
                or any(c in value for c in "\r\n\t\\")
                or any(p in {".", ".."} for p in decoded.split("/"))
                or any(c in decoded for c in "\x00\\\r\n")):
            raise ValueError("Unapproved URL")
    except ValueError:
        raise DomainError("DOWNLOAD_URL_BLOCKED", "Only reviewed direct HTTPS URLs on official hosts are allowed", 403) from None
    return value


def validate_hf_redirect(source: str, location: str) -> str:
    """Permit the observed Hugging Face LFS redirect without trusting arbitrary hosts."""
    try:
        if urlsplit(source).hostname != "huggingface.co":
            raise ValueError("Redirect source is not Hugging Face")
        target = urlsplit(urljoin(source, location))
        decoded = unquote(unquote(target.path))
        if (target.scheme != "https" or target.hostname not in HF_STORAGE_HOSTS or target.port not in (None, 443)
                or target.username or target.password or target.fragment
                or any(c in target.geturl() for c in "\r\n\t\\")
                or any(p in {".", ".."} for p in decoded.split("/"))
                or any(c in decoded for c in "\x00\\\r\n")):
            raise ValueError("Unapproved Hugging Face storage redirect")
    except ValueError:
        raise DomainError("DOWNLOAD_REDIRECT_BLOCKED", "Model host redirected outside the reviewed Hugging Face storage host", 403) from None
    return target.geturl()


def validator(headers: httpx.Headers) -> tuple[str, str] | None:
    etag = headers.get("etag", "")
    if re.fullmatch(r'"[\x21\x23-\x7e]*"', etag):
        return "etag", etag
    modified = headers.get("last-modified", "")
    if modified:
        try:
            date = parsedate_to_datetime(modified)
            if date.tzinfo and not any(c in modified for c in "\r\n"):
                return "last-modified", modified
        except (ValueError, TypeError, OverflowError):
            pass
    return None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    downloaded: bool
    resumed: bool


class Downloader:
    def __init__(self, root: Path, *, transport: httpx.BaseTransport | None = None,
                 max_bytes: int = MAX_ARTIFACT_BYTES):
        self.root, self.transport, self.max_bytes = root, transport, max_bytes

    def _path(self, relative: str) -> Path:
        path = owned(self.root, relative)
        if os.path.lexists(path):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise DomainError("CACHE_FILE_UNSAFE", "Model cache entries must be regular unlinked files")
        return path

    def paths(self, record: ModelRecord) -> tuple[str, str, str]:
        base = f"roop_3d/models/cache/{record.model_id}"
        suffix = Path(record.filenames[0]).suffix if record.filenames else ".artifact"
        return base + ".model" + suffix, base + ".part", base + ".part.json"

    def cached(self, record: ModelRecord) -> Path | None:
        final, _, _ = self.paths(record)
        path = self._path(final)
        if not path.exists():
            return None
        if not record.expected_sha256 or not record.byte_size:
            raise DomainError("MODEL_IDENTITY_UNKNOWN", "No artifact identity is registered")
        try:
            return verify_file(path, record.expected_sha256, record.byte_size)
        except DomainError as error:
            quarantined = self._path(final + "." + uuid.uuid4().hex + ".quarantined")
            os.replace(path, quarantined)
            raise DomainError("MODEL_QUARANTINED", "Managed model cache failed verification and was quarantined", 409) from error

    def _remove(self, relative: str) -> None:
        path = self._path(relative)
        if path.exists():
            path.unlink()

    def _quarantine(self, part: str, metadata: str) -> None:
        source = self._path(part)
        if source.exists():
            destination = self._path(part + "." + uuid.uuid4().hex + ".corrupt")
            os.replace(source, destination)
        self._remove(metadata)

    def _resume(self, record: ModelRecord, part: str, metadata: str) -> tuple[int, tuple[str, str] | None]:
        path, meta = self._path(part), self._path(metadata)
        if path.exists() and meta.exists() and meta.stat().st_size <= 16384:
            try:
                state = json.loads(meta.read_text("utf-8"))
                if not isinstance(state, dict) or not isinstance(state.get("validator"), dict):
                    raise ValueError("Invalid resume metadata")
                identity = validator(httpx.Headers(state["validator"]))
                if (identity and state["url"] == record.upstream_url
                        and state["sha256"] == record.expected_sha256 and state["size"] == record.byte_size
                        and 0 < path.stat().st_size < record.byte_size):
                    return path.stat().st_size, identity
            except (ValueError, KeyError, TypeError):
                pass
        self._remove(part)
        self._remove(metadata)
        return 0, None

    def download(self, record: ModelRecord, *, accepted_license: bool = False,
                 commercial: bool = False, cancel_event: threading.Event | None = None,
                 progress_callback: Callable[[int, int], None] | None = None,
                 state_callback: Callable[[str], None] | None = None) -> DownloadResult:
        authorize(record, action="download", accepted_license=accepted_license, commercial=commercial)
        url = validate_url(record.upstream_url)  # type: ignore[arg-type]
        suffix = Path(record.filenames[0]).suffix.lower() if record.filenames else ""
        if suffix in BLOCKED_SUFFIXES or record.serialization_format.upper() in {"PICKLE", "PYTORCH", "TORCH"}:
            raise DomainError("DOWNLOAD_FORMAT_BLOCKED", "Executable, pickle, and archive model formats are not accepted", 415)
        if record.byte_size > self.max_bytes:  # type: ignore[operator]
            raise DomainError("DOWNLOAD_SIZE_LIMIT", "Artifact exceeds the configured download limit", 413)
        final, part, metadata = self.paths(record)
        lock = f"roop_3d/models/cache/{record.model_id}.lock"
        self._path(lock)
        with exclusive(self.root, lock):
            self._path(lock)
            existing = self.cached(record)
            if existing:
                if state_callback:
                    state_callback("READY")
                return DownloadResult(existing, False, False)
            # A verified complete part can be recovered after a crash before rename.
            partial = self._path(part)
            if partial.exists() and partial.stat().st_size == record.byte_size:
                return self._finish(record, final, part, metadata, False, state_callback)
            offset, previous = self._resume(record, part, metadata)
            if progress_callback is not None:
                try:
                    progress_callback(offset, record.byte_size)
                except Exception:
                    pass
            if state_callback:
                state_callback("DOWNLOADING")
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(30, read=60),
                              follow_redirects=False, trust_env=False, verify=tls_context()) as client:
                for _ in range(2):
                    if cancel_event is not None and cancel_event.is_set():
                        raise DomainError("DOWNLOAD_CANCELLED", "Model download was cancelled; verified partial bytes are retained for resume", 409)
                    headers = {"Accept-Encoding": "identity"}
                    if offset and previous:
                        headers.update({"Range": f"bytes={offset}-", "If-Range": previous[1]})
                    try:
                        with ExitStack() as stack:
                            response = stack.enter_context(client.stream("GET", url, headers=headers))
                            if 300 <= response.status_code < 400:
                                location = response.headers.get("location", "")
                                response = stack.enter_context(client.stream(
                                    "GET", validate_hf_redirect(url, location), headers=headers))
                            if 300 <= response.status_code < 400:
                                raise DomainError("DOWNLOAD_REDIRECT_BLOCKED", "Redirected model downloads require a reviewed direct URL", 403)
                            identity = validator(response.headers)
                            if offset and (response.status_code == 416 or (
                                    response.status_code == 206 and identity != previous)):
                                self._remove(part)
                                self._remove(metadata)
                                offset, previous = 0, None
                                continue
                            if response.status_code not in {200, 206}:
                                raise DomainError("DOWNLOAD_HTTP_ERROR", "Model host did not return artifact bytes", 502)
                            if response.headers.get("content-encoding", "identity").lower() != "identity":
                                raise DomainError("DOWNLOAD_ENCODING_BLOCKED", "Compressed HTTP representations are not accepted", 502)
                            if response.status_code == 206:
                                expected = f"bytes {offset}-{record.byte_size - 1}/{record.byte_size}"
                                if not offset or response.headers.get("content-range") != expected:
                                    raise DomainError("DOWNLOAD_RANGE_INVALID", "Partial response does not match the requested artifact range", 502)
                            else:
                                offset = 0
                                self._remove(part)
                                self._remove(metadata)
                            length = response.headers.get("content-length")
                            if length is not None and (not re.fullmatch(r"[0-9]+", length)
                                                       or int(length) != record.byte_size - offset):
                                raise DomainError("DOWNLOAD_SIZE_MISMATCH", "Response length differs from the registered artifact", 502)
                            state = {"url": url, "sha256": record.expected_sha256, "size": record.byte_size,
                                     "validator": dict([identity]) if identity else {}}
                            atomic_json(self.root, metadata, state)
                            resumed = bool(offset)
                            path = self._path(part)
                            flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                            flags |= os.O_APPEND if offset else os.O_CREAT | os.O_EXCL
                            descriptor = os.open(path, flags, 0o600)
                            with os.fdopen(descriptor, "ab" if offset else "wb") as stream:
                                info = os.fstat(stream.fileno())
                                current = self._path(part).stat()
                                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != offset
                                        or (info.st_ino, info.st_dev) != (current.st_ino, current.st_dev)):
                                    raise DomainError("CACHE_FILE_UNSAFE", "Partial artifact changed before writing")
                                total = offset
                                try:
                                    for block in response.iter_bytes():
                                        if cancel_event is not None and cancel_event.is_set():
                                            raise DomainError("DOWNLOAD_CANCELLED", "Model download was cancelled; verified partial bytes are retained for resume", 409)
                                        if total + len(block) > record.byte_size:
                                            raise DomainError("DOWNLOAD_SIZE_LIMIT", "Response exceeded the registered artifact size", 413)
                                        stream.write(block)
                                        total += len(block)
                                        if progress_callback is not None:
                                            try:
                                                progress_callback(total, record.byte_size)
                                            except Exception:
                                                pass
                                finally:
                                    stream.flush()
                                    os.fsync(stream.fileno())
                            if total != record.byte_size:
                                raise DomainError("DOWNLOAD_INCOMPLETE", "Download is incomplete; retry to resume", 502)
                            return self._finish(record, final, part, metadata, resumed, state_callback)
                    except httpx.HTTPError:
                        raise DomainError("DOWNLOAD_INTERRUPTED", "Download was interrupted; retry to resume", 502) from None
                    except DomainError as error:
                        if error.code == "DOWNLOAD_SIZE_LIMIT":
                            self._quarantine(part, metadata)
                        raise
            raise DomainError("DOWNLOAD_IDENTITY_CHANGED", "Host identity changed while resuming; retry a fresh download", 502)

    def _finish(self, record: ModelRecord, final: str, part: str, metadata: str,
                resumed: bool, state_callback: Callable[[str], None] | None = None) -> DownloadResult:
        if state_callback:
            state_callback("VERIFYING")
        try:
            verify_file(self._path(part), record.expected_sha256, record.byte_size)  # type: ignore[arg-type]
        except DomainError:
            self._quarantine(part, metadata)
            if state_callback:
                state_callback("QUARANTINED")
            raise DomainError("DOWNLOAD_DIGEST_MISMATCH", "Downloaded artifact failed verification and was quarantined", 502) from None
        source, destination = self._path(part), self._path(final)
        # Windows FlushFileBuffers requires a writable handle, including recovery
        # of a complete part left behind before the atomic rename.
        with source.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(source, destination)
        self._remove(metadata)
        if os.name != "nt":
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if state_callback:
            state_callback("READY")
        return DownloadResult(destination, True, resumed)
