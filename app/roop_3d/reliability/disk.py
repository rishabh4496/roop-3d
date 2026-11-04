"""Disk-space budgeting for long renders.

A render stores every completed frame as a PNG checkpoint under ``roop_3d/.runtime/jobs/<id>/frames`` and then muxes the
final file into the output directory. A 60-minute 1080p30 job is 108,000 frames; running out of disk part-way is the
most predictable long-job failure, so it is checked up front (``require``) and, because disks also fill from other
programs mid-job, recognised when it happens (``is_disk_full``).
"""
from __future__ import annotations

import errno
import shutil
from dataclasses import dataclass
from pathlib import Path

from roop_3d.core.errors import DomainError

# Windows: ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL. ENOSPC covers POSIX and Python's own mapping.
_DISK_FULL_WINERRORS = {112, 39}
_QUOTA_ERRNOS = {getattr(errno, "EDQUOT", -1)}

# PNG of a photographic BGR frame is ~45-65 % of raw; flat/synthetic frames are far smaller. Be conservative.
PNG_RATIO = 0.65
# Mux output (h264/hevc at the render bitrate) is bounded by the source file; transcodes can be larger.
OUTPUT_FACTOR = 1.5
# Never plan to fill the volume to the last byte: the OS, the checkpoint rewrites and the manifest need room.
DEFAULT_RESERVE_BYTES = 512 * 1024 * 1024


def is_disk_full(error: BaseException | None) -> bool:
    """True when ``error`` (or anything in its cause/context chain) is an out-of-space condition."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, OSError):
            if error.errno == errno.ENOSPC or error.errno in _QUOTA_ERRNOS or getattr(error, "winerror", None) in _DISK_FULL_WINERRORS:
                return True
        # PyAV/FFmpeg report ENOSPC through av.error.FFmpegError subclasses that carry errno as well.
        if getattr(error, "errno", None) == errno.ENOSPC:
            return True
        message = str(error).lower()
        if "no space left on device" in message or "not enough space on the disk" in message:
            return True
        error = error.__cause__ or error.__context__
    return False


@dataclass(frozen=True)
class SpaceEstimate:
    frame_store_bytes: int
    output_bytes: int
    reserve_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.frame_store_bytes + self.output_bytes + self.reserve_bytes

    def as_dict(self) -> dict[str, int]:
        return {"frame_store_bytes": self.frame_store_bytes, "output_bytes": self.output_bytes,
                "reserve_bytes": self.reserve_bytes, "required_bytes": self.required_bytes}


def estimate(width: int, height: int, frames: int | None, source_bytes: int, *,
             remaining_frames: int | None = None, reserve_bytes: int = DEFAULT_RESERVE_BYTES) -> SpaceEstimate:
    """Space a job still needs. ``remaining_frames`` discounts frames a checkpoint already stored."""
    frames = max(0, frames or 0)
    pending = frames if remaining_frames is None else max(0, min(frames, remaining_frames))
    return SpaceEstimate(int(pending * width * height * 3 * PNG_RATIO), int(source_bytes * OUTPUT_FACTOR), reserve_bytes)


def free_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def require(path: Path, needed: SpaceEstimate) -> None:
    """Refuse to start a job the volume cannot hold. The failure is actionable and keeps no partial state."""
    available = free_bytes(path)
    if available < needed.required_bytes:
        raise DomainError(
            "DISK_SPACE_LOW",
            f"Rendering needs about {needed.required_bytes // (1024 * 1024):,} MiB free on the output volume "
            f"but only {available // (1024 * 1024):,} MiB is available. Free some space or change the output folder, then retry.",
            507)


def tree_bytes(path: Path) -> int:
    """Total size of regular files below ``path`` (symlinks are not followed). Missing paths are 0."""
    total = 0
    try:
        stack = [Path(path)]
        while stack:
            current = stack.pop()
            with __import__("os").scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
    except (FileNotFoundError, NotADirectoryError):
        return 0
    return total
