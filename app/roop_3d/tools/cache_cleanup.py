"""Dry-run report for stale temporary files in the package-owned cache only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time


TEMP_SUFFIXES = {".tmp", ".part", ".partial"}


@dataclass(frozen=True)
class CacheCandidate:
    relative_path: str
    size_bytes: int
    modified_at: float


def find_candidates(cache_root: Path, *, older_than_seconds: int = 24 * 60 * 60,
                    now: float | None = None) -> tuple[CacheCandidate, ...]:
    """List stale temporary regular files without opening or changing them."""
    if older_than_seconds < 0:
        raise ValueError("older_than_seconds cannot be negative")
    package_root = Path(__file__).absolute().parents[1]
    expected = package_root / "cache"
    requested = cache_root.absolute()
    if requested != expected.absolute():
        raise ValueError("Cache cleanup is restricted to roop_3d/cache")
    if requested.is_symlink() or not requested.exists():
        return ()
    current = time.time() if now is None else now
    cutoff = current - older_than_seconds
    result: list[CacheCandidate] = []
    for path in requested.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_file() or path.suffix.casefold() not in TEMP_SUFFIXES:
            continue
        info = path.stat(follow_symlinks=False)
        if info.st_mtime <= cutoff:
            result.append(CacheCandidate(path.relative_to(requested).as_posix(), info.st_size, info.st_mtime))
    return tuple(sorted(result, key=lambda item: item.relative_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-than-hours", type=int, default=24)
    args = parser.parse_args()
    if args.older_than_hours < 0:
        parser.error("--older-than-hours must be nonnegative")
    cache = Path(__file__).absolute().parents[1] / "cache"
    candidates = find_candidates(cache, older_than_seconds=args.older_than_hours * 3600)
    for item in candidates:
        print(f"DRY_RUN\t{item.size_bytes}\t{item.relative_path}")
    print(f"DRY_RUN\tcandidates={len(candidates)}\troot=roop_3d/cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
