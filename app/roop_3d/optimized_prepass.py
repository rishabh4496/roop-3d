"""Sparse candidates and dense verification; checkpoints contain numbers only."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.storage import atomic_bytes, owned
from roop_3d.media.video_reader import VideoReader
from roop_3d.render_policy import RenderPolicy

SPARSE = "STAGE_1_SPARSE_PRESCAN"
PREPASS = "STAGE_2_ACTIVE_PREPASS"
MAIN = "STAGE_3_MAIN_SWAP"


@dataclass(frozen=True)
class LockedFace:
    frame: int
    track_id: int
    box: np.ndarray
    landmarks5: np.ndarray


@dataclass(frozen=True)
class PrepassResult:
    ranges: tuple[tuple[int, int], ...]
    faces: tuple[LockedFace, ...]
    total_frames: int
    restored: bool = False

    @property
    def verified_target_frames(self):
        return len({v.frame for v in self.faces})

    @property
    def skipped_frames_pct(self):
        return 100.0 * (self.total_frames - self.verified_target_frames) / max(1, self.total_frames)

    def by_frame(self):
        result = {}
        for face in self.faces:
            result.setdefault(face.frame, []).append(face)
        return result


def merge_ranges(indices, total: int, padding: int):
    merged = []
    for index in sorted(set(indices)):
        start, end = max(0, index - padding), min(total, index + padding + 1)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return tuple(tuple(v) for v in merged)


class SparsePrepass:
    def __init__(self, root: Path, project_id: str, source: Path, *, detect: Callable,
                 track: Callable, policy: RenderPolicy = RenderPolicy(), emit: Callable[[dict], None] | None = None,
                 identity: dict | None = None, reader_factory=None, check: Callable[[], None] = lambda: None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
            raise ValueError("Invalid checkpoint project ID")
        self.root, self.project_id, self.source = root, project_id, source
        self.detect, self.track, self.policy, self.emit, self.check = detect, track, policy, emit, check
        self.identity = identity or {}
        def frames():
            if policy.strict_nvdec:
                from roop_3d.hardware_streamer import NvdecFrameSource
                from roop_3d.media.video_reader import DecodedFrame
                from roop_3d.media.timestamps import Timestamp
                from fractions import Fraction
                for image, metadata in NvdecFrameSource(source, strict=True).frames():
                    yield DecodedFrame(metadata.index, Timestamp(metadata.pts, Fraction(metadata.time_base_num,
                        metadata.time_base_den)), 1, metadata.keyframe, image)
            else:
                yield from VideoReader(source, hardware="auto" if policy.hardware_profile != "cpu" else "software").frames()
        self.reader_factory = reader_factory or frames
        self.relative = f"roop_3d/.runtime/prepass/{project_id}.stage2.npz"

    def signal(self, stage, state, **values):
        if self.emit:
            self.emit({"stage": stage, "state": state, **values})

    def _identity(self):
        return {"version": 3, "stabilizer": {"min_cutoff": self.policy.temporal_min_cutoff, "beta": self.policy.temporal_beta, "d_cutoff": self.policy.temporal_d_cutoff}, "source_sha256": sha256_file(self.source)[1], "identity": self.identity,
                "interval": self.policy.sparse_interval, "padding": self.policy.range_padding}

    def _load(self, identity):
        path = owned(self.root, self.relative)
        if not path.exists():
            return None
        # Header and expansion checks precede np.load; never accept arbitrary npz imports.
        import zipfile
        if path.stat().st_size > 128 * 1024**2:
            raise ValueError("Prepass checkpoint exceeds budget")
        with zipfile.ZipFile(path) as archive:
            if (len(archive.infolist()) != 5 or sum(v.file_size for v in archive.infolist()) > 256 * 1024**2
                    or set(archive.namelist()) != {"identity.npy", "ranges.npy", "faces.npy", "total.npy", "digest.npy"}):
                raise ValueError("Prepass checkpoint archive is invalid")
            for entry in archive.infolist():
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version != (1, 0):
                        raise ValueError("Unsupported checkpoint NumPy header version")
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    import math
                    size = math.prod(shape) * dtype.itemsize
                    if dtype.hasobject or fortran or size + stream.tell() != entry.file_size or size > 256 * 1024**2:
                        raise ValueError("Checkpoint NumPy shape/size/object header rejected")
        with np.load(path, allow_pickle=False) as archive:
            saved = bytes(archive["identity"].tolist()).decode()
            if json.loads(saved) != identity:
                raise ValueError("Prepass checkpoint identity differs; checkpoint preserved")
            faces, ranges, total = archive["faces"], archive["ranges"], int(archive["total"].item())
            if faces.ndim != 2 or faces.shape[1] != 16 or ranges.ndim != 2 or ranges.shape[1] != 2 or total < 1:
                raise ValueError("Invalid prepass checkpoint tensor layout")
            import hashlib
            digest = hashlib.sha256(faces.tobytes() + ranges.tobytes() + str(total).encode() + saved.encode()).hexdigest()
            if archive["digest"].item() != digest or not np.isfinite(faces).all():
                raise ValueError("Prepass checkpoint integrity failed")
            if (np.any(faces[:, 0] < 0) or np.any(faces[:, 0] >= total) or np.any(faces[:, :2] != np.floor(faces[:, :2]))
                    or np.any(faces[:, 1] < 0) or np.any(ranges < 0) or np.any(ranges[:, 1] > total)
                    or np.any(ranges[:, 0] >= ranges[:, 1])):
                raise ValueError("Prepass checkpoint index bounds failed")
            locked = tuple(LockedFace(int(row[0]), int(row[1]), row[2:6].astype(np.float32),
                                      row[6:].reshape(5, 2).astype(np.float32)) for row in faces)
            return PrepassResult(tuple(map(tuple, ranges.tolist())), locked, total, True)

    def run(self):
        identity = self._identity()
        restored = self._load(identity)
        if restored:
            self.signal(SPARSE, "RESTORED", active_ranges_count=len(restored.ranges))
            self.signal(PREPASS, "RESTORED", verified_target_frames=restored.verified_target_frames,
                        skipped_frames_pct=restored.skipped_frames_pct)
            return restored
        self.signal(SPARSE, "RUNNING")
        hits, total, sampled = [], 0, 0
        for decoded in self.reader_factory():
            self.check()
            total = decoded.index + 1
            if decoded.key_frame or decoded.index % self.policy.sparse_interval == 0:
                sampled += 1
                if len(self.detect(decoded.image)):
                    hits.append(decoded.index)
        if not total:
            raise ValueError("No source frames")
        ranges = merge_ranges(hits, total, self.policy.range_padding)
        # An estimate describes sampled candidate ranges, never a dense absence guarantee.
        estimated_skip = 100 * (total - sum(b - a for a, b in ranges)) / total
        self.signal(SPARSE, "DONE", active_ranges_count=len(ranges), skipped_frames_pct=estimated_skip,
                    sparse_sampled_frames=sampled, absence_evidence="SPARSE_ESTIMATE")
        self.signal(PREPASS, "RUNNING", active_ranges_count=len(ranges))
        faces = []
        from roop_3d.temporal_stabilizer import TemporalStabilizer
        stabilizer = TemporalStabilizer(min_cutoff=self.policy.temporal_min_cutoff, beta=self.policy.temporal_beta, d_cutoff=self.policy.temporal_d_cutoff)
        verified_count = 0
        for decoded in self.reader_factory():
            self.check()
            verified_count = decoded.index + 1
            active_ids = set()
            if any(start <= decoded.index < end for start, end in ranges):
                for track_id, box, points in self.track(decoded.image):
                    active_ids.add(int(track_id))
                    box, points = np.asarray(box, np.float32), np.asarray(points, np.float32)
                    if (box.shape != (4,) or points.shape != (5, 2) or not np.isfinite(box).all()
                            or not np.isfinite(points).all() or int(track_id) != track_id or track_id < 0):
                        raise ValueError("Prepass tracker returned invalid geometry")
                    if box[2] > box[0] and box[3] > box[1]:
                        points = stabilizer.filter(int(track_id), "landmarks5", points, float(decoded.timestamp.seconds))
                        faces.append(LockedFace(decoded.index, int(track_id), box.copy(), points.astype(np.float32)))
            stabilizer.retain(active_ids)
        if verified_count != total or sha256_file(self.source)[1] != identity["source_sha256"]:
            raise ValueError("Prepass source changed between scan/verification passes")
        result = PrepassResult(ranges, tuple(faces), total)
        rows = np.array([[v.frame, v.track_id, *v.box.tolist(), *v.landmarks5.reshape(-1).tolist()] for v in faces], np.float64).reshape(-1, 16)
        ranges_array = np.array(ranges, np.int64).reshape(-1, 2)
        saved = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        import hashlib
        digest = hashlib.sha256(rows.tobytes() + ranges_array.tobytes() + str(total).encode() + saved.encode()).hexdigest()
        buffer = BytesIO()
        np.savez_compressed(buffer, identity=np.frombuffer(saved.encode(), np.uint8), faces=rows,
                            ranges=ranges_array, total=np.int64(total), digest=np.array(digest))
        atomic_bytes(self.root, self.relative, buffer.getvalue())
        self.signal(PREPASS, "DONE", verified_target_frames=result.verified_target_frames,
                    skipped_frames_pct=result.skipped_frames_pct)
        return result
