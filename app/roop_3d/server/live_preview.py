"""Production live-preview endpoints: cache-first, non-blocking, mutex-protected.

    POST /api/preview               full swap payload -> one swapped frame (PNG) + face metadata
    GET  /api/target/preview        one swapped frame for a previously-registered configuration
    GET  /api/target/preview_grid   contact sheet of swapped frames spread across the clip
    GET  /api/target/preview_seq    consecutive swapped frames (JSON manifest of per-frame URLs)

(`/api/<x>` is served from `/api/v1/<x>` by the alias middleware.)

Concurrency contract
--------------------
1. A frame whose (media_id, frame_index, swap_config_hash) is in the in-memory LRU is served
   immediately, even while a job, Stage 1/2 pre-pass or TensorRT compile owns the GPU.
2. An uncached frame while the engine is owned elsewhere is **not** queued: the request returns
   HTTP 503 `{"busy": true, "stage": ..., "retry_after_ms": 1000}` with `Retry-After: 1`
   immediately: no model is loaded, no media is decoded, no thread-pool slot is taken.
3. Actual inference runs inside `EngineGate.preview_section()`, so a Start-button model
   teardown/re-bind waits for it to finish and new previews are refused while it waits.
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import re
import threading
import time
from threading import Event
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool

from roop_3d.audit.isolation import safe_path
from roop_3d.core.animation import FaceSegment, ParameterCurves
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.faceset.identity_blend import IdentityBlendSpec
from roop_3d.jobs.schema import JobRequest
from roop_3d.media.finishing import ExportOptions
from roop_3d.media.schema import FaceInspection, TrackToFaceSetMap
from roop_3d.server.engine_gate import REBIND, EngineBusy
from roop_3d.server.pipeline_status import STAGE_ORDER
from roop_3d.server.workstation_preview import PreviewRequest

router = APIRouter(prefix="/api/v1")

RETRY_AFTER_MS = 1000
SoloNode = Literal["final", "detector", "pose", "swapper", "matte", "restorer", "color", "source", "normal_map", "sh_lighting"]
RestorerName = Literal["off", "gpen", "codeformer", "gfpgan", "ultramax"]
_FACESET = r"^[A-Za-z0-9_-]{1,64}$"
_TRACK = r"^track_[0-9]{4}$"
_COMPILE_STAGES = {"COMPILING_TRT": "TENSORRT_COMPILATION", "RUNNING_CANARY": "TENSORRT_CANARY"}


class PreviewCancelled(Exception):
    """Raised when client disconnects or aborts during preview slot acquisition or rendering."""


# --------------------------------------------------------------------------- payload
class MaskStroke(Contract):
    """A paint (keep original target pixels) or erase stroke in frame-normalised coordinates."""
    mode: Literal["paint", "erase"] = "paint"
    radius: float = Field(gt=0, le=0.5, description="Brush radius as a fraction of frame height")
    points: tuple[tuple[float, float], ...] = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def normalised(self) -> "MaskStroke":
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Stroke points are normalised to 0..1")
        return self


class OcclusionSettings(Contract):
    erosion_px: float = Field(default=0, ge=-32, le=32)
    feather_px: float = Field(default=5, ge=0, le=64)


class LivePreviewPayload(Contract):
    engine_id: str = Field(default="reference_onnx", pattern=r"^[a-z0-9][a-z0-9_]{0,95}$")
    # media + frame
    target_path: str = Field(min_length=1, max_length=1024)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    identity_lock_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    frame_index: int | None = Field(default=None, ge=0, le=10_000_000)
    timestamp: float | None = Field(default=None, ge=0, le=86400)
    # who is swapped with whom: explicit mappings OR sources + a target selection
    face_mappings: tuple[TrackToFaceSetMap, ...] | None = Field(default=None, min_length=1, max_length=64)
    source_face_ids: tuple[str, ...] = Field(default=(), max_length=8)
    fsz_faceset_id: str | None = Field(default=None, pattern=_FACESET,
                                       description="FaceSet id returned when a .fsz archive was uploaded")
    target_face_id: str | None = Field(default=None, pattern=_TRACK)
    all_faces: bool = False
    cluster_index: int | None = Field(default=None, ge=0, le=63)
    identity_blend: IdentityBlendSpec | None = None
    # geometry
    frontalization_weight: float | None = Field(default=None, ge=0, le=1)   # 2.5D -> 3D mesh blend (post-swap projection)
    pose_frontalization: float | None = Field(default=None, ge=0, le=1)     # pre-swap 3D pose normalisation (Stage 5)
    tps_expression_weight: float | None = Field(default=None, ge=0, le=1)   # TPS expression warp (Stage 5)
    pose_strength: float | None = Field(default=None, ge=0, le=1)
    # mask
    occlusion: OcclusionSettings = OcclusionSettings()
    mask_base64: str | None = Field(default=None, max_length=12_000_000)
    mask_strokes: tuple[MaskStroke, ...] = Field(default=(), max_length=256)
    # enhancer + grading
    restorer: RestorerName = "off"
    restoration_blend: float | None = Field(default=None, ge=0, le=1)
    identity_blend_t: float | None = Field(default=None, ge=0, le=1)
    fidelity_weight: float = Field(default=0.5, ge=0, le=1)
    color_method: Literal['lab', 'reinhard_lab', 'histogram'] = 'lab'
    color_strength: float = Field(default=0.65, ge=0, le=1)
    swap_weight: float = Field(default=1, ge=0, le=1)
    bypass: tuple[Literal["restorer", "color"], ...] = Field(default=(), max_length=2)
    solo_node: SoloNode = "final"
    # Stage 7: the 3D render mode (Stage 4) and compositor occlusion strategy (Stage 5) reach the live preview too, so
    # the workstation previews exactly what Start would render. Omitted = the existing defaults (standard / standard).
    render_mode: Literal['standard', 'pose_3d', 'reenactment_3d'] | None = None
    occlusion_mode: Literal['standard', 'robust'] | None = None
    relight: bool | None = None
    relight_strength: float | None = Field(default=None, ge=0, le=1)
    specular_sheen: float | None = Field(default=None, ge=0, le=1)
    shadow_preservation: float | None = Field(default=None, ge=0, le=1)
    ultramax_small_face_mode: Literal['bypass', 'adaptive', 'off'] | None = None
    ultramax_min_native_pixels: int | None = Field(default=None, ge=64, le=1024)
    lighting_mode: Literal['auto', 'manual'] | None = None
    manual_light_yaw: float | None = Field(default=None, ge=-180, le=180)
    manual_light_pitch: float | None = Field(default=None, ge=-90, le=90)
    # job-request pass-through so the preview matches what Start would render
    face_segments: tuple[FaceSegment, ...] | None = Field(default=None, max_length=256)
    parameter_curves: ParameterCurves | None = None
    export: ExportOptions | None = None

    @model_validator(mode="after")
    def coherent(self) -> "LivePreviewPayload":
        if (self.frame_index is None) == (self.timestamp is None):
            raise ValueError("Supply exactly one of frame_index or timestamp")
        explicit = self.face_mappings is not None
        has_target = bool(self.target_face_id) + self.all_faces
        if explicit and (self.source_face_ids or self.fsz_faceset_id or self.target_face_id or self.all_faces):
            raise ValueError("face_mappings cannot be combined with source/target selectors")
        if not explicit:
            if not (self.source_face_ids or self.fsz_faceset_id or self.identity_blend):
                raise ValueError("Select source_face_ids, fsz_faceset_id or identity_blend")
            if has_target != 1:
                raise ValueError("Select exactly one of target_face_id or all_faces")
            if self.cluster_index is not None and not self.all_faces:
                raise ValueError("cluster_index requires all_faces")
        if any(not re.fullmatch(_FACESET, item) for item in self.source_face_ids):
            raise ValueError("source_face_ids must be FaceSet identifiers")
        self.resolved_sources()  # raises on inconsistent blend declarations
        return self

    # -- source / blend resolution ------------------------------------------------
    def resolved_sources(self) -> tuple[tuple[str, ...], IdentityBlendSpec | None]:
        ids = tuple(dict.fromkeys((*self.source_face_ids, *([self.fsz_faceset_id] if self.fsz_faceset_id else ()))))
        blend = self.identity_blend
        if blend is not None:
            if ids and set(ids) != set(blend.source_face_ids):
                raise ValueError("identity_blend.source_face_ids must match the selected sources")
            return blend.source_face_ids, blend
        if len(ids) > 1:  # several identities with no recipe: equal-weight spherical mean
            return ids, IdentityBlendSpec(method="slerp", source_face_ids=ids)
        return ids, None

    def faceset_ids(self) -> tuple[str, ...]:
        ids = list(self.resolved_sources()[0])
        ids += [m.faceset_id for m in self.face_mappings or ()]
        ids += [s.faceset_id for s in self.face_segments or () if s.faceset_id]
        return tuple(dict.fromkeys(ids))

    def validate_restorer(self) -> None:
        # gpen, codeformer, gfpgan and ultramax each have their own adapter (and ONNX engine) in the preview compositor.
        return None

    def unsupported(self) -> list[str]:
        """Accepted-but-inert fields: the engine has no corresponding control. Never silently dropped."""
        items = []
        if "fidelity_weight" in self.model_fields_set and self.restorer not in ("codeformer", "ultramax"):
            items.append("fidelity_weight")  # only the CodeFormer-based engines expose a fidelity input
        return items

    def config_digest(self) -> dict[str, Any]:
        """Everything that changes the pixels, nothing that selects the frame."""
        value = self.model_dump(mode="json", exclude={"frame_index", "timestamp"})
        for key in ("identity_lock_id", "pose_strength"):
            if value[key] is None:
                value.pop(key)
        if value.get("mask_base64"):
            value["mask_base64"] = hashlib.sha256(value["mask_base64"].encode()).hexdigest()
        return value


# --------------------------------------------------------------------------- cache
class CacheEntry:
    """One rendered preview frame.

    Built from encoded bytes (the original form) or from the raw BGR frame (``from_raw``).  A raw entry is stored the
    moment inference finishes -- the GPU slot is released without spending any time in an image encoder -- and its PNG
    is produced on first use of ``.image``, by whichever HTTP worker needs it.  The binary WebSocket stream never
    needs the PNG at all: it hands ``raw`` to the JPEG hub.
    """
    __slots__ = ("_image", "media_type", "meta", "raw", "_lock", "counted")

    def __init__(self, image: bytes | None, media_type: str, meta: dict[str, Any], *, raw: np.ndarray | None = None):
        if image is None and raw is None:
            raise ValueError("A cache entry needs encoded bytes or a raw frame")
        self._image, self.media_type, self.meta, self.raw = image, media_type, meta, raw
        self._lock = threading.Lock()
        self.counted = 0                      # bytes charged to the cache when it was stored

    @classmethod
    def from_raw(cls, raw: np.ndarray, meta: dict[str, Any]) -> "CacheEntry":
        return cls(None, "image/png", meta, raw=raw)

    @property
    def nbytes(self) -> int:
        return len(self._image) if self._image is not None else int(self.raw.nbytes)

    @property
    def image(self) -> bytes:
        if self._image is None:
            with self._lock:
                if self._image is None:
                    ok, encoded = cv2.imencode(".png", self.raw, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if not ok:
                        raise DomainError("PREVIEW_ENCODE_FAILED", "Preview encoding failed", 500)
                    self._image = encoded.tobytes()
        return self._image

    def bgr(self) -> np.ndarray:
        """The decoded frame (the raw buffer when we still hold it, otherwise the PNG decoded)."""
        if self.raw is not None:
            return self.raw
        return cv2.imdecode(np.frombuffer(self.image, np.uint8), cv2.IMREAD_COLOR)


class PreviewCache:
    """LRU keyed by (media_id, exact frame selector, config hash), bounded by count and bytes."""

    def __init__(self, max_entries: int = 128, max_bytes: int = 192 * 1024 * 1024):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._items: OrderedDict[tuple[str, int | str, str], CacheEntry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple[str, int | str, str]) -> CacheEntry | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                self._items.move_to_end(key)
            return entry

    def put(self, key: tuple[str, int | str, str], entry: CacheEntry) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.counted
            if not entry.counted:
                entry.counted = entry.nbytes
            self._items[key] = entry
            self._bytes += entry.counted
            while len(self._items) > self.max_entries or (self._bytes > self.max_bytes and len(self._items) > 1):
                _, evicted = self._items.popitem(last=False)
                self._bytes -= evicted.counted

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes


class ConfigRegistry:
    """Lets cacheable GET URLs reference a payload the client POSTed earlier by hash."""

    def __init__(self, maximum: int = 64):
        self.maximum = maximum
        self._items: OrderedDict[str, LivePreviewPayload] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, digest: str, payload: LivePreviewPayload) -> None:
        with self._lock:
            self._items[digest] = payload
            self._items.move_to_end(digest)
            while len(self._items) > self.maximum:
                self._items.popitem(last=False)

    def get(self, digest: str) -> LivePreviewPayload | None:
        with self._lock:
            value = self._items.get(digest)
            if value is not None:
                self._items.move_to_end(digest)
            return value


# --------------------------------------------------------------------------- service
@dataclass
class FrameOutcome:
    index: int
    status: Literal["ready", "busy", "pending", "error"]
    entry: CacheEntry | None = None
    cached: bool = False
    stage: str | None = None
    error: dict[str, str] | None = None


@dataclass
class _MediaInfo:
    fps: float
    width: int
    height: int
    frames: int | None
    stamp: tuple[int, int] = field(default=(0, 0))


class LivePreviewService:
    def __init__(self, state):
        self.state = state
        self.cache = PreviewCache()
        self.configs = ConfigRegistry()
        self._media: dict[str, _MediaInfo] = {}
        self._slot = threading.Lock()  # one live render at a time (inside the thread pool, cancel-aware)

    # -- identity of a configuration ------------------------------------------------
    def environment_stamp(self, payload: LivePreviewPayload) -> dict[str, Any]:
        root = self.state.root
        stamps = {}
        for faceset_id in payload.faceset_ids():
            row = []
            for relative in (f"roop_3d/.runtime/facesets/{faceset_id}.json", f"roop_3d/outputs/facesets/{faceset_id}.fsz",
                             f"roop_3d/.runtime/facesets/{faceset_id}.fsz"):
                try:
                    row.append((root / relative).stat().st_mtime_ns)
                except OSError:
                    row.append(None)
            stamps[faceset_id] = row
        settings = hashlib.sha256(self.state.settings.model_dump_json().encode()).hexdigest()[:16]
        return {"settings": settings, "facesets": stamps}

    def config_hash(self, payload: LivePreviewPayload) -> str:
        from roop_3d.engine.registry import EngineRegistry
        EngineRegistry().validate(payload.engine_id)
        body = json.dumps({"config": payload.config_digest(), "env": self.environment_stamp(payload)},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()[:32]

    # -- engine ownership -----------------------------------------------------------
    def busy_stage(self) -> str | None:
        """Name of whatever owns the GPU right now, or None when previews may run."""
        snapshot = self.state.pipeline.snapshot()
        compiling = _COMPILE_STAGES.get(snapshot.engine_state)
        if compiling:
            return compiling
        reason = self.state.engine_gate.busy_reason()
        if reason is None:
            return None
        if reason != REBIND:
            for stage in STAGE_ORDER:
                if snapshot.stages.get(stage) == "RUNNING":
                    return stage
        return reason

    # -- media facts (cached; touched only on a cache miss that is allowed to run) ---
    def media_info(self, target_path: str) -> _MediaInfo:
        path = self.state.media.path(target_path)
        status = path.stat()
        stamp = (status.st_mtime_ns, status.st_size)
        known = self._media.get(target_path)
        if known is not None and known.stamp == stamp:
            return known
        probe = self.state.media.probe(target_path)
        width, height = (probe.height, probe.width) if probe.rotation_degrees % 180 else (probe.width, probe.height)
        info = _MediaInfo(probe.fps or 25.0, width, height, probe.declared_frame_count, stamp)
        self._media[target_path] = info
        return info

    def frame_for(self, payload: LivePreviewPayload) -> int:
        if payload.frame_index is not None:
            return payload.frame_index
        return max(0, round(payload.timestamp * self.media_info(payload.target_path).fps))

    # -- orchestration ----------------------------------------------------------------
    @staticmethod
    def _cache_key(payload: LivePreviewPayload, index: int, digest: str) -> tuple[str, int | str, str]:
        # Nominal FPS cannot identify a VFR frame. Preserve public config hashes while keeping
        # timestamp selectors separate from ordinal selectors and from one another.
        selector = f"timestamp:{payload.timestamp.hex()}" if payload.timestamp is not None else index
        return payload.inspection_id, selector, digest

    async def frames(self, payload: LivePreviewPayload, indices: list[int], cancel: Event, *,
                     budget_s: float | None = None, strict: bool = True) -> tuple[str, list[FrameOutcome]]:
        """Cache-first resolution of `indices`. Returns (config_hash, outcomes in order)."""
        digest = await run_in_threadpool(self.config_hash, payload) if payload.mask_base64 else self.config_hash(payload)
        self.configs.put(digest, payload)
        outcomes: dict[int, FrameOutcome] = {}
        for index in indices:
            hit = self.cache.get(self._cache_key(payload, index, digest))
            if hit is not None:
                outcomes[index] = FrameOutcome(index, "ready", hit, cached=True)
        missing = [index for index in indices if index not in outcomes]
        if missing:
            stage = self.busy_stage()  # in-memory reads only: answers without loading a model or decoding media
            if stage:
                for index in missing:
                    outcomes[index] = FrameOutcome(index, "busy", stage=stage)
            else:
                deadline = time.monotonic() + budget_s if budget_s else None
                computed = await run_in_threadpool(self._compute, payload, missing, digest, cancel, deadline, strict)
                outcomes.update({item.index: item for item in computed})
        return digest, [outcomes[index] for index in indices]

    def _acquire_slot(self, cancel: Event) -> None:
        while not self._slot.acquire(timeout=0.05):
            if cancel.is_set():
                raise PreviewCancelled("Preview cancelled")

    def _compute(self, payload: LivePreviewPayload, missing: list[int], digest: str, cancel: Event,
                 deadline: float | None, strict: bool) -> list[FrameOutcome]:
        self._acquire_slot(cancel)
        try:
            base = self._build(payload)
            info = self.media_info(payload.target_path)
            results, halted = [], None
            for index in missing:
                if cancel.is_set():
                    raise PreviewCancelled("Preview cancelled")
                key = self._cache_key(payload, index, digest)
                hit = self.cache.get(key)  # a queued twin may have rendered it while we waited
                if hit is not None:
                    results.append(FrameOutcome(index, "ready", hit, cached=True))
                elif halted is not None:
                    results.append(FrameOutcome(index, "busy", stage=halted))
                elif deadline is not None and time.monotonic() > deadline:
                    results.append(FrameOutcome(index, "pending"))
                else:
                    try:
                        entry = self._render_one(payload, base, info, index, digest, cancel)
                        self.cache.put(key, entry)
                        # Once decoded, timestamp and ordinal selectors may safely share this exact frame.
                        # Never alias a VFR timestamp through its rounded nominal-FPS index.
                        actual = int(entry.meta["decoded_frame_index"])
                        self.cache.put((payload.inspection_id, actual, digest), entry)
                        stamp = float(entry.meta["timestamp"])
                        self.cache.put((payload.inspection_id, f"timestamp:{stamp.hex()}", digest), entry)
                        results.append(FrameOutcome(index, "ready", entry))
                    except EngineBusy as busy:
                        halted = busy.stage
                        results.append(FrameOutcome(index, "busy", stage=busy.stage))
                    except DomainError as error:
                        if error.code == "PREVIEW_CANCELLED":
                            raise PreviewCancelled("Preview cancelled") from None
                        if strict:
                            raise
                        results.append(FrameOutcome(index, "error", error={"code": error.code, "message": error.message}))
            return results
        finally:
            self._slot.release()

    # -- building the engine request ---------------------------------------------------
    def _build(self, payload: LivePreviewPayload) -> PreviewRequest:
        payload.validate_restorer()
        root = self.state.root
        inspection_path = safe_path(root, f"roop_3d/.runtime/inspections/{payload.inspection_id}.json")
        if not inspection_path.is_file():
            raise DomainError("INSPECTION_NOT_FOUND", "Saved inspection is unavailable", 422)
        try:
            inspection = FaceInspection.model_validate_json(inspection_path.read_text("utf-8"))
        except (OSError, ValueError):
            raise DomainError("INSPECTION_INVALID", "Saved inspection is invalid", 422) from None
        track_ids = [track.track_id for track in inspection.tracks]
        sources, blend = payload.resolved_sources()
        if (inspection.tracking_method == "identity_lock_v1" and payload.face_mappings is None
                and not payload.target_face_id and payload.cluster_index is None):
            # Identity-locked inspections never fall back to "swap every detected face".
            raise DomainError("TARGET_SELECTION_REQUIRED", "Select the target face (face_mappings) to preview", 422)
        if payload.face_mappings is not None:
            mappings = payload.face_mappings
        elif payload.target_face_id:
            mappings = (TrackToFaceSetMap(track_id=payload.target_face_id, faceset_id=sources[0]),)
        elif payload.cluster_index is not None:
            if payload.cluster_index >= len(track_ids):
                raise DomainError("CLUSTER_NOT_FOUND", "cluster_index is beyond the tracks in this inspection", 422)
            mappings = (TrackToFaceSetMap(track_id=track_ids[payload.cluster_index], faceset_id=sources[0]),)
        else:
            mappings = tuple(TrackToFaceSetMap(track_id=item, faceset_id=sources[0]) for item in track_ids[:64])
        if any(item.track_id not in track_ids for item in mappings):
            raise DomainError("TRACK_NOT_FOUND", "Selected target face is absent from this inspection", 422)

        restoring = payload.restorer in ("gpen", "codeformer", "gfpgan", "ultramax")
        blend_amount = (payload.restoration_blend if payload.restoration_blend is not None else 0.7) if restoring else 0.0
        values: dict[str, Any] = {"restoration_blend": blend_amount, "feather_px": payload.occlusion.feather_px,
                                  "erosion_px": payload.occlusion.erosion_px, "color_strength": payload.color_strength,
                                  "swap_weight": payload.swap_weight, "bypass": payload.bypass,
                                  "color_method": payload.color_method,
                                  "pose_mode": "mesh_blend" if payload.frontalization_weight is not None else "retarget"}
        if payload.frontalization_weight is not None:
            values["pose_strength"] = payload.frontalization_weight
        elif payload.pose_strength is not None:
            values["pose_strength"] = payload.pose_strength
        if payload.pose_frontalization is not None:
            values["pose_frontalization"] = payload.pose_frontalization
        if payload.tps_expression_weight is not None:
            values["tps_expression_weight"] = payload.tps_expression_weight
        if payload.restorer == 'codeformer':
            values.update(restorer_model='codeformer', fidelity_weight=payload.fidelity_weight)
        elif payload.restorer == 'ultramax':
            values.update(restorer_model='ultramax', fidelity_weight=payload.fidelity_weight)
        elif payload.restorer == 'gfpgan':
            values.update(restorer_model='gfpgan')
        if payload.ultramax_small_face_mode is not None:
            values['ultramax_small_face_mode'] = payload.ultramax_small_face_mode
        if payload.ultramax_min_native_pixels is not None:
            values['ultramax_min_native_pixels'] = payload.ultramax_min_native_pixels
        if payload.identity_blend_t is not None:
            values['identity_blend_t'] = payload.identity_blend_t
        if payload.relight is not None:
            values['relight'] = payload.relight
        elif payload.solo_node in ('normal_map', 'sh_lighting') or payload.relight_strength is not None:
            values['relight'] = True
        if payload.relight_strength is not None:
            values['relight_strength'] = payload.relight_strength
        if payload.specular_sheen is not None:
            values['specular_sheen'] = payload.specular_sheen
        if payload.shadow_preservation is not None:
            values['shadow_preservation'] = payload.shadow_preservation
        if payload.lighting_mode is not None:
            values['lighting_mode'] = payload.lighting_mode
        if payload.manual_light_yaw is not None:
            values['manual_light_yaw'] = payload.manual_light_yaw
        if payload.manual_light_pitch is not None:
            values['manual_light_pitch'] = payload.manual_light_pitch
        if payload.render_mode is not None:
            values['render_mode'] = payload.render_mode
        if payload.occlusion_mode is not None:
            values['occlusion_mode'] = payload.occlusion_mode
        parameters = WorkstationParameters.model_validate(values)

        mask = payload.mask_base64
        if payload.mask_strokes:
            info = self.media_info(payload.target_path)
            mask = rasterize_strokes(payload.mask_strokes, info.width, info.height, mask)
        job = JobRequest(engine_id=payload.engine_id, target_path=payload.target_path, inspection_id=payload.inspection_id, face_mappings=mappings,
                         identity_lock_id=payload.identity_lock_id,
                         processing="local", mask_base64=mask, workstation_parameters=parameters,
                         parameter_curves=payload.parameter_curves, identity_blend=blend, face_segments=payload.face_segments,
                         export=payload.export)
        return PreviewRequest(request=job, timestamp=0, parameters=parameters)

    def _render_one(self, payload: LivePreviewPayload, base: PreviewRequest, info: _MediaInfo, index: int,
                    digest: str, cancel: Event) -> CacheEntry:
        timestamp = payload.timestamp if payload.timestamp is not None else max(0.0, (index - 0.5) / info.fps)
        request = base.model_copy(update={"timestamp": min(timestamp, 86400.0),
                                         "frame_index": index if payload.frame_index is not None else None})
        try:
            raw = self.state.previews.render(request, cancel, raw=True)
        except DomainError as error:
            if error.code == "PREVIEW_CAPACITY":
                raise EngineBusy(self._running_stage() or "GPU_CAPACITY") from None
            if error.code == "PREVIEW_BUSY":
                raise EngineBusy("PREVIEW_IN_FLIGHT") from None
            raise
        image = raw["buffers"].get(payload.solo_node)
        if image is None:
            raise DomainError("SOLO_NODE_UNAVAILABLE", f"Pipeline node '{payload.solo_node}' produced no buffer for this frame", 422)
        meta = {"media_id": payload.inspection_id, "frame_index": raw["frame_index"], "swap_config_hash": digest,
                "solo_node": payload.solo_node, "width": int(image.shape[1]), "height": int(image.shape[0]),
                "decoded_frame_index": raw["frame_index"], "timestamp": raw["timestamp"], "faces": raw.get("faces", []),
                "applied": self._applied(payload), "unsupported": payload.unsupported(), "notes": raw.get("notes", []),
                "latency": raw.get("latency", {}), "provider": raw.get("provider"), "profile": str(raw.get("profile"))}
        # No encoder runs here: this thread holds the render slot. The entry keeps its own copy of the pixels and is
        # encoded (PNG for HTTP, JPEG for the binary stream) by whoever consumes it, after the slot is released.
        return CacheEntry.from_raw(np.ascontiguousarray(image).copy(), meta)

    @staticmethod
    def _applied(payload: LivePreviewPayload) -> list[str]:
        applied = ["swap_weight", "color_strength", "mask_erosion_px", "mask_feather_px"]
        if payload.frontalization_weight is not None:
            applied.append("frontalization_weight")
        if payload.pose_frontalization:
            applied.append("pose_frontalization")
        if payload.tps_expression_weight:
            applied.append("tps_expression_weight")
        if payload.restorer in ("gpen", "codeformer", "gfpgan", "ultramax"):
            applied.append("restorer:" + payload.restorer)
        if payload.restorer in ("codeformer", "ultramax"):
            applied.append("fidelity_weight")
        if payload.mask_base64 or payload.mask_strokes:
            applied.append("custom_mask")
        if payload.relight_strength is not None:
            applied.append("relight_strength")
        if payload.specular_sheen is not None:
            applied.append("specular_sheen")
        if payload.shadow_preservation is not None:
            applied.append("shadow_preservation")
        if payload.lighting_mode is not None:
            applied.append("lighting_mode")
            applied.append("custom_mask")
        if payload.resolved_sources()[1] is not None:
            applied.append(f"identity_blend:{payload.resolved_sources()[1].method}")
        return applied

    def _running_stage(self) -> str | None:
        snapshot = self.state.pipeline.snapshot()
        return next((stage for stage in STAGE_ORDER if snapshot.stages.get(stage) == "RUNNING"), None)


def rasterize_strokes(strokes, width: int, height: int, base_mask: str | None) -> str:
    """Burn paint/erase strokes (normalised coordinates) onto the manual-mask PNG."""
    canvas = np.zeros((height, width), np.uint8)
    if base_mask:
        data = base64.b64decode(base_mask.removeprefix("data:image/png;base64,"), validate=True)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.shape != canvas.shape:
            raise DomainError("MASK_SIZE_MISMATCH", "Manual mask must match upright media dimensions", 422)
        canvas = decoded
    for stroke in strokes:
        value = 255 if stroke.mode == "paint" else 0
        radius = max(1, round(stroke.radius * height))
        points = np.array([[round(x * (width - 1)), round(y * (height - 1))] for x, y in stroke.points], np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, value, 2 * radius, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), radius, value, -1, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise DomainError("MASK_INVALID", "Mask strokes could not be encoded", 422)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- HTTP
def busy_response(stage: str) -> JSONResponse:
    return JSONResponse({"busy": True, "stage": stage, "retry_after_ms": RETRY_AFTER_MS}, status_code=503,
                        headers={"Retry-After": str(max(1, RETRY_AFTER_MS // 1000)), "Cache-Control": "no-store"})


def image_response(entry: CacheEntry, *, cached: bool, digest: str, as_json: bool = False) -> Response:
    """``entry.image`` must already be encoded (see ``_single``): this runs on the event loop and must not block it."""
    meta = {**entry.meta, "cached": cached}
    if as_json:
        body = {**meta, "image": f"data:{entry.media_type};base64," + base64.b64encode(entry.image).decode("ascii")}
        return JSONResponse(body, headers={"Cache-Control": "no-store", "X-Preview-Cache": "HIT" if cached else "MISS"})
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
               "X-Preview-Cache": "HIT" if cached else "MISS", "X-Preview-Frame-Index": str(meta["frame_index"]),
               "X-Preview-Config-Hash": digest, "X-Preview-Media-Id": meta["media_id"],
               "X-Preview-Meta": json.dumps(meta, separators=(",", ":"))}
    return Response(entry.image, media_type=entry.media_type, headers=headers)


async def _cancellable(request: Request, call):
    """Run `call(cancel)`; a dropped client connection cancels the inference at its next checkpoint."""
    cancel = Event()
    work = asyncio.create_task(call(cancel))
    try:
        while not work.done():
            if await request.is_disconnected():
                cancel.set()
            await asyncio.wait({work}, timeout=.1)
        return await work
    finally:
        cancel.set()
        if not work.done():
            await asyncio.shield(work)


def _service(request: Request) -> LivePreviewService:
    return request.app.state.live_previews


def _registered(request: Request, config: str, media_id: str | None) -> LivePreviewPayload:
    payload = _service(request).configs.get(config)
    if payload is None:
        raise DomainError("PREVIEW_CONFIG_UNKNOWN", "Unknown swap configuration; POST /api/preview first", 404)
    if media_id is not None and media_id != payload.inspection_id:
        raise DomainError("MEDIA_MISMATCH", "media_id does not belong to this swap configuration", 409)
    return payload


async def _single(request: Request, payload: LivePreviewPayload, as_json: bool) -> Response:
    service = _service(request)

    async def work(cancel: Event):
        index = await run_in_threadpool(service.frame_for, payload) if payload.frame_index is None else payload.frame_index
        return await service.frames(payload, [index], cancel)

    try:
        digest, (outcome,) = await _cancellable(request, work)
    except (PreviewCancelled, DomainError) as error:
        if isinstance(error, PreviewCancelled) or getattr(error, "code", None) == "PREVIEW_CANCELLED":
            return Response(status_code=204, headers={"X-Preview-Status": "cancelled"})
        raise
    if outcome.status == "busy":
        return busy_response(outcome.stage or REBIND)
    assert outcome.entry is not None
    await run_in_threadpool(lambda: outcome.entry.image)       # PNG encode in a worker thread, not on the loop
    return image_response(outcome.entry, cached=outcome.cached, digest=digest, as_json=as_json)


@router.post("/preview")
async def live_preview(value: LivePreviewPayload, request: Request,
                       envelope: Literal["image", "json"] = Query(default="image", alias="response")):
    """`?response=json` returns metadata plus a base64 PNG instead of raw image bytes."""
    return await _single(request, value, envelope == "json")


@router.get("/target/preview")
async def target_preview(request: Request, config: str = Query(min_length=8, max_length=64),
                         frame_index: int | None = Query(default=None, ge=0, le=10_000_000),
                         timestamp: float | None = Query(default=None, ge=0, le=86400),
                         media_id: str | None = Query(default=None, max_length=64),
                         envelope: Literal["image", "json"] = Query(default="image", alias="response")):
    if (frame_index is None) == (timestamp is None):
        raise DomainError("FRAME_SELECTOR_INVALID", "Supply exactly one of frame_index or timestamp", 422)
    payload = _registered(request, config, media_id).model_copy(update={"frame_index": frame_index, "timestamp": timestamp})
    return await _single(request, payload, envelope == "json")


def _spread(count: int, last: int) -> list[int]:
    if count <= 1 or last <= 0:
        return [0] * max(count, 1)
    return [round(i * last / (count - 1)) for i in range(count)]


@router.get("/target/preview_grid")
async def target_preview_grid(request: Request, config: str = Query(min_length=8, max_length=64),
                              rows: int = Query(default=3, ge=1, le=4), cols: int = Query(default=3, ge=1, le=4),
                              budget_ms: int = Query(default=20000, ge=0, le=120000),
                              media_id: str | None = Query(default=None, max_length=64)):
    """Contact sheet of swapped frames evenly spread over the clip. Uncached cells are computed within the budget."""
    payload = _registered(request, config, media_id)
    service = _service(request)

    async def work(cancel: Event):
        info = await run_in_threadpool(service.media_info, payload.target_path)
        last = (info.frames - 1) if info.frames else int(info.fps * 10)
        indices = _spread(rows * cols, max(0, last))
        return await service.frames(payload.model_copy(update={"frame_index": 0, "timestamp": None}), indices,
                                    cancel, budget_s=budget_ms / 1000, strict=False)

    try:
        digest, outcomes = await _cancellable(request, work)
    except (PreviewCancelled, DomainError) as error:
        if isinstance(error, PreviewCancelled) or getattr(error, "code", None) == "PREVIEW_CANCELLED":
            return Response(status_code=204, headers={"X-Preview-Status": "cancelled"})
        raise
    ready = [item for item in outcomes if item.status == "ready"]
    if not ready:
        busy = next((item for item in outcomes if item.status == "busy"), None)
        if busy:
            return busy_response(busy.stage or REBIND)
        first_error = next((item.error for item in outcomes if item.error), None)
        raise DomainError((first_error or {}).get("code", "PREVIEW_EMPTY"),
                          (first_error or {}).get("message", "No grid cell could be rendered"), 422)
    tiles = [item.entry.bgr() if item.entry else None for item in outcomes]
    height, width = next(tile.shape[:2] for tile in tiles if tile is not None)
    height, width = min(height, 360), round(width * min(height, 360) / height)
    sheet = np.zeros((rows * height, cols * width, 3), np.uint8)
    for position, tile in enumerate(tiles):
        if tile is not None:
            r, c = divmod(position, cols)
            sheet[r * height:(r + 1) * height, c * width:(c + 1) * width] = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", sheet, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cells = [{"cell": i, "frame_index": item.index, "status": item.status, "cached": item.cached,
              **({"stage": item.stage} if item.stage else {}), **({"error": item.error} if item.error else {})}
             for i, item in enumerate(outcomes)]
    return Response(png.tobytes(), media_type="image/png", headers={
        "Cache-Control": "no-store", "X-Preview-Config-Hash": digest, "X-Preview-Cells": json.dumps(cells, separators=(",", ":")),
        "X-Preview-Pending": ",".join(str(c["cell"]) for c in cells if c["status"] != "ready")})


@router.get("/target/preview_seq")
async def target_preview_seq(request: Request, config: str = Query(min_length=8, max_length=64),
                             start: int = Query(default=0, ge=0, le=10_000_000), count: int = Query(default=8, ge=1, le=16),
                             step: int = Query(default=1, ge=1, le=240),
                             budget_ms: int = Query(default=20000, ge=0, le=120000),
                             media_id: str | None = Query(default=None, max_length=64)):
    """Consecutive swapped frames as a JSON manifest; each ready frame is fetched from /target/preview by URL."""
    payload = _registered(request, config, media_id)
    service = _service(request)
    indices = [start + i * step for i in range(count)]

    async def work(cancel: Event):
        return await service.frames(payload.model_copy(update={"frame_index": 0, "timestamp": None}), indices, cancel,
                                    budget_s=budget_ms / 1000, strict=False)

    try:
        digest, outcomes = await _cancellable(request, work)
    except (PreviewCancelled, DomainError) as error:
        if isinstance(error, PreviewCancelled) or getattr(error, "code", None) == "PREVIEW_CANCELLED":
            return Response(status_code=204, headers={"X-Preview-Status": "cancelled"})
        raise
    if not any(item.status == "ready" for item in outcomes):
        busy = next((item for item in outcomes if item.status == "busy"), None)
        if busy:
            return busy_response(busy.stage or REBIND)
    frames = [{"frame_index": item.index, "status": item.status, "cached": item.cached,
               **({"stage": item.stage} if item.stage else {}), **({"error": item.error} if item.error else {}),
               **({"url": f"/api/target/preview?config={digest}&frame_index={item.index}"} if item.status == "ready" else {})}
              for item in outcomes]
    return JSONResponse({"media_id": payload.inspection_id, "swap_config_hash": digest, "frames": frames},
                        headers={"Cache-Control": "no-store"})
