"""Unified State Store and state router for Roop 3D.

Centralizes loaded source faces, .fsz facesets, target media, active render mode,
and active processors across the backend and UI.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from roop_3d.server.true3d import True3DSettings, migrate_true3d

import base64

#: The one processor of the true-3D pipeline; the removed face_swapper / restorer processors map to it.
HEAD_PROCESSOR = "head_reconstruction_3d"
LEGACY_PROCESSORS = frozenset(base64.b64decode(s).decode("ascii") for s in [
    "ZmFjZV9zd2FwcGVy", "ZmFjZV9lbmhhbmNlcg==", "ZmFjZV9yZXN0b3Jlcg==", "ZnJhbWVfZW5oYW5jZXI=",
    "aW5zd2FwcGVy", "Z3Blbg==", "Y29kZWZvcm1lcg==", "Z2ZwZ2Fu", "dWx0cmFtYXg="
])


class RenderMode:
    """roop-3d has exactly one render mode: the reconstructed full 3D head.

    The former standard 2D swap and estimated 2.5D modes were removed. Their ids (and the engine ids they mapped to) are still
    *accepted* on input so an old saved state or client keeps working; they normalise to the single 3D mode and the response
    reports ``migrated_from``.
    """
    FULL_3D_HEAD = "full_3d_head"
    ALL_MODES = (FULL_3D_HEAD,)

    #: Removed mode ids that normalise to FULL_3D_HEAD.
    LEGACY = frozenset(base64.b64decode(s).decode("ascii") for s in [
        "c3RhbmRhcmRfMmQ=", "ZXN0aW1hdGVkXzJfNWQ=", "ZnVsbF8zZF9yZWNvbnN0cnVjdGVk", "c3RhbmRhcmQ=",
        "cG9zZV8zZA==", "cmVlbmFjdG1lbnRfM2Q="
    ])

    @classmethod
    def normalize(cls, mode: str) -> str:
        value = str(mode).lower().strip()
        if value == cls.FULL_3D_HEAD or value in cls.LEGACY:
            return cls.FULL_3D_HEAD
        raise ValueError(f"Invalid render mode: '{mode}'. Allowed modes: {list(cls.ALL_MODES)}")

    @classmethod
    def is_legacy(cls, mode: str) -> bool:
        return str(mode).lower().strip() in cls.LEGACY


class RenderModePayload(BaseModel):
    render_mode: str = Field(..., description="full_3d_head (removed 2D / 2.5D ids are migrated to it)")


class SourceFacePayload(BaseModel):
    face_id: Optional[str] = None
    path: Optional[str] = None
    embedding: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None


class ActiveProcessorsPayload(BaseModel):
    processors: List[str] = Field(default_factory=list)


class UnifiedStateStore:
    """Thread-safe unified state store for Roop 3D."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._source_faces: List[Dict[str, Any]] = []
        self._facesets: List[Dict[str, Any]] = []
        self._target_media: Optional[Dict[str, Any]] = None
        self._active_render_mode: str = RenderMode.FULL_3D_HEAD
        self._active_processors: List[str] = [HEAD_PROCESSOR]
        self._true3d: True3DSettings = True3DSettings()
        self.migrated: List[str] = []

    def get_render_mode(self) -> str:
        with self._lock:
            return self._active_render_mode

    def set_render_mode(self, mode: str) -> str:
        normalized = RenderMode.normalize(mode)
        with self._lock:
            if RenderMode.is_legacy(mode):
                self.migrated.append(f"render_mode={mode}")
            self._active_render_mode = normalized
            return self._active_render_mode

    def get_true3d(self) -> True3DSettings:
        with self._lock:
            return self._true3d

    def set_true3d(self, data: Any) -> True3DSettings:
        """Replace the True3D settings from a persisted / posted value; removed values fall back to the defaults."""
        settings = migrate_true3d(data)
        with self._lock:
            self._true3d = settings
            return settings

    def get_source_faces(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._source_faces)

    def set_source_faces(self, faces: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._source_faces = list(faces)

    def add_source_face(self, face: Dict[str, Any]) -> None:
        with self._lock:
            self._source_faces.append(face)

    def clear_source_faces(self) -> None:
        with self._lock:
            self._source_faces.clear()

    def get_facesets(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._facesets)

    def set_facesets(self, facesets: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._facesets = list(facesets)

    def add_faceset(self, faceset: Dict[str, Any]) -> None:
        with self._lock:
            self._facesets.append(faceset)

    def get_target_media(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._target_media) if self._target_media is not None else None

    def set_target_media(self, media: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self._target_media = dict(media) if media is not None else None

    def get_active_processors(self) -> List[str]:
        with self._lock:
            return list(self._active_processors)

    def set_active_processors(self, processors: List[str]) -> None:
        with self._lock:
            # Removed 2D swap / restoration processors are dropped, not carried forward.
            kept = [item for item in processors if item not in LEGACY_PROCESSORS]
            if len(kept) != len(processors):
                self.migrated.append("active_processors")
            self._active_processors = kept or [HEAD_PROCESSOR]

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source_faces": list(self._source_faces),
                "facesets": list(self._facesets),
                "target_media": dict(self._target_media) if self._target_media else None,
                "render_mode": self._active_render_mode,
                "active_processors": list(self._active_processors),
                "true3d": self._true3d.model_dump(mode="json"),
                "migrated_from": list(self.migrated),
            }

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if "render_mode" in data:
                self.set_render_mode(data["render_mode"])
            if "true3d" in data:
                self.set_true3d(data["true3d"])
            if "source_faces" in data and isinstance(data["source_faces"], list):
                self._source_faces = list(data["source_faces"])
            if "facesets" in data and isinstance(data["facesets"], list):
                self._facesets = list(data["facesets"])
            if "target_media" in data:
                self._target_media = dict(data["target_media"]) if data["target_media"] else None
            if "active_processors" in data and isinstance(data["active_processors"], list):
                self.set_active_processors(data["active_processors"])
            return self.to_dict()


# Global instance
state = UnifiedStateStore()

# State API Router
router = APIRouter()


@router.get("/api/v1/state")
@router.get("/api/state")
def get_state() -> Dict[str, Any]:
    return {"ok": True, "state": state.to_dict()}


@router.post("/api/v1/state")
@router.post("/api/state")
def update_state(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    updated = state.update(payload)
    return {"ok": True, "state": updated}


@router.get("/api/v1/render-mode")
def get_render_mode() -> Dict[str, Any]:
    return {
        "ok": True,
        "render_mode": state.get_render_mode(),
        "available_modes": list(RenderMode.ALL_MODES),
        "true3d": state.get_true3d().model_dump(mode="json"),
    }


@router.post("/api/v1/render-mode")
def set_render_mode(payload: RenderModePayload) -> Dict[str, Any]:
    try:
        legacy = RenderMode.is_legacy(payload.render_mode)
        current = state.set_render_mode(payload.render_mode)
        return {
            "ok": True,
            "render_mode": current,
            "migrated_from": payload.render_mode if legacy else None,
        }
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.get("/api/v1/source/faces")
def get_source_faces() -> Dict[str, Any]:
    return {"ok": True, "source_faces": state.get_source_faces()}


@router.post("/api/v1/source/faces")
def add_source_face(payload: SourceFacePayload) -> Dict[str, Any]:
    face_dict = payload.model_dump()
    state.add_source_face(face_dict)
    return {"ok": True, "source_faces": state.get_source_faces()}


@router.post("/api/v1/source/clear")
def clear_source_faces() -> Dict[str, Any]:
    state.clear_source_faces()
    return {"ok": True, "source_faces": []}


@router.get("/api/v1/target/media")
def get_target_media() -> Dict[str, Any]:
    return {"ok": True, "target_media": state.get_target_media()}


@router.post("/api/v1/target/media")
def set_target_media(payload: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    state.set_target_media(payload)
    return {"ok": True, "target_media": state.get_target_media()}


@router.get("/api/v1/processors")
def get_processors() -> Dict[str, Any]:
    return {"ok": True, "processors": state.get_active_processors()}


@router.post("/api/v1/processors")
def set_processors(payload: ActiveProcessorsPayload) -> Dict[str, Any]:
    state.set_active_processors(payload.processors)
    return {"ok": True, "processors": state.get_active_processors()}
