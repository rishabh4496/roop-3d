"""Unified State Store and state router for Roop 3D.

Centralizes loaded source faces, .fsz facesets, target media, active render mode,
and active processors across the backend and UI.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field


class RenderMode:
    STANDARD_2D = "standard_2d"
    ESTIMATED_2_5D = "estimated_2_5d"
    FULL_3D_RECONSTRUCTED = "full_3d_reconstructed"

    ALL_MODES = (STANDARD_2D, ESTIMATED_2_5D, FULL_3D_RECONSTRUCTED)

    # Bi-directional mapping between UI render modes and internal engine modes
    TO_ENGINE = {
        STANDARD_2D: "standard",
        ESTIMATED_2_5D: "pose_3d",
        FULL_3D_RECONSTRUCTED: "reenactment_3d",
        "standard": "standard",
        "pose_3d": "pose_3d",
        "reenactment_3d": "reenactment_3d",
    }

    FROM_ENGINE = {
        "standard": STANDARD_2D,
        "pose_3d": ESTIMATED_2_5D,
        "reenactment_3d": FULL_3D_RECONSTRUCTED,
        STANDARD_2D: STANDARD_2D,
        ESTIMATED_2_5D: ESTIMATED_2_5D,
        FULL_3D_RECONSTRUCTED: FULL_3D_RECONSTRUCTED,
    }

    @classmethod
    def normalize(cls, mode: str) -> str:
        canonical = cls.FROM_ENGINE.get(mode.lower().strip())
        if canonical:
            return canonical
        raise ValueError(
            f"Invalid render mode: '{mode}'. Allowed modes: {list(cls.ALL_MODES)}"
        )

    @classmethod
    def to_engine_mode(cls, mode: str) -> str:
        return cls.TO_ENGINE.get(mode, "standard")


class RenderModePayload(BaseModel):
    render_mode: str = Field(..., description="One of: standard_2d, estimated_2_5d, full_3d_reconstructed")


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
        self._active_render_mode: str = RenderMode.STANDARD_2D
        self._active_processors: List[str] = ["face_swapper"]

    def get_render_mode(self) -> str:
        with self._lock:
            return self._active_render_mode

    def get_engine_render_mode(self) -> str:
        with self._lock:
            return RenderMode.to_engine_mode(self._active_render_mode)

    def set_render_mode(self, mode: str) -> str:
        normalized = RenderMode.normalize(mode)
        with self._lock:
            self._active_render_mode = normalized
            return self._active_render_mode

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
            self._active_processors = list(processors)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source_faces": list(self._source_faces),
                "facesets": list(self._facesets),
                "target_media": dict(self._target_media) if self._target_media else None,
                "render_mode": self._active_render_mode,
                "engine_render_mode": RenderMode.to_engine_mode(self._active_render_mode),
                "active_processors": list(self._active_processors),
            }

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if "render_mode" in data:
                self.set_render_mode(data["render_mode"])
            if "source_faces" in data and isinstance(data["source_faces"], list):
                self._source_faces = list(data["source_faces"])
            if "facesets" in data and isinstance(data["facesets"], list):
                self._facesets = list(data["facesets"])
            if "target_media" in data:
                self._target_media = dict(data["target_media"]) if data["target_media"] else None
            if "active_processors" in data and isinstance(data["active_processors"], list):
                self._active_processors = list(data["active_processors"])
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
        "engine_mode": state.get_engine_render_mode(),
        "available_modes": list(RenderMode.ALL_MODES),
    }


@router.post("/api/v1/render-mode")
def set_render_mode(payload: RenderModePayload) -> Dict[str, Any]:
    try:
        current = state.set_render_mode(payload.render_mode)
        return {
            "ok": True,
            "render_mode": current,
            "engine_mode": state.get_engine_render_mode(),
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
