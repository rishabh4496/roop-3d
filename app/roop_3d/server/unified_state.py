"""Unified state store and router for sources, target media, and true-3D settings."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict

from roop_3d.server.true3d import True3DSettings


class SourceFacePayload(BaseModel):
    face_id: Optional[str] = None
    path: Optional[str] = None
    embedding: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None


class StateUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true3d: True3DSettings | None = None
    source_faces: List[Dict[str, Any]] | None = None
    facesets: List[Dict[str, Any]] | None = None
    target_media: Dict[str, Any] | None = None



class UnifiedStateStore:
    """Thread-safe unified state store for Roop 3D."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._source_faces: List[Dict[str, Any]] = []
        self._facesets: List[Dict[str, Any]] = []
        self._target_media: Optional[Dict[str, Any]] = None
        self._true3d = True3DSettings()


    def get_true3d(self) -> True3DSettings:
        with self._lock:
            return self._true3d

    def set_true3d(self, data: Any) -> True3DSettings:
        """Validate and replace the true-3D settings; unsupported fields fail validation."""
        settings = True3DSettings.model_validate(data)
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

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source_faces": list(self._source_faces),
                "facesets": list(self._facesets),
                "target_media": dict(self._target_media) if self._target_media else None,
                "true3d": self._true3d.model_dump(mode="json"),
            }

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if "true3d" in data:
                self.set_true3d(data["true3d"])
            if "source_faces" in data and isinstance(data["source_faces"], list):
                self._source_faces = list(data["source_faces"])
            if "facesets" in data and isinstance(data["facesets"], list):
                self._facesets = list(data["facesets"])
            if "target_media" in data:
                self._target_media = dict(data["target_media"]) if data["target_media"] else None
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
def update_state(payload: StateUpdatePayload) -> Dict[str, Any]:
    updated = state.update(payload.model_dump(exclude_unset=True))
    return {"ok": True, "state": updated}




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
