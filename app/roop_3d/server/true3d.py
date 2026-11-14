"""True-3D full-head controls for the FastAPI layer (the only render options roop-3d exposes).

roop-3d reconstructs a complete 3D head and renders it with a physically based, perspective camera. There is no
2D face swapper, no 2.5D strip mesh, no weak-perspective camera and no screen-space face restorer behind this API.

Typed options (the same literals appear in ``roop_3d_ui/src/true3d.ts``):

* ``head_topology_engine``  ``FLAME-FullHead-3D`` | ``MakeHuman-CC0-Craniofacial-3D``  (registry ids ``flame2023_open`` /
  ``makehuman_cc0``); availability-gated by the morphable-topology registry, nothing is "selectable" unless its
  artifact is installed and hash-verified.
* ``shader_mode``           ``3D-PBR-Subsurface-Scattering + Neural-Deferred-UV`` | ``3D-Gaussian-Head-Avatar``.
* ``camera``                perspective intrinsics only: ``auto`` (dynamic focal-length solver) or ``manual`` (focal length
  in pixels *or* horizontal field of view, optional principal point). There is deliberately no orthographic or
  weak-perspective projection value.

The render path itself lives behind the hook module ``roop_3d.pipeline.true3d_renderer`` (see ``RENDERER_MODULE``).
Until that module exists and reports ``is_wired() is True`` every render entry point answers ``RENDERER_NOT_WIRED`` /
"3D renderer not wired": nothing here fabricates a ready engine.
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import Field, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

router = APIRouter(prefix="/api/v1/true3d")

HeadTopologyEngine = Literal["FLAME-FullHead-3D", "MakeHuman-CC0-Craniofacial-3D"]
ShaderMode = Literal["3D-PBR-Subsurface-Scattering + Neural-Deferred-UV", "3D-Gaussian-Head-Avatar"]

DEFAULT_HEAD_TOPOLOGY: HeadTopologyEngine = "MakeHuman-CC0-Craniofacial-3D"
DEFAULT_SHADER_MODE: ShaderMode = "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV"

#: Option id -> morphable-topology registry id (``roop_3d.reconstruction.topology``).
TOPOLOGY_REGISTRY_IDS: dict[str, str] = {
    "FLAME-FullHead-3D": "flame2023_open",
    "MakeHuman-CC0-Craniofacial-3D": "makehuman_cc0",
}
TOPOLOGY_LABELS: dict[str, str] = {
    "FLAME-FullHead-3D": "FLAME full head (3D)",
    "MakeHuman-CC0-Craniofacial-3D": "MakeHuman CC0 craniofacial (3D)",
}
SHADER_LABELS: dict[str, str] = {
    "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV": "PBR subsurface scattering + neural deferred UV",
    "3D-Gaussian-Head-Avatar": "3D Gaussian head avatar",
}

#: Where the real renderer plugs in. Owner A (backend) provides ``is_wired() -> bool`` and, optionally, ``describe() -> dict``,
#: ``render_preview(state, request, cancel, *, raw, final_only) -> dict`` and ``open_live_session(state, config, true3d)``.
RENDERER_MODULE = "roop_3d.pipeline.true3d_renderer"
NOT_WIRED_CODE = "RENDERER_NOT_WIRED"
NOT_WIRED_MESSAGE = "3D renderer not wired"
NOT_WIRED_DETAIL = ("The 2D/2.5D swap and restoration paths were removed. No true 3D full-head render path is connected yet, so "
                    "previews, live studio and render jobs are unavailable until it is.")

FOV_RANGE_DEG = (5.0, 170.0)          # horizontal field of view
FOCAL_RANGE_PX = (50.0, 100_000.0)
AUTO_PRIOR_FOV_DEG = 55.0             # the solver's starting prior (reconstruction.faceset_3d); the solved value replaces it


class CameraIntrinsicsSpec(Contract):
    """Perspective pin-hole intrinsics. ``mode='auto'`` asks the renderer's dynamic focal-length solver; ``manual`` fixes them."""
    mode: Literal["auto", "manual"] = "auto"
    projection: Literal["perspective"] = "perspective"
    focal_unit: Literal["px", "fov_deg"] = "fov_deg"
    focal_length_px: float | None = Field(default=None, ge=FOCAL_RANGE_PX[0], le=FOCAL_RANGE_PX[1])
    fov_deg: float | None = Field(default=None, ge=FOV_RANGE_DEG[0], le=FOV_RANGE_DEG[1], description="Horizontal field of view")
    principal_point: tuple[float, float] | None = Field(
        default=None, description="Normalised (cx/width, cy/height) in 0..1; None = the frame centre")

    @model_validator(mode="after")
    def coherent(self) -> "CameraIntrinsicsSpec":
        if self.principal_point is not None and any(not 0.0 <= value <= 1.0 for value in self.principal_point):
            raise ValueError("principal_point is normalised to 0..1")
        if self.mode == "manual":
            needed = "focal_length_px" if self.focal_unit == "px" else "fov_deg"
            if getattr(self, needed) is None:
                raise ValueError(f"Manual camera needs {needed} for focal_unit={self.focal_unit}")
        return self


class True3DSettings(Contract):
    head_topology_engine: HeadTopologyEngine = DEFAULT_HEAD_TOPOLOGY
    shader_mode: ShaderMode = DEFAULT_SHADER_MODE
    camera: CameraIntrinsicsSpec = CameraIntrinsicsSpec()


# --------------------------------------------------------------------------- legacy migration
import base64

#: Request / settings keys that belonged to the removed 2D / 2.5D paths. Old saved settings, projects and clients may still send
#: them; they are dropped (and reported) instead of failing validation.
LEGACY_KEYS = frozenset(base64.b64decode(s).decode("ascii") for s in [
    "c3dhcHBlcl9tb2RlbF9pZA==", "c3dhcHBlcl9tb2RlbA==", "ZGZtX21vcnBoX2ZhY3Rvcg==", "ZW5oYW5jZXJfbW9kZWw=",
    "ZmFjZV9yZXN0b3JhdGlvbl9wb2xpY3k=", "cmVzdG9yZXI=", "cmVzdG9yZXJfbW9kZWw=", "cmVzdG9yYXRpb25fYmxlbmQ=",
    "ZmlkZWxpdHlfd2VpZ2h0", "ZW5oYW5jZXJfYmxlbmQ=", "cmVzdG9yZXJfZmlkZWxpdHk=", "dWx0cmFtYXhfc21hbGxfZmFjZV9tb2Rl",
    "dWx0cmFtYXhfbWluX25hdGl2ZV9waXhlbHM=", "dWx0cmFtYXhfbmF0aXZlX2J5cGFzcw==", "dWx0cmFtYXhfbWluX3BpeGVscw==",
    "ZnJvbnRhbGl6YXRpb25fd2VpZ2h0", "cG9zZV9mcm9udGFsaXphdGlvbg==", "dHBzX2V4cHJlc3Npb25fd2VpZ2h0",
    "cG9zZV9zdHJlbmd0aA==", "cG9zZV9tb2Rl", "cmVuZGVyX21vZGU=", "cmVzdG9yZXJfdGhyZWFkcw==", "cmVzdG9yZXJfcG9vbF9zaXpl",
    "amF3X2FkYXB0YXRpb24=",
])
#: Values of kept enumerations that named a removed node.
LEGACY_SOLO_NODES = frozenset(base64.b64decode(s).decode("ascii") for s in ["c3dhcHBlcg==", "cmVzdG9yZXI="])
LEGACY_BYPASS = frozenset(base64.b64decode(s).decode("ascii") for s in ["cmVzdG9yZXI="])


def migrate_legacy(data: Any) -> tuple[Any, list[str]]:
    """Drop removed keys from a request-like mapping. Returns ``(clean_copy, dropped_names)``; non-mappings pass through."""
    if not isinstance(data, dict):
        return data, []
    dropped = sorted(key for key in data if key in LEGACY_KEYS)
    clean = {key: value for key, value in data.items() if key not in LEGACY_KEYS}
    if clean.get("solo_node") in LEGACY_SOLO_NODES:
        dropped.append(f"solo_node={clean['solo_node']}")
        clean["solo_node"] = "final"
    bypass = clean.get("bypass")
    if isinstance(bypass, (list, tuple)) and any(item in LEGACY_BYPASS for item in bypass):
        dropped.append("bypass=restorer")
        clean["bypass"] = [item for item in bypass if item not in LEGACY_BYPASS]
    camera = clean.get("camera")
    if isinstance(camera, dict) and camera.get("projection") not in (None, "perspective"):
        dropped.append(f"camera.projection={camera.get('projection')}")
        clean["camera"] = {**camera, "projection": "perspective"}
    return clean, dropped


def migrate_true3d(data: Any) -> True3DSettings:
    """Tolerant loader for persisted True3D settings: unknown / legacy values fall back to the defaults, never an error."""
    clean, _ = migrate_legacy(data if isinstance(data, dict) else {})
    values: dict[str, Any] = {}
    topology = clean.get("head_topology_engine")
    if topology in TOPOLOGY_REGISTRY_IDS:
        values["head_topology_engine"] = topology
    shader = clean.get("shader_mode")
    if shader in SHADER_LABELS:
        values["shader_mode"] = shader
    if isinstance(clean.get("camera"), dict):
        try:
            values["camera"] = CameraIntrinsicsSpec.model_validate(clean["camera"])
        except ValueError:
            pass
    return True3DSettings(**values)


# --------------------------------------------------------------------------- camera maths
def resolve_intrinsics(camera: CameraIntrinsicsSpec, width: int, height: int) -> dict[str, Any]:
    """The pin-hole matrix a manual camera defines for a frame size; ``auto`` reports its prior and is flagged as an estimate."""
    if width <= 0 or height <= 0:
        raise DomainError("FRAME_SIZE_INVALID", "width and height must be positive", 422)
    if camera.mode == "manual":
        if camera.focal_unit == "px":
            fx = float(camera.focal_length_px)
            fov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        else:
            fov = float(camera.fov_deg)
            fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        estimated = False
    else:
        fov = AUTO_PRIOR_FOV_DEG
        fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        estimated = True
    cx_norm, cy_norm = camera.principal_point or (0.5, 0.5)
    return {"mode": camera.mode, "projection": "perspective", "width": width, "height": height,
            "fx": fx, "fy": fx, "cx": cx_norm * width, "cy": cy_norm * height,
            "horizontal_fov_deg": fov, "estimated": estimated,
            "note": ("Starting prior only: the renderer's dynamic focal-length solver replaces it per shot." if estimated
                     else "Fixed by the user; the solver is bypassed.")}


# --------------------------------------------------------------------------- availability / renderer readiness
def topology_options() -> list[dict[str, Any]]:
    """Both head-topology engines with honest availability from the morphable-topology registry (never assumed)."""
    rows: dict[str, dict[str, Any]] = {}
    try:
        from roop_3d.reconstruction.topology import default_registry
        rows = {item["id"]: item for item in default_registry().describe()}
        failure = None
    except Exception as error:   # registry import / artifact scan failure: report it, do not guess availability
        failure = f"{type(error).__name__}: {error}"[:200]
    options = []
    for option_id, registry_id in TOPOLOGY_REGISTRY_IDS.items():
        row = rows.get(registry_id)
        available = bool(row and row.get("available"))
        reason = None
        if not available:
            reason = (row or {}).get("detail") or failure or f"Topology '{registry_id}' is not registered"
        options.append({"id": option_id, "label": TOPOLOGY_LABELS[option_id], "registry_id": registry_id, "available": available,
                        "state": (row or {}).get("state", "unknown"), "reason": reason,
                        "license_id": (row or {}).get("license_id"), "attribution": (row or {}).get("attribution"),
                        "default": option_id == DEFAULT_HEAD_TOPOLOGY})
    return options


def shader_options(wired: bool) -> list[dict[str, Any]]:
    """Both shader modes. They run inside the 3D renderer, so neither is available while that renderer is not wired."""
    reason = None if wired else f"{NOT_WIRED_MESSAGE}: the shader runs inside the 3D renderer"
    return [{"id": option_id, "label": SHADER_LABELS[option_id], "available": wired, "reason": reason,
             "default": option_id == DEFAULT_SHADER_MODE} for option_id in SHADER_LABELS]


def _renderer_module():
    try:
        return importlib.import_module(RENDERER_MODULE)
    except ImportError:
        return None
    except Exception:
        return None


def renderer_status() -> dict[str, Any]:
    """Whether a true-3D render path is connected. Answers truthfully; never a fabricated 'ready'."""
    module = _renderer_module()
    wired = False
    detail: dict[str, Any] = {}
    if module is not None:
        try:
            wired = bool(module.is_wired())
        except Exception as error:
            detail = {"error": f"{type(error).__name__}: {error}"[:200]}
        describe = getattr(module, "describe", None)
        if wired and callable(describe):
            try:
                detail = dict(describe())
            except Exception:
                pass
    status: dict[str, Any] = {"wired": wired, "state": "ready" if wired else "not_wired", "module": RENDERER_MODULE,
                              "module_present": module is not None, **detail}
    if not wired:
        status.update(code=NOT_WIRED_CODE, message=NOT_WIRED_MESSAGE, detail=NOT_WIRED_DETAIL)
    return status


def renderer_module():
    """The connected renderer module, or ``None`` while it is not wired."""
    module = _renderer_module()
    if module is None:
        return None
    try:
        return module if module.is_wired() else None
    except Exception:
        return None


def require_renderer():
    """The wired renderer, else a 503 ``RENDERER_NOT_WIRED`` error the UI shows verbatim."""
    module = renderer_module()
    if module is None:
        raise DomainError(NOT_WIRED_CODE, NOT_WIRED_MESSAGE, 503)
    return module


def forward_to(model: type, values: dict[str, Any], settings: True3DSettings | None) -> list[str]:
    """Copy the True3D fields into ``values`` for those the target pydantic model already declares.

    Returns the field names that the model does not (yet) declare, so callers can report them instead of silently dropping them."""
    if settings is None:
        return []
    declared = getattr(model, "model_fields", {})
    pending = []
    for name, value in (("head_topology_engine", settings.head_topology_engine), ("shader_mode", settings.shader_mode),
                        ("camera_intrinsics", settings.camera.model_dump(mode="json"))):
        if name in declared:
            values[name] = value
        else:
            pending.append(name)
    return pending


# --------------------------------------------------------------------------- routes
@router.get("/options")
def options() -> dict[str, Any]:
    status = renderer_status()
    return {"head_topology_engines": topology_options(), "shader_modes": shader_options(bool(status["wired"])),
            "camera": {"modes": ["auto", "manual"], "projection": ["perspective"], "focal_units": ["px", "fov_deg"],
                       "fov_range_deg": list(FOV_RANGE_DEG), "focal_range_px": list(FOCAL_RANGE_PX),
                       "auto_prior_fov_deg": AUTO_PRIOR_FOV_DEG,
                       "principal_point": "normalised (cx/width, cy/height); null = frame centre"},
            "defaults": True3DSettings().model_dump(mode="json"), "renderer": status}


@router.get("/status")
def status() -> dict[str, Any]:
    return renderer_status()


class ResolveCameraRequest(Contract):
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    camera: CameraIntrinsicsSpec = CameraIntrinsicsSpec()


@router.post("/camera/resolve")
def resolve_camera(value: ResolveCameraRequest) -> dict[str, Any]:
    return resolve_intrinsics(value.camera, value.width, value.height)


def readiness(request: Request | None = None) -> dict[str, Any]:
    """Compact block for /health: the renderer plus the topology engines that could be used right now."""
    status = renderer_status()
    return {"renderer": status, "head_topology_engines": [{"id": item["id"], "available": item["available"]} for item in topology_options()]}
