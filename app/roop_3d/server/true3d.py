"""True-3D full-head controls for the FastAPI layer (the only render options roop-3d exposes).

roop-3d fits a complete MakeHuman head to the FaceSet and renders it through a perspective Z-buffer raster, shaded by a
per-frame spherical-harmonics irradiance. There is no 2D face swapper, no 2.5D strip mesh, no parallel-projection camera and no
screen-space face restorer behind this API.

Typed options (the same literals appear in ``roop_3d_ui/src/true3d.ts``). A literal is a protocol id; whether the connected
renderer implements it is reported by ``GET /true3d/options`` and enforced with a structured 422 (never substituted):

* ``head_topology_engine``  ``FLAME-FullHead-3D`` | ``MakeHuman-CC0-Craniofacial-3D``. Only the MakeHuman CC0 head is
  implemented (registry id ``makehuman_cc0``, availability-gated by the morphable-topology registry); FLAME is listed, disabled.
* ``shader_mode``           ``3D-PBR-Subsurface-Scattering + Neural-Deferred-UV`` | ``3D-Gaussian-Head-Avatar``. The first id is
  a legacy protocol spelling kept so saved presets keep loading: what it selects is the perspective spherical-harmonics
  Lambertian shader (no PBR material, no subsurface scattering, no neural deferred shading). The Gaussian avatar is listed,
  disabled.
* ``camera``                perspective only. ``auto`` is the renderer's assumed 35 mm-equivalent focal length with the
  principal point at the image centre (no per-shot solver runs); ``manual`` and a custom principal point are refused.

The render path itself lives behind the hook module ``roop_3d.pipeline.true3d_renderer`` (see ``RENDERER_MODULE``). Until that
module reports ``is_wired() is True`` every render entry point answers ``RENDERER_NOT_WIRED`` / "3D renderer not wired":
nothing here fabricates a ready engine.
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
#: Exact labels: the first id is a legacy protocol spelling, so its label says what it really selects.
SHADER_LABELS: dict[str, str] = {
    "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV": "Perspective SH shading (legacy protocol id; no PBR, no subsurface scattering)",
    "3D-Gaussian-Head-Avatar": "3D Gaussian head avatar (not implemented)",
}
SHADER_DESCRIPTIONS: dict[str, str] = {
    "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV":
        "Lambertian UV albedo lit by a per-frame second-order spherical-harmonics irradiance fitted to the visible target "
        "skin. The id keeps its historical spelling for saved presets; the shader has no PBR material model, no subsurface "
        "scattering, no specular lobe and no neural deferred shading.",
    "3D-Gaussian-Head-Avatar": "3D Gaussian head avatars are not implemented by the connected renderer.",
}

#: Where the real renderer plugs in. The hook module provides ``is_wired() -> bool``, ``describe() -> dict``,
#: ``blockers() -> list[str]``, ``validate_true3d(settings) -> dict``, ``render_preview(state, request, cancel, *, raw,
#: final_only) -> dict`` and ``open_live_session(state, config, true3d, admission)``.
RENDERER_MODULE = "roop_3d.pipeline.true3d_renderer"
NOT_WIRED_CODE = "RENDERER_NOT_WIRED"
NOT_WIRED_MESSAGE = "3D renderer not wired"
NOT_WIRED_DETAIL = ("The 2D/2.5D swap and restoration paths were removed. The true 3D full-head render path is not connected, so "
                    "previews, live studio and render jobs are unavailable until it is.")

FOV_RANGE_DEG = (5.0, 170.0)          # horizontal field of view
FOCAL_RANGE_PX = (50.0, 100_000.0)
AUTO_PRIOR_FOV_DEG = 54.4             # horizontal FOV of the assumed 35 mm-equivalent focal length (landscape frames)


class CameraIntrinsicsSpec(Contract):
    """Perspective pin-hole intrinsics. ``mode='auto'`` is the renderer's assumed 35 mm-equivalent focal length; ``manual`` names
    a fixed focal length, which the connected renderer refuses (``TRUE3D_CAMERA_MANUAL_UNSUPPORTED``) instead of ignoring."""
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


# --------------------------------------------------------------------------- the connected renderer
def _renderer_module():
    try:
        return importlib.import_module(RENDERER_MODULE)
    except ImportError:
        return None
    except Exception:
        return None


def _support(table: str) -> dict[str, dict[str, Any]]:
    """What the renderer module declares it implements (``TOPOLOGY_SUPPORT`` / ``SHADER_SUPPORT`` / ``CAMERA_SUPPORT``).
    Without a module there is no declaration, and ``wired`` is False anyway."""
    module = _renderer_module()
    value = getattr(module, table, None) if module is not None else None
    return value if isinstance(value, dict) else {}


def _camera_refusal(camera: CameraIntrinsicsSpec) -> DomainError | None:
    """The structured refusal the renderer would raise for this camera, or None when it honours it exactly."""
    validate = getattr(_renderer_module(), "validate_true3d", None)
    if not callable(validate):
        return None
    try:
        validate({"camera": camera})
    except DomainError as error:
        return error
    return None


# --------------------------------------------------------------------------- camera maths
def resolve_intrinsics(camera: CameraIntrinsicsSpec, width: int, height: int) -> dict[str, Any]:
    """The pin-hole matrix for a frame size. ``auto`` is exactly what the renderer uses (assumed 35 mm-equivalent focal length,
    principal point at the image centre); ``manual`` reports what the spec would define and says it is not honoured."""
    if width <= 0 or height <= 0:
        raise DomainError("FRAME_SIZE_INVALID", "width and height must be positive", 422)
    if camera.mode == "manual":
        if camera.focal_unit == "px":
            fx = float(camera.focal_length_px)
            fov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        else:
            fov = float(camera.fov_deg)
            fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        cx_norm, cy_norm = camera.principal_point or (0.5, 0.5)
        cx, cy, estimated = cx_norm * width, cy_norm * height, False
        note = "What this camera would define. The connected renderer does not honour a manual camera."
    else:
        from roop_3d.reconstruction.pose_solver import default_intrinsics_matrix
        matrix = default_intrinsics_matrix(width, height)
        fx, cx, cy = float(matrix[0, 0]), float(matrix[0, 2]), float(matrix[1, 2])
        fov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        estimated = True
        note = ("Assumed 35 mm-equivalent focal length (f = 35/36 * max(width, height) px) with the principal point at the "
                "image centre; no per-shot focal solver runs.")
    refusal = _camera_refusal(camera)
    return {"mode": camera.mode, "projection": "perspective", "width": width, "height": height,
            "fx": fx, "fy": fx, "cx": cx, "cy": cy, "horizontal_fov_deg": fov, "estimated": estimated,
            "honored": refusal is None, "unsupported": None if refusal is None else {"code": refusal.code, "reason": refusal.message},
            "note": note}


def camera_mode_reasons() -> dict[str, str | None]:
    """Per camera mode: None when the connected renderer honours it, otherwise why not (the convention of the option maps)."""
    support = _support("CAMERA_SUPPORT")
    return {mode: (None if support.get(mode, {"supported": True})["supported"] else support[mode]["reason"])
            for mode in ("auto", "manual")}


# --------------------------------------------------------------------------- availability / renderer readiness
def topology_options() -> list[dict[str, Any]]:
    """Both head-topology engines with honest availability: the morphable-topology registry says whether the asset is installed
    and hash-verified, the renderer says whether it implements the topology. Neither is ever assumed."""
    rows: dict[str, dict[str, Any]] = {}
    try:
        from roop_3d.reconstruction.topology import default_registry
        rows = {item["id"]: item for item in default_registry().describe()}
        failure = None
    except Exception as error:   # registry import / artifact scan failure: report it, do not guess availability
        failure = f"{type(error).__name__}: {error}"[:200]
    support = _support("TOPOLOGY_SUPPORT")
    options = []
    for option_id, registry_id in TOPOLOGY_REGISTRY_IDS.items():
        row = rows.get(registry_id)
        installed = bool(row and row.get("available"))
        implemented = support.get(option_id, {"supported": True})["supported"]
        reason = None
        if not implemented:
            reason = support[option_id].get("reason") or "Not implemented by the connected renderer"
        elif not installed:
            reason = (row or {}).get("detail") or failure or f"Topology '{registry_id}' is not registered"
        options.append({"id": option_id, "label": TOPOLOGY_LABELS[option_id], "registry_id": registry_id,
                        "available": implemented and installed, "implemented": implemented, "installed": installed,
                        "state": (row or {}).get("state", "unknown"), "reason": reason,
                        "license_id": (row or {}).get("license_id"), "attribution": (row or {}).get("attribution"),
                        "default": option_id == DEFAULT_HEAD_TOPOLOGY})
    return options


def shader_options(wired: bool) -> list[dict[str, Any]]:
    """Both shader modes. They run inside the 3D renderer, so none is available while it is not wired, and a mode the renderer
    does not implement is never available."""
    support = _support("SHADER_SUPPORT")
    options = []
    for option_id in SHADER_LABELS:
        info = support.get(option_id, {"supported": True, "implementation": None, "reason": None})
        if not info["supported"]:
            reason = info.get("reason") or "Not implemented by the connected renderer"
        else:
            reason = None if wired else f"{NOT_WIRED_MESSAGE}: the shader runs inside the 3D renderer"
        options.append({"id": option_id, "label": SHADER_LABELS[option_id], "description": SHADER_DESCRIPTIONS[option_id],
                        "available": bool(wired and info["supported"]), "implemented": bool(info["supported"]),
                        "implementation": info.get("implementation"), "reason": reason,
                        "default": option_id == DEFAULT_SHADER_MODE})
    return options


def renderer_status() -> dict[str, Any]:
    """Whether a true-3D render path is connected. Answers truthfully; never a fabricated 'ready'."""
    module = _renderer_module()
    wired = False
    detail: dict[str, Any] = {}
    blockers: list[str] = []
    if module is not None:
        try:
            wired = bool(module.is_wired())
        except Exception as error:
            blockers = [f"{type(error).__name__}: {error}"[:200]]
        if not wired and not blockers:
            try:
                blockers = list(module.blockers())
            except Exception:
                blockers = []
        describe = getattr(module, "describe", None)
        if wired and callable(describe):
            try:
                detail = dict(describe())
            except Exception:
                pass
    status: dict[str, Any] = {"wired": wired, "state": "ready" if wired else "not_wired", "module": RENDERER_MODULE,
                              "module_present": module is not None, **detail}
    if not wired:
        status.update(code=NOT_WIRED_CODE, message=NOT_WIRED_MESSAGE, blockers=blockers,
                      detail=NOT_WIRED_DETAIL + (f" Blockers: {'; '.join(blockers)}" if blockers else ""))
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


def validate_settings(settings: True3DSettings) -> dict[str, Any]:
    """The configuration the connected renderer will honour for ``settings``; a structured 422 for anything it does not
    implement (503 ``RENDERER_NOT_WIRED`` while no renderer is connected)."""
    return require_renderer().validate_true3d(settings)


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
            "camera": {"modes": ["auto", "manual"], "mode_reasons": camera_mode_reasons(), "projection": ["perspective"],
                       "focal_units": ["px", "fov_deg"], "fov_range_deg": list(FOV_RANGE_DEG),
                       "focal_range_px": list(FOCAL_RANGE_PX), "auto_prior_fov_deg": AUTO_PRIOR_FOV_DEG,
                       "auto_focal": "assumed 35 mm-equivalent: f = 35/36 * max(width, height) px, no per-shot solver",
                       "principal_point": "image centre only; a custom principal point is refused"},
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
