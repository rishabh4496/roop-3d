"""Hook module of the true-3D full-head render path (``roop_3d.server.true3d`` and ``roop_3d.core.true3d_guard`` look it up).

Exactly what is connected here, and what is not (``describe()`` reports the same facts):

* Geometry: the MakeHuman CC0 head-and-neck template (4289 base vertices, UV-seam-split render vertices, fixed triangle
  topology). One shared linear shape identity is fitted to the FaceSet's trusted real views from their landmarks; whatever the
  landmarks do not constrain stays the template prior. Five expression axes are fitted per frame from dense-68 landmark
  residuals. There is no FLAME topology and no scalp-hair / eyeball geometry.
* Pose and camera: 6-DoF PnP on the target's measured dense-68 landmarks (no 5-point fallback), smoothed per track. The camera
  is a perspective pin-hole with an assumed 35 mm-equivalent focal length and the principal point at the image centre. No
  focal-length solver runs, so a manual camera is refused instead of being ignored.
* Shader: a perspective-correct Z-buffer raster of the UV albedo atlas, lit by a second-order spherical-harmonics irradiance
  (9 coefficients per colour channel) fitted per frame to the visible target skin. It is Lambertian: no PBR material, no
  subsurface scattering, no specular lobe, no cast shadows, no 3D Gaussian avatar, no neural deferred shading.

Entry points (all of them run the same ``ReferencePipeline`` path as a render job)::

    is_wired() / blockers() / describe()                         readiness and exact capability
    validate_true3d(settings) / validate_job_request(request)    structured refusal of anything not implemented
    render_frame(session, image, face, *, timestamp, scene_cut)  one tracked face -> ``Replacement`` (called by infer_frame)
    render_preview(state, request, cancel, *, raw, final_only)   one bounded workstation preview
    open_live_session(state, config, true3d, admission)          the camera studio's session
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
import dataclasses
from fractions import Fraction
import math
import threading
import time
from typing import Any
import uuid

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.true3d_guard import FRAME_PIPELINE_STAGES

RENDERER_ID = "volumetric_makehuman_perspective_sh"
MAKEHUMAN_ENGINE = "MakeHuman-CC0-Craniofacial-3D"
FLAME_ENGINE = "FLAME-FullHead-3D"
#: Wire id of the one implemented shader. It keeps its historical spelling so saved presets and the UI enum keep loading; it
#: names no PBR, subsurface-scattering or neural-deferred behaviour (see ``describe()["shader"]["claims"]``).
LEGACY_SHADER_ID = "3D-PBR-Subsurface-Scattering + Neural-Deferred-UV"
GAUSSIAN_SHADER_ID = "3D-Gaussian-Head-Avatar"
SHADER_IMPLEMENTATION = "perspective_sh_lambertian"
DEFAULT_RESTORATION_WEIGHT = 0.35
DEFAULT_TEXTURE_SIZE = 1024

#: Per-frame fit failures of the volumetric renderer: this frame's face cannot be rendered, the next one may be. Previews and the
#: live studio skip these (with a diagnostic); the job adapter's own ``FIT_SKIP_CODES`` is unioned in at use.
FRAME_FIT_CODES = frozenset({
    "DENSE_HEAD_LANDMARKS_REQUIRED", "HEAD_DENSE_LANDMARKS_INVALID", "HEAD_106_LANDMARKS_INVALID",
    "HEAD_5POINT_LANDMARKS_INVALID", "TARGET_LANDMARKS_UNTRUSTED", "HEAD_FACE_BOX_INVALID", "HEAD_FACE_BOUNDS_MISSING",
    "HEAD_INTERSECTS_CAMERA", "HEAD_PROJECTION_INVALID", "STAGE4_LIGHTING_EVIDENCE_INSUFFICIENT",
    "STAGE4_LIGHTING_FIT_FAILED", "STAGE4_LIGHTING_FIT_INVALID"})

TOPOLOGY_SUPPORT: dict[str, dict[str, Any]] = {
    MAKEHUMAN_ENGINE: {"supported": True, "implementation": "makehuman_cc0_head_neck_template", "reason": None},
    FLAME_ENGINE: {"supported": False, "implementation": None,
                   "reason": "FLAME is not implemented by the volumetric renderer: it fits and renders the MakeHuman CC0 head only"},
}
SHADER_SUPPORT: dict[str, dict[str, Any]] = {
    LEGACY_SHADER_ID: {"supported": True, "implementation": SHADER_IMPLEMENTATION, "reason": None},
    GAUSSIAN_SHADER_ID: {"supported": False, "implementation": None,
                         "reason": "3D Gaussian head avatars are not implemented: the renderer shades a UV-albedo mesh "
                                   "with fitted spherical harmonics"},
}
CAMERA_SUPPORT: dict[str, dict[str, Any]] = {
    "auto": {"supported": True, "reason": None},
    "manual": {"supported": False,
               "reason": "Manual camera intrinsics are not supported: the renderer assumes a 35 mm-equivalent focal length "
                         "with the principal point at the image centre"},
}

REPLACEMENT_FIELDS = frozenset({"track_id", "rendered_frame", "coverage", "skin_mask", "hair_mask", "normals",
                                "pipeline_stages", "full_head", "occluder"})
HEAD_LAYER_FIELDS = frozenset({"image", "coverage", "normals", "diagnostics"})


# --------------------------------------------------------------------------- readiness
_PROBE_RETRY_S = 30.0
_PROBE_LOCK = threading.Lock()
_probe_result: tuple[bool, list[str], float] | None = None


def _probe() -> tuple[bool, list[str]]:
    """Import every implementation module the hook calls and check the contracts between them (no weights are loaded)."""
    try:
        from roop_3d.engine.volumetric.session import Replacement
        from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
        from roop_3d.pipeline.session_adapter import SessionFrameProcessor
        from roop_3d.reconstruction.stage4_head import HeadLayer, VolumetricHeadRenderer
    except Exception as error:   # a missing dependency or a broken module: the hook cannot run, so say why
        return False, [f"import failed: {type(error).__name__}: {error}"[:240]]
    problems: list[str] = []
    absent = sorted(REPLACEMENT_FIELDS - {field.name for field in dataclasses.fields(Replacement)})
    if absent:
        problems.append(f"Replacement lacks {absent}")
    absent = sorted(HEAD_LAYER_FIELDS - {field.name for field in dataclasses.fields(HeadLayer)})
    if absent:
        problems.append(f"HeadLayer lacks {absent}")
    for name in ("render", "release"):
        if not callable(getattr(VolumetricHeadRenderer, name, None)):
            problems.append(f"VolumetricHeadRenderer.{name} is missing")
    for owner, name in ((Photorealistic3DSwapCore, "process_face"), (SessionFrameProcessor, "process_result")):
        if not callable(getattr(owner, name, None)):
            problems.append(f"{owner.__name__}.{name} is missing")
    return not problems, problems


def _status() -> tuple[bool, list[str]]:
    """A success is cached for the process; a failure is probed again after ``_PROBE_RETRY_S`` (a repaired install heals)."""
    global _probe_result
    with _PROBE_LOCK:
        cached = _probe_result
        if cached is not None and (cached[0] or time.monotonic() - cached[2] < _PROBE_RETRY_S):
            return cached[0], list(cached[1])
        wired, problems = _probe()
        _probe_result = (wired, problems, time.monotonic())
        return wired, list(problems)


def is_wired() -> bool:
    """True when the whole render path imports and its contracts hold. It does not claim that weights or assets are installed."""
    return _status()[0]


def blockers() -> list[str]:
    """Why the hook is not wired (empty while it is)."""
    return _status()[1]


class _Stats:
    """Process-wide render counters: measured evidence for readiness, never an assumption."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames_rendered = 0
        self.frames_failed = 0
        self.engines_built = 0
        self.last_render_ms: float | None = None
        self.last_error: dict[str, Any] | None = None

    def engine_built(self) -> None:
        with self._lock:
            self.engines_built += 1

    def rendered(self, milliseconds: float) -> None:
        with self._lock:
            self.frames_rendered += 1
            self.last_render_ms = round(milliseconds, 3)

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self.frames_failed += 1
            self.last_error = {"code": getattr(error, "code", type(error).__name__), "message": str(error)[:240],
                               "at": time.time()}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"frames_rendered": self.frames_rendered, "frames_failed": self.frames_failed,
                    "engines_built": self.engines_built, "last_render_ms": self.last_render_ms,
                    "last_error": dict(self.last_error) if self.last_error else None}


_STATS = _Stats()


def describe() -> dict[str, Any]:
    """The exact capability of the connected renderer. Cheap and static apart from the measured ``runtime`` counters."""
    wired, problems = _status()
    return {
        "renderer": RENDERER_ID, "wired": wired, "blockers": problems,
        "summary": "MakeHuman CC0 head, perspective Z-buffer raster, per-frame spherical-harmonics shading",
        "geometry": {
            "topology_engine": MAKEHUMAN_ENGINE, "registry_id": "makehuman_cc0",
            "mesh": "MakeHuman CC0 head-and-neck template: 4289 base vertices, UV-seam-split render vertices, fixed triangles",
            "identity": "one shared linear shape fitted to the FaceSet's trusted real views from their landmarks; whatever "
                        "the landmarks do not constrain stays the template prior",
            "expression": "five MakeHuman axes (mouth open, mouth-corner pull, pursing, brows up, eyes closed) fitted per "
                          "frame from dense-68 landmark residuals; unchanged when the fit does not explain them",
            "not_modelled": ["scalp hair", "eyeballs, teeth and tongue"]},
        "pose": {
            "method": "6-DoF PnP on the target's measured dense-68 landmarks against the fitted MakeHuman landmark vertices",
            "fallback": "none: a frame without a consistent dense-68 pose is refused (DENSE_HEAD_LANDMARKS_REQUIRED)",
            "smoothing": "per-track One-Euro filters on rotation and translation"},
        "camera": {
            "projection": "perspective", "model": "pin-hole K[R|t], square pixels, no lens distortion",
            "auto": "assumed 35 mm-equivalent focal length f = 35/36 * max(width, height) px, principal point at the image "
                    "centre; no per-shot focal solver runs",
            "manual": CAMERA_SUPPORT["manual"]["reason"]},
        "rasterizer": {"method": "CPU triangle Z-buffer with perspective-correct UV interpolation and back-face rejection",
                       "silhouette_feather_px": 1.5},
        "shader": {
            "id": SHADER_IMPLEMENTATION, "wire_id": LEGACY_SHADER_ID,
            "model": "Lambertian UV albedo lit by a per-frame second-order spherical-harmonics irradiance (9 coefficients "
                     "per colour channel) fitted by robust least squares to the visible target skin",
            "evidence": "needs at least 32 visible skin samples overlapping the head, otherwise the frame is refused "
                        "(STAGE4_LIGHTING_EVIDENCE_INSUFFICIENT)",
            "claims": {"pbr": False, "subsurface_scattering": False, "neural_deferred_uv": False,
                       "gaussian_splatting": False, "specular": False, "cast_shadows": False}},
        "texture": {
            "atlas": f"MakeHuman template UV layout, square, {DEFAULT_TEXTURE_SIZE} px by default (64..4096)",
            "synthesis": "visibility-weighted projection of the real source views through their fitted cameras; unseen "
                         "texels are symmetry or harmonic completion and are reported as such, never as observations",
            "restoration": "skin-only neural detail restoration in canonical UV space before projection; weight "
                           f"restoration_weight (default {DEFAULT_RESTORATION_WEIGHT}, 0 disables); no target-frame crop "
                           "is ever restored"},
        "semantic_masks": "face parser, XSeg visible-face network and occluder network over the head raster's region",
        "pipeline_stages": list(FRAME_PIPELINE_STAGES[:3]),
        "support": {"head_topology_engines": {key: dict(value) for key, value in TOPOLOGY_SUPPORT.items()},
                    "shader_modes": {key: dict(value) for key, value in SHADER_SUPPORT.items()},
                    "camera_modes": {key: dict(value) for key, value in CAMERA_SUPPORT.items()}},
        "components": ["dense_landmarks_3d", "dense_landmarks_2d", "head_reconstructor", "uv_synthesis", "face_parser",
                       "occlusion_mask", "uv_enhancer"],
        "not_implemented": ["FLAME topology", "3D Gaussian head avatar", "PBR and subsurface-scattering shading",
                            "neural deferred UV shading", "specular reflection and cast shadows",
                            "manual camera intrinsics and focal-length solving",
                            "manual lighting controls (relight, specular sheen, manual key light)",
                            "render-pass export (alpha, occluder, uv, normal, depth, relit)", "identity blends",
                            "force-include mask strokes"],
        "limitations": [
            "Head, ear, scalp and neck geometry outside the landmark constraints is the MakeHuman template prior.",
            "Texture without direct source visibility is symmetry or harmonic completion, not observation.",
            "There is no scalp-hair geometry; semantic masks exclude hair, glasses, neck and occluders.",
            "Dense-68 correspondences are template-derived; inconsistent dense poses are rejected."],
        "runtime": _STATS.snapshot()}


# --------------------------------------------------------------------------- refusing what is not implemented
class UnsupportedTrue3DOption(DomainError):
    """A valid request names something the connected renderer does not implement: it is refused, never substituted."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(code, message, 422)
        self.details = dict(details or {})


def _field(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def validate_true3d(settings: Any) -> dict[str, Any]:
    """Refuse a topology, shader or camera this renderer does not implement; return the configuration that will be honoured.

    ``settings`` is a ``True3DSettings`` (or any object / mapping with ``head_topology_engine``, ``shader_mode`` and ``camera``);
    a missing value means the renderer's own default. Nothing is ever substituted for an unsupported choice."""
    topology = _field(settings, "head_topology_engine") or MAKEHUMAN_ENGINE
    shader = _field(settings, "shader_mode") or LEGACY_SHADER_ID
    camera = _field(settings, "camera")
    support = TOPOLOGY_SUPPORT.get(topology)
    if support is None or not support["supported"]:
        raise UnsupportedTrue3DOption(
            "TRUE3D_TOPOLOGY_UNSUPPORTED", (support or {}).get("reason") or f"Unknown head topology engine '{topology}'",
            {"field": "head_topology_engine", "value": topology, "supported": [MAKEHUMAN_ENGINE]})
    support = SHADER_SUPPORT.get(shader)
    if support is None or not support["supported"]:
        raise UnsupportedTrue3DOption(
            "TRUE3D_SHADER_UNSUPPORTED", (support or {}).get("reason") or f"Unknown shader mode '{shader}'",
            {"field": "shader_mode", "value": shader, "supported": [LEGACY_SHADER_ID]})
    mode = _field(camera, "mode") or "auto"
    if mode != "auto":
        raise UnsupportedTrue3DOption(
            "TRUE3D_CAMERA_MANUAL_UNSUPPORTED", CAMERA_SUPPORT.get(mode, CAMERA_SUPPORT["manual"])["reason"],
            {"field": "camera.mode", "value": mode, "supported": ["auto"]})
    if (_field(camera, "projection") or "perspective") != "perspective":
        raise UnsupportedTrue3DOption("TRUE3D_CAMERA_PROJECTION_UNSUPPORTED", "Only a perspective camera is supported",
                                      {"field": "camera.projection", "supported": ["perspective"]})
    principal = _field(camera, "principal_point")
    if principal is not None and any(abs(float(value) - 0.5) > 1e-9 for value in principal):
        raise UnsupportedTrue3DOption(
            "TRUE3D_CAMERA_PRINCIPAL_POINT_UNSUPPORTED",
            "A custom principal point is not supported: the renderer's principal point is the image centre",
            {"field": "camera.principal_point", "value": list(principal), "supported": [0.5, 0.5]})
    return {"head_topology_engine": topology, "shader_mode": shader, "shader_implementation": SHADER_IMPLEMENTATION,
            "camera": {"mode": "auto", "projection": "perspective", "intrinsics": "assumed_35mm_equivalent",
                       "principal_point": "image_centre"}}


def validate_job_request(request: Any) -> None:
    """Refuse job features the volumetric renderer cannot honour: identity blends, render passes, unsupported 3D options."""
    if _field(request, "identity_blend") is not None:
        raise UnsupportedTrue3DOption(
            "TRUE3D_IDENTITY_BLEND_UNSUPPORTED",
            "Identity blends are not supported: the MakeHuman head is fitted from FaceSet views, which an embedding blend "
            "does not change. Select a single FaceSet", {"field": "identity_blend"})
    export = _field(request, "export")
    if export is not None and _field(export, "passes"):
        raise UnsupportedTrue3DOption(
            "TRUE3D_RENDER_PASSES_UNAVAILABLE",
            "Render-pass export (alpha, occluder, uv, normal, depth, relit) is not implemented for the volumetric MakeHuman "
            "renderer", {"field": "export.passes"})
    parameters = _field(request, "workstation_parameters")
    if parameters is not None:
        validate_true3d({"head_topology_engine": _field(parameters, "head_topology_engine"),
                         "shader_mode": _field(parameters, "shader_mode"),
                         "camera": _field(parameters, "camera_intrinsics")})


def frame_skip_codes() -> frozenset[str]:
    """Error codes that mean "this frame's face cannot be rendered" (skipped with a diagnostic, never fatal for a preview
    warm-up frame or a live frame): the job adapter's fit codes plus this renderer's own per-frame ones."""
    from roop_3d.pipeline.session_adapter import SessionFrameProcessor
    return frozenset(SessionFrameProcessor.FIT_SKIP_CODES) | FRAME_FIT_CODES


# --------------------------------------------------------------------------- one frame, one face
def _track_id(face: Any) -> int:
    value = _field(face, "track_id")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise DomainError("HEAD_TRACK_ID_INVALID", "A rendered face needs a nonnegative integer track id", 422)
    return int(value)


def _engine_for(session: Any, track_id: int, document: Any):
    """The track's head engine, built lazily from the track's own FaceSet and rebuilt whenever that source, the restoration
    weight or the atlas size no longer match what it was fitted with."""
    from roop_3d.reconstruction.stage4_head import VolumetricHeadRenderer
    engines = session.full_head_engines
    weight = float(getattr(session, "restoration_weight", DEFAULT_RESTORATION_WEIGHT))
    size = int(getattr(session, "texture_size", DEFAULT_TEXTURE_SIZE))
    engine = engines.get(track_id)
    if engine is not None and (getattr(engine, "source_document", engine.document) is not document or engine.restoration_weight != weight
                               or engine.texture_size != size):
        del engines[track_id]
        engine.release()
        engine = None
    if engine is None:
        source = document
        if sum(view.landmarks_68 is not None and view.landmarks_trusted for view in document.views) < 2:
            from roop_3d.faceset_loader import LandmarkEngine, enrich_document
            from roop_3d.engine.runtime import session_factory
            landmarker = LandmarkEngine(session.catalog.root, catalog=session.catalog,
                                       session_factory=session_factory(session.catalog, session.config))
            source = enrich_document(document, landmarker).document
        engine = VolumetricHeadRenderer(session.catalog, session.config, source, texture_size=size,
                                        restoration_weight=weight)
        engine.source_document = document
        engines[track_id] = engine
        _STATS.engine_built()
    return engine


def _replacement(track_id: int, layer: Any, frame: np.ndarray):
    """The head layer as a full-frame ``Replacement``: raster, coverage and normals from the mesh; the semantic skin / hair
    planes are neutral (all skin, no hair) because the compositor's parser decides those after the 3D render."""
    from roop_3d.engine.volumetric.session import Replacement
    shape = frame.shape[:2]
    image, coverage, normals = layer.image, layer.coverage, layer.normals
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != frame.shape
            or not isinstance(coverage, np.ndarray) or coverage.shape != shape
            or not isinstance(normals, np.ndarray) or normals.shape != (*shape, 3)):
        raise DomainError("TRUE3D_RENDERER_CONTRACT",
                          "The head renderer returned buffers that do not match the frame (uint8 BGR raster, HxW coverage, "
                          "HxWx3 normals expected)", 500)
    return Replacement(track_id=track_id, rendered_frame=image, coverage=coverage,
                       skin_mask=np.ones(shape, np.float32), hair_mask=np.zeros(shape, np.float32), normals=normals,
                       pipeline_stages=FRAME_PIPELINE_STAGES[:3], full_head=True, occluder=None)


def render_frame(session: Any, image: np.ndarray, face: Any, *, timestamp: float, scene_cut: bool = False):
    """Render one tracked face as a complete perspective head layer (``Replacement``, stages 1-3 of the frame pipeline).

    The head engine of the track is built lazily from ``session._document_for(face.track_id)`` with the session's
    ``restoration_weight`` / ``texture_size`` and cached in ``session.full_head_engines`` (the session releases it when the track
    ends). Errors keep their structured codes; nothing is substituted for a frame that cannot be fitted."""
    track_id = _track_id(face)
    document = session._document_for(track_id)
    if document is None:
        raise DomainError("TRUE3D_SOURCE_MISSING", f"Track {track_id} has no FaceSet source to build a head from", 409)
    began = time.perf_counter()
    try:
        engine = _engine_for(session, track_id, document)
        replacement = _replacement(track_id, engine.render(image, face, timestamp, scene_cut=bool(scene_cut)), image)
    except BaseException as error:
        _STATS.failed(error)
        raise
    _STATS.rendered((time.perf_counter() - began) * 1000.0)
    return replacement


# --------------------------------------------------------------------------- workstation preview
def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _layer_summary(layer: Any) -> dict[str, Any]:
    """Measured facts of one rendered head layer, as plain JSON."""
    diagnostics = layer.diagnostics
    pose, light = diagnostics.get("pose", {}), diagnostics.get("lighting", {})
    uv, fit = diagnostics.get("uv_synthesis", {}), diagnostics.get("identity_fit", {})
    return _json_safe({
        "pose": {key: pose.get(key) for key in ("source", "yaw_deg", "pitch_deg", "roll_deg", "reprojection_rms_px",
                                                 "inliers", "translation_m", "camera_intrinsics", "expression")},
        "lighting": {key: light.get(key) for key in ("method", "samples", "linear_bgr_rms", "key_direction_camera")},
        "uv": {key: uv.get(key) for key in ("observed_ratio", "symmetry_ratio", "harmonic_ratio", "view_count",
                                             "views_with_observed_uv")},
        "identity_fit": {key: fit.get(key) for key in ("fit_method", "fit_converged", "fit_cost")},
        "texture_restoration": diagnostics.get("texture_restoration"),
        "rasterization": diagnostics.get("rasterization"),
        "limitations": diagnostics.get("limitations")})


def _nearest_source_view(document: Any, yaw: float, pitch: float) -> np.ndarray | None:
    """The FaceSet view whose recorded pose is nearest the target pose (display only: the head uses every trusted view)."""
    from roop_3d.core.pose import as_pose3d
    views = [view for view in document.views if not view.is_synthetic and view.landmarks_trusted] or list(document.views)
    if not views:
        return None

    def distance(view) -> float:
        pose = as_pose3d((view.yaw, view.pitch, view.roll))
        return (pose.yaw - yaw) ** 2 + (pose.pitch - pitch) ** 2

    best = min(views, key=distance)
    return cv2.imdecode(np.frombuffer(best.image_bytes, np.uint8), cv2.IMREAD_COLOR)


@dataclasses.dataclass
class _Selected:
    frame: Any
    image: np.ndarray        # the working-resolution BGR frame every stage ran on
    result: Any              # FrameResult
    seconds: float
    index: int
    replayed: bool


class _PreviewRun:
    """One preview: resolve the real target / FaceSet / mapping inputs exactly like a render job, replay the pipeline to the
    playhead frame, composite it, and report what was measured. Every failure is a structured ``DomainError``."""

    def __init__(self, state: Any, value: Any, cancel: Any, raw: bool, final_only: bool, effective: dict[str, Any]):
        self.state, self.value, self.cancel = state, value, cancel
        self.raw, self.final_only, self.effective = raw, final_only, effective
        service = state.previews
        self.max_frames = int(getattr(service, "MAX_FRAMES", 1200))
        self.budget = float(getattr(service, "TIME_BUDGET", 120))
        self.began = time.monotonic()
        self.session = None
        self.reader = None
        self.warmup_skipped = 0
        self.preroll = 0
        self.downscaled: tuple[int, int, int, int] | None = None

    # -- control ------------------------------------------------------------------------------------------------------
    def check(self) -> None:
        if self.cancel.is_set():
            raise DomainError("PREVIEW_CANCELLED", "Preview cancelled", 409)
        if time.monotonic() - self.began > self.budget:
            raise DomainError("PREVIEW_BUDGET", f"Preview exceeded the {self.budget:g} second budget; use an earlier frame "
                                                "or shorter clip", 422)

    def _pack(self, image: np.ndarray):
        if self.raw:
            return image
        from roop_3d.server.workstation_preview import encode
        return encode(image)

    # -- orchestration ------------------------------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        scheduler = self.state.previews.scheduler
        key = "preview-" + uuid.uuid4().hex
        admission = None
        try:
            self.check()
            settings = self.state.settings
            admission = self._admit(scheduler, key, settings)
            edit_plan, target, tracking = self._open_session(admission, settings)
            self.check()
            selected = self._select(edit_plan, target, tracking, admission)
            return self._finish(edit_plan, selected, admission)
        finally:
            try:
                if self.session is not None:
                    self.session.release_session()
            finally:
                try:
                    release = getattr(self.reader, "release", None)
                    if callable(release):
                        release()
                finally:
                    if admission is not None:
                        scheduler.release(key)

    def _admit(self, scheduler: Any, key: str, settings: Any):
        admission = scheduler.try_admit(key, settings)
        if admission is None:
            raise DomainError("PREVIEW_CAPACITY", "Rendering currently owns the available execution capacity", 409)
        wants_gpu = settings.hardware_profile != "cpu" and settings.preferred_provider != "CPUExecutionProvider"
        if admission.profile == "cpu" and wants_gpu:
            # Admission falls back to the CPU for the WHOLE preview when live free VRAM looks too small. Part of that VRAM is
            # this service's own warm sessions, which are reclaimable: release them and ask once more before accepting the CPU.
            from roop_3d.core.ort_session import evict_pooled_sessions
            if evict_pooled_sessions():
                scheduler.release(key)
                admission = scheduler.try_admit(key, settings)
                if admission is None:
                    raise DomainError("PREVIEW_CAPACITY", "Rendering currently owns the available execution capacity", 409)
            if admission.profile == "cpu":
                try:
                    from roop_3d.telemetry.pinokio_terminal import HUD
                    HUD.milestone("ADMISSION", "preview is running on the CPU",
                                  reason=getattr(admission.lease.admission, "reason", "?"),
                                  hint="free GPU memory or close other GPU apps")
                except Exception:
                    pass
        return admission

    def _open_session(self, admission: Any, settings: Any):
        from roop_3d.engine.edit_plan import FrameEditPlan
        from roop_3d.engine.pipeline_runner import MasterPipelineRunner
        from roop_3d.engine.volumetric.session import ReferencePipeline
        from roop_3d.jobs.schema import JobRecord
        state, value = self.state, self.value
        runner = MasterPipelineRunner(state.root, state.models, state.media, lambda: settings)
        self.reader = runner.faceset_reader
        # Reuse target hashes, inspection identity and FaceSet validation without creating a job or a checkpoint.
        target, _, _, sources, labels_to_ids, _ = runner._prepare_inputs(JobRecord(request=value.request), settings)
        _, identity_guard = runner.identity_context(value.request, labels_to_ids, settings)
        edit_plan = FrameEditPlan(runner, value.request, sources)
        runtime = runner._runtime_config(admission, runner.settings_for(value.request, settings))
        self.session = session = ReferencePipeline(state.models, runtime)
        tracking = runner.tracking_options(value.request)
        session.prepare_session(next(iter(sources.values())), track_ids=set(sources), sources_by_track=sources,
                                restoration_weight=value.effective_parameters.restoration_weight, **tracking)
        if hasattr(session, "set_target_registry") and hasattr(state.media, "target_locks"):
            session.set_target_registry(state.media.target_locks.registry(value.request.target_path))
        if identity_guard is not None:
            session.set_identity_guard(identity_guard)
        if hasattr(session, "configure_temporal"):
            session.configure_temporal(settings)
        return edit_plan, target, tracking

    # -- frame selection ----------------------------------------------------------------------------------------------
    def _working_image(self, frame: Any, admission: Any) -> np.ndarray:
        image = frame.image
        if image.shape[0] * image.shape[1] > 3840 * 2160:
            raise DomainError("PREVIEW_SIZE_BUDGET", "Preview input exceeds the 4K frame budget", 422)
        if admission.profile == "6gb" and max(image.shape[:2]) > 512:
            height, width = image.shape[:2]
            scale = 512.0 / max(height, width)
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            self.downscaled = (width, height, *size)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    def _infer(self, image: np.ndarray, tracked: Any, *, final: bool):
        """``infer_frame`` for the playhead frame (any error is the preview's error) or for a warm-up frame (a frame whose face
        cannot be fitted is skipped, exactly as the job adapter skips it)."""
        try:
            return self.session.infer_frame(image, **({"tracked": tracked} if tracked is not None else {}))
        except DomainError as error:
            if final or error.code not in frame_skip_codes():
                raise
            self.warmup_skipped += 1
            return None

    def _select(self, edit_plan: Any, target: Any, tracking: Any, admission: Any) -> _Selected:
        from roop_3d.media.probe import open_media
        value = self.value
        target_time = Fraction(str(value.timestamp))
        with open_media(target) as container:
            stream = container.streams.video[0]
            time_base = Fraction(stream.time_base)
            start_offset = (stream.start_time or 0) * time_base
            rate = stream.average_rate or stream.base_rate or 25
            fps = float(rate) if rate else 25.0
        replay = (target_time <= Fraction(1, 10) or value.frame_index is not None or bool(tracking)
                  or value.request.identity_lock_id is not None
                  or self.state.media.probe(value.request.target_path).vfr is True)
        if replay:
            return self._select_replayed(edit_plan, target, target_time, fps, admission)
        return self._select_seeked(edit_plan, target, target_time, start_offset, time_base, fps, admission)

    def _select_replayed(self, edit_plan: Any, target: Any, target_time: Fraction, fps: float, admission: Any) -> _Selected:
        """Replay tracking from frame zero so tracker state matches a job; only the last ``preview_preroll()`` frames before
        the playhead also render, which is what gives the pose smoothing its history."""
        from roop_3d.media.video_reader import VideoReader
        from roop_3d.server.workstation_preview import preview_preroll
        value, session = self.value, self.session
        self.preroll = preroll = preview_preroll()
        estimated = value.frame_index if value.frame_index is not None else int(round(float(target_time) * fps))
        full_from = estimated - preroll if preroll > 0 else None
        origin = selected = None
        with closing(VideoReader(target, hardware="software", cancel=self.cancel).frames()) as frames:
            for frame in frames:
                self.check()
                if frame.index >= self.max_frames:
                    raise DomainError("PREVIEW_FRAME_BUDGET", "Preview replays tracking from the beginning; select a frame "
                                                              f"within the first {self.max_frames} frames", 422)
                image = self._working_image(frame, admission)
                origin = frame.timestamp.seconds if origin is None else origin
                edit_plan.apply(session, frame.index)
                session.set_frame_timestamp(float(frame.timestamp.seconds))
                session.set_angle_frame(frame.index)
                tracked = session.track_frame(image)
                reached = (frame.index >= value.frame_index if value.frame_index is not None
                           else frame.timestamp.seconds - origin >= target_time)
                if full_from is not None and frame.index < full_from and not reached:
                    continue
                result = self._infer(image, tracked, final=reached)
                if result is None:
                    continue
                selected = _Selected(frame, image, result, float(frame.timestamp.seconds - origin), frame.index, True)
                if reached:
                    break
            else:
                raise DomainError("PREVIEW_FRAME_MISSING", "No frame exists at the requested playhead time", 422)
        return selected

    def _select_seeked(self, edit_plan: Any, target: Any, target_time: Fraction, start_offset: Fraction,
                       time_base: Fraction, fps: float, admission: Any) -> _Selected:
        """Seek to the keyframe before the playhead and decode forward: only the playhead frame is tracked and rendered, so
        temporal smoothing starts cold."""
        from roop_3d.media.timestamps import Timestamp
        from roop_3d.media.video_reader import VideoReader
        session = self.session
        seek = Timestamp(int(round((start_offset + target_time) / time_base)), time_base)
        step = Fraction(1, max(1, int(round(fps))))
        last = None

        def infer_at(frame: Any, image: np.ndarray) -> _Selected:
            seconds = frame.timestamp.seconds - start_offset
            index = int(round(float(seconds * Fraction(str(fps)))))
            edit_plan.apply(session, index)
            session.set_frame_timestamp(float(frame.timestamp.seconds))
            return _Selected(frame, image, self._infer(image, None, final=True), float(seconds), index, False)

        with closing(VideoReader(target, hardware="software", cancel=self.cancel).frames(seek=seek)) as frames:
            for frame in frames:
                self.check()
                if frame.index >= self.max_frames:
                    raise DomainError("PREVIEW_FRAME_BUDGET", "Preview seek exceeded decoder frame budget", 422)
                image = self._working_image(frame, admission)
                last = (frame, image)
                if frame.timestamp.seconds - start_offset >= target_time:
                    return infer_at(frame, image)
            if last is not None and target_time - (last[0].timestamp.seconds - start_offset) <= step:
                return infer_at(*last)
        raise DomainError("PREVIEW_FRAME_MISSING", "No frame exists at the requested playhead time", 422)

    # -- compositing and reporting ------------------------------------------------------------------------------------
    def _composite(self, edit_plan: Any, selected: _Selected, options: Any) -> np.ndarray:
        result = selected.result
        if edit_plan.has_masks:
            result = edit_plan.with_occluders(selected.image, result, selected.index)
        return self.session.composite_frame(selected.image, result, options=options, frame_index=selected.index,
                                            timestamp_seconds=selected.seconds)

    def _missing_message(self, result: Any) -> str:
        if result.withheld:
            first = result.withheld[0]
            return (f"No mapped face is rendered at this playhead frame: the identity gate withheld track {first.track_id} "
                    f"({first.reason})")
        return "No mapped face is visible at this playhead frame"

    def _finisher(self, selected: _Selected):
        """Export finishing (LUT, grain) exactly as the render applies it; the identity function when the request has none."""
        export = self.value.request.export
        if not export:
            return lambda picture: picture
        from roop_3d.media.finishing import Finisher
        finisher = Finisher(export.finishing)
        return lambda picture: finisher.apply(picture, selected.index)

    def _finish(self, edit_plan: Any, selected: _Selected, admission: Any) -> dict[str, Any]:
        value, session = self.value, self.session
        result, image = selected.result, selected.image
        if not result.replacements and value.request.face_segments is None:
            raise DomainError("PREVIEW_TRACK_MISSING", self._missing_message(result), 422)
        base = value.effective_parameters
        parameters = value.request.parameter_curves.evaluate(selected.index, base) if value.request.parameter_curves else base
        self.check()
        began = time.perf_counter()
        composite = self._composite(edit_plan, selected, parameters)
        composite_ms = (time.perf_counter() - began) * 1000.0
        # Read straight after the compositor ran: a sweep composites again and replaces both.
        processor = getattr(session, "_frame_processor", None)
        alpha = getattr(getattr(processor, "core", None), "last_alpha", None)
        report = session.last_frame_report if result.replacements else None
        if report is not None and not report.faces:
            skipped = report.skipped[0] if report.skipped else {}
            raise DomainError(skipped.get("code") or "PREVIEW_FRAME_NOT_RENDERED",
                              skipped.get("message") or "The head render could not be composited for this frame: "
                              f"{skipped.get('code', 'unknown reason')}", 422)
        layers = {}
        for item in result.replacements:
            engine = session.full_head_engines.get(item.track_id)
            if engine is not None and engine.last_layer is not None:
                layers[item.track_id] = engine.last_layer
        finish = self._finisher(selected)
        final = finish(composite)
        response = {
            "frame_index": selected.index, "timestamp": selected.seconds, "width": int(image.shape[1]),
            "height": int(image.shape[0]), "profile": str(admission.profile), "provider": admission.provider,
            "notes": self._notes(selected, layers, report), "faces": self._faces(result, layers),
            "withheld": [{"track_id": int(item.track_id), "reason": item.reason, "state": item.state,
                          "confidence": float(item.confidence)} for item in result.withheld],
            "renderer": {"id": RENDERER_ID, **self.effective}}
        latency = dataclasses.asdict(result.latency)
        latency.update(render_ms=latency["swap_ms"], composite_ms=round(composite_ms, 3))
        response.update(parameters=_json_safe(parameters.model_dump(mode="json")), latency=_json_safe(latency))
        if value.x and value.y:
            response["cells"] = self._cells(edit_plan, selected, parameters, finish)
        buffers = {"final": self._pack(final)}
        if not self.final_only:
            buffers.update({name: self._pack(picture)
                            for name, picture in self._buffers(selected, layers, alpha).items()})
        response["buffers"] = buffers
        return response

    def _cells(self, edit_plan: Any, selected: _Selected, parameters: Any, finish: Any) -> list[dict[str, Any]]:
        from roop_3d.core.workstation import WorkstationParameters
        value, cells = self.value, []
        for row, y in enumerate(value.y.values):
            for column, x in enumerate(value.x.values):
                self.check()
                coordinates = {value.x.parameter: x, value.y.parameter: y}
                cell = WorkstationParameters.model_validate({**parameters.model_dump(), **coordinates})
                picture = finish(self._composite(edit_plan, selected, cell))
                cells.append({"row": row, "column": column, "coordinates": coordinates,
                              "parameters": _json_safe(cell.model_dump(mode="json")), "image": self._pack(picture)})
        return cells

    def _faces(self, result: Any, layers: dict[int, Any]) -> list[dict[str, Any]]:
        """Detected boxes, 5-point landmarks and yaw/pitch/roll of every tracked face, plus the render facts of rendered ones."""
        from roop_3d.core.pose import as_pose3d
        faces = []
        for tracked in result.observations:
            face = tracked.observation
            pose = as_pose3d(face.pose)
            entry = {"track_id": int(tracked.track_id), "box": [float(v) for v in np.asarray(face.box).tolist()],
                     "landmarks5": np.asarray(face.landmarks5, float).tolist() if face.landmarks5 is not None else None,
                     "yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll, "state": tracked.state,
                     "confidence": float(tracked.confidence), "swap_allowed": bool(tracked.swap_allowed)}
            if tracked.track_id in layers:
                entry["render"] = _layer_summary(layers[tracked.track_id])
            faces.append(_json_safe(entry))
        return faces

    def _notes(self, selected: _Selected, layers: dict[int, Any], report: Any) -> list[str]:
        notes = [
            ("Single-frame preview: tracking replayed from frame zero; " + (
                f"the {self.preroll} frame(s) before the playhead also render so pose smoothing has history."
                if self.preroll > 0 else "every replayed frame renders.")) if selected.replayed else
            "Single-frame preview, seeked to the playhead keyframe; temporal smoothing starts cold.",
            "Head: MakeHuman CC0 template fitted to the FaceSet views, not a scan; unobserved texture is symmetry or harmonic "
            "completion.",
            "Shading: perspective-correct Z-buffer raster lit by spherical harmonics fitted to the visible target skin; no "
            "specular, no subsurface scattering.",
            "Camera: perspective pin-hole with an assumed 35 mm-equivalent focal length and the principal point at the image "
            "centre (no focal-length solver).",
            "Matte: face parser + XSeg visible-face + occluder networks over the head raster; hair, neck and occluders keep "
            "the target pixels."]
        if self.warmup_skipped:
            notes.append(f"{self.warmup_skipped} warm-up frame(s) before the playhead could not be fitted and were skipped.")
        if self.downscaled:
            notes.append("The 6 GB profile previews at most 512 px: the frame was downscaled from "
                         "{}x{} to {}x{}.".format(*self.downscaled))
        if len(layers) > 1:
            notes.append("The matte buffer shows the alpha of the last composited face only.")
        if report is not None and report.skipped:
            notes.extend(f"Face {item.get('track_id', '?')} was not composited: {item.get('code')}" for item in report.skipped)
        return notes

    def _buffers(self, selected: _Selected, layers: dict[int, Any], alpha: Any) -> dict[str, np.ndarray]:
        """Diagnostic buffers from the real 3D layers; a stage that produced nothing for this frame has no buffer."""
        from roop_3d.core.pose import as_pose3d
        image, result = selected.image, selected.result
        shape = image.shape[:2]
        detection, pose_view = image.copy(), image.copy()
        for tracked in result.observations:
            face = tracked.observation
            x1, y1, x2, y2 = (int(round(v)) for v in np.asarray(face.box).tolist()[:4])
            cv2.rectangle(detection, (x1, y1), (x2, y2), (80, 245, 180), 2)
            cv2.putText(detection, f"#{tracked.track_id}", (max(0, x1), max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (80, 245, 180), 2)
            points = face.landmarks68 if face.landmarks68 is not None else face.landmarks106
            for px, py in np.asarray(points, np.float64)[:, :2]:
                cv2.circle(pose_view, (int(round(px)), int(round(py))), 2, (80, 245, 180), -1)
            pose = as_pose3d(face.pose)
            cv2.putText(pose_view, "Y/P/R " + "/".join(f"{angle:.1f}" for angle in (pose.yaw, pose.pitch, pose.roll)),
                        (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 245, 180), 2)
        render = np.zeros_like(image)
        normal_map = np.zeros_like(image)
        depth_map = np.zeros_like(image)
        sh_lighting = np.zeros_like(image)
        from roop_3d.relighting_3d import SphericalHarmonicsLighting, SurfaceNormalEstimator
        from roop_3d.rendering.skin_calibration import linear_to_srgb
        for track_id, layer in layers.items():
            covered = layer.coverage > 0
            if not covered.any():
                continue
            ys, xs = np.nonzero(covered)
            box = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            region = covered[box] & np.isfinite(layer.depth[box])
            if not region.any():
                continue
            render[box][region] = layer.image[box][region]
            normal_map[box][region] = SurfaceNormalEstimator.encode_normal_map_rgb(layer.normals[box])[region]
            depth = layer.depth[box][region]
            span = float(depth.max() - depth.min())
            tone = np.clip(255.0 * (1.0 - (depth - depth.min()) / max(span, 1e-6)), 0, 255).astype(np.uint8)
            depth_map[box][region] = cv2.applyColorMap(tone.reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
            coefficients = np.asarray(layer.diagnostics["lighting"]["coefficients_bgr"], np.float32)
            shading = SphericalHarmonicsLighting().evaluate_shading(layer.normals[box], coefficients)
            display = np.clip(linear_to_srgb(np.clip(shading, 0.0, 1.0)) * 255.0, 0, 255).astype(np.uint8)
            sh_lighting[box][region] = display[region]
            contours, _ = cv2.findContours((layer.coverage > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(pose_view, contours, True, (255, 180, 65), 1, cv2.LINE_AA)
        buffers = {"detector": detection, "pose": pose_view}
        if layers:
            buffers.update(render=render, normal_map=normal_map, depth=depth_map, sh_lighting=sh_lighting)
        if alpha is not None and np.asarray(alpha).shape == shape:
            buffers["matte"] = cv2.cvtColor(np.clip(np.asarray(alpha) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if layers:
            track_id, layer = next(iter(layers.items()))
            pose = layer.diagnostics["pose"]
            source = _nearest_source_view(self.session._document_for(track_id), float(pose["yaw_deg"]),
                                          float(pose["pitch_deg"]))
            if source is not None:
                buffers["source"] = source
        return buffers


def render_preview(state: Any, request: Any, cancel: Any, *, raw: bool = False, final_only: bool = False) -> dict[str, Any]:
    """One bounded workstation preview through the same ``ReferencePipeline`` path as a render job.

    Returns ``{"buffers": {...}, "frame_index", "timestamp", "width", "height", "profile", "provider", "notes", "faces",
    "withheld", "parameters", "latency", "renderer"}`` (plus ``cells`` for a two-axis sweep). ``raw=True`` keeps the buffers as
    ndarrays, otherwise each is a JPEG data URL; ``final_only`` renders only ``final``. Unsupported options (manual camera,
    FLAME, Gaussian avatar, identity blends, render passes, inert sweep axes) raise before any model is touched."""
    effective = validate_true3d(request.true3d)
    validate_job_request(request.request)
    if _field(request, "identity_blend") is not None:
        validate_job_request({"identity_blend": request.identity_blend})
    return _PreviewRun(state, request, cancel, raw, final_only, effective).run()


# --------------------------------------------------------------------------- camera studio
class LiveSession:
    """A prepared ``ReferencePipeline`` for the camera studio: ``render(frame)`` returns the frame (same size, uint8 BGR) with
    every tracked head replaced by the rendered 3D head, ``close()`` releases the session and its FaceSet reader.

    A frame whose face cannot be fitted passes through unchanged and is counted in ``stats()`` (never hidden); any other
    error (missing weights, invalid tensors) propagates and stops the studio."""

    def __init__(self, session: Any, reader: Any, options: Any, effective: dict[str, Any]):
        self.session, self.reader, self.options, self.effective = session, reader, options, effective
        self._origin = time.monotonic()
        self._index = 0
        self._closed = False
        self._lock = threading.Lock()
        self._counts = {"frames": 0, "frames_without_face": 0, "faces_rendered": 0, "faces_skipped": 0, "faces_withheld": 0}
        self._last_skip: dict[str, Any] | None = None
        self._skip_codes = frame_skip_codes()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._counts, "last_skip": dict(self._last_skip) if self._last_skip else None}

    def _skipped(self, code: str, message: str) -> None:
        with self._lock:
            self._counts["frames"] += 1
            self._counts["faces_skipped"] += 1
            self._last_skip = {"code": code, "message": str(message)[:240], "frame": self._index - 1}

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self._closed:
            raise DomainError("LIVE_SESSION_CLOSED", "The live 3D session has been closed", 409)
        seconds = time.monotonic() - self._origin
        index = self._index
        self._index += 1
        session = self.session
        session.set_frame_timestamp(seconds)
        try:
            result = session.infer_frame(frame)
        except DomainError as error:
            if error.code not in self._skip_codes:
                raise
            self._skipped(error.code, error.message)
            return frame
        output = session.composite_frame(frame, result, options=self.options, frame_index=index, timestamp_seconds=seconds)
        report = session.last_frame_report if result.replacements else None
        with self._lock:
            self._counts["frames"] += 1
            self._counts["faces_withheld"] += len(result.withheld)
            if report is None:
                self._counts["frames_without_face"] += 1
                return output
            self._counts["faces_rendered"] += len(report.faces)
            self._counts["faces_skipped"] += len(report.skipped)
            if report.skipped:
                first = report.skipped[0]
                self._last_skip = {"code": first.get("code"), "message": str(first.get("message") or first.get("details"))[:240],
                                   "frame": index}
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.release_session()
        finally:
            release = getattr(self.reader, "release", None)
            if callable(release):
                release()


def open_live_session(state: Any, config: Any, true3d: Any, admission: Any) -> LiveSession:
    """Prepare the camera studio's session: the FaceSet is read from owned storage, the pipeline is admitted on ``admission``'s
    profile and provider, and every tracked face is rendered by the volumetric renderer (no track selection: the studio
    replaces all faces with its one source)."""
    effective = validate_true3d(true3d)
    from roop_3d.core.workstation import WorkstationParameters
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner
    from roop_3d.engine.volumetric.session import ReferencePipeline
    settings = state.settings
    weight = _field(config, "restoration_weight")
    weight = DEFAULT_RESTORATION_WEIGHT if weight is None else float(weight)
    runner = MasterPipelineRunner(state.root, state.models, state.media, lambda: settings)
    reader = runner.faceset_reader
    session = None
    try:
        source = reader.read(runner._faceset_path(config.source_faceset_id), faceset_id=config.source_faceset_id)
        session = ReferencePipeline(state.models, runner._runtime_config(admission, settings))
        session.prepare_session(source, restoration_weight=weight)
        if hasattr(session, "configure_temporal"):
            session.configure_temporal(settings)
        return LiveSession(session, reader, WorkstationParameters(restoration_weight=weight), effective)
    except BaseException:
        try:
            if session is not None:
                session.release_session()
        finally:
            release = getattr(reader, "release", None)
            if callable(release):
                release()
        raise


__all__ = ["CAMERA_SUPPORT", "FRAME_FIT_CODES", "LiveSession", "RENDERER_ID", "SHADER_SUPPORT", "TOPOLOGY_SUPPORT",
           "UnsupportedTrue3DOption", "blockers", "describe", "frame_skip_codes", "is_wired", "open_live_session",
           "render_frame", "render_preview", "validate_job_request", "validate_true3d"]
