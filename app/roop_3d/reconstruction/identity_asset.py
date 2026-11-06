"""Persistent, identity-specific 3D face asset built from an authorised multi-view FaceSet (Stage 3, Phases D+E).

Pipeline: FaceSet quality analysis -> multi-view landmark fit (MAP, with posterior confidence) -> smooth template
deformation onto the fitted landmarks -> visibility/normal/sharpness-weighted texture atlas with explicit
OBSERVED / INTERPOLATED / UNSEEN states -> content-addressed storage independent of the FaceSet.

Honesty contract (enforced by tests): a vertex or texel is OBSERVED only if the views actually constrain it;
a front-only FaceSet yields a template-prior mesh whose depth confidence is zero and whose manifest lists what
is missing; nothing outside the facial region (ears, skull, back of head) is ever generated.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid
from typing import Any, Literal, Mapping

import cv2
import numpy as np
from pydantic import Field

from roop_3d.audit.isolation import safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.faceset.pose_matrix import view_pose
from roop_3d.faceset.schemas import FaceSetDocument, FaceView
from roop_3d.reconstruction.faceset_quality import FaceSetCoverageReport, analyze_faceset
from roop_3d.reconstruction.geometry_types import ConfidenceState, Mesh3D
from roop_3d.reconstruction.mesh_io import export_mesh_glb
from roop_3d.reconstruction.multiview_fit import (
    IdentityShapeFit,
    ViewLandmarks,
    deform_template,
    fit_identity_landmarks,
    identity_template,
    kernel_support,
)
from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation

SCHEMA = "roop.identity_asset.v1"
ALGORITHM_VERSION = "identity-mv-2"          # v2: grazing-angle gate + dominant-view blending (real-photo ghosting)
TEMPLATE_VERSION = "canonical68-delaunay+midpoint-subdiv"
TEMPLATES = ("canonical", "makehuman")
FRAME_DESCRIPTION = ("mesh frame (metres): +x image-right, +y down, +z toward the viewer; a frontal head has "
                     "yaw = pitch = roll = 0 (core.pose standard frame, pitch + = up)")
OBSERVED_VERTEX_CONFIDENCE = 0.6


class IdentityAssetSettings(Contract):
    uv_size: int = Field(default=512, ge=64, le=2048)
    subdivisions: int = Field(default=2, ge=0, le=3)
    max_views: int = Field(default=12, ge=1, le=32)
    holdout_validation: bool = True
    # Which head template the landmarks are carried onto. 'canonical' is the original 68-point facial mesh (the only one the
    # expression rig / pose blend understand); 'makehuman' is the open CC0 head+neck (5-point landmarks only; `subdivisions`
    # does not apply). It must be built first: python -m roop_3d.reconstruction.makehuman_build
    template: Literal["canonical", "makehuman"] = "canonical"
    max_interpolation_fraction: float = Field(default=0.02, ge=0.0, le=0.1)
    # Hashes of external models whose output the inputs depend on (e.g. the detector that produced the stored
    # landmarks). Part of the cache key: change one and the asset is rebuilt.
    model_hashes: dict[str, str] = Field(default_factory=dict)


class SourceReference(Contract):
    index: int
    image_sha256: str
    used: bool
    exclusion_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    identity_distance: float
    refined_yaw: float | None = None
    refined_pitch: float | None = None
    refined_roll: float | None = None
    reprojection_rms_mm: float | None = None
    holdout_error_mm: float | None = None


class IdentityAssetManifest(Contract):
    schema_id: str = SCHEMA
    algorithm_version: str = ALGORITHM_VERSION
    build_key: str
    created_at: str
    faceset_id: str
    faceset_source_sha256: str
    reconstruction_level: str
    landmark_layout: str
    claims: dict[str, bool]
    missing_information: list[str]
    limitations: list[str]
    sources: list[SourceReference]
    settings: dict[str, Any]
    model_version: dict[str, Any]
    model_hashes: dict[str, str | None]
    flame: dict[str, Any]
    canonical_head_pose: dict[str, Any]
    shape: dict[str, Any]
    expression: dict[str, Any]
    fit: dict[str, Any]
    coverage: dict[str, Any]
    statistics: dict[str, Any]
    arrays_sha256: dict[str, str]


@dataclass
class IdentityAsset:
    manifest: IdentityAssetManifest
    vertices: np.ndarray            # (V, 3) float32, mesh frame
    faces: np.ndarray               # (F, 3) int32
    normals: np.ndarray             # (V, 3) float32
    uvs: np.ndarray                 # (V, 2) float32, OBJ convention (v up); atlas row = (1 - v) * (H - 1)
    vertex_confidence: np.ndarray   # (V,) float32
    vertex_state: np.ndarray        # (V,) uint8 ConfidenceState
    texture: np.ndarray             # (H, W, 3) uint8 BGR
    texel_state: np.ndarray         # (H, W) uint8 ConfidenceState
    texel_confidence: np.ndarray    # (H, W) float32
    texel_view_count: np.ndarray    # (H, W) uint8
    texel_source_view: np.ndarray   # (H, W) int32, FaceSet view index of the strongest contributor (-1 none)
    landmarks_3d: np.ndarray        # (L, 3)
    landmark_confidence: np.ndarray # (L,)
    landmark_sigma: np.ndarray      # (L, 3) posterior std-dev, metres
    template_landmarks: np.ndarray  # (L, 3)
    shape_params: np.ndarray        # (L, 3) landmark displacement from the template, metres
    expression_params: np.ndarray   # (n_views, n_measurements)

    @property
    def build_key(self) -> str:
        return self.manifest.build_key

    def mesh(self) -> Mesh3D:
        return Mesh3D(self.vertices, self.faces, uvs=self.uvs, normals=self.normals)

    @property
    def array_fields(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _ARRAY_FIELDS}


_ARRAY_FIELDS = ("vertices", "faces", "normals", "uvs", "vertex_confidence", "vertex_state", "texture", "texel_state",
                 "texel_confidence", "texel_view_count", "texel_source_view", "landmarks_3d", "landmark_confidence",
                 "landmark_sigma", "template_landmarks", "shape_params", "expression_params")


# --------------------------------------------------------------------------- hashing / keys
def hash_array(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(tuple(array.shape)).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def flame_status() -> dict[str, Any]:
    """Whether FLAME may be used. It is never used silently: blocked / unlicensed / unverified => unavailable."""
    try:
        from roop_3d.models.registry import ModelRegistry
        record = ModelRegistry().get("flame-2020")
    except Exception as err:  # registry unreadable is also "unavailable", never "assume available"
        return {"model_id": "flame-2020", "available": False, "used": False, "reason": f"registry unavailable: {type(err).__name__}"}
    verified = record.status != "blocked" and bool(record.expected_sha256) and bool(record.locator)
    reason = ("weights are not installed or verified, or the academic license has not been accepted"
              if not verified else "registered and hashed, but this build ships no FLAME shape-fitting adapter")
    return {"model_id": record.model_id, "registry_status": record.status, "license_id": record.license_id,
            "expected_sha256": record.expected_sha256, "available": False, "used": False, "reason": reason}


def _template_hash(subdivisions: int) -> str:
    mesh = identity_template(subdivisions)
    return hash_array(mesh.vertices) + hash_array(mesh.faces)


def template_for(settings: IdentityAssetSettings) -> Mesh3D:
    """The template mesh the settings select (a fresh copy)."""
    if settings.template == "makehuman":
        from roop_3d.reconstruction.makehuman_template import makehuman_template
        return makehuman_template()
    return identity_template(settings.subdivisions)


def _settings_template_hash(settings: IdentityAssetSettings) -> str:
    if settings.template == "makehuman":
        from roop_3d.reconstruction.makehuman_template import template_hash
        return template_hash()
    return _template_hash(settings.subdivisions)


def _template_supports_dense(template: Mesh3D) -> bool:
    """Only the canonical template has the 68-point correspondence."""
    return getattr(template, "landmark_vertices", None) is None


def _key_settings(settings: IdentityAssetSettings) -> dict[str, Any]:
    """Settings that enter the build key. The default template is omitted so canonical keys (and stored assets) are unchanged."""
    dumped = settings.model_dump()
    if dumped.get("template") == "canonical":
        dumped.pop("template", None)
    return dumped


def _template_version(settings: IdentityAssetSettings) -> str:
    if settings.template == "makehuman":
        from roop_3d.reconstruction.makehuman_template import TEMPLATE_VERSION as MH_VERSION
        return MH_VERSION
    return TEMPLATE_VERSION


def compute_build_key(document: FaceSetDocument, settings: IdentityAssetSettings,
                      occlusion_masks: Mapping[int, np.ndarray] | None = None) -> str:
    """Content address of everything that determines the asset. Unrelated FaceSet edits (rename) do not change it."""
    views = []
    for v in document.views:
        views.append({
            "image": hashlib.sha256(v.image_bytes).hexdigest(),
            "embedding": hashlib.sha256(np.asarray(v.embedding, np.float32).tobytes()).hexdigest(),
            "landmarks": [[round(x, 3), round(y, 3)] for x, y in v.landmarks],
            "landmarks_68": None if v.landmarks_68 is None else [[round(x, 3), round(y, 3)] for x, y in v.landmarks_68],
            "trusted": v.landmarks_trusted, "bbox": [round(x, 2) for x in v.bbox],
            "pose": [round(v.yaw, 3), round(v.pitch, 3), round(v.roll, 3)],
            "quality": [round(v.quality.score, 4), round(v.quality.sharpness, 4)],
        })
    masks = {str(i): hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for i, m in sorted((occlusion_masks or {}).items())}
    payload = {"schema": SCHEMA, "algorithm": ALGORITHM_VERSION, "settings": _key_settings(settings), "views": views,
               "masks": masks, "template": _settings_template_hash(settings), "flame": flame_status()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- expression measurements
EXPRESSION_FIELDS = ("mouth_width_over_iod", "mouth_open_over_iod", "brow_raise_over_iod")


def expression_measurements(view: FaceView, yaw_deg: float) -> dict[str, float | None]:
    """Image-space facial-expression ratios, corrected for yaw foreshortening. These are MEASUREMENTS, not FLAME
    coefficients; ``None`` where the landmark layout cannot measure them or the head is too turned."""
    corr = max(math.cos(math.radians(yaw_deg)), 0.5)
    if abs(yaw_deg) > 60.0:
        return {k: None for k in EXPRESSION_FIELDS}
    five = np.asarray(view.landmarks, np.float64)
    iod = float(np.linalg.norm(five[0] - five[1])) / corr
    out: dict[str, float | None] = {k: None for k in EXPRESSION_FIELDS}
    if iod < 1e-6:
        return out
    out["mouth_width_over_iod"] = float(np.linalg.norm(five[3] - five[4])) / corr / iod
    if view.landmarks_68 is not None:
        p = np.asarray(view.landmarks_68, np.float64)
        out["mouth_open_over_iod"] = float(np.mean(np.linalg.norm(p[[61, 62, 63]] - p[[67, 66, 65]], axis=1))) / iod
        eyes_y, brows_y = p[36:48, 1].mean(), p[17:27, 1].mean()
        out["brow_raise_over_iod"] = float(eyes_y - brows_y) / iod
    return out


# --------------------------------------------------------------------------- build
def _decode(view: FaceView) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise DomainError("IMAGE_UNDECODABLE", f"View {view.index} image cannot be decoded", 422)
    return img


def _landmarks_of(view: FaceView, use68: bool) -> np.ndarray:
    return np.asarray(view.landmarks_68 if use68 else view.landmarks, np.float64)


def build_identity_asset(
    document: FaceSetDocument,
    settings: IdentityAssetSettings | None = None,
    *,
    occlusion_masks: Mapping[int, np.ndarray] | None = None,
    report: FaceSetCoverageReport | None = None,
) -> IdentityAsset:
    """Reconstruct the identity asset. Raises ``DomainError(IDENTITY_ASSET_NOT_BUILDABLE)`` with concrete image
    requests when the FaceSet cannot support it (nothing usable, or more than one person)."""
    settings = settings or IdentityAssetSettings()
    report = report or analyze_faceset(document, occlusion_masks=occlusion_masks)
    if report.readiness in ("insufficient", "blocked_inconsistent_identity"):
        excluded = "; ".join(f"view {q.index}: {', '.join(q.exclusion_reasons)}" for q in report.views if q.exclusion_reasons)
        parts = [report.readiness, *([f"excluded -> {excluded}"] if excluded else []), *report.requests]
        raise DomainError("IDENTITY_ASSET_NOT_BUILDABLE", " | ".join(parts), 422)

    by_index = {v.index: v for v in document.views}
    q_by_index = {q.index: q for q in report.views}
    chosen = sorted(report.usable_indices, key=lambda i: -by_index[i].quality.score)[:settings.max_views]
    template = template_for(settings)
    use68 = all(by_index[i].landmarks_68 is not None for i in chosen) and _template_supports_dense(template)

    def observations(indices: list[int]) -> list[ViewLandmarks]:
        out = []
        for i in indices:
            v, q = by_index[i], q_by_index[i]
            pose = view_pose(v)
            out.append(ViewLandmarks(i, _landmarks_of(v, use68), (pose.yaw, pose.pitch, pose.roll),
                                     float(np.clip(v.quality.score * max(q.landmark_score, 0.05), 0.05, 1.0))))
        return out

    fit = fit_identity_landmarks(observations(chosen), template=template, holdout=settings.holdout_validation)
    # Iteratively drop the single worst inconsistent view and refit. (A corrupted view inflates every hold-out
    # refit, so several views may be flagged at first; only the worst is a genuine outlier until it is removed.)
    dropped: list[int] = []
    while len(chosen) > 1:
        flagged = [vf for vf in fit.views if not vf.consistent]
        if not flagged:
            break
        worst = max(flagged, key=lambda vf: vf.holdout_mm if vf.holdout_mm is not None else vf.rms_mm)
        dropped.append(worst.view_index)
        chosen = [i for i in chosen if i != worst.view_index]
        fit = fit_identity_landmarks(observations(chosen), template=template, holdout=settings.holdout_validation)
    fit_by_view = {vf.view_index: vf for vf in fit.views}

    # ---- geometry: template carried onto the fitted landmarks, with per-vertex support
    vertices, _, strength = deform_template(template, fit.template_landmarks, fit.landmarks_3d, fit.confidence_xy)
    geo = Mesh3D(vertices, template.faces, uvs=template.uvs)
    support_xy = kernel_support(template.vertices, fit.template_landmarks, fit.confidence_xy)
    support_z = kernel_support(template.vertices, fit.template_landmarks, fit.confidence_z)
    geometry_confidence = np.minimum(support_xy, support_z)

    # ---- texture atlas (the synthesizer addresses rows as v*(H-1); the asset keeps OBJ v-up, so flip for it)
    uv_for_atlas = np.column_stack([template.uvs[:, 0], 1.0 - template.uvs[:, 1]])
    synth = UVSynthesizer(Mesh3D(vertices, template.faces, uvs=uv_for_atlas), uv_size=settings.uv_size,
                          max_interpolation_fraction=settings.max_interpolation_fraction)
    views_obs = []
    for i in chosen:
        vf, q = fit_by_view[i], q_by_index[i]
        consistency = math.exp(-((max(vf.rms_mm, vf.holdout_mm or 0.0) / 6.0) ** 2))
        # Real-photo finding: near-profile views are registered poorly from 5 landmarks (the template geometry at 85 deg is a
        # guess), so their texture lands on background. Full trust up to 60 deg, fading to 10 % at 90 deg.
        registration = float(np.clip(1.0 - (abs(vf.yaw) - 60.0) / 30.0, 0.1, 1.0))
        views_obs.append(ViewObservation(
            image=_decode(by_index[i]), camera=fit.camera_for(vf), sharpness=max(q.sharpness, 0.1),
            view_quality=float(np.clip(q.landmark_score * consistency * registration * (0.5 + 0.5 * q.exposure_score), 0.0, 1.0)),
            view_index=i, occlusion_mask=None if not occlusion_masks else occlusion_masks.get(i)))
    atlas = synth.synthesize(views_obs)
    assert atlas.confidence_score is not None and atlas.view_count is not None and atlas.dominant_view is not None

    # ---- vertex states: visible in an OBSERVED texel AND geometrically constrained
    h = w = settings.uv_size
    px = np.clip(np.round(uv_for_atlas[:, 0] * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(uv_for_atlas[:, 1] * (h - 1)).astype(int), 0, h - 1)
    near_observed = cv2.dilate((atlas.confidence_mask == int(ConfidenceState.OBSERVED)).astype(np.uint8), np.ones((3, 3), np.uint8))
    visible = near_observed[py, px] > 0
    vertex_state = np.where(~visible, int(ConfidenceState.UNSEEN),
                            np.where(geometry_confidence >= OBSERVED_VERTEX_CONFIDENCE,
                                     int(ConfidenceState.OBSERVED), int(ConfidenceState.INTERPOLATED))).astype(np.uint8)
    vertex_conf = np.where(visible, geometry_confidence, 0.0).astype(np.float32)

    # ---- expression measurements per used view
    names = list(EXPRESSION_FIELDS)
    per_view_expr: dict[str, Any] = {}
    matrix = np.full((len(chosen), len(names)), np.nan, np.float32)
    for r, i in enumerate(chosen):
        m = expression_measurements(by_index[i], fit_by_view[i].yaw)
        per_view_expr[str(i)] = {"measurements": m, "stored": dict(by_index[i].expression)}
        for c, name in enumerate(names):
            if m[name] is not None:
                matrix[r, c] = m[name]

    # ---- honesty bookkeeping
    states = atlas.confidence_mask
    res = asset_statistics(atlas, vertex_state)
    missing = list(dict.fromkeys(report.requests))
    if fit.level == "single_view_prior":
        missing.insert(0, "Depth is not triangulated: with this yaw coverage the 3D shape of every landmark equals the "
                          "anthropometric template (depth confidence ~0). Complete 3D recovery is NOT claimed.")
    if res["texel_unseen_ratio"] > 0.05:
        missing.append(f"{res['texel_unseen_ratio']:.0%} of the facial texture atlas was never observed and is left empty.")
    flame = flame_status()
    if settings.template == "makehuman":
        coverage_limit = ("Geometry is the MakeHuman (CC0) head+neck template carried onto the fitted landmarks. Ears, skull, "
                          "neck and everything beyond the landmarks' influence are the template prior, never measured: they are "
                          "marked UNSEEN / INTERPOLATED and carry low per-vertex confidence.")
    else:
        coverage_limit = ("Geometry covers the FACIAL REGION only (anthropometric mesh mask). Ears, skull and back of head are "
                          "not reconstructed or generated.")
    limitations = [
        coverage_limit,
        "Geometry is the template deformed to fitted landmarks; surface detail between landmarks is not measured (see per-vertex confidence).",
        "Metric head size is a prior: a scale cannot be recovered from images without a known reference length.",
        "Pose comes from a 5-point scaled-orthographic estimate refined during the fit; typical error a few degrees.",
        "Expression is stored as image-space measurements, not FLAME coefficients.",
        "Texture lighting is only gain-compensated per view; shading/specular are baked into OBSERVED texels.",
    ]
    if not flame["used"]:
        limitations.append(f"FLAME not used: {flame['reason']}.")
    if settings.template == "makehuman" and any(v.landmarks_68 is not None for v in document.views):
        limitations.append("The MakeHuman template has no 68-point correspondence: dense landmarks were ignored and the 5-point layout was used.")
    if fit.layout == "5pt":
        limitations.append("Only 5 landmarks per view are available: lateral and vertical shape detail is far less constrained than with 68.")

    arrays = {
        "vertices": vertices.astype(np.float32), "faces": template.faces.astype(np.int32),
        "normals": geo.normals.astype(np.float32), "uvs": template.uvs.astype(np.float32),
        "vertex_confidence": vertex_conf, "vertex_state": vertex_state, "texture": atlas.texture_map,
        "texel_state": states, "texel_confidence": atlas.confidence_score, "texel_view_count": atlas.view_count,
        "texel_source_view": atlas.dominant_view, "landmarks_3d": fit.landmarks_3d.astype(np.float32),
        "landmark_confidence": fit.landmark_confidence.astype(np.float32), "landmark_sigma": fit.sigma_post.astype(np.float32),
        "template_landmarks": fit.template_landmarks.astype(np.float32), "shape_params": fit.displacement.astype(np.float32),
        "expression_params": matrix,
    }

    frontal = min(chosen, key=lambda i: abs(fit_by_view[i].yaw) + abs(fit_by_view[i].pitch))
    sources = []
    for v in document.views:
        q, vf = q_by_index[v.index], fit_by_view.get(v.index)
        reasons = list(q.exclusion_reasons)
        if v.index in dropped:
            reasons.append("MULTIVIEW_INCONSISTENT")
        elif q.usable and v.index not in chosen:
            reasons.append("NOT_SELECTED_MAX_VIEWS")
        sources.append(SourceReference(
            index=v.index, image_sha256=hashlib.sha256(v.image_bytes).hexdigest(), used=v.index in chosen,
            exclusion_reasons=reasons, warnings=list(q.warnings), identity_distance=q.identity_distance,
            refined_yaw=vf.yaw if vf else None, refined_pitch=vf.pitch if vf else None, refined_roll=vf.roll if vf else None,
            reprojection_rms_mm=vf.rms_mm if vf else None, holdout_error_mm=vf.holdout_mm if vf else None))

    depth_ok = fit.level != "single_view_prior" and fit.depth_observed_fraction >= 0.5
    manifest = IdentityAssetManifest(
        build_key=compute_build_key(document, settings, occlusion_masks),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        faceset_id=document.faceset_id, faceset_source_sha256=document.source_sha256,
        reconstruction_level=fit.level, landmark_layout=fit.layout,
        claims={"complete_3d_recovery": False, "back_of_head_recovered": False, "full_head_geometry": False,
                "identity_specific_landmarks": bool(np.any(fit.confidence_xy >= 0.5)), "depth_triangulated": bool(depth_ok),
                "texture_from_observations_only": True},
        missing_information=missing, limitations=limitations, sources=sources, settings=settings.model_dump(),
        model_version={"asset_schema": SCHEMA, "algorithm": ALGORITHM_VERSION, "template": _template_version(settings),
                       "landmark_fit": "weak-perspective MAP bundle adjustment", "texture": "z-buffer visibility-weighted blend",
                       "mesh_frame": FRAME_DESCRIPTION},
        model_hashes={"template_geometry_sha256": _settings_template_hash(settings),
                      "flame_weights_sha256": None, **settings.model_hashes},
        flame=flame,
        canonical_head_pose={"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "frame": FRAME_DESCRIPTION, "reference_view": frontal,
                             "reference_view_refined_pose": [fit_by_view[frontal].yaw, fit_by_view[frontal].pitch, fit_by_view[frontal].roll]},
        shape={"parameterization": "landmark_displacement_m", "shape": list(fit.displacement.shape),
               "flame_shape_coefficients": None, "posterior_sigma_m_mean": [float(x) for x in fit.sigma_post.mean(axis=0)]},
        expression={"representation": "image-space measurements", "fields": names, "per_view": per_view_expr,
                    "flame_expression_coefficients": None},
        fit={"layout": fit.layout, "converged": fit.converged, "iterations": fit.iterations, "rms_mm": fit.rms_mm,
             "holdout_rms_mm": fit.holdout_rms_mm, "yaw_span_deg": fit.yaw_span_deg,
             "depth_observed_fraction": fit.depth_observed_fraction, "deformation_strength": strength,
             "dropped_inconsistent_views": dropped, "notes": fit.notes},
        coverage={"readiness": report.readiness, "yaw_range_deg": report.yaw_range_deg, "yaw_coverage": report.yaw_coverage,
                  "pitch_coverage": report.pitch_coverage, "missing": report.missing, "identity": report.identity,
                  "duplicate_groups": report.duplicate_groups},
        statistics={**res, "rejected_photometric_samples": atlas.rejected_samples,
                    "view_gains": {str(k): list(v) for k, v in (atlas.view_gains or {}).items()}},
        arrays_sha256={name: hash_array(arr) for name, arr in arrays.items()})
    return IdentityAsset(manifest=manifest, **arrays)


def asset_statistics(atlas: Any, vertex_state: np.ndarray) -> dict[str, Any]:
    """Ratios are over the facial UV footprint (texels covered by the mesh), straight from the synthesizer."""
    return {
        "texel_observed_ratio": float(atlas.observed_ratio),
        "texel_interpolated_ratio": float(atlas.interpolated_ratio),
        "texel_unseen_ratio": float(atlas.unseen_ratio),
        "vertex_observed": int((vertex_state == int(ConfidenceState.OBSERVED)).sum()),
        "vertex_interpolated": int((vertex_state == int(ConfidenceState.INTERPOLATED)).sum()),
        "vertex_unseen": int((vertex_state == int(ConfidenceState.UNSEEN)).sum()),
        "vertex_count": int(len(vertex_state)),
    }


# --------------------------------------------------------------------------- GLB / inspection
def to_gltf_frame(mesh: Mesh3D) -> Mesh3D:
    """Project mesh frame (y down, z to viewer, left-handed) -> glTF (y up, right-handed). The y mirror flips
    triangle orientation, so the winding is swapped to keep outward faces counter-clockwise."""
    vertices = mesh.vertices * np.array([1.0, -1.0, 1.0], np.float32)
    faces = mesh.faces[:, [0, 2, 1]]
    normals = mesh.normals * np.array([1.0, -1.0, 1.0], np.float32)
    return Mesh3D(vertices, faces, uvs=mesh.uvs, normals=normals)


def export_asset_glb(asset: IdentityAsset, directory: Path, filename: str = "head.glb") -> Path:
    m = asset.manifest
    extras = {
        "roop3d_identity_asset": {
            "schema": m.schema_id, "build_key": m.build_key, "faceset_id": m.faceset_id,
            "reconstruction_level": m.reconstruction_level, "claims": m.claims, "missing_information": m.missing_information,
            "limitations": m.limitations, "arrays_sha256": m.arrays_sha256, "model_hashes": m.model_hashes,
            "sources": [{"index": s.index, "image_sha256": s.image_sha256, "used": s.used} for s in m.sources],
            "attribute_legend": {"_CONFIDENCE": "per-vertex geometry confidence 0..1",
                                 "_STATE": "0 UNSEEN, 1 INTERPOLATED, 2 OBSERVED"},
        }}
    return export_mesh_glb(
        to_gltf_frame(asset.mesh()), filename, texture_image=asset.texture, output_dir=directory, extras=extras,
        vertex_attributes={"_CONFIDENCE": asset.vertex_confidence, "_STATE": asset.vertex_state.astype(np.float32)})


# --------------------------------------------------------------------------- persistence
class IdentityAssetStore:
    """Content-addressed store under ``roop_3d/.runtime/identity_assets/<faceset_id>/<build_key[:24]>/``.

    Independent of the FaceSet (never modifies or links to it); an asset is reused iff the build key
    (images, landmarks, poses, settings, template and model hashes) is unchanged and its stored arrays still
    hash to the manifest."""

    KEEP_PER_FACESET = 3

    def __init__(self, root: Path | None = None):
        self.app_root = application_root()
        if root is not None:
            self.base = Path(root)
        else:
            self.base = safe_path(self.app_root, "roop_3d/.runtime/identity_assets", write=True)

    def _dir(self, faceset_id: str, build_key: str) -> Path:
        if not faceset_id or not all(c.isalnum() or c in "_-" for c in faceset_id):
            raise DomainError("INVALID_FACESET_ID", "FaceSet id may only contain letters, digits, '_' and '-'", 422)
        return self.base / faceset_id / build_key[:24]

    def save(self, asset: IdentityAsset) -> Path:
        final = self._dir(asset.manifest.faceset_id, asset.manifest.build_key)
        if (final / "manifest.json").exists():
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            np.savez_compressed(tmp / "arrays.npz", **asset.array_fields)
            ok, png = cv2.imencode(".png", asset.texture)
            if not ok:
                raise DomainError("TEXTURE_ENCODE_FAILED", "Failed to encode the identity atlas", 500)
            (tmp / "texture.png").write_bytes(png.tobytes())
            export_asset_glb(asset, tmp)
            (tmp / "manifest.json").write_text(asset.manifest.model_dump_json(indent=2), encoding="utf-8")
            try:
                os.replace(tmp, final)
            except OSError:  # a concurrent writer won the race; its content is identical by construction
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._prune(final.parent)
        return final

    def load(self, faceset_id: str, build_key: str) -> IdentityAsset | None:
        """The stored asset, or None when absent / stale / corrupted (any array not matching its manifest hash)."""
        directory = self._dir(faceset_id, build_key)
        try:
            manifest = IdentityAssetManifest.model_validate_json((directory / "manifest.json").read_text("utf-8"))
            if manifest.build_key != build_key or manifest.schema_id != SCHEMA or manifest.algorithm_version != ALGORITHM_VERSION:
                return None
            with np.load(directory / "arrays.npz") as npz:
                arrays = {name: npz[name] for name in _ARRAY_FIELDS}
        except (OSError, ValueError, KeyError):
            return None
        if any(hash_array(arrays[name]) != manifest.arrays_sha256.get(name) for name in _ARRAY_FIELDS):
            return None
        return IdentityAsset(manifest=manifest, **arrays)

    def glb_path(self, faceset_id: str, build_key: str) -> Path:
        return self._dir(faceset_id, build_key) / "head.glb"

    def get_or_build(
        self,
        document: FaceSetDocument,
        settings: IdentityAssetSettings | None = None,
        *,
        occlusion_masks: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[IdentityAsset, bool]:
        """(asset, reused). Rebuilds only when the content-addressed key changed or the stored copy is invalid."""
        settings = settings or IdentityAssetSettings()
        key = compute_build_key(document, settings, occlusion_masks)
        cached = self.load(document.faceset_id, key)
        if cached is not None:
            return cached, True
        asset = build_identity_asset(document, settings, occlusion_masks=occlusion_masks)
        if asset.manifest.build_key != key:
            raise DomainError("BUILD_KEY_MISMATCH", "Internal error: build key changed during the build", 500)
        self.save(asset)
        return asset, False

    def _prune(self, faceset_dir: Path) -> None:
        builds = sorted((p for p in faceset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        for old in builds[self.KEEP_PER_FACESET:]:
            shutil.rmtree(old, ignore_errors=True)
