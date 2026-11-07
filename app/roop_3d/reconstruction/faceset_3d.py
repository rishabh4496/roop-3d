"""Local FaceSet pose coverage and best-effort dense facial surface reconstruction."""
from __future__ import annotations

import base64
import math
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.faceset.pose_matrix import view_pose
from roop_3d.faceset.schemas import FaceSetDocument, FaceView
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, Mesh3D
from roop_3d.reconstruction.multiview_fit import (
    ViewLandmarks, deform_template, fit_identity_landmarks, five_from_68, identity_template,
)
from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks, solve_pose_pnp

YAW_BINS, PITCH_BINS = 12, 8
TOTAL_BINS = YAW_BINS * PITCH_BINS


def _pose(view: FaceView) -> tuple[float, float, float, float]:
    """Return PnP pose and reprojection RMS; use the stored measured pose on degeneracy."""
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    points = np.asarray(view.landmarks_68 if view.landmarks_68 is not None else view.landmarks, np.float64)
    if points.shape[0] == 68:
        # PnP against the canonical 68-point model is ill-conditioned (flip ambiguity: measured pitch -177 deg on a frontal
        # photo), so the dense set is reduced to the same five landmarks every other consumer of pose uses.
        points = five_from_68(points)
    if view.landmarks_trusted and image is not None and points.shape[0] in (5, 68) and np.isfinite(points).all():
        try:
            h, w = image.shape[:2]
            pose = solve_pose_pnp(points, CameraIntrinsics.from_fov(55.0, w, h),
                                  get_canonical_3d_landmarks(points.shape[0]))
            values = (pose.yaw, pose.pitch, pose.roll, pose.reprojection_error)
            if all(math.isfinite(x) for x in values):
                return values
        except (DomainError, cv2.error, ValueError, np.linalg.LinAlgError):
            pass
    stored = view_pose(view)
    return float(stored.yaw), float(stored.pitch), float(stored.roll), 0.0


def sphere_position(yaw: float, pitch: float) -> tuple[float, float, float]:
    yaw = min(90.0, max(-90.0, yaw))
    pitch = min(60.0, max(-60.0, pitch))
    yr, pr = math.radians(yaw), math.radians(pitch)
    return (math.sin(yr) * math.cos(pr), math.sin(pr), math.cos(yr) * math.cos(pr))


def pose_bin(yaw: float, pitch: float) -> int:
    yi = min(YAW_BINS - 1, max(0, int((min(90.0, max(-90.0, yaw)) + 90.0) / 180.0 * YAW_BINS)))
    pi = min(PITCH_BINS - 1, max(0, int((min(60.0, max(-60.0, pitch)) + 60.0) / 120.0 * PITCH_BINS)))
    return pi * YAW_BINS + yi


def observe_poses(document: FaceSetDocument) -> list[dict[str, Any]]:
    output = []
    for view in document.views:
        yaw, pitch, roll, reprojection = _pose(view)
        # Coverage is defined for the frontal viewing hemisphere. Clamping maps edge
        # views onto the edge bins while preserving their measured pose in the payload.
        index = pose_bin(yaw, pitch)
        output.append({"view_index": view.index, "yaw": round(yaw, 3), "pitch": round(pitch, 3),
                       "roll": round(roll, 3), "bin_index": index,
                       "position_xyz": [round(v, 7) for v in sphere_position(yaw, pitch)],
                       "reprojection_error_px": round(reprojection, 3)})
    return output


def _png_base64(image: np.ndarray) -> str:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if image.ndim == 3 and image.shape[2] == 3 else image
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise DomainError("TEXTURE_ENCODE_FAILED", "Could not encode the reconstructed UV texture", 500)
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _serialize_mesh(mesh: Mesh3D, texture: np.ndarray, *, level: str, limitations: list[str],
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**(extra or {}),"vertices": np.round(mesh.vertices.astype(np.float64), 6).tolist(),
            "uvs": np.round(mesh.uvs.astype(np.float64), 6).tolist() if mesh.uvs is not None else [],
            "indices": mesh.faces.astype(np.int64).tolist(),
            "normals": np.round(mesh.normals.astype(np.float64), 6).tolist(),
            "texture_base64": _png_base64(texture), "reconstruction_level": level,
            "limitations": limitations}


def _estimated_mesh(view: FaceView, pose: tuple[float, float, float]) -> tuple[Mesh3D, np.ndarray, str, list[str]]:
    template = identity_template(2)
    points = np.asarray(view.landmarks_68 if view.landmarks_68 is not None else view.landmarks, np.float64)
    limitations = ["Single-view depth is an anthropometric prior; image-only data does not triangulate hidden depth."]
    if not view.landmarks_trusted:
        # The canonical triangulated template is an always-available fallback.
        mesh = template
        level = "canonical_template_fallback"
        limitations.append("This view has no trusted landmarks; canonical facial topology is shown without identity deformation.")
    else:
        try:
            fit = fit_identity_landmarks([ViewLandmarks(view.index, points, pose, max(.05, view.quality.score))],
                                         template=template, holdout=False, max_iterations=24)
            vertices, _, _ = deform_template(template, fit.template_landmarks, fit.landmarks_3d, fit.confidence_xy)
            mesh = Mesh3D(vertices, template.faces, uvs=template.uvs)
            level = fit.level
        except Exception:
            # The canonical triangulated template is an always-available fallback.
            mesh = template
            level = "canonical_template_fallback"
            limitations.append("Landmark fitting was unavailable; canonical facial topology is shown without identity deformation.")
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DomainError("IMAGE_UNDECODABLE", "A source FaceSet view cannot be decoded", 422)
    texture = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA if max(image.shape[:2]) > 512 else cv2.INTER_CUBIC)
    return mesh, texture, level, limitations


def _display_texture(asset, mesh: Mesh3D) -> tuple[np.ndarray, dict[str, Any]]:
    """Hole-free BGRA atlas for the viewer: symmetry projection + harmonic skin-tone fill of never-observed texels.

    The asset's own atlas stays observation-only (the reenactment/blending code relies on that); this is display only."""
    from roop_3d.reconstruction.geometry_types import ConfidenceState
    from roop_3d.reconstruction.uv_completion import complete_texture
    from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer
    size = int(asset.texture.shape[0])
    uv_for_atlas = np.column_stack([asset.uvs[:, 0], 1.0 - asset.uvs[:, 1]])
    synth = UVSynthesizer(Mesh3D(asset.vertices, asset.faces, uvs=uv_for_atlas), uv_size=size)
    state = np.asarray(asset.texel_state)
    known = (state == int(ConfidenceState.OBSERVED)) | (state == int(ConfidenceState.INTERPOLATED))
    mirror_x = float(np.median(asset.vertices[:, 0]))
    result = complete_texture(asset.texture, known, synth.face_texel_mask, synth.pos_map.astype(np.float64), mirror_x=mirror_x)
    return result.texture, {"method": "symmetry_projection+harmonic_skin_tone", **result.stats,
                            "note": "Completed texels are generated for display only; they are not observations."}


def _deformation_report(asset, settings) -> dict[str, Any]:
    from roop_3d.reconstruction.identity_asset import template_for
    base = template_for(settings).vertices.astype(np.float64)
    dev = np.linalg.norm(asset.vertices.astype(np.float64) - base, axis=1) * 1000.0
    fit = asset.manifest.fit
    return {"template": settings.template, "landmark_layout": asset.manifest.landmark_layout,
            "reconstruction_level": asset.manifest.reconstruction_level, "method": "wendland_rbf+laplacian_regularised",
            "vertex_count": int(len(base)),
            "deviation_from_neutral_mm": {"mean": round(float(dev.mean()), 3), "p95": round(float(np.percentile(dev, 95)), 3),
                                          "max": round(float(dev.max()), 3)},
            "landmark_fit_rms_mm": round(float(fit.get("rms_mm", float("nan"))), 3),
            "yaw_span_deg": round(float(fit.get("yaw_span_deg", 0.0)), 2),
            "deformation_strength": round(float(fit.get("deformation_strength", 1.0)), 3),
            "dropped_inconsistent_views": list(fit.get("dropped_inconsistent_views", []))}


def _full_mesh(document: FaceSetDocument, estimated: Mesh3D, fallback_texture: np.ndarray):
    from roop_3d.reconstruction.identity_asset import IdentityAssetSettings, build_identity_asset
    errors = []
    # Prefer the bundled CC0 head+neck template when present. Its geometry includes the full head volume;
    # identity variation outside the observed landmarks remains explicitly marked as template prior.
    for template in ("makehuman", "canonical"):
        try:
            settings = IdentityAssetSettings(uv_size=512, subdivisions=2, max_views=32,
                                             holdout_validation=False, template=template)
            asset = build_identity_asset(document, settings)
            mesh = asset.mesh()
            limitations = list(asset.manifest.limitations) + list(asset.manifest.missing_information)
            texture, completion = _display_texture(asset, mesh)
            extra = {"template": template, "deformation": _deformation_report(asset, settings), "texture_completion": completion}
            return mesh, texture, asset.manifest.reconstruction_level, limitations, extra
        except Exception as error:
            errors.append(f"{template}: {str(error)[:160]}")
    limitations = ["Identity-specific multiview fitting was unavailable; symmetric canonical facial geometry is shown.",
                   "The fallback mesh covers the facial region, not the skull or back of the head."]
    return estimated, fallback_texture, "canonical_symmetric_fallback", limitations + errors, {"template": "canonical_fallback"}


def reconstruct_faceset(document: FaceSetDocument, *, live_pose: dict[str, float] | None = None,
                        loaded: Any | None = None) -> dict[str, Any]:
    """``loaded`` (a ``faceset_loader.LoadedFaceSet``) adds the per-sample landmark/pose/bin report to the payload."""
    if not document.views:
        raise DomainError("FACESET_EMPTY", "At least one face sample is required for reconstruction", 422)
    observations = observe_poses(document)
    pose_by_index = {item["view_index"]: item for item in observations}
    primary = min(document.views, key=lambda v: (abs(pose_by_index[v.index]["yaw"]) +
                                                  abs(pose_by_index[v.index]["pitch"]),
                                                  -v.quality.score))
    primary_pose = pose_by_index[primary.index]
    estimated, face_texture, estimated_level, estimated_limits = _estimated_mesh(
        primary, (primary_pose["yaw"], primary_pose["pitch"], primary_pose["roll"]))
    full, full_texture, full_level, full_limits, full_extra = _full_mesh(document, estimated, face_texture)
    populated = {item["bin_index"] for item in observations}
    count = len(populated)
    compatibility_views = [{"yaw": item["yaw"], "pitch": item["pitch"], "roll": item["roll"],
                            "pose_bin": f"bin_{item['bin_index']}"} for item in observations]
    bins = {f"bin_{index}": sum(item["bin_index"] == index for item in observations)
            for index in range(TOTAL_BINS) if index in populated}
    return {"status": "ready", "faceset_id": document.faceset_id, "views": compatibility_views, "bins": bins,
            "observed_views": len(observations), "sphere_coverage_count": count,
            "sphere_coverage_total": TOTAL_BINS, "sphere_coverage_pct": round(count / TOTAL_BINS * 100.0, 2),
            "live_pose": ({"yaw": float(live_pose["yaw"]), "pitch": float(live_pose["pitch"]),
                           "roll": float(live_pose.get("roll", 0.0))} if live_pose else
                          {"yaw": primary_pose["yaw"], "pitch": primary_pose["pitch"], "roll": primary_pose["roll"]}),
            "observed_pose_points": observations,
            "mesh_2_5d": _serialize_mesh(estimated, face_texture, level=estimated_level, limitations=estimated_limits),
            "mesh_full_3d": _serialize_mesh(full, full_texture, level=full_level, limitations=full_limits, extra=full_extra),
            "uv_texture_data_url": "data:image/png;base64," + _png_base64(full_texture),
            "texture_size": [int(full_texture.shape[1]), int(full_texture.shape[0])],
            "texture_channels": int(full_texture.shape[2]) if full_texture.ndim == 3 else 1,
            **({"faceset_samples": (lambda s: {key: s[key] for key in ("samples", "pose_matrix_3x3_roll", "landmark_models",
                                                                       "notes", "sphere_grid")})(loaded.summary())}
               if loaded is not None else {})}
