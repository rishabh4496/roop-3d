"""MakeHuman-based, full-head Stage 4 geometry, UV synthesis, and perspective rendering.

The public seam is deliberately small: construct once for a FaceSet, then call
``render`` for each target frame and ``release`` when the owning track ends.
MakeHuman geometry is a CC0 template prior; only the landmark-supported identity
axes and observed UV texels are reported as fitted/observed. Unseen geometry and
appearance stay explicitly tagged as template/symmetry/harmonic completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import euler_from_rotation
from roop_3d.engine.runtime import image_bgr
from roop_3d.faceset.schemas import FaceSetDocument, FaceView
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    PerspectiveCamera,
    UVOrigin,
    uv_to_pixel,
)
from roop_3d.reconstruction.makehuman_head import (
    MakeHumanHeadModel,
    ViewObservation as MakeHumanViewObservation,
    fit_multiview,
)
from roop_3d.reconstruction.makehuman_template import (
    expression_basis,
    makehuman_template,
    shape_basis,
)
from roop_3d.reconstruction.pose_solver import (
    HeadPose6DoF,
    default_intrinsics_matrix,
    rotation_to_standard,
    solve_head_pose_6dof,
)
from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation as UVViewObservation
from roop_3d.reconstruction.uv_completion import FullHeadUVCompleter
from roop_3d.reconstruction.rbf_deform import weld_ids, welded_vertex_normals
from roop_3d.rendering.cpu_perspective_raster import rasterize_layer
from roop_3d.rendering.intrinsic_texture import IntrinsicTextureDecomposer
from roop_3d.rendering.skin_calibration import linear_to_srgb, srgb_to_linear
from roop_3d.reconstruction.photometry import fit_sh
from roop_3d.relighting_3d import SphericalHarmonicsLighting
from roop_3d.temporal import TrackTemporalStabilizer


@dataclass(frozen=True, slots=True)
class HeadLayer:
    """One complete perspective-rendered head layer in target-frame coordinates."""

    image: np.ndarray                    # uint8 HxWx3 BGR
    coverage: np.ndarray                 # float32 HxW, geometry coverage with a 1.5 px edge ramp
    normals: np.ndarray                  # float32 HxWx3 camera normals, +z toward the viewer
    projected_vertices: np.ndarray       # float32 Nx2, every seam-split MakeHuman vertex
    faces: np.ndarray                    # int32 Mx3, complete MakeHuman triangle topology
    depth: np.ndarray                    # float32 HxW camera-space Z in metres; inf outside the head
    diagnostics: dict[str, Any]
    eye_mask: np.ndarray | None = None                 # float32 HxW, 1 where the front surface is an eyeball (pre-feather)
    eye_projected_vertices: np.ndarray | None = None   # float32 Gx2, both eyeball surfaces
    eye_faces: np.ndarray | None = None                # int32 Fx3, eyeball triangles indexing eye_projected_vertices


class VolumetricHeadRenderer:
    """Jointly fit one MakeHuman identity and render it through a perspective Z-buffer.

    ``catalog`` is the only model-weight resolver used by semantic parsing and
    visible-face masking. The MakeHuman CC0 mesh itself is a verified local
    geometry asset. Source views are decoded from ``FaceSetDocument`` without
    changing the archive or any file next to the repository.
    """

    SOURCE_FOV_DEGREES = 45.0
    MAX_DENSE_REPROJECTION_FRACTION = 0.08
    MIN_DENSE_REPROJECTION_PX = 5.0
    SILHOUETTE_FEATHER_PX = 1.5
    FACE_CLASSES = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
    FORBIDDEN_CLASSES = (0, 6, 9, 14, 15, 16, 17, 18)
    EXPRESSION_AXES = ("mouth_open", "mouth_corner_puller", "mouth_pursing", "brows_up", "eyes_closed")
    BOUNDARY_ERODE_FRACTION = 0.018       # observation support erosion, as a fraction of the source face size
    TONE_DECAY_FRACTION = 0.12            # distance (fraction of the atlas) over which completed skin relaxes to the tone
    DELIGHT_MAX_LOG_SHADING = 0.75        # |ln shading| bound: removes directional light, never albedo features

    def __init__(self, catalog, config, document: FaceSetDocument, *, texture_size: int = 1024,
                 restoration_weight: float = 0.35):
        if isinstance(texture_size, bool) or not isinstance(texture_size, (int, np.integer)) or not 64 <= texture_size <= 4096:
            raise DomainError("INVALID_UV_SIZE", "texture_size must be an integer in [64, 4096]", 422)
        if isinstance(restoration_weight, (bool, np.bool_)):
            raise DomainError("INVALID_RESTORATION_WEIGHT", "restoration_weight must be numeric and within [0, 1]", 422)
        try:
            restoration_weight = float(restoration_weight)
        except (TypeError, ValueError) as error:
            raise DomainError("INVALID_RESTORATION_WEIGHT", "restoration_weight must be numeric and within [0, 1]", 422) from error
        if not np.isfinite(restoration_weight) or not 0.0 <= restoration_weight <= 1.0:
            raise DomainError("INVALID_RESTORATION_WEIGHT", "restoration_weight must be finite and within [0, 1]", 422)
        if not getattr(document, "views", None):
            raise DomainError("FACESET_EMPTY", "A full-head renderer needs at least one source view", 422)
        self.catalog, self.config, self.document = catalog, config, document
        self.texture_size = int(texture_size)
        self.restoration_weight = restoration_weight
        self.last_layer: HeadLayer | None = None
        self._released = False
        self._parser = None
        self._visible_face = None
        self._texture_restorer: Callable | None = None
        self._texture_restoration = {"status": "not_checked", "weight": self.restoration_weight}
        self._temporal = TrackTemporalStabilizer()
        self._source_fov = self._configured_fov()

        try:
            self._load_geometry_and_fit_identity()
            self._build_eye_geometry()
            self._load_semantic_models()
            self._fit_source_cameras()
            self._synthesize_texture()
            self._build_eye_textures()
            self._install_default_restorer()
        except Exception:
            self.release()
            raise

    def _configured_fov(self) -> float:
        value = self.SOURCE_FOV_DEGREES
        for name in ("source_fov_degrees", "camera_fov_degrees", "fov_degrees"):
            if isinstance(self.config, Mapping) and name in self.config:
                value = self.config[name]
                break
            if hasattr(self.config, name):
                value = getattr(self.config, name)
                break
        if isinstance(value, (bool, np.bool_)):
            raise DomainError("INVALID_FOV", "Configured source camera FOV must be numeric", 422)
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise DomainError("INVALID_FOV", "Configured source camera FOV must be numeric", 422) from error
        if not np.isfinite(value) or not 1.0 <= value <= 170.0:
            raise DomainError("INVALID_FOV", f"Configured source camera FOV is invalid: {value}", 422)
        return value

    @staticmethod
    def _decode_view(view: FaceView) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise DomainError("FACESET_IMAGE_INVALID", f"FaceSet view {view.index} cannot be decoded as BGR", 422)
        return image

    def _load_geometry_and_fit_identity(self) -> None:
        self._model = MakeHumanHeadModel.load()
        self._template = makehuman_template()
        from roop_3d.reconstruction.makehuman_landmarks import correspondence_report
        self._landmark_correspondence = correspondence_report(self._template)
        self._shape_basis = shape_basis()
        self._expression_basis = expression_basis()
        self._landmarks5 = np.asarray(self._template.landmark_vertices["5pt"], np.int64)
        self._landmarks68 = np.asarray(self._template.landmark_vertices["68pt"], np.int64)
        if (len(self._model.base_mesh.vertices) != 4289
                or self._shape_basis.shape[:2] != self._template.vertices.shape
                or self._expression_basis.shape[:2] != self._template.vertices.shape
                or self._landmarks5.shape != (5,) or self._landmarks68.shape != (68,)
                or len(np.unique(self._landmarks68)) != 68):
            raise DomainError("MAKEHUMAN_TOPOLOGY_INVALID",
                              "Verified MakeHuman head topology or dense landmark correspondence is inconsistent", 422)

        source_views: list[tuple[FaceView, np.ndarray, np.ndarray]] = []
        excluded: list[dict[str, Any]] = []
        fit_observations: list[MakeHumanViewObservation] = []
        for view in self.document.views:
            if view.is_synthetic:
                excluded.append({"view_index": int(view.index), "reason": "synthetic_view_not_used_as_source_evidence"})
                continue
            if not view.landmarks_trusted:
                excluded.append({"view_index": int(view.index), "reason": "untrusted_landmarks"})
                continue
            image = self._decode_view(view)
            points = np.asarray(view.landmarks, np.float64)
            if points.shape != (5, 2) or not np.isfinite(points).all():
                dense68 = None if view.landmarks_68 is None else np.asarray(view.landmarks_68, np.float64)
                if dense68 is None or dense68.shape != (68, 2) or not np.isfinite(dense68).all():
                    excluded.append({"view_index": int(view.index), "reason": "no_finite_trusted_5_or_68_point_landmarks"})
                    continue
                from roop_3d.reconstruction.multiview_fit import five_from_68
                points = five_from_68(dense68)
            height, width = image.shape[:2]
            if (np.any(points[:, 0] < -0.05 * width) or np.any(points[:, 0] > 1.05 * width)
                    or np.any(points[:, 1] < -0.05 * height) or np.any(points[:, 1] > 1.05 * height)):
                excluded.append({"view_index": int(view.index), "reason": "landmarks_outside_source_image"})
                continue
            source_views.append((view, image, points))
            fit_observations.append(MakeHumanViewObservation(
                landmarks=points, width=width, height=height, name=str(view.index),
                weight=max(float(view.quality.score), 0.05), reference_yaw=float(view.yaw)))
        if not fit_observations:
            raise DomainError("NO_TRUSTED_FACESET_VIEWS", "No real FaceSet view has usable trusted landmarks", 422)

        fit = fit_multiview(self._model, fit_observations, fov_degrees=self._source_fov, holdout=False)
        beta = np.asarray(fit.shape_params, np.float64).reshape(-1)
        if beta.shape != (self._model.num_shape,) or not np.isfinite(beta).all():
            raise DomainError("MAKEHUMAN_IDENTITY_FIT_INVALID", "Joint MakeHuman identity fit returned invalid coefficients", 422)
        self._shape_params = beta.astype(np.float32)
        vertices = self._template.vertices + np.einsum("vck,k->vc", self._shape_basis, self._shape_params).astype(np.float32)
        if not np.isfinite(vertices).all():
            raise DomainError("MAKEHUMAN_IDENTITY_FIT_INVALID", "Fitted MakeHuman vertices are non-finite", 422)

        self._atlas_uv = self._template.uvs.copy().astype(np.float32)
        self._mesh = Mesh3D(vertices, self._template.faces.copy(), uvs=self._atlas_uv.copy())
        self._mesh.landmark_vertices = {"5pt": self._landmarks5.copy(), "68pt": self._landmarks68.copy()}
        from roop_3d.reconstruction.dense_surface_fit import fit_dense_surface
        surface = fit_dense_surface(self._mesh, source_views, fov_degrees=self._source_fov)
        self._mesh.vertices = surface.vertices
        self._mesh.normals = welded_vertex_normals(surface.vertices, self._mesh.faces)
        self._source_surface_cameras = surface.cameras
        # Structural validation uses the actual fitted surface after welding UV seams.
        from dataclasses import asdict
        from roop_3d.core.true3d_guard import close_open_boundaries, validate_head_mesh, validate_render_config
        _, representatives, welded = np.unique(np.round(surface.vertices, 6), axis=0,
                                               return_index=True, return_inverse=True)
        closed_vertices, closed_faces = close_open_boundaries(
            surface.vertices[representatives], welded[self._mesh.faces])
        self._mesh_validation = asdict(validate_head_mesh(
            closed_vertices, closed_faces, face_direction=(0, 0, 1), require_parts=False))
        validate_render_config({"projection": "perspective", "rasterizer": "z_buffer",
                                "shader": "perspective_sh_lambertian"})
        self._mesh.base_vertex_indices = np.asarray(self._template.base_vertex_indices, np.int32).copy()
        self._weld = weld_ids(self._mesh.vertices)
        self._base_normals = welded_vertex_normals(self._mesh.vertices, self._mesh.faces, self._weld)
        self._mesh.faces.setflags(write=False)
        self._mesh.uvs.setflags(write=False)
        self._mesh.vertices.setflags(write=False)
        self._mesh.landmark_vertices["5pt"].setflags(write=False)
        self._mesh.landmark_vertices["68pt"].setflags(write=False)
        self._source_fit = fit
        self._source_views = source_views
        self._source_excluded = excluded
        self._identity_diagnostics = {
            "fit_method": "joint_makehuman_multiview_map_and_dense68_perspective_surface_fit",
            "dense_identity_fit": surface.diagnostics,
            "fit_converged": bool(fit.converged),
            "fit_iterations": int(fit.iterations),
            "fit_cost": float(fit.cost),
            "shape_coefficients": [float(value) for value in beta],
            "shape_axes": self._model.shape_names,
            "notes": list(fit.notes),
            "dense68_vertex_correspondence": self._landmark_correspondence,
            "source_views_excluded": excluded,
            "base_topology_vertices": int(len(self._model.base_mesh.vertices)),
            "render_vertices_after_uv_seam_split": int(len(self._mesh.vertices)),
            "template_geometry_is_prior": True,
        }


    def _load_semantic_models(self) -> None:
        from roop_3d.compositor.masks import BisenetFaceParser
        from roop_3d.compositor.occlusion_model import VisibleFaceNet
        self._parser = BisenetFaceParser(self.catalog, self.config, normalization="imagenet")
        self._parser.cache_enabled = False
        self._visible_face = VisibleFaceNet(self.catalog, self.config, model_name="xseg.onnx")

    @staticmethod
    def _face_value(face: Any, name: str, default=None):
        if isinstance(face, Mapping):
            return face.get(name, default)
        return getattr(face, name, default)

    @classmethod
    def _expanded_crop(cls, image: np.ndarray, box: Any, landmarks: np.ndarray | None) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        if box is not None:
            bounds = np.asarray(box, np.float64).reshape(-1)
            if bounds.shape != (4,) or not np.isfinite(bounds).all():
                raise DomainError("HEAD_FACE_BOX_INVALID", "Face box must contain four finite image coordinates", 422)
            x0, y0, x1, y1 = map(float, bounds)
        elif landmarks is not None and len(landmarks):
            lo, hi = np.min(landmarks, axis=0), np.max(landmarks, axis=0)
            x0, y0, x1, y1 = float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
        else:
            raise DomainError("HEAD_FACE_BOUNDS_MISSING", "Semantic masking needs a measured face box or landmarks", 422)
        bw, bh = x1 - x0, y1 - y0
        if not np.isfinite((bw, bh)).all() or bw < 2.0 or bh < 2.0:
            raise DomainError("HEAD_FACE_BOX_INVALID", "Face box has no positive image area", 422)
        px, py_top, py_bottom = 0.35 * bw, 0.65 * bh, 0.28 * bh
        left = max(0, int(math.floor(x0 - px)))
        top = max(0, int(math.floor(y0 - py_top)))
        right = min(width, int(math.ceil(x1 + px)))
        bottom = min(height, int(math.ceil(y1 + py_bottom)))
        if right - left < 2 or bottom - top < 2:
            raise DomainError("HEAD_FACE_BOX_INVALID", "Expanded face crop falls outside the image", 422)
        return left, top, right, bottom

    def _semantic_masks(self, image: np.ndarray, box: Any, landmarks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        allowed, skin, _, stats = self._semantic_planes(image, box, landmarks)
        return allowed, skin, stats

    def _semantic_planes(self, image: np.ndarray, box: Any, landmarks: np.ndarray | None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Full-frame allowed-face, skin and BiSeNet eyeball (classes 4+5) planes, plus statistics."""
        x0, y0, x1, y1 = self._expanded_crop(image, box, landmarks)
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        probabilities = np.asarray(self._parser.predict(crop), np.float32)
        visible = np.asarray(self._visible_face.predict(crop), np.float32)
        if (probabilities.ndim != 3 or probabilities.shape[:2] != crop.shape[:2]
                or probabilities.shape[2] < 19 or not np.isfinite(probabilities).all()
                or np.any(probabilities < -1e-4) or np.any(probabilities > 1.0001)
                or visible.shape != crop.shape[:2] or not np.isfinite(visible).all()
                or np.any(visible < -1e-4) or np.any(visible > 1.0001)):
            raise DomainError("STAGE4_SEMANTIC_TENSOR_INVALID", "Parser or visible-face network returned invalid masks", 503)
        from roop_3d.compositor.stage4 import semantic_face_masks
        face_probability, forbidden, skin_probability, ears = semantic_face_masks(probabilities)
        visible = np.where(ears, 1.0, np.clip(visible, 0.0, 1.0))
        allowed = face_probability * (1.0 - forbidden) * visible
        skin = skin_probability * (1.0 - forbidden) * visible
        eyes = np.clip(probabilities[..., 4] + probabilities[..., 5], 0.0, 1.0) * (1.0 - forbidden)
        allowed_full = np.zeros(image.shape[:2], np.float32)
        skin_full = np.zeros(image.shape[:2], np.float32)
        eyes_full = np.zeros(image.shape[:2], np.float32)
        allowed_full[y0:y1, x0:x1] = np.clip(allowed, 0.0, 1.0)
        skin_full[y0:y1, x0:x1] = np.clip(skin, 0.0, 1.0)
        eyes_full[y0:y1, x0:x1] = eyes
        stats = {
            "bisenet_class_mapping": "CelebAMask-HQ 19-class partition, including ears",
            "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "semantic_face_fraction": float(np.mean(face_probability > 0.5)),
            "forbidden_fraction": float(np.mean(forbidden > 0.5)),
            "visible_face_fraction": float(np.mean(visible > 0.5)),
            "eye_class_pixels": int(np.count_nonzero(eyes_full > 0.5)),
        }
        return allowed_full, skin_full, eyes_full, stats

    @staticmethod
    def _camera(rotation: np.ndarray, translation: np.ndarray, intrinsics: CameraIntrinsics) -> PerspectiveCamera:
        matrix = np.asarray(rotation, np.float64)
        offset = np.asarray(translation, np.float64).reshape(3)
        return PerspectiveCamera(
            intrinsics=intrinsics,
            rotation=tuple(tuple(float(value) for value in row) for row in matrix),
            translation=tuple(float(value) for value in offset),
            mesh_z_toward_viewer=True,
        )

    def _fit_source_cameras(self) -> None:
        cameras: dict[int, PerspectiveCamera] = {}
        view_diagnostics: list[dict[str, Any]] = []
        for (view, image, points5), fitted in zip(self._source_views, self._source_fit.views):
            height, width = image.shape[:2]
            intrinsics = CameraIntrinsics.from_fov(self._source_fov, width, height)
            camera_pose = solve_head_pose_6dof(
                points5, intrinsics=intrinsics,
                model_points=self._mesh.vertices[self._landmarks5].astype(np.float64))
            pose_source = "fitted_identity_5point_pnp"
            dense_reason = "no_trusted_68_point_landmarks"
            dense = None if view.landmarks_68 is None else np.asarray(view.landmarks_68, np.float64)
            if dense is not None and view.landmarks_trusted and dense.shape == (68, 2) and np.isfinite(dense).all():
                try:
                    solved = solve_head_pose_6dof(
                        dense, intrinsics=intrinsics,
                        model_points=self._mesh.vertices[self._landmarks68].astype(np.float64))
                    extent = float(np.linalg.norm(dense.max(axis=0) - dense.min(axis=0)))
                    max_rms = max(self.MIN_DENSE_REPROJECTION_PX, self.MAX_DENSE_REPROJECTION_FRACTION * extent)
                    if solved.reproj_rms_px > max_rms:
                        dense_reason = "dense_pnp_reprojection_too_high"
                    else:
                        camera_pose = solved
                        pose_source = "makehuman_dense68_pnp"
                        dense_reason = "accepted"
                except DomainError as error:
                    dense_reason = f"dense_pnp_failed:{error.code}"
            elif view.landmarks_106 is not None:
                dense_reason = "106_point_layout_has_no_verified_makehuman_vertex_map"
            source_camera = self._camera(camera_pose.R, camera_pose.t, intrinsics)
            reprojection = float(camera_pose.reproj_rms_px)
            if int(view.index) in self._source_surface_cameras:
                rotation, translation, reprojection = self._source_surface_cameras[int(view.index)]
                source_camera = self._camera(rotation, translation, intrinsics)
                pose_source = "joint_dense68_surface_bundle"
            cameras[int(view.index)] = source_camera
            view_diagnostics.append({
                "view_index": int(view.index),
                "pose_source": pose_source,
                "dense_pose_status": dense_reason,
                "reprojection_rms_px": reprojection,
                "landmark_layout": "68pt" if "dense68" in pose_source else "5pt",
                "shared_identity_view_rejected": bool(fitted.rejected),
                "source_image_size": [int(width), int(height)],
                "camera": {
                    "rotation_cv": [list(row) for row in source_camera.rotation],
                    "translation_m": list(source_camera.translation),
                    "intrinsics": {name: float(getattr(intrinsics, name)) for name in ("fx", "fy", "cx", "cy")},
                },
            })
        self._source_cameras = cameras
        self._source_view_diagnostics = view_diagnostics

    def _synthesize_texture(self) -> None:
        uv_mesh = Mesh3D(self._mesh.vertices, self._mesh.faces, uvs=self._atlas_uv, normals=self._base_normals)
        uv_mesh.landmark_vertices = self._mesh.landmark_vertices
        synthesizer = UVSynthesizer(
            uv_mesh, uv_size=self.texture_size, max_interpolation_fraction=0,
            blend_mode="weighted", visibility_backend="cpu",
            intrinsic_texture_decomposer=IntrinsicTextureDecomposer(),
        )
        view_textures: list[np.ndarray] = []
        view_weights: list[np.ndarray] = []
        per_view: list[dict[str, Any]] = []
        self._source_eye_planes = {}
        socket_mask = self._socket_uv_mask()
        source_diagnostics = {int(item["view_index"]): item for item in self._source_view_diagnostics}
        for view, image, _ in self._source_views:
            camera_diagnostics = source_diagnostics[int(view.index)]
            landmark_layout = str(camera_diagnostics["landmark_layout"])
            reprojection_rms = float(camera_diagnostics["reprojection_rms_px"])
            registration_confidence = 1.0
            if landmark_layout == "68pt":
                # Dense-fit residual is measured camera-registration error. Keep these texels
                # observed, but let better-aligned views win where their UV support overlaps.
                registration_confidence = min(
                    1.0, self.MIN_DENSE_REPROJECTION_PX / max(reprojection_rms, self.MIN_DENSE_REPROJECTION_PX))
            camera = self._source_cameras.get(int(view.index))
            if camera is None:
                raise DomainError("FACESET_CAMERA_MISSING", f"Source view {view.index} has no fitted perspective camera", 422)
            allowed, _, eye_plane, mask_stats = self._semantic_planes(
                image, view.bbox, np.asarray(view.landmarks, np.float64))
            # The eyeball is not skin: BiSeNet eye pixels (dilated to catch the lid margin / lash line) never enter
            # the skin atlas, so the source photograph cannot ghost an iris or sclera onto the lid surface.
            face_width = (float(np.ptp(np.asarray(view.landmarks, np.float64)[:, 0])) if view.bbox is None
                          else float(np.asarray(view.bbox, np.float64).reshape(-1)[2] - np.asarray(view.bbox, np.float64).reshape(-1)[0]))
            grow = max(2, int(round(0.012 * face_width)))
            eye_block = cv2.dilate((eye_plane > 0.35).astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
            allowed = allowed * (1.0 - eye_block.astype(np.float32))
            # Parser/XSeg boundaries sit a few pixels off the true skin edge (and the fitted silhouette is off by the camera
            # residual): pixels adjacent to hair, neck, background or occluders carry their shadow and colour, so the
            # observation support is eroded away from every excluded region.
            extent = (float(np.ptp(np.asarray(view.landmarks, np.float64)[:, 1])) if view.bbox is None
                      else float(np.asarray(view.bbox, np.float64).reshape(-1)[3] - np.asarray(view.bbox, np.float64).reshape(-1)[1]))
            shrink = max(2, int(round(self.BOUNDARY_ERODE_FRACTION * max(face_width, extent))))
            allowed = cv2.erode(allowed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1)))
            self._source_eye_planes[int(view.index)] = eye_plane
            mask_stats = {**mask_stats, "eye_exclusion_dilation_px": int(grow),
                          "eye_excluded_pixels": int(np.count_nonzero(eye_block)),
                          "boundary_erosion_px": int(shrink)}
            observation = UVViewObservation(
                image=image,
                camera=camera,
                sharpness=max(float(view.quality.sharpness), 0.1),
                occlusion_mask=np.clip(np.rint(allowed * 255.0), 0, 255).astype(np.uint8),
                view_quality=float(np.clip(view.quality.score, 0.0, 1.0)),
                view_index=int(view.index),
            )
            synthesized = synthesizer.synthesize([observation])
            observed = synthesized.confidence_mask == int(ConfidenceState.OBSERVED)
            score = (np.ones(observed.shape, np.float32) if synthesized.confidence_score is None
                     else np.asarray(synthesized.confidence_score, np.float32))
            weights = np.where(observed, score * registration_confidence, 0.0).astype(np.float32)
            texture_map, delight_stats = self._delight_view(
                np.asarray(synthesized.texture_map, np.uint8), weights, synthesizer.norm_map)
            view_textures.append(texture_map)
            view_weights.append(weights)
            mask_stats = {**mask_stats, "delighting": delight_stats}
            per_view.append({
                "view_index": int(view.index),
                "observed_uv_ratio": float(synthesized.observed_ratio),
                "observed_texels": int(observed.sum()),
                "semantic_mask": mask_stats,
                "landmark_layout": landmark_layout,
                "camera_reprojection_rms_px": reprojection_rms,
                "registration_confidence": registration_confidence,
            })
        colors = np.stack(view_textures).astype(np.float32)
        weights = np.stack(view_weights).astype(np.float32)
        flat_gains = UVSynthesizer._solve_gains(colors.reshape(len(colors), -1, 3), weights.reshape(len(weights), -1))
        colors = np.clip(colors * flat_gains[:, None, None, :].astype(np.float32), 0, 255)
        for item, gain in zip(per_view, flat_gains):
            item["cross_view_exposure_gain_bgr"] = [float(value) for value in gain]
        # Specular glints and cast-shadow spots survive the SH delighting and, once completed, spread through the
        # harmonic extension: bound each observed texel's luminance to a soft band around its broad neighbourhood.
        sigma = max(2.0, 0.025 * self.texture_size)
        weights[:, socket_mask] = 0.0
        clamped_texels = 0
        for index in range(len(colors)):
            support = (weights[index] > 0).astype(np.float32)
            luma = colors[index] @ np.array((0.114, 0.587, 0.299), np.float32)
            local = cv2.GaussianBlur(luma * support, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(support, (0, 0), sigma), 1e-3)
            ratio = (luma + 3.0) / (local + 3.0)
            scale = np.where(support > 0, np.clip(ratio, 0.7, 1.24) / np.maximum(ratio, 1e-6), 1.0).astype(np.float32)
            clamped_texels += int(np.count_nonzero(np.abs(scale - 1.0) > 0.01))
            colors[index] = np.clip(colors[index] * scale[..., None], 0, 255)
        for item in per_view:
            item["luminance_outlier_band"] = [0.7, 1.24]
        weights[:, socket_mask] = 0.0                    # the orbital cavity is anatomy, never a skin observation
        footprint = synthesizer.face_texel_mask.astype(bool)
        if not np.any(weights > 0):
            raise DomainError("UV_SYNTHESIS_NO_OBSERVATIONS",
                              "No source view contributed semantically visible MakeHuman UV texels", 422)
        scale = self.texture_size / 1024.0
        completion = FullHeadUVCompleter(
            poisson_seams=True, tone_decay_px=self.TONE_DECAY_FRACTION * self.texture_size,
            feather_px=6.0 * scale, min_fragment_texels=int(0.0008 * self.texture_size ** 2)).complete(
            colors, weights, footprint, synthesizer.pos_map,
            mirror_x=float(np.median(self._mesh.vertices[:, 0])),
        )
        tone = np.asarray(completion.stats["skin_tone_bgr"], np.float32)
        # Cavity / caruncle prior: the shadowed orbital wall is a darker, redder version of the subject's skin tone.
        cavity = np.clip(tone * np.array((0.72, 0.55, 0.62), np.float32), 0, 255)
        soft = cv2.GaussianBlur(socket_mask.astype(np.float32), (0, 0), 1.2)[..., None]
        base = completion.texture[..., :3].astype(np.float32)
        base = base * (1.0 - soft) + cavity[None, None, :] * soft
        self._base_texture = np.ascontiguousarray(np.clip(np.rint(base), 0, 255), dtype=np.uint8)
        self._texture = self._base_texture
        self._texture_coverage = np.ascontiguousarray(completion.texture[..., 3].astype(np.float32) / 255.0)
        self._uv_completion_source = np.asarray(completion.source, np.uint8)
        self._socket_mask = socket_mask
        observed_any = (weights.max(axis=0) > 0) & footprint
        surface = synthesizer.pos_map
        front = observed_any & (surface[..., 2] > -0.02) & (np.abs(surface[..., 0]) < 0.07)
        brow_y = float(np.median(self._mesh.vertices[self._landmarks68[17:27], 1]))
        midline = (np.abs(self._mesh.vertices[:, 0]) < 0.012) & (self._mesh.vertices[:, 2] > -0.06)
        skull_top_y = float(self._mesh.vertices[midline, 1].min())
        observed_top_y = float(np.percentile(surface[..., 1][front], 0.5)) if front.any() else brow_y
        forehead = {
            "brow_line_to_scalp_midline_mm": 1000.0 * (brow_y - skull_top_y),
            "observed_above_brow_mm": 1000.0 * max(brow_y - observed_top_y, 0.0),
            "observed_fraction_of_forehead_height": float(max(brow_y - observed_top_y, 0.0) / max(brow_y - skull_top_y, 1e-6)),
            "views_with_texels_above_brow_plus_10mm": int(sum(
                bool(np.any((weights[i] > 0) & front & (surface[..., 1] < brow_y - 0.010))) for i in range(len(weights)))),
            "cause_of_missing_forehead": "BiSeNet hair class and XSeg exclude the hair-covered forehead in every source view; "
                                         "unobserved forehead is symmetry/harmonic skin-tone completion",
        }
        self._uv_diagnostics = {
            "forehead": forehead,
            "luminance_outlier_texels_clamped": clamped_texels,
            **completion.stats,
            "observed_ratio": float(completion.stats["observed_ratio"]),
            "symmetry_ratio": float(completion.stats["symmetry_ratio"]),
            "harmonic_ratio": float(completion.stats["harmonic_ratio"]),
            "view_count": int(len(view_textures)),
            "views_with_observed_uv": int(sum(item["observed_texels"] > 0 for item in per_view)),
            "per_view": per_view,
            "semantic_exclusion": "BiSeNet 19-class face/ears intersected with XSeg; hair, glasses, neck and background forbidden",
            "unseen_texels_are_observations": False,
            "eye_socket_cavity_texels": int(np.count_nonzero(socket_mask)),
            "eye_socket_cavity_source": "prior: shadowed skin-tone cavity; excluded from observation and restoration",
        }

    def _delight_view(self, texture: np.ndarray, weights: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Divide one view's observed texels by a robust second-order SH shading fitted in the head frame.

        The source photographs carry their own directional light (a shadowed jaw underside, a lit forehead). Baking that into
        the albedo and relighting it again at render time double-darkens every surface that faces away from the source key
        light. A 9-term SH regression of log-luminance on the UV surface normals, with a Huber IRLS so lips, brows and
        freckles stay albedo, estimates and removes only the smooth directional component; the DC term (brightness) is kept.
        """
        valid = weights > 0.05
        count = int(valid.sum())
        if count < 400:
            return texture, {"status": "skipped_too_few_texels", "texels": count}
        linear = srgb_to_linear(texture)
        luma = linear @ np.array((0.114, 0.587, 0.299), np.float32)
        valid &= luma > 0.002
        indices = np.flatnonzero(valid.ravel())
        stride = max(1, len(indices) // 30000)
        sample = indices[::stride]
        basis = SphericalHarmonicsLighting.evaluate_basis(normals.reshape(-1, 3)[sample]).astype(np.float64)
        target = np.log(luma.reshape(-1)[sample].astype(np.float64))
        base = np.clip(weights.reshape(-1)[sample].astype(np.float64), 0.0, 1.0)
        ridge = np.diag([1e-9, 4e-3, 4e-3, 4e-3, 4e-2, 4e-2, 4e-2, 4e-2, 4e-2]) * len(sample)
        w = base.copy()
        coefficients = np.zeros(9)
        for _ in range(6):
            coefficients = np.linalg.solve(basis.T @ (basis * w[:, None]) + ridge, basis.T @ (target * w))
            residual = np.abs(target - basis @ coefficients)
            scale = max(float(np.median(residual)) * 1.4826, 0.04)
            w = base * np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-6))
        directional = coefficients.copy()
        directional[0] = 0.0
        all_basis = SphericalHarmonicsLighting.evaluate_basis(normals.reshape(-1, 3)).astype(np.float32)
        log_shading = np.clip(all_basis @ directional.astype(np.float32), -self.DELIGHT_MAX_LOG_SHADING,
                              self.DELIGHT_MAX_LOG_SHADING).reshape(luma.shape)
        log_shading -= float(np.median(log_shading[valid]))        # keep the observed brightness
        flat = linear / np.exp(log_shading)[..., None]
        flat[~(weights > 0)] = linear[~(weights > 0)]
        result = np.clip(np.rint(linear_to_srgb(np.clip(flat, 0.0, 1.0)) * 255.0), 0, 255).astype(np.uint8)
        result[~(weights > 0)] = texture[~(weights > 0)]
        return result, {"status": "applied", "texels_fitted": int(len(sample)),
                        "log_shading_range": [float(log_shading[valid].min()), float(log_shading[valid].max())],
                        "log_shading_std": float(log_shading[valid].std()),
                        "sh_coefficients_log_luma": [float(v) for v in coefficients]}

    def _socket_uv_mask(self) -> np.ndarray:
        """UV texels of the two orbital-cavity islands (dilated by two texels so bilinear taps never reach skin)."""
        size = self.texture_size
        pixels = uv_to_pixel(self._atlas_uv, size, size, origin=UVOrigin.TOP_LEFT)
        mask = np.zeros((size, size), np.uint8)
        for triangle in self._mesh.faces[self._socket_faces]:
            corners = np.rint(pixels[triangle].astype(np.float64) * 16.0).astype(np.int32)
            cv2.fillConvexPoly(mask, corners, 255, lineType=cv2.LINE_8, shift=4)
        return cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0

    def _build_eye_geometry(self) -> None:
        """Locate the two orbital cavities and rims and fit an anatomical globe into each from the lid-margin loop."""
        from roop_3d.reconstruction.eye_globes import build_eye_globes
        from roop_3d.reconstruction.makehuman_landmarks import eye_apertures
        faces = np.asarray(self._mesh.faces, np.int64)
        vertices = np.asarray(self._mesh.vertices, np.float64)
        apertures = eye_apertures(vertices, faces)
        bound = np.concatenate([item["landmarks"] for item in apertures])
        if not np.array_equal(bound, self._landmarks68[36:48]):
            raise DomainError("EYE_LANDMARK_BINDING_INVALID",
                              "Dense eye landmarks 36-47 are not bound to the anatomical lid-margin loop", 422)
        sockets = np.concatenate([item["socket"] for item in apertures])
        self._socket_vertices = np.sort(sockets).astype(np.int64)
        self._socket_faces = np.isin(faces[:, 0], self._socket_vertices)
        self._eye_globes = build_eye_globes(vertices, faces, apertures)
        self._eye_diagnostics_base = {**self._eye_globes.diagnostics, "socket_faces": int(self._socket_faces.sum()),
                                      "socket_vertices": int(len(self._socket_vertices)),
                                      "landmark_binding": "dense68 points 36-47 are the anatomical lid-margin loop vertices"}

    def _scene_depth(self, camera: PerspectiveCamera, shape: tuple[int, int]) -> np.ndarray:
        """Camera-space Z-buffer of the skin mesh and both globes, for depth-tested eye measurement."""
        globes = self._eye_globes
        vertices = np.concatenate((self._mesh.vertices, globes.vertices)).astype(np.float64)
        camera_vertices = camera.camera_points(vertices)
        projected = camera.project(vertices)
        normals = camera.rotate_normals(np.concatenate((self._base_normals, globes.normals)))
        faces = np.concatenate((self._mesh.faces, globes.faces + len(self._mesh.vertices)))[:, ::-1]
        buffers = rasterize_layer(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32), projected, camera_vertices,
                                  np.zeros((len(vertices), 2), np.float32), faces, normals, shape, roi=None)
        return buffers.depth

    def _build_eye_textures(self) -> None:
        """Texture the globes from depth-tested source-view projections, falling back to a labelled anatomical prior."""
        from roop_3d.reconstruction.eye_globes import compose_eye_tiles, sample_measured_tiles
        from roop_3d.reconstruction.uv_synthesizer import remap_points
        views = []
        for view, image, _ in self._source_views:
            camera = self._source_cameras[int(view.index)]
            views.append({"camera": camera, "image": image, "eye": self._source_eye_planes[int(view.index)],
                          "scene_depth": self._scene_depth(camera, image.shape[:2]),
                          "quality": float(np.clip(view.quality.score, 0.0, 1.0)), "view_index": int(view.index)})
        measured, weight, per_view = sample_measured_tiles(self._eye_globes, views, remap_points)
        self._eye_tiles, texture_stats = compose_eye_tiles(self._eye_globes, measured, weight)
        self._eye_diagnostics = {**self._eye_diagnostics_base, "texture": texture_stats, "per_view_measurement": per_view,
                                 "limitations": [
                                     "Gaze is the head-forward prior; the target person's gaze direction is not measured.",
                                     "Source eye photographs are ~10-30 px wide: measured sclera/iris colour is real, fine iris "
                                     "structure comes from the labelled procedural prior."]}
        self._assemble_atlas()

    def _assemble_atlas(self) -> None:
        """Stack the (possibly restored) skin atlas above the two eye tiles so one Z-buffer pass textures everything."""
        from roop_3d.reconstruction.eye_globes import TILE_SIZE
        size = self.texture_size
        width = max(size, 2 * TILE_SIZE)
        atlas = np.zeros((size + TILE_SIZE, width, 3), np.uint8)
        coverage = np.zeros((size + TILE_SIZE, width), np.float32)
        atlas[:size, :size] = self._texture
        coverage[:size, :size] = self._texture_coverage
        for eye in range(2):
            atlas[size:, eye * TILE_SIZE:(eye + 1) * TILE_SIZE] = self._eye_tiles[eye]
            coverage[size:, eye * TILE_SIZE:(eye + 1) * TILE_SIZE] = 1.0
        globes = self._eye_globes
        offset = np.stack((globes.eye_of_vertex.astype(np.float32) * TILE_SIZE,
                           np.full(len(globes.eye_of_vertex), float(size), np.float32)), axis=1)
        self._raster_atlas = atlas
        self._raster_coverage = coverage
        self._skin_uv_pixels = uv_to_pixel(self._atlas_uv, size, size, origin=UVOrigin.TOP_LEFT)
        self._eye_uv_pixels = (globes.tile_pixels + offset).astype(np.float32)

    def _install_default_restorer(self) -> None:
        if self.restoration_weight == 0.0:
            self._texture_restoration = {"status": "disabled", "weight": 0.0}
            return
        from roop_3d.enhancers.stage4_skin import restore_uv_texture
        self.set_texture_restorer(restore_uv_texture)

    def set_texture_restorer(self, restorer: Callable) -> None:
        """Install a canonical-UV restorer; it runs before any perspective projection."""
        if self._released:
            raise DomainError("RENDERER_RELEASED", "Cannot configure a released head renderer", 409)
        if not callable(restorer):
            raise DomainError("STAGE4_RESTORER_INVALID", "Texture restorer must be callable", 422)
        if self.restoration_weight == 0.0:
            self._texture_restorer = restorer
            self._texture = self._base_texture
            self._texture_restoration = {"status": "disabled", "weight": 0.0}
            return
        valid = self._texture_coverage * (~self._socket_mask)
        restored = restorer(self._base_texture, self.catalog, self.config, self.restoration_weight,
                            mesh=self._mesh, valid_mask=valid)
        texture = np.asarray(restored)
        if (texture.shape != self._base_texture.shape or texture.dtype != np.uint8
                or not np.isfinite(texture).all()):
            raise DomainError("STAGE4_RESTORER_OUTPUT_INVALID",
                              "UV restorer must return finite uint8 BGR pixels with the canonical atlas shape", 503)
        self._texture_restorer = restorer
        texture = np.ascontiguousarray(texture)
        texture[self._socket_mask] = self._base_texture[self._socket_mask]    # the cavity is never restored
        self._texture = texture
        self._assemble_atlas()
        self._texture_restoration = {
            "status": "applied", "weight": self.restoration_weight,
            "implementation": f"{getattr(restorer, '__module__', type(restorer).__module__)}.{getattr(restorer, '__name__', type(restorer).__name__)}",
            "space": "canonical_full_head_uv",
        }

    @staticmethod
    def _landmarks(face: Any, name: str, aliases: tuple[str, ...] = ()) -> np.ndarray | None:
        value = VolumetricHeadRenderer._face_value(face, name)
        if value is None:
            for alias in aliases:
                value = VolumetricHeadRenderer._face_value(face, alias)
                if value is not None:
                    break
        if value is None:
            return None
        try:
            return np.asarray(value, np.float64)
        except (TypeError, ValueError) as error:
            raise DomainError("HEAD_LANDMARKS_INVALID", f"{name} must be numeric image-space landmarks", 422) from error

    def _target_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        k = default_intrinsics_matrix(width, height)
        return CameraIntrinsics(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
                                width=width, height=height)

    def _fit_expression(self, pose: HeadPose6DoF, landmarks68: np.ndarray, extent: float,
                        intrinsics: CameraIntrinsics) -> tuple[np.ndarray, dict[str, Any]]:
        zero = np.zeros(self._expression_basis.shape[2], np.float32)
        axis_ids = [self._model.expr_names.index(name) for name in self.EXPRESSION_AXES if name in self._model.expr_names]
        if len(axis_ids) < 3:
            return zero, {"status": "unavailable", "reason": "makehuman_expression_axes_insufficient"}
        observed = landmarks68[:, :2]
        if observed.shape != (68, 2) or not np.isfinite(observed).all():
            return zero, {"status": "unavailable", "reason": "dense68_landmarks_invalid"}
        points = self._mesh.vertices[self._landmarks68].astype(np.float64)
        basis = self._expression_basis[self._landmarks68][:, :, axis_ids].astype(np.float64)
        flip = np.array([1.0, 1.0, -1.0], np.float64)
        rotation = np.asarray(pose.R, np.float64)
        camera_points = (points * flip) @ rotation.T + np.asarray(pose.t, np.float64)
        if np.any(camera_points[:, 2] <= 1e-4):
            return zero, {"status": "unavailable", "reason": "dense_landmark_model_behind_camera"}
        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

        z = camera_points[:, 2]
        projected = np.column_stack((fx * camera_points[:, 0] / z + cx,
                                     fy * camera_points[:, 1] / z + cy))
        delta = basis * flip[None, :, None]
        delta_camera = np.einsum("nce,dc->nde", delta, rotation)
        jacobian = np.empty((68, 2, len(axis_ids)), np.float64)
        jacobian[:, 0] = fx * (delta_camera[:, 0] * z[:, None] - camera_points[:, 0, None] * delta_camera[:, 2]) / z[:, None] ** 2
        jacobian[:, 1] = fy * (delta_camera[:, 1] * z[:, None] - camera_points[:, 1, None] * delta_camera[:, 2]) / z[:, None] ** 2
        residual = observed - projected
        point_weight = np.ones(68, np.float64)
        point_weight[:17] = 0.35
        point_weight[17:27] = 0.6
        point_weight[27:36] = 0.4
        point_weight[36:48] = 1.0
        point_weight[48:68] = 1.2
        delta_px = max(3.0, 0.03 * extent)
        residual_length = np.linalg.norm(residual, axis=1)
        point_weight *= np.minimum(1.0, delta_px / np.maximum(residual_length, 1e-6))
        weights = np.repeat(point_weight, 2)
        matrix = jacobian.reshape(136, len(axis_ids))
        rhs_pixels = residual.reshape(-1)
        regularization = 1.0 / (0.35 * 0.35)
        normal = matrix.T @ (matrix * weights[:, None]) + np.eye(len(axis_ids)) * regularization
        rhs = matrix.T @ (rhs_pixels * weights)
        try:
            if not np.isfinite(normal).all() or np.linalg.cond(normal) > 1e10:
                return zero, {"status": "unavailable", "reason": "expression_fit_ill_conditioned"}
            values = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return zero, {"status": "unavailable", "reason": "expression_fit_singular"}
        values = np.clip(values, -1.0, 1.0)
        coefficients = zero.copy()
        coefficients[np.asarray(axis_ids, np.int64)] = values.astype(np.float32)
        if not np.isfinite(coefficients).all() or np.linalg.norm(coefficients) < 0.025:
            return zero, {"status": "unchanged", "reason": "expression_delta_below_observation_floor"}
        deformed = points + np.einsum("nce,e->nc", self._expression_basis[self._landmarks68], coefficients)
        deformed_camera = (deformed * flip) @ rotation.T + np.asarray(pose.t, np.float64)
        if np.any(deformed_camera[:, 2] <= 1e-4):
            return zero, {"status": "rejected", "reason": "expression_fit_crossed_camera_near_plane"}
        deformed_projection = np.column_stack((fx * deformed_camera[:, 0] / deformed_camera[:, 2] + cx,
                                               fy * deformed_camera[:, 1] / deformed_camera[:, 2] + cy))
        before = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        after = float(np.sqrt(np.mean(np.sum((observed - deformed_projection) ** 2, axis=1))))
        min_gain = max(0.5, 0.01 * extent)
        if after > before - min_gain or after > max(self.MIN_DENSE_REPROJECTION_PX,
                                                   self.MAX_DENSE_REPROJECTION_FRACTION * extent):
            return zero, {"status": "rejected", "reason": "expression_fit_did_not_explain_dense_residual",
                          "rms_before_px": before, "rms_after_px": after}
        return coefficients, {
            "status": "applied", "source": "robust_dense68_landmark_residual",
            "rms_before_px": before, "rms_after_px": after,
            "coefficients": {name: float(coefficients[index]) for index, name in enumerate(self._model.expr_names)},
            "axes_fit": [self._model.expr_names[index] for index in axis_ids],
        }

    def _target_pose(self, face: Any, width: int, height: int) -> tuple[HeadPose6DoF, np.ndarray, dict[str, Any]]:
        if self._face_value(face, "landmarks_trusted", True) is False:
            raise DomainError("TARGET_LANDMARKS_UNTRUSTED", "Target face landmarks are explicitly untrusted", 422)
        dense = self._landmarks(face, "landmarks68", ("landmarks_68",))
        if dense is not None:
            if dense.ndim != 2 or dense.shape[0] < 68 or dense.shape[1] < 2:
                raise DomainError("HEAD_DENSE_LANDMARKS_INVALID", "Dense target landmarks must contain 68 image-space points", 422)
            dense = np.ascontiguousarray(dense[:68, :])
            if not np.isfinite(dense[:, :2]).all():
                raise DomainError("HEAD_DENSE_LANDMARKS_INVALID", "Dense target landmark coordinates must be finite", 422)
        dense106 = self._landmarks(face, "landmarks106", ("landmarks_106",))
        if dense106 is not None:
            if dense106.ndim != 2 or dense106.shape[0] < 106 or dense106.shape[1] < 2:
                raise DomainError("HEAD_106_LANDMARKS_INVALID", "106-point target landmarks must contain 106 image-space points", 422)
            dense106 = np.ascontiguousarray(dense106[:106, :])
            if not np.isfinite(dense106[:, :2]).all():
                raise DomainError("HEAD_106_LANDMARKS_INVALID", "106-point target landmark coordinates must be finite", 422)
        intrinsics = self._target_intrinsics(width, height)
        pose: HeadPose6DoF | None = None
        pose_failure = None
        dense_extent = 0.0
        if dense is not None and np.isfinite(dense[:, :2]).all():
            dense_extent = float(np.linalg.norm(dense[:, :2].max(axis=0) - dense[:, :2].min(axis=0)))
            try:
                candidate = solve_head_pose_6dof(
                    dense[:, :2], intrinsics=intrinsics,
                    model_points=self._mesh.vertices[self._landmarks68].astype(np.float64))
                max_rms = max(self.MIN_DENSE_REPROJECTION_PX, self.MAX_DENSE_REPROJECTION_FRACTION * dense_extent)
                if candidate.reproj_rms_px > max_rms:
                    pose_failure = "dense_pnp_reprojection_too_high"
                else:
                    pose = candidate
            except DomainError as error:
                pose_failure = f"dense_pnp_failed:{error.code}"
        elif dense is not None:
            pose_failure = "dense68_landmarks_non_finite"
        else:
            pose_failure = ("106_point_layout_has_no_verified_makehuman_vertex_map" if dense106 is not None
                            else "no_verified_makehuman_dense68_correspondence")

        if pose is None:
            raise DomainError("DENSE_HEAD_LANDMARKS_REQUIRED",
                              f"A measured dense MakeHuman pose is required: {pose_failure}", 422)
        pose_source = "makehuman_pnp_dense68"

        expression = np.zeros(self._expression_basis.shape[2], np.float32)
        expression_diagnostics: dict[str, Any] = {"status": "unchanged", "reason": "no_reliable_expression_measurement"}
        supplied = self._face_value(face, "makehuman_expression_coefficients")
        explicit_expression = supplied is not None
        if supplied is None:
            generic = self._face_value(face, "expression_coefficients")
            if isinstance(generic, Mapping):
                names = set(self._model.expr_names)
                unknown = set(generic).difference(names)
                if not unknown:
                    mapped = np.zeros(len(self._model.expr_names), np.float64)
                    for name, value in generic.items():
                        try:
                            mapped[self._model.expr_names.index(name)] = float(value)
                        except (TypeError, ValueError) as error:
                            raise DomainError("HEAD_EXPRESSION_INVALID", f"Expression coefficient {name!r} is not numeric", 422) from error
                    supplied, explicit_expression = mapped, True
                else:
                    expression_diagnostics = {
                        "status": "unavailable", "reason": "expression_axes_not_mapped_to_makehuman",
                        "unknown_axes": sorted(str(name) for name in unknown),
                    }
            elif generic is not None:
                model_id = self._face_value(face, "expression_model")
                if model_id in ("makehuman", "makehuman-head-cc0"):
                    supplied, explicit_expression = generic, True
                else:
                    expression_diagnostics = {
                        "status": "unavailable", "reason": "unlabelled_expression_vector_not_assumed_to_be_makehuman",
                    }
        try:
            face_score = float(self._face_value(face, "score", 1.0))
        except (TypeError, ValueError) as error:
            raise DomainError("HEAD_FACE_SCORE_INVALID", "Target face score must be finite and within [0, 1]", 422) from error
        if not np.isfinite(face_score) or not 0.0 <= face_score <= 1.0:
            raise DomainError("HEAD_FACE_SCORE_INVALID", "Target face score must be finite and within [0, 1]", 422)
        if supplied is not None:
            try:
                values = np.asarray(supplied, np.float64).reshape(-1)
            except (TypeError, ValueError) as error:
                raise DomainError("HEAD_EXPRESSION_INVALID", "Target expression coefficients must be a numeric vector", 422) from error
            if (values.shape != (self._expression_basis.shape[2],) or not np.isfinite(values).all()
                    or np.any(np.abs(values) > 1.0)):
                raise DomainError("HEAD_EXPRESSION_INVALID", "Target expression coefficients violate the MakeHuman basis", 422)
            if face_score >= 0.5:
                expression = values.astype(np.float32)
                expression_diagnostics = {"status": "applied", "source": "explicit_makehuman_expression_coefficients",
                                          "coefficients": {name: float(value) for name, value in zip(self._model.expr_names, expression)}}
            else:
                expression_diagnostics = {"status": "rejected", "reason": "target_face_confidence_below_0_5"}
        elif pose_source == "makehuman_pnp_dense68" and dense is not None and face_score >= 0.5:
            expression, expression_diagnostics = self._fit_expression(pose, dense, dense_extent, intrinsics)
        elif dense is not None and face_score < 0.5:
            expression_diagnostics = {"status": "unavailable", "reason": "dense_pose_or_face_confidence_insufficient"}

        if np.any(expression):
            deformed_vertices = self._identity_vertices_with_expression(expression)
            model_points = deformed_vertices[self._landmarks68]
            observed = dense[:, :2]
            try:
                refined = solve_head_pose_6dof(observed, intrinsics=intrinsics, model_points=model_points,
                                               prev_rotation=pose.R)
                permitted = max(self.MIN_DENSE_REPROJECTION_PX,
                                self.MAX_DENSE_REPROJECTION_FRACTION * dense_extent)
                if refined.reproj_rms_px <= permitted:
                    pose = refined
                else:
                    raise DomainError("HEAD_EXPRESSION_POSE_INVALID", "Expression-deformed PnP exceeded its reprojection bound", 422)
            except DomainError as error:
                if explicit_expression:
                    raise DomainError("HEAD_EXPRESSION_POSE_INVALID",
                                      f"Explicit expression coefficients do not fit target landmarks: {error.code}", 422) from error
                expression = np.zeros_like(expression)
                expression_diagnostics = {"status": "rejected", "reason": f"expression_pose_refinement_failed:{error.code}"}

        return pose, expression, {
            "source": pose_source,
            "dense_failure": pose_failure,
            "yaw_deg": float(pose.yaw_deg), "pitch_deg": float(pose.pitch_deg), "roll_deg": float(pose.roll_deg),
            "reprojection_rms_px": float(pose.reproj_rms_px), "inliers": int(pose.inliers),
            "dense_landmark_count": 68 if dense is not None else 0,
            "detector_106_landmark_count": 106 if dense106 is not None else 0,
            "expression": expression_diagnostics,
        }

    def _identity_vertices_with_expression(self, coefficients: np.ndarray) -> np.ndarray:
        if not np.any(coefficients):
            return self._mesh.vertices
        vertices = self._mesh.vertices + np.einsum("vck,k->vc", self._expression_basis, coefficients).astype(np.float32)
        if not np.isfinite(vertices).all():
            raise DomainError("HEAD_EXPRESSION_INVALID", "Expression deformation produced non-finite vertices", 422)
        return vertices

    def _smooth_pose(self, pose: HeadPose6DoF, face: Any, timestamp: float, scene_cut: bool) -> tuple[np.ndarray, np.ndarray, int]:
        raw_id = self._face_value(face, "track_id", 0)
        if isinstance(face, Mapping):
            raw_id = face.get("track_id", raw_id)
        if hasattr(face, "track_id"):
            raw_id = getattr(face, "track_id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)) or int(raw_id) < 0:
            raise DomainError("HEAD_TRACK_ID_INVALID", "Target track id must be a nonnegative integer", 422)
        track_id = int(raw_id)
        if scene_cut:
            self._temporal.forget_track(track_id)
        rotation = self._temporal.filter_rotation_matrix(track_id, "stage4_head_pose", pose.R, timestamp)
        translation = self._temporal.filter(track_id, "stage4_head_translation", pose.t, timestamp)
        return rotation, translation, track_id

    @staticmethod
    def _project_all(camera: PerspectiveCamera, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_vertices = camera.camera_points(vertices)
        depth = camera_vertices[:, 2]
        if not np.isfinite(camera_vertices).all() or np.any(depth <= 1e-4):
            raise DomainError("HEAD_INTERSECTS_CAMERA", "The full MakeHuman head is not in front of the camera near plane", 422)
        k = camera.intrinsics
        projected = np.column_stack((k.fx * camera_vertices[:, 0] / depth + k.cx,
                                     k.fy * camera_vertices[:, 1] / depth + k.cy))
        if not np.isfinite(projected).all():
            raise DomainError("HEAD_PROJECTION_INVALID", "Full-head perspective projection is non-finite", 422)
        return camera_vertices, projected.astype(np.float32)

    @staticmethod
    def _roi(projected: np.ndarray, width: int, height: int) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        minimum = projected.min(axis=0).astype(np.float64)
        maximum = projected.max(axis=0).astype(np.float64)
        x0, y0 = int(math.floor(minimum[0])), int(math.floor(minimum[1]))
        x1, y1 = int(math.ceil(maximum[0])) + 1, int(math.ceil(maximum[1])) + 1
        clipped = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        return clipped, {
            "all_vertex_bounds_xyxy": [float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])],
            "clipped_raster_roi_xyxy": [int(value) for value in clipped],
            "projected_vertex_count": int(len(projected)),
            "projected_vertices_in_frame": int(np.count_nonzero((projected[:, 0] >= 0) & (projected[:, 0] < width)
                                                                   & (projected[:, 1] >= 0) & (projected[:, 1] < height))),
        }

    @staticmethod
    def _feather_coverage(coverage: np.ndarray) -> np.ndarray:
        hard = (coverage > 0.0).astype(np.uint8)
        if not hard.any():
            return coverage
        distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
        ramp = np.clip(distance / VolumetricHeadRenderer.SILHOUETTE_FEATHER_PX, 0.0, 1.0)
        return np.asarray(coverage * ramp, np.float32)

    def _fit_and_apply_lighting(self, image: np.ndarray, face: Any, camera_normals: np.ndarray,
                                albedo_image: np.ndarray, geometric_coverage: np.ndarray, *,
                                eye_mask: np.ndarray, cornea_mask: np.ndarray,
                                intrinsics: CameraIntrinsics) -> tuple[np.ndarray, dict[str, Any]]:
        box = self._face_value(face, "box")
        if box is None:
            box = self._face_value(face, "bbox")
        dense = self._landmarks(face, "landmarks68", ("landmarks_68",))
        if dense is None:
            dense = self._landmarks(face, "landmarks5", ("landmarks_5", "landmarks"))
        allowed, skin, semantic_stats = self._semantic_masks(image, box, dense)
        weights = skin * np.clip(geometric_coverage, 0.0, 1.0)
        sample_count = int(np.count_nonzero(weights > 0.05))
        if sample_count < 32:
            raise DomainError("STAGE4_LIGHTING_EVIDENCE_INSUFFICIENT",
                              f"Only {sample_count} visible target-skin samples overlap the rendered head", 422)
        target_linear = srgb_to_linear(image)
        albedo_linear = srgb_to_linear(albedo_image)
        try:
            coefficients = fit_sh(camera_normals, target_linear, weights, albedo=albedo_linear)
        except (np.linalg.LinAlgError, ValueError) as error:
            raise DomainError("STAGE4_LIGHTING_FIT_FAILED", f"Target SH fit failed: {error}", 422) from error
        if coefficients.shape != (9, 3) or not np.isfinite(coefficients).all():
            raise DomainError("STAGE4_LIGHTING_FIT_INVALID", "Target SH fit returned invalid coefficients", 422)
        sh = SphericalHarmonicsLighting()
        shading = sh.evaluate_shading(camera_normals, coefficients)
        usable = weights > 0.05
        # Least squares on the SH basis is biased by the clipped, saturated and shadowed samples it down-weights; match the
        # first moment of the visible skin per channel so the raw render's tone equals the scene's before any compositing.
        w_usable = weights[usable][:, None].astype(np.float64)
        predicted_mean = (w_usable * (albedo_linear[usable] * shading[usable])).sum(axis=0)
        target_mean = (w_usable * target_linear[usable]).sum(axis=0)
        gain = np.clip(target_mean / np.maximum(predicted_mean, 1e-8), 0.8, 1.25).astype(np.float32)
        shading = shading * gain
        lit_linear = np.clip(albedo_linear * shading, 0.0, 1.0)
        eye_stats = self._shade_eyes(lit_linear, camera_normals, eye_mask, cornea_mask, intrinsics, coefficients, sh)
        rendered = np.clip(np.rint(linear_to_srgb(lit_linear) * 255.0), 0, 255).astype(np.uint8)
        rendered[geometric_coverage <= 0.0] = 0
        predicted = albedo_linear[usable] * shading[usable]
        residual = float(np.sqrt(np.mean((predicted - target_linear[usable]) ** 2)))
        return rendered, {
            "method": "robust_second_order_spherical_harmonics_with_first_moment_tone_calibration",
            "tone_gain_bgr": [float(value) for value in gain],
            "eye_shading": eye_stats,
            "samples": sample_count,
            "linear_bgr_rms": residual,
            "coefficients_bgr": coefficients.tolist(),
            "key_direction_camera": sh.extract_key_light_direction(coefficients).astype(float).tolist(),
            "target_skin_mask": semantic_stats,
            "mask_sources": ["BiSeNet skin class (candidate conventional 19-class mapping)",
                             "XSeg visible-face probability", "rendered mesh coverage"],
            "target_face_probability_fraction": float(np.mean(allowed > 0.5)),
        }

    @staticmethod
    def _shade_eyes(lit_linear: np.ndarray, camera_normals: np.ndarray, eye_mask: np.ndarray, cornea_mask: np.ndarray,
                    intrinsics: CameraIntrinsics, coefficients: np.ndarray, sh: SphericalHarmonicsLighting) -> dict[str, Any]:
        """Contact shadow of the lids on the globes and the key-light specular highlight of the wet cornea/sclera.

        Both terms are driven by measured quantities: the shadow by the rasterized lid/globe silhouette distance, the
        highlight by the SH-fitted key light direction and each pixel's true view ray and surface normal.
        """
        eyes = eye_mask > 0.5
        if not eyes.any():
            return {"status": "no_eye_pixels"}
        eye_radius = float(np.sqrt(np.count_nonzero(eyes) / (2.0 * np.pi)))      # two visible apertures
        distance = cv2.distanceTransform(eyes.astype(np.uint8), cv2.DIST_L2, 3)
        scale = max(0.35 * eye_radius, 1.0)
        occlusion = 1.0 - 0.38 * np.exp(-distance / scale)
        lit_linear[eyes] *= occlusion[eyes][:, None]
        luma_coefficients = coefficients @ np.array((0.114, 0.587, 0.299), np.float32)
        key = sh.extract_key_light_direction(coefficients).astype(np.float64)
        ambient = max(float(luma_coefficients[0]), 1e-4)
        key_strength = float(np.clip(2.0 * np.linalg.norm(luma_coefficients[1:4]) / ambient, 0.1, 0.9))
        pixel_y, pixel_x = np.nonzero(eyes)
        ray = np.stack(((pixel_x - intrinsics.cx) / intrinsics.fx, (pixel_y - intrinsics.cy) / intrinsics.fy,
                        np.full(len(pixel_x), -1.0)), axis=1)            # camera -> scene, +z toward viewer convention
        view = -ray / np.linalg.norm(ray, axis=1, keepdims=True)
        half = view + key[None, :]
        half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-9)
        normal = camera_normals[pixel_y, pixel_x].astype(np.float64)
        facing = np.clip(np.einsum("ij,ij->i", normal, half), 0.0, 1.0)
        on_cornea = cornea_mask[pixel_y, pixel_x]
        highlight = np.where(on_cornea, 0.9 * facing ** 220, 0.07 * facing ** 40) * key_strength
        lit_linear[pixel_y, pixel_x] = np.clip(lit_linear[pixel_y, pixel_x] + highlight[:, None], 0.0, 1.0)
        return {"status": "applied", "contact_shadow": "distance_to_lid_silhouette_exp_falloff",
                "specular": "blinn_phong_key_light_from_sh_fit", "key_strength": key_strength,
                "highlight_pixels_gt_0_2": int(np.count_nonzero(highlight > 0.2))}

    def render(self, image: np.ndarray, face: Any, timestamp: float, *, scene_cut: bool = False) -> HeadLayer:
        """Fit target 6DoF pose and render a full-frame volumetric head layer."""
        if self._released:
            raise DomainError("RENDERER_RELEASED", "Volumetric head renderer has been released", 409)
        image = image_bgr(image)
        if isinstance(timestamp, (bool, np.bool_)):
            raise DomainError("HEAD_TIMESTAMP_INVALID", "Render timestamp must be a finite scalar", 422)
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as error:
            raise DomainError("HEAD_TIMESTAMP_INVALID", "Render timestamp must be a finite scalar", 422) from error
        if not np.isfinite(timestamp):
            raise DomainError("HEAD_TIMESTAMP_INVALID", "Render timestamp must be a finite scalar", 422)
        observation = self._face_value(face, "observation")
        if observation is not None:
            pose_face = observation
        else:
            pose_face = face
        height, width = image.shape[:2]
        raw_pose, expression, pose_stats = self._target_pose(pose_face, width, height)
        rotation, translation, track_id = self._smooth_pose(raw_pose, face, float(timestamp), bool(scene_cut))
        camera = self._camera(rotation, translation, self._target_intrinsics(width, height))
        vertices = self._identity_vertices_with_expression(expression)
        normals = self._base_normals if not np.any(expression) else welded_vertex_normals(vertices, self._mesh.faces, self._weld)
        camera_vertices, projected = self._project_all(camera, vertices)
        from roop_3d.core.true3d_guard import validate_camera
        validate_camera(camera.intrinsics.matrix(), rotation, translation,
                        vertices=vertices, mesh_z_toward_viewer=True)
        roi, roi_stats = self._roi(projected, width, height)
        camera_normals = camera.rotate_normals(normals).astype(np.float32)
        globes = self._eye_globes
        eye_camera_vertices, eye_projected = self._project_all(camera, globes.vertices)
        eye_camera_normals = camera.rotate_normals(globes.normals).astype(np.float32)
        skin_triangles = len(self._mesh.faces)
        # Reflecting the mesh's viewer-facing Z axis reverses triangle winding. Skin and both eyeballs share one Z-buffer,
        # so the lid geometry occludes the globes exactly where it is in front of them.
        triangles = np.concatenate((self._mesh.faces, globes.faces + len(vertices)))[:, ::-1]
        raster = rasterize_layer(
            self._raster_atlas, self._raster_coverage,
            np.concatenate((projected, eye_projected)), np.concatenate((camera_vertices, eye_camera_vertices)),
            np.concatenate((self._skin_uv_pixels, self._eye_uv_pixels)), triangles,
            np.concatenate((camera_normals, eye_camera_normals)), (height, width),
            roi=roi, perspective=True, sampling_device="cpu")
        if raster.normals is None or raster.face_index is None:
            raise DomainError("PERSPECTIVE_RASTER_NORMALS_MISSING", "Perspective rasterizer did not return surface normals", 500)
        eye_mask = (raster.face_index >= skin_triangles).astype(np.float32)
        coverage = self._feather_coverage(raster.coverage)
        owner = raster.face_index - skin_triangles
        cornea_mask = np.zeros(raster.face_index.shape, bool)
        eye_pixels = owner >= 0
        cornea_mask[eye_pixels] = globes.cornea_faces[owner[eye_pixels]]
        rendered, lighting_stats = self._fit_and_apply_lighting(
            image, pose_face, raster.normals, raster.image, raster.coverage,
            eye_mask=eye_mask, cornea_mask=cornea_mask, intrinsics=camera.intrinsics)
        standard_rotation = rotation_to_standard(rotation)
        yaw, pitch, roll = euler_from_rotation(standard_rotation)
        diagnostics = {
            "model": {"id": "makehuman-head-cc0", "license": "CC0-1.0",
                      "base_vertices": int(len(self._model.base_mesh.vertices)),
                      "render_vertices": int(len(vertices)), "triangles": int(len(self._mesh.faces)),
                      "shape_prior": "MakeHuman CC0 full head and neck; unobserved geometry remains template prior"},
            "architecture_validation": self._mesh_validation,
            "identity_fit": self._identity_diagnostics,
            "source_views": self._source_view_diagnostics,
            "uv_synthesis": self._uv_diagnostics,
            "texture_restoration": dict(self._texture_restoration),
            "pose": {**pose_stats, "yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll),
                     "translation_m": [float(value) for value in translation], "track_id": track_id,
                     "camera_intrinsics": {name: float(getattr(camera.intrinsics, name)) for name in ("fx", "fy", "cx", "cy")},
                     "temporal_smoothing": "SO(3) quaternion One-Euro plus translation One-Euro/Kalman"},
            "projection_roi": roi_stats,
            "rasterization": {"method": "CPU perspective triangle Z-buffer with perspective-correct UVs",
                               "visible_triangle_count": int(len(raster.faces)),
                               "visible_eyeball_pixels": int(np.count_nonzero(eye_mask)),
                               "eyeball_triangles": int(len(globes.faces)),
                               "silhouette_feather_px": self.SILHOUETTE_FEATHER_PX},
            "lighting": lighting_stats,
            "eyes": self._eye_diagnostics,
            "limitations": [
                "MakeHuman head, ears, scalp, and neck geometry outside landmark constraints remain a template prior.",
                "Texture regions without direct source visibility are symmetry or harmonic completion, not observations.",
                "MakeHuman has no scalp-hair geometry; semantic masks exclude hair, glasses, neck, and occluders.",
                "Dense68 correspondences are template-derived; unsupported or inconsistent dense poses are rejected.",
            ],
        }
        layer = HeadLayer(
            image=np.ascontiguousarray(rendered, dtype=np.uint8),
            coverage=np.ascontiguousarray(coverage, dtype=np.float32),
            normals=np.ascontiguousarray(raster.normals, dtype=np.float32),
            projected_vertices=np.ascontiguousarray(projected, dtype=np.float32),
            faces=self._mesh.faces,
            depth=np.ascontiguousarray(raster.depth, dtype=np.float32),
            diagnostics=diagnostics,
            eye_mask=np.ascontiguousarray(eye_mask),
            eye_projected_vertices=np.ascontiguousarray(eye_projected, dtype=np.float32),
            eye_faces=globes.faces,
        )
        self.last_layer = layer
        return layer

    def release(self) -> None:
        """Release renderer-owned inference sessions and frame-history buffers."""
        if self._released:
            return
        self._released = True
        self.last_layer = None
        self._temporal.reset()
        parser, visible = self._parser, self._visible_face
        self._parser = self._visible_face = None
        if parser is not None:
            parser.release()
        if visible is not None:
            visible.release()
        for name in ("_model", "_template", "_shape_basis", "_expression_basis", "_shape_params", "_mesh",
                     "_base_normals", "_weld", "_landmarks5", "_landmarks68", "_landmark_correspondence",
                     "_atlas_uv", "_base_texture", "_texture", "_texture_coverage", "_uv_completion_source",
                     "_source_views", "_source_cameras", "_source_fit", "_source_excluded",
                     "_source_view_diagnostics", "_source_surface_cameras", "_mesh_validation",
                     "_identity_diagnostics", "_uv_diagnostics", "_eye_globes", "_eye_tiles", "_eye_diagnostics",
                     "_eye_diagnostics_base", "_socket_mask", "_socket_vertices", "_socket_faces", "_raster_atlas",
                     "_raster_coverage", "_skin_uv_pixels", "_eye_uv_pixels", "_source_eye_planes"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.document = None


__all__ = ["HeadLayer", "VolumetricHeadRenderer"]
