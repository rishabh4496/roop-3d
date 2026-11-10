"""Face-only multiview correspondence conditioning followed by neural synthesis."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import AlignedCrop, aligned_crop, transform_points
from roop_3d.reconstruction.dense_correspondence import (
    GPUSurfaceRasterizer, WarpField, correspondence_field, project,
)
from roop_3d.reconstruction.morphable_backbone import MorphableBackbone


@dataclass(frozen=True)
class SourceCameraSurface:
    camera_vertices: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class SourceSurface:
    view: object
    fit: SourceCameraSurface
    image: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray
    observed: np.ndarray
    original_inverse: np.ndarray  # aligned photograph cache -> original .fsz image pixels


@dataclass(frozen=True)
class NeuralFaceResult:
    image: np.ndarray
    inverse: np.ndarray
    coverage: np.ndarray
    landmarks5: np.ndarray


def front_faces(camera_vertices, faces):
    triangles = camera_vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    return faces[np.einsum("ij,ij->i", normal, triangles.mean(1)) < 0]


class NeuralFaceEngine:
    def __init__(self, backbone, rasterizer, enhancer, *, profile="6gb"):
        self.backbone, self.rasterizer, self.enhancer = backbone, rasterizer, enhancer
        self.surface = getattr(backbone, "surface", backbone.topology)
        self.profile = profile
        self.sources = {}
        self.source_shapes = {}
        self.last_source_shape_scale_ratio = 1.0
        self.last_fields = ()
        self.last_original_fields = ()
        self.last_source_view_indices = ()

    @classmethod
    def load(cls, catalog, config):
        from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
        backbone = MorphableBackbone.load(catalog, config)
        rasterizer = GPUSurfaceRasterizer()
        try:
            rasterizer._context()
            return cls(backbone, rasterizer, UltraMaxPipelineRestorer(catalog, config),
                       profile=config.hardware_profile)
        except BaseException:
            rasterizer.release()
            backbone.release()
            raise

    def prepare(self, document):
        if document.faceset_id in self.sources:
            return
        real = [v for v in document.views if not v.is_synthetic and v.landmarks_trusted]
        if not real:
            raise DomainError("DENSE_FACE_OBSERVATIONS_REQUIRED", "3D warping requires real source photographs and measured landmarks", 422)
        # Balance view directions before filling remaining quality-ranked slots.
        ordered = sorted(real, key=lambda v: (-v.quality.score, v.index))
        buckets = {}
        for view in ordered:
            buckets.setdefault((round(view.yaw/30), round(view.pitch/25)), view)
        limit = 24 if self.profile == "12gb" else 12
        selected = list(buckets.values())[:limit]
        selected += [v for v in ordered if v.index not in {s.index for s in selected}][:limit-len(selected)]
        side = 1024 if self.profile == "12gb" else 512
        surfaces = []
        identity_shapes = []
        identity_weights = []
        for view in selected:
            image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise DomainError("DENSE_FACE_IMAGE_INVALID", "Could not decode source photograph", 422)
            fit = self.backbone.fit(image, view.landmarks)
            identity = getattr(fit, "identity", None)
            if identity is not None:
                identity = np.asarray(identity, np.float32)
                if identity.ndim != 2 or identity.shape[1] != 3 or not np.isfinite(identity).all():
                    raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID", "Source identity geometry must be finite (V,3)", 422)
                identity_shapes.append(identity)
                identity_weights.append(max(float(view.quality.score), .01))
            # Sample the original high-resolution photograph once, never a UV
            # shader or a prefiltered diffuse-albedo atlas.
            matrix = fit.alignment.matrix * (side/128)
            photograph = cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_LINEAR)
            observed = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (side, side),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            k = np.diag((side/224, side/224, 1)) @ fit.intrinsics
            faces = front_faces(fit.camera_vertices, self.surface.faces)
            if not len(faces):
                raise DomainError("DENSE_FACE_WINDING_INVALID", "No outward-facing source triangles; verify export winding and coordinates", 422)
            surface = self.rasterizer.rasterize(fit.camera_vertices, faces,
                k, fit.camera_vertices, (side, side))
            # The displacement has already changed the camera surface. Do not
            # retain a potentially 2048-square HRN map for every cached view.
            camera_surface = SourceCameraSurface(fit.camera_vertices, fit.rotation)
            surfaces.append(SourceSurface(view, camera_surface, photograph, k, surface.depth, observed,
                                          cv2.invertAffineTransform(matrix)))
        if not identity_shapes:
            raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID",
                              "Dense neural synthesis requires a neutral source mesh from the FaceSet", 422)
        if any(shape.shape != identity_shapes[0].shape for shape in identity_shapes):
            raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID", "FaceSet identity meshes do not share one topology", 422)
        weights = np.asarray(identity_weights, np.float64)
        source_shape = np.average(np.stack(identity_shapes).astype(np.float64), axis=0,
                                  weights=weights).astype(np.float32)
        self.sources[document.faceset_id] = tuple(surfaces)
        self.source_shapes[document.faceset_id] = source_shape

    @staticmethod
    def _metric_scale(vertices):
        """Geometric mean of metric head extents, used to normalize expression deltas."""
        extent = np.ptp(np.asarray(vertices, np.float64), axis=0)
        return float(np.prod(np.maximum(extent, 1e-6)) ** (1.0 / 3.0))

    @classmethod
    def _source_geometry_in_target_pose(cls, source_identity, target_fit):
        """Put target articulation on source identity geometry, then project with target camera pose."""
        source_identity = np.asarray(source_identity, np.float32)
        target_identity = np.asarray(target_fit.identity, np.float32)
        expression_geometry = getattr(target_fit, "expression_vertices", None)
        if expression_geometry is None:
            raise DomainError("DENSE_FACE_EXPRESSION_GEOMETRY_REQUIRED",
                              "Target fit must expose expression-only geometry before detail offsets", 422)
        target_expressed = np.asarray(expression_geometry, np.float32)
        if (source_identity.shape != target_identity.shape or target_identity.shape != target_expressed.shape
                or not np.isfinite(source_identity).all() or not np.isfinite(target_identity).all()
                or not np.isfinite(target_expressed).all()):
            raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID", "Source and target geometry must share finite topology", 422)
        scale_ratio = cls._metric_scale(source_identity) / max(cls._metric_scale(target_identity), 1e-6)
        # The base is 100% source identity. Only target expression displacements
        # cross the identity boundary, converted into source metric scale; the
        # target fit's HRN/detail geometry is deliberately excluded.
        vertices = source_identity + (target_expressed - target_identity) * scale_ratio
        camera_vertices = (vertices @ np.asarray(target_fit.rotation, np.float32).T
                           * float(target_fit.scale) + np.asarray(target_fit.translation, np.float32))
        if not np.isfinite(camera_vertices).all():
            raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID", "Target pose produced non-finite source geometry", 422)
        return vertices.astype(np.float32), camera_vertices.astype(np.float32), scale_ratio

    @staticmethod
    def _mesh_crop_affine(projected_pixels, size: int, padding: float = 0.12) -> np.ndarray:
        """Frame the projected source mesh in swapper space without target 2D landmark alignment."""
        points = np.asarray(projected_pixels, np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or not 0 <= padding < 0.5):
            raise DomainError("DENSE_FACE_SOURCE_PROJECTION_INVALID", "Projected source mesh bounds are invalid", 422)
        lower, upper = points.min(axis=0), points.max(axis=0)
        extent = float(np.max(upper - lower))
        if extent <= 1e-4:
            raise DomainError("DENSE_FACE_SOURCE_PROJECTION_INVALID", "Projected source mesh is degenerate", 422)
        scale = size * (1.0 - 2.0 * padding) / extent
        center = (lower + upper) * 0.5
        destination = np.array((size * 0.5, size * 0.52), np.float64)
        translation = destination - scale * center
        return np.array(((scale, 0.0, translation[0]),
                         (0.0, scale, translation[1])), np.float64)

    @staticmethod
    def _compose_affines(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        return (np.vstack((outer, (0.0, 0.0, 1.0)))
                @ np.vstack((inner, (0.0, 0.0, 1.0))))[:2]

    def process(self, image, face, document, swapper, *, track_id):
        self.last_fields = self.last_original_fields = self.last_source_view_indices = ()
        if document.faceset_id not in self.sources:
            raise DomainError("DENSE_FACE_NOT_PREPARED", "Prepare source reconstruction before frame inference", 409)
        target = self.backbone.fit(image, face.landmarks5)
        model = swapper.model
        if model is None or not callable(getattr(model, "infer", None)):
            raise DomainError("DENSE_FACE_NEURAL_REQUIRED", "Dense face output requires a prepared neural swapper", 503)
        size = model.resolution
        surfaces = sorted(self.sources[document.faceset_id], key=lambda s:
            (np.linalg.norm(s.fit.rotation-target.rotation), -s.view.quality.score))
        color = np.zeros((size, size, 3), np.float32)
        support = np.zeros((size, size), np.float32)
        fields, original_fields, view_indices = [], [], []
        source_identity = self.source_shapes.get(document.faceset_id)
        if source_identity is None or not hasattr(target, "identity") or not hasattr(target, "vertices"):
            raise DomainError("DENSE_FACE_SOURCE_SHAPE_INVALID",
                              "Dense synthesis needs source and target neutral geometry", 422)
        _, target_camera_vertices, expression_scale_ratio = self._source_geometry_in_target_pose(
            source_identity, target)
        self.last_source_shape_scale_ratio = expression_scale_ratio
        target_faces = front_faces(target_camera_vertices, self.surface.faces)
        if not len(target_faces):
            raise DomainError("DENSE_FACE_NOT_VISIBLE", "Target has no front-facing anatomical face surface", 422)
        # Project the source-identity mesh through the target camera first.
        # This geometry-derived transform defines swapper pixels; target 2D
        # landmarks are not used to align source appearance to target anatomy.
        fit_pixels = project(target_camera_vertices, target.intrinsics)
        source_vertices = np.unique(target_faces)
        mesh_affine = self._mesh_crop_affine(fit_pixels[source_vertices], size)
        source_to_fit = np.asarray(target.alignment.matrix, np.float64) * (224.0 / 128.0)
        crop_matrix = self._compose_affines(mesh_affine, source_to_fit)
        aligned = AlignedCrop(np.zeros((size, size, 3), np.uint8), crop_matrix,
                             cv2.invertAffineTransform(crop_matrix))
        k = np.vstack((mesh_affine, (0.0, 0.0, 1.0))) @ target.intrinsics
        # Appearance confidence governs photograph sampling. The replacement
        # boundary is the target's anatomical surface: the neural model owns
        # unobserved facial skin too, rather than leaving patches of target ID.
        target_surface = self.rasterizer.rasterize(target_camera_vertices, target_faces,
            k, target_camera_vertices, (size, size)).valid
        for source in surfaces[:4 if self.profile == "12gb" else 2]:
            field = correspondence_field(self.rasterizer, target_faces,
                source.fit.camera_vertices, source.intrinsics, target_camera_vertices, k, (size, size),
                source.depth, (source.image.shape[1], source.image.shape[0]), source_support=source.observed)
            angular = float(np.exp(-2*np.linalg.norm(source.fit.rotation-target.rotation)))
            weight = field.confidence * max(source.view.quality.score, .01) * angular
            color += field.sample(source.image).astype(np.float32) * weight[..., None]
            support += weight
            fields.append(field)
            # Keep a separately labelled W_3d_to_target in original photograph
            # coordinates for diagnostics/export. Sampling above uses the bounded
            # aligned photo cache, not an unbounded full-resolution image cache.
            original_xy = cv2.transform(field.source_xy, source.original_inverse)
            original_fields.append(WarpField(np.where(field.confidence[..., None] > 0,
                                                     original_xy, -1).astype(np.float32), field.confidence))
            view_indices.append(source.view.index)
        self.last_fields = tuple(fields)
        self.last_original_fields = tuple(original_fields)
        self.last_source_view_indices = tuple(view_indices)
        valid = support > 1e-8
        if not valid.any():
            raise DomainError("DENSE_FACE_NO_CORRESPONDENCE", "No observed source surface is visible in the target view", 422)
        warped = color / np.maximum(support[..., None], 1e-8)
        # Missing source texels stay source-colored. Target pixels here would
        # teach a 2D swapper the target actor's jaw/cheek appearance and undo
        # the source-geometry projection that produced this conditioning crop.
        mirrored = cv2.flip(warped, 1)
        mirrored_valid = cv2.flip(valid.astype(np.uint8), 1).astype(bool)
        source_skin = np.median(warped[valid], axis=0)
        fallback = np.broadcast_to(source_skin, warped.shape).copy()
        use_mirror = ~valid & mirrored_valid
        fallback[use_mirror] = mirrored[use_mirror]
        conditioning = np.where(valid[..., None], warped, fallback).round().clip(0, 255).astype(np.uint8)
        # Full neural output owns ALL replacement pixels. No residual extraction
        # or fall-through to the photographic conditioning image is permitted.
        neural = swapper.synthesize_correspondence(conditioning, pose=face.pose, track_id=track_id)
        # Enhancement sees the geometry-conditioned source crop. CodeFormer/UltraMax are trained on five-point FFHQ
        # alignment and fail closed without it, so the restorer is given the target's five landmarks expressed in this
        # crop's own frame. They only choose the restorer's internal alignment: the result is warped straight back
        # into the same crop frame, so they cannot recenter or reshape the source face.
        crop_points = transform_points(face.landmarks5, aligned.matrix)
        enhanced = self.enhancer.restore_hires(neural, crop_points)
        if enhanced is None or enhanced.dtype != np.uint8 or enhanced.ndim != 3 or enhanced.shape[2] != 3:
            raise DomainError("DENSE_FACE_ENHANCER_FAILED", "UltraMax did not return a neural face image", 503)
        height, width = enhanced.shape[:2]
        # A feather stays strictly inside the anatomical facial surface. Source
        # holes are neural synthesis from the identity embedding, not observed
        # appearance and never a shader/albedo completion.
        coverage = cv2.resize(target_surface.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        edge = cv2.distanceTransform(coverage.astype(np.uint8), cv2.DIST_L2, 3)
        coverage *= np.clip(edge / max(width*.01, 1), 0, 1)
        inverse = aligned.inverse.copy()
        inverse[:, 0] *= size/width
        inverse[:, 1] *= size/height
        points = crop_points * (width/size, height/size)
        return NeuralFaceResult(enhanced, inverse, coverage, points)

    def capability(self):
        return {**self.backbone.capability(), "prepared_identities": len(self.sources),
                "source_texture": "original_photographs", "visibility": "source_and_target_depth_test",
                "geometry_identity_source": "faceset_multiview_neutral",
                "target_geometry_conditioning": "source_shape+normalized_target_expression",
                "target_pixels_in_source_holes": False,
                "conditioning_crop": "projected_source_mesh_in_target_camera",
                "warp_coordinate_space": "aligned_source_crop_pixels", "original_warp_fields": True,
                "source_views_per_frame": 4 if self.profile == "12gb" else 2,
                "enhancer": self.enhancer.capability()}

    def release(self):
        self.sources.clear()
        self.source_shapes.clear()
        self.last_source_shape_scale_ratio = 1.0
        self.last_fields = ()
        self.last_original_fields = ()
        self.last_source_view_indices = ()
        self.enhancer.release()
        self.rasterizer.release()
        self.backbone.release()
