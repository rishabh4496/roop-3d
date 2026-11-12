"""Face-only multiview correspondence conditioning followed by neural synthesis."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import aligned_crop, transform_points
from roop_3d.reconstruction.dense_correspondence import GPUSurfaceRasterizer, WarpField, correspondence_field
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
        for view in selected:
            image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise DomainError("DENSE_FACE_IMAGE_INVALID", "Could not decode source photograph", 422)
            fit = self.backbone.fit(image, view.landmarks)
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
        self.sources[document.faceset_id] = tuple(surfaces)

    def process(self, image, face, document, swapper, *, track_id):
        self.last_fields = self.last_original_fields = self.last_source_view_indices = ()
        if document.faceset_id not in self.sources:
            raise DomainError("DENSE_FACE_NOT_PREPARED", "Prepare source reconstruction before frame inference", 409)
        target = self.backbone.fit(image, face.landmarks5)
        model = swapper.model
        if model is None or not callable(getattr(model, "infer", None)):
            raise DomainError("DENSE_FACE_NEURAL_REQUIRED", "Dense face output requires a prepared neural swapper", 503)
        size = model.resolution
        aligned = aligned_crop(image, face.landmarks5, size,
                               template=model.spec.get("alignment_template", "arcface_112"))
        # Transform the fit's 224px intrinsics into this architecture's native
        # target crop, including FFHQ-vs-ArcFace offsets where applicable.
        h = np.vstack((aligned.matrix, (0, 0, 1))) @ np.vstack((target.alignment.inverse, (0, 0, 1)))
        k = h @ np.diag((128/224, 128/224, 1)) @ target.intrinsics
        surfaces = sorted(self.sources[document.faceset_id], key=lambda s:
            (np.linalg.norm(s.fit.rotation-target.rotation), -s.view.quality.score))
        color = np.zeros((size, size, 3), np.float32)
        support = np.zeros((size, size), np.float32)
        fields, original_fields, view_indices = [], [], []
        target_faces = front_faces(target.camera_vertices, self.surface.faces)
        if not len(target_faces):
            raise DomainError("DENSE_FACE_NOT_VISIBLE", "Target has no front-facing anatomical face surface", 422)
        # Appearance confidence governs photograph sampling. The replacement
        # boundary is the target's anatomical surface: the neural model owns
        # unobserved facial skin too, rather than leaving patches of target ID.
        target_surface = self.rasterizer.rasterize(target.camera_vertices, target_faces,
            k, target.camera_vertices, (size, size)).valid
        for source in surfaces[:4 if self.profile == "12gb" else 2]:
            field = correspondence_field(self.rasterizer, target_faces,
                source.fit.camera_vertices, source.intrinsics, target.camera_vertices, k, (size, size),
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
        conditioning = np.where(valid[..., None], warped, aligned.image).round().clip(0, 255).astype(np.uint8)
        # Full neural output owns ALL replacement pixels. No residual extraction
        # or fall-through to the photographic conditioning image is permitted.
        neural = swapper.synthesize_correspondence(conditioning, pose=face.pose, track_id=track_id)
        points = transform_points(face.landmarks5, aligned.matrix)
        enhanced = self.enhancer.restore_hires(neural, points)
        if enhanced is None or enhanced.dtype != np.uint8 or enhanced.ndim != 3 or enhanced.shape[2] != 3:
            raise DomainError("DENSE_FACE_ENHANCER_FAILED", "UltraMax did not return a neural face image", 503)
        height, width = enhanced.shape[:2]
        # A feather stays strictly inside the anatomical facial surface. Source
        # holes are neural synthesis from the identity embedding, not observed
        # appearance and never a shader/albedo completion.
        if width == size and height == size:
            target_surface_hi = target_surface
        else:
            k_hi = np.diag((width / size, height / size, 1.0)) @ k
            target_surface_hi = self.rasterizer.rasterize(
                target.camera_vertices, target_faces, k_hi, target.camera_vertices, (width, height)
            ).valid
        coverage = target_surface_hi.astype(np.float32)
        edge = cv2.distanceTransform(target_surface_hi.astype(np.uint8), cv2.DIST_L2, 5)
        coverage *= np.clip(edge / max(width * 0.015, 2.0), 0.0, 1.0)
        inverse = aligned.inverse.copy()
        inverse[:, 0] *= size/width
        inverse[:, 1] *= size/height
        points = points * (width/size, height/size)
        return NeuralFaceResult(enhanced, inverse, coverage, points)

    def capability(self):
        return {**self.backbone.capability(), "prepared_identities": len(self.sources),
                "source_texture": "original_photographs", "visibility": "source_and_target_depth_test",
                "warp_coordinate_space": "aligned_source_crop_pixels", "original_warp_fields": True,
                "source_views_per_frame": 4 if self.profile == "12gb" else 2,
                "enhancer": self.enhancer.capability()}

    def release(self):
        self.sources.clear()
        self.last_fields = ()
        self.last_original_fields = ()
        self.last_source_view_indices = ()
        self.enhancer.release()
        self.rasterizer.release()
        self.backbone.release()
