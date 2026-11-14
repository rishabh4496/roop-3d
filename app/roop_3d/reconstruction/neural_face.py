"""Face-only multiview source surface preparation and dense correspondence (no 2D swap synthesis: that path was removed)."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace

import cv2
import numpy as np

from roop_3d.compositor.masks import STRICT_FORBIDDEN_CLASSES
from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import transform_points
from roop_3d.reconstruction.dense_correspondence import (
    GPUSurfaceRasterizer, ViewWorkers, WarpField, correspondence_field,
)
from roop_3d.reconstruction.fit_ahead import FitAhead, fit_ahead_enabled
from roop_3d.reconstruction.morphable_backbone import MorphableBackbone
from roop_3d.reconstruction.view_selection import ViewSelector, neutral_pitch, view_cost
from roop_3d.temporal import CoupledTrackSmoother

log = logging.getLogger(__name__)


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


def front_faces(camera_vertices, faces):
    triangles = camera_vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    return faces[np.einsum("ij,ij->i", normal, triangles.mean(1)) < 0]


class WarmEngineSlot:
    """One idle ``NeuralFaceEngine`` kept between live previews.

    Building it loads the native backbone, re-verifies its weights and re-fits every FaceSet photograph (10-15 s) and was
    repeated for every preview.  The slot is single-use-at-a-time (``take`` removes the engine), is keyed by the runtime
    configuration, and is emptied together with the ORT session pool the moment a job or compile needs the GPU.
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._key = None
        self._engine = None
        from roop_3d.core.ort_session import on_pool_eviction
        on_pool_eviction(self.clear)

    def take(self, key):
        with self._lock:
            if self._engine is not None and self._key == key:
                engine, self._engine = self._engine, None
                return engine
        return None

    def put(self, key, engine) -> None:
        with self._lock:
            previous, self._engine, self._key = self._engine, engine, key
        if previous is not None and previous is not engine:
            previous.release()

    def clear(self) -> None:
        with self._lock:
            engine, self._engine, self._key = self._engine, None, None
        if engine is not None:
            engine.release()


WARM_ENGINE = WarmEngineSlot()


def view_worker_count(profile: str) -> int:
    """Threads for the per-view correspondence jobs: one per source view of the profile (4 on 12 GB, 2 otherwise);
    ``ROOP3D_VIEW_WORKERS`` overrides it, and ``1`` runs them sequentially."""
    import os
    default = 4 if profile == "12gb" else 2
    try:
        return max(1, min(8, int(os.environ.get("ROOP3D_VIEW_WORKERS", default))))
    except ValueError:
        return default


class NeuralFaceEngine:
    def __init__(self, backbone, rasterizer, *, profile="6gb", parser=None):
        self.backbone, self.rasterizer = backbone, rasterizer
        # The 2-4 source views of a frame are independent; ROOP3D_VIEW_WORKERS=1 restores the one-after-another loop.
        self.view_workers = ViewWorkers(view_worker_count(profile), rasterizer)
        # Target fits for frame N+1 are made on the (idle) track thread while the swap thread works on frame N.
        self.fit_ahead = FitAhead(backbone) if fit_ahead_enabled() else None
        # Face parser (``predict(bgr) -> HxWx19 class probabilities``) that keeps hair, neck, cloth and background texels
        # of the source photographs out of the appearance sampled for the source surface.
        self.parser = parser
        self.parser_unavailable: str | None = None
        self.surface = getattr(backbone, "surface", backbone.topology)
        self.profile = profile
        self.sources = {}
        self.source_digests = {}
        self.source_shapes = {}
        self.last_source_shape_scale_ratio = 1.0
        self.last_fields = ()
        self.last_original_fields = ()
        self.last_source_view_indices = ()
        self.last_diag = None       # angle_diagnostics input of the latest frame (see engine.volumetric.angle_diagnostics)
        # Per-track, size-normalised filter of the target fit and the landmarks that register the crop (see process()).
        self.fit_smoother = CoupledTrackSmoother()
        self.view_selector = ViewSelector()
        self.neutral_pitch: dict[str, float] = {}

    @classmethod
    def load(cls, catalog, config):
        from roop_3d.compositor.masks import BisenetFaceParser
        backbone = MorphableBackbone.load(catalog, config)
        rasterizer = GPUSurfaceRasterizer()
        try:
            rasterizer._context()
            return cls(backbone, rasterizer, profile=config.hardware_profile, parser=BisenetFaceParser(catalog, config))
        except BaseException:
            rasterizer.release()
            backbone.release()
            raise

    def prepare(self, document):
        # A warm engine outlives one preview: reuse a prepared FaceSet only while its content is unchanged.
        if document.faceset_id in self.sources and self.source_digests.get(document.faceset_id) == getattr(document, "source_sha256", None):
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
            photograph = self._photograph_cache(image, matrix, side)
            observed = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (side, side),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            observed[self._non_skin(photograph)] = 0
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
        if self.parser is not None and hasattr(self.parser, "release"):
            self.parser.release()           # parsing is a one-off per FaceSet; the compositor owns the resident parser
        self.sources[document.faceset_id] = tuple(surfaces)
        self.source_digests[document.faceset_id] = getattr(document, "source_sha256", None)
        self.source_shapes[document.faceset_id] = source_shape
        self.neutral_pitch[document.faceset_id] = neutral_pitch([s.view for s in surfaces])

    # BiSeNet classes that are not the face: background, glasses, ears, neck, cloth, hair, hat.
    NON_SKIN_CLASSES = (0, *STRICT_FORBIDDEN_CLASSES)

    def _non_skin(self, photograph) -> np.ndarray:
        """Texels of a source photograph that are hair, neck, cloth, hat, glasses, ears or background (bool, dilated).

        Being inside the photograph is not being skin: without this, ``observed`` accepted a fringe of hair, a collar or
        a hairline shadow, and the correspondence warp then painted that texture onto the face. A texel rejected here is
        left to the same fallback fill as any other unobserved texel (mirror, else the source's median skin tone).
        """
        if self.parser is None or self.parser_unavailable:
            return np.zeros(photograph.shape[:2], bool)
        try:
            probabilities = np.asarray(self.parser.predict(photograph), np.float32)
        except DomainError as error:
            # The parser model is optional everywhere else in the pipeline; the loss is reported, never hidden.
            self.parser_unavailable = f"{error.code}: {error.message}"
            log.warning("Source photograph parsing unavailable (%s); non-skin texels are not rejected", self.parser_unavailable)
            return np.zeros(photograph.shape[:2], bool)
        bad = (probabilities[..., list(self.NON_SKIN_CLASSES)].sum(-1) > 0.5).astype(np.uint8)
        radius = max(2, round(photograph.shape[0] * 0.006))
        return cv2.dilate(bad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0

    @staticmethod
    def _photograph_cache(image, matrix, side: int):
        """The original photograph in the aligned ``side`` px cache (one resampling of the original pixels).

        A face photograph is usually larger than its cache; ``warpAffine`` has no mip-mapping and aliases a shrink, so
        the photograph is first area-averaged to about the cache scale. Up-scaling is bicubic; the border replicates so
        no dark bleed enters the taps next to the photograph's edge (``observed`` marks that region unusable).
        """
        matrix = np.asarray(matrix, np.float64)
        height, width = image.shape[:2]
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        if scale < 0.75:
            reduced = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                                 interpolation=cv2.INTER_AREA)
            sx, sy = reduced.shape[1] / width, reduced.shape[0] / height
            # INTER_AREA maps original pixel x to reduced pixel sx*x + (sx-1)/2.
            to_original = np.array(((1 / sx, 0, -(sx - 1) / (2 * sx)), (0, 1 / sy, -(sy - 1) / (2 * sy)), (0, 0, 1)))
            matrix = (np.vstack((matrix, (0, 0, 1))) @ to_original)[:2]
            image = reduced
        return cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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
    def _compose_affines(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        return (np.vstack((outer, (0.0, 0.0, 1.0)))
                @ np.vstack((inner, (0.0, 0.0, 1.0))))[:2]

    @staticmethod
    def _smoothed_fit(target, step):
        """The target fit with every quantity that moves the projected surface filtered under the frame's shared beta.

        The monocular regressor answers each frame independently, so its rotation, translation, scale, intrinsics and
        expression carry frame-to-frame noise the mask edge then shows as shimmer. Rotation is filtered on SO(3), the scale
        in log space, and the expression as a displacement from the (equally filtered) neutral shape.
        """
        if getattr(target, "expression_vertices", None) is None:
            raise DomainError("DENSE_FACE_EXPRESSION_GEOMETRY_REQUIRED",
                              "Target fit must expose expression-only geometry before detail offsets", 422)
        identity = np.asarray(target.identity, np.float64)
        displacement = np.asarray(target.expression_vertices, np.float64) - identity
        identity = step.filter("identity", identity)
        intrinsics = step.filter("intrinsics", target.intrinsics)
        intrinsics[2] = (0.0, 0.0, 1.0)          # the homogeneous row stays exact whatever beta rounds to
        return SimpleNamespace(
            identity=identity.astype(np.float32), vertices=target.vertices, alignment=target.alignment,
            expression_vertices=(identity + step.filter("expression", displacement)).astype(np.float32),
            rotation=step.filter_rotation("rotation", target.rotation),
            translation=step.filter("translation", target.translation).astype(np.float32),
            scale=float(np.exp(step.filter("log_scale", np.log(float(target.scale))))),
            intrinsics=intrinsics)

    def reset_tracks(self) -> None:
        """Forget every track's filtered fit and view selection (a warm engine must not continue an earlier render)."""
        self.fit_smoother.reset()
        self.view_selector.reset()
        if self.fit_ahead is not None:
            self.fit_ahead.clear()

    def begin_fit_ahead(self, base_frame: int) -> None:
        """A pipelined render is starting: ``base_frame`` is the ``frame_index`` its first frame will be processed with."""
        if self.fit_ahead is not None:
            self.fit_ahead.begin(base_frame)

    def retain_tracks(self, active_ids) -> None:
        self.fit_smoother.retain(active_ids)
        self.view_selector.retain(active_ids)

    def capability(self):
        return {**self.backbone.capability(), "prepared_identities": len(self.sources),
                "source_texture": "original_photographs", "visibility": "source_and_target_depth_test",
                "geometry_identity_source": "faceset_multiview_neutral",
                "target_geometry_conditioning": "source_shape+normalized_target_expression",
                "target_pixels_in_source_holes": False,
                "conditioning_crop": "projected_source_mesh_in_target_camera",
                "warp_coordinate_space": "aligned_source_crop_pixels", "original_warp_fields": True,
                "source_texel_parsing": (f"unavailable: {self.parser_unavailable}" if self.parser_unavailable
                                         else "bisenet_non_skin_rejected" if self.parser is not None else "none"),
                "source_views_per_frame": 4 if self.profile == "12gb" else 2}

    def release(self):
        self.sources.clear()
        self.source_digests.clear()
        self.source_shapes.clear()
        self.last_source_shape_scale_ratio = 1.0
        self.last_fields = ()
        self.last_original_fields = ()
        self.last_source_view_indices = ()
        self.last_diag = None
        self.reset_tracks()
        self.neutral_pitch.clear()
        self.view_workers.close()
        self.rasterizer.release()
        self.backbone.release()
