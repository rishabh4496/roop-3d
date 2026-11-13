"""Face-only multiview correspondence conditioning followed by neural synthesis."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace

import cv2
import numpy as np

from roop_3d.compositor.masks import STRICT_FORBIDDEN_CLASSES
from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import AlignedCrop, registration_matrix, transform_points
from roop_3d.reconstruction.dense_correspondence import (
    GPUSurfaceRasterizer, ViewWorkers, WarpField, correspondence_field,
)
from roop_3d.reconstruction.fit_ahead import FitAhead, fit_ahead_enabled
from roop_3d.reconstruction.morphable_backbone import MorphableBackbone
from roop_3d.reconstruction.swap_coverage import feathered_coverage
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
    def __init__(self, backbone, rasterizer, enhancer, *, profile="6gb", parser=None):
        self.backbone, self.rasterizer, self.enhancer = backbone, rasterizer, enhancer
        # The 2-4 source views of a frame are independent; ROOP3D_VIEW_WORKERS=1 restores the one-after-another loop.
        self.view_workers = ViewWorkers(view_worker_count(profile), rasterizer)
        # Target fits for frame N+1 are made on the (idle) track thread while the swap thread works on frame N.
        self.fit_ahead = FitAhead(backbone) if fit_ahead_enabled() else None
        # Face parser (``predict(bgr) -> HxWx19 class probabilities``) that keeps hair, neck, cloth and background texels
        # of the source photographs out of the appearance the swapper is conditioned on.
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
        from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
        backbone = MorphableBackbone.load(catalog, config)
        rasterizer = GPUSurfaceRasterizer()
        try:
            rasterizer._context()
            return cls(backbone, rasterizer, UltraMaxPipelineRestorer(catalog, config),
                       profile=config.hardware_profile, parser=BisenetFaceParser(catalog, config))
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

    def process(self, image, face, document, swapper, *, track_id, frame_index=None, timestamp=None, scene_cut=False,
                pose_confidence=1.0):
        """``frame_index`` (consecutive per render), ``timestamp`` (seconds) and ``scene_cut`` let the per-track fit filter
        know which frames follow each other; without a ``frame_index`` nothing is filtered (a one-off preview)."""
        self.last_fields = self.last_original_fields = self.last_source_view_indices = ()
        self.last_diag = None
        if document.faceset_id not in self.sources:
            raise DomainError("DENSE_FACE_NOT_PREPARED", "Prepare source reconstruction before frame inference", 409)
        # The same filtered landmarks align the image for the regressor, register the swap crop and align UltraMax: the
        # fit, the crop and the surface rendered into it share one trajectory.
        landmarks5, step = np.asarray(face.landmarks5, np.float64), None
        if frame_index is not None:
            step = self.fit_smoother.advance(track_id, frame_index, landmarks5, timestamp=timestamp, scene_cut=scene_cut)
            landmarks5 = step.landmarks
        # In a pipelined render the track thread may already have fitted this exact frame (see fit_ahead); the fit is used
        # only if it was made from these very landmarks and this very image, otherwise it is computed here as always.
        target = None
        if step is not None and self.fit_ahead is not None:
            target = self.fit_ahead.take(track_id, frame_index, image, landmarks5, timestamp)
        if target is None:
            target = self.backbone.fit(image, landmarks5)
        if step is not None:
            target = self._smoothed_fit(target, step)
        model = swapper.model
        if model is None or not callable(getattr(model, "infer", None)):
            raise DomainError("DENSE_FACE_NEURAL_REQUIRED", "Dense face output requires a prepared neural swapper", 503)
        size = model.resolution
        surfaces = {s.view.index: s for s in self.sources[document.faceset_id]}
        color = np.zeros((size, size, 3), np.float32)
        support = np.zeros((size, size), np.float32)
        fields, original_fields, view_indices, view_log = [], [], [], []
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
        # The swap crop is registered exactly as the 2D swap path registers its crop: a similarity of the tracker-filtered
        # five landmarks onto the swapper's template. Eyes, nose and mouth therefore land on the same crop pixels at every
        # yaw/pitch and mouth opening, and the swapper and UltraMax see the framing they were trained on. The mesh is only
        # rasterised into this crop (never used to frame it): a silhouette bounding box breathes with yaw and jaw opening.
        template = getattr(model, "spec", {}).get("alignment_template", "arcface_112")
        crop_matrix = registration_matrix(landmarks5, size, template)
        source_to_fit = np.asarray(target.alignment.matrix, np.float64) * (224.0 / 128.0)
        mesh_affine = self._compose_affines(crop_matrix, cv2.invertAffineTransform(source_to_fit))
        aligned = AlignedCrop(np.zeros((size, size, 3), np.uint8), crop_matrix,
                             cv2.invertAffineTransform(crop_matrix))
        k = np.vstack((mesh_affine, (0.0, 0.0, 1.0))) @ target.intrinsics
        # Which photographs, and how strongly: lowest pose cost against the filtered tracker pose (view pitches relative to
        # the FaceSet's own neutral), switched only with hysteresis and faded in and out over a few frames.
        pitch_zero = self.neutral_pitch.get(document.faceset_id, 0.0)
        distances = {i: float(np.linalg.norm(s.fit.rotation - target.rotation)) for i, s in surfaces.items()}
        costs = {i: view_cost(face.pose, (s.view.yaw, s.view.pitch - pitch_zero, getattr(s.view, "roll", 0.0)),
                              s.view.quality.score, pose_confidence, rotation_distance=distances[i])
                 for i, s in surfaces.items()}
        chosen = self.view_selector.select(
            track_id, frame_index, costs, 4 if self.profile == "12gb" else 2,
            angular={i: float(np.exp(-2 * d)) for i, d in distances.items()}, timestamp=timestamp, scene_cut=scene_cut)
        ordered = sorted(chosen.items(), key=lambda item: (costs[item[0]], item[0]))

        def field_job(source):
            return lambda: correspondence_field(self.rasterizer, target_faces,
                source.fit.camera_vertices, source.intrinsics, target_camera_vertices, k, (size, size),
                source.depth, (source.image.shape[1], source.image.shape[0]), source_support=source.observed)
        # The fields are computed side by side but consumed in `ordered`, so the accumulated colour is unchanged.
        computed = self.view_workers.run([field_job(surfaces[index]) for index, _ in ordered])
        for (index, factor), field in zip(ordered, computed):
            source = surfaces[index]
            weight = field.confidence * max(source.view.quality.score, .01) * factor
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
            view_log.append({"index": source.view.index, "yaw": source.view.yaw, "pitch": source.view.pitch,
                             "raw": float(costs[index]),
                             "weight": float(weight.sum())})
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
        crop_points = transform_points(landmarks5, aligned.matrix)
        enhanced = self.enhancer.restore_hires(neural, crop_points)
        if enhanced is None or enhanced.dtype != np.uint8 or enhanced.ndim != 3 or enhanced.shape[2] != 3:
            raise DomainError("DENSE_FACE_ENHANCER_FAILED", "UltraMax did not return a neural face image", 503)
        # The anatomical facial surface of the target is the replacement boundary: appearance confidence only governs
        # photograph sampling, the neural model owns unobserved facial skin too (source holes are synthesis from the
        # identity embedding, never observed appearance and never a shader/albedo completion). It is rasterised at the
        # resolution the face is composited at, never upsampled from the swap crop, then intersected with the landmark hull
        # and feathered in proportion to the face's own extent.
        height, width = enhanced.shape[:2]
        to_output = np.diag((width / size, height / size, 1.0))
        surface = self.rasterizer.rasterize(target_camera_vertices, target_faces, to_output @ k,
                                            target_camera_vertices, (width, height)).valid
        hull_landmarks = getattr(face, "landmarks106", None)
        coverage = feathered_coverage(
            surface,
            landmarks106=None if hull_landmarks is None else transform_points(hull_landmarks, crop_matrix) * (width / size, height / size),
            landmarks5=crop_points * (width / size, height / size))
        inverse = aligned.inverse.copy()
        inverse[:, 0] *= size/width
        inverse[:, 1] *= size/height
        points = crop_points * (width/size, height/size)
        total_weight = max(sum(v["weight"] for v in view_log), 1e-12)
        for entry in view_log:
            entry["weight"] /= total_weight
        self.last_diag = {"crop_matrix": crop_matrix, "landmarks5": np.asarray(face.landmarks5, np.float64),
                          "coverage": coverage, "coverage_scale": width / size, "views": view_log}
        return NeuralFaceResult(enhanced, inverse, coverage, points)

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
                "source_views_per_frame": 4 if self.profile == "12gb" else 2,
                "enhancer": self.enhancer.capability()}

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
        self.enhancer.release()
        self.view_workers.close()
        self.rasterizer.release()
        self.backbone.release()
