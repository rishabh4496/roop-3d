"""Verified local INSwapper identity transfer; no application-code reuse."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from collections.abc import Mapping
from typing import Any, BinaryIO, Iterator
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_pose3d, pose_cell
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, tensor
from roop_3d.engine.volumetric.preprocess import AlignedCrop, roll_compensated_aligned_crop, transform_points
from roop_3d.engine.volumetric.swapper_engine import SwapperEngineAdapter
from roop_3d.faceset.pose_selector import PoseSelector
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.models.catalog import ModelCatalog
from roop_3d.pose_matrix import FrozenSourcePoseMatrix, FrozenPoseSelection
from roop_3d.reconstruction.frontalization import frontalize_crop

log = logging.getLogger("roop3d.stage13")
stage0_log = logging.getLogger("roop3d.stage0")


def _varint(stream: BinaryIO) -> int:
    value = 0
    for index in range(10):
        byte = stream.read(1)
        if not byte:
            raise ValueError("Truncated protobuf field")
        value |= (byte[0] & 127) << (7 * index)
        if byte[0] < 128:
            return value
    raise ValueError("Oversized protobuf varint")


def _fields(stream: BinaryIO, end: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield field number, wire type, value/offset and length; skip tensor payloads."""
    while stream.tell() < end:
        tag = _varint(stream)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            yield field, wire, _varint(stream), 0
        elif wire in (1, 2, 5):
            size = _varint(stream) if wire == 2 else (8 if wire == 1 else 4)
            offset = stream.tell()
            if offset + size > end:
                raise ValueError("Protobuf length exceeds verified artifact")
            yield field, wire, offset, size
            stream.seek(offset + size)
        else:
            raise ValueError("Unsupported protobuf wire type")


def embedding_map(path: Path) -> np.ndarray:
    """Read the observed buff2fs tensor without importing ONNX or loading all weights.

    ModelProto.graph=7; GraphProto.initializer=5; TensorProto dims/type/name/raw
    fields are 1/2/8/9. Only the verified float32 512x512 map is materialized.
    """
    try:
        with path.open("rb") as stream:
            for field, wire, graph_offset, graph_size in _fields(stream, path.stat().st_size):
                if field != 7 or wire != 2:
                    continue
                stream.seek(graph_offset)
                for number, kind, offset, size in _fields(stream, graph_offset + graph_size):
                    if number != 5 or kind != 2:
                        continue
                    stream.seek(offset)
                    name, dims, dtype, raw = "", [], None, None
                    for member, encoding, position, length in _fields(stream, offset + size):
                        if member == 1 and encoding == 0:
                            dims.append(position)
                        elif member == 2 and encoding == 0:
                            dtype = position
                        elif member == 8 and encoding == 2:
                            if length > 1024:
                                raise ValueError("Tensor name exceeds bound")
                            name = stream.read(length).decode("utf-8")
                        elif member in (4, 9) and encoding == 2:
                            raw = (position, length)
                    if name == "buff2fs":
                        if dims != [512, 512] or dtype not in (1, 10) or raw is None:
                            raise ValueError("Unsupported embedding projection contract")
                        item_size = 4 if dtype == 1 else 2
                        if raw[1] != 512 * 512 * item_size:
                            raise ValueError("Unsupported embedding projection payload")
                        stream.seek(raw[0])
                        dtype_name = "<f4" if dtype == 1 else "<f2"
                        return tensor(np.frombuffer(stream.read(raw[1]), dtype=dtype_name).reshape(512, 512).astype(np.float32),
                                      (512, 512))
    except (OSError, ValueError):
        raise DomainError("SWAPPER_PROJECTION_INVALID", "Verified swapper projection could not be read", 503) from None
    raise DomainError("SWAPPER_PROJECTION_MISSING", "Swapper has no supported embedding projection", 503)


def unit_embedding(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    tensor(value, (512,))
    norm = np.linalg.norm(value)
    if norm < 1e-8:
        raise DomainError("IDENTITY_EMBEDDING_INVALID", "Identity embedding must have nonzero norm", 422)
    return np.ascontiguousarray(value / norm)


@dataclass(frozen=True)
class SwapResult:
    image: np.ndarray
    alignment: AlignedCrop
    reference_index: int
    reference_weight: float


class IdentitySwapper:
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig(),
                 model_id: str = "inswapper_128"):
        self.catalog, self.config = catalog or ModelCatalog(), config
        self.model_id = model_id
        self.model: SwapperEngineAdapter | None = None
        self.projection: np.ndarray | None = None
        self.source: FaceSetDocument | None = None
        self.canonical: np.ndarray | None = None
        self.sources_by_track: dict[int, FaceSetDocument] = {}
        self.canonical_by_track: dict[int, np.ndarray] = {}
        self._selectors: dict[int, PoseSelector] = {}
        self._matrices: dict[int | None, FrozenSourcePoseMatrix | None] = {}
        self._logged_cells: set[tuple[int | None, int]] = set()

    def prepare_session(self, source: FaceSetDocument, *,
                        sources_by_track: Mapping[int, FaceSetDocument] | None = None) -> None:
        if not source.views:
            raise DomainError("FACESET_EMPTY", "Identity transfer requires observed source views", 422)
        mapped = dict(sources_by_track or {})
        if any(type(track_id) is not int or track_id < 0 or not document.views
               for track_id, document in mapped.items()):
            raise DomainError("FACESET_MAPPING_INVALID", "Mapped tracks require nonempty verified FaceSets", 422)
        canonical = unit_embedding(np.asarray(source.canonical_embedding))
        canonical_by_track = {track_id: unit_embedding(np.asarray(document.canonical_embedding))
                              for track_id, document in mapped.items()}
        self.release_session()
        try:
            self.model = SwapperEngineAdapter(self.catalog, self.model_id, self.config)
            self.projection = None
            if self.model.spec.get("identity_mode") == "arcface_emap":
                from roop_3d.models.emap_cache import cached_emap
                path = self.model.path or getattr(self.model.model, "path", None)
                record = getattr(self.model.model, "record", None)
                digest = record.get("sha256") if isinstance(record, dict) else None
                self.projection = cached_emap(path, embedding_map, source_sha256=digest) if digest else embedding_map(path)
            self.source, self.canonical = source, canonical
            self.sources_by_track = mapped
            self.canonical_by_track = canonical_by_track
            self._selectors = {track_id: PoseSelector(document) for track_id, document in mapped.items()}
            # Stage 13: freeze pose-selected identities and EMAP projections before frame 0.
            self._logged_cells = set()
            device = "cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu"
            self._matrices = {None: FrozenSourcePoseMatrix.from_document(
                source, projection=self.projection, device=device)}
            self._matrices.update({track_id: FrozenSourcePoseMatrix.from_document(
                document, projection=self.projection, device=device) for track_id, document in mapped.items()})
        except BaseException:
            self.release_session()
            raise

    def set_track_source(self, track_id: int, document: FaceSetDocument) -> None:
        """Change an explicit source without restarting detection or loading another model."""
        self.sources_by_track[track_id] = document
        self.canonical_by_track[track_id] = unit_embedding(np.asarray(document.canonical_embedding))
        self._selectors[track_id] = PoseSelector(document)
        device = "cuda" if self.config.provider == "CUDAExecutionProvider" else "cpu"
        self._matrices[track_id] = FrozenSourcePoseMatrix.from_document(
            document, projection=self.projection, device=device)

    def _matrix_for(self, track_id: int | None) -> FrozenSourcePoseMatrix | None:
        return self._matrices.get(track_id) if track_id in self._matrices else self._matrices.get(None)

    def pose_selection(self, track_id: int | None, pose: Any) -> FrozenPoseSelection | None:
        matrix = self._matrix_for(track_id)
        return matrix.lookup(pose) if matrix is not None else None

    def reference_embedding(self, track_id: int | None, pose: Any, reference_blend: float = 0.15
                            ) -> tuple[np.ndarray, int, float]:
        """The identity vector for a target pose: (unit embedding, reference view index, reference weight).

        `pose` may be a 2-/3-tuple, array, dict or `Pose3D` (legacy frame unless it is a `Pose3D`).
        The render-session matrix uses smooth bilinear interpolation over its frozen 3x3 or 96-bin vectors;
        `PoseSelector` remains the fallback for a source without a usable pose matrix.
        """
        if self.source is None or self.canonical is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare identity source before inference", 409)
        target = as_pose3d(pose)
        source = self.sources_by_track.get(track_id, self.source) if track_id is not None else self.source
        canonical = self.canonical_by_track.get(track_id, self.canonical) if track_id is not None else self.canonical
        matrix = self._matrix_for(track_id)
        if matrix is not None:
            selection = matrix.lookup(target)
            marker = (track_id, selection.view_index)
            if marker not in self._logged_cells:
                self._logged_cells.add(marker)
                log.info("[Stage13] lookup track=%s yaw=%.1f pitch=%.1f roll=%.1f -> source=%s view=%d cells=%s",
                         track_id, target.yaw, target.pitch, target.roll, getattr(selection, "source", "matrix"),
                         selection.view_index, getattr(selection, "weights", ()))
                stage0_log.info("[Stage0] lookup track=%s yaw=%.1f pitch=%.1f roll=%.1f cell=%s view=%d",
                                track_id, target.yaw, target.pitch, target.roll, pose_cell(target).name,
                                selection.view_index)
            else:
                log.debug("[Stage13] lookup track=%s -> view=%d", track_id, selection.view_index)
                stage0_log.debug("[Stage0] lookup track=%s -> cell=%s view=%d", track_id,
                                 pose_cell(target).name, selection.view_index)
            weight, view_vector, index = reference_blend * selection.confidence, selection.vector, selection.view_index
            if (hasattr(matrix, "reference_blend") and hasattr(selection, "pose_selected_vector")
                    and abs(reference_blend - matrix.reference_blend) < 1e-8):
                return selection.pose_selected_vector.copy(), index, weight
        else:
            selector = self._selectors.get(track_id) if track_id is not None else None
            if selector is None:
                selector = PoseSelector(source)
            match = selector.select_best_reference(target.legacy_tuple())
            view = next(v for v in source.views if v.index == match.view_index)
            weight, view_vector, index = reference_blend * match.confidence, np.asarray(view.embedding), match.view_index
        return unit_embedding(canonical * (1 - weight) + unit_embedding(np.asarray(view_vector)) * weight), index, weight

    def infer_frame(self, image: np.ndarray, landmarks5: np.ndarray, *, pose: Any = (0, 0, 0),
                    reference_blend: float = 0.15, track_id: int | None = None, frontalization: float = 0.0, identity_blend_t: float | None = None) -> SwapResult:
        if self.model is None or self.source is None or self.canonical is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare identity source before inference", 409)
        if not 0 <= reference_blend <= 0.25:
            raise ValueError("Pose reference weight must be within 0..0.25")
        if not np.isfinite(frontalization) or not 0 <= frontalization <= 1:
            raise ValueError("Pose frontalization must be within 0..1")
        if track_id is not None and (type(track_id) is not int or track_id < 0):
            raise ValueError("track_id must be a nonnegative integer")
        embedding, view_index, weight = self.reference_embedding(track_id, pose, reference_blend)
        if identity_blend_t is not None:
            if not np.isfinite(identity_blend_t) or not 0 <= identity_blend_t <= 1:
                raise ValueError('Identity blend must be within 0..1')
            source = self.sources_by_track.get(track_id, self.source)
            pair = source.metadata.get('identity_curve_pair')
            if pair is None or len(pair) != 2:
                raise DomainError('IDENTITY_PAIR_REQUIRED','Choose a two-source identity mixer FaceSet before animating identity_blend_t',422)
            a,b = (unit_embedding(np.asarray(v)) for v in pair)
            cosine=float(np.clip(a@b,-1,1))
            angle=float(np.arccos(cosine))
            if cosine < -.9999:
                raise DomainError('IDENTITY_PAIR_ANTIPODAL','Antipodal identities have no unique Slerp path',422)
            t=identity_blend_t
            embedding=unit_embedding((1-t)*a+t*b if angle < 1e-5 else
                (np.sin((1-t)*angle)*a+np.sin(t*angle)*b)/np.sin(angle))
        selection = self.pose_selection(track_id, pose)
        matrix = self._matrix_for(track_id)
        if (identity_blend_t is None and matrix is not None and hasattr(matrix, "reference_blend")
                and hasattr(selection, "projected_vector")
                and abs(reference_blend - matrix.reference_blend) < 1e-8):
            # Per-cell projection (EMAP or identity) was completed during session preparation.
            projected = selection.projected_vector.reshape(1, 512)
        elif self.projection is not None:
            projected = unit_embedding(embedding @ self.projection).reshape(1, 512)
        else:
            projected = unit_embedding(embedding).reshape(1, 512)
        target_pose = as_pose3d(pose)
        compensation = None
        if matrix is not None and callable(getattr(matrix, "roll_affine", None)):
            center = tuple(float(v) for v in np.asarray(landmarks5, np.float64)[:2].mean(axis=0))
            compensation = matrix.roll_affine(target_pose.roll, center)
        model_resolution = getattr(self.model, "resolution", 128)
        model_spec = getattr(self.model, "spec", {})
        model_id = getattr(self.model, "model_id", "inswapper_128")
        aligned = roll_compensated_aligned_crop(image, landmarks5, target_pose.roll, model_resolution,
                                                compensation=compensation,
                                                template=model_spec.get("alignment_template", "arcface_112"))
        model_input, flat = aligned.image, None
        if frontalization > 0 and model_id == "inswapper_128":
            # Reduce severe profile mismatch before inswapper_128 sees the crop; undone on the output below.
            flat = frontalize_crop(aligned.image, transform_points(np.asarray(landmarks5, np.float64), aligned.matrix), frontalization)
            model_input = flat.image
        document = self.sources_by_track.get(track_id, self.source) if track_id is not None else self.source
        source_view = next((view for view in document.views if view.index == view_index), document.views[0])
        if callable(getattr(self.model, "infer", None)):
            bgr = self.model.infer(model_input, projected.reshape(512), source_view=source_view)
        else:
            # Keep the established raw-session seam used by diagnostics and test doubles.
            blob = model_input[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
            result = self.model.run({"target": np.ascontiguousarray(blob), "source": projected})["output"]
            tensor(result, (1, 3, model_resolution, model_resolution))
            bgr = np.clip(result[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
        if flat is not None:
            bgr = flat.restore(bgr)
        return SwapResult(bgr, aligned, view_index, weight)

    def release_session(self) -> None:
        if self.model is not None:
            self.model.release()
        self.model = self.projection = self.source = self.canonical = None
        self.sources_by_track.clear()
        self.canonical_by_track.clear()
        self._selectors.clear()
        self._matrices.clear()
        self._logged_cells.clear()

    def refine_head_texture(self, rendered, coverage, *, pose, track_id=None):
        """Legacy entry point: return full neural synthesis, never a CG residual."""
        import cv2
        if self.model is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare a neural swapper before refinement", 409)
        size = self.model.resolution
        neural = self.synthesize_correspondence(cv2.resize(rendered, (size, size), interpolation=cv2.INTER_AREA),
                                               pose=pose, track_id=track_id)
        return cv2.resize(neural, (rendered.shape[1], rendered.shape[0]), interpolation=cv2.INTER_LINEAR)

    def synthesize_correspondence(self, conditioned_bgr, *, pose, track_id=None):
        """Native aligned photographic warp + identity embedding -> full neural face."""
        if self.model is None or self.source is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare identity source before neural synthesis", 409)
        embedding, index, _ = self.reference_embedding(track_id, pose, .15)
        projected = unit_embedding(embedding @ self.projection) if self.projection is not None else unit_embedding(embedding)
        document = self.sources_by_track.get(track_id, self.source)
        view = next((v for v in document.views if v.index == index), document.views[0])
        if conditioned_bgr.shape != (self.model.resolution, self.model.resolution, 3):
            raise ValueError("Correspondence image must match the selected swapper's native alignment and resolution")
        neural = self.model.infer(conditioned_bgr, projected, source_view=view)
        if neural.shape != conditioned_bgr.shape or neural.dtype != np.uint8:
            raise DomainError("DENSE_FACE_NEURAL_OUTPUT_INVALID", "Neural swapper output has an invalid image contract", 503)
        return neural
