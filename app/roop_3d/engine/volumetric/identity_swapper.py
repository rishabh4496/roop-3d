"""Verified local INSwapper identity transfer; no application-code reuse."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import BinaryIO, Iterator
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, tensor
from roop_3d.engine.volumetric.preprocess import AlignedCrop, aligned_crop
from roop_3d.faceset.pose_selector import PoseSelector
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.models.catalog import ModelCatalog


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
                        if dims != [512, 512] or dtype != 1 or raw is None or raw[1] != 512 * 512 * 4:
                            raise ValueError("Unsupported embedding projection contract")
                        stream.seek(raw[0])
                        return tensor(np.frombuffer(stream.read(raw[1]), dtype="<f4").reshape(512, 512).copy(), (512, 512))
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
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog or ModelCatalog(), config
        self.model: OrtModel | None = None
        self.projection: np.ndarray | None = None
        self.source: FaceSetDocument | None = None
        self.canonical: np.ndarray | None = None
        self.sources_by_track: dict[int, FaceSetDocument] = {}
        self.canonical_by_track: dict[int, np.ndarray] = {}
        self._selectors: dict[int, PoseSelector] = {}

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
            self.model = OrtModel(self.catalog, "inswapper_128.onnx", self.config)
            self.projection = embedding_map(self.model.path)
            self.source, self.canonical = source, canonical
            self.sources_by_track = mapped
            self.canonical_by_track = canonical_by_track
            self._selectors = {track_id: PoseSelector(document) for track_id, document in mapped.items()}
        except BaseException:
            self.release_session()
            raise

    def set_track_source(self, track_id: int, document: FaceSetDocument) -> None:
        """Change an explicit source without restarting detection or loading another model."""
        self.sources_by_track[track_id] = document
        self.canonical_by_track[track_id] = unit_embedding(np.asarray(document.canonical_embedding))
        self._selectors[track_id] = PoseSelector(document)

    def infer_frame(self, image: np.ndarray, landmarks5: np.ndarray, *, pose: tuple[float, float, float] = (0, 0, 0),
                    reference_blend: float = 0.15, track_id: int | None = None) -> SwapResult:
        if self.model is None or self.source is None or self.canonical is None or self.projection is None:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare identity source before inference", 409)
        if not 0 <= reference_blend <= 0.25:
            raise ValueError("Pose reference weight must be within 0..0.25")
        if track_id is not None and (type(track_id) is not int or track_id < 0):
            raise ValueError("track_id must be a nonnegative integer")
        source = self.sources_by_track.get(track_id, self.source) if track_id is not None else self.source
        selector = self._selectors.get(track_id) if track_id is not None else None
        if selector is None:
            selector = PoseSelector(source)
        match = selector.select_best_reference(*pose)
        view = next(v for v in source.views if v.index == match.view_index)
        weight = reference_blend * match.confidence
        canonical = self.canonical_by_track.get(track_id, self.canonical) if track_id is not None else self.canonical
        assert canonical is not None
        embedding = unit_embedding(canonical * (1 - weight) + unit_embedding(np.asarray(view.embedding)) * weight)
        projected = unit_embedding(embedding @ self.projection).reshape(1, 512)
        aligned = aligned_crop(image, landmarks5, 128)
        blob = aligned.image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        result = self.model.run({"target": np.ascontiguousarray(blob), "source": projected})["output"]
        tensor(result, (1, 3, 128, 128))
        bgr = np.clip(result[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
        return SwapResult(bgr, aligned, match.view_index, weight)

    def release_session(self) -> None:
        if self.model is not None:
            self.model.release()
        self.model = self.projection = self.source = self.canonical = None
        self.sources_by_track.clear()
        self.canonical_by_track.clear()
        self._selectors.clear()
