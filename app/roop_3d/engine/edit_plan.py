"""Validated role intervals, source changes and manual masks; excluded/reference tracks never render a head.

A manual mask is a frame-space occluder plane (float32 ``H x W``, 1 = keep the target pixel). It reaches the compositor as
``Replacement.occluder`` (:meth:`FrameEditPlan.with_occluders`): the full-frame 3D raster has no face-crop space to warp it into.
Force-include brush strokes (the studio brush's red channel) have no equivalent in that compositor, so a plan carrying them is
refused when it is built instead of silently ignoring them.
"""
from contextlib import nullcontext
from dataclasses import replace

from roop_3d.core.errors import DomainError
from roop_3d.audit.isolation import safe_path
from roop_3d.media.schema import FaceInspection
import base64
import cv2
import numpy as np


class FrameEditPlan:
    def __init__(self, runner, request, sources):
        self.request = request
        self.segments = request.face_segments
        self.sources = dict(sources)
        self.current = {}
        self.documents = {}
        self.mask = None
        self.mask_channels = None
        self.mask_keyframes = []
        self._planes = {}
        if request.mask_base64:
            self.mask, self.mask_channels = self._decode_mask(request.mask_base64)
            self._require_supported(self.mask_channels)
        for keyframe in sorted(getattr(request, 'mask_keyframes', None) or (), key=lambda item: item.start_frame):
            decoded = self._decode_mask(keyframe.mask_base64)
            self._require_supported(decoded[1])
            self.mask_keyframes.append((keyframe, *decoded))
        if self.segments is None:
            return
        path = safe_path(runner.root, f'roop_3d/.runtime/inspections/{request.inspection_id}.json')
        inspection = FaceInspection.model_validate_json(path.read_text('utf-8'))
        tracks = {track.track_id: track for track in inspection.tracks}
        self.ids = {}
        for segment in self.segments:
            track = tracks.get(segment.track_id)
            if track is None:
                raise DomainError('TRACK_NOT_FOUND', 'Face role refers to a track absent from inspection', 422)
            track_id = track.persistent_id if track.persistent_id is not None else int(segment.track_id.rsplit('_', 1)[1])
            self.ids[segment.track_id] = track_id
            if segment.role != 'swap':
                continue
            key = (segment.faceset_id, segment.source_view_index)
            if key in self.documents:
                continue
            document = next(iter(sources.values())) if request.identity_blend else next((item for item in sources.values() if item.faceset_id == segment.faceset_id), None)
            if document is None:
                document = runner.faceset_reader.read(runner._faceset_path(segment.faceset_id), faceset_id=segment.faceset_id)
            if segment.source_view_index is not None:
                view = next((view for view in document.views if view.index == segment.source_view_index), None)
                if view is None:
                    raise DomainError('SOURCE_VIEW_NOT_FOUND', 'Selected source face is absent from FaceSet', 422)
                document = document.model_copy(update={'views': (view,), 'canonical_embedding': view.embedding,
                    'medoid_embedding': view.embedding, 'pose_bins': {}})
            self.documents[key] = document

    @staticmethod
    def _decode_mask(encoded: str):
        """Return ``(keep_target_mask, (include, exclude))``; exactly one of them is set."""
        try:
            value = encoded.removeprefix('data:image/png;base64,')
            data = base64.b64decode(value, validate=True)
            if len(data) > 8 * 1024 * 1024 or not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError()
            width, height = int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
            if not 0 < width <= 3840 or not 0 < height <= 3840 or width * height > 3840 * 2160:
                raise ValueError()
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError()
            if decoded.ndim == 2:
                # Existing workstation masks are grayscale keep-target masks.
                return decoded, None
            if decoded.ndim == 3 and decoded.shape[2] in (3, 4):
                alpha = decoded[..., 3].astype(np.float32) / 255 if decoded.shape[2] == 4 else 1.0
                blue, green, red = (decoded[..., i].astype(np.float32) / 255 for i in range(3))
                if np.any(blue > 0):  # Legacy white PNG with alpha; retain exclude-only meaning.
                    return np.clip(blue * alpha * 255, 0, 255).astype(np.uint8), None
                return None, (np.clip(red * alpha, 0, 1), np.clip(green * alpha, 0, 1))
            raise ValueError()
        except (ValueError, cv2.error):
            raise DomainError('MASK_INVALID', 'Manual mask must be a bounded local PNG', 422) from None

    @staticmethod
    def _require_supported(channels) -> None:
        """Only keep-target (exclude) strokes can be composited; a force-include stroke would be dropped, so it is refused."""
        if channels is not None and bool(np.any(channels[0] > 0)):
            raise DomainError('TRUE3D_MASK_INCLUDE_UNSUPPORTED',
                              'Force-include brush strokes cannot be honoured by the full-frame 3D compositor; '
                              'remove them or paint keep-target (exclude) strokes instead', 422)

    @property
    def has_masks(self) -> bool:
        return self.mask is not None or self.mask_channels is not None or bool(self.mask_keyframes)

    def _mask_for(self, frame):
        """The mask active on ``frame``: a painted frame range wins over the whole-clip mask."""
        if frame is not None:
            for keyframe, mask, channels in self.mask_keyframes:
                if keyframe.contains(frame):
                    return mask, channels
        return self.mask, self.mask_channels

    def apply(self, session, frame: int):
        """Select the swap tracks and their sources for ``frame``; a track whose source changed or ended drops its head engine."""
        if self.segments is None:
            return
        active = {}
        for segment in self.segments:
            if segment.role == 'swap' and segment.contains(frame):
                active[self.ids[segment.track_id]] = (segment.faceset_id, segment.source_view_index)
        lock = getattr(session, 'lock', None)
        with lock if lock is not None else nullcontext():
            session.selected_ids = frozenset(active)
            for track_id, key in active.items():
                if self.current.get(track_id) != key:
                    session.sources_by_track[track_id] = self.documents[key]
                    self._drop_engine(session, track_id)
            for track_id in set(self.current) - set(active):
                self._drop_engine(session, track_id)
        self.current = active

    @staticmethod
    def _drop_engine(session, track_id: int) -> None:
        """Release the cached full-head engine of ``track_id``: it was fitted to a source that is no longer this track's."""
        engines = getattr(session, 'full_head_engines', None)
        engine = engines.pop(track_id, None) if engines is not None else None
        if engine is not None:
            engine.release()

    def _plane(self, source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """``source`` (uint8 keep-target mask or float 0..1 plane) as a read-only float32 plane of the frame size."""
        key = (id(source), shape)
        plane = self._planes.get(key)
        if plane is None:
            plane = source.astype(np.float32)
            if source.dtype == np.uint8:
                plane /= 255
            if plane.shape != shape:
                plane = cv2.resize(plane, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            plane = np.ascontiguousarray(np.clip(plane, 0.0, 1.0), np.float32)
            plane.setflags(write=False)
            if len(self._planes) >= 8:
                self._planes.clear()
            self._planes[key] = plane
        return plane

    def occluders(self, image, result, frame=None):
        """``{track_id: plane}`` of frame-space keep-target planes (1 = the target pixel stays) for every rendered track, or None."""
        mask, channels = self._mask_for(frame)
        if mask is None and channels is None:
            return None
        plane = self._plane(mask if mask is not None else channels[1], image.shape[:2])
        return {item.track_id: plane for item in result.replacements}

    def with_occluders(self, image, result, frame=None):
        """``result`` with the manual keep-target plane merged into every ``Replacement.occluder``; nothing is left unapplied."""
        planes = self.occluders(image, result, frame)
        if not planes:
            return result
        return replace(result, replacements=tuple(
            replace(item, occluder=planes[item.track_id] if item.occluder is None else np.maximum(item.occluder, planes[item.track_id]))
            for item in result.replacements))
