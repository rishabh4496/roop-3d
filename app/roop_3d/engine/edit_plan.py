"""Validated role intervals and source changes; excluded/reference tracks never infer swaps."""
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
        if request.mask_base64:
            try:
                value = request.mask_base64.removeprefix('data:image/png;base64,')
                data = base64.b64decode(value, validate=True)
                if len(data) > 8 * 1024 * 1024 or not data.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError()
                width, height = int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
                if not 0 < width <= 3840 or not 0 < height <= 3840 or width * height > 3840 * 2160:
                    raise ValueError()
                self.mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if self.mask is None:
                    raise ValueError()
            except (ValueError, cv2.error):
                raise DomainError('MASK_INVALID', 'Manual mask must be a bounded local PNG', 422) from None
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

    def apply(self, session, frame: int):
        if self.segments is None:
            return
        active = {}
        for segment in self.segments:
            if segment.role == 'swap' and segment.contains(frame):
                active[self.ids[segment.track_id]] = (segment.faceset_id, segment.source_view_index)
        session.selected_ids = frozenset(active)
        for track_id, key in active.items():
            if self.current.get(track_id) != key:
                session.swapper.set_track_source(track_id, self.documents[key])
                # Rebuild portrait conditioning after an identity change.
                session.portrait.tracks.pop(track_id, None)
        self.current = active

    def occluders(self, image, result):
        if self.mask is None:
            return None
        if self.mask.shape != image.shape[:2]:
            raise DomainError('MASK_SIZE_MISMATCH', 'Manual mask must match upright media dimensions', 422)
        return {item.track_id: cv2.warpAffine(self.mask.astype(np.float32) / 255,
            cv2.invertAffineTransform(item.inverse), (item.image.shape[1], item.image.shape[0])) for item in result.replacements}
