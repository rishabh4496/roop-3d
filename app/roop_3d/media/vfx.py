"""Unfinished, full-frame 16-bit compositing passes and lossless FFV1 previews."""
from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
import os
import uuid

import av
import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned, atomic_bytes, atomic_json
from roop_3d.core.settings import runtime_path
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.probe import probe

DEPTH_FAR = 2048.0  # estimated camera units, never claimed as metres


def geometry_passes(replacement, shape: tuple[int, int], weight: float) -> tuple[np.ndarray, np.ndarray]:
    projection = getattr(replacement, 'projection', None)
    if projection is None:
        raise DomainError('VFX_GEOMETRY_UNAVAILABLE', 'Normal/depth passes require mesh blend mode and frame-aligned projection geometry', 422)
    geometry = projection.geometry
    vertices = geometry.flat_camera_vertices * (1 - weight) + geometry.mesh_camera_vertices * weight
    pixels = projection.blend_pixels
    h, w = replacement.image.shape[:2]
    zbuffer = np.full((h, w), np.inf)
    normals = np.zeros((h, w, 3), np.float32)
    # Crop alignment removes roll. Rotate normal XY back into target-frame axes;
    # exclude the crop-to-frame scale so normal vectors retain unit length.
    left, _, right = np.linalg.svd(np.asarray(replacement.inverse[:, :2], np.float64))
    normal_rotation = left @ right
    for face in projection.visible_faces:
        p, v = pixels[face], vertices[face]
        normal = np.cross(v[1] - v[0], v[2] - v[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10 or not np.isfinite(v).all() or np.any(v[:, 2] <= 0):
            continue
        x0, x1 = max(0, int(np.floor(p[:, 0].min()))), min(w - 1, int(np.ceil(p[:, 0].max())))
        y0, y1 = max(0, int(np.floor(p[:, 1].min()))), min(h - 1, int(np.ceil(p[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = p
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-8:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        u = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / det
        t = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / det
        third = 1 - u - t
        z = 1 / np.maximum(u / v[0, 2] + t / v[1, 2] + third / v[2, 2], 1e-12)
        old = zbuffer[y0:y1 + 1, x0:x1 + 1]
        visible = (u >= -1e-6) & (t >= -1e-6) & (third >= -1e-6) & (z < old)
        old[visible] = z[visible]
        normal /= norm
        normal[:2] = normal_rotation @ normal[:2]
        normals[y0:y1 + 1, x0:x1 + 1][visible] = normal * .5 + .5
    active = np.isfinite(zbuffer) & (projection.coverage > 0)
    if np.any(zbuffer[active] >= DEPTH_FAR):
        raise DomainError('VFX_DEPTH_RANGE', 'Geometry exceeds the documented depth encoding range', 422)
    normals[~active] = 0
    depth = np.where(active, zbuffer, 0).astype(np.float32) / DEPTH_FAR
    size = (shape[1], shape[0])
    return (cv2.warpAffine(normals[..., ::-1], replacement.inverse, size),
            cv2.warpAffine(depth, replacement.inverse, size))


class PassCollector:
    def __init__(self, shape: tuple[int, int], selected: tuple[str, ...]):
        self.shape, self.selected = shape, selected
        self.buffers = {name: np.zeros((*shape, 3), np.float32) if name in {'uv', 'normal', 'relit'}
                        else np.zeros(shape, np.float32) for name in selected}

    def observe(self, stage, crop, inverse, alpha):
        size = (self.shape[1], self.shape[0])
        if stage == 'vfx':
            weight = cv2.warpAffine(alpha, inverse, size)
            if 'alpha' in self.buffers:
                self.buffers['alpha'] = weight + self.buffers['alpha'] * (1 - weight)
            if 'uv' in self.buffers:
                # Unpremultiplied RGB with a separate straight alpha pass.
                image = cv2.warpAffine(crop.astype(np.float32) / 255, inverse, size)
                self.buffers['uv'] = np.where((weight > 0)[..., None], image, self.buffers['uv'])
        elif stage == 'color' and 'relit' in self.buffers:
            # The harmonised face after SH relighting, before it is blended with the target: straight RGB, no background or
            # occluders. Coverage is the separate alpha pass. The compositor already scaled `inverse` for high-resolution crops.
            weight = cv2.warpAffine(alpha.astype(np.float32), inverse, size)
            image = cv2.warpAffine(crop.astype(np.float32) / 255, inverse, size)
            self.buffers['relit'] = np.where((weight > 0)[..., None], image, self.buffers['relit'])
        elif stage == 'occluder' and 'occluder' in self.buffers:
            self.buffers['occluder'] = np.maximum(self.buffers['occluder'], cv2.warpAffine(crop, inverse, size))

    def geometry(self, replacements, weight: float):
        if not {'normal', 'depth'} & set(self.selected):
            return
        for replacement in sorted(replacements, key=lambda item: item.track_id):
            normal, depth = geometry_passes(replacement, self.shape, weight)
            for name, value in [('normal', normal), ('depth', depth)]:
                if name in self.buffers:
                    self.buffers[name] = np.where((depth > 0)[..., None], value, self.buffers[name]) if name == 'normal' else np.where(depth > 0, value, self.buffers[name])

    def save(self, root: Path, job_id: str, index: int):
        for name, image in self.buffers.items():
            ok, encoded = cv2.imencode('.png', np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16))
            if not ok:
                raise DomainError('VFX_ENCODE_FAILED', 'Could not checkpoint a 16-bit pass', 500)
            atomic_bytes(root, f'roop_3d/.runtime/jobs/{job_id}/passes/{name}/{index:08d}.png', encoded.tobytes())


def publish_passes(root: Path, job_id: str, source: Path, selected: tuple[str, ...], output_dir: str,
                   *, start: Fraction | None, end: Fraction | None, check, timeline: dict | None = None) -> dict:
    """PNG sequences retain exact source frame IDs; FFV1 carries matching frame order."""
    output_dir = runtime_path(output_dir, root).relative_to(root).as_posix()
    paths = {name: f'{output_dir}/roop3d_{job_id}_{name}.mkv' for name in selected}
    rate = probe(source).videos[0].average_rate
    if rate is None or rate <= 0:
        raise DomainError('VFX_RATE_UNKNOWN', 'Pass export requires a verified source rate', 422)
    temporary = {name: owned(root, f'{output_dir}/.{uuid.uuid4().hex}.mkv') for name in selected}
    published = []
    sequence_relative = f'{output_dir}/roop3d_{job_id}_sequences'
    sequence_destination = owned(root, sequence_relative)
    published_sequences: list[str] = []
    metadata = {'schema_version': 1, 'passes': paths, 'frames': [], 'input_sha256': sha256_file(source)[1],
                'encoding': '16-bit PNG sequences and lossless FFV1; no LUT, grain, grading or display transform',
                'alpha': 'straight; final temporal face coverage after occluder subtraction',
                'occluder': 'actual parser/manual foreground masks; black where none detected; not an XSeg inference claim',
                'normal': 'flat camera-space normals; RGB=(normal+1)/2, OpenCV +X right +Y down +Z away',
                'depth': {'encoding': 'linear camera Z / 2048', 'far': DEPTH_FAR, 'background': 0,
                          'units': 'estimated canonical camera units, not physical distance'},
                'relit': 'clean relit face: harmonised swap after SH relighting, before blending with the target; straight RGB, pair with the alpha pass',
                'geometry': 'frame-aligned mesh blend projection, estimated canonical proxy and camera',
                'sequence_root': f'roop_3d/.runtime/jobs/{job_id}/passes'}
    if timeline is not None:
        metadata['timeline'] = timeline
    try:
        for relative in paths.values():
            path = owned(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise DomainError('OUTPUT_EXISTS', 'Pass output already exists', 409)
        if sequence_destination.exists():
            raise DomainError('OUTPUT_EXISTS', 'Pass sequence output already exists', 409)
        with ExitStack() as stack:
            containers = {name: stack.enter_context(av.open(str(path), 'w')) for name, path in temporary.items()}
            streams = {}
            frames = VideoReader(source, hardware='software').frames()
            stack.callback(frames.close)
            for decoded in frames:
                check()
                stamp = decoded.timestamp.seconds
                if start is not None and stamp < start:
                    continue
                if end is not None and stamp >= end:
                    break
                entry = {'source_frame': decoded.index, 'pts_seconds': str(stamp - (start or 0)), 'sha256': {}}
                for name in selected:
                    relative = f"{metadata['sequence_root']}/{name}/{decoded.index:08d}.png"
                    image = cv2.imread(str(owned(root, relative)), cv2.IMREAD_UNCHANGED)
                    if image is None or image.dtype != np.uint16 or image.shape[:2] != decoded.image.shape[:2]:
                        raise DomainError('VFX_CHECKPOINT_INVALID', 'A pass checkpoint is missing or invalid', 409)
                    entry['sha256'][name] = sha256_file(owned(root, relative))[1]
                    if name not in streams:
                        stream = containers[name].add_stream('ffv1', rate=rate)
                        stream.width, stream.height = image.shape[1], image.shape[0]
                        stream.pix_fmt = 'gbrp16le' if image.ndim == 3 else 'gray16le'
                        stream.time_base = stream.codec_context.time_base = decoded.timestamp.time_base
                        stream.codec_context.thread_count = 1
                        streams[name] = stream
                    frame = av.VideoFrame.from_ndarray(image[..., ::-1].copy() if image.ndim == 3 else image,
                                                        format='rgb48le' if image.ndim == 3 else 'gray16le')
                    frame = frame.reformat(format=streams[name].pix_fmt)
                    frame.time_base = decoded.timestamp.time_base
                    relative_stamp = (stamp - (start or 0)) / frame.time_base
                    frame.pts = round(relative_stamp)
                    frame.duration = decoded.duration
                    for packet in streams[name].encode(frame):
                        containers[name].mux(packet)
                metadata['frames'].append(entry)
            if not metadata['frames']:
                raise DomainError('VFX_EMPTY', 'No pass frames in selected range', 422)
            for name in selected:
                for packet in streams[name].encode(None):
                    containers[name].mux(packet)
        metadata['output_sha256'] = {}
        for name in selected:
            check()
            with av.open(str(temporary[name])) as container:
                decoded_frames = container.decode(video=0)
                count = 0
                for frame in decoded_frames:
                    check()
                    if count >= len(metadata['frames']):
                        raise DomainError('VFX_VERIFY_FAILED', 'Pass contains unexpected extra frames', 422)
                    entry = metadata['frames'][count]
                    expected = cv2.imread(str(owned(root, f"{metadata['sequence_root']}/{name}/{entry['source_frame']:08d}.png")), cv2.IMREAD_UNCHANGED)
                    observed = frame.to_ndarray(format='rgb48le' if expected.ndim == 3 else 'gray16le')
                    if expected.ndim == 3:
                        observed = observed[..., ::-1]
                    if not np.array_equal(observed, expected) or abs(frame.pts * frame.time_base - Fraction(entry['pts_seconds'])) > Fraction(1, 1000):
                        raise DomainError('VFX_VERIFY_FAILED', 'Lossless pass read-back or timestamp verification failed', 422)
                    count += 1
                if count != len(metadata['frames']):
                    raise DomainError('VFX_VERIFY_FAILED', 'Pass frame count differs', 422)
            metadata['output_sha256'][name] = sha256_file(temporary[name])[1]
            os.replace(temporary[name], owned(root, paths[name]))
            published.append(paths[name])
        # Publish only selected frames; retain checkpoints until the job succeeds
        # so pause/cancel/OOM during baked encoding remains resumable.
        for entry in metadata['frames']:
            for name in selected:
                check()
                checkpoint = owned(root, f"{metadata['sequence_root']}/{name}/{entry['source_frame']:08d}.png")
                relative = f"{sequence_relative}/{name}/{entry['source_frame']:08d}.png"
                atomic_bytes(root, relative, checkpoint.read_bytes())
                published_sequences.append(relative)
        metadata['sequence_root'] = sequence_relative
        manifest_path = f'{output_dir}/roop3d_{job_id}_passes.manifest.json'
        atomic_json(root, manifest_path, metadata)
        return {'paths': paths, 'manifest_path': manifest_path, 'metadata': metadata}
    except BaseException:
        for relative in published_sequences:
            owned(root, relative).unlink(missing_ok=True)
        for name in selected:
            directory = owned(root, f'{sequence_relative}/{name}')
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        if sequence_destination.is_dir() and not any(sequence_destination.iterdir()):
            sequence_destination.rmdir()
        for relative in published:
            owned(root, relative).unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            owned(root, path.relative_to(root).as_posix()).unlink(missing_ok=True)


def discard_sequences(root: Path, manifest_relative: str) -> None:
    """Delete only the sequence directory derived from this generated manifest."""
    import re
    path = owned(root, manifest_relative)
    match = re.fullmatch(r'roop3d_([0-9a-f]{32})_passes.manifest.json', path.name)
    if not match or not path.is_file():
        return
    relative = (path.parent / f'roop3d_{match[1]}_sequences').relative_to(root).as_posix()
    directory = owned(root, relative)
    if not directory.is_dir():
        return
    # Check every descendant before deleting any; never follow a junction.
    paths = sorted(directory.rglob('*'), key=lambda value: len(value.parts), reverse=True)
    for item in paths:
        owned(root, item.relative_to(root).as_posix())
    for item in paths:
        if item.is_dir():
            item.rmdir()
        else:
            item.unlink()
    directory.rmdir()
