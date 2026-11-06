"""Bounded, rank-preserving media curation and self-contained Stage 7 archives."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import threading
import uuid
import zipfile
import cv2
import numpy as np
from roop_3d.core.errors import DomainError
from roop_3d.faceset.pose_selector import classify_pose_bin
from roop_3d.faceset.schemas import FaceView, QualityMetrics
from roop_3d.faceset.faceset_reader import FaceSetReader, export_v2
from roop_3d.reconstruction.faceset_3d import _pose, pose_bin, reconstruct_faceset
from roop_3d.engine.volumetric.preprocess import aligned_crop, transform_points


class FacesetBuilder:
    def __init__(self, state):
        self.state = state
        self.lock = threading.RLock()
        self.jobs = {}
        self.save_lock = threading.Lock()

    def start(self, paths, *, faceset_id=None, max_frames=300, frame_stride=5):
        with self.lock:
            if any(j['status'] == 'extracting' for j in self.jobs.values()):
                raise DomainError('BUILDER_BUSY', 'An extraction is already running', 409)
            # Drafts retain only two samples per identity/bin and at most four sessions.
            if len(self.jobs) >= 4:
                del self.jobs[next(iter(self.jobs))]
            key = uuid.uuid4().hex
            job = dict(job_id=key, status='extracting', progress=0, scanned=0, clusters=[], samples={}, error=None)
            self.jobs[key] = job
        threading.Thread(target=self._run, args=(job, paths, faceset_id, max_frames, frame_stride), daemon=True).start()
        return self.snapshot(key)

    def snapshot(self, key):
        with self.lock:
            if key not in self.jobs:
                raise DomainError('DRAFT_NOT_FOUND', 'Extraction draft expired; extract again', 404)
            job = self.jobs[key]
            payload = {k: v for k, v in job.items() if k not in ('samples', 'centroids', 'counts')}
            payload['clusters'] = [{**cluster, 'samples': [{k: v for k, v in sample.items() if k not in ('landmarks_106', 'landmarks_3d_68')} for sample in cluster['samples']]} for cluster in job['clusters']]
            return payload

    def _frames(self, paths, limit, stride):
        files = []
        for value in paths:
            path = (self.state.root / value).resolve()
            if not path.is_relative_to(self.state.root.resolve()) or not path.exists():
                raise DomainError('MEDIA_PATH_DENIED', 'Media must exist inside this checkout; upload external files first', 422)
            files.extend(sorted(path.glob('*')) if path.is_dir() else [path])
        count = 0
        for path in files:
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
                image = cv2.imread(str(path))
                if image is None:
                    raise DomainError('MEDIA_UNREADABLE', f'Cannot decode {path.name}', 422)
                yield image
                count += 1
            elif path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
                cap = cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        raise DomainError('MEDIA_UNREADABLE', f'Cannot open {path.name}', 422)
                    index = 0
                    while count < limit:
                        ok, image = cap.read()
                        if not ok:
                            break
                        if index % stride == 0:
                            yield image
                            count += 1
                        index += 1
                finally:
                    cap.release()
            if count >= limit:
                break

    def _run(self, job, paths, faceset_id, limit, stride):
        from contextlib import nullcontext
        gate = getattr(self.state, 'engine_gate', None)
        try:
            with (gate.preview_section() if gate is not None else nullcontext()):
                self._extract(job, paths, faceset_id, limit, stride)
        except Exception as error:
            with self.lock:
                job['status'], job['error'] = 'error', str(error)

    def _extract(self, job, paths, faceset_id, limit, stride):
        tracker = analyzer = dense68 = None
        try:
            job['centroids'], job['counts'] = [], []
            if faceset_id:
                _, archive = self.state.face_extract._resolve_faceset(faceset_id)
                document = FaceSetReader(self.state.models).read(archive, faceset_id)
                with zipfile.ZipFile(archive) as container:
                    extras = json.loads(container.read('curation.json')) if 'curation.json' in container.namelist() else {}
                for view in document.views:
                    extra = {**extras.get(str(view.index), {}), 'origin': 'archive', 'occlusion_tier': 'archive', 'eligible': True}
                    for field in ('sample_id', 'cluster_rank', 'bin_index', 'yaw', 'pitch', 'roll', 'score', 'thumbnail_base64'):
                        extra.pop(field, None)
                    self.add(job, view, extra)
            if not paths:
                with self.lock:
                    job['status'], job['progress'] = 'ready', 1
                return
            from roop_3d.engine.runtime import RuntimeConfig, OrtModel
            from roop_3d.engine.detection.face_tracker import FaceTracker
            from roop_3d.compositor.occlusion_analysis import OcclusionAnalyzer
            settings = self.state.settings
            profile = settings.hardware_profile
            if profile == 'auto':
                from roop_3d.core.hardware import probe_hardware
                hardware = probe_hardware(settings)
                profile = hardware.resolved_hardware_profile
            provider = 'CPUExecutionProvider' if profile == 'cpu' or settings.preferred_provider == 'CPUExecutionProvider' else 'CUDAExecutionProvider'
            config = RuntimeConfig(provider=provider, hardware_profile=profile)
            tracker = FaceTracker(self.state.models, config=config)
            analyzer = OcclusionAnalyzer(self.state.models, config)
            dense68 = OrtModel(self.state.models, '1k3d68.onnx', config)
            # Match InsightFace's graph-dependent normalization and last-68 3D output contract.
            import onnx
            graph = onnx.load(str(dense68.path), load_external_data=False).graph
            nodes = graph.node[:8]
            internal = (any(n.name.startswith(('Sub', '_minus')) for n in nodes) and
                        any(n.name.startswith(('Mul', '_mul')) for n in nodes)) or any(n.name == 'bn_data' for n in nodes[:3])
            del graph, nodes
            input_node = dense68.session.get_inputs()[0]
            side = int(input_node.shape[2])
            for image in self._frames(paths, limit, stride):
                if max(image.shape[:2]) > 2048:
                    scale = 2048 / max(image.shape[:2])
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                for face in sorted(tracker.observe(image), key=lambda f: float(f.box[0])):
                    crop = aligned_crop(image, face.landmarks5, 512)
                    box = face.box
                    scale = side / (max(box[2] - box[0], box[3] - box[1]) * 1.5)
                    matrix = np.column_stack((np.eye(2) * scale, side / 2 - (box[:2] + box[2:]) / 2 * scale))
                    portrait = cv2.warpAffine(image, matrix, (side, side))
                    blob = cv2.dnn.blobFromImage(portrait, 1. if internal else 1./128., (side, side),
                                               (0.,)*3 if internal else (127.5,)*3, swapRB=True)
                    raw = next(iter(dense68.run({input_node.name: blob}).values())).reshape(-1, 3)[-68:].copy()
                    raw[:, :2] = (raw[:, :2] + 1) * (side / 2)
                    raw[:, 2] *= side / 2 / scale
                    raw[:, :2] = transform_points(raw[:, :2], cv2.invertAffineTransform(matrix))
                    points68 = transform_points(raw[:, :2], crop.matrix)
                    points3d = np.column_stack((points68, raw[:, 2] * np.linalg.norm(crop.matrix[0, :2])))
                    five = transform_points(face.landmarks5, crop.matrix)
                    dense = transform_points(face.landmarks106, crop.matrix)
                    analysis = analyzer.analyze(crop.image, dense)
                    visible = analysis.visible_fraction
                    lap = float(cv2.Laplacian(cv2.cvtColor(crop.image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    pixels = float(np.prod(face.box[2:] - face.box[:2]))
                    sharp = min(1., lap / 500.)
                    score = (.65 * sharp + .35 * min(1., pixels / 65536.)) * (visible if visible is not None else .5)
                    _, png = cv2.imencode('.png', crop.image)
                    yaw, pitch, roll = face.pose
                    view = FaceView(index=0, image_bytes=png.tobytes(), embedding=tuple(face.embedding.tolist()),
                        landmarks=tuple(map(tuple, five.tolist())), bbox=(0., 0., 512., 512.),
                        yaw=max(-90., min(90., yaw)), pitch=max(-60., min(60., pitch)), roll=roll,
                        landmarks_68=tuple(map(tuple, points68.tolist())), quality=QualityMetrics(sharpness=sharp, score=score, laplacian_variance=lap, face_pixels=pixels), pose_bin=classify_pose_bin(max(-90., min(90., yaw)), max(-60., min(60., pitch)))[0])
                    self.add(job, view, {'landmarks_106': dense.tolist(), 'landmarks_3d_68': points3d.tolist(), 'occlusion_tier': analysis.tier,
                        'visible_fraction': visible, 'eligible': visible is not None and visible >= .65})
                with self.lock:
                    job['scanned'] += 1
                    job['progress'] = min(.99, job['scanned'] / limit)
            with self.lock:
                job['inference'] = {'requested_provider': provider, 'hardware_profile': profile, 'landmarks_3d': dense68.capability()}
                job['status'], job['progress'] = 'ready', 1
                if not job['samples']:
                    job['error'] = 'No faces detected. Try a clearer reference or a smaller frame stride.'
        except Exception as error:
            with self.lock:
                job['status'], job['error'] = 'error', str(error)
        finally:
            if tracker is not None:
                tracker.release()
            if analyzer is not None:
                analyzer.release()
            if dense68 is not None:
                dense68.release()

    def add(self, job, view, extra):
        """Cluster numbers never change when identities gain more observations."""
        emb = np.asarray(view.embedding, np.float32)
        emb /= max(float(np.linalg.norm(emb)), 1e-9)
        with self.lock:
            similarities = [float(emb @ center) for center in job['centroids']]
            rank = int(np.argmax(similarities)) if similarities and max(similarities) >= .60 else len(similarities)
            if rank == len(similarities):
                if rank >= 16:
                    return
                job['centroids'].append(emb.copy())
                job['counts'].append(0)
            job['counts'][rank] += 1
            center = job['centroids'][rank] + (emb - job['centroids'][rank]) / job['counts'][rank]
            job['centroids'][rank] = center / max(float(np.linalg.norm(center)), 1e-9)
            yaw, pitch, roll, _ = _pose(view)
            bin_id = pose_bin(yaw, pitch)
            sample_id = uuid.uuid4().hex
            _, jpg = cv2.imencode('.jpg', cv2.resize(cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), 1), (112, 112)))
            item = dict(sample_id=sample_id, cluster_rank=rank, bin_index=bin_id, yaw=yaw, pitch=pitch, roll=roll,
                        score=view.quality.score, thumbnail_base64=base64.b64encode(jpg).decode(), **extra)
            job['samples'][sample_id] = (view, item)
            same = sorted([(key, pair) for key, pair in job['samples'].items() if pair[1]['cluster_rank'] == rank and pair[1]['bin_index'] == bin_id],
                          key=lambda pair: (pair[1][1].get('eligible', True), pair[1][0].quality.score), reverse=True)
            for key, _ in same[2:]:
                del job['samples'][key]
            clusters = []
            for r, count in enumerate(job['counts']):
                items = [dict(pair[1]) for pair in job['samples'].values() if pair[1]['cluster_rank'] == r]
                selected = {}
                for item in sorted(items, key=lambda x: x['score'], reverse=True):
                    if item.get('eligible', True):
                        selected.setdefault(item['bin_index'], item['sample_id'])
                clusters.append(dict(rank=r, detections=count, samples=items, selected_ids=list(selected.values()), coverage=len(selected)))
            job['clusters'] = clusters

    def save(self, key, rank, selected_ids, name, faceset_id=None):
        with self.save_lock:
            return self._save(key, rank, selected_ids, name, faceset_id)

    def _save(self, key, rank, selected_ids, name, faceset_id=None):
        with self.lock:
            job = self.jobs.get(key)
            if not job or job['status'] != 'ready':
                raise DomainError('DRAFT_NOT_READY', 'Wait for extraction to finish', 409)
            if not selected_ids or len(selected_ids) > 96 or len(set(selected_ids)) != len(selected_ids):
                raise DomainError('SELECTION_INVALID', 'Select between 1 and 96 unique crops', 422)
            pairs = [job['samples'].get(key) for key in selected_ids]
            if any(pair is None or pair[1]['cluster_rank'] != rank for pair in pairs):
                raise DomainError('IDENTITY_MISMATCH', 'All selected samples must belong to the chosen cluster', 422)
            if len({pair[1]['bin_index'] for pair in pairs}) != len(pairs):
                raise DomainError('SELECTION_INVALID', 'Choose one crop per sphere bin', 422)
            views = tuple(pair[0].model_copy(update={'index': i}) for i, pair in enumerate(pairs))
            extras = {str(i): pair[1] for i, pair in enumerate(pairs)}
        fid = faceset_id or 'faceset_' + uuid.uuid4().hex[:16]
        document = self.state.face_extract._document(fid, name, views, {'created_via': 'stage7_builder'})
        reconstruction = reconstruct_faceset(document)
        final = self.state.root / 'facesets' / (fid + '.fsz')
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = final.with_name('.' + uuid.uuid4().hex + '.fsz')
        try:
            export_v2(document, temp, app_root=self.state.root)
            with zipfile.ZipFile(temp, 'a', compression=zipfile.ZIP_STORED) as archive:
                archive.writestr('sphere_coverage.json', json.dumps(reconstruction['observed_pose_points']))
                archive.writestr('curation.json', json.dumps(extras))
                archive.writestr('uv_albedo.png', base64.b64decode(reconstruction['mesh_full_3d']['texture_base64']))
            # Verify the complete archive before publication.
            reloaded = FaceSetReader(self.state.models).read(temp, fid)
            self.state.face_extract._write_faceset(reloaded, replace=bool(faceset_id))
            # Publish identical bytes to the compatibility path used by render workers.
            compatibility = self.state.root / 'roop_3d/outputs/facesets' / (fid + '.fsz')
            copy_temp = compatibility.with_suffix('.new.fsz')
            copy_temp.write_bytes(temp.read_bytes())
            os.replace(copy_temp, compatibility)
            os.replace(temp, final)
        finally:
            temp.unlink(missing_ok=True)
        from roop_3d.faceset import pose_matrix
        pose_matrix.invalidate(fid)
        cache = getattr(self.state, 'faceset_3d_state', None)
        if cache is None:
            cache = self.state.faceset_3d_state = {'lock': threading.RLock(), 'values': {}, 'latest': None}
        reconstruction['source_sha256'] = reloaded.source_sha256
        with cache['lock']:
            cache['values'][fid] = {'source_sha256': reloaded.source_sha256, 'payload': reconstruction}
            cache['latest'] = fid
        from roop_3d.server.unified_state import state
        state.set_source_faces([{'face_id': fid, 'path': str(final), 'embedding': list(document.canonical_embedding)}])
        return dict(faceset_id=fid, name=name, path='facesets/' + final.name, reconstruction=reconstruction)
