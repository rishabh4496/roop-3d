"""Sequential single-photo mesh/LivePortrait/UltraMax faceset synthesis.

Generated evidence is labelled, never promoted to observed source geometry.
All work is local, uses the engine admission gate and shares GPU profile budgets.
"""
from __future__ import annotations
from contextlib import nullcontext
import base64
import json
import hashlib
import threading
import uuid
import cv2
import numpy as np
from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig, record_named
from roop_3d.faceset.schemas import FaceView, QualityMetrics
from roop_3d.faceset.pose_selector import classify_pose_bin
from roop_3d.reconstruction.faceset_3d import reconstruct_faceset, sphere_position, pose_bin

# The requested 11 x 3 lattice contains 33 views. Three intermediate frontal
# views complete the preset without exceeding the requested hemisphere bounds.
SYNTHESIS_POSES = tuple((float(y), float(p)) for p in (-30, 0, 30) for y in range(-75, 76, 15)) + ((0., -15.), (0., 15.), (7.5, 0.))


def synthesis_config(state):
    settings = state.settings
    profile = settings.hardware_profile
    if profile == 'auto':
        from roop_3d.core.hardware import probe_hardware
        profile = probe_hardware(settings).resolved_hardware_profile
    provider = settings.preferred_provider
    if profile == 'cpu' or provider == 'CPUExecutionProvider':
        provider = 'CPUExecutionProvider'
    elif provider not in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'):
        provider = 'CUDAExecutionProvider'
    return RuntimeConfig(provider=provider, hardware_profile=profile,
                         reserve_mib=settings.memory_safety_reserve_mb)


def start_synthesis(builder, key, rank, selected_ids, name):
    with builder.lock:
        draft = builder.jobs.get(key)
        if not draft or draft['status'] != 'ready':
            raise DomainError('DRAFT_NOT_READY', 'Finish importing reference photos first', 409)
        if not 1 <= len(selected_ids) <= 3 or len(set(selected_ids)) != len(selected_ids):
            raise DomainError('SYNTHESIS_INPUT_INVALID', 'Select one to three unique source photos', 422)
        pairs = [draft['samples'].get(s) for s in selected_ids]
        if any(pair is None or pair[1]['cluster_rank'] != rank or pair[0].is_synthetic for pair in pairs):
            raise DomainError('IDENTITY_MISMATCH', 'Select original photos of one person', 422)
        if any(job['status'] in ('extracting', 'synthesizing') for job in builder.jobs.values()):
            raise DomainError('BUILDER_BUSY', 'Another faceset operation is running', 409)
        # Preflight resolves model identities without downloading or loading GPU sessions.
        from roop_3d.engine.volumetric.reference_adapter import CORE_MODELS
        for filename in (*CORE_MODELS.values(), 'liveportrait_stitching_eye.onnx', '1k3d68.onnx', 'CodeFormerv0.1.onnx'):
            record_named(builder.state.models, filename)
        config = synthesis_config(builder.state)
        views = tuple(pair[0].model_copy(update={'index': i}) for i, pair in enumerate(pairs))
        if len(builder.jobs) >= 4:
            del builder.jobs[next(iter(builder.jobs))]
        job = dict(job_id=uuid.uuid4().hex, status='synthesizing', progress=0., scanned=0,
                   clusters=[], samples={}, error=None, result=None, total=36, phase='Preparing local models')
        builder.jobs[job['job_id']] = job
    threading.Thread(target=_run, args=(builder, job, views, name, config), daemon=True,
                     name='roop3d-faceset-synthesis').start()
    return builder.snapshot(job['job_id'])


def _png(image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise DomainError('SYNTHESIS_ENCODE_FAILED', 'Cannot encode a generated view', 500)
    return encoded.tobytes()


def start_photo_synthesis(builder, images, name):
    """One-click bridge from Face Manager's already imported image crops."""
    if not 1 <= len(images) <= 3:
        raise DomainError('SYNTHESIS_INPUT_INVALID', 'Import one to three source photos', 422)
    paths = []
    try:
        for value in images:
            if len(value) > 16*1024*1024:
                raise DomainError('MEDIA_TOO_LARGE', 'Use source crops smaller than 12 MiB', 413)
            try:
                raw = base64.b64decode(value.split(',', 1)[-1], validate=True)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[0]*image.shape[1] > 16777216:
                    raise ValueError('Invalid source crop')
            except (ValueError, cv2.error):
                raise DomainError('SYNTHESIS_INPUT_INVALID', 'Provide bounded, encoded source photos', 422) from None
            path = builder.state.root / 'roop_3d/.runtime/media' / (uuid.uuid4().hex+'.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_png(image))
            paths.append(path)
        config = synthesis_config(builder.state)
        with builder.lock:
            if any(j['status'] in ('extracting', 'synthesizing') for j in builder.jobs.values()):
                raise DomainError('BUILDER_BUSY', 'Another faceset operation is running', 409)
            if len(builder.jobs) >= 4:
                del builder.jobs[next(iter(builder.jobs))]
            job = dict(job_id=uuid.uuid4().hex, status='synthesizing', progress=0., scanned=0,
                       clusters=[], samples={}, error=None, result=None, total=36, phase='Reading source photos')
            builder.jobs[job['job_id']] = job
        threading.Thread(target=_photos_run, args=(builder, job, paths, name, config), daemon=True,
                         name='roop3d-photo-synthesis').start()
        return builder.snapshot(job['job_id'])
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def _photos_run(builder, job, paths, name, config):
    try:
        gate = getattr(builder.state, 'engine_gate', None)
        draft = dict(status='extracting', progress=0., scanned=0, clusters=[], samples={}, error=None)
        with (gate.preview_section() if gate is not None else nullcontext()):
            builder._extract(draft, [str(p.relative_to(builder.state.root)) for p in paths], None, 3, 1)
        if draft['status'] != 'ready' or len(draft['clusters']) != 1 or not draft['samples']:
            raise DomainError('SYNTHESIS_INPUT_INVALID', draft['error'] or 'Photos must show exactly one identity', 422)
        pairs = sorted(draft['samples'].values(), key=lambda pair: pair[0].quality.score, reverse=True)[:3]
        originals = tuple(pair[0].model_copy(update={'index': i}) for i, pair in enumerate(pairs))
        _run(builder, job, originals, name, config)
    except Exception as error:
        with builder.lock:
            job.update(status='error', error=str(error), error_code=getattr(error, 'code', 'SYNTHESIS_FAILED'))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def angle_uv(image, buffers, size=512):
    """Splat visible generated pixels into UV space; alpha records generated support."""
    pixels = cv2.resize(image, (buffers.uv.shape[1], buffers.uv.shape[0]))
    valid = buffers.silhouette > .5
    uv = buffers.uv[valid]
    if not len(uv):
        raise DomainError('SYNTHESIS_VIEW_EMPTY', 'Mesh has no visible UV samples', 422)
    x = np.clip(np.rint(uv[:, 0]*(size-1)).astype(int), 0, size-1)
    y = np.clip(np.rint((1-uv[:, 1])*(size-1)).astype(int), 0, size-1)
    sums, counts = np.zeros((size, size, 3), np.float32), np.zeros((size, size), np.float32)
    np.add.at(sums, (y, x), pixels[valid].astype(np.float32))
    np.add.at(counts, (y, x), 1)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[:, :, :3] = np.clip(sums/np.maximum(counts[:, :, None], 1), 0, 255).round().astype(np.uint8)
    rgba[:, :, 3] = (counts > 0).astype(np.uint8)*255
    return _png(rgba)


def register_uv(buffers, projected_five, detected_five):
    """Register the posed mesh's visible UV field to the generated portrait.

    The generator supplies newly visible facial appearance. A single-view mesh
    projection alone stretches frontal pixels at profiles and is not a portrait
    generator. Mesh UV/depth visibility still owns where each generated texel goes.
    """
    from types import SimpleNamespace
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(projected_five, np.float32),
        np.asarray(detected_five, np.float32), method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-6:
        raise DomainError('SYNTHESIS_UV_ALIGNMENT_FAILED', 'Generated portrait could not be registered to the posed head', 422)
    size = (buffers.uv.shape[1], buffers.uv.shape[0])
    return SimpleNamespace(uv=cv2.warpAffine(buffers.uv, matrix, size, flags=cv2.INTER_NEAREST),
                           silhouette=cv2.warpAffine(buffers.silhouette, matrix, size, flags=cv2.INTER_NEAREST))


def detect_portrait(tracker, image):
    """Give close-cropped generated faces detector context, preserving crop coordinates."""
    faces = tracker.observe(image)
    if faces:
        return faces
    from types import SimpleNamespace
    margin = max(image.shape[:2])//4
    padded = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    return [SimpleNamespace(**{**vars(face), 'box': face.box-margin,
        'landmarks5': face.landmarks5-margin, 'landmarks106': face.landmarks106-margin})
        for face in tracker.observe(padded)]


def _run(builder, job, originals, name, config):
    neural = restorer = tracker = geometry = rasterizer = None
    try:
        gate = getattr(builder.state, 'engine_gate', None)
        with (gate.preview_section() if gate is not None else nullcontext()):
            from roop_3d.expression_restorer import ExpressionRestorer
            from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
            from roop_3d.engine.detection.face_tracker import FaceTracker
            from roop_3d.reconstruction.full_head import GPUHeadRasterizer, HeadPose, head_roi
            from roop_3d.reconstruction.multiview_fit import five_from_68
            from roop_3d.engine.volumetric.reference_adapter import rotation
            fid = 'faceset_' + uuid.uuid4().hex[:16]
            document = builder.state.face_extract._document(fid, name, originals, {})
            with builder.lock:
                job['phase'] = 'Reconstructing source head'
            reconstruction = reconstruct_faceset(document)
            mesh = reconstruction['mesh_full_3d']
            vertices, faces, uvs = np.asarray(mesh['vertices'], np.float32), np.asarray(mesh['indices'], np.int32), np.asarray(mesh['uvs'], np.float32)
            vertices -= (vertices.min(0)+vertices.max(0))/2
            if mesh.get('template') == 'makehuman':
                from roop_3d.reconstruction.makehuman_template import makehuman_template
                landmark_ids = makehuman_template().landmark_vertices['68pt']
            else:
                landmark_ids = np.arange(68)
            texture = cv2.imdecode(np.frombuffer(base64.b64decode(mesh['texture_base64']), np.uint8), cv2.IMREAD_COLOR)
            neural = ExpressionRestorer(builder.state.models, config)
            with builder.lock:
                job['phase'] = 'Preparing local models'
            neural.prepare()
            restorer = UltraMaxPipelineRestorer(builder.state.models, config)
            tracker = FaceTracker(builder.state.models, config=config, stabilize=False)
            geometry = neural.geometry
            rasterizer = GPUHeadRasterizer()
            side = 1024 if config.hardware_profile == '12gb' else 512
            camera = np.array([[side, 0, side/2], [0, side, side/2], [0, 0, 1]], np.float64)
            distance = max(float(np.ptp(vertices, axis=0).max())*3, .5)
            views, entries, poses = [], {}, []
            for original in originals:
                entries[f'references/{original.index}.png'] = _png(cv2.imdecode(np.frombuffer(original.image_bytes, np.uint8), cv2.IMREAD_COLOR))
            entries['source_mesh.json'] = json.dumps({key: mesh[key] for key in
                ('vertices', 'indices', 'uvs', 'normals', 'template', 'reconstruction_level') if key in mesh}).encode('utf-8')
            reference = np.asarray(document.canonical_embedding, np.float32)
            for index, (yaw, pitch) in enumerate(SYNTHESIS_POSES):
                with builder.lock:
                    job['phase'] = f'Generating view {index+1} of 36'
                pose = HeadPose(rotation((pitch, yaw, 0)).T, np.array([0., 0., distance]), camera)
                inverse = head_roi(pose.project(vertices), side=side)
                buffers = rasterizer.render_buffers(vertices, faces, uvs, texture,
                    np.ones(texture.shape[:2], np.float32), pose, inverse, side=side)
                original = min(originals, key=lambda view: (view.yaw-yaw)**2+(view.pitch-pitch)**2)
                source = cv2.imdecode(np.frombuffer(original.image_bytes, np.uint8), cv2.IMREAD_COLOR)
                generated = cv2.resize(neural.generate_pose(source, yaw=yaw, pitch=pitch), (side, side))
                detected = detect_portrait(tracker, generated)
                if len(detected) != 1:
                    raise DomainError('SYNTHESIS_FACE_INVALID', f'Angle {yaw}/{pitch} did not contain one detectable face', 422)
                with builder.lock:
                    job['phase'] = f'Enhancing view {index+1} of 36'
                restored = restorer.restore_hires(generated, detected[0].landmarks5)
                if restored is None:
                    raise DomainError('SYNTHESIS_RESTORER_FAILED', 'UltraMax did not return an image', 503)
                detected = detect_portrait(tracker, restored)
                if len(detected) != 1:
                    raise DomainError('SYNTHESIS_FACE_INVALID', f'Enhanced angle {yaw}/{pitch} did not contain one face', 422)
                face = detected[0]
                similarity = float(np.asarray(face.embedding) @ reference)
                embedding = np.asarray(face.embedding, np.float32)
                if embedding.shape != (512,) or not np.isfinite(embedding).all() or abs(float(np.linalg.norm(embedding))-1) > 1e-4:
                    raise DomainError('SYNTHESIS_EMBEDDING_INVALID', 'Generated view did not produce a finite unit ArcFace embedding', 422)
                # ArcFace alignment is unreliable when one eye is self-occluded.
                # Never call a low-score extreme-angle prior identity-verified.
                extreme = abs(yaw) > 30 or abs(pitch) > 25
                if not np.isfinite(similarity) or (not extreme and similarity < .45):
                    raise DomainError('SYNTHESIS_IDENTITY_DRIFT', f'Angle {yaw}/{pitch} failed ArcFace identity verification ({similarity:.3f})', 422)
                dense = geometry.landmarks(restored, face.box)
                uv_name = f'uv/{index}.png'
                mesh_points = cv2.transform(pose.project(vertices[landmark_ids])[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
                scale = side / restored.shape[0]
                registered = register_uv(buffers, five_from_68(mesh_points), face.landmarks5*scale)
                uv_bytes = angle_uv(restored, registered)
                lap = float(cv2.Laplacian(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                views.append(FaceView(index=index, image_bytes=_png(restored), embedding=tuple(map(float, face.embedding)),
                    landmarks=tuple(map(tuple, face.landmarks5.tolist())), bbox=tuple(map(float, face.box)),
                    landmarks_68=tuple(map(tuple, dense.tolist())), landmarks_106=tuple(map(tuple, face.landmarks106.tolist())),
                    yaw=yaw, pitch=pitch, roll=0., pose_bin=classify_pose_bin(yaw, pitch)[0],
                    quality=QualityMetrics(sharpness=min(1., lap/500), score=min(1., lap/500)*float(np.clip(similarity, 0, 1)), laplacian_variance=lap),
                    is_synthetic=True, uv_map_path=uv_name, uv_map_bytes=uv_bytes))
                poses.append({'view_index': index, 'yaw': yaw, 'pitch': pitch, 'roll': 0., 'bin_index': pose_bin(yaw, pitch),
                    'position_xyz': sphere_position(yaw, pitch), 'is_synthetic': True, 'arcface_similarity': similarity,
                    'identity_status': 'passed' if similarity >= .45 else 'low_confidence_generated_prior',
                    'pose_method': 'mesh_camera_command', 'source_index': original.index})
                with builder.lock:
                    job.update(scanned=index+1, progress=(index+1)/37)
            provenance = {'schema': 'roop.synthesis.v1', 'source_count': len(originals), 'view_count': 36,
                'source_references': [{'index': v.index, 'filename': f'references/{v.index}.png',
                    'sha256': hashlib.sha256(entries[f'references/{v.index}.png']).hexdigest(),
                    'input_sha256': hashlib.sha256(v.image_bytes).hexdigest(), 'yaw': v.yaw, 'pitch': v.pitch, 'roll': v.roll} for v in originals],
                'poses': poses, 'hardware_profile': config.hardware_profile, 'requested_provider': config.provider,
                'grid_note': '33 requested lattice views plus (yaw,pitch) (0,-15), (0,15), (7.5,0)',
                'limitations': ['Hidden surfaces use a single-view mesh/appearance prior; these are generated references, not observations.',
                                'Pose metadata records mesh camera commands, not independently calibrated neural head-pose measurements.',
                                'Extreme-view ArcFace scores are recorded and reduce view quality; low scores at self-occluded poses do not verify identity.'],
                'models': neural.portrait.capability(), 'ultramax': {**restorer.capability(),
                    **({'model': restorer.runner.model.capability()} if getattr(getattr(restorer, 'runner', None), 'model', None) is not None else {})}}
            entries['synthesis.json'] = json.dumps(provenance).encode('utf-8')
            output = builder.state.face_extract._document(fid, name, tuple(views), {'synthesis': provenance})
            # The mesh is fitted exclusively to original photographs, never to hallucinated views.
            reconstruction['generated_pose_points'] = poses
            reconstruction['generated_views'] = len(poses)
            all_points = [*reconstruction['observed_pose_points'], *poses]
            bins = {}
            for point in all_points:
                key = f"bin_{point['bin_index']}"
                bins[key] = bins.get(key, 0)+1
            reconstruction.update(bins=bins, sphere_coverage_count=len(bins),
                sphere_coverage_pct=round(len(bins)/96*100, 2),
                views=[{'yaw': point['yaw'], 'pitch': point['pitch'], 'roll': point.get('roll', 0.),
                        'pose_bin': f"bin_{point['bin_index']}"} for point in all_points])
            with builder.save_lock:
                with builder.lock:
                    job['phase'] = 'Verifying and saving faceset'
                result = builder.publish(output, reconstruction, {}, extra_entries=entries)
            with builder.lock:
                job.update(status='ready', progress=1., phase='Complete', result={'faceset_id': fid, 'path': result['path'], 'view_count': 36})
    except Exception as error:
        with builder.lock:
            job.update(status='error', error=str(error), error_code=getattr(error, 'code', 'SYNTHESIS_FAILED'))
    finally:
        for resource in (neural, restorer, rasterizer):
            if resource is not None:
                resource.release()
        if tracker is not None:
            tracker.release()
