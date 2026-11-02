"""Synthetic ground-truth world for the Stage 3 identity-asset tests (test-only; never imported by production code).

A "subject" is a known 3D head (anthropometric template + a known landmark displacement) with a known UV texture.
Views are produced by really rendering that textured mesh through a scaled-orthographic camera with painter's
visibility, so the images contain what a camera would see (self-occlusion included) and the production fitter /
synthesizer must recover geometry and texture from pixels and noisy landmarks alone. The ground truth is only
used by the assertions, never by the code under test.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from roop_3d.core.pose import as_pose3d, rotation_from_euler
from roop_3d.faceset.pose_selector import classify_pose_bin
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, PoseBinSummary, QualityMetrics
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.multiview_fit import (
    LAYOUT_5,
    LAYOUT_68,
    deform_template,
    identity_template,
    template_landmarks,
)

ATLAS = 256
IMG_W, IMG_H = 560, 600
SCALE = 2300.0          # pixels per metre
CENTER = (280.0, 310.0)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def ground_truth_texture(size: int = ATLAS, seed: int = 0) -> np.ndarray:
    """Deterministic, detailed, skin-ish BGR atlas (row 0 = top of the face)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    base = np.stack([150 + 25 * xx, 165 + 20 * yy, 205 + 25 * np.sin(3 * xx)], axis=-1)
    checker = (((xx * 24).astype(int) + (yy * 24).astype(int)) % 2).astype(np.float32)[..., None] * 22
    tex = base + checker + rng.normal(0, 4, (size, size, 1))
    for cx, cy, color in ((0.3, 0.38, (60, 60, 200)), (0.7, 0.38, (200, 70, 60)), (0.5, 0.75, (70, 190, 70))):
        cv2.circle(tex, (int(cx * size), int(cy * size)), size // 14, color, -1)
    return np.clip(tex, 0, 255).astype(np.uint8)


@dataclass
class Subject:
    name: str
    layout: str
    seed: int
    template: Mesh3D
    template_lm: np.ndarray
    true_lm: np.ndarray
    mesh: Mesh3D                       # ground-truth head (deformed template)
    texture: np.ndarray
    center: np.ndarray
    embedding: np.ndarray
    uv_raster: np.ndarray = field(default=None)  # (V, 2) atlas raster coords


def make_subject(name: str, layout: str, seed: int, *, width: float = 0.12, nose_depth: float = 0.016,
                 jaw_drop: float = 0.010, subdivisions: int = 2) -> Subject:
    template = identity_template(subdivisions)
    t_lm = template_landmarks(layout, template)
    d = np.zeros_like(t_lm)
    d[:, 0] = t_lm[:, 0] * width
    d[:, 1] = np.where(t_lm[:, 1] > 0.02, jaw_drop, 0.0)
    nose = int(np.argmin(np.linalg.norm(t_lm - template.vertices[30], axis=1)))
    d[nose, 2] = nose_depth
    d[:, 2] += (np.abs(t_lm[:, 0]) > 0.03) * -0.006
    true_lm = t_lm + d
    verts, _, _ = deform_template(template, t_lm, true_lm, np.ones(len(t_lm)))
    mesh = Mesh3D(verts, template.faces, uvs=template.uvs)
    rng = np.random.default_rng(seed)
    uv_raster = np.column_stack([template.uvs[:, 0] * (ATLAS - 1), (1.0 - template.uvs[:, 1]) * (ATLAS - 1)])
    return Subject(name, layout, seed, template, t_lm, true_lm, mesh, ground_truth_texture(ATLAS, seed),
                   t_lm.mean(axis=0), _unit(rng.normal(size=512)), uv_raster)


def camera(subject: Subject, yaw: float, pitch: float, roll: float):
    rot = rotation_from_euler(yaw, pitch, roll)
    return rot, SCALE, np.array(CENTER) - SCALE * (rot @ subject.center)[:2]


def render(subject: Subject, yaw: float, pitch: float = 0.0, roll: float = 0.0, *, light: float = 1.0,
           cast: tuple[float, float, float] = (1.0, 1.0, 1.0), seed: int = 0) -> np.ndarray:
    """Painter's-algorithm render of the textured ground-truth mesh over a noisy neutral background."""
    rng = np.random.default_rng(1000 + seed)
    img = np.clip(rng.normal(95, 10, (IMG_H, IMG_W, 3)), 0, 255).astype(np.uint8)
    rot, scale, t = camera(subject, yaw, pitch, roll)
    verts = subject.mesh.vertices.astype(np.float64)
    cam = verts @ rot.T
    xy = scale * cam[:, :2] + t
    depth = cam[:, 2]
    normals_z = (subject.mesh.compute_face_normals().astype(np.float64) @ rot.T)[:, 2]
    faces = subject.mesh.faces
    order = np.argsort(depth[faces].mean(axis=1))
    tex = subject.texture
    for k in order:
        if normals_z[k] <= 0.0:
            continue
        tri = faces[k]
        dst = xy[tri].astype(np.float32)
        x0, y0 = np.floor(dst.min(axis=0)).astype(int) - 1
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, IMG_W), min(y1, IMG_H)
        if x1 <= x0 or y1 <= y0:
            continue
        m = cv2.getAffineTransform(subject.uv_raster[tri].astype(np.float32), dst - np.array([x0, y0], np.float32))
        patch = cv2.warpAffine(tex, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst - np.array([x0, y0], np.float32)).astype(np.int32), 255)
        region = img[y0:y1, x0:x1]
        np.copyto(region, patch, where=mask[..., None] > 0)
    out = img.astype(np.float32) * light * np.array(cast, np.float32)[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def project_landmarks(subject: Subject, yaw: float, pitch: float, roll: float, *, noise_px: float = 0.7,
                      seed: int = 0, expression: str | None = None) -> np.ndarray:
    lm = subject.true_lm.copy()
    if expression == "open" and subject.layout == LAYOUT_68:
        lm[[57, 58, 59, 65, 66, 67], 1] += 0.012   # lower lip / inner lip drop
        lm[[48, 54], 0] += np.array([-0.002, 0.002])
    elif expression == "open" and subject.layout == LAYOUT_5:
        lm[[3, 4], 1] += 0.006
    rot, scale, t = camera(subject, yaw, pitch, roll)
    pts = scale * (lm @ rot.T)[:, :2] + t
    return pts + np.random.default_rng(seed + 7).normal(0, noise_px, pts.shape)


def five_from_dense(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts[36:42].mean(axis=0), pts[42:48].mean(axis=0), pts[30], pts[48], pts[54]])


def make_view(subject: Subject, index: int, yaw: float, pitch: float = 0.0, roll: float = 0.0, *, light: float = 1.0,
              cast=(1.0, 1.0, 1.0), pose_noise: float = 2.0, seed: int = 0, expression: str | None = None,
              occlude_eyes: bool = False, stored_pose: tuple[float, float, float] | None = None,
              embed_noise: float = 0.15, embedding: np.ndarray | None = None, image: np.ndarray | None = None,
              with_dense: bool | None = None) -> FaceView:
    img = image if image is not None else render(subject, yaw, pitch, roll, light=light, cast=cast, seed=seed + index)
    pts = project_landmarks(subject, yaw, pitch, roll, seed=seed + index, expression=expression)
    if occlude_eyes:
        eyes = pts[[0, 1]] if subject.layout == LAYOUT_5 else pts[36:48]
        x0, y0 = eyes.min(axis=0) - 14
        x1, y1 = eyes.max(axis=0) + 14
        img = img.copy()
        img[int(y0):int(y1), int(x0):int(x1)] = 8
    five = pts if subject.layout == LAYOUT_5 else five_from_dense(pts)
    ok, png = cv2.imencode(".png", img)
    assert ok
    rng = np.random.default_rng(subject.seed * 100 + index + seed)
    emb = (embedding if embedding is not None else subject.embedding) + rng.normal(0, embed_noise / np.sqrt(512), 512)
    emb = _unit(emb)
    syaw, spitch, sroll = stored_pose if stored_pose is not None else (
        yaw + rng.normal(0, pose_noise), pitch + rng.normal(0, pose_noise), roll)
    legacy = as_pose3d((syaw, spitch, sroll), frame="standard").legacy_tuple()   # FaceView stores pitch + = down
    lo, hi = five.min(axis=0), five.max(axis=0)
    bbox = (float(max(lo[0] - 90, 0)), float(max(lo[1] - 110, 0)), float(min(hi[0] + 90, IMG_W)), float(min(hi[1] + 90, IMG_H)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))
    dense = with_dense if with_dense is not None else subject.layout == LAYOUT_68
    return FaceView(
        index=index, image_bytes=png.tobytes(), embedding=tuple(float(x) for x in emb),
        landmarks=tuple((float(x), float(y)) for x, y in five),
        landmarks_68=tuple((float(x), float(y)) for x, y in pts) if dense and subject.layout == LAYOUT_68 else None,
        bbox=bbox, yaw=float(np.clip(legacy[0], -90, 90)), pitch=float(np.clip(legacy[1], -60, 60)), roll=float(legacy[2]),
        quality=QualityMetrics(sharpness=sharp, detector_confidence=0.95, score=round(0.7 * sharp + 0.3 * 0.95, 4),
                               laplacian_variance=float(sharp * 500), face_pixels=float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))),
        pose_bin=classify_pose_bin(legacy[0], legacy[1])[0], expression={"openness": 0.8 if expression == "open" else 0.0})


def make_faceset(faceset_id: str, views: list[FaceView]) -> FaceSetDocument:
    embeddings = np.stack([np.asarray(v.embedding, np.float64) for v in views])
    mean = _unit(embeddings.mean(axis=0))
    sims = embeddings @ embeddings.T
    medoid = embeddings[int(np.argmax(sims.sum(axis=1)))]
    bins: dict[str, list[int]] = {}
    for v in views:
        bins.setdefault(v.pose_bin, []).append(v.index)
    summaries = {}
    for name, members in bins.items():
        yb, pb = name.split("_")
        avg = _unit(embeddings[[m for m in range(len(views)) if views[m].index in members]].mean(axis=0))
        summaries[name] = PoseBinSummary(yaw_bin=yb, pitch_bin=pb, members=tuple(members), support=len(members),
                                         representative_index=members[0], embedding=tuple(float(x) for x in avg))
    digest = hashlib.sha256(b"".join(v.image_bytes for v in views)).hexdigest()
    return FaceSetDocument(
        faceset_id=faceset_id, name=faceset_id, source_sha256=digest, archive_format="image_directory",
        canonical_embedding=tuple(float(x) for x in mean), medoid_embedding=tuple(float(x) for x in _unit(medoid)),
        views=tuple(views), pose_bins=summaries, metadata={})


def sim_align(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove the unobservable global translation and uniform scale."""
    mc, rc = moving - moving.mean(0), reference - reference.mean(0)
    s = float((mc * rc).sum() / (mc * mc).sum())
    return s * mc + reference.mean(0)


def full_set(subject: Subject, yaws=(0, -40, 40, -72, 72), pitches=None, **kw) -> FaceSetDocument:
    pitches = pitches or [0.0] * len(yaws)
    return make_faceset(f"{subject.name}_{len(yaws)}v", [
        make_view(subject, i, y, p, **kw) for i, (y, p) in enumerate(zip(yaws, pitches))])
