"""Anatomical eyeball geometry and texture for the volumetric Stage 4 head.

The MakeHuman CC0 head contains eyelids and an empty orbital cavity but no globe. This module builds two real
sclera/limbus/cornea surfaces of revolution (not a 2D paste) and places them *inside* the fitted cavity from the dense
68-point eye landmarks (36-41 subject-right, 42-47 subject-left), so the mesh lids occlude the globes through the same
perspective Z-buffer as the skin.

Frame: the identity-mesh frame (metres; +x image-right, +y down, +z toward the viewer, outward CCW cross-product normals).
The gaze axis is the head-forward +z axis. Each eye owns a square texture tile addressed by an azimuthal-equidistant map
around the gaze axis (tile x = head x, tile y = head y, so a tile reads like a frontal photograph of the eye).

Texture provenance is explicit: texels sampled from real source photographs (through a depth-tested projection, only where
the BiSeNet eye class sees the eyeball) are ``measured``; everything else is a labelled ``prior`` built from the measured iris
and sclera colours when they exist, or from neutral anatomical constants when they do not.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

EYE_NAMES = ("subject_right_image_left", "subject_left_image_right")

# Anatomical proportions as fractions of the globe radius (adult: R ~ 12 mm, iris diameter ~ 11.7 mm, cornea bulge ~ 1.6 mm).
GLOBE_FRACTION_OF_FISSURE = 0.42      # globe radius / lid-margin aperture width (adult: 12 mm / ~29 mm)
MIN_GLOBE_RADIUS_M = 0.0080
MAX_GLOBE_RADIUS_M = 0.0135
LIMBUS_FRACTION = 0.4875
BULGE_FRACTION = 0.133
PUPIL_FRACTION = 0.38                 # pupil radius / limbus radius (mesopic)
LID_CONTACT_M = 0.0003                # the lid margin rests on the tear film just outside the sclera
PSI_MAX = math.radians(100.0)         # polar angle mapped onto the tile edge; nothing beyond is ever visible
TILE_SIZE = 256
RINGS_CORNEA = 7
RINGS_SCLERA = 16
SEGMENTS = 32

# Neutral anatomical priors (BGR), used only when the source photographs give no measurement.
PRIOR_SCLERA_BGR = np.array((186.0, 196.0, 206.0), np.float32)       # albedo (not a lit highlight): sclera reflects ~0.7
PRIOR_IRIS_BGR = np.array((46.0, 74.0, 108.0), np.float32)           # mid-brown, the most common human iris


@dataclass(frozen=True)
class EyeGlobes:
    """Two eyeball surfaces in the mesh frame plus the maps needed to texture them."""

    vertices: np.ndarray        # float32 (2*G, 3)
    normals: np.ndarray         # float32 (2*G, 3) analytic, +outward
    faces: np.ndarray           # int32 (2*F, 3), cross-product outward like the head template
    tile_pixels: np.ndarray     # float32 (2*G, 2) pixel position inside the vertex's own eye tile
    eye_of_vertex: np.ndarray   # int8 (2*G,)
    centers: np.ndarray         # float64 (2, 3)
    radius_m: float
    limbus_radius_m: float
    cornea_center_offset_m: float
    cornea_radius_m: float
    limbus_psi: float
    vertices_per_eye: int
    faces_per_eye: int
    cornea_faces: np.ndarray    # bool (2*F,) triangles lying wholly within the limbus
    diagnostics: dict[str, Any]

    def surface(self, eye: int, psi: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points and analytic outward normals at polar angle ``psi`` (from +z) and azimuth ``phi`` of one eye."""
        return _globe_surface(self.radius_m, self.cornea_center_offset_m, self.cornea_radius_m, self.limbus_psi,
                              self.centers[eye], psi, phi)


def _globe_surface(radius: float, offset: float, corneal_radius: float, limbus_psi: float, centre: np.ndarray,
                   psi: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi = np.asarray(psi, np.float64)
    phi = np.asarray(phi, np.float64)
    direction = np.stack((np.sin(psi) * np.cos(phi), np.sin(psi) * np.sin(phi), np.cos(psi)), axis=-1)
    cornea = psi < limbus_psi
    s_cornea = offset * np.cos(psi) + np.sqrt(np.maximum(corneal_radius ** 2 - offset ** 2 * np.sin(psi) ** 2, 0.0))
    local = direction * np.where(cornea, s_cornea, radius)[..., None]
    corneal_normal = (local - np.array((0.0, 0.0, offset))) / corneal_radius
    normal = np.where(cornea[..., None], corneal_normal, direction)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    return local + np.asarray(centre, np.float64), normal


def _tile_pixel(psi: np.ndarray, phi: np.ndarray, size: int = TILE_SIZE) -> np.ndarray:
    """Tile pixel (x, y) of a surface direction; the exact inverse of :func:`tile_grid`."""
    reach = size / 2.0 - 2.0
    radius = np.minimum(np.asarray(psi, np.float64), PSI_MAX) / PSI_MAX * reach
    centre = (size - 1) / 2.0
    return np.stack((centre + radius * np.cos(phi), centre + radius * np.sin(phi)), axis=-1)


def tile_grid(size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, phi, valid) of every tile texel under the azimuthal-equidistant eye map."""
    reach = size / 2.0 - 2.0
    centre = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = xs - centre, ys - centre
    radius = np.hypot(dx, dy)
    psi = radius / reach * PSI_MAX
    return psi, np.arctan2(dy, dx), psi <= PSI_MAX


def _outward_sign(template_vertices: np.ndarray, template_faces: np.ndarray) -> float:
    v = np.asarray(template_vertices, np.float64)
    f = np.asarray(template_faces, np.int64)
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    outward = v[f].mean(axis=1) - v.mean(axis=0)
    score = float(np.mean(np.sign(np.einsum("ij,ij->i", cross, outward))))
    return 1.0 if score >= 0 else -1.0


def build_eye_globes(vertices: np.ndarray, head_faces: np.ndarray, apertures: Sequence[dict[str, np.ndarray]]) -> EyeGlobes:
    """Fit two anatomical globes into the fitted head's eye sockets.

    ``apertures`` come from :func:`makehuman_landmarks.eye_apertures` (image-left eye first): each carries the ``socket``
    cavity vertex ids and the ordered ``rim`` loop where the lid skin meets the cavity - the same vertices the dense
    eye landmarks 36-47 are bound to. The globe centre lies on the head-forward axis through the rim's bounding-box
    centre; its depth is solved so the central lid margin rests on the globe surface (tear-film contact), i.e. the lids
    hug the sphere and occlude its upper and lower rim through the shared Z-buffer.
    """
    verts = np.asarray(vertices, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(apertures) != 2:
        raise DomainError("EYE_GLOBE_INPUT_INVALID", "Eye globes need fitted head vertices and two eyelid apertures", 422)
    rims = [verts[np.asarray(item["rim"], np.int64)] for item in apertures]
    socket_vertices = np.concatenate([np.asarray(item["socket"], np.int64) for item in apertures])
    widths = [float(np.ptp(rim[:, 0])) for rim in rims]
    fissure = float(np.mean(widths))
    if not np.isfinite(fissure) or fissure < 0.012 or fissure > 0.05:
        raise DomainError("EYE_GLOBE_APERTURE_INVALID", f"Fitted palpebral fissure width {fissure:.4f} m is implausible", 422)
    radius = float(np.clip(GLOBE_FRACTION_OF_FISSURE * fissure, MIN_GLOBE_RADIUS_M, MAX_GLOBE_RADIUS_M))
    limbus = LIMBUS_FRACTION * radius
    z_limbus = math.sqrt(radius * radius - limbus * limbus)
    apex = radius * (1.0 + BULGE_FRACTION)
    a = z_limbus - apex
    corneal_radius = (a * a + limbus * limbus) / (2.0 * abs(a))
    corneal_offset = apex - corneal_radius
    limbus_psi = math.atan2(limbus, z_limbus)

    centers = np.zeros((2, 3), np.float64)
    contact = []
    for eye, rim in enumerate(rims):
        centre_xy = 0.5 * (rim[:, :2].min(axis=0) + rim[:, :2].max(axis=0))
        half_width = 0.5 * float(np.ptp(rim[:, 0]))
        central = rim[np.abs(rim[:, 0] - centre_xy[0]) < 0.6 * half_width]
        dxy = np.linalg.norm(central[:, :2] - centre_xy, axis=1)
        depth = np.sqrt(np.maximum((radius + LID_CONTACT_M) ** 2 - dxy ** 2, 0.0))
        cz = float(np.mean(central[:, 2] - depth))
        centers[eye] = (centre_xy[0], centre_xy[1], cz)
        contact.append({"aperture_width_m": float(np.ptp(rim[:, 0])), "aperture_height_m": float(np.ptp(rim[:, 1])),
                        "rim_vertices": int(len(rim)),
                        "apex_in_front_of_lid_plane_m": float(cz + apex - central[:, 2].mean())})

    sign = _outward_sign(verts, head_faces)
    reach_psi = np.concatenate((np.linspace(0.0, limbus_psi, RINGS_CORNEA),
                                np.linspace(limbus_psi, math.pi, RINGS_SCLERA + 1)[1:]))
    azimuth = np.arange(SEGMENTS) * (2.0 * math.pi / SEGMENTS)
    ring_psi, ring_phi = [reach_psi[0]], [0.0]
    for psi in reach_psi[1:-1]:
        ring_psi.extend([psi] * SEGMENTS)
        ring_phi.extend(azimuth)
    ring_psi.append(reach_psi[-1])
    ring_phi.append(0.0)
    ring_psi, ring_phi = np.asarray(ring_psi), np.asarray(ring_phi)
    per_eye = len(ring_psi)
    faces_local: list[tuple[int, int, int]] = []
    for j in range(SEGMENTS):
        faces_local.append((0, 1 + j, 1 + (j + 1) % SEGMENTS))
    for ring in range(len(reach_psi) - 3):
        a0 = 1 + ring * SEGMENTS
        b0 = a0 + SEGMENTS
        for j in range(SEGMENTS):
            j1 = (j + 1) % SEGMENTS
            faces_local.extend(((a0 + j, b0 + j, a0 + j1), (a0 + j1, b0 + j, b0 + j1)))
    last = 1 + (len(reach_psi) - 3) * SEGMENTS
    south = per_eye - 1
    for j in range(SEGMENTS):
        faces_local.append((last + j, south, last + (j + 1) % SEGMENTS))
    faces_local_arr = np.asarray(faces_local, np.int32)

    all_vertices, all_normals, all_faces, all_tiles, all_eye = [], [], [], [], []
    for eye in range(2):
        points, normals = _globe_surface(radius, corneal_offset, corneal_radius, limbus_psi, centers[eye], ring_psi, ring_phi)
        tri = faces_local_arr.copy()
        edge = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
        outward = points[tri].mean(axis=1) - centers[eye]
        flip = np.einsum("ij,ij->i", edge, outward) * sign < 0
        tri[flip] = tri[flip][:, ::-1]
        # Heads share the template's winding: cross-product outward when sign > 0, inward (mirrored) otherwise.
        all_vertices.append(points)
        all_normals.append(normals)
        all_faces.append(tri + eye * per_eye)
        all_tiles.append(_tile_pixel(ring_psi, ring_phi))
        all_eye.append(np.full(per_eye, eye, np.int8))

    glob_vertices = np.concatenate(all_vertices).astype(np.float32)
    clearance: dict[str, Any] = {}
    if socket_vertices is not None and len(socket_vertices):
        sockets = verts[np.asarray(socket_vertices, np.int64)]
        for eye in range(2):
            near = sockets[np.linalg.norm(sockets[:, :2] - centers[eye, :2], axis=1) < 0.03]
            if len(near):
                distance = np.linalg.norm(near - centers[eye], axis=1)
                clearance[EYE_NAMES[eye]] = {
                    "socket_vertices": int(len(near)),
                    "min_wall_distance_m": float(distance.min()),
                    "wall_vertices_inside_globe": int(np.count_nonzero(distance < radius))}
    not_socket = np.ones(len(verts), bool)
    if socket_vertices is not None:
        not_socket[np.asarray(socket_vertices, np.int64)] = False
    skin = verts[not_socket]
    diagnostics = {
        "model": "procedural_sclera_limbus_cornea_surface_of_revolution",
        "frame": "identity mesh frame: x right, y down, z toward viewer; gaze axis +z (head-forward)",
        "globe_radius_m": radius, "limbus_radius_m": limbus, "cornea_radius_m": corneal_radius,
        "cornea_bulge_m": radius * BULGE_FRACTION, "palpebral_fissure_width_m": fissure,
        "placement": "landmark_aperture_centre_xy; depth_from_lid_margin_contact_with_globe_surface",
        "centers_m": centers.tolist(), "per_eye": dict(zip(EYE_NAMES, contact)), "socket_clearance": clearance,
        "vertices_per_eye": per_eye, "triangles_per_eye": int(len(faces_local_arr)),
        "gaze": "head_forward_prior_target_gaze_not_measured",
        "min_skin_vertex_distance_to_globe_m": float(min(
            np.min(np.linalg.norm(skin - centers[0], axis=1)), np.min(np.linalg.norm(skin - centers[1], axis=1)))),
    }
    is_cornea = ring_psi[faces_local_arr].max(axis=1) <= limbus_psi + 1e-9
    return EyeGlobes(glob_vertices, np.concatenate(all_normals).astype(np.float32),
                     np.concatenate(all_faces).astype(np.int32), np.concatenate(all_tiles).astype(np.float32),
                     np.concatenate(all_eye), centers, radius, limbus, corneal_offset, corneal_radius, limbus_psi,
                     per_eye, int(len(faces_local_arr)), np.concatenate((is_cornea, is_cornea)), diagnostics)


def sample_measured_tiles(globes: EyeGlobes, views: Sequence[dict[str, Any]], remap) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Project every eye-tile texel into the real source views and keep depth-tested BiSeNet-eye observations.

    Each view dict carries ``camera`` (PerspectiveCamera), ``image`` (BGR), ``eye`` (HxW float eye-class probability),
    ``scene_depth`` (HxW camera-space Z of the full head+globes raster, inf outside) and ``quality`` (0..1).
    Returns measured BGR (2, T, T, 3) float32, weight (2, T, T) in 0..1 and per-view statistics.
    """
    psi, phi, valid = tile_grid()
    measured = np.zeros((2, TILE_SIZE, TILE_SIZE, 3), np.float32)
    weight = np.zeros((2, TILE_SIZE, TILE_SIZE), np.float32)
    # Only texels the lids can ever expose are worth projecting; the rest of the tile stays a prior.
    reachable = valid & (psi < math.radians(62.0))
    stats: list[dict[str, Any]] = []
    for eye in range(2):
        ys, xs = np.nonzero(reachable)
        points, normals = globes.surface(eye, psi[ys, xs], phi[ys, xs])
        per_view_w, per_view_c = [], []
        for view in views:
            camera, image, eye_prob, scene_depth = view["camera"], view["image"], view["eye"], view["scene_depth"]
            height, width = image.shape[:2]
            xy = camera.project(points)
            inside = (xy[:, 0] >= 1) & (xy[:, 0] <= width - 2) & (xy[:, 1] >= 1) & (xy[:, 1] <= height - 2)
            cosine = camera.view_cosine(points, normals)
            ok = inside & (cosine > 0.35)
            w = np.zeros(len(points), np.float32)
            colors = np.zeros((len(points), 3), np.float32)
            visible = 0
            if ok.any():
                xi = np.clip(np.rint(xy[ok, 0]).astype(np.int64), 0, width - 1)
                yi = np.clip(np.rint(xy[ok, 1]).astype(np.int64), 0, height - 1)
                depth = camera.depth(points[ok])
                front = np.isfinite(scene_depth[yi, xi]) & (np.abs(scene_depth[yi, xi] - depth) < 0.0007)
                probability = remap(eye_prob.astype(np.float32), xy[ok, 0], xy[ok, 1], cv2.INTER_LINEAR).reshape(-1)
                soft = np.clip((probability - 0.35) / 0.35, 0.0, 1.0)
                keep = front & (soft > 0)
                index = np.nonzero(ok)[0][keep]
                visible = int(len(index))
                if visible:
                    colors[index] = remap(image, xy[index, 0], xy[index, 1], cv2.INTER_LINEAR).reshape(-1, 3)
                    w[index] = (cosine[index] ** 2 * soft[keep] * max(float(view["quality"]), 0.05)).astype(np.float32)
            per_view_w.append(w)
            per_view_c.append(colors)
            stats.append({"eye": EYE_NAMES[eye], "view_index": int(view["view_index"]), "measured_texels": visible})
        weights = np.stack(per_view_w)
        colors = np.stack(per_view_c)
        weights *= weights >= 0.25 * np.maximum(weights.max(axis=0, keepdims=True), 1e-9)
        total = weights.sum(axis=0)
        mean = (weights[..., None] * colors).sum(axis=0) / np.maximum(total, 1e-9)[:, None]
        measured[eye][ys, xs] = mean
        weight[eye][ys, xs] = np.clip(weights.max(axis=0) / 0.30, 0.0, 1.0)
    return measured, weight, stats


def _luma(bgr: np.ndarray) -> np.ndarray:
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def compose_eye_tiles(globes: EyeGlobes, measured: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend measured source texels over the anatomical prior; returns (2, T, T, 3) uint8 and provenance diagnostics."""
    psi, phi, valid = tile_grid()
    limbus = globes.limbus_psi
    iris_zone = psi < limbus
    stats: dict[str, Any] = {"per_eye": {}}
    iris_candidates, sclera_candidates = [], []
    weight = weight.copy()
    rejected: dict[str, int] = {}
    for eye in range(2):
        luma = _luma(measured[eye])
        outside = (psi > 1.1 * limbus) & (weight[eye] > 0.3)
        if int(outside.sum()) >= 80:
            reference = float(np.percentile(luma[outside], 85))
            # Beyond the limbus only sclera is wanted: lashes / lid shadow are dark, lid skin is red-dominant.
            bad = outside & ((luma < 0.66 * reference) | ((measured[eye][..., 2] - measured[eye][..., 0]) > 42.0))
            weight[eye][bad] = 0.0
            rejected[EYE_NAMES[eye]] = int(bad.sum())
    stats["non_sclera_texels_rejected"] = rejected
    solid = (weight > 0.6)
    for eye in range(2):
        m = solid[eye]
        iris_sel = m & (psi > 0.45 * limbus) & (psi < 0.92 * limbus)
        sclera_sel = m & (psi > 1.35 * limbus) & (psi < math.radians(56.0))
        if iris_sel.sum() >= 40:
            iris_candidates.append(measured[eye][iris_sel])
        if sclera_sel.sum() >= 60:
            pixels = measured[eye][sclera_sel]
            bright = pixels[_luma(pixels) >= np.percentile(_luma(pixels), 55)]
            sclera_candidates.append(bright)
    iris_state = "measured" if iris_candidates else "prior_neutral_brown"
    sclera_state = "measured" if sclera_candidates else "prior_neutral_white"
    iris_bgr = np.median(np.concatenate(iris_candidates), axis=0).astype(np.float32) if iris_candidates else PRIOR_IRIS_BGR
    sclera_bgr = np.median(np.concatenate(sclera_candidates), axis=0).astype(np.float32) if sclera_candidates else PRIOR_SCLERA_BGR
    if sclera_candidates:
        # Photo shading, lashes and lid shadow darken the measured sclera; anatomy bounds it from below.
        sclera_bgr = np.maximum(sclera_bgr, 0.85 * PRIOR_SCLERA_BGR)
    prior = _prior_tile(iris_bgr, sclera_bgr, psi, phi, limbus)
    tiles = np.zeros((2, TILE_SIZE, TILE_SIZE, 3), np.uint8)
    for eye in range(2):
        w = weight[eye].copy()
        shift = (0.0, 0.0)
        iris_texels = int(((w > 0.6) & iris_zone).sum())
        colour = measured[eye]
        if iris_texels >= 120:
            dark = (w > 0.6) & (psi < 1.15 * limbus)
            luma = _luma(colour)
            threshold = np.percentile(luma[dark], 45)
            sel = dark & (luma <= threshold)
            cx = float((np.nonzero(sel)[1]).mean())
            cy = float((np.nonzero(sel)[0]).mean())
            centre = (TILE_SIZE - 1) / 2.0
            # Cancel source gaze / landmark registration offset so the rendered gaze is the head-forward prior.
            shift = (centre - cx, centre - cy)
            limit = 0.16 * (TILE_SIZE / 2.0)
            shift = (float(np.clip(shift[0], -limit, limit)), float(np.clip(shift[1], -limit, limit)))
            matrix = np.array(((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1])), np.float32)
            colour = cv2.warpAffine(colour, matrix, (TILE_SIZE, TILE_SIZE), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w = cv2.warpAffine(w, matrix, (TILE_SIZE, TILE_SIZE), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        premultiplied = cv2.GaussianBlur(colour * w[..., None], (0, 0), 2.0)
        w = cv2.GaussianBlur(w, (0, 0), 2.0) * valid
        colour = premultiplied / np.maximum(w[..., None], 1e-3)
        # Beyond the limbus the photo is ~10 px of noisy, lash- and lid-shadowed sclera: keep its broad measured shading only.
        broad = cv2.GaussianBlur(premultiplied, (0, 0), 7.0) / np.maximum(cv2.GaussianBlur(w, (0, 0), 7.0), 1e-3)[..., None]
        colour = np.where((psi > 1.1 * limbus)[..., None], 0.35 * colour + 0.65 * broad, colour)
        blended = prior * (1.0 - w[..., None]) + colour * w[..., None]
        # Lids and brow ridge occlude the globe's upper cap far more than its lower one (geometric ambient occlusion in the
        # head frame, tile y = head y); it is albedo-side shading the SH relight, which has no lid geometry, cannot add.
        top = np.clip(-np.sin(phi), 0.0, 1.0) ** 1.4
        bottom = np.clip(np.sin(phi), 0.0, 1.0) ** 1.4
        reach = np.clip((psi - math.radians(12.0)) / math.radians(32.0), 0.0, 1.0)
        occlusion = 1.0 - reach * (0.36 * top + 0.12 * bottom)
        tiles[eye] = np.clip(blended * occlusion[..., None], 0, 255).astype(np.uint8)
        stats["per_eye"][EYE_NAMES[eye]] = {
            "measured_weight_mean_iris": float(w[iris_zone & valid].mean()),
            "measured_weight_mean_sclera_visible": float(w[(psi >= limbus) & (psi < math.radians(56.0))].mean()),
            "measured_iris_texel_fraction": float(((w > 0.5) & iris_zone).sum() / max(int(iris_zone.sum()), 1)),
            "measured_sclera_texel_fraction": float(((w > 0.5) & (psi >= limbus) & (psi < math.radians(56.0))).sum()
                                                    / max(int(((psi >= limbus) & (psi < math.radians(56.0))).sum()), 1)),
            "gaze_recentering_shift_tile_px": [float(shift[0]), float(shift[1])],
            "gaze_recentering_shift_deg": float(math.hypot(*shift) / (TILE_SIZE / 2.0 - 2.0) * math.degrees(PSI_MAX)),
        }
    stats.update({"iris_colour_bgr": [float(x) for x in iris_bgr], "iris_colour_source": iris_state,
                  "sclera_colour_bgr": [float(x) for x in sclera_bgr], "sclera_colour_source": sclera_state,
                  "tile_size": TILE_SIZE, "texture_map": "azimuthal_equidistant_about_gaze_axis",
                  "prior_detail": "procedural limbal ring, radial iris fibres and pupil (prior, not measured)",
                  "baked_lid_occlusion": "geometric prior: up to 36 % upper-cap, 12 % lower-cap albedo occlusion beyond 12 deg"})
    return tiles, stats


def _prior_tile(iris_bgr: np.ndarray, sclera_bgr: np.ndarray, psi: np.ndarray, phi: np.ndarray, limbus: float) -> np.ndarray:
    """Anatomical prior: sclera, iris with radial fibres + limbal ring, and a pupil. float32 (T, T, 3)."""
    t = psi / limbus                                    # 1.0 at the limbus
    rng = np.random.default_rng(0x45594532)
    fibres = rng.standard_normal(180).astype(np.float32)
    fibre = np.interp(((phi + math.pi) / (2 * math.pi)) * 180.0, np.arange(181), np.append(fibres, fibres[0]))
    fibre = cv2.GaussianBlur(fibre.astype(np.float32), (0, 0), 0.8)
    iris = iris_bgr[None, None, :] * (0.78 + 0.34 * np.clip(1.0 - np.abs(t - 0.62) / 0.5, 0, 1))[..., None]
    iris = iris * (1.0 + 0.10 * fibre)[..., None] * (1.0 - 0.45 * np.clip((t - 0.88) / 0.12, 0, 1) ** 2)[..., None]
    pupil = np.clip((t - PUPIL_FRACTION) / 0.06, 0.0, 1.0)[..., None]
    iris = iris * (0.14 + 0.86 * pupil)
    sclera = np.broadcast_to(sclera_bgr, iris.shape).astype(np.float32)
    # Sclera darkens gently toward the lids/orbit (periphery) and carries a faint warm tint near the limbus.
    periph = np.clip((psi - math.radians(28.0)) / math.radians(40.0), 0, 1)[..., None]
    sclera = sclera * (1.0 - 0.18 * periph)
    edge = np.clip((t - 1.0) / 0.06 + 0.5, 0.0, 1.0)[..., None]     # 0 inside the limbus, 1 outside (soft 6 % band)
    return (iris * (1.0 - edge) + sclera * edge).astype(np.float32)


__all__ = ["EYE_NAMES", "EyeGlobes", "TILE_SIZE", "build_eye_globes", "compose_eye_tiles",
           "sample_measured_tiles", "tile_grid"]
