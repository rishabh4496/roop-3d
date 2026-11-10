"""UV-atlas blending and completion for the multi-view head texture.

* :func:`multiband_blend` - Burt-Adelson Laplacian-pyramid blend of per-view UV layers. Each view contributes through a
  hard winner mask that is low-passed more and more at coarser pyramid levels, so colour/exposure differences are
  feathered over a wide band while fine detail switches over a narrow one: no hard seams and no double-image ghosts.
* :func:`complete_texture` - makes the atlas hole free for DISPLAY: (1) bilateral facial symmetry projection copies
  observed texels onto their mirror position (+x <-> -x on the fitted surface) where one side was never seen;
  (2) whatever is still empty is filled by a harmonic (Laplace) extension of the surrounding texture that relaxes
  towards the subject's own skin tone with distance, so the back/side of the head never shows black holes.
  Completed texels are reported separately (``symmetry`` / ``harmonic``); they are NOT observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- multiband blend
def _pyr_down(image: np.ndarray) -> np.ndarray:
    return cv2.pyrDown(image)


def _pyr_up(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.pyrUp(image, dstsize=(shape[1], shape[0]))


def multiband_blend(colors: np.ndarray, weights: np.ndarray, levels: int = 5) -> np.ndarray:
    """Blend per-view layers.

    colors  (V, H, W, 3) float32; weights (V, H, W) float32 (>= 0, 0 = view has no data at that texel).
    Returns (H, W, 3) float32; texels where every weight is 0 are 0.
    """
    n_views, h, w = weights.shape
    levels = max(1, min(levels, int(np.log2(min(h, w))) - 2))
    winner = weights.argmax(axis=0)
    has_data = weights.max(axis=0) > 0
    mask = np.stack([((winner == i) & has_data).astype(np.float32) for i in range(n_views)])   # hard masks

    # Gaussian pyramids of the masks, and of colour using normalised convolution so empty texels never darken a band.
    mask_pyr: list[list[np.ndarray]] = []
    color_pyr: list[list[np.ndarray]] = []
    for i in range(n_views):
        valid = (weights[i] > 0).astype(np.float32)
        cm, mm = [mask[i]], [mask[i]]
        premult, vv = colors[i] * valid[..., None], valid
        cp = []
        for _ in range(levels):
            norm = premult / np.maximum(vv[..., None], 1e-4)
            cp.append(np.where(vv[..., None] > 1e-3, norm, 0.0).astype(np.float32))
            premult, vv = _pyr_down(premult), _pyr_down(vv)
            cm.append(_pyr_down(cm[-1]))
        norm = premult / np.maximum(vv[..., None], 1e-4)
        cp.append(np.where(vv[..., None] > 1e-3, norm, 0.0).astype(np.float32))
        mask_pyr.append(cm)
        color_pyr.append(cp)

    out: np.ndarray | None = None
    for level in range(levels, -1, -1):
        total = sum(mask_pyr[i][level] for i in range(n_views))
        total = np.maximum(total, 1e-6)
        band = np.zeros_like(color_pyr[0][level])
        for i in range(n_views):
            cur = color_pyr[i][level]
            if level == levels:
                laplace = cur
            else:
                laplace = cur - _pyr_up(color_pyr[i][level + 1], cur.shape[:2])
            band += (mask_pyr[i][level] / total)[..., None] * laplace
        out = band if out is None else _pyr_up(out, band.shape[:2]) + band
    assert out is not None
    out[~has_data] = 0.0
    return np.clip(out, 0, 255).astype(np.float32)


# --------------------------------------------------------------------------- geometry helpers
def nearest_texels(query: np.ndarray, positions: np.ndarray, voxel: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """Nearest of ``positions`` (K, 3) for each ``query`` (Q, 3) point, by voxel hashing (numpy only).

    Returns (index into positions, distance); distance is inf where no point lies within one voxel ring.
    """
    offset = 1 << 20

    def key(cells: np.ndarray) -> np.ndarray:
        c = cells.astype(np.int64) + offset
        return (c[..., 0] << 42) | (c[..., 1] << 21) | c[..., 2]

    cell = np.floor(positions / voxel).astype(np.int64)
    pkeys = key(cell)
    order = np.argsort(pkeys, kind="stable")
    sorted_keys = pkeys[order]
    qcell = np.floor(query / voxel).astype(np.int64)
    best = np.full(len(query), -1, np.int64)
    best_d = np.full(len(query), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                qk = key(qcell + np.array([dx, dy, dz]))
                lo = np.searchsorted(sorted_keys, qk, side="left")
                hi = np.searchsorted(sorted_keys, qk, side="right")
                # up to a few texels per voxel: test the first 4
                for j in range(4):
                    ok = lo + j < hi
                    if not ok.any():
                        break
                    cand = order[np.minimum(lo + j, len(order) - 1)]
                    d = np.linalg.norm(positions[cand] - query, axis=1)
                    better = ok & (d < best_d)
                    best[better], best_d[better] = cand[better], d[better]
    return best, best_d


def _harmonic_extend(image: np.ndarray, known: np.ndarray, levels: int = 5) -> np.ndarray:
    """Laplace extension of ``image`` into ``~known`` texels (coarse-to-fine Jacobi relaxation, known texels pinned)."""
    h, w = known.shape
    pyr_img, pyr_known = [image.astype(np.float32)], [known.astype(np.float32)]
    for _ in range(levels):
        prem = pyr_img[-1] * pyr_known[-1][..., None]
        d_prem, d_known = _pyr_down(prem), _pyr_down(pyr_known[-1])
        pyr_img.append(np.where(d_known[..., None] > 1e-3, d_prem / np.maximum(d_known[..., None], 1e-4), 0.0).astype(np.float32))
        pyr_known.append((d_known > 0.5).astype(np.float32))
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4.0
    current = pyr_img[-1]
    for level in range(levels, -1, -1):
        pinned = pyr_known[level] > 0
        base = pyr_img[level]
        if level != levels:
            current = _pyr_up(current, base.shape[:2])
        current = np.where(pinned[..., None], base, current)
        for _ in range(60 if level >= 3 else 25):
            current = cv2.filter2D(current, -1, kernel, borderType=cv2.BORDER_REFLECT)
            current = np.where(pinned[..., None], base, current)
    return current


def skin_tone(texture: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Subject skin colour (BGR): median of observed texels inside a skin-like YCrCb range, else the overall median."""
    pix = texture[observed].reshape(-1, 1, 3).astype(np.uint8)
    if len(pix) == 0:
        return np.array([150, 160, 190], np.float32)
    ycc = cv2.cvtColor(pix, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    skin = (ycc[:, 1] >= 133) & (ycc[:, 1] <= 173) & (ycc[:, 2] >= 77) & (ycc[:, 2] <= 127) & (ycc[:, 0] > 50)
    chosen = pix.reshape(-1, 3)[skin] if skin.sum() >= 50 else pix.reshape(-1, 3)
    return np.median(chosen, axis=0).astype(np.float32)


@dataclass
class CompletionResult:
    texture: np.ndarray            # (H, W, 4) BGRA uint8, alpha 255 on the mesh footprint, 0 elsewhere
    source: np.ndarray             # (H, W) uint8: 0 outside footprint, 1 observed, 2 symmetry, 3 harmonic
    stats: dict


def complete_texture(texture: np.ndarray, observed: np.ndarray, footprint: np.ndarray, positions: np.ndarray,
                     *, mirror_x: float, symmetry_max_mm: float = 5.0, tone_decay_px: float = 60.0,
                     feather_px: float = 6.0, far_field: np.ndarray | None = None,
                     symmetry_source: np.ndarray | None = None) -> CompletionResult:
    """Fill every footprint texel. ``positions`` is the (H, W, 3) surface position of each texel (metres).

    ``far_field`` (optional (H, W, 3) BGR) replaces the flat skin tone as the colour the harmonic extension relaxes to far from
    any texture (see ``albedo_prior``); ``None`` keeps the flat tone."""
    h, w = footprint.shape
    tex = texture.astype(np.float32).copy()
    known = observed & footprint
    source = np.zeros((h, w), np.uint8)
    source[footprint] = 3
    source[known] = 1
    tone = skin_tone(texture, known)

    # (1) bilateral symmetry: callers may restrict donors to skin-only samples,
    # preserving asymmetrical features while completing unseen skin patches.
    missing = footprint & ~known
    symmetry_known = known if symmetry_source is None else known & np.asarray(symmetry_source, bool)
    if symmetry_known.shape != footprint.shape:
        raise ValueError("symmetry_source must match the UV footprint shape")
    n_sym = 0
    if symmetry_known.any() and missing.any():
        ys, xs = np.nonzero(missing)
        q = positions[ys, xs].astype(np.float64).copy()
        q[:, 0] = 2.0 * mirror_x - q[:, 0]
        ky, kx = np.nonzero(symmetry_known)
        idx, dist = nearest_texels(q, positions[ky, kx].astype(np.float64))
        ok = (idx >= 0) & (dist <= symmetry_max_mm / 1000.0)
        if ok.any():
            src_y, src_x = ky[idx[ok]], kx[idx[ok]]
            tex[ys[ok], xs[ok]] = texture[src_y, src_x]
            source[ys[ok], xs[ok]] = 2
            n_sym = int(ok.sum())
    placed = source == 1
    placed |= source == 2

    # (2) harmonic extension relaxed towards the subject's skin tone with distance from any texture
    rest = footprint & ~placed
    n_harm = int(rest.sum())
    if n_harm:
        if placed.any():
            ext = _harmonic_extend(tex, placed)
            dist = cv2.distanceTransform((~placed).astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.exp(-dist / max(tone_decay_px, 1.0))[..., None]
            target = tone[None, None, :] if far_field is None else np.asarray(far_field, np.float32)
            filled = target + (ext - target) * alpha
        else:
            filled = np.broadcast_to(tone, tex.shape) if far_field is None else np.asarray(far_field, np.float32)
        tex[rest] = filled[rest]

    # soften the seam between completed and observed texels (normalised blur: never pulls in the empty atlas)
    if feather_px > 0 and (source >= 2).any():
        comp = (source >= 2).astype(np.uint8)
        dist_out = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 3)
        fp = footprint.astype(np.float32)
        blurred = cv2.GaussianBlur(tex * fp[..., None], (0, 0), 2.5) / np.maximum(cv2.GaussianBlur(fp, (0, 0), 2.5), 1e-4)[..., None]
        ramp = np.where(comp > 0, 1.0, np.clip(1.0 - dist_out / feather_px, 0.0, 1.0))[..., None]
        tex = np.where(footprint[..., None], tex * (1 - 0.6 * ramp) + blurred * 0.6 * ramp, tex)

    # edge padding outside the footprint so bilinear/mipmap taps at island borders never pull in black
    out = np.clip(tex, 0, 255).astype(np.uint8)
    out[~footprint] = 0
    pad = footprint.copy()
    padded = out.copy()
    for _ in range(4):
        grown = cv2.dilate(pad.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ring = grown & ~pad
        blur = cv2.blur(padded * pad[..., None], (3, 3)).astype(np.float32)
        cnt = cv2.blur(pad.astype(np.float32), (3, 3))
        fillv = np.where(cnt[..., None] > 0, blur / np.maximum(cnt[..., None], 1e-6), 0)
        padded[ring] = np.clip(fillv[ring], 0, 255).astype(np.uint8)
        pad = grown
    alpha_channel = np.where(footprint, 255, 0).astype(np.uint8)
    bgra = np.dstack([padded, alpha_channel])
    total = max(int(footprint.sum()), 1)
    stats = {"observed_ratio": float(known.sum() / total), "symmetry_ratio": float(n_sym / total),
             "harmonic_ratio": float(n_harm / total), "skin_tone_bgr": [float(x) for x in tone]}
    return CompletionResult(texture=bgra, source=source, stats=stats)


def _poisson_blend_view_seams(composite: np.ndarray, colors: np.ndarray, weights: np.ndarray,
                              footprint: np.ndarray, *, threshold: float = 1e-4) -> tuple[np.ndarray, int]:
    """Use local Poisson cloning on overlap seams after the broad multiband stitch."""
    result = np.clip(composite, 0, 255).round().astype(np.uint8)
    present = weights > threshold
    winner = weights.argmax(axis=0)
    seam_count = 0
    for index in range(len(weights)):
        other = np.max(np.where(np.arange(len(weights))[:, None, None] == index, 0.0, weights), axis=0)
        seam = (present[index] & (other > threshold) & (winner != index) & footprint)
        if not seam.any():
            continue
        seam = cv2.dilate(seam.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        seam &= present[index] & footprint
        ys, xs = np.nonzero(seam)
        pad = 8
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, result.shape[0])
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, result.shape[1])
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue
        source = np.clip(colors[index, y0:y1, x0:x1], 0, 255).round().astype(np.uint8)
        destination = result[y0:y1, x0:x1]
        mask = seam[y0:y1, x0:x1] * 255
        try:
            result[y0:y1, x0:x1] = cv2.seamlessClone(
                source, destination, mask, ((x1 - x0) // 2, (y1 - y0) // 2), cv2.NORMAL_CLONE)
            seam_count += 1
        except cv2.error:
            # Tiny or degenerate overlap regions keep their multiband result.
            continue
    return result.astype(np.float32), seam_count


class FullHeadUVCompleter:
    """Stitch observed FaceSet views and complete a full-head UV atlas.

    View layers are already projected into the same canonical UV topology and
    z-buffered by the caller. This class blends their overlap seams, then uses
    symmetric *skin* donors and a harmonic prior for texels unseen at every
    angle. Generated texels remain tagged separately from observations.
    """

    def __init__(self, *, poisson_seams: bool = True, symmetry_max_mm: float = 5.0,
                 tone_decay_px: float = 60.0, feather_px: float = 6.0) -> None:
        if (not np.isfinite((symmetry_max_mm, tone_decay_px, feather_px)).all()
                or symmetry_max_mm <= 0 or tone_decay_px <= 0 or feather_px < 0):
            raise ValueError("UV completion distances must be positive")
        self.poisson_seams = bool(poisson_seams)
        self.symmetry_max_mm = float(symmetry_max_mm)
        self.tone_decay_px = float(tone_decay_px)
        self.feather_px = float(feather_px)

    @staticmethod
    def _skin_donors(texture: np.ndarray, observed: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(np.clip(texture, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        return (observed & (ycc[..., 1] >= 128) & (ycc[..., 1] <= 178)
                & (ycc[..., 2] >= 72) & (ycc[..., 2] <= 132) & (ycc[..., 0] > 35))

    def stitch(self, view_textures: np.ndarray, view_weights: np.ndarray,
               footprint: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """Return an observation-only atlas with multiband and Poisson seam blending."""
        colors = np.asarray(view_textures, np.float32)
        weights = np.asarray(view_weights, np.float32)
        support = np.asarray(footprint, bool)
        if (colors.ndim != 4 or colors.shape[-1] != 3 or weights.shape != colors.shape[:3]
                or support.shape != colors.shape[1:3] or not np.isfinite(colors).all()
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Expected finite view layers (N,H,W,3), weights (N,H,W), and a matching footprint")
        weights = np.where(support[None], weights, 0.0)
        if not (weights > 0).any():
            raise ValueError("At least one observed UV texel is required")
        stitched = multiband_blend(colors, weights)
        poisson_regions = 0
        if self.poisson_seams and len(colors) > 1:
            stitched, poisson_regions = _poisson_blend_view_seams(stitched, colors, weights, support)
        return stitched, {"view_count": int(len(colors)), "poisson_seam_regions": poisson_regions}

    def complete(self, view_textures: np.ndarray, view_weights: np.ndarray, footprint: np.ndarray,
                 positions: np.ndarray, *, mirror_x: float,
                 far_field: np.ndarray | None = None) -> CompletionResult:
        colors = np.asarray(view_textures, np.float32)
        weights = np.asarray(view_weights, np.float32)
        support = np.asarray(footprint, bool)
        surface = np.asarray(positions, np.float32)
        if (colors.ndim != 4 or colors.shape[-1] != 3 or weights.shape != colors.shape[:3]
                or support.shape != colors.shape[1:3] or surface.shape != (*support.shape, 3)
                or not np.isfinite(colors).all() or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isfinite(surface).all()):
            raise ValueError("Expected finite view layers (N,H,W,3), weights (N,H,W), and matching UV geometry")
        weights = np.where(support[None], weights, 0.0)
        observed = weights.max(axis=0) > 0
        if not observed.any():
            raise ValueError("At least one observed UV texel is required")
        stitched, stitch_stats = self.stitch(colors, weights, support)
        observed &= support
        symmetry_source = self._skin_donors(stitched, observed)
        result = complete_texture(stitched, observed, support, surface, mirror_x=mirror_x,
            symmetry_max_mm=self.symmetry_max_mm, tone_decay_px=self.tone_decay_px,
            feather_px=self.feather_px, far_field=far_field, symmetry_source=symmetry_source)
        result.stats.update({**stitch_stats, "skin_symmetry_donors": int(symmetry_source.sum()),
                             "completion_pipeline": "multiband+poisson+skin_symmetry+harmonic"})
        return result
