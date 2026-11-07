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
                     feather_px: float = 6.0) -> CompletionResult:
    """Fill every footprint texel. ``positions`` is the (H, W, 3) surface position of each texel (metres)."""
    h, w = footprint.shape
    tex = texture.astype(np.float32).copy()
    known = observed & footprint
    source = np.zeros((h, w), np.uint8)
    source[footprint] = 3
    source[known] = 1
    tone = skin_tone(texture, known)

    # (1) bilateral symmetry: a missing texel takes the colour at its mirrored surface position when that was observed
    missing = footprint & ~known
    n_sym = 0
    if known.any() and missing.any():
        ys, xs = np.nonzero(missing)
        q = positions[ys, xs].astype(np.float64).copy()
        q[:, 0] = 2.0 * mirror_x - q[:, 0]
        ky, kx = np.nonzero(known)
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
            filled = tone[None, None, :] + (ext - tone[None, None, :]) * alpha
        else:
            filled = np.broadcast_to(tone, tex.shape)
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
