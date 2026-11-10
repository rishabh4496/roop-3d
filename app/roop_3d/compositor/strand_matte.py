# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Directional strand matte: thin hair strands and facial hair over the face, from the target's own pixels.

NeuralHaircut-inspired in its goal only (hair as thin oriented structures rather than a blob); it is a single-frame 2D
filter bank, not a strand reconstruction. A bank of even Gabor filters finds thin ridges; the structure tensor's coherence
rejects isotropic skin texture and noise; the hair-flow direction measured inside the hair region gates orientation; a local
skin-to-hair colour line rejects ridges that are not hair-coloured; and a ridge only qualifies near hair or when connected to
such a ridge (hysteresis, up to ``strand_max_length_px``), so a skin crease touching no hair is never promoted.

It lives in ``compositor`` (matting) so both the robust compositor and ``rendering.strand_occlusion`` can use it without a cycle.
All lengths are in pixels of the image passed in; :func:`scaled_for_face` rescales them to the face's size in that image.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from pydantic import Field

from roop_3d.core.settings import Contract

_LUMA = np.array((0.114, 0.587, 0.299), np.float32)         # BGR
REFERENCE_FACE_PX = 250.0          # the face width the default pixel lengths were calibrated on (strands ~1.25 px wide)


class StrandMatteOptions(Contract):
    gabor_orientations: int = Field(default=12, ge=4, le=32)
    gabor_wavelength_px: float = Field(default=4.0, ge=2.0, le=24.0, description="Ridge period; scale with the face's resolution")
    gabor_sigma_px: float = Field(default=1.7, ge=0.8, le=12.0)
    strand_reach_px: float = Field(default=10.0, ge=1.0, le=64.0, description="Ridges this close to hair qualify on their own")
    strand_max_length_px: float = Field(default=96.0, ge=2.0, le=512.0, description="Longest strand kept when connected to hair")
    strand_response_low: float = Field(default=0.018, ge=0.0, le=1.0)
    strand_response_high: float = Field(default=0.06, gt=0.0, le=1.0)
    strand_gain: float = Field(default=1.0, ge=0.0, le=1.0)
    flow_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="How much the hair-flow direction gates strand orientation")
    colour_gate: bool = True


def scaled_for_face(options: StrandMatteOptions, face_px: float) -> StrandMatteOptions:
    """Options with every pixel length rescaled from the calibration face width to ``face_px`` (clamped to the legal ranges)."""
    k = float(np.clip(face_px / REFERENCE_FACE_PX, 0.5, 6.0))
    return options.model_copy(update={
        "gabor_wavelength_px": float(np.clip(options.gabor_wavelength_px * k, 2.0, 24.0)),
        "gabor_sigma_px": float(np.clip(options.gabor_sigma_px * k, 0.8, 12.0)),
        "strand_reach_px": float(np.clip(options.strand_reach_px * k, 1.0, 64.0)),
        "strand_max_length_px": float(np.clip(options.strand_max_length_px * k, 2.0, 512.0))})


def _smoothstep(x, lo: float, hi: float):
    t = np.clip((np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _normalised_blur(data: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    k = (2 * radius + 1, 2 * radius + 1)
    w = weight.astype(np.float32)
    den = cv2.GaussianBlur(w, k, sigma, borderType=cv2.BORDER_REPLICATE)
    num = cv2.GaussianBlur(data * (w[..., None] if data.ndim == 3 else w), k, sigma, borderType=cv2.BORDER_REPLICATE)
    return num / (np.maximum(den, 1e-4)[..., None] if data.ndim == 3 else np.maximum(den, 1e-4))


def gabor_bank(count: int, wavelength: float, sigma: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Zero-mean, L1-normalised even Gabor kernels; ``theta`` is the direction of the kernel's wave (the ridge normal)."""
    size = int(2 * math.ceil(3.0 * sigma) + 1)
    thetas = np.pi * np.arange(count) / count
    kernels = []
    for theta in thetas:
        kernel = cv2.getGaborKernel((size, size), sigma, float(theta), wavelength, 0.6, 0.0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        kernels.append((kernel / max(float(np.abs(kernel).sum()), 1e-6)).astype(np.float32))
    return thetas, kernels


def strand_evidence(target_bgr: np.ndarray, hair_prior: np.ndarray, skin_prior: np.ndarray,
                    options: StrandMatteOptions) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Strand alpha (0..1) for a crop, plus the intermediate maps (ridge, coherence, flow gate, colour, reach)."""
    o = options
    image = np.asarray(target_bgr, np.float32) / 255.0
    luma = image @ _LUMA
    hair = np.clip(hair_prior, 0.0, 1.0).astype(np.float32)
    band_pass = luma - cv2.GaussianBlur(luma, (0, 0), 1.5 * o.gabor_wavelength_px)
    thetas, kernels = gabor_bank(o.gabor_orientations, o.gabor_wavelength_px, o.gabor_sigma_px)
    responses = np.stack([np.abs(cv2.filter2D(band_pass, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)) for k in kernels])
    best = responses.argmax(axis=0)
    ridge = responses.max(axis=0)
    theta_best = thetas[best].astype(np.float32)

    # orientation coherence of the neighbourhood (structure tensor of the band-passed signal)
    gx = cv2.Sobel(band_pass, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(band_pass, cv2.CV_32F, 0, 1, ksize=3)
    sigma_t = max(1.0, 0.75 * o.gabor_wavelength_px)
    jxx, jxy, jyy = (cv2.GaussianBlur(a, (0, 0), sigma_t) for a in (gx * gx, gx * gy, gy * gy))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / np.maximum(jxx + jyy, 1e-8)
    phi = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)                     # gradient (= ridge normal) orientation
    consistency = np.cos(theta_best - phi) ** 2

    # hair-flow direction from inside the hair region, extended outward by normalised convolution
    flow_sigma = 3.0 * o.gabor_wavelength_px
    wx, wy, wc = (_normalised_blur(a, hair, flow_sigma) for a in (gx * gx - gy * gy, 2.0 * gx * gy, np.ones_like(gx)))
    flow_phi = 0.5 * np.arctan2(wy, wx)
    flow_confidence = np.clip(cv2.GaussianBlur(hair, (0, 0), flow_sigma) * 4.0, 0.0, 1.0)
    alignment = np.cos(theta_best - flow_phi) ** 2
    flow_gate = 1.0 - o.flow_weight * flow_confidence * (1.0 - alignment)

    # Strands only exist where hair does. Two ways to qualify: a ridge within ``strand_reach_px`` of hair (flyaways), or a
    # ridge CONNECTED to such a ridge (a bang hanging from the hairline), up to ``strand_max_length_px`` from the hair.
    # Skin lines that touch no hair are never promoted. (Hysteresis: an isolated 40 px wrinkle is rejected, a 40 px
    # strand attached to the hairline is kept whole.)
    far = (hair < 0.5).astype(np.uint8)
    if far.all():
        reach = np.zeros_like(hair)
    else:
        distance = cv2.distanceTransform(far, cv2.DIST_L2, 3)
        near = _smoothstep(1.0 - distance / o.strand_reach_px, 0.0, 0.6)
        reach_far = 1.0 - _smoothstep(distance, 0.7 * o.strand_max_length_px, o.strand_max_length_px)
        reach = near
    # colour line between the local skin and hair colours
    colour = np.ones_like(hair)
    if o.colour_gate and (hair > 0.6).sum() >= 16 and (skin_prior > 0.5).sum() >= 16:
        sigma_c = max(4.0, 2.0 * o.strand_reach_px)
        hair_colour = _normalised_blur(image, (hair > 0.6).astype(np.float32), sigma_c)
        skin_colour = _normalised_blur(image, (skin_prior > 0.5).astype(np.float32), sigma_c)
        direction = hair_colour - skin_colour
        norm2 = (direction ** 2).sum(-1)
        along = np.clip(((image - skin_colour) * direction).sum(-1) / np.maximum(norm2, 1e-4), 0.0, 1.0)
        informative = _smoothstep(np.sqrt(norm2), 0.04, 0.10)
        colour = (1.0 - informative) + informative * along
    strength = _smoothstep(ridge * coherence * consistency, o.strand_response_low, o.strand_response_high)
    evidence = strength * flow_gate * colour
    if not far.all():
        candidate = (evidence > 0.25).astype(np.uint8)
        joined = cv2.dilate(candidate, np.ones((3, 3), np.uint8))
        count, labels = cv2.connectedComponents(joined, connectivity=8)
        anchored = np.unique(labels[(near > 0.3) & (joined > 0)])
        anchored = anchored[anchored > 0]
        if len(anchored):
            attached = np.isin(labels, anchored) & (candidate > 0)
            reach = np.maximum(near, attached.astype(np.float32) * reach_far)
    strand = np.clip(o.strand_gain * evidence * reach, 0.0, 1.0)
    strand = cv2.GaussianBlur(strand.astype(np.float32), (0, 0), 0.6)
    return strand.astype(np.float32), {"ridge": ridge, "coherence": coherence, "flow_gate": flow_gate,
                                       "colour": colour, "reach": reach}
