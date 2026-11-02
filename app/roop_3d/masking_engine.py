"""19-Class Region Masking + DFL XSeg + Text-Prompted CLIPSeg Masking Engine.

Composes four GPU-resident layers:
1. Geometric 3D hull mask (derived from 3ddfa_v2 / 2d106det / 1k3d68 landmarks).
2. Semantic region selector (bisenet_resnet_34.onnx) for skin, eyebrows, eyes, nose, mouth/lips, eyeglasses.
3. Neural foreground occluder (xseg_3.onnx) preserving hands, hair strands, microphones, and props.
4. Text-prompted CLIPSeg mask (clipseg_rd64_refined.onnx) for user-prompted custom object protection.
Includes independent per-face, per-thread watchdog seam tracking.

STATUS (Stage 5 audit): this module is a *mask-combination helper*, not an inference engine, and it is not on the
compositor's call path (only ``tests/test_model_registry.py`` imports it). It never loads BiSeNet, XSeg or CLIPSeg:
``generate_semantic_mask`` / ``generate_xseg_occlusion_mask`` consume arrays the caller already computed, and
``generate_clipseg_prompt_mask`` draws a caller-supplied rectangle (``simulated_activation_box``); there is no
CLIPSeg runtime in this repository, so a text prompt alone returns an empty mask. Do not treat it as evidence that
text-prompted segmentation exists. Real inference lives in ``compositor/masks.py`` (BiSeNet),
``compositor/occlusion_model.py`` (XSeg / face occluder) and ``compositor/occlusion_analysis.py`` (unified analysis).
"""
from __future__ import annotations

import threading
from typing import Any, Iterable, Literal, Tuple

import cv2
import numpy as np

# BiSeNet 19 classes
BISENET_REGIONS: dict[str, list[int]] = {
    "skin": [1],
    "eyebrows": [2, 3],
    "brows": [2, 3],
    "eyes": [4, 5],
    "nose": [10],
    "mouth": [11, 12, 13],
    "lips": [12, 13],
    "eyeglasses": [6],
    "glasses": [6],
    "ears": [7, 8, 9],
    "neck": [14, 15],
    "hair": [17],
    "hat": [18],
    "cloth": [16],
}

DEFAULT_ACTIVE_REGIONS = ("skin", "eyebrows", "eyes", "nose", "mouth")


class MaskingEngine:
    """Multi-layer composable masking engine with watchdog seam baseline isolation."""

    def __init__(self):
        self._seam_baselines: dict[tuple[int, int], dict[str, float]] = {}
        self._lock = threading.Lock()

    def generate_3d_hull_mask(
        self,
        landmarks: np.ndarray | None,
        shape: tuple[int, int],
        feather_radius: int = 5,
    ) -> np.ndarray:
        """Generate smooth geometric convex hull mask from 2D/3D facial landmarks."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)

        if landmarks is None or len(landmarks) < 3:
            # Fallback to centered ellipse
            center = (w // 2, h // 2)
            axes = (int(w * 0.38), int(h * 0.46))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        else:
            # Extract 2D points (x, y)
            pts = np.asarray(landmarks, dtype=np.float32)[:, :2]
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)

        if feather_radius > 0:
            k = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return np.clip(mask, 0.0, 1.0)

    def generate_semantic_mask(
        self,
        bisenet_probs_or_classes: np.ndarray | None,
        selected_regions: Iterable[str] | None = None,
        shape: tuple[int, int] = (512, 512),
        grow_pixels: int = 2,
    ) -> np.ndarray:
        """Generate semantic region mask selecting only requested facial classes."""
        h, w = shape
        active_regions = set(selected_regions) if selected_regions else set(DEFAULT_ACTIVE_REGIONS)

        # Map active region names to class IDs
        active_classes: set[int] = set()
        for reg in active_regions:
            reg_norm = reg.lower().strip()
            if reg_norm in BISENET_REGIONS:
                active_classes.update(BISENET_REGIONS[reg_norm])

        if bisenet_probs_or_classes is None:
            # Synthetic default inner-face mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.36), 1.0, -1)
            return mask

        # If classes map (H, W) or (1, H, W)
        arr = np.asarray(bisenet_probs_or_classes)
        if arr.ndim == 3 and arr.shape[0] == 19:  # (19, H, W) logits/probs
            class_map = np.argmax(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            class_map = arr[0]
        else:
            class_map = arr

        if class_map.shape != (h, w):
            class_map = cv2.resize(class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Build binary selection mask
        mask = np.isin(class_map, list(active_classes)).astype(np.float32)

        if grow_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_pixels * 2 + 1, grow_pixels * 2 + 1))
            mask = cv2.dilate(mask, kernel)

        # Soft boundary anti-aliasing
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return np.clip(mask, 0.0, 1.0)

    def generate_xseg_occlusion_mask(
        self,
        xseg_output: np.ndarray | None,
        shape: tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Extract foreground neural occluder mask (hands, hair, props)."""
        h, w = shape
        if xseg_output is None:
            return np.zeros((h, w), dtype=np.float32)

        arr = np.asarray(xseg_output, dtype=np.float32).squeeze()
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to 0..1
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Subtractive occluder mask (where occluder exists)
        occluder = np.where(arr >= threshold, arr, 0.0)
        occluder = cv2.GaussianBlur(occluder, (3, 3), 0)
        return np.clip(occluder, 0.0, 1.0)

    def generate_clipseg_prompt_mask(
        self,
        prompt: str | None,
        image_shape: tuple[int, int] = (512, 512),
        simulated_activation_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Extract text-prompted CLIPSeg object occlusion mask."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.float32)

        if not prompt or not prompt.strip():
            return mask

        # If specific bounding coordinates are detected or mocked
        if simulated_activation_box:
            bx, by, bw, bh = simulated_activation_box
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return np.clip(mask, 0.0, 1.0)

    def composite_4_layer_mask(
        self,
        geometric_hull: np.ndarray,
        semantic_mask: np.ndarray,
        xseg_occlusion: np.ndarray | None = None,
        clipseg_occlusion: np.ndarray | None = None,
        feather_radius: int = 3,
    ) -> np.ndarray:
        """Composite all 4 mask layers into a single high-precision float32 alpha mask."""
        # 1. Base inner-face area
        h, w = geometric_hull.shape[:2]
        sem_res = cv2.resize(semantic_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        base_face = geometric_hull * sem_res

        # 2. Union of all occlusions
        occlusion = np.zeros((h, w), dtype=np.float32)
        if xseg_occlusion is not None:
            xseg_res = cv2.resize(xseg_occlusion, (w, h), interpolation=cv2.INTER_LINEAR)
            occlusion = np.maximum(occlusion, xseg_res)

        if clipseg_occlusion is not None:
            clip_res = cv2.resize(clipseg_occlusion, (w, h), interpolation=cv2.INTER_LINEAR)
            occlusion = np.maximum(occlusion, clip_res)

        # 3. Subtraction of occluders from base face
        final_mask = base_face * (1.0 - occlusion)

        if feather_radius > 0:
            k = feather_radius * 2 + 1
            final_mask = cv2.GaussianBlur(final_mask, (k, k), 0)

        return np.clip(final_mask, 0.0, 1.0)

    def update_watchdog_seam_baseline(
        self,
        thread_id: int,
        face_id: int,
        mask: np.ndarray,
        seam_tolerance: float = 0.25,
    ) -> tuple[bool, float]:
        """Track independent per-face, per-thread mask seam baselines to avoid
        false-positive watchdog seam resets when neural occluders engage.
        """
        key = (thread_id, face_id)
        current_energy = float(np.mean(mask))

        with self._lock:
            if key not in self._seam_baselines:
                self._seam_baselines[key] = {
                    "mean_energy": current_energy,
                    "count": 1.0,
                }
                return True, 0.0

            b = self._seam_baselines[key]
            prev_mean = b["mean_energy"]
            deviation = abs(current_energy - prev_mean)

            # Update moving baseline
            b["count"] += 1.0
            alpha = 1.0 / min(10.0, b["count"])
            b["mean_energy"] = (1.0 - alpha) * prev_mean + alpha * current_energy

            is_valid = deviation <= seam_tolerance
            return is_valid, deviation
