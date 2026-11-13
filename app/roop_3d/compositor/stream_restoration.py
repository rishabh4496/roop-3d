"""Independent restorer sessions, explicit training alignment and native-size bypass."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from roop_3d.compositor.mask_flow import FlowEstimator
from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import similarity
from roop_3d.render_policy import CudaWorkerContext

# Reference preprocessing coordinates, not a claim of measured output parity.
# github.com/yangxy/GPEN/blob/main/{align_faces.py,face_enhancement.py}
GPEN_REFERENCE = (np.array([[30.29459953, 51.69630051], [65.53179932, 51.50139999],
    [48.02519989, 71.73660278], [33.54930115, 92.3655014], [62.72990036, 92.20410156]]) + [36, 28]) * (512 / 168)
# github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
CODEFORMER_REFERENCE = np.array([[192.98138, 239.94708], [318.90277, 240.1936],
    [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]])


@dataclass(frozen=True)
class RestorerSpec:
    name: str
    model_name: str
    template: np.ndarray
    size: int = 512
    normalization: str = "minus_one_one"

    def __post_init__(self):
        if (np.asarray(self.template).shape != (5, 2) or not np.isfinite(self.template).all()
                or self.size not in (128, 256, 512, 1024) or self.normalization not in ("minus_one_one", "zero_one")):
            raise ValueError("Restorer requires an explicit finite training alignment/input contract")


def gpen_spec(model_name="gpen_bfr_512.onnx"):
    return RestorerSpec("gpen_realistic", model_name, GPEN_REFERENCE.copy())


def codeformer_spec(model_name="CodeFormerv0.1.onnx"):
    return RestorerSpec("codeformer", model_name, CODEFORMER_REFERENCE.copy())


def gfpgan_spec(model_name="GFPGANv1.4.onnx"):
    """GFPGAN v1.4: facexlib FFHQ-512 five-point alignment (same template as CodeFormer), RGB, [-1, 1]."""
    return RestorerSpec("gfpgan", model_name, CODEFORMER_REFERENCE.copy())


def ultramax_spec():
    """Observed preserved processor: CodeFormer fp16, FFHQ-512, RGB [-1,1]."""
    return RestorerSpec("ultramax", "codeformer.fp16.onnx", CODEFORMER_REFERENCE.copy())


def load_ultramax_spec(root):
    import json
    from roop_3d.core.storage import owned
    path = owned(root, "roop_3d/.runtime/restorers/ultramax.json")
    if not path.exists():
        return ultramax_spec()
    if path.stat().st_size > 16384:
        raise ValueError("UltraMax preprocessing contract exceeds the metadata budget")
    info = json.loads(path.read_text())
    if set(info) != {"model_name", "template", "size", "normalization", "evidence"} or not info["evidence"]:
        raise ValueError("UltraMax alignment/normalization evidence is required")
    return RestorerSpec("ultramax", info["model_name"], np.asarray(info["template"], np.float64),
                        info["size"], info["normalization"])


def restore_in_crop_space(restore, image: np.ndarray, points: np.ndarray, spec: RestorerSpec, *, blend: float = 1.0):
    """Align ``image`` (square swap crop) to ``spec``'s training template, restore it and put the result back into
    crop space AT THE ENHANCER'S RESOLUTION (``max(crop, spec.size)``) with a single bicubic warp.

    The crop->frame matrix of the returned image is ``scale_matrix_input(inverse, out / crop)``. Returns
    ``(uint8 BGR out x out, float32 support mask out x out)``; ``blend`` (0..1) is the dynamic enhancer weight.
    """
    from roop_3d.hires_handoff import scale_matrix_input
    side = image.shape[0]
    matrix = similarity(points, spec.template * spec.size / 512)
    crop = cv2.warpAffine(image, matrix, (spec.size, spec.size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    restored = restore(crop)
    if restored.shape != crop.shape or restored.dtype != np.uint8:
        raise ValueError("Restoration returned an invalid crop")
    scale = max(1.0, spec.size / side)
    out = int(round(side * scale))
    to_aligned = scale_matrix_input(matrix, scale)
    warped = cv2.warpAffine(restored, to_aligned, (out, out), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    # Blend only where the aligned crop actually sampled valid input pixels.
    support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (spec.size, spec.size))
    mask = np.clip(cv2.warpAffine(support, to_aligned, (out, out), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), 0, 1)
    base = image if out == side else cv2.resize(image, (out, out), interpolation=cv2.INTER_CUBIC)
    weight = mask * float(blend)
    result = np.clip(warped.astype(np.float32) * weight[:, :, None] + base * (1 - weight[:, :, None]), 0, 255)
    return result.round().astype(np.uint8), mask


class MaskSeamWatchdog:
    """Never compare occlusion coverage to a geometric-face-mask baseline."""
    def __init__(self, maximum=256):
        self.maximum = maximum
        self._baselines = {}
        self._edge_levels = {}
        self._temporal_masks = {}
        self._lock = threading.Lock()

    def observe_temporal(self, face_id: int, mask: np.ndarray, *, frame_index: int | None,
                         scene_cut: bool = False, context: dict | None = None) -> dict:
        report, _ = self.observe_and_recover(face_id, mask, frame_index=frame_index,
                                             scene_cut=scene_cut, context=context)
        return report

    @staticmethod
    def _gray_small(value, size=(128, 128)):
        guide = np.asarray(value)
        if guide.dtype != np.uint8 or guide.ndim not in (2, 3):
            raise ValueError("Mask recovery guides must be uint8 grayscale or BGR images")
        if guide.ndim == 3:
            if guide.shape[2] != 3:
                raise ValueError("Mask recovery guides must be grayscale or BGR")
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        return cv2.resize(guide, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _registered_correlation(first: np.ndarray, second: np.ndarray) -> float:
        a, b = first.astype(np.float32), second.astype(np.float32)
        a -= float(a.mean())
        b -= float(b.mean())
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / denominator) if denominator > 1e-6 else (
            1.0 if np.array_equal(first, second) else 0.0)

    @staticmethod
    def _recover_triplet(mask: np.ndarray, context: dict):
        required = {"previous_mask", "next_mask", "previous_guide", "current_guide", "next_guide"}
        if not isinstance(context, dict) or not required.issubset(context):
            return None, None
        h, w = mask.shape
        previous_mask = np.asarray(context["previous_mask"], np.float32)
        next_mask = np.asarray(context["next_mask"], np.float32)
        if (previous_mask.ndim != 2 or next_mask.ndim != 2
                or min(previous_mask.shape, default=0) < 1 or min(next_mask.shape, default=0) < 1
                or not np.isfinite(previous_mask).all() or not np.isfinite(next_mask).all()
                or np.any((previous_mask < 0) | (previous_mask > 1))
                or np.any((next_mask < 0) | (next_mask > 1))):
            return None, None
        previous_mask = cv2.resize(previous_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        next_mask = cv2.resize(next_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        size = (128, 128)
        try:
            prev_guide, current_guide, next_guide = [
                MaskSeamWatchdog._gray_small(context[name], size)
                for name in ("previous_guide", "current_guide", "next_guide")]
        except (TypeError, ValueError, cv2.error):
            return None, None

        # For a current-frame output pixel, backward flow samples the previous
        # mask, while forward flow samples the next mask. Check each direction
        # against its inverse before allowing either temporal observation.
        try:
            flow = FlowEstimator()
            current_to_previous = flow.calc(current_guide, prev_guide, None)
            previous_to_current = flow.calc(prev_guide, current_guide, None)
            current_to_next = flow.calc(current_guide, next_guide, None)
            next_to_current = flow.calc(next_guide, current_guide, None)
        except (DomainError, cv2.error):
            return None, None
        side = size[0]
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        prev_x, prev_y = xx + current_to_previous[..., 0], yy + current_to_previous[..., 1]
        next_x, next_y = xx + current_to_next[..., 0], yy + current_to_next[..., 1]
        registered_previous = cv2.remap(prev_guide, prev_x, prev_y, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        registered_next = cv2.remap(next_guide, next_x, next_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        previous_correlation = MaskSeamWatchdog._registered_correlation(registered_previous, current_guide)
        next_correlation = MaskSeamWatchdog._registered_correlation(registered_next, current_guide)
        if min(previous_correlation, next_correlation) < .35:
            return None, {"appearance_correlation_previous": previous_correlation,
                          "appearance_correlation_next": next_correlation}
        prev_cycle = cv2.remap(previous_to_current, prev_x, prev_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        next_cycle = cv2.remap(next_to_current, next_x, next_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        prev_error = np.linalg.norm(current_to_previous + prev_cycle, axis=2)
        next_error = np.linalg.norm(current_to_next + next_cycle, axis=2)
        prev_mag = np.linalg.norm(current_to_previous, axis=2)
        next_mag = np.linalg.norm(current_to_next, axis=2)
        prev_valid = ((prev_x >= 0) & (prev_x <= side - 1) & (prev_y >= 0) & (prev_y <= side - 1))
        next_valid = ((next_x >= 0) & (next_x <= side - 1) & (next_y >= 0) & (next_y <= side - 1))
        prev_conf = np.exp(-np.square(prev_error / (.75 + .05 * prev_mag))) * prev_valid
        next_conf = np.exp(-np.square(next_error / (.75 + .05 * next_mag))) * next_valid

        previous_small = cv2.resize(previous_mask, size, interpolation=cv2.INTER_AREA)
        next_small = cv2.resize(next_mask, size, interpolation=cv2.INTER_AREA)
        warped_previous = cv2.remap(previous_small, prev_x, prev_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_next = cv2.remap(next_small, next_x, next_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        confidence = np.minimum(prev_conf, next_conf)
        supported = confidence > .15
        prev_binary, next_binary = (warped_previous >= .5) & supported, (warped_next >= .5) & supported
        union = int(np.logical_or(prev_binary, next_binary).sum())
        agreement = float(np.logical_and(prev_binary, next_binary).sum() / union) if union else 1.0
        confidence_mean = float(confidence.mean())
        area_delta = abs(float((warped_previous >= .5).mean()) - float((warped_next >= .5).mean()))
        if confidence_mean < .20 or agreement < .45 or area_delta > .20:
            return None, {"flow_confidence": confidence_mean, "context_iou": agreement,
                          "context_area_delta": area_delta}
        weights = prev_conf + next_conf
        repaired_small = np.divide(warped_previous * prev_conf + warped_next * next_conf,
                                   weights, out=np.zeros_like(weights), where=weights > 1e-4)
        repaired = cv2.resize(repaired_small, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        occluder = context.get("occluder_mask")
        if occluder is not None:
            occluder = np.asarray(occluder, np.float32)
            if occluder.shape != (h, w) or not np.isfinite(occluder).all() or np.any((occluder < 0) | (occluder > 1)):
                return None, None
            repaired *= (1.0 - occluder)
        return repaired.astype(np.float32), {"flow_confidence": confidence_mean,
                                              "context_iou": agreement,
                                              "context_area_delta": area_delta}

    def observe_and_recover(self, face_id: int, mask: np.ndarray, *, frame_index: int | None,
                            scene_cut: bool = False, context: dict | None = None) -> tuple[dict, np.ndarray]:
        """Measure combined-mask continuity independently for this worker thread and face.

        Recovery is attempted only for a single-frame anomaly and only when explicit
        previous/current/next guides and masks are supplied. With no buffered lookahead,
        this remains a report-only watchdog.
        """
        value = np.asarray(mask, np.float32)
        if (value.ndim != 2 or min(value.shape, default=0) < 1 or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError("Invalid temporal seam watchdog mask")
        small = cv2.resize(value, (128, 128), interpolation=cv2.INTER_AREA) if max(value.shape) > 128 else value
        binary = small >= .5
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge_level = float(np.percentile(edge[edge > .01], 95)) if np.any(edge > .01) else 0.0
        key = (threading.get_ident(), face_id)
        with self._lock:
            previous = None if scene_cut else self._temporal_masks.get(key)
            if previous is not None and previous[0].shape == binary.shape:
                old_binary, old_area, old_edge, old_frame = previous
                consecutive = frame_index is None or old_frame is None or 0 < frame_index - old_frame <= 2
                if consecutive:
                    intersection = float(np.logical_and(old_binary, binary).sum())
                    union = float(np.logical_or(old_binary, binary).sum())
                    iou = intersection / union if union else 1.0
                    area_jump = abs(float(binary.mean()) - old_area)
                    boundary_jump = abs(edge_level - old_edge)
                    anomaly = bool(iou < .45 and area_jump > .12 and boundary_jump > .08)
                    reset_reason = None
                else:
                    iou, area_jump, boundary_jump, anomaly = 1.0, 0.0, 0.0, False
                    reset_reason = "FRAME_GAP"
            else:
                iou, area_jump, boundary_jump, anomaly = 1.0, 0.0, 0.0, False
                reset_reason = ("SCENE_CUT" if scene_cut else
                                "MASK_SHAPE_CHANGED" if previous is not None else "INITIAL")
        repaired = None
        recovery_info = None
        recovery = ("not_attempted_no_lookahead" if context is None else "rejected_context") if anomaly else None
        if anomaly and not scene_cut and context is not None:
            repaired, recovery_info = self._recover_triplet(value, context)
            if repaired is not None:
                recovery = "bidirectional_optical_flow"
        observed = repaired if repaired is not None else value
        observed_small = cv2.resize(observed, (128, 128), interpolation=cv2.INTER_AREA) if max(observed.shape) > 128 else observed
        observed_binary = observed_small >= .5
        observed_gx = cv2.Sobel(observed_small, cv2.CV_32F, 1, 0, ksize=3)
        observed_gy = cv2.Sobel(observed_small, cv2.CV_32F, 0, 1, ksize=3)
        observed_edge = np.sqrt(observed_gx * observed_gx + observed_gy * observed_gy)
        observed_level = float(np.percentile(observed_edge[observed_edge > .01], 95)) if np.any(observed_edge > .01) else 0.0
        with self._lock:
            if len(self._temporal_masks) >= self.maximum and key not in self._temporal_masks:
                self._temporal_masks.pop(next(iter(self._temporal_masks)))
            self._temporal_masks[key] = (observed_binary.copy(), float(observed_binary.mean()), observed_level, frame_index)
        report = {"iou": float(iou), "area_jump": float(area_jump), "boundary_gradient": edge_level,
                  "boundary_gradient_jump": float(boundary_jump), "anomaly": anomaly,
                  "reset_reason": reset_reason, "recovery": recovery}
        if recovery_info is not None:
            report["recovery_metrics"] = recovery_info
        return report, observed.astype(np.float32, copy=False)

    def observe(self, face_id: int, mask: np.ndarray, *, kind: str):
        if kind not in ("geometric", "occluder", "restoration", "composite"):
            raise ValueError("Mask baseline type must be explicit")
        mask = np.asarray(mask, np.float32)
        if mask.ndim != 2 or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("Invalid seam watchdog mask")
        key = (threading.get_ident(), face_id, kind)
        with self._lock:
            old = self._baselines.get(key)
            drift = float(np.abs(mask - old).mean()) if old is not None and old.shape == mask.shape else 0.0
            if key not in self._baselines and len(self._baselines) >= self.maximum:
                oldest = next(iter(self._baselines))
                self._baselines.pop(oldest)
                self._edge_levels.pop(oldest, None)
            self._baselines[key] = mask.copy()
        return drift

    def feather(self, face_id: int, mask: np.ndarray, *, kind: str = "composite",
                minimum_sigma: float = .35, maximum_sigma: float = 2.5) -> np.ndarray:
        """Reduce sharp boundary spikes using a baseline from the same mask class only."""
        if kind not in ("geometric", "occluder", "restoration", "composite"):
            raise ValueError("Mask baseline type must be explicit")
        value = np.asarray(mask, np.float32)
        if value.ndim != 2 or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Invalid seam watchdog mask")
        drift = self.observe(face_id, value, kind=kind)
        gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        boundary = edge > .01
        level = float(np.percentile(edge[boundary], 95)) if np.any(boundary) else 0.0
        key = (threading.get_ident(), face_id, kind)
        with self._lock:
            previous = self._edge_levels.get(key)
            self._edge_levels[key] = level
        spike = max(0.0, level - previous * 1.15) if previous is not None else 0.0
        sigma = float(np.clip(minimum_sigma + drift * 8.0 + spike * 1.5,
                              minimum_sigma, maximum_sigma))
        if sigma <= .5:
            return np.clip(value, 0, 1).astype(np.float32)
        softened = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        return np.clip(softened, 0, 1).astype(np.float32)

    def clear(self):
        with self._lock:
            self._baselines.clear()
            self._edge_levels.clear()
            self._temporal_masks.clear()


class OnnxFaceRestorer:
    def __init__(self, catalog, config, spec: RestorerSpec, *, fidelity_weight=0.5):
        if not np.isfinite(fidelity_weight) or not 0 <= fidelity_weight <= 1:
            raise ValueError("CodeFormer fidelity must be in [0,1]")
        self.catalog, self.config, self.spec, self.fidelity_weight = catalog, config, spec, fidelity_weight
        self.model = None
        self._binding = self._output = self._lookup = None

    def restore(self, aligned):
        from roop_3d.engine.runtime import OrtModel
        if self.model is None:
            from dataclasses import replace
            config = (replace(self.config, graph_optimization="extended")
                      if self.spec.name == "ultramax" and self.config.provider == "CPUExecutionProvider" else self.config)
            self.model = OrtModel(self.catalog, self.spec.model_name, config)
        inputs = self.model.record["graph"]["inputs"]
        if getattr(self, 'require_dynamic_fidelity', False) and not any(item['name'] in ('w','fidelity_weight') for item in inputs):
            from roop_3d.core.errors import DomainError
            raise DomainError('FIDELITY_UNAVAILABLE','This CodeFormer artifact has no runtime fidelity input; select a verified dynamic-w artifact',422)
        image_input = next((v for v in inputs if len(v["shape"]) == 4), None)
        if image_input is None or image_input["shape"][1:] != [3, self.spec.size, self.spec.size]:
            raise ValueError("Restorer graph does not match its registered training input contract")
        dtype = {"FLOAT": np.float32, "FLOAT16": np.float16}.get(image_input["dtype"])
        if dtype is None:
            raise ValueError("Unsupported restorer image dtype")
        if self.spec.name == "ultramax":
            if self._lookup is None:
                values = np.arange(256, dtype=np.float32) / 255
                self._lookup = (values * 2 - 1 if self.spec.normalization == "minus_one_one" else values).astype(dtype)
            values = self._lookup[aligned.transpose(2, 0, 1)[::-1]][None]
        else:
            values = aligned[:, :, ::-1].transpose(2, 0, 1)[None].astype(dtype) / 255
            if self.spec.normalization == "minus_one_one":
                values = values * 2 - 1
        feeds = {image_input["name"]: np.ascontiguousarray(values)}
        for item in inputs:
            if item["name"] == image_input["name"]:
                continue
            if item["name"] not in ("w", "fidelity_weight") or item["dtype"] not in ("FLOAT", "DOUBLE"):
                raise ValueError("Unknown restorer auxiliary input")
            if any(not isinstance(v, int) or v < 1 for v in item["shape"]):
                raise ValueError("Unknown fidelity input shape")
            feeds[item["name"]] = np.full(tuple(item["shape"]), self.fidelity_weight,
                                           np.float64 if item["dtype"] == "DOUBLE" else np.float32)
        if self.spec.name == "ultramax":
            if self.model._stat() != self.model._identity:
                raise ValueError("UltraMax model changed after session initialization")
            if self._binding is None:
                self._binding = self.model.session.io_binding()
            self._binding.clear_binding_inputs()
            self._binding.clear_binding_outputs()
            for name, value in feeds.items():
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite UltraMax input")
                self._binding.bind_cpu_input(name, value)
            image_outputs = [v for v in self.model.record["graph"]["outputs"]
                             if len(v["shape"]) == 4 and v["shape"][1] == 3]
            if len(image_outputs) != 1:
                raise ValueError("UltraMax requires an unambiguous RGB image output")
            name = image_outputs[0]["name"]
            if self._output is None:
                self._binding.bind_output(name, "cuda" if self.model.provider == "CUDAExecutionProvider" else "cpu")
            else:
                self._binding.bind_ortvalue_output(name, self._output)
            self.model.session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()
            self._output = self._binding.get_outputs()[0]
            outputs = {name: self._output.numpy()}
            self.model.executions += 1
        else:
            outputs = self.model.run(feeds)
        candidates = [v for v in outputs.values() if v.shape == (1, 3, self.spec.size, self.spec.size)]
        if len(candidates) != 1 or not np.isfinite(candidates[0]).all():
            raise ValueError("Invalid restorer output contract")
        output = candidates[0].astype(np.float32)
        if self.spec.normalization == "minus_one_one":
            output = (output + 1) * 0.5
        restored = np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).round().astype(np.uint8)
        if self.spec.name in ("ultramax", "gpen_realistic"):
            # Network luminance on the supplied crop's chrominance. No legacy
            # eye/detail filters or legacy performance measurements are inherited.
            delta = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.int16) - cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.int16)
            restored = np.clip(aligned.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)
        return restored

    def close(self):
        if self.model:
            self.model.release()
            self.model = None
        self._binding = self._output = self._lookup = None


class DecoupledRestorer:
    """Worker-pooled restorer. Results are returned at the enhancer's NATIVE resolution in the (square) swap-crop
    frame of reference, i.e. ``max(crop, spec.size)`` pixels, together with ``hires_scale`` so the caller can
    re-base the crop->frame matrix (``roop_3d.hires_handoff.scale_matrix_input``) instead of shrinking the face.

    ``ultramax_native_bypass`` is OFF by default: UltraMax runs on every face unless the user explicitly opts into
    skipping faces whose destination box in the target frame is ``<= ultramax_min_pixels``.
    """
    def __init__(self, factory, spec: RestorerSpec, *, workers=1, pool_size=2, cuda=False,
                 ultramax_native_bypass=False, ultramax_min_pixels=320,
                 cuda_provider='CUDAExecutionProvider', hardware_profile='6gb',
                 enhancer_blend=1.0, fidelity_weight=0.5):
        if not 1 <= workers <= 8 or not 1 <= pool_size <= 32 or ultramax_min_pixels < 1:
            raise ValueError("Invalid restoration pool limits")
        for name, value in (("enhancer_blend", enhancer_blend), ("fidelity_weight", fidelity_weight)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within 0..1")
        # Both are read on every call, so a slider change applies to the next frame without rebuilding a session.
        self.enhancer_blend, self.fidelity_weight = float(enhancer_blend), float(fidelity_weight)
        self.factory, self.spec, self.cuda = factory, spec, cuda
        if cuda_provider not in ('CUDAExecutionProvider', 'TensorrtExecutionProvider') or hardware_profile not in ('6gb', '12gb'):
            raise ValueError('Invalid CUDA restorer provider/profile')
        self.cuda_provider, self.hardware_profile = cuda_provider, hardware_profile
        self.bypass, self.threshold = ultramax_native_bypass, ultramax_min_pixels
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="roop3d-restorer")
        self.slots, self.local = threading.BoundedSemaphore(pool_size), threading.local()
        self.instances, self.lock = [], threading.Lock()
        self.watchdog = MaskSeamWatchdog()
        self.closed = False

    def hires_scale(self, crop_side: int) -> float:
        """S = enhancer_res / swap_res (never below 1): how much larger a restored face is than its swap crop."""
        return max(1.0, self.spec.size / crop_side)

    def cuda_adapter(self, stream):
        """Worker-owned device restoration, preserving explicit training geometry.

        This opt-in adapter uses the registered graph's actual I/O metadata and
        normalization contract. It never calls the host restore/submit path.
        """
        import torch
        from dataclasses import replace
        from roop_3d.engine.runtime import OrtModel
        from roop_3d.gpu_compositor import TorchIOBinding, stream_session, warp, invert_affine
        from roop_3d.gpu_transforms import normalize_model_input, model_output_to_rgb01
        from roop_3d.vectorized_pipeline import crop_matrix
        adapter = self.factory()
        if not isinstance(adapter, OnnxFaceRestorer):
            raise ValueError('CUDA restoration requires the registered ONNX restorer adapter')
        model_config = (replace(adapter.config, provider='CUDAExecutionProvider')
                        if self.cuda_provider == 'TensorrtExecutionProvider' else adapter.config)
        adapter.model = OrtModel(adapter.catalog, self.spec.model_name, model_config)
        with self.lock:
            self.instances.append(adapter)
        if self.cuda_provider == 'TensorrtExecutionProvider':
            from roop_3d.trt_session_builder import TensorRTSessionBuilder, BatchProfile
            session = TensorRTSessionBuilder(hardware_profile=self.hardware_profile).build(
                adapter.model.path, profile=BatchProfile((1,), 1), compute_stream=stream.cuda_stream)
        else:
            session = stream_session(adapter.model.session, stream)
        adapter.model.session = session
        runner = TorchIOBinding(session, stream)
        image_inputs = [v for v in session.get_inputs() if len(v.shape) == 4]
        image_outputs = [v for v in session.get_outputs() if len(v.shape) == 4 and v.shape[1:] == [3, self.spec.size, self.spec.size]]
        dtypes = {'tensor(float)': torch.float32, 'tensor(float16)': torch.float16, 'tensor(double)': torch.float64}
        if (len(image_inputs) != 1 or len(image_outputs) != 1
                or image_inputs[0].shape[1:] != [3, self.spec.size, self.spec.size]
                or image_inputs[0].type not in dtypes or image_outputs[0].type not in dtypes):
            raise ValueError('CUDA restorer image contract is unsupported')
        image_input, image_output = image_inputs[0], image_outputs[0]
        auxiliary = {}
        with torch.cuda.stream(stream):
            for value in session.get_inputs():
                if value.name == image_input.name:
                    continue
                if (value.name not in ('w', 'fidelity_weight') or value.type not in dtypes
                        or any(type(d) is not int or d < 1 for d in value.shape)):
                    raise ValueError('Unknown CUDA restorer auxiliary input contract')
                auxiliary[value.name] = torch.full(value.shape, adapter.fidelity_weight,
                                                   dtype=dtypes[value.type], device=stream.device)

        def restore(image, face):
            """Return the restored face in crop space at the enhancer's native resolution (``max(crop, spec.size)``).

            The enhanced tensor is NOT shrunk back to the crop size: the compositor re-bases the crop->frame matrix by
            ``out_size / image.shape[-1]`` and warps this tensor straight into the frame.
            """
            import torch.nn.functional as F
            from roop_3d.hires_handoff import bicubic_warp, scale_matrix_input
            if self.closed or adapter.model._stat() != adapter.model._identity:
                raise ValueError('CUDA restorer is closed or its model changed')
            if self.spec.name == 'ultramax' and self.bypass:
                # Opt-in only. Judged on the destination face box in the target frame, never on the working crop.
                native = min(max(0, face.box[2] - face.box[0]), max(0, face.box[3] - face.box[1]))
                if native <= self.threshold:
                    return image
            in_size = image.shape[-1]
            crop = crop_matrix(face.landmarks5, in_size)
            points = face.landmarks5 @ crop[:, :2].T + crop[:, 2]
            matrix_np = similarity(points, self.spec.template * self.spec.size / 512)
            matrix = torch.as_tensor(matrix_np, device=image.device, dtype=image.dtype)[None]
            aligned = warp(image, invert_affine(matrix), (self.spec.size, self.spec.size))
            values = normalize_model_input(aligned, channel_order='RGB', value_range=self.spec.normalization)
            values = values.to(dtypes[image_input.type]).contiguous()
            for tensor in auxiliary.values():  # dynamic fidelity slider, applied to the very next frame
                tensor.fill_(self.fidelity_weight)
            output = runner.run({image_input.name: values, **auxiliary},
                {image_output.name: ((1, 3, self.spec.size, self.spec.size), dtypes[image_output.type])})[image_output.name]
            output = model_output_to_rgb01(output, value_range=self.spec.normalization, channel_order='RGB')
            if self.spec.name in ('ultramax', 'gpen_realistic'):
                coefficients = image.new_tensor((.299, .587, .114))[None, :, None, None]
                output = (aligned + ((output - aligned) * coefficients).sum(1, keepdim=True)).clamp(0, 1)
            scale = max(1.0, self.spec.size / in_size)
            out_size = int(round(in_size * scale))
            to_aligned = torch.as_tensor(scale_matrix_input(matrix_np, scale), device=image.device,
                                         dtype=image.dtype)[None]
            restored = bicubic_warp(output, to_aligned, (out_size, out_size))
            support = bicubic_warp(warp(torch.ones_like(image[:, :1]), invert_affine(matrix),
                                        (self.spec.size, self.spec.size)), to_aligned, (out_size, out_size))
            base = image if out_size == in_size else F.interpolate(image, size=(out_size, out_size), mode='bicubic',
                                                                    align_corners=False).clamp(0, 1)
            return base + (restored - base) * (support * self.enhancer_blend)
        return restore

    def submit(self, image, landmarks5, native_box, face_id, *, native_shape=None):
        box = np.asarray(native_box, np.float64).copy()
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Native face box must be finite XYXY")
        if native_shape is not None:
            box = box.clip([0, 0, 0, 0], [native_shape[1], native_shape[0], native_shape[1], native_shape[0]])
        native = min(max(0, box[2] - box[0]), max(0, box[3] - box[1]))
        if self.spec.name.casefold() == "ultramax" and self.bypass and native <= self.threshold:
            future = Future()
            # Explicit user opt-in only. Same output geometry as a restored face so callers can batch both.
            out = int(round(image.shape[0] * self.hires_scale(image.shape[0])))
            future.set_result(image.copy() if out == image.shape[0]
                              else cv2.resize(image, (out, out), interpolation=cv2.INTER_CUBIC))
            return future  # No model construction, alignment or queue contention.
        if self.closed:
            raise RuntimeError("Restorer pool is closed")
        self.slots.acquire()
        try:
            future = self.executor.submit(self._restore, image.copy(), np.asarray(landmarks5).copy(), face_id)
            future.add_done_callback(lambda _: self.slots.release())
            return future
        except BaseException:
            self.slots.release()
            raise

    def _restore(self, image, points, face_id):
        if not hasattr(self.local, "model"):
            self.local.context = CudaWorkerContext(enabled=self.cuda)
            with self.local.context.activate():
                self.local.model = self.factory()
            with self.lock:
                self.instances.append(self.local.model)
        with self.local.context.activate():
            self.local.model.fidelity_weight = self.fidelity_weight  # dynamic slider, read per call
            result, mask = restore_in_crop_space(self.local.model.restore, image, points, self.spec,
                                                 blend=self.enhancer_blend)
            self.watchdog.observe(face_id, mask, kind="restoration")
            return result

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        for model in self.instances:
            if hasattr(model, "close"):
                model.close()
        self.instances.clear()
        self.watchdog.clear()
