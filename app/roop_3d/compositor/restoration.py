"""Optional ROI restoration, blended only through the face alpha."""
from __future__ import annotations

import time

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


def blend_restored_roi(image: np.ndarray, restored: np.ndarray, face_mask: np.ndarray, blend: float,
                       protect: np.ndarray | None = None) -> np.ndarray:
    """Blend an enhancer result inside the facial mask, preserving outside pixels.

    ``protect`` (0..1, optional) removes the restorer's influence where it must not rewrite pixels:
    eyes, brows, lips, mouth interior and teeth (Stage 5 restoration guard).
    """
    source = np.asarray(image)
    enhanced = np.asarray(restored)
    mask = np.asarray(face_mask, dtype=np.float32)
    if protect is not None:
        guard = np.asarray(protect, dtype=np.float32)
        if guard.shape != mask.shape or not np.isfinite(guard).all():
            raise DomainError("RESTORATION_INPUT_INVALID", "Restoration protection map must match the face mask", 422)
        mask = mask * (1.0 - np.clip(guard, 0.0, 1.0))
    if (source.ndim != 3 or source.shape[2] != 3 or enhanced.shape != source.shape
            or mask.shape != source.shape[:2] or not np.isfinite(source).all()
            or not np.isfinite(enhanced).all() or not np.isfinite(mask).all()):
        raise DomainError("RESTORATION_INPUT_INVALID", "Restoration image and face mask dimensions are invalid", 422)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("Restoration blend must be finite and within 0..1")
    if blend == 0:
        return source.copy()
    result = source.astype(np.float32, copy=True)
    active = mask > 0
    weight = np.clip(mask, 0.0, 1.0) * float(blend)
    result[active] = (source.astype(np.float32)[active] * (1.0 - weight[active, None])
                      + enhanced.astype(np.float32)[active] * weight[active, None])
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        return np.clip(np.nan_to_num(result), limits.min, limits.max).round().astype(source.dtype)
    return np.nan_to_num(result).astype(source.dtype)


class GpenRestorer:
    """Lazy local GPEN adapter. Model bytes are hash checked by the Stage 03 catalog."""

    def __init__(self, catalog=None, config=None, *, model_name: str = "gpen_bfr_512.onnx",
                 normalization: str = "minus_one_one"):
        from roop_3d.engine.runtime import RuntimeConfig
        from roop_3d.models.catalog import ModelCatalog

        if normalization not in {"zero_one", "minus_one_one"}:
            raise ValueError("Unsupported GPEN normalization")
        self.catalog = catalog or ModelCatalog()
        self.config = config or RuntimeConfig()
        self.model_name = model_name
        self.normalization = normalization
        self.model = None
        self.inference_verified = False

    def _load(self):
        if self.model is None:
            from roop_3d.engine.runtime import OrtModel
            self.model = OrtModel(self.catalog, self.model_name, self.config)
        return self.model

    def restore(self, image: np.ndarray) -> np.ndarray:
        from roop_3d.engine.runtime import image_bgr as validate_image

        validate_image(image)
        shape = (512, 512)
        rgb = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        if self.normalization == "minus_one_one":
            rgb = rgb * 2.0 - 1.0
        values = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self._load().run({"input": values})
        if "output" not in output:
            raise DomainError("RESTORATION_OUTPUT_INVALID", "GPEN output tensor is unavailable", 503)
        result = np.asarray(output["output"], dtype=np.float32)
        if result.shape != (1, 3, 512, 512) or not np.isfinite(result).all():
            raise DomainError("RESTORATION_OUTPUT_INVALID", "GPEN output tensor has an unsupported contract", 503)
        result = result[0].transpose(1, 2, 0)
        if self.normalization == "minus_one_one":
            result = (result + 1.0) * 0.5
        bgr = np.clip(result[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
        self.inference_verified = True
        return cv2.resize(bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    def restore_hires(self, image: np.ndarray, landmarks5=None, **_unused) -> np.ndarray:
        """GPEN output at its native 512 px (BGR uint8), expressed in the crop's frame of reference.

        The caller re-bases the crop->frame matrix by ``512 / crop`` and warps this tensor straight into the frame;
        ``restore`` (which shrinks back to the crop) stays only for callers that need crop-sized pixels.
        """
        from roop_3d.engine.runtime import image_bgr as validate_image

        validate_image(image)
        shape = (512, 512)
        interpolation = cv2.INTER_CUBIC if image.shape[0] < 512 else cv2.INTER_AREA
        rgb = cv2.resize(image, shape, interpolation=interpolation)[:, :, ::-1].astype(np.float32) / 255.0
        if self.normalization == "minus_one_one":
            rgb = rgb * 2.0 - 1.0
        values = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self._load().run({"input": values})
        if "output" not in output:
            raise DomainError("RESTORATION_OUTPUT_INVALID", "GPEN output tensor is unavailable", 503)
        result = np.asarray(output["output"], dtype=np.float32)
        if result.shape != (1, 3, 512, 512) or not np.isfinite(result).all():
            raise DomainError("RESTORATION_OUTPUT_INVALID", "GPEN output tensor has an unsupported contract", 503)
        result = result[0].transpose(1, 2, 0)
        if self.normalization == "minus_one_one":
            result = (result + 1.0) * 0.5
        self.inference_verified = True
        return np.clip(result[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)

    def capability(self) -> dict[str, str]:
        return {"weights": "VERIFIED_LOCAL", "provenance": "UNKNOWN", "license": "UNKNOWN",
                "inference": "VERIFIED_LOCAL" if self.inference_verified else "UNKNOWN",
                "preprocessing": "CANDIDATE"}

    def release(self) -> None:
        if self.model is not None:
            self.model.release()
        self.model = None


class CodeFormerRestorer:
    """Lazy, hash-verified CodeFormer with training alignment in replacement-crop space."""
    def __init__(self, catalog, config, spec=None):
        from roop_3d.compositor.stream_restoration import OnnxFaceRestorer, codeformer_spec
        self.spec = spec if spec is not None else codeformer_spec()
        self.runner = OnnxFaceRestorer(catalog, config, self.spec)
        self.runner.require_dynamic_fidelity = True
        self.fidelity_weight = 0.5
        self.inference_verified = False
        self.last_inference_ms: float | None = None

    def _infer(self, aligned: np.ndarray) -> np.ndarray:
        """One model run on a spec-aligned crop; its wall time is observed here, around the real inference call."""
        began = time.perf_counter()
        restored = self.runner.restore(aligned)
        self.last_inference_ms = (time.perf_counter() - began) * 1000
        return restored

    def restore(self, image: np.ndarray, landmarks5: np.ndarray | None = None) -> np.ndarray:
        if landmarks5 is not None:
            return self.restore_aligned(image, landmarks5)
        self.runner.fidelity_weight = self.fidelity_weight
        restored = self._infer(image)
        self.inference_verified = True
        return restored

    def restore_aligned(self, image, landmarks5):
        from roop_3d.engine.volumetric.preprocess import similarity
        points = np.asarray(landmarks5, np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise DomainError('RESTORER_LANDMARKS_REQUIRED', 'CodeFormer requires five crop-space landmarks', 422)
        matrix = similarity(points, self.spec.template)
        crop = cv2.warpAffine(image, matrix, (self.spec.size, self.spec.size), borderMode=cv2.BORDER_CONSTANT)
        self.runner.fidelity_weight = self.fidelity_weight
        restored = self._infer(crop)
        inverse = cv2.invertAffineTransform(matrix)
        size = (image.shape[1], image.shape[0])
        warped = cv2.warpAffine(restored, inverse, size)
        support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (self.spec.size, self.spec.size))
        alpha = np.clip(cv2.warpAffine(support, inverse, size), 0, 1)[..., None]
        self.inference_verified = True
        return np.clip(warped * alpha + image * (1-alpha), 0, 255).round().astype(np.uint8)

    def restore_hires(self, image, landmarks5=None, **_unused):
        """Restored face at the enhancer's native resolution, in the crop's frame of reference (BGR uint8)."""
        from roop_3d.compositor.stream_restoration import restore_in_crop_space
        self.runner.fidelity_weight = getattr(self, 'fidelity_weight', 0.5)
        # CodeFormer / UltraMax are trained on five-point FFHQ alignment: without landmarks there is no valid input, and
        # running the model on an unaligned crop would silently degrade the result. Fail closed, as before.
        points = np.asarray(landmarks5 if landmarks5 is not None else (), np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise DomainError('RESTORER_LANDMARKS_REQUIRED', 'CodeFormer requires five crop-space landmarks', 422)
        result, _ = restore_in_crop_space(self._infer, image, points, self.spec)
        self.inference_verified = True
        return result

    def capability(self):
        return {'architecture': 'codeformer', 'alignment': 'CODEFORMER_REFERENCE_5PTS',
                'inference': 'VERIFIED_THIS_SESSION' if self.inference_verified else 'NOT_RUN',
                'weights': 'VERIFIED_LOCAL' if self.runner.model is not None else 'NOT_LOADED'}

    def release(self):
        self.runner.close()
        self.inference_verified = False


class UltraMaxPipelineRestorer(CodeFormerRestorer):
    """UltraMax high-detail restoration: CodeFormer at 512 px training alignment, then a 3-level Laplacian pyramid
    fusion that keeps the swap crop's colour and low/mid frequencies and takes only the restorer's finest detail.

    The weights, alignment template and normalization come from ``load_ultramax_spec`` (the same contract the
    optimized path, ``/system/capabilities`` and the model rack use), so every route runs the registered
    ``codeformer.fp16.onnx`` artifact through the UltraMax runner. It is CodeFormer plus Laplacian detail fusion: there
    is no Real-ESRGAN cascade. Native face size controls whether small background faces bypass inference or receive an
    adaptively reduced detail pass; only that explicit user setting may return/skip without inference, and every
    outcome is recorded in ``telemetry``. Inference errors always propagate.
    """
    def __init__(self, catalog, config, spec=None):
        from roop_3d.compositor.stream_restoration import RestorationTelemetry, ultramax_spec
        if spec is None:
            resolve = getattr(catalog, "ultramax_spec", None)
            spec = resolve() if callable(resolve) else ultramax_spec()
        if spec.name != "ultramax":
            raise ValueError("UltraMax requires the ultramax restorer contract")
        super().__init__(catalog, config, spec)
        self.fidelity_weight = 0.7
        # "off" = UltraMax runs on every face. "bypass" / "adaptive" are explicit user settings, judged on the face's
        # destination size in the target frame (never on the size of the swap crop).
        self.small_face_mode = "off"
        self.min_native_pixels = 320
        self.telemetry = RestorationTelemetry(spec.model_name)

    def _small(self, native_face_size_px) -> bool:
        return native_face_size_px is not None and native_face_size_px <= self.min_native_pixels

    def _bypassed(self, native_face_size_px) -> None:
        self.telemetry.record_bypassed("small_face_bypass", native_face_px=round(float(native_face_size_px), 1),
                                       threshold_px=self.min_native_pixels)

    def _applied(self, began: float, **detail) -> None:
        self.telemetry.record_applied((time.perf_counter() - began) * 1000, self.last_inference_ms, **detail)

    def restore_aligned(self, image, landmarks5, *, native_face_size_px=None):
        from roop_3d.enhancers.ultramax import laplacian_pyramid_fusion
        small = self._small(native_face_size_px)
        if small and self.small_face_mode == "bypass":
            self._bypassed(native_face_size_px)
            return image.copy()
        began = time.perf_counter()
        try:
            restored = super().restore_aligned(image, landmarks5)
            fused = laplacian_pyramid_fusion(np.ascontiguousarray(image), restored, levels=3)
            strength = 1.0
            if small and self.small_face_mode == "adaptive":
                strength = float(np.clip(native_face_size_px / max(self.min_native_pixels, 1), 0, 1))
                fused = np.clip(image.astype(np.float32) + (fused.astype(np.float32) - image) * strength,
                                0, 255).round().astype(np.uint8)
        except Exception as error:
            self.telemetry.record_failed(error)
            raise
        self._applied(began, strength=round(strength, 3))
        return fused

    def restore_hires(self, image, landmarks5=None, *, native_face_size_px=None, **_unused):
        """UltraMax at native resolution: CodeFormer at 512, luma-only recolour, then 3-level Laplacian fusion with
        the (cubic-enlarged) swap so colour and low/mid bands stay the swap's and the finest bands are the restorer's.
        Returns ``None`` only when the user explicitly opted into skipping this (small) face."""
        from roop_3d.enhancers.ultramax import laplacian_pyramid_fusion
        small = self._small(native_face_size_px)
        if small and self.small_face_mode == "bypass":
            self._bypassed(native_face_size_px)
            return None
        began = time.perf_counter()
        try:
            restored = super().restore_hires(image, landmarks5)
            base = (image if image.shape[:2] == restored.shape[:2]
                    else cv2.resize(image, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_CUBIC))
            # S = restorer / swap resolution; the log2(S) finest bands lie above the swap's own Nyquist limit.
            detail = max(1, int(round(np.log2(max(restored.shape[0] / image.shape[0], 1.0)))))
            fused = laplacian_pyramid_fusion(np.ascontiguousarray(base), restored, levels=max(3, detail + 1),
                                             detail_levels=detail)
            strength = 1.0
            if small and self.small_face_mode == "adaptive":
                strength = float(np.clip(native_face_size_px / max(self.min_native_pixels, 1), 0, 1))
                fused = np.clip(base.astype(np.float32) + (fused.astype(np.float32) - base) * strength,
                                0, 255).round().astype(np.uint8)
        except Exception as error:
            self.telemetry.record_failed(error)
            raise
        self._applied(began, strength=round(strength, 3))
        return fused

    def restore(self, image: np.ndarray, landmarks5: np.ndarray | None = None) -> np.ndarray:
        from roop_3d.enhancers.ultramax import laplacian_pyramid_fusion
        if landmarks5 is not None:
            return self.restore_aligned(image, landmarks5)
        began = time.perf_counter()
        try:
            restored = super().restore(image)
            if restored.shape != image.shape:
                restored = cv2.resize(restored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            fused = laplacian_pyramid_fusion(np.ascontiguousarray(image), restored, levels=3)
        except Exception as error:
            self.telemetry.record_failed(error)
            raise
        self._applied(began, strength=1.0)
        return fused

    def capability(self):
        model = getattr(self.runner, "model", None)
        record = getattr(model, "record", None)
        observed = {"model_sha256": record.get("sha256"), "provider": getattr(model, "provider", None)} \
            if isinstance(record, dict) else {}
        return {**super().capability(), 'architecture': 'ultramax', 'model': self.spec.model_name,
                **observed, 'telemetry': self.telemetry.snapshot()}


class GfpganRestorer:
    """Lazy, hash-verified GFPGAN v1.4: facexlib FFHQ-512 alignment, RGB, [-1, 1]; no fidelity input (blend only)."""
    def __init__(self, catalog, config):
        from roop_3d.compositor.stream_restoration import OnnxFaceRestorer, gfpgan_spec
        self.spec = gfpgan_spec()
        self.runner = OnnxFaceRestorer(catalog, config, self.spec)
        self.inference_verified = False

    def restore_hires(self, image, landmarks5=None, **_unused):
        from roop_3d.compositor.stream_restoration import restore_in_crop_space
        points = np.asarray(landmarks5 if landmarks5 is not None else (), np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise DomainError('RESTORER_LANDMARKS_REQUIRED', 'GFPGAN requires five crop-space landmarks', 422)
        result, _ = restore_in_crop_space(self.runner.restore, image, points, self.spec)
        self.inference_verified = True
        return result

    def restore_aligned(self, image, landmarks5):
        """Crop-sized result (legacy callers); the hi-res path uses ``restore_hires``."""
        restored = self.restore_hires(image, landmarks5)
        return cv2.resize(restored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    def capability(self):
        return {'architecture': 'gfpgan', 'alignment': 'FFHQ_512_FACEXLIB_5PTS', 'preprocessing': 'RGB_MINUS_ONE_ONE',
                'inference': 'VERIFIED_THIS_SESSION' if self.inference_verified else 'NOT_RUN',
                'weights': 'VERIFIED_LOCAL' if self.runner.model is not None else 'NOT_LOADED'}

    def release(self):
        self.runner.close()
        self.inference_verified = False
