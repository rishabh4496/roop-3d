"""Optional ROI restoration, blended only through the face alpha."""
from __future__ import annotations

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
    def __init__(self, catalog, config):
        from roop_3d.compositor.stream_restoration import OnnxFaceRestorer, codeformer_spec
        self.spec = codeformer_spec()
        self.runner = OnnxFaceRestorer(catalog, config, self.spec)
        self.runner.require_dynamic_fidelity = True
        self.fidelity_weight = 0.5
        self.inference_verified = False

    def restore(self, image: np.ndarray, landmarks5: np.ndarray | None = None) -> np.ndarray:
        if landmarks5 is not None:
            return self.restore_aligned(image, landmarks5)
        self.runner.fidelity_weight = self.fidelity_weight
        restored = self.runner.restore(image)
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
        restored = self.runner.restore(crop)
        inverse = cv2.invertAffineTransform(matrix)
        size = (image.shape[1], image.shape[0])
        warped = cv2.warpAffine(restored, inverse, size)
        support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (self.spec.size, self.spec.size))
        alpha = np.clip(cv2.warpAffine(support, inverse, size), 0, 1)[..., None]
        self.inference_verified = True
        return np.clip(warped * alpha + image * (1-alpha), 0, 255).round().astype(np.uint8)

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

    It reuses the verified CodeFormer weights and alignment (no separate model is claimed). Native face size controls
    whether small background faces bypass inference or receive an adaptively reduced detail pass.
    """
    def __init__(self, catalog, config):
        super().__init__(catalog, config)
        self.fidelity_weight = 0.7
        self.small_face_mode = "bypass"
        self.min_native_pixels = 320

    def restore_aligned(self, image, landmarks5, *, native_face_size_px=None):
        from roop_3d.enhancers.ultramax import laplacian_pyramid_fusion
        small = native_face_size_px is not None and native_face_size_px <= self.min_native_pixels
        if small and self.small_face_mode == "bypass":
            return image.copy()
        restored = super().restore_aligned(image, landmarks5)
        fused = laplacian_pyramid_fusion(np.ascontiguousarray(image), restored, levels=3)
        if small and self.small_face_mode == "adaptive":
            strength = float(np.clip(native_face_size_px / max(self.min_native_pixels, 1), 0, 1))
            fused = np.clip(image.astype(np.float32) + (fused.astype(np.float32) - image) * strength,
                            0, 255).round().astype(np.uint8)
        return fused

    def restore(self, image: np.ndarray, landmarks5: np.ndarray | None = None) -> np.ndarray:
        from roop_3d.enhancers.ultramax import laplacian_pyramid_fusion
        if landmarks5 is not None:
            return self.restore_aligned(image, landmarks5)
        restored = super().restore(image)
        if restored.shape != image.shape:
            restored = cv2.resize(restored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return laplacian_pyramid_fusion(np.ascontiguousarray(image), restored, levels=3)

    def capability(self):
        return {**super().capability(), 'architecture': 'ultramax'}
