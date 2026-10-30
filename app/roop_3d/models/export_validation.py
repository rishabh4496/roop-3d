"""Deterministic numerical gates; no candidate becomes active on shape alone."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import math

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class Tolerance:
    max_absolute: float = 0.001
    mean_absolute: float = 0.00005
    embedding_cosine: float | None = None
    image_psnr_db: float | None = None
    image_ssim: float | None = None
    border_absolute: float | None = None

    def __post_init__(self) -> None:
        values = [v for v in asdict(self).values() if v is not None]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Parity tolerances must be finite and nonnegative")
        if self.embedding_cosine is not None and not 0.995 <= self.embedding_cosine <= 1:
            raise ValueError("Embedding cosine gate must be at least 0.995")
        if self.image_ssim is not None and self.image_ssim > 1:
            raise ValueError("SSIM threshold must not exceed one")


@dataclass(frozen=True)
class OutputMetric:
    name: str
    max_absolute: float
    mean_absolute: float
    cosine_minimum: float | None
    psnr_db: float | None
    psnr_infinite: bool
    ssim: float | None
    border_absolute: float | None
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ParityReport:
    passed: bool
    outputs: tuple[OutputMetric, ...]
    tolerance: Tolerance

    def require(self) -> "ParityReport":
        if not self.passed:
            raise DomainError("EXPORT_PARITY_FAILED", "Candidate failed the fixed numerical parity gates", 422)
        return self


def fixture_digest(cases: tuple[Mapping[str, np.ndarray], ...]) -> str:
    if not cases:
        raise ValueError("Golden fixtures must not be empty")
    digest = hashlib.sha256()
    for index, case in enumerate(cases):
        digest.update(str(index).encode())
        for name, value in sorted(case.items()):
            if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
                raise ValueError("Golden inputs must be finite NumPy arrays")
            digest.update(json.dumps([name, value.dtype.str, value.shape], separators=(",", ":")).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _images(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 3 or min(value.shape[2:]) < 11:
        raise DomainError("PARITY_IMAGE_LAYOUT", "Image gates require NCHW RGB with dimensions >= 11", 422)
    return value.transpose(0, 2, 3, 1).astype(np.float64)


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """11x11 Gaussian-window SSIM, unit data range, averaged over valid pixels/channels."""
    scores = []
    for a, b in zip(_images(left), _images(right)):
        mean_a, mean_b = (cv2.GaussianBlur(x, (11, 11), 1.5) for x in (a, b))
        variance_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mean_a * mean_a
        variance_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mean_b * mean_b
        covariance = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mean_a * mean_b
        numerator = (2 * mean_a * mean_b + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
        denominator = (mean_a ** 2 + mean_b ** 2 + 0.01 ** 2) * (variance_a + variance_b + 0.03 ** 2)
        scores.append(float(np.mean((numerator / denominator)[5:-5, 5:-5])))
    return float(np.mean(scores))


def compare_outputs(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray],
                    tolerance: Tolerance) -> ParityReport:
    if not reference or set(reference) != set(candidate):
        raise DomainError("PARITY_OUTPUT_NAMES", "Reference/candidate output names differ", 422)
    metrics = []
    for name, expected in reference.items():
        actual = candidate[name]
        if (not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray)
                or expected.shape != actual.shape or expected.dtype != actual.dtype or expected.size == 0
                or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
            raise DomainError("PARITY_OUTPUT_CONTRACT", "Output shape, dtype or finite-value contract failed", 422)
        delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        maximum, mean = float(delta.max()), float(delta.mean())
        failures = []
        if maximum > tolerance.max_absolute:
            failures.append("max_absolute")
        if mean > tolerance.mean_absolute:
            failures.append("mean_absolute")
        cosine = psnr = ssim = border = None
        infinite = False
        if tolerance.embedding_cosine is not None:
            if expected.shape[-1] != 512:
                raise DomainError("PARITY_EMBEDDING_LAYOUT", "Embedding parity requires a 512-D final axis", 422)
            a, b = expected.reshape(-1, 512).astype(np.float64), actual.reshape(-1, 512).astype(np.float64)
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            if np.any(norms < 1e-12):
                raise DomainError("PARITY_EMBEDDING_ZERO", "Zero embeddings cannot establish cosine parity", 422)
            cosine = float(np.clip(np.sum(a * b, axis=1) / norms, -1, 1).min())
            if cosine < tolerance.embedding_cosine:
                failures.append("embedding_cosine")
        if tolerance.image_psnr_db is not None or tolerance.image_ssim is not None or tolerance.border_absolute is not None:
            _images(expected)
            mse = float(np.mean(delta * delta))
            infinite = mse == 0
            psnr = None if infinite else -10 * math.log10(mse)
            ssim = structural_similarity(expected, actual)
            border = float(max(delta[:, :, :4].max(), delta[:, :, -4:].max(),
                               delta[:, :, :, :4].max(), delta[:, :, :, -4:].max()))
            if tolerance.image_psnr_db is not None and not infinite and psnr < tolerance.image_psnr_db:
                failures.append("image_psnr")
            if tolerance.image_ssim is not None and ssim < tolerance.image_ssim:
                failures.append("image_ssim")
            if tolerance.border_absolute is not None and border > tolerance.border_absolute:
                failures.append("border_absolute")
        metrics.append(OutputMetric(name, maximum, mean, cosine, psnr, infinite, ssim, border, tuple(failures)))
    return ParityReport(all(not item.failures for item in metrics), tuple(metrics), tolerance)
