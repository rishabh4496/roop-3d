# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Clean-room ARKit 52 <-> 3DMM expression bridge.

The coefficient names and ordering are the public ARKit blend-shape contract.
No third-party mapper, weights, or training data are included. A new bridge uses
an identity seed for the first 52 expression dimensions; applications with
paired local calibration samples can call :meth:`fit` to learn linear and
quadratic residual terms in both directions.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


ARKIT52_NAMES: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
)
ARKIT_INDEX = {name: index for index, name in enumerate(ARKIT52_NAMES)}
EXPRESSION_DIM = 100
JAW_DIM = 3
EYES_DIM = 6
_EYE_ANGLE_DIM = 4
_QUADRATIC_INDEX = {size: np.triu_indices(size) for size in (len(ARKIT52_NAMES), EXPRESSION_DIM)}


def _finite_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return np.ascontiguousarray(array)


def _quadratic_features(value: np.ndarray) -> np.ndarray:
    """All unique second-order products, including squares, in stable order."""
    rows = np.asarray(value, np.float32)
    i, j = _QUADRATIC_INDEX[rows.shape[-1]]
    return rows[..., i] * rows[..., j]


def _ridge_map(source: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit affine linear plus quadratic residual maps without a heavyweight runtime dependency."""
    x = np.asarray(source, np.float64)
    y = np.asarray(target, np.float64)
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mean_x, y - mean_y
    gram = xc.T @ xc
    regularizer = max(float(ridge), 0.0) * (float(np.trace(gram)) / max(len(gram), 1) + 1e-12)
    linear = np.linalg.solve(gram + regularizer * np.eye(len(gram)), xc.T @ yc).T
    residual = yc - xc @ linear.T
    q = _quadratic_features(x).astype(np.float64)
    q_mean = q.mean(axis=0)
    qc = q - q_mean
    # The interaction space has 5,050 columns on the reverse direction. Use
    # the sample-space dual solve so calibration remains inexpensive even when
    # a user supplies only a few hundred paired frames.
    qgram = qc @ qc.T
    qreg = max(float(ridge), 0.0) * (float(np.trace(qgram)) / max(len(qgram), 1) + 1e-12)
    correction = (qc.T @ np.linalg.solve(qgram + qreg * np.eye(len(qgram)), residual)).T
    bias = mean_y - linear @ mean_x - correction @ q_mean
    return (linear.astype(np.float32), bias.astype(np.float32), correction.astype(np.float32))


@dataclass(frozen=True)
class ARKitBridgeOutput:
    psi: np.ndarray                 # (..., 100)
    theta_jaw: np.ndarray           # (..., 3), axis-angle radians
    theta_eyes: np.ndarray          # (..., 6), axis-angle + translation contract
    gaze_angles: np.ndarray         # (..., 4), left yaw/pitch, right yaw/pitch radians


class ARKit52To3DMMBridge:
    """Bidirectional, calibration-ready ARKit and 3DMM coefficient mapper.

    ``to_3dmm`` is linear + quadratic-residual and emits ``psi (100)``,
    ``theta_jaw (3)`` and ``theta_eyes (6)``. The eye output follows the
    articulated-head contract (axis-angle plus translation); ``gaze_angles``
    also retains separate left/right yaw and pitch for the video tracker.

    Coefficients are kept in contiguous float32 arrays for low-overhead frame
    inference. The default is a transparent ARKit-ordered seed, not a claim
    that arbitrary model bases share ARKit semantics. Paired calibration data
    should replace it where the target 3DMM has a different basis ordering.
    """

    target_latency_ms = 0.4

    def __init__(self, *, forward: np.ndarray | None = None, forward_bias: np.ndarray | None = None,
                 forward_residual: np.ndarray | None = None, reverse: np.ndarray | None = None,
                 reverse_bias: np.ndarray | None = None, reverse_residual: np.ndarray | None = None,
                 jaw_map: np.ndarray | None = None, eye_map: np.ndarray | None = None):
        f = np.zeros((EXPRESSION_DIM, len(ARKIT52_NAMES)), np.float32)
        f[:len(ARKIT52_NAMES), :] = np.eye(len(ARKIT52_NAMES), dtype=np.float32)
        r = np.zeros((len(ARKIT52_NAMES), EXPRESSION_DIM), np.float32)
        r[:, :len(ARKIT52_NAMES)] = np.eye(len(ARKIT52_NAMES), dtype=np.float32)
        nf = len(ARKIT52_NAMES) * (len(ARKIT52_NAMES) + 1) // 2
        nr = EXPRESSION_DIM * (EXPRESSION_DIM + 1) // 2
        self.forward = _finite_matrix(f if forward is None else forward, f.shape, "forward")
        self.forward_bias = self._finite_vector(np.zeros(EXPRESSION_DIM) if forward_bias is None else forward_bias,
                                                EXPRESSION_DIM, "forward_bias")
        self.forward_residual = _finite_matrix(np.zeros((EXPRESSION_DIM, nf)) if forward_residual is None else forward_residual,
                                               (EXPRESSION_DIM, nf), "forward_residual")
        self.reverse = _finite_matrix(r if reverse is None else reverse, r.shape, "reverse")
        self.reverse_bias = self._finite_vector(np.zeros(len(ARKIT52_NAMES)) if reverse_bias is None else reverse_bias,
                                                len(ARKIT52_NAMES), "reverse_bias")
        self.reverse_residual = _finite_matrix(np.zeros((len(ARKIT52_NAMES), nr)) if reverse_residual is None else reverse_residual,
                                               (len(ARKIT52_NAMES), nr), "reverse_residual")
        default_jaw = np.zeros((JAW_DIM, len(ARKIT52_NAMES)), np.float32)
        default_jaw[0, ARKIT_INDEX["jawOpen"]] = 0.60
        default_jaw[1, ARKIT_INDEX["jawLeft"]] = -0.15
        default_jaw[1, ARKIT_INDEX["jawRight"]] = 0.15
        default_jaw[2, ARKIT_INDEX["jawForward"]] = 0.10
        self.jaw_map = _finite_matrix(default_jaw if jaw_map is None else jaw_map, (JAW_DIM, len(ARKIT52_NAMES)), "jaw_map")
        self.eye_map = _finite_matrix(np.zeros((EYES_DIM, len(ARKIT52_NAMES))) if eye_map is None else eye_map,
                                      (EYES_DIM, len(ARKIT52_NAMES)), "eye_map")
        self._has_forward_residual = bool(np.any(self.forward_residual))
        self._has_reverse_residual = bool(np.any(self.reverse_residual))

    @staticmethod
    def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
        array = np.asarray(value, np.float32)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{label} must be finite with shape ({size},)")
        return np.ascontiguousarray(array)

    @staticmethod
    def _input(value: Any, size: int, label: str, *, clip: tuple[float, float] | None = None) -> tuple[np.ndarray, bool]:
        array = np.asarray(value, np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if array.ndim != 2 or array.shape[-1] != size or not np.isfinite(array).all():
            raise ValueError(f"{label} must be finite with shape (..., {size})")
        if clip is not None:
            array = np.clip(array, *clip)
        return np.ascontiguousarray(array), single

    @classmethod
    def fit(cls, arkit_samples: np.ndarray, psi_samples: np.ndarray, *, jaw_samples: np.ndarray | None = None,
            eye_samples: np.ndarray | None = None, ridge: float = 1e-4) -> "ARKit52To3DMMBridge":
        """Fit both directions from aligned local calibration arrays.

        ``arkit_samples`` is ``(N,52)`` in [0,1], ``psi_samples`` is ``(N,100)``.
        Optional joint arrays are ``(N,3)`` and ``(N,6)``. Samples are caller
        supplied; this project bundles no licensed or proprietary calibration set.
        """
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        x, _ = cls._input(arkit_samples, 52, "arkit_samples", clip=(0.0, 1.0))
        y, _ = cls._input(psi_samples, EXPRESSION_DIM, "psi_samples")
        if len(x) < 3 or len(x) != len(y):
            raise ValueError("calibration requires at least three aligned ARKit/psi samples")
        forward, fbias, fres = _ridge_map(x, y, ridge)
        reverse, rbias, rres = _ridge_map(y, x, ridge)
        jaw_map = None
        eye_map = None
        if jaw_samples is not None:
            jaw, _ = cls._input(jaw_samples, JAW_DIM, "jaw_samples")
            if len(jaw) != len(x):
                raise ValueError("jaw_samples must match calibration rows")
            jaw_map = np.linalg.lstsq(x.astype(np.float64), jaw.astype(np.float64), rcond=None)[0].T.astype(np.float32)
        if eye_samples is not None:
            eyes, _ = cls._input(eye_samples, EYES_DIM, "eye_samples")
            if len(eyes) != len(x):
                raise ValueError("eye_samples must match calibration rows")
            eye_map = np.linalg.lstsq(x.astype(np.float64), eyes.astype(np.float64), rcond=None)[0].T.astype(np.float32)
        return cls(forward=forward, forward_bias=fbias, forward_residual=fres, reverse=reverse,
                   reverse_bias=rbias, reverse_residual=rres, jaw_map=jaw_map, eye_map=eye_map)

    def to_3dmm(self, arkit52: Any) -> ARKitBridgeOutput:
        x, single = self._input(arkit52, len(ARKIT52_NAMES), "arkit52", clip=(0.0, 1.0))
        psi = x @ self.forward.T + self.forward_bias
        if self._has_forward_residual:
            psi += _quadratic_features(x) @ self.forward_residual.T
        jaw = x @ self.jaw_map.T
        # Directional controls are made explicit and stable even when no eye calibration was fitted.
        def eye_axis(side: str) -> tuple[np.ndarray, np.ndarray]:
            if side == "Left":
                up, down = "eyeLookUpLeft", "eyeLookDownLeft"
                inward, outward = "eyeLookInLeft", "eyeLookOutLeft"
            else:
                up, down = "eyeLookUpRight", "eyeLookDownRight"
                inward, outward = "eyeLookInRight", "eyeLookOutRight"
            yaw = x[:, ARKIT_INDEX[outward]] - x[:, ARKIT_INDEX[inward]]
            pitch = x[:, ARKIT_INDEX[up]] - x[:, ARKIT_INDEX[down]]
            return yaw * np.float32(0.35), pitch * np.float32(0.30)
        ly, lp = eye_axis("Left")
        ry, rp = eye_axis("Right")
        gaze = np.column_stack((ly, lp, ry, rp)).astype(np.float32)
        calibrated_eye = x @ self.eye_map.T
        # First 3 dims are shared eyeball rotation, last 3 are per-eye differential offsets.
        eyes = calibrated_eye.copy()
        eyes[:, 0] += 0.5 * (lp + rp)
        eyes[:, 1] += 0.5 * (ly + ry)
        eyes[:, 3] += 0.5 * (ly - ry)
        eyes[:, 4] += 0.5 * (lp - rp)
        if single:
            return ARKitBridgeOutput(psi[0], jaw[0], eyes[0], gaze[0])
        return ARKitBridgeOutput(psi.astype(np.float32), jaw.astype(np.float32), eyes.astype(np.float32), gaze)

    def to_arkit(self, psi: Any, theta_jaw: Any | None = None, theta_eyes: Any | None = None) -> np.ndarray:
        coefficients, single = self._input(psi, EXPRESSION_DIM, "psi")
        result = coefficients @ self.reverse.T + self.reverse_bias
        if self._has_reverse_residual:
            result += _quadratic_features(coefficients) @ self.reverse_residual.T
        # Joint inputs are accepted for symmetry with to_3dmm. Learned models
        # fold their effects into psi; defaults avoid counting them twice.
        if theta_jaw is not None:
            joint, joint_single = self._input(theta_jaw, JAW_DIM, "theta_jaw")
            if len(joint) not in (1, len(result)):
                raise ValueError("theta_jaw batch must match psi")
            if len(joint) == 1 and len(result) > 1:
                joint = np.repeat(joint, len(result), axis=0)
            result += joint @ self.jaw_map
            single = single and joint_single
        if theta_eyes is not None:
            joint, joint_single = self._input(theta_eyes, EYES_DIM, "theta_eyes")
            if len(joint) not in (1, len(result)):
                raise ValueError("theta_eyes batch must match psi")
            if len(joint) == 1 and len(result) > 1:
                joint = np.repeat(joint, len(result), axis=0)
            result += joint @ self.eye_map
            single = single and joint_single
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        return result[0] if single else result

    def regularize(self, psi: Any, arkit52: Any, *, strength: float = 0.35) -> np.ndarray:
        """Blend an existing expression estimate toward the mapped observation."""
        weight = float(strength)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("strength must be within [0, 1]")
        current, single = self._input(psi, EXPRESSION_DIM, "psi")
        mapped = self.to_3dmm(arkit52).psi
        mapped, _ = self._input(mapped, EXPRESSION_DIM, "mapped psi")
        if len(mapped) == 1 and len(current) > 1:
            mapped = np.repeat(mapped, len(current), axis=0)
        if len(mapped) != len(current):
            raise ValueError("ARKit and psi batch sizes must match")
        out = (1 - weight) * current + weight * mapped
        return out[0].astype(np.float32) if single else out.astype(np.float32)

    @staticmethod
    def slider_definitions() -> list[dict[str, Any]]:
        """Stable UI metadata for the 52 standard 0..1 controls."""
        return [{"name": name, "label": re.sub(r"([a-z])([A-Z])", r"\1 \2", name),
                 "minimum": 0.0, "maximum": 1.0, "default": 0.0}
                for name in ARKIT52_NAMES]


__all__ = ["ARKIT52_NAMES", "ARKIT_INDEX", "ARKitBridgeOutput", "ARKit52To3DMMBridge"]
