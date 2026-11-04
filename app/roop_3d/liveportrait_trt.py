"""Modular LivePortrait retargeting using explicitly prepared checked sessions.

The reference adapter remains intact. Generator graph lowering is an offline
step in tools/convert_dynamic_shapes.py --grid-sample-4d. Eye-open retargeting
and independently calibrated eyeball gaze are separate capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from roop_3d.engine.volumetric.reference_adapter import Motion, angle
from roop_3d.flame_3d_engine import EyeGaze, finite_array, rgb_tensor


class GazeRetargeter(Protocol):
    def delta(self, source: EyeGaze, target: EyeGaze, points: np.ndarray) -> np.ndarray: ...


class OnnxGazeRetargeter:
    """Optional calibrated export, not a reinterpretation of eye-open weights.

    Declared export inputs: source_gaze/target_gaze [1,4] in radians and
    points [1,21,3]; output delta [1,21,3] in LivePortrait canonical units.
    A caller must provide an inspected, hash-checked session and calibration evidence.
    """
    def __init__(self, session, *, calibration_evidence: str):
        if not calibration_evidence.strip():
            raise ValueError("Gaze retargeting requires calibration evidence")
        self.session = session

    def delta(self, source, target, points):
        output = self.session.run({"source_gaze": source.angles()[None], "target_gaze": target.angles()[None],
                                   "points": np.ascontiguousarray(points, dtype=np.float32)})
        return finite_array(output["delta"], (1, 21, 3), "calibrated gaze delta")


@dataclass(frozen=True)
class LivePortraitNames:
    """Defaults match the preserved export's observed graph metadata."""
    image: str = "img"
    appearance_output: str = "output"
    feature: str = "feature_3d"
    source_points: str = "kp_source"
    driving_points: str = "kp_driving"
    generated: str = "out"
    retarget_input: str = "input"
    retarget_output: str = "output"


class LivePortraitTRT:
    def __init__(self, appearance, motion, generator, stitching, eye=None, *,
                 names=LivePortraitNames(), gaze: GazeRetargeter | None = None):
        self.models = {"appearance": appearance, "motion": motion, "generator": generator,
                       "stitching": stitching}
        if eye is not None:
            self.models["eye"] = eye
        self.names, self.gaze = names, gaze

    @classmethod
    def from_registry(cls, *, provider="CPUExecutionProvider", profile="6gb", catalog=None,
                      gaze: GazeRetargeter | None = None):
        """Explicit opt-in preparation of the preserved modular export family."""
        from roop_3d.models.catalog import ModelCatalog
        from roop_3d.models.stage16 import CheckedSession, prepare_generator
        catalog = catalog or ModelCatalog()
        generator, digest, size = prepare_generator(catalog)
        sessions = {}
        for role, filename in (("appearance", "liveportrait_appearance_feature_extractor.onnx"),
                               ("motion", "liveportrait_motion_extractor.onnx"),
                               ("stitching", "liveportrait_stitching.onnx"),
                               ("eye", "liveportrait_stitching_eye.onnx")):
            record = next((r for r in catalog.registry.records() if r.name == filename and r.locator), None)
            if record is None:
                raise RuntimeError(f"The inspected modular LivePortrait export is unavailable: {filename}")
            path = catalog.resolve_for_inference(record.model_id, precision="FP32")
            sessions[role] = CheckedSession(path, record.expected_sha256, record.byte_size,
                                            provider=provider, profile=profile)
        sessions["generator"] = CheckedSession(generator, digest, size, provider=provider, profile=profile)
        return cls(**sessions, gaze=gaze)

    def extract_motion(self, rgb_crop):
        values = self.models["motion"].run({self.names.image: rgb_tensor(rgb_crop, 256)})
        angles = []
        for name in ("pitch", "yaw", "roll"):
            value = np.asarray(values[name])
            if value.shape == (1, 66):
                angles.append(angle(value))
            elif value.shape == (1, 1) and np.isfinite(value).all():
                angles.append(float(value[0, 0]))
            else:
                raise ValueError("Motion angle must match an inspected logits or degree output")
        canonical = finite_array(values["kp"].reshape(1, 21, 3), (1, 21, 3), "canonical motion")
        expression = finite_array(values["exp"].reshape(1, 21, 3), (1, 21, 3), "expression motion")
        translation = finite_array(values["t"], (1, 3), "portrait translation")
        scale = finite_array(values["scale"], (1, 1), "portrait scale")
        if scale[0, 0] <= 0:
            raise ValueError("LivePortrait scale must be positive")
        return Motion(canonical, expression, translation, scale, tuple(angles))

    def eye_delta(self, points, source_ratios, target_ratio):
        if "eye" not in self.models:
            raise RuntimeError("Eye-open retargeter has not been prepared")
        source = finite_array(source_ratios, (2,), "source left/right eye-open ratios")
        if not np.isfinite(target_ratio) or target_ratio < 0 or np.any(source < 0):
            raise ValueError("Eye-open distance ratios must be finite and nonnegative")
        features = np.concatenate((points.reshape(1, 63), source[None], [[target_ratio]]), axis=1).astype(np.float32)
        output = self.models["eye"].run({self.names.retarget_input: features})[self.names.retarget_output]
        return finite_array(output.reshape(1, 21, 3), (1, 21, 3), "eye-open delta")

    def retarget(self, source_rgb, driving_rgb, *, strength=1.0, baseline_rgb=None,
                 source_eye_ratios=None, driving_eye_ratio=None,
                 source_gaze: EyeGaze | None = None, driving_gaze: EyeGaze | None = None):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Retarget strength must be in [0,1]")
        if (source_gaze is None) != (driving_gaze is None):
            raise ValueError("Supply both source and driving gaze")
        if driving_gaze is not None and self.gaze is None:
            raise RuntimeError("Independent eyeball transfer requires a calibrated gaze retargeter")
        if (source_eye_ratios is None) != (driving_eye_ratio is None):
            raise ValueError("Supply both source and driving eye-open ratios")
        source, driving = self.extract_motion(source_rgb), self.extract_motion(driving_rgb)
        baseline = self.extract_motion(baseline_rgb) if baseline_rgb is not None else source
        points_source = source.points()
        moved = Motion(source.canonical,
                       source.expression + strength * (driving.expression - baseline.expression),
                       source.translation + strength * (driving.translation - baseline.translation),
                       source.scale * (1 + strength * (driving.scale / baseline.scale - 1)),
                       tuple(s + strength * (d-b) for s, d, b in zip(source.angles, driving.angles, baseline.angles)))
        points = moved.points()
        if source_eye_ratios is not None:
            # Remove the reference delta so zero strength leaves the eye ratio unchanged.
            points += strength * (self.eye_delta(points_source, source_eye_ratios, driving_eye_ratio)
                                  - self.eye_delta(points_source, source_eye_ratios, float(np.mean(source_eye_ratios))))
        correction = self.models["stitching"].run({
            self.names.retarget_input: np.concatenate((points_source.reshape(1, 63), points.reshape(1, 63)), axis=1)
        })[self.names.retarget_output]
        correction = finite_array(correction, (1, 65), "stitching correction")
        points += strength * correction[:, :63].reshape(1, 21, 3)
        points[:, :, :2] += strength * correction[:, 63:].reshape(1, 1, 2)
        if driving_gaze is not None:
            points += strength * self.gaze.delta(source_gaze, driving_gaze, points)
        feature = self.models["appearance"].run({self.names.image: rgb_tensor(source_rgb, 256)})[self.names.appearance_output]
        feature = finite_array(feature, (1, 32, 16, 64, 64), "appearance feature volume")
        output = self.models["generator"].run({self.names.feature: feature,
                                               self.names.source_points: points_source,
                                               self.names.driving_points: np.ascontiguousarray(points)})[self.names.generated]
        output = finite_array(output, (1, 3, 512, 512), "generated portrait")
        return np.rint(np.clip(output[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

    def finish_evidence(self):
        return {role: session.finish_evidence() for role, session in self.models.items() if session.runs}
