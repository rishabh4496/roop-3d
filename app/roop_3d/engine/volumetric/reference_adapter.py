"""NumPy/ONNX LivePortrait reference path: relative motion, stitching and 3D warp.

The verified warp graph includes the SPADE decoder. No torch or TensorRT is
required. Features and baseline motion are bounded per active identity track.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, image_bgr, record_named, tensor
from roop_3d.models.catalog import ModelCatalog

CORE_MODELS = {
    "appearance": "liveportrait_appearance_feature_extractor.onnx",
    "motion": "liveportrait_motion_extractor.onnx",
    "warp_spade": "liveportrait_warping_spade.onnx",
    "stitch": "liveportrait_stitching.onnx",
}


def portrait_tensor(image: np.ndarray) -> np.ndarray:
    image_bgr(image)
    return np.ascontiguousarray(cv2.resize(image, (256, 256))[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def angle(logits: np.ndarray) -> float:
    tensor(logits, (1, 66))
    scores = np.exp(logits.astype(np.float64) - logits.max())
    return float((scores / scores.sum() * np.arange(66)).sum() * 3 - 97.5)


def rotation(angles: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.deg2rad(angles)
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    around_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    around_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    around_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return (around_z @ around_y @ around_x).T.astype(np.float32)


@dataclass(frozen=True)
class Motion:
    canonical: np.ndarray
    expression: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    angles: tuple[float, float, float]

    def points(self) -> np.ndarray:
        transformed = (self.canonical @ rotation(self.angles) + self.expression) * self.scale.reshape(1, 1, 1)
        transformed[:, :, :2] += self.translation[:, None, :2]
        return tensor(transformed.astype(np.float32), (1, 21, 3))


def expression_ratios(landmarks203: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(landmarks203, np.float32)
    tensor(points, (203, 2))
    def distance_ratio(a: int, b: int, c: int, d: int) -> float:
        return float(np.linalg.norm(points[a] - points[b]) / (np.linalg.norm(points[c] - points[d]) + 1e-6))
    eyes = np.array([[distance_ratio(6, 18, 0, 12), distance_ratio(30, 42, 24, 36)]], np.float32)
    lips = np.array([[distance_ratio(90, 102, 48, 66)]], np.float32)
    return eyes, lips


@dataclass(frozen=True)
class TrackReference:
    feature: np.ndarray
    source: Motion
    driving_baseline: Motion
    source_points: np.ndarray
    source_eyes: np.ndarray
    source_lips: np.ndarray
    baseline_eyes: np.ndarray
    baseline_lips: np.ndarray


class LivePortraitAdapter:
    def __init__(self, catalog: ModelCatalog | None = None, config: RuntimeConfig = RuntimeConfig(), *, max_tracks: int = 32):
        if not 1 <= max_tracks <= 32:
            raise ValueError("Invalid portrait track budget")
        self.catalog, self.config, self.max_tracks = catalog or ModelCatalog(), config, max_tracks
        self.models: dict[str, OrtModel] = {}
        self.tracks: dict[int, TrackReference] = {}
        self.available = False
        self.fallback_reason: str | None = None

    def prepare_session(self, *, required: bool = False) -> bool:
        self.release_session()
        try:
            for name in CORE_MODELS.values():
                record_named(self.catalog, name)
            for key, name in CORE_MODELS.items():
                self.models[key] = OrtModel(self.catalog, name, self.config)
            for key, filename in (("eye", "liveportrait_stitching_eye.onnx"), ("lip", "liveportrait_stitching_lip.onnx")):
                if any(record["name"] == filename for record in self.catalog.records()):
                    self.models[key] = OrtModel(self.catalog, filename, self.config)
            self.available = True
            return True
        except DomainError as error:
            self.release_session()
            if error.code != "MODEL_NOT_INSTALLED" or required:
                raise
            self.fallback_reason = "Required LivePortrait weights are not installed; verified identity swap remains available"
            return False

    def motion(self, image: np.ndarray) -> Motion:
        output = self.models["motion"].run({"img": portrait_tensor(image)})
        return Motion(output["kp"].reshape(1, 21, 3), output["exp"].reshape(1, 21, 3), output["t"],
                      output["scale"], tuple(angle(output[key]) for key in ("pitch", "yaw", "roll")))

    def prepare_track(self, track_id: int, swapped_portrait: np.ndarray, target_portrait: np.ndarray,
                      source_landmarks203: np.ndarray, target_landmarks203: np.ndarray) -> None:
        if not self.available:
            raise DomainError("LIVEPORTRAIT_UNAVAILABLE", "LivePortrait is not prepared", 409)
        if track_id not in self.tracks and len(self.tracks) >= self.max_tracks:
            raise DomainError("PORTRAIT_TRACK_BUDGET", "Portrait feature cache is full", 422)
        source, baseline = self.motion(swapped_portrait), self.motion(target_portrait)
        feature = self.models["appearance"].run({"img": portrait_tensor(swapped_portrait)})["output"]
        tensor(feature, (1, 32, 16, 64, 64))
        source_eyes, source_lips = expression_ratios(source_landmarks203)
        baseline_eyes, baseline_lips = expression_ratios(target_landmarks203)
        self.tracks[track_id] = TrackReference(feature, source, baseline, source.points(), source_eyes, source_lips,
                                               baseline_eyes, baseline_lips)

    def _ratio_delta(self, key: str, reference: TrackReference, ratio: np.ndarray) -> np.ndarray:
        source_ratios = reference.source_eyes if key == "eye" else reference.source_lips
        target_ratio = np.asarray([[ratio.mean()]], np.float32)
        features = np.concatenate((reference.source_points.reshape(1, 63), source_ratios, target_ratio), axis=1)
        return self.models[key].run({"input": features})["output"].reshape(1, 21, 3)

    def infer_frame(self, track_id: int, target_portrait: np.ndarray, target_landmarks203: np.ndarray, *,
                    strength: float = 0.35) -> np.ndarray:
        if not self.available or track_id not in self.tracks:
            raise DomainError("PORTRAIT_TRACK_MISSING", "Prepare a source reference for this track", 409)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Retargeting strength must be within 0..1")
        reference, driving = self.tracks[track_id], self.motion(target_portrait)
        source, baseline = reference.source, reference.driving_baseline
        new_angles = tuple(s + strength * (d - b) for s, d, b in zip(source.angles, driving.angles, baseline.angles))
        ratio = driving.scale / np.maximum(baseline.scale, 1e-6)
        moved = Motion(source.canonical,
                       source.expression + strength * (driving.expression - baseline.expression),
                       source.translation + strength * (driving.translation - baseline.translation),
                       source.scale * (1 + strength * (ratio - 1)), new_angles)
        points = moved.points()
        eyes, lips = expression_ratios(target_landmarks203)
        for key, current, initial in (("eye", eyes, reference.baseline_eyes), ("lip", lips, reference.baseline_lips)):
            if key in self.models:
                points += strength * (self._ratio_delta(key, reference, current) - self._ratio_delta(key, reference, initial))
        correction = self.models["stitch"].run({"input": np.concatenate((reference.source_points.reshape(1, 63),
                                                                        points.reshape(1, 63)), axis=1)})["output"]
        tensor(correction, (1, 65))
        points += correction[:, :63].reshape(1, 21, 3)
        points[:, :, :2] += correction[:, 63:].reshape(1, 1, 2)
        output = self.models["warp_spade"].run({"feature_3d": reference.feature,
                                               "kp_driving": np.ascontiguousarray(points),
                                               "kp_source": reference.source_points})["out"]
        tensor(output, (1, 3, 512, 512))
        return np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)

    def retain_tracks(self, active_ids: frozenset[int]) -> None:
        for track_id in set(self.tracks) - active_ids:
            del self.tracks[track_id]

    def capability(self) -> dict:
        return {"liveportrait": "VERIFIED_LOCAL" if self.available and all(m.executions for m in self.models.values()) else "UNKNOWN",
                "available": self.available, "fallback_reason": self.fallback_reason,
                "models": {name: model.capability() for name, model in self.models.items()},
                "learned_eye_retargeting": "VERIFIED_LOCAL" if "eye" in self.models and self.models["eye"].executions else "UNKNOWN",
                "learned_lip_retargeting": "VERIFIED_LOCAL" if "lip" in self.models and self.models["lip"].executions else "UNKNOWN",
                "lip_note": "Mouth motion is driven by motion-extractor expression deltas; a separate lip network is used only when verified weights exist",
                "cached_tracks": len(self.tracks), "max_tracks": self.max_tracks}

    def release_session(self) -> None:
        self.tracks.clear()
        for model in self.models.values():
            model.release()
        self.models.clear()
        self.available = False
        self.fallback_reason = None
