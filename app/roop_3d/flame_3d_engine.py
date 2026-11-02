"""FLAME coefficient extraction with explicit camera, mesh and gaze evidence.

SMIRK's weak-perspective camera is not metric translation, eyelids are not gaze,
and MediaPipe/3DDFA coefficients are not FLAME coefficients. Unknowns stay None.
Optional encoders/landmarkers are supplied on demand, never at API startup.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np


def finite_array(value, shape, label):
    result = np.array(value, dtype=np.float32, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} requires finite shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EyeGaze:
    """Head-local angles in radians; separate left/right anatomical eyes."""
    left_eye_yaw: float
    left_eye_pitch: float
    right_eye_yaw: float
    right_eye_pitch: float

    def __post_init__(self):
        if not np.isfinite(self.angles()).all() or np.any(np.abs(self.angles()) > np.pi):
            raise ValueError("Gaze angles must be finite radians in [-pi, pi]")

    def angles(self):
        return np.array([self.left_eye_yaw, self.left_eye_pitch,
                         self.right_eye_yaw, self.right_eye_pitch], np.float32)

    def vectors(self):
        yaw, pitch = self.angles().reshape(2, 2).T
        return np.stack((np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                         np.cos(yaw) * np.cos(pitch)), axis=1)


@dataclass(frozen=True)
class FlameParameters:
    beta: np.ndarray
    psi: np.ndarray
    jaw_rotation: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray | None = None
    weak_camera: np.ndarray | None = None
    eyelids: np.ndarray | None = None
    gaze: EyeGaze | None = None

    def __post_init__(self):
        for field, shape in (("beta", (100,)), ("psi", (50,)), ("jaw_rotation", (3,)), ("rotation", (3, 3))):
            object.__setattr__(self, field, finite_array(getattr(self, field), shape, field))
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-5):
            raise ValueError("Head rotation must belong to SO(3)")
        for field, shape in (("translation", (3,)), ("weak_camera", (3,)), ("eyelids", (2,))):
            if getattr(self, field) is not None:
                object.__setattr__(self, field, finite_array(getattr(self, field), shape, field))


def blend_shape(source: FlameParameters, target: FlameParameters, shape_retention_weight: float) -> FlameParameters:
    """0 retains source skull, 1 retains target skull; target motion stays intact."""
    weight = float(shape_retention_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("shape_retention_weight must be in [0, 1]")
    return replace(target, beta=(1-weight) * source.beta + weight * target.beta)


class FlameEncoder(Protocol):
    def extract(self, rgb_crop: np.ndarray) -> FlameParameters: ...


class MeshEstimator(Protocol):
    def extract(self, rgb_crop: np.ndarray) -> np.ndarray: ...


class GazeEstimator(Protocol):
    def estimate(self, rgb_crop: np.ndarray, mesh478: np.ndarray, head: FlameParameters) -> EyeGaze: ...


class OnnxGazeEstimator:
    """Optional calibrated gaze export with an explicit, independently tested contract.

    Inputs rgb [1,3,224,224] in [0,1], mesh478 [1,478,3] in normalized
    crop coordinates, rotation [1,3,3]. Output angles [1,4], head-local radians
    ordered left yaw/pitch, right yaw/pitch. No such weights are bundled.
    """
    def __init__(self, session, *, calibration_evidence: str):
        if not calibration_evidence.strip():
            raise ValueError("Gaze estimation requires calibration evidence")
        self.session = session

    def estimate(self, rgb_crop, mesh478, head):
        result = self.session.run({"rgb": rgb_tensor(rgb_crop, 224),
                                   "mesh478": finite_array(mesh478, (478, 3), "dense mesh")[None],
                                   "rotation": head.rotation[None]})
        angles = finite_array(result["angles"], (1, 4), "independent gaze")[0]
        return EyeGaze(*map(float, angles))


class OnnxFlameEncoder:
    """Named-output SMIRK/EMOCA export adapter with declared axis-angle pose.

    SMIRK default outputs: shape_params (300), expression_params (50),
    jaw_params (3), pose_params (3), cam (3), eyelid_params (2).
    The first 100 ordered FLAME shape components are retained. EMOCA exports
    must explicitly supply their names and RGB crop size; no assumed packed layout.
    """
    def __init__(self, session, *, family="smirk", output_names=None, crop_size=224):
        if family not in {"smirk", "emoca"}:
            raise ValueError("3DDFA/MediaPipe bases cannot be decoded as FLAME")
        if family == "emoca" and output_names is None:
            raise ValueError("EMOCA requires an inspected named-output export contract")
        self.session, self.crop_size = session, crop_size
        self.names = output_names or {"beta": "shape_params", "psi": "expression_params",
                                     "jaw": "jaw_params", "pose": "pose_params",
                                     "camera": "cam", "eyelids": "eyelid_params"}
        if not {"beta", "psi", "jaw", "pose"} <= self.names.keys():
            raise ValueError("The export must supply shape, expression, jaw and pose outputs")

    def extract(self, rgb_crop):
        import cv2
        crop = rgb_tensor(rgb_crop, self.crop_size)
        inputs = self.session.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("FLAME encoder requires a reviewed single RGB input")
        outputs = self.session.run({inputs[0].name: crop})

        def value(key):
            name = self.names.get(key)
            if name is None:
                return None
            arr = np.asarray(outputs[name], np.float32)
            if arr.ndim != 2 or arr.shape[0] != 1:
                raise ValueError(f"FLAME output must be a single face vector: {name}")
            return arr[0]

        beta, pose = value("beta"), value("pose")
        if beta.shape not in {(100,), (300,)}:
            raise ValueError("Export must declare 100 or 300 ordered FLAME shape components")
        pose = finite_array(pose, (3,), "axis-angle pose")
        rotation, _ = cv2.Rodrigues(pose)
        return FlameParameters(beta[:100], value("psi"), value("jaw"), rotation,
                               translation=value("translation"), weak_camera=value("camera"),
                               eyelids=value("eyelids"))


def rgb_tensor(rgb_crop, size):
    import cv2
    image = np.asarray(rgb_crop)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
        raise ValueError("Supply an aligned RGB uint8 face crop")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255


class MediaPipe478Estimator:
    """Optional local FaceLandmarker task. Coordinates are normalized, not metric."""
    def __init__(self, task_path: Path, digest: str, size: int):
        from roop_3d.models.owned_files import model_source
        from roop_3d.models.verifier import verify_file
        self.path = verify_file(model_source(task_path), digest, size)
        self.landmarker = None

    def extract(self, rgb_crop):
        import mediapipe as mp
        # Validate without resizing; mesh coordinates belong to this exact crop.
        rgb_tensor(rgb_crop, 224)
        if self.landmarker is None:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.path)), num_faces=1,
                running_mode=mp.tasks.vision.RunningMode.IMAGE)
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_crop)))
        if len(result.face_landmarks) != 1:
            raise RuntimeError("No single dense face mesh was detected")
        return finite_array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], (478, 3), "MediaPipe mesh")

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


@dataclass(frozen=True)
class Face3D:
    flame: FlameParameters
    mesh478: np.ndarray | None

    def __post_init__(self):
        if self.mesh478 is not None:
            object.__setattr__(self, "mesh478", finite_array(self.mesh478, (478, 3), "dense mesh"))


class Flame3DEngine:
    def __init__(self, encoder: FlameEncoder, mesh: MeshEstimator | None = None,
                 gaze: GazeEstimator | None = None):
        self.encoder, self.mesh, self.gaze = encoder, mesh, gaze

    def extract(self, rgb_crop: np.ndarray, *, require_gaze=False) -> Face3D:
        head = self.encoder.extract(rgb_crop)
        mesh = self.mesh.extract(rgb_crop) if self.mesh is not None else None
        if self.gaze is not None:
            if mesh is None:
                raise RuntimeError("Calibrated gaze extraction requires the dense mesh")
            head = replace(head, gaze=self.gaze.estimate(rgb_crop, mesh, head))
        if require_gaze and head.gaze is None:
            raise RuntimeError("Independent eyeball gaze is unavailable; SMIRK eyelids are not gaze")
        return Face3D(head, mesh)
