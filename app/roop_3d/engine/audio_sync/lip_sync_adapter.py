"""Local Wav2Lip ONNX mouth inference; unavailable models preserve the swapped crop."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import image_bgr


@dataclass(frozen=True)
class LipSyncFrameInput:
    mel_window: np.ndarray
    rms: float
    face_confidence: float
    timestamp: Fraction | None = None


@dataclass(frozen=True)
class LipSyncResult:
    image: np.ndarray
    mouth_mask: np.ndarray
    blend: float
    applied: bool
    reason: str | None


@dataclass
class _GateState:
    value: float
    timestamp: Fraction | None


class LipSyncGate:
    """Per-track RMS/VAD and face-confidence gate with a smooth release to neutral."""

    def __init__(self, *, max_tracks: int = 32, silence_rms: float = 0.008,
                 speech_rms: float = 0.04, minimum_face_confidence: float = 0.45,
                 rise_seconds: float = 0.06, decay_seconds: float = 0.14):
        if (max_tracks <= 0 or not 0 <= silence_rms < speech_rms
                or not 0 <= minimum_face_confidence < 1 or rise_seconds <= 0 or decay_seconds <= 0):
            raise ValueError("Invalid lip-sync gate configuration")
        self.max_tracks = int(max_tracks)
        self.silence_rms, self.speech_rms = float(silence_rms), float(speech_rms)
        self.minimum_face_confidence = float(minimum_face_confidence)
        self.rise_seconds, self.decay_seconds = float(rise_seconds), float(decay_seconds)
        self._states: OrderedDict[int, _GateState] = OrderedDict()

    @property
    def track_count(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        self._states.clear()

    def begin_frame(self, *, active_track_ids: set[int] | None = None, scene_cut: bool = False) -> None:
        if scene_cut:
            self.clear()
        elif active_track_ids is not None:
            for track_id in tuple(self._states):
                if track_id not in active_track_ids:
                    self._states.pop(track_id, None)

    def update(self, track_id: int, rms: float, face_confidence: float, *,
               timestamp: Fraction | None = None, scene_cut: bool = False) -> float:
        if type(track_id) is not int or track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        if (not np.isfinite((rms, face_confidence)).all() or rms < 0
                or not 0 <= face_confidence <= 1 or timestamp is not None and not isinstance(timestamp, Fraction)):
            raise DomainError("LIPSYNC_GATE_INPUT_INVALID", "RMS, face confidence or timestamp is invalid", 422)
        if scene_cut:
            self.clear()
        state = self._states.get(track_id)
        dt = 1.0 / 30.0
        if state is not None and timestamp is not None and state.timestamp is not None:
            delta = timestamp - state.timestamp
            if delta <= 0 or delta > 1:
                state = None
                self._states.pop(track_id, None)
            else:
                dt = float(delta)
        elif state is not None and timestamp is not None:
            state.timestamp = timestamp

        energy = float(np.clip((rms - self.silence_rms) / (self.speech_rms - self.silence_rms), 0, 1))
        confidence = float(np.clip((face_confidence - self.minimum_face_confidence)
                                   / (1.0 - self.minimum_face_confidence), 0, 1))
        target = energy * confidence
        previous = state.value if state is not None else 0.0
        tau = self.rise_seconds if target > previous else self.decay_seconds
        smoothing = 1.0 - math.exp(-dt / tau)
        value = float(np.clip(previous + smoothing * (target - previous), 0, 1))
        self._states[track_id] = _GateState(value, timestamp if timestamp is not None else (
            state.timestamp if state is not None else None))
        self._states.move_to_end(track_id)
        while len(self._states) > self.max_tracks:
            self._states.popitem(last=False)
        return value


class LipSyncAdapter:
    """2D lower-face adapter with explicit local ONNX graph contracts."""

    def __init__(self, catalog=None, *, gate: LipSyncGate | None = None):
        from roop_3d.models.catalog import ModelCatalog

        self.catalog = catalog or ModelCatalog()
        self.gate = gate or LipSyncGate()
        records = self.catalog.registry.records()
        self.record = next((item for item in records if item.role == "lip_sync"
                            or "wav2lip" in item.name.casefold()), None)
        self.available = False
        self.reason = "No verified local 2D lip-sync weights and supported tensor contract are registered"

    def prepare(self, mode: str, config) -> None:
        """Activate only a known architecture; TensorRT is selected through RuntimeConfig."""
        self.release()
        self.session = None
        if mode not in ("wav2lip", "wav2lip_gan_fp16"):
            raise DomainError("LIPSYNC_ARCHITECTURE_UNSUPPORTED",
                              "MuseTalk requires Whisper/VAE conditioning; EDTalk requires its pose/emotion encoder bundle", 422)
        from roop_3d.engine.volumetric.swapper_engine import load_local_session
        candidates = [row for row in self.catalog.public()
                      if row.get("installed") and "wav2lip" in row["name"].casefold()
                      and row.get("local_format") == "onnx"
                      and (mode != "wav2lip_gan_fp16" or "gan" in row["name"].casefold())]
        if not candidates:
            raise DomainError("LIPSYNC_MODEL_UNAVAILABLE", "Install a Wav2Lip ONNX export in app/models", 404)
        path = self.catalog.resolve_local_model(candidates[0]["id"])
        # Reuse graph validation and the profile-aware session factory.
        self.loaded_model_id = candidates[0]["id"]
        self.session = load_local_session(self.catalog, self.loaded_model_id, path, config)
        inputs = self.session.get_inputs()
        self.image_node = next((n for n in inputs if len(n.shape) == 4 and tuple(n.shape[1:]) == (6, 96, 96)), None)
        self.mel_node = next((n for n in inputs if len(n.shape) == 4 and tuple(n.shape[1:]) == (1, 80, 16)), None)
        if len(inputs) != 2 or self.image_node is None or self.mel_node is None:
            self.release()
            raise DomainError("LIPSYNC_GRAPH_UNSUPPORTED", "Expected Wav2Lip NCHW [N,6,96,96] and [N,1,80,16] inputs", 422)
        if mode == "wav2lip_gan_fp16" and any(n.type != "tensor(float16)" for n in inputs):
            self.release()
            raise DomainError("LIPSYNC_PRECISION", "Wav2Lip-GAN FP16 requires float16 graph inputs", 422)
        self.available, self.reason = True, None

    def process_frame(self, face_crop: np.ndarray, track_id: int, frame_input: LipSyncFrameInput, *,
                      scene_cut: bool = False) -> LipSyncResult:
        """Run a prepared local adapter; unprepared previews retain their original crop."""
        image_bgr(face_crop)
        mel = np.asarray(frame_input.mel_window, dtype=np.float32)
        if mel.ndim != 2 or not mel.shape[0] or not mel.shape[1] or not np.isfinite(mel).all():
            raise DomainError("LIPSYNC_FEATURES_INVALID", "Mel window must be finite bins-by-frames data", 422)
        self.gate.update(track_id, frame_input.rms, frame_input.face_confidence,
                         timestamp=frame_input.timestamp, scene_cut=scene_cut)
        if not self.available:
            return LipSyncResult(face_crop.copy(), np.zeros(face_crop.shape[:2], np.float32), 0.0,
                                 False, "LIPSYNC_MODEL_UNAVAILABLE")
        import cv2
        if mel.shape != (80, 16):
            raise DomainError("LIPSYNC_FEATURES_INVALID", "Wav2Lip needs an 80 by 16 mel window", 422)
        crop = cv2.resize(face_crop, (96, 96)).astype(np.float32) / 255
        masked = crop.copy()
        masked[48:] = 0
        packed = np.concatenate((masked, crop), axis=2).transpose(2, 0, 1)[None]
        dtype = lambda node: np.float16 if node.type == "tensor(float16)" else np.float32
        values = self.session.run(None, {self.image_node.name: packed.astype(dtype(self.image_node)),
                                       self.mel_node.name: mel[None, None].astype(dtype(self.mel_node))})
        output = np.asarray(values[0])
        if output.shape != (1, 3, 96, 96) or not np.isfinite(output).all():
            raise DomainError("LIPSYNC_OUTPUT_INVALID", "Wav2Lip returned an invalid image", 503)
        image = cv2.resize(np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8),
                           (face_crop.shape[1], face_crop.shape[0]))
        mask = np.zeros(face_crop.shape[:2], np.float32)
        mask[int(mask.shape[0] * .5):] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, mask.shape[0] * .015))
        # Do not gate closures on RMS: silent consonants still need the model's mouth prediction.
        return LipSyncResult(image, mask, float(np.clip(frame_input.face_confidence, 0, 1)), True, None)

    def capability(self) -> dict:
        record = self.record
        weights = record.status.upper() if record is not None else "UNKNOWN"
        return {"state": "UNSUPPORTED" if not self.available else "LOCAL_INFERENCE",
                "model": record.name if record is not None else None,
                "weights": weights,
                "sha256": record.expected_sha256 if record is not None else None,
                "provenance": record.provenance_state if record is not None else "UNKNOWN",
                "license": record.license_id if record is not None else "UNKNOWN",
                "inference": "LOCAL_ONNX" if self.available else "UNKNOWN", "three_d_visemes": "UNSUPPORTED",
                # Pixels only: this feature has no path to any audio track, enabled or not.
                "audio_modification": "never", "default": "off", "selection": "explicit (lip_sync_mode setting)",
                "reason": self.reason,
                "gate_tracks": self.gate.track_count}

    def release(self) -> None:
        self.gate.clear()
        if getattr(self, "loaded_model_id", None):
            self.catalog.set_vram_loaded(self.loaded_model_id, False)
            self.loaded_model_id = None
        self.session = None
        self.available = False
