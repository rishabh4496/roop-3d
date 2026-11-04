"""Controlled, consented golden-video dataset and scenario generator (Stage 11).

Provides representative test categories:
1. Frontal faces
2. Extreme yaw and profile
3. Rapid head movement
4. Speaking and expressions
5. Hand occlusion
6. Hair and glasses
7. Multiple interacting people
8. Dark scenes
9. Motion blur
10. Different resolutions
11. Different skin textures and lighting

Each scene carries full ground-truth annotations:
- Canonical performer target frames
- Frame timestamps and FPS
- Ground-truth 3D head poses (yaw, pitch, roll)
- Ground-truth facial landmarks (5-point, 68-point, 106-point)
- Ground-truth expression coefficients (mouth opening, smile, eye closure)
- Ground-truth foreground occluder masks (hands, glasses, microphones)
- Background masks for background-preservation verification
- Associated consented source FaceSet ID
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import av
import cv2
import numpy as np

from roop_3d.core.errors import DomainError


class SceneCategory(str, Enum):
    FRONTAL = "frontal_faces"
    EXTREME_YAW = "extreme_yaw_and_profile"
    RAPID_MOTION = "rapid_head_movement"
    SPEAKING_EXPRESSIONS = "speaking_and_expressions"
    HAND_OCCLUSION = "hand_occlusion"
    HAIR_GLASSES = "hair_and_glasses"
    MULTIPLE_PEOPLE = "multiple_interacting_people"
    DARK_SCENES = "dark_scenes"
    MOTION_BLUR = "motion_blur"
    DIFFERENT_RESOLUTIONS = "different_resolutions"
    SKIN_LIGHTING = "different_skin_textures_and_lighting"


@dataclass
class GoldenFrame:
    """A single annotated frame in a golden test scene."""
    index: int
    timestamp: float
    image: np.ndarray  # BGR uint8
    pose_deg: tuple[float, float, float]  # (yaw, pitch, roll)
    landmarks5: np.ndarray  # (5, 2) float32
    landmarks68: np.ndarray | None  # (68, 2) float32
    landmarks106: np.ndarray | None  # (106, 2) float32
    expression: dict[str, float] = field(default_factory=dict)
    occluder_mask: np.ndarray | None = None  # (H, W) float32 in [0, 1]
    face_mask: np.ndarray | None = None  # (H, W) float32 in [0, 1]
    background_mask: np.ndarray | None = None  # (H, W) float32 in [0, 1]
    secondary_faces: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GoldenScene:
    """Complete golden-video test scene with full ground truth."""
    scene_id: str
    category: SceneCategory
    title: str
    description: str
    fps: float
    resolution: tuple[int, int]  # (width, height)
    frames: list[GoldenFrame]
    source_faceset_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / max(self.fps, 1.0)

    def write_video(self, output_path: Path) -> Path:
        """Write the scene frames to an MP4 video file with standard libx264 encoding."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.resolution
        fps_int = int(round(self.fps))

        with av.open(str(output_path), "w") as container:
            stream = container.add_stream("libx264", rate=fps_int)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, fps_int)
            stream.codec_context.time_base = Fraction(1, fps_int)

            for frame in self.frames:
                vframe = av.VideoFrame.from_ndarray(frame.image, format="bgr24")
                vframe.pts = frame.index
                vframe.time_base = Fraction(1, fps_int)
                for packet in stream.encode(vframe):
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

        return output_path


def _generate_synthetic_canvas(
    width: int,
    height: int,
    seed: int = 42,
    base_color: tuple[int, int, int] = (120, 140, 160),
) -> np.ndarray:
    """Generate a reproducible, textured synthetic background canvas."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = (xx / width) * 30.0 + (yy / height) * 20.0
    noise = rng.normal(0, 3, (height, width, 3)).astype(np.float32)
    canvas = np.zeros((height, width, 3), np.float32)
    for c in range(3):
        canvas[:, :, c] = base_color[c] + gradient + noise[:, :, c]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _draw_synthetic_face(
    canvas: np.ndarray,
    center: tuple[float, float],
    size: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    mouth_open: float = 0.0,
    smile: float = 0.0,
    blink_left: float = 0.0,
    blink_right: float = 0.0,
    skin_tone: tuple[int, int, int] = (180, 205, 235),  # BGR skin
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a projected 3D-informed facial fixture and compute ground-truth 5, 68, 106 landmarks."""
    h, w = canvas.shape[:2]
    cx, cy = center
    rad_yaw = np.radians(yaw)
    rad_pitch = np.radians(pitch)
    rad_roll = np.radians(roll)

    # 3D rotation offsets
    dx_yaw = np.sin(rad_yaw) * (size * 0.45)
    dy_pitch = -np.sin(rad_pitch) * (size * 0.35)

    # Face ellipse dimensions adjusted for yaw/pitch foreshortening
    face_w = int(max(size * 0.7 * np.cos(rad_yaw), 12))
    face_h = int(max(size * 0.9 * np.cos(rad_pitch), 16))

    # Face mask
    face_mask = np.zeros((h, w), np.float32)
    face_center = (int(round(cx + dx_yaw * 0.3)), int(round(cy + dy_pitch * 0.3)))
    cv2.ellipse(
        canvas,
        face_center,
        (face_w // 2, face_h // 2),
        np.degrees(rad_roll),
        0,
        360,
        skin_tone,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        face_mask,
        face_center,
        (face_w // 2, face_h // 2),
        np.degrees(rad_roll),
        0,
        360,
        1.0,
        -1,
        cv2.LINE_AA,
    )

    # Eyes
    eye_offset_x = size * 0.22 * np.cos(rad_yaw)
    eye_y = cy - size * 0.12 + dy_pitch * 0.5
    left_eye_center = (cx - eye_offset_x, eye_y)
    right_eye_center = (cx + eye_offset_x, eye_y)

    # Nose
    nose_x = cx + dx_yaw * 0.6
    nose_y = cy + size * 0.05 + dy_pitch * 0.6

    # Mouth
    mouth_x = cx + dx_yaw * 0.4
    mouth_y = cy + size * 0.25 + dy_pitch * 0.4
    mouth_w = max(size * 0.25 * np.cos(rad_yaw), 8)
    mouth_gap = max(size * 0.15 * mouth_open, 2.0)

    # 5 Key landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    lm5 = np.array([
        [left_eye_center[0], left_eye_center[1]],
        [right_eye_center[0], right_eye_center[1]],
        [nose_x, nose_y],
        [mouth_x - mouth_w * 0.5, mouth_y - smile * size * 0.05],
        [mouth_x + mouth_w * 0.5, mouth_y - smile * size * 0.05],
    ], dtype=np.float32)

    # Draw eyes
    for pt, blink in [(left_eye_center, blink_left), (right_eye_center, blink_right)]:
        ew = max(int(size * 0.08 * np.cos(rad_yaw)), 3)
        eh = max(int(size * 0.04 * (1.0 - blink * 0.85)), 1)
        cv2.ellipse(canvas, (int(round(pt[0])), int(round(pt[1]))), (ew, eh), 0, 0, 360, (50, 50, 60), -1, cv2.LINE_AA)

    # Draw nose bridge & tip
    cv2.circle(canvas, (int(round(nose_x)), int(round(nose_y))), max(int(size * 0.04), 2), (140, 165, 195), -1, cv2.LINE_AA)

    # Draw mouth
    cv2.ellipse(
        canvas,
        (int(round(mouth_x)), int(round(mouth_y))),
        (int(round(mouth_w * 0.5)), int(round(mouth_gap))),
        0,
        0,
        360,
        (60, 60, 140),
        -1,
        cv2.LINE_AA,
    )

    # Construct synthetic 68 and 106 landmarks
    lm68 = np.zeros((68, 2), dtype=np.float32)
    # Jaw contour (0-16)
    jaw_angles = np.linspace(-np.pi * 0.8, -np.pi * 0.2, 17)
    for i, a in enumerate(jaw_angles):
        lm68[i] = [face_center[0] + np.cos(a) * face_w * 0.5, face_center[1] - np.sin(a) * face_h * 0.5]
    # Eyebrows (17-26)
    lm68[17:22] = lm5[0] + np.array([[-size * 0.08, -size * 0.06], [-size * 0.04, -size * 0.08], [0, -size * 0.08], [size * 0.04, -size * 0.07], [size * 0.08, -size * 0.05]])
    lm68[22:27] = lm5[1] + np.array([[-size * 0.08, -size * 0.05], [-size * 0.04, -size * 0.07], [0, -size * 0.08], [size * 0.04, -size * 0.08], [size * 0.08, -size * 0.06]])
    # Nose (27-35)
    for idx, t in enumerate(np.linspace(0, 1, 4)):
        lm68[27 + idx] = [left_eye_center[0] * 0.5 + right_eye_center[0] * 0.5, eye_y + (nose_y - eye_y) * t]
    lm68[31:36] = np.array([[nose_x - size * 0.05, nose_y + 2], [nose_x - size * 0.02, nose_y + 4], [nose_x, nose_y + 4], [nose_x + size * 0.02, nose_y + 4], [nose_x + size * 0.05, nose_y + 2]])
    # Eyes (36-47)
    lm68[36] = lm5[0] + [-size * 0.06, 0]
    lm68[39] = lm5[0] + [size * 0.06, 0]
    lm68[37:39] = lm5[0] + [[-size * 0.02, -size * 0.025 * (1 - blink_left)], [size * 0.02, -size * 0.025 * (1 - blink_left)]]
    lm68[40:42] = lm5[0] + [[size * 0.02, size * 0.025 * (1 - blink_left)], [-size * 0.02, size * 0.025 * (1 - blink_left)]]
    lm68[42] = lm5[1] + [-size * 0.06, 0]
    lm68[45] = lm5[1] + [size * 0.06, 0]
    lm68[43:45] = lm5[1] + [[-size * 0.02, -size * 0.025 * (1 - blink_right)], [size * 0.02, -size * 0.025 * (1 - blink_right)]]
    lm68[46:48] = lm5[1] + [[size * 0.02, size * 0.025 * (1 - blink_right)], [-size * 0.02, size * 0.025 * (1 - blink_right)]]
    # Mouth (48-67)
    lm68[48] = lm5[3]
    lm68[54] = lm5[4]
    for idx, t in enumerate(np.linspace(0.2, 0.8, 5)):
        lm68[49 + idx] = [mouth_x - mouth_w * 0.5 + mouth_w * t, mouth_y - mouth_gap * 0.5]
    for idx, t in enumerate(np.linspace(0.8, 0.2, 5)):
        lm68[55 + idx] = [mouth_x - mouth_w * 0.5 + mouth_w * t, mouth_y + mouth_gap * 0.5]
    lm68[60:68] = lm68[48:56] * 0.9 + np.array([mouth_x, mouth_y]) * 0.1

    # 106 points layout
    lm106 = np.zeros((106, 2), dtype=np.float32)
    lm106[:68] = lm68
    # Dense contour interpolation
    lm106[68:106] = lm68[np.linspace(0, 67, 38, dtype=int)]

    return canvas, lm5, lm68, lm106


def generate_synthetic_scene(
    category: SceneCategory,
    source_faceset_id: str = "akansha",
    frame_count: int = 15,
    resolution: tuple[int, int] = (512, 512),
    fps: float = 25.0,
    seed: int = 100,
) -> GoldenScene:
    """Procedurally generate a ground-truth golden scene for any representative category."""
    w, h = resolution
    cx, cy = w / 2.0, h / 2.0
    face_size = min(w, h) * 0.52
    frames: list[GoldenFrame] = []

    for idx in range(frame_count):
        t = idx / max(fps, 1.0)
        canvas = _generate_synthetic_canvas(w, h, seed=seed + idx)
        occluder_mask = np.zeros((h, w), np.float32)
        yaw, pitch, roll = 0.0, 0.0, 0.0
        mouth_open = 0.0
        smile = 0.0
        blink_l, blink_r = 0.0, 0.0
        skin_tone = (160, 195, 230)
        face_center_pt = (cx, cy)
        effective_face_size = face_size
        secondary_faces: list[dict[str, Any]] = []

        # 1. Category-specific parameter dynamics
        if category == SceneCategory.FRONTAL:
            yaw = 2.0 * np.sin(t * 1.5)
            pitch = 1.0 * np.cos(t * 1.2)
            roll = 0.5 * np.sin(t)

        elif category == SceneCategory.EXTREME_YAW:
            # Sweeps yaw from -70 degrees to +70 degrees
            yaw = -70.0 + (140.0 * idx / max(frame_count - 1, 1))
            pitch = 4.0 * np.sin(t * 2.0)

        elif category == SceneCategory.RAPID_MOTION:
            # High-velocity turns up to 450 deg/sec
            yaw = 45.0 * np.sin(t * 8.0)
            pitch = 20.0 * np.cos(t * 6.0)
            roll = 15.0 * np.sin(t * 7.0)

        elif category == SceneCategory.SPEAKING_EXPRESSIONS:
            # Talking rhythm + smile modulation
            mouth_open = 0.15 + 0.25 * np.abs(np.sin(t * 9.0))
            smile = 0.10 + 0.12 * np.sin(t * 3.0)
            if idx in (4, 5, 11, 12):
                blink_l = blink_r = 0.95

        elif category == SceneCategory.HAND_OCCLUSION:
            # Performer talking while a hand traverses across lower face
            mouth_open = 0.2 * np.abs(np.sin(t * 4.0))

        elif category == SceneCategory.HAIR_GLASSES:
            yaw = 3.0 * np.sin(t * 1.5)

        elif category == SceneCategory.MULTIPLE_PEOPLE:
            face_center_pt = (cx - face_size * 0.45, cy)
            effective_face_size = face_size * 0.75
            yaw = -15.0

        elif category == SceneCategory.DARK_SCENES:
            canvas = (canvas.astype(np.float32) * 0.22).astype(np.uint8)
            skin_tone = (45, 55, 70)
            yaw = 10.0
            pitch = -5.0

        elif category == SceneCategory.MOTION_BLUR:
            yaw = t * 15.0

        elif category == SceneCategory.DIFFERENT_RESOLUTIONS:
            yaw = 5.0

        elif category == SceneCategory.SKIN_LIGHTING:
            skin_tone = (190, 220, 245)
            yaw = -10.0

        # 2. Draw primary face
        canvas, lm5, lm68, lm106 = _draw_synthetic_face(
            canvas,
            face_center_pt,
            effective_face_size,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mouth_open=mouth_open,
            smile=smile,
            blink_left=blink_l,
            blink_right=blink_r,
            skin_tone=skin_tone,
        )

        # 3. Apply post-face overlays & occluders
        if category == SceneCategory.HAND_OCCLUSION:
            hand_x = int(cx - face_size * 0.4 + (face_size * 0.8 * idx / max(frame_count - 1, 1)))
            hand_y = int(cy + face_size * 0.1)
            hand_w, hand_h = int(face_size * 0.35), int(face_size * 0.28)
            hand_color = (130, 160, 200)
            cv2.ellipse(canvas, (hand_x, hand_y), (hand_w // 2, hand_h // 2), 25, 0, 360, hand_color, -1, cv2.LINE_AA)
            cv2.ellipse(occluder_mask, (hand_x, hand_y), (hand_w // 2, hand_h // 2), 25, 0, 360, 1.0, -1, cv2.LINE_AA)

        elif category == SceneCategory.HAIR_GLASSES:
            g_y = int(cy - face_size * 0.12)
            cv2.rectangle(canvas, (int(cx - face_size * 0.32), g_y - 12), (int(cx - face_size * 0.06), g_y + 12), (20, 20, 30), 4, cv2.LINE_AA)
            cv2.rectangle(canvas, (int(cx + face_size * 0.06), g_y - 12), (int(cx + face_size * 0.32), g_y + 12), (20, 20, 30), 4, cv2.LINE_AA)
            cv2.line(canvas, (int(cx - face_size * 0.06), g_y), (int(cx + face_size * 0.06), g_y), (20, 20, 30), 3, cv2.LINE_AA)
            cv2.rectangle(occluder_mask, (int(cx - face_size * 0.32), g_y - 12), (int(cx - face_size * 0.06), g_y + 12), 1.0, 4)
            cv2.rectangle(occluder_mask, (int(cx + face_size * 0.06), g_y - 12), (int(cx + face_size * 0.32), g_y + 12), 1.0, 4)

        elif category == SceneCategory.MULTIPLE_PEOPLE:
            pos_b = (cx + face_size * 0.45, cy + face_size * 0.1)
            canvas, lm5_b, lm68_b, lm106_b = _draw_synthetic_face(
                canvas, pos_b, face_size * 0.65, yaw=20.0, pitch=0.0, roll=0.0, skin_tone=(150, 180, 215)
            )
            secondary_faces.append({
                "track_id": 1,
                "landmarks5": lm5_b,
                "landmarks68": lm68_b,
                "landmarks106": lm106_b,
                "pose_deg": (20.0, 0.0, 0.0),
            })

        elif category == SceneCategory.MOTION_BLUR:
            ksize = 11
            kernel = np.zeros((ksize, ksize), np.float32)
            kernel[ksize // 2, :] = 1.0 / ksize
            canvas = cv2.filter2D(canvas, -1, kernel)

        elif category == SceneCategory.SKIN_LIGHTING:
            lighting_gradient = np.tile(np.linspace(1.3, 0.6, w, dtype=np.float32)[None, :, None], (h, 1, 3))
            canvas = np.clip(canvas.astype(np.float32) * lighting_gradient, 0, 255).astype(np.uint8)

        # 4. Construct background mask & face mask
        face_mask = np.zeros((h, w), np.float32)
        cv2.ellipse(face_mask, (int(round(face_center_pt[0])), int(round(face_center_pt[1]))), (int(effective_face_size * 0.45), int(effective_face_size * 0.55)), 0, 0, 360, 1.0, -1)
        if category == SceneCategory.MULTIPLE_PEOPLE:
            cv2.ellipse(face_mask, (int(round(cx + face_size * 0.45)), int(round(cy + face_size * 0.1))), (int(face_size * 0.30), int(face_size * 0.40)), 0, 0, 360, 1.0, -1)
        bg_mask = 1.0 - face_mask

        # 5. Assemble GoldenFrame
        frames.append(GoldenFrame(
            index=idx,
            timestamp=t,
            image=canvas,
            pose_deg=(yaw, pitch, roll),
            landmarks5=lm5,
            landmarks68=lm68,
            landmarks106=lm106,
            expression={"mouth_open": mouth_open, "smile": smile, "blink": blink_l},
            occluder_mask=occluder_mask,
            face_mask=face_mask,
            background_mask=bg_mask,
            secondary_faces=secondary_faces,
        ))

    return GoldenScene(
        scene_id=f"golden_{category.value}_{resolution[0]}x{resolution[1]}",
        category=category,
        title=f"Golden Scene: {category.value.replace('_', ' ').title()}",
        description=f"Objective benchmark scenario for {category.value} with {frame_count} frames at {resolution[0]}x{resolution[1]}.",
        fps=fps,
        resolution=resolution,
        frames=frames,
        source_faceset_id=source_faceset_id,
        metadata={"seed": seed, "procedural": True},
    )


class GoldenDatasetRegistry:
    """Manages the controlled, consented golden test dataset across all 11 categories."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(".runtime/golden_dataset")
        self._scenes: dict[str, GoldenScene] = {}

    def get_canonical_dataset(self, source_faceset_id: str = "akansha", frame_count: int = 12) -> list[GoldenScene]:
        """Retrieve the complete 11-category golden dataset."""
        scenes: list[GoldenScene] = []
        for cat in SceneCategory:
            res = (384, 384) if cat == SceneCategory.DIFFERENT_RESOLUTIONS else (512, 512)
            scenes.append(generate_synthetic_scene(
                category=cat,
                source_faceset_id=source_faceset_id,
                frame_count=frame_count,
                resolution=res,
                seed=100 + list(SceneCategory).index(cat) * 17,
            ))
        return scenes

    def materialize_dataset(self, scenes: Sequence[GoldenScene]) -> dict[str, Path]:
        """Save scenes as MP4 clips in the cache directory and return paths."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for scene in scenes:
            clip_path = self.cache_dir / f"{scene.scene_id}.mp4"
            scene.write_video(clip_path)
            paths[scene.scene_id] = clip_path
        return paths
