"""Subpixel iris and oral-aperture constraints in the full-head ROI.

478-point MediaPipe includes irises; 468-point meshes do not. Gaze is geometric
relative to the eye aperture, not a calibrated point of regard. No temporal
averaging is applied to iris positions, so saccades do not acquire filter lag.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def polygon_mask(points, side):
    mask = np.zeros((side, side), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(points) * 16).astype(np.int32)], 255,
                 lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32) / 255


@dataclass(frozen=True)
class EyeGaze:
    center: np.ndarray
    vector: np.ndarray
    pitch: float
    yaw: float
    visible: bool


@dataclass(frozen=True)
class FaceDetail:
    iris: np.ndarray
    mouth: np.ndarray
    excluded_skin: np.ndarray
    eyes: tuple
    openness: float
    tau_mouth: float = .12
    eye_surface: np.ndarray | None = None

    def masks(self, side):
        def scale(value):
            return cv2.resize(value, (side, side), interpolation=cv2.INTER_LINEAR)
        return scale(self.iris), scale(self.mouth), scale(self.excluded_skin)

    def condition(self, image, target, *, mouth=True):
        """Image-space conditioning for models without trained gaze/depth inputs.

        Measured iris pixels are locked after synthesis/restoration. Oral detail
        combines synthesized high frequencies with the target's cavity shadows
        and tooth gaps, at the enhancer's native resolution. Never blur teeth.
        """
        from roop_3d.reconstruction.photometry import linearize
        iris, aperture, _ = self.masks(image.shape[0])
        ocular = iris
        if self.eye_surface is not None:
            # Replace the measured ocular aperture as well as its iris. Merely
            # painting the new iris can leave the old synthesized pupil behind.
            ocular = np.maximum(iris,cv2.resize(self.eye_surface,(image.shape[0],image.shape[0])))
        result = image.astype(np.float32) * (1 - ocular[..., None]) + target * ocular[..., None]
        if mouth and self.openness > self.tau_mouth and aperture.any():
            source, driving = linearize(image), linearize(target)
            weights = np.array([.114, .587, .299], np.float32)
            sl, tl = source @ weights, driving @ weights
            # Low frequencies determine depth, high frequencies retain both the
            # synthesized tooth texture and the actor's visible interdental gaps.
            low_s = cv2.GaussianBlur(sl, (0, 0), 1.2)
            low_t = cv2.GaussianBlur(tl, (0, 0), 1.2)
            detail = .35 * (sl - low_s) + .65 * (tl - low_t)
            luminance = np.clip(low_t + detail, 0, 1)
            luminance = np.minimum(luminance, tl + .004)  # dark cavity cannot turn into a white plate
            oral = source * (luminance / np.maximum(sl, .015))[..., None]
            # Synthesized black gaps must not erase target teeth/tongue.
            oral = np.where((sl < .015)[..., None], driving, oral)
            oral = np.where(oral <= .0031308, oral * 12.92, 1.055 * np.maximum(oral, 0) ** (1 / 2.4) - .055)
            gate = aperture[..., None] * np.clip((self.openness - self.tau_mouth) / .06, 0, 1)
            result = result * (1 - gate) + np.clip(oral * 255, 0, 255) * gate
        return np.clip(result, 0, 255).round().astype(np.uint8)


def extract_detail(mesh, points68, inverse, side):
    """Input dense points are frame pixels (z, if present, in pixel-width units)."""
    matrix = cv2.invertAffineTransform(np.asarray(inverse, np.float32))
    iris = np.zeros((side, side), np.float32)
    mouth = iris.copy()
    excluded = iris.copy()
    eye_surface = iris.copy()
    eyes = []
    openness = 0.
    if mesh is not None:
        mesh = np.asarray(mesh, np.float32)
        if mesh.ndim != 2 or len(mesh) not in (468, 478) or mesh.shape[1] not in (2, 3) or not np.isfinite(mesh).all():
            raise ValueError("Dense eye/mouth landmarks must be finite 468/478 x 2/3 frame coordinates")
        xy = cv2.transform(mesh[None, :, :2], matrix)[0]
        aperture = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        mouth = polygon_mask(xy[aperture], side)
        openness = float(np.linalg.norm(xy[14] - xy[13]) / max(np.linalg.norm(xy[308] - xy[78]), 1))
        for corners, lids, contour, iris_ids in (
            ((33, 133), (159, 145), (33, 160, 158, 133, 153, 144), (468, 469, 470, 471, 472)),
            ((362, 263), (386, 374), (362, 385, 387, 263, 373, 380), (473, 474, 475, 476, 477)),
        ):
            eye = polygon_mask(xy[list(contour)], side)
            excluded = np.maximum(excluded, eye)
            eye_surface = np.maximum(eye_surface,eye)
            if len(mesh) == 468:
                continue
            a, b = xy[list(corners)]
            axis = b - a
            width = max(float(np.linalg.norm(axis)), 1.)
            axis /= width
            down = np.array([-axis[1], axis[0]])
            center = xy[iris_ids[0]]
            offset = center - (a + b) / 2
            # Eyeball radius prior ~ half the fissure width. Independent eyes
            # retain measured vergence; do not force equal yaw/pitch.
            right3 = np.r_[axis, 0.]
            down3 = np.r_[down, 0.]
            forward3 = np.array([0., 0., 1.])
            sx = float(offset @ axis) / (width * .5)
            sy = float(offset @ down) / (width * .5)
            if mesh.shape[1] == 3:
                # MediaPipe z is width-relative depth, positive away. Use the
                # measured 3D eye plane to remove head yaw/pitch/roll from the
                # aperture-relative angles; z is not treated as metric depth.
                cloud = mesh * np.array([1, 1, -1], np.float32)
                corner_a, corner_b = cloud[list(corners)]
                right3 = corner_b - corner_a
                radius = max(float(np.linalg.norm(right3)) * .5, .5)
                right3 /= radius * 2
                down3 = cloud[lids[1]] - cloud[lids[0]]
                down3 -= (down3 @ right3) * right3
                if np.linalg.norm(down3) > 1e-5:
                    down3 /= np.linalg.norm(down3)
                else:
                    down3 = np.r_[down, 0.]
                forward3 = np.cross(right3, down3)
                forward3 /= max(float(np.linalg.norm(forward3)), 1e-5)
                if forward3[2] < 0:
                    forward3 *= -1
                displacement = cloud[iris_ids[0]] - (corner_a + corner_b) / 2
                sx, sy = float(displacement @ right3) / radius, float(displacement @ down3) / radius
            sx, sy = np.clip([sx, sy], -.95, .95)
            local = np.array([sx, sy, np.sqrt(max(1 - sx*sx - sy*sy, .01))], np.float32)
            local /= np.linalg.norm(local)
            vector = (right3*local[0] + down3*local[1] + forward3*local[2]).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            visible = np.linalg.norm(xy[lids[0]] - xy[lids[1]]) / width > .08
            if visible:
                rim = polygon_mask(xy[list(iris_ids[1:])], side) * eye
                iris = np.maximum(iris, rim)
            eyes.append(EyeGaze(center, vector, float(np.arctan2(local[1], local[2])),
                                float(np.arctan2(local[0], local[2])), bool(visible)))
    elif points68 is not None:
        xy = cv2.transform(np.asarray(points68, np.float32)[None, :, :2], matrix)[0]
        mouth = polygon_mask(xy[60:68], side)
        openness = float(np.linalg.norm(xy[66] - xy[62]) / max(np.linalg.norm(xy[64] - xy[60]), 1))
        for ids in (range(36, 42), range(42, 48)):
            excluded = np.maximum(excluded, polygon_mask(xy[list(ids)], side))
    excluded = np.maximum(excluded, mouth)
    return FaceDetail(iris, mouth, excluded, tuple(eyes), openness,eye_surface=eye_surface)
