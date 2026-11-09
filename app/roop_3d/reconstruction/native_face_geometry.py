"""NumPy decoders for native reconstruction encoders; no appearance renderer.

BFM coordinates are the original 3DDFA/HRN coordinates (one unit = 0.1m).
FLAME rigs are metric, +y up, +z toward the viewer, before registration.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from roop_3d.reconstruction.morphable_backbone import finite

CV_AXES = np.diag([1., -1., -1.]).astype(np.float32)


@dataclass(frozen=True)
class NativeCrop:
    image: np.ndarray
    matrix: np.ndarray
    inverse: np.ndarray


def deep3d_crop(image_bgr, landmarks5, standard5, size=224):
    """Upstream POS + integer PIL resize/crop, retaining its exact pixel transform.

    Matches 3DDFA-V3 util/preprocess.py, including y-up landmarks and the
    integer resize/crop rounding. Image origins in the returned transform are
    pixel centres, so PIL's half-pixel resize offset is retained.
    """
    height, width = image_bgr.shape[:2]
    points = finite(landmarks5, (5, 2), "five measured landmarks").copy()
    points[:, 1] = height - 1 - points[:, 1]
    standard = finite(standard5, (5, 3), "Deep3D standard landmarks")
    a = np.zeros((10, 8), np.float64)
    a[::2, :3], a[::2, 3] = standard, 1
    a[1::2, 4:7], a[1::2, 7] = standard, 1
    solution, _, rank, _ = np.linalg.lstsq(a, points.reshape(-1), rcond=None)
    scale = (np.linalg.norm(solution[:3]) + np.linalg.norm(solution[4:7])) / 2
    if rank < 8 or scale <= 1e-8:
        raise ValueError("Degenerate reconstruction crop landmarks")
    scale = (102. * size / 224) / scale
    w, h = int(width * scale), int(height * scale)
    if min(w, h) <= 0 or max(w, h) > 32768:
        raise ValueError("Reconstruction crop exceeds image budget")
    tx, ty = solution[3], solution[7]
    left = int(w / 2 - size / 2 + (tx - width / 2) * scale)
    top = int(h / 2 - size / 2 + (height / 2 - ty) * scale)
    rgb = Image.fromarray(image_bgr[:, :, ::-1]).resize((w, h), Image.Resampling.BICUBIC)
    cropped = np.asarray(rgb.crop((left, top, left + size, top + size)))[:, :, ::-1].copy()
    sx, sy = w / width, h / height
    matrix = np.array([[sx, 0, (sx-1)/2-left], [0, sy, (sy-1)/2-top]], np.float64)
    return NativeCrop(cropped, matrix, cv2.invertAffineTransform(matrix))


def bfm_rotation(angles):
    x, y, z = finite(angles, (3,), "BFM Euler angles")
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def bfm_camera(coefficients):
    """Native 257-vector -> proper OpenCV rotation, metric t and pinhole K.

    3DDFA has three translations, a camera distance of 10 units and a 1015px
    focal length. The metric gauge is explicitly fixed to 0.1m/unit, s=1.
    """
    alpha = finite(coefficients, (257,), "3DDFA-V3 coefficients")
    rotation = CV_AXES @ bfm_rotation(alpha[224:227]) @ CV_AXES
    translation = np.array([alpha[254], -alpha[255], 10.-alpha[256]], np.float32) * .1
    # Native image landmarks use y-up; upstream flips y with 224-1-y.
    k = np.array([[1015., 0, 112.], [0, 1015., 111.], [0, 0, 1]], np.float32)
    return rotation, translation, k


class SurfaceTransfer:
    """Registered barycentric correspondence, never an index reshape."""
    def __init__(self, indices, weights, source_count, target_count):
        self.indices = np.asarray(indices)
        if (self.indices.shape != (target_count, 3) or self.indices.dtype.kind not in "iu"
                or np.any(self.indices < 0) or np.any(self.indices >= source_count)):
            raise ValueError("Invalid registered surface indices")
        self.weights = finite(weights, (target_count, 3), "registered barycentric weights")
        if np.any(self.weights < -1e-6) or not np.allclose(self.weights.sum(1), 1, atol=1e-5):
            raise ValueError("Surface transfer requires convex barycentric weights")
        self.source_count = source_count

    def __call__(self, values):
        values = np.asarray(values, np.float32)
        if values.shape != (self.source_count, 3) or not np.isfinite(values).all():
            raise ValueError("Invalid surface transfer geometry")
        return np.einsum("vkc,vk->vc", values[self.indices], self.weights).astype(np.float32)


class FlameExpressionRig:
    """FLAME LBS on supplied identity anatomy, including jaw and separate eyelids.

    ``smirk_to_expression`` must map the checkpoint's actual expression basis
    into this rig's basis. Identity and Open-model conversion are prepared
    offline; this decoder never relabels FLAME2020 coefficients as Open/ARKit.
    """
    def __init__(self, data):
        self.template = finite(data["template_m"], (5023, 3), "FLAME Open template")
        expression = np.asarray(data["expression_dirs_m"], np.float32)
        if expression.ndim != 3 or expression.shape[:2] != (5023, 3) or not np.isfinite(expression).all():
            raise ValueError("Invalid FLAME expression directions")
        conversion = finite(data["smirk_to_expression"], (expression.shape[2], 50), "SMIRK basis conversion")
        self.expression_dirs = np.einsum("vck,ke->vce", expression, conversion)
        self.parents = np.asarray(data["parents"])
        joints = len(self.parents)
        if (joints != 5 or self.parents.dtype.kind not in "iu" or self.parents[0] != -1
                or any(not 0 <= self.parents[j] < j for j in range(1, joints))):
            raise ValueError("Expected ordered five-joint FLAME hierarchy")
        self.regressor = finite(data["joint_regressor"], (joints, 5023), "joint regressor")
        self.weights = finite(data["skinning_weights"], (5023, joints), "skinning weights")
        if np.any(self.weights < 0) or not np.allclose(self.weights.sum(1), 1, atol=1e-4):
            raise ValueError("Invalid FLAME skinning weights")
        self.pose_dirs = finite(data["pose_dirs_m"], ((joints-1)*9, 5023*3), "pose correctives")
        self.eyelids = finite(data["eyelid_dirs_m"], (2, 5023, 3), "left/right eyelid directions")

    def express(self, identity_m, expression50, jaw, eyelids):
        identity = finite(identity_m, (5023, 3), "FLAME identity")
        coefficients = finite(expression50, (50,), "SMIRK expression")
        jaw = finite(jaw, (3,), "jaw articulation")
        eyelids = finite(eyelids, (2,), "left/right eyelids")
        if np.any((eyelids < 0) | (eyelids > 1)):
            raise ValueError("SMIRK eyelids must be in [0,1]")
        shaped = identity + np.einsum("vce,e->vc", self.expression_dirs, coefficients)
        joints = self.regressor @ shaped
        rotations = np.repeat(np.eye(3)[None], 5, axis=0)
        rotations[2] = cv2.Rodrigues(jaw.astype(np.float64))[0]
        feature = (rotations[1:] - np.eye(3)).reshape(-1)
        posed = shaped + (feature @ self.pose_dirs).reshape(-1, 3)
        transforms = []
        for j, parent in enumerate(self.parents):
            local = np.eye(4)
            local[:3, :3] = rotations[j]
            local[:3, 3] = joints[j] if parent == -1 else joints[j] - joints[parent]
            transforms.append(local if parent == -1 else transforms[parent] @ local)
        transforms = np.asarray(transforms)
        transforms[:, :3, 3] -= np.einsum("jab,jb->ja", transforms[:, :3, :3], joints)
        blended = np.einsum("vj,jab->vab", self.weights, transforms)
        result = np.einsum("vab,vb->va", blended, np.column_stack((posed, np.ones(5023))))[:, :3]
        result += np.einsum("evc,e->vc", self.eyelids, eyelids)
        return result.astype(np.float32)


class UVSurfaceMap:
    """Static UV triangle lookup; runtime interpolation uses registered geometry.

    UV rasterization only generates correspondence attributes. The output is
    never a mesh color render. Maps are built once at 256px per session.
    """
    def __init__(self, uvs, faces, size=256):
        self.size = size
        self.indices = np.zeros((size, size, 3), np.int32)
        self.weights = np.zeros((size, size, 3), np.float32)
        self.valid = np.zeros((size, size), bool)
        # Match PyTorch3D's NDC pixel-centre convention used by HRN world2uv.
        pixels = np.column_stack((uvs[:, 0], 1-uvs[:, 1])) * size - .5
        for face in faces:
            p = pixels[face]
            low = np.maximum(np.ceil(p.min(0)).astype(int), 0)
            high = np.minimum(np.floor(p.max(0)).astype(int), size-1)
            if np.any(low > high):
                continue
            basis = np.column_stack((p[1]-p[0], p[2]-p[0]))
            if abs(np.linalg.det(basis)) < 1e-10:
                continue
            y, x = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
            uv = (np.stack((x, y), -1)-p[0]) @ np.linalg.inv(basis).T
            bary = np.concatenate((1-uv.sum(-1, keepdims=True), uv), -1)
            inside = bary.min(-1) >= -1e-6
            ys, xs = y[inside], x[inside]
            self.indices[ys, xs], self.weights[ys, xs] = face, bary[inside]
            self.valid[ys, xs] = True

    def interpolate(self, vertices):
        result = np.einsum("hwkc,hwk->hwc", vertices[self.indices], self.weights)
        return np.where(self.valid[..., None], result, 0).astype(np.float32)


def sample_hrn_mid(deformation, uvs):
    """Reproduce HRN add_nonlinear_offset's native UV convention exactly.

    HRN uses w/h, y-up indexing and reversed vertical interpolation weights;
    substituting conventional align_corners bilinear sampling changes results.
    """
    h, w = deformation.shape[:2]
    xy = uvs * (w, h)
    integer = np.floor(xy).astype(int)
    a, b = (xy-integer).T
    x, y = integer.T
    left, right = np.clip(x, 0, w-1), np.clip(x+1, 0, w-1)
    top, bottom = np.clip(h-1-y, 0, h-1), np.clip(h-y, 0, h-1)
    return (deformation[top, left] * ((1-a)*b)[:, None]
            + deformation[bottom, left] * ((1-a)*(1-b))[:, None]
            + deformation[top, right] * (a*b)[:, None]
            + deformation[bottom, right] * (a*(1-b))[:, None]).astype(np.float32)
