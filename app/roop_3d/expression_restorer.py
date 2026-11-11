"""Stage 2.5: current-frame expression residuals with identity/pose locked.

No temporal neutral reference: a blink on the first frame must also transfer.
Uses the existing catalog and profile-bounded ONNX/TensorRT FP16 sessions.
Manual semantic edits operate on LivePortrait's 21 implicit 3D keypoints;
these are artistic controls, not calibrated anatomical measurements.
"""
from __future__ import annotations
import cv2
import numpy as np
from roop_3d.core.errors import DomainError
from roop_3d.core.face_editor import FaceEditor
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, tensor, image_bgr
from roop_3d.engine.volumetric.reference_adapter import LivePortraitAdapter, Motion, portrait_tensor


def eye_aspect_ratio(eye):
    p = np.asarray(eye, np.float64)
    if p.shape != (6, 2) or not np.isfinite(p).all():
        raise ValueError('EAR requires six finite eye landmarks')
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-4:
        raise DomainError('EXPRESSION_GEOMETRY_INVALID', 'Degenerate eye landmarks', 422)
    return float((np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*width))


class PortraitGeometry:
    """Inspected InsightFace 68-point graph; coordinates remain in crop pixels."""
    def __init__(self, catalog, config):
        self.model = OrtModel(catalog, '1k3d68.onnx', config)
        import onnx
        nodes = onnx.load(str(self.model.path), load_external_data=False).graph.node[:8]
        self.internal = (any(n.name.startswith(('Sub', '_minus')) for n in nodes) and
                         any(n.name.startswith(('Mul', '_mul')) for n in nodes)) or any(n.name == 'bn_data' for n in nodes[:3])
        self.input = self.model.session.get_inputs()[0]

    def landmarks(self, image, box=None):
        image_bgr(image)
        h, w = image.shape[:2]
        box = np.asarray(box if box is not None else (w*.12, h*.1, w*.88, h*.9), np.float32)
        side = int(self.input.shape[2])
        scale = side / (max(box[2]-box[0], box[3]-box[1])*1.5)
        matrix = np.column_stack((np.eye(2)*scale, side/2-(box[:2]+box[2:])/2*scale))
        crop = cv2.warpAffine(image, matrix, (side, side))
        blob = cv2.dnn.blobFromImage(crop, 1 if self.internal else 1/128, (side, side),
                                   (0,)*3 if self.internal else (127.5,)*3, swapRB=True)
        raw = next(iter(self.model.run({self.input.name: blob}).values())).reshape(-1, 3)[-68:]
        points = cv2.transform(((raw[:, :2]+1)*(side/2))[None], cv2.invertAffineTransform(matrix))[0]
        return tensor(points.astype(np.float32), (68, 2))

    def release(self):
        self.model.release()


def edit_motion(source: Motion, expression, edit: FaceEditor):
    """Keep canonical geometry immutable except for explicitly requested morphs."""
    canonical, delta = source.canonical.copy(), expression.copy()
    arkit = edit.arkit_blendshapes
    get = lambda name: float(arkit.get(name, 0.0))
    smile = 0.5 * (get('mouthSmileLeft') + get('mouthSmileRight'))
    smile += 0.5 * (get('mouthDimpleLeft') + get('mouthDimpleRight'))
    smile -= 0.5 * (get('mouthFrownLeft') + get('mouthFrownRight'))
    smile = float(np.clip(edit.mouth_smile + smile, -1, 1))
    blink_left = max(edit.blink_left, get('eyeBlinkLeft'), get('eyeSquintLeft'))
    blink_right = max(edit.blink_right, get('eyeBlinkRight'), get('eyeSquintRight'))
    blink_left = float(np.clip(blink_left - 0.5 * get('eyeWideLeft'), 0, 1))
    blink_right = float(np.clip(blink_right - 0.5 * get('eyeWideRight'), 0, 1))
    mouth_open = max(edit.mouth_open, get('jawOpen'), get('mouthLowerDownLeft'), get('mouthLowerDownRight'),
                     0.35 * get('tongueOut'))
    mouth_open = float(np.clip(mouth_open - max(get('mouthClose'), get('mouthPressLeft'), get('mouthPressRight')), 0, 1))
    mouth_pout = max(edit.mouth_pout, get('mouthFunnel'), get('mouthPucker'), get('mouthRollLower'),
                     get('mouthRollUpper'), get('mouthShrugUpper'))
    mouth_pout = float(np.clip(mouth_pout, 0, 1))
    brow_raise = float(np.clip(edit.eyebrow_raise + get('browInnerUp')
                               + 0.5 * (get('browOuterUpLeft') + get('browOuterUpRight'))
                               - 0.5 * (get('browDownLeft') + get('browDownRight')), -1, 1))
    brow_tilt = float(np.clip(edit.eyebrow_tilt + 0.5 * (get('browOuterUpRight') - get('browOuterUpLeft'))
                              + 0.5 * (get('browDownLeft') - get('browDownRight')), -1, 1))
    gaze_left_x = float(np.clip(edit.gaze_x + get('eyeLookOutLeft') - get('eyeLookInLeft'), -1, 1))
    gaze_right_x = float(np.clip(edit.gaze_x + get('eyeLookOutRight') - get('eyeLookInRight'), -1, 1))
    gaze_left_y = float(np.clip(edit.gaze_y + get('eyeLookUpLeft') - get('eyeLookDownLeft'), -1, 1))
    gaze_right_y = float(np.clip(edit.gaze_y + get('eyeLookUpRight') - get('eyeLookDownRight'), -1, 1))
    jaw_width = float(np.clip(edit.jaw_width + 0.5 * get('cheekPuff')
                              + 0.25 * (get('mouthStretchLeft') + get('mouthStretchRight')), -1, 1))
    # Spatial lower-face weights make jaw/chin edits independent of target bone structure.
    y = canonical[0, :, 1]
    lower = np.clip((y-np.median(y)) / max(float(np.ptp(y))*.5, 1e-5), 0, 1)
    canonical[0, :, 0] *= 1 + .15*jaw_width*lower
    canonical[0, :, 1] += .08*edit.chin_length*lower
    delta[0, 11, 0] += .015*gaze_left_x
    delta[0, 15, 0] += .015*gaze_right_x
    delta[0, 11, 1] -= .01*gaze_left_y
    delta[0, 15, 1] -= .01*gaze_right_y
    delta[0, 1, 1] -= .015*(brow_raise+brow_tilt)
    delta[0, 2, 1] -= .015*(brow_raise-brow_tilt)
    delta[0, (14, 20), 1] -= .02*smile
    delta[0, (14, 20), 2] += .02*mouth_pout
    delta[0, 14, 0] += .012 * (get('mouthLeft') - get('mouthRight'))
    delta[0, 20, 0] += .012 * (get('mouthRight') - get('mouthLeft'))
    delta[0, 19, 1] += .025*mouth_open
    delta[0, 17, 1] -= .01*mouth_open
    delta[0, 17, 1] -= .012 * get('mouthShrugUpper')
    delta[0, 19, 2] += .02 * get('jawForward')
    cheek = float(np.clip(get('cheekPuff') + 0.5 * (get('cheekSquintLeft') + get('cheekSquintRight')), 0, 1))
    canonical[0, (5, 6, 9, 10), 0] *= 1 + 0.06 * cheek
    canonical[0, (5, 6, 9, 10), 2] += 0.025 * cheek
    delta[0, 7, 2] += 0.012 * (get('noseSneerLeft') + get('noseSneerRight'))
    delta[0, 7, 0] += 0.006 * (get('jawLeft') - get('jawRight'))
    delta[0, 19, 1] += 0.012 * (get('mouthShrugLower') + get('mouthUpperUpLeft') + get('mouthUpperUpRight'))
    # Eye apertures from the named ARKit controls share the same local keypoints
    # as the existing blink sliders; target-driven transfer remains independent.
    delta[0, (3, 11, 13), 1] += .012 * blink_left
    delta[0, (7, 15, 16), 1] += .012 * blink_right
    return Motion(canonical, delta, source.translation, source.scale,
                  tuple(a+b for a,b in zip(source.angles, (edit.head_pitch, edit.head_yaw, edit.head_roll))))


def close_apertures(image, points, left=False, right=False, lips=False):
    """Cover an aperture with adjacent identity skin, retaining a one-pixel lid/lip seam.

    Runs AFTER enhancement too, so a restorer cannot reopen an enforced blink.
    No target pixels are pasted into the swapped identity.
    """
    output = image.copy()
    for indices, active in ((range(36, 42), left), (range(42, 48), right), (range(60, 68), lips)):
        if not active:
            continue
        p = np.asarray(points)[list(indices)]
        centre = p.mean(axis=0)
        axis = p[np.argmax(p[:, 0])] - p[np.argmin(p[:, 0])]
        width = float(np.linalg.norm(axis))
        if width < 2:
            continue
        across = axis / width
        down = np.array([-across[1], across[0]])
        v = (p-centre) @ down
        radius = max(float(np.max(np.abs(v)))+2, 2)
        yy, xx = np.mgrid[:image.shape[0], :image.shape[1]].astype(np.float32)
        u = (xx-centre[0])*across[0]+(yy-centre[1])*across[1]
        d = (xx-centre[0])*down[0]+(yy-centre[1])*down[1]
        mask = (np.abs(u) <= width*.52) & (np.abs(d) <= radius)
        # Sample skin immediately outside the upper/lower aperture boundary.
        shift = np.where(d < 0, -radius-2-d, radius+2-d)
        skin = cv2.remap(output, (xx+down[0]*shift).astype(np.float32),
                         (yy+down[1]*shift).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        seam = np.abs(d) < .6
        skin[seam] = np.rint(skin[seam].astype(np.float32)*.65).astype(np.uint8)
        output[mask] = skin[mask]
    return output


class ExpressionRestorer:
    def __init__(self, catalog=None, config=RuntimeConfig(), *, portrait=None, geometry=None):
        self.catalog, self.config = catalog, config
        self.portrait, self.geometry = portrait, geometry
        self._owns_portrait = portrait is None
        self.last_report = {}

    def prepare(self):
        if self.portrait is None:
            portrait = LivePortraitAdapter(self.catalog, self.config)
            try:
                portrait.prepare_session(required=True)
            except Exception:
                portrait.release_session()
                raise
            self.portrait = portrait
        try:
            if 'eye' not in self.portrait.models:
                raise DomainError('EXPRESSION_MODEL_MISSING', 'Install LivePortrait eye retargeting weights', 503)
            self.geometry = self.geometry or PortraitGeometry(self.portrait.catalog, self.config)
        except Exception:
            if self._owns_portrait:
                self.portrait.release_session()
                self.portrait = None
            raise

    def restore(self, swapped, target, *, factor=0., edit=None):
        edit = edit or FaceEditor()
        if not np.isfinite(factor) or not 0 <= factor <= 1:
            raise ValueError('expression_restore_factor must be within 0..1')
        if factor == 0 and not edit.active():
            return swapped
        self.prepare()
        image_bgr(swapped); image_bgr(target)
        source, driving = self.portrait.motion(swapped), self.portrait.motion(target)
        moved = edit_motion(source, source.expression + factor*(driving.expression-source.expression), edit)
        points_source, points = source.points(), moved.points()
        source68, target68 = self.geometry.landmarks(swapped), self.geometry.landmarks(target)
        ears = [eye_aspect_ratio(target68[i:i+6]) for i in (36, 42)]
        # Strict threshold and independent eyes: never average a wink into a half-blink.
        arkit = edit.arkit_blendshapes
        closed = [bool(factor > 0 and ear < .16) or blink >= 1 for ear, blink in
                  zip(ears, (max(edit.blink_left, arkit.get('eyeBlinkLeft', 0.0)),
                             max(edit.blink_right, arkit.get('eyeBlinkRight', 0.0))))]
        source_ears = [eye_aspect_ratio(source68[i:i+6]) for i in (36, 42)]
        # The export has one driving ratio; evaluate each eye independently and
        # restrict its residual to that side's implicit eye/lid/cheek points.
        for i, (blink, ids) in enumerate(((max(edit.blink_left, arkit.get('eyeBlinkLeft', 0.0)), (11, 13, 3)),
                                           (max(edit.blink_right, arkit.get('eyeBlinkRight', 0.0)), (15, 16, 7)))):
            ratio = (factor*ears[i]+(1-factor)*source_ears[i])*(1-blink)
            features = np.array([list(points_source.reshape(-1))+source_ears+[0. if closed[i] else ratio]], np.float32)
            eye = self.portrait.models['eye'].run({'input': features})['output'].reshape(1, 21, 3)
            tensor(eye.astype(np.float32), (1, 21, 3))
            points[:, ids, :] += max(factor, blink)*eye[:, ids, :]
        def lip_ratio(p):
            return float(np.linalg.norm(p[62]-p[66])/max(np.linalg.norm(p[48]-p[54]), 1e-5))
        arkit_open = max(arkit.get('jawOpen', 0.0), arkit.get('mouthLowerDownLeft', 0.0),
                         arkit.get('mouthLowerDownRight', 0.0), 0.35 * arkit.get('tongueOut', 0.0))
        arkit_close = max(arkit.get('mouthClose', 0.0), arkit.get('mouthPressLeft', 0.0), arkit.get('mouthPressRight', 0.0))
        edit_open = float(np.clip(max(edit.mouth_open, arkit_open) - arkit_close, 0.0, 1.0))
        lip_closed = factor > 0 and lip_ratio(target68) < .02 and edit_open == 0
        lip_target = max(edit_open*.5, factor*lip_ratio(target68)+(1-factor)*lip_ratio(source68))
        features = np.array([list(points_source.reshape(-1))+[lip_ratio(source68), 0. if lip_closed else lip_target]], np.float32)
        if 'lip' in self.portrait.models:
            lip = self.portrait.models['lip'].run({'input': features})['output'].reshape(1, 21, 3)
            points += max(factor, edit_open)*tensor(lip.astype(np.float32), (1, 21, 3))
        correction = self.portrait.models['stitch'].run({'input': np.concatenate((points_source.reshape(1, 63), points.reshape(1, 63)), axis=1)})['output']
        tensor(correction, (1, 65))
        points += correction[:, :63].reshape(1, 21, 3)
        points[:, :, :2] += correction[:, 63:].reshape(1, 1, 2)
        feature = self.portrait.models['appearance'].run({'img': portrait_tensor(swapped)})['output']
        tensor(feature, (1, 32, 16, 64, 64))
        generated = self.portrait.models['warp_spade'].run({'feature_3d': feature, 'kp_source': points_source,
                                                        'kp_driving': np.ascontiguousarray(points, dtype=np.float32)})['out']
        tensor(generated, (1, 3, 512, 512))
        output = np.clip(generated[0].transpose(1, 2, 0)[:, :, ::-1]*255, 0, 255).round().astype(np.uint8)
        output = cv2.resize(output, (swapped.shape[1], swapped.shape[0]))
        self.last_report = {'target_ear': ears, 'closed_left': closed[0], 'closed_right': closed[1], 'closed_lips': lip_closed,
                            'factor': factor, 'identity_canonical_locked': not (edit.jaw_width or edit.chin_length),
                            'gaze_mode': 'implicit_keypoint_edit', 'learned_lip_retargeting': 'lip' in self.portrait.models,
                            'pose_locked': not (edit.head_pitch or edit.head_yaw or edit.head_roll)}
        return self.enforce(output)

    def enforce(self, image):
        r = self.last_report
        if not any(r.get(k) for k in ('closed_left', 'closed_right', 'closed_lips')):
            return image
        return close_apertures(image, self.geometry.landmarks(image), r['closed_left'], r['closed_right'], r['closed_lips'])

    def generate_pose(self, image, *, yaw, pitch):
        """Absolute neural pose for the source identity; no driving identity geometry."""
        self.prepare()
        source = self.portrait.motion(image)
        points = Motion(source.canonical, source.expression, source.translation, source.scale, (pitch, yaw, 0.)).points()
        original = source.points()
        correction = self.portrait.models['stitch'].run({'input': np.concatenate((original.reshape(1, 63), points.reshape(1, 63)), axis=1)})['output']
        tensor(correction, (1, 65))
        points += correction[:, :63].reshape(1, 21, 3)
        points[:, :, :2] += correction[:, 63:].reshape(1, 1, 2)
        feature = self.portrait.models['appearance'].run({'img': portrait_tensor(image)})['output']
        tensor(feature, (1, 32, 16, 64, 64))
        output = self.portrait.models['warp_spade'].run({'feature_3d': feature, 'kp_source': original,
                                                      'kp_driving': np.ascontiguousarray(points, dtype=np.float32)})['out']
        tensor(output, (1, 3, 512, 512))
        return np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1]*255, 0, 255).round().astype(np.uint8)

    def release(self):
        if self.portrait is not None and self._owns_portrait:
            self.portrait.release_session()
        if self.geometry is not None:
            self.geometry.release()
        self.portrait = self.geometry = None
        self.last_report = {}
