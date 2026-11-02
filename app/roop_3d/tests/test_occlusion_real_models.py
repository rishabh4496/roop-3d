"""Stage 5 on REAL local inference: real BiSeNet, real XSeg / face-occluder networks, real tracker, real video.

Skipped, with a reason, when the local models or the local videos are not installed; it never fakes inference.
No identity transfer happens here: the replacement is the target face's own aligned crop with a green tint (see
``roop_3d.tools.occlusion_acceptance``), so every tinted output pixel is a pixel the compositor chose to paint.
Only face regions are analysed; nothing leaves the machine.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2

import numpy as np
import pytest

from roop_3d.compositor.face_regions import build_regions
from roop_3d.compositor.masks import BisenetFaceParser
from roop_3d.compositor.occlusion_model import VisibleFaceNet
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor, RobustOptions
from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.models.catalog import ModelCatalog
from roop_3d.tools import occlusion_acceptance as acceptance

CONFIG = RuntimeConfig(cpu_threads=4)


@pytest.fixture(scope="module")
def real():
    try:
        catalog = ModelCatalog()
        catalog.detector()
        nets = [VisibleFaceNet(catalog, CONFIG, model_name=name) for name in ("xseg_3.onnx", "face_occluder.onnx")]
        if not all(net.available() for net in nets):
            pytest.skip("visible-face networks are not installed locally")
        BisenetFaceParser(catalog, CONFIG)._load()
        tracker = FaceTracker(catalog, config=CONFIG)
        film = acceptance._video("03de")
    except (DomainError, OSError, KeyError, FileNotFoundError) as error:
        pytest.skip(f"local models or video unavailable: {error}")
    yield SimpleNamespace(catalog=catalog, tracker=tracker, film=film)
    tracker.release()


def faces_at(real, path, start, count):
    frames = acceptance.read_clip(path, start, count)
    if len(frames) < count:
        pytest.skip("local video is shorter than expected")
    real.tracker.mark_scene_cut()
    observed, cuts = acceptance.track_clip(real.tracker, frames)
    if not all(observed):
        pytest.skip("no trackable face in the selected local clip")
    return frames, observed, cuts


def work_crop(frame, face, size=256):
    replacement = acceptance.probe(frame, face)
    inverse = FrameCompositor._scaled_inverse(replacement.inverse, size / replacement.image.shape[0])
    return FrameCompositor._crop_to_frame(frame, inverse, (size, size))


def test_visible_face_networks_run_real_inference_and_agree_on_a_frontal_face(real):
    frames, observed, _ = faces_at(real, real.film, 600, 1)
    crop = work_crop(frames[0], observed[0][0])
    maps = []
    for name in ("xseg_3.onnx", "face_occluder.onnx"):
        net = VisibleFaceNet(real.catalog, CONFIG, model_name=name)
        visible = net.predict(crop)
        assert net.inference_verified and net.capability()["inference"] == "VERIFIED_LOCAL"
        assert visible.shape == crop.shape[:2] and 0.0 <= visible.min() and visible.max() <= 1.0
        assert 0.15 < float((visible > 0.5).mean()) < 0.8             # a face-sized region, not empty or everything
        maps.append(visible > 0.5)
        net.release()
    iou = (maps[0] & maps[1]).sum() / (maps[0] | maps[1]).sum()
    assert iou > 0.75


def test_bisenet_regions_are_anatomically_ordered_on_a_real_face(real):
    frames, observed, _ = faces_at(real, real.film, 600, 1)
    crop = work_crop(frames[0], observed[0][0])
    parser = BisenetFaceParser(real.catalog, CONFIG, normalization="imagenet")
    regions = build_regions(parser.predict(crop))
    ys = np.arange(crop.shape[0], dtype=np.float32)[:, None]
    centroid = lambda m: float((m * ys).sum() / max(float(m.sum()), 1.0))
    assert regions["skin"].sum() > regions["eyes"].sum() * 20
    assert centroid(regions["brows"]) < centroid(regions["eyes"]) < centroid(regions["nose"]) < centroid(regions["lips"])
    assert regions["nose"].sum() > 20 and regions["lips"].sum() > 20
    parser.release()


def test_imagenet_preprocessing_agrees_with_the_networks_better_than_the_legacy_default(real):
    """Evidence for using imagenet normalisation in the robust path (the standard path keeps zero_one unchanged)."""
    net = VisibleFaceNet(real.catalog, CONFIG, model_name="xseg_3.onnx")
    scores = {"zero_one": [], "imagenet": []}
    parsers = {mode: BisenetFaceParser(real.catalog, CONFIG, normalization=mode) for mode in scores}
    for start in (600, 900, 3000):
        frames, observed, _ = faces_at(real, real.film, start, 1)
        crop = work_crop(frames[0], observed[0][0])
        reference = net.predict(crop) > 0.5
        for mode, parser in parsers.items():
            face = build_regions(parser.predict(crop)).face_interior > 0.5
            scores[mode].append((face & reference).sum() / max((face | reference).sum(), 1))
    assert np.mean(scores["imagenet"]) >= np.mean(scores["zero_one"]) - 0.02, scores
    assert np.mean(scores["imagenet"]) > 0.7
    for parser in parsers.values():
        parser.release()
    net.release()


def _painted_stats(frames, outputs, observed, compositor):
    """Fraction of interior pixels BOTH networks call occluded that the compositor still painted."""
    painted_total, occluded_total = 0, 0
    for frame, out, row in zip(frames, outputs, observed):
        mask = acceptance.painted(frame, out)
        for face in row:
            _, occ = acceptance.reference_regions(compositor, frame, acceptance.probe(frame, face), face.landmarks106)
            occluded_total += int(occ.sum())
            painted_total += int((mask & occ).sum())
    return painted_total / max(occluded_total, 1), occluded_total


def test_robust_compositor_protects_hands_on_real_footage_where_standard_paints_over_them(real):
    try:
        path = acceptance._video("5256")
    except FileNotFoundError:
        pytest.skip("local clip with hands in front of a face is unavailable")
    frames, observed, cuts = faces_at(real, path, 10800, 3)
    standard = FrameCompositor(real.catalog, CONFIG)
    robust = FrameCompositor(real.catalog, CONFIG)
    try:
        s_opts = CompositeOptions(color_strength=0, feather_px=5)
        r_opts = CompositeOptions(color_strength=0, feather_px=5, occlusion_mode="robust",
                                  robust=RobustOptions(local_illumination=0, texture_strength=0))
        outs = {"standard": [], "robust": []}
        for index, (frame, row) in enumerate(zip(frames, observed)):
            replacements = [acceptance.probe(frame, face) for face in row]
            landmarks = {face.track_id: face.landmarks106 for face in row}
            for name, comp, opts in (("standard", standard, s_opts), ("robust", robust, r_opts)):
                out = comp.compose(frame, replacements, landmarks, scene_cut=cuts[index], frame_index=index, options=opts)
                assert out.dtype == np.uint8 and out.shape == frame.shape and np.isfinite(out).all()
                outs[name].append(out)
        painted_standard, occluded = _painted_stats(frames, outs["standard"], observed, robust)
        painted_robust, _ = _painted_stats(frames, outs["robust"], observed, robust)
        assert occluded > 500, "the clip must actually contain occluded face pixels for this test to mean anything"
        assert painted_standard > 0.5 and painted_robust < 0.1
        report = robust.last_reports[observed[-1][0].track_id]
        assert report["tier"] in {"full", "neural"} and report["networks"] == ["xseg_3.onnx", "face_occluder.onnx"]
        assert report["fusion"]["occluded_fraction"] > 0.05
        assert robust.capability()["robust_occlusion"]["parser"]["inference"] == "VERIFIED_LOCAL"
    finally:
        standard.release()
        robust.release()


def test_robust_compositor_survives_a_real_profile_that_defeats_the_parser(real):
    frames, observed, cuts = faces_at(real, real.film, 6750, 3)
    robust = FrameCompositor(real.catalog, CONFIG)
    try:
        options = CompositeOptions(color_strength=0, feather_px=5, occlusion_mode="robust",
                                   robust=RobustOptions(local_illumination=0, texture_strength=0))
        painted = []
        for index, (frame, row) in enumerate(zip(frames, observed)):
            replacements = [acceptance.probe(frame, face) for face in row]
            out = robust.compose(frame, replacements, {f.track_id: f.landmarks106 for f in row},
                                 scene_cut=cuts[index], frame_index=index, options=options)
            painted.append(int(acceptance.painted(frame, out).sum()))
        assert min(painted) > 200, "a profile must still be composited, not silently dropped"
        assert all(report["confidence"] >= 0.0 for report in robust.last_reports.values())
    finally:
        robust.release()


def test_occlusion_tracker_detects_a_real_shot_change_from_pixels_alone(real):
    """The dangerous case: a face is detected at a similar place after a cut and keeps its track id."""
    from roop_3d.compositor.occlusion_tracker import OcclusionTracker

    # A real shot change with a face on both sides (cut = first frame of the new shot).
    for cut in acceptance.find_scene_cuts(real.film, 11700, 11900):
        frames = acceptance.read_clip(real.film, cut - 4, 6)       # 4 frames before the cut, the cut frame, 1 after
        real.tracker.mark_scene_cut()
        observed, _ = acceptance.track_clip(real.tracker, frames[:4])
        if observed[-1]:
            break
    else:
        pytest.skip("no shot change with a face before it in this local video")
    face = max(observed[-1], key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
    inverse = FrameCompositor._scaled_inverse(acceptance.probe(frames[3], face).inverse, 2.0)
    crops = [FrameCompositor._crop_to_frame(frame, inverse, (256, 256)) for frame in frames]   # same geometry on both sides
    occluder = np.zeros((256, 256), np.float32)
    occluder[100:160, 100:160] = 0.5
    tracker = OcclusionTracker()
    reasons = [tracker.update(1, occluder, crop, frame_index=index).reset_reason for index, crop in enumerate(crops)]
    assert reasons[:4] == [None, None, None, None], f"ordinary real frames must not reset history: {reasons}"
    assert reasons[4] == "APPEARANCE_CUT", reasons


def test_restoration_guard_protects_eyes_and_mouth_with_the_real_gpen_restorer(real):
    """Real GPEN on a real face: protected pixels (eyes, brows, lips, mouth interior, teeth) must not change."""
    from roop_3d.compositor.restoration import GpenRestorer

    restorer = GpenRestorer(real.catalog, CONFIG)
    try:
        restorer._load()
    except (DomainError, OSError, KeyError, RuntimeError) as error:
        pytest.skip(f"GPEN weights unavailable: {error}")
    frames, observed, _ = faces_at(real, real.film, 600, 1)
    frame, face = frames[0], observed[0][0]
    probe = acceptance.probe(frame, face)
    landmarks = {face.track_id: face.landmarks106}
    outputs = {}
    for name, guard in (("none", None), ("guarded", True), ("unguarded", False)):
        compositor = FrameCompositor(real.catalog, CONFIG, restorer=restorer)
        blend = 0.0 if guard is None else 1.0
        options = CompositeOptions(color_strength=0, feather_px=3, restoration_blend=blend, occlusion_mode="robust",
                                   robust=RobustOptions(local_illumination=0, texture_strength=0, sharpness_matching=False,
                                                        restoration_guard=bool(guard)))
        try:
            for index in range(3):                                   # parser trust (hence the protection map) is earned over 3 frames
                outputs[name] = compositor.compose(frame, [probe], landmarks, frame_index=index, options=options)
            if name == "guarded":
                analysis = compositor.analyzer.analyze(
                    FrameCompositor._crop_to_frame(frame, FrameCompositor._scaled_inverse(probe.inverse, 2.0), (256, 256)),
                    FrameCompositor._landmarks_to_crop(face.landmarks106, FrameCompositor._scaled_inverse(probe.inverse, 2.0)),
                    feather_px=10)
        finally:
            compositor.release()
    if analysis.protect is None:
        pytest.skip("parser was not trusted on this frame, so no protection map exists to verify")
    inverse = FrameCompositor._scaled_inverse(probe.inverse, 2.0)
    shape = (frame.shape[1], frame.shape[0])
    protect = cv2.warpAffine((analysis.protect > 0.5).astype(np.uint8), inverse, shape, flags=cv2.INTER_NEAREST) > 0
    skin = cv2.warpAffine((analysis.skin > 0.5).astype(np.uint8), inverse, shape, flags=cv2.INTER_NEAREST) > 0
    assert protect.sum() > 200 and skin.sum() > 1000
    delta = lambda a, b, region: float(np.abs(outputs[a].astype(np.float32) - outputs[b].astype(np.float32))[region].mean())
    core = cv2.erode(protect.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0         # inside the protected parts, not their soft rim
    guarded, unguarded = delta("guarded", "none", core), delta("unguarded", "none", core)
    restored_skin = delta("guarded", "none", skin & ~protect)
    # the restorer rewrites eyes/mouth when unguarded, barely touches them when guarded, and still restores skin
    assert unguarded > 3.0, (guarded, unguarded, restored_skin)
    assert guarded < 0.35 * unguarded, (guarded, unguarded, restored_skin)
    assert restored_skin > 1.0, (guarded, unguarded, restored_skin)
