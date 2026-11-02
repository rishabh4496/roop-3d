"""Stage 5 acceptance harness: standard vs robust compositor on REAL local video.

Rules this harness follows (and reports):

* **No identity transfer.** The replacement is an identity-neutral *probe*: the target face's own
  aligned crop with a strong green tint, composited with colour matching disabled. Any tinted pixel in
  the output is, by construction, a pixel the compositor chose to paint, so leakage onto hands, hair,
  glasses, props and background is directly measurable without inventing a "ground truth".
* **Real inference only.** Face tracking, BiSeNet and the visible-face networks are the local ONNX
  models. Nothing is simulated. Scenarios without real footage are reported as ``NOT_COVERED``.
* **Metrics are proxies, not ground truth.** The face-region reference uses the same families of models
  the robust path uses, so ``leak`` and ``overpaint`` measure *self-consistency end to end* (matte
  refinement, temporal logic, fusion, depth ordering), not an independent pixel-accurate truth. The
  flow-compensated ``flicker`` metric needs no segmentation at all. Montages exist for human review.
* Only non-explicit local clips are listed below; the harness never writes full frames, only face
  regions.

Run from ``app/``:  ``python -m roop_3d.tools.occlusion_acceptance --out roop_3d/.runtime/acceptance/occlusion_stage5``
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor, RobustOptions
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.preprocess import aligned_crop
from roop_3d.models.catalog import ModelCatalog

MEDIA = application_root() / "roop_3d/.runtime/media"
TINT = np.array((0.45, 1.35, 0.55), np.float32)          # BGR gain: unmistakably green
PAINT_DELTA = 14                                         # max-channel difference that counts as "painted"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    candidates: tuple[tuple[str, int], ...] = ()          # (media-file prefix, first frame); empty = no real footage
    select: str = "first"                                 # first | fastest | darkest
    frames: int = 10


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("hand_over_face", "Folded hands / fingers held in front of a face", (("5256", 10800),)),
    Scenario("hair_in_front_of_face", "Hair strands crossing cheeks and eyes", (("5256", 13200),)),
    Scenario("glasses", "Spectacles / tinted glasses on a face", (("5256", 7800), ("5256", 3900))),
    Scenario("fast_movement", "Largest measured face displacement among candidate clips",
             (("03de", 10650), ("03de", 11250), ("5256", 3900), ("5256", 15000)), select="fastest"),
    Scenario("microphone_crossing_mouth", "A microphone moving across the mouth", ()),
    Scenario("multiple_interacting_subjects", "Two faces in close contact", (("5256", 14400), ("5256", 7800))),
    Scenario("dark_lighting", "Darkest face crop among candidate clips",
             (("5256", 24450), ("5256", 14400), ("03de", 6750)), select="darkest"),
    Scenario("side_profile", "Face in profile", (("03de", 6750), ("5256", 150), ("5256", 3600))),
    Scenario("motion_blur", "Motion-blurred face", (("03de", 10650), ("03de", 5700))),
    Scenario("rapid_scene_transition", "A real shot change inside the clip", (("03de", -1),)),
)


def _video(prefix: str) -> Path:
    for path in sorted(MEDIA.glob("*.mp4")):
        if path.name.startswith(prefix):
            return path
    raise FileNotFoundError(prefix)


def read_clip(path: Path, start: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


def find_scene_cuts(path: Path, first: int = 0, last: int = 12000, threshold: float = 0.5) -> list[int]:
    """Frame indices (first frame of the new shot) whose correlation with the previous frame is below ``threshold``,
    most abrupt first. Fully black frames (correlation 0 with a flat frame) are ignored."""
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    previous, found = None, []
    for index in range(first, last):
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous is not None and gray.std() > 2.0 and previous.std() > 2.0:
            a, b = previous - previous.mean(), gray - gray.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
            if ncc < threshold:
                found.append((ncc, index))
        previous = gray
    capture.release()
    return [index for _, index in sorted(found)]


def find_scene_cut(path: Path, first: int = 0, last: int = 12000) -> int | None:
    cuts = find_scene_cuts(path, first, last)
    return cuts[0] if cuts else None


@dataclass
class Observed:
    track_id: int
    landmarks5: np.ndarray
    landmarks106: np.ndarray
    box: tuple[float, float, float, float]


def track_clip(tracker: FaceTracker, frames: list[np.ndarray]) -> tuple[list[list[Observed]], list[bool]]:
    observed, cuts = [], []
    for frame in frames:
        faces = tracker.process(frame)
        cuts.append(bool(tracker.scene_cut))
        row = []
        for tracked in faces:
            obs = tracked.observation
            if obs.landmarks5 is None or obs.landmarks106 is None:
                continue
            row.append(Observed(tracked.track_id, np.asarray(obs.landmarks5, np.float64), np.asarray(obs.landmarks106, np.float32),
                                tuple(float(v) for v in obs.box)))
        observed.append(row)
    return observed, cuts


def probe(frame: np.ndarray, face: Observed) -> SimpleNamespace:
    aligned = aligned_crop(frame, face.landmarks5, 128)
    tinted = np.clip(aligned.image.astype(np.float32) * TINT, 0, 255).astype(np.uint8)
    return SimpleNamespace(track_id=face.track_id, image=tinted, inverse=np.asarray(aligned.inverse, np.float32),
                           landmarks5=face.landmarks5.astype(np.float32))


def natural_replacement(frame: np.ndarray, face: Observed, reference_frame: np.ndarray, reference: Observed | None) -> SimpleNamespace:
    """Identity-neutral but realistic replacement: the SAME person's aligned crop from another moment of the clip
    (different lighting / expression / sharpness), placed with the current frame's geometry."""
    rep = probe(frame, face)
    if reference is not None:
        rep.image = aligned_crop(reference_frame, reference.landmarks5, 128).image
    return rep


def natural_run(frames, observed, cuts, options, catalog, config):
    """Compose with colour matching on and a different-moment crop; returns the outputs."""
    compositor = FrameCompositor(catalog, config)
    outputs = []
    first = {face.track_id: face for face in observed[0]}
    for index, (frame, row) in enumerate(zip(frames, observed)):
        replacements = [natural_replacement(frame, face, frames[0], first.get(face.track_id)) for face in row]
        outputs.append(compositor.compose(frame, replacements, {f.track_id: f.landmarks106 for f in row}, scene_cut=cuts[index],
                                          frame_index=index, options=options))
    compositor.release()
    return outputs


def natural_montage(path: Path, frames, observed, standard_out, robust_out, indices):
    rows = []
    for index in indices:
        row = observed[index]
        if not row:
            continue
        face = max(row, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        x0, y0, x1, y1 = face.box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.95
        X0, Y0, X1, Y1 = (int(max(0, v)) for v in (cx - half, cy - half, cx + half, cy + half))
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            continue
        tile = lambda im: cv2.resize(im[Y0:Y1, X0:X1], (320, 320), interpolation=cv2.INTER_CUBIC)
        panels = [tile(frames[index]), tile(standard_out[index]), tile(robust_out[index])]
        for panel, name in zip(panels, ("original", "standard", "robust (quality on)")):
            cv2.putText(panel, f"{name} f{index}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))
    return bool(rows)


def clip_speed(observed) -> float:
    centres = {}
    speeds = []
    for row in observed:
        for face in row:
            c = np.array(((face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2))
            if face.track_id in centres:
                speeds.append(float(np.linalg.norm(c - centres[face.track_id])))
            centres[face.track_id] = c
    return float(np.mean(speeds)) if speeds else 0.0


def clip_darkness(frames, observed) -> float:
    values = []
    for frame, row in zip(frames, observed):
        for face in row:
            x0, y0, x1, y1 = (int(max(0, v)) for v in face.box)
            patch = frame[y0:y1, x0:x1]
            if patch.size:
                values.append(float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()))
    return float(np.mean(values)) if values else 255.0


def painted(frame: np.ndarray, out: np.ndarray) -> np.ndarray:
    return (np.abs(out.astype(np.int16) - frame.astype(np.int16)).max(axis=2) > PAINT_DELTA)


def reference_regions(compositor: FrameCompositor, frame, replacement, landmarks106):
    """Frame-space (reference face region, occluded-by-both-networks) masks for one face, via the analyser's models."""
    work = 256
    scale = work / replacement.image.shape[0]
    inverse = FrameCompositor._scaled_inverse(replacement.inverse, scale)
    crop = FrameCompositor._crop_to_frame(frame, inverse, (work, work))
    points = FrameCompositor._landmarks_to_crop(landmarks106, inverse)
    analyzer = compositor.analyzer
    visible = [net.predict(crop) for net in analyzer.nets if net.available()]
    from roop_3d.compositor.face_regions import build_regions
    from roop_3d.compositor.masks import landmark_hull_mask
    face = np.zeros((work, work), np.float32)
    for v in visible:
        face = np.maximum(face, v)
    try:
        face = np.maximum(face, build_regions(analyzer.parser.predict(crop)).face_interior)
    except Exception:
        pass
    face = cv2.dilate((face > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8))
    hull = landmark_hull_mask((work, work), points, feather_px=0) > 0.5
    # Interior only: at the hull border the networks' tighter outline is not an occluder.
    hull = cv2.erode(hull.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(work * 0.08) + 1,) * 2)) > 0
    both = hull & (np.minimum.reduce([1 - v for v in visible]) > 0.5) if visible else np.zeros((work, work), bool)
    shape = frame.shape[1], frame.shape[0]
    ref = cv2.warpAffine(face, inverse, shape, flags=cv2.INTER_NEAREST) > 0
    occ = cv2.warpAffine(both.astype(np.uint8), inverse, shape, flags=cv2.INTER_NEAREST) > 0
    return ref, occ


def flow_warp(previous_mask: np.ndarray, previous: np.ndarray, current: np.ndarray, flow_engine) -> np.ndarray:
    g0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = flow_engine.calc(g1, g0, None)
    height, width = g1.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(previous_mask.astype(np.float32), grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def run_mode(label, frames, observed, cuts, options, catalog, config, reference=None):
    compositor = FrameCompositor(catalog, config)
    captured: list[dict] = []
    outputs, timings, reports = [], [], []
    for index, (frame, row) in enumerate(zip(frames, observed)):
        replacements = [probe(frame, face) for face in row]
        landmarks = {face.track_id: face.landmarks106 for face in row}
        grabbed: dict[int, np.ndarray] = {}

        def observer(name, image, inverse, alpha, _g=grabbed):
            if name == "occluder":
                _g[len(_g)] = cv2.warpAffine(np.asarray(image, np.float32), np.asarray(inverse, np.float32),
                                             (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

        began = time.perf_counter()
        out = compositor.compose(frame, replacements, landmarks, scene_cut=cuts[index], frame_index=index, options=options,
                                 observer=observer)
        timings.append((time.perf_counter() - began) * 1000 / max(1, len(replacements)))
        outputs.append(out)
        captured.append(grabbed)
        reports.append({str(k): v for k, v in compositor.last_reports.items()})
    compositor.release()
    return {"label": label, "outputs": outputs, "ms_per_face": timings, "mattes": captured, "reports": reports}


def evaluate(frames, observed, runs, reference_compositor):
    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    refs = []
    for frame, row in zip(frames, observed):
        ref = np.zeros(frame.shape[:2], bool)
        occ = np.zeros(frame.shape[:2], bool)
        for face in row:
            r, o = reference_regions(reference_compositor, frame, probe(frame, face), face.landmarks106)
            ref |= r
            occ |= o
        refs.append((ref, occ))
    result = {}
    for run in runs:
        leak, overpaint, flicker, flicker_abs, area = [], [], [], [], []
        previous_mask = None
        for index, (frame, out) in enumerate(zip(frames, run["outputs"])):
            mask = painted(frame, out)
            ref, occ = refs[index]
            total = int(mask.sum())
            area.append(total)
            if total:
                leak.append(float((mask & ~ref).sum() / total))
            if occ.any():
                overpaint.append(float((mask & occ).sum() / occ.sum()))
            if previous_mask is not None and total:
                warped = flow_warp(previous_mask, frames[index - 1], frame, flow_engine)
                change = float(np.abs(mask.astype(np.float32) - warped).sum())
                flicker.append(change / total)
                # Normalised by the (mode-independent) face-region area: a mode that paints less is not penalised
                # for the same absolute boundary change.
                flicker_abs.append(change / max(int(ref.sum()), 1))
            previous_mask = mask
        mean = lambda v: round(float(np.mean(v)), 4) if v else None
        result[run["label"]] = {"leak_outside_face": mean(leak), "occluded_pixels_painted": mean(overpaint),
                                "flicker": mean(flicker), "flicker_vs_face_area": mean(flicker_abs), "painted_px_mean": mean(area),
                                "ms_per_face": round(float(np.mean(run["ms_per_face"])), 1)}
    return result, refs


def montage(path: Path, frames, observed, runs, indices):
    rows = []
    for index in indices:
        row = observed[index]
        if not row:
            continue
        face = max(row, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        x0, y0, x1, y1 = face.box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.95
        X0, Y0, X1, Y1 = (int(max(0, v)) for v in (cx - half, cy - half, cx + half, cy + half))
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            continue
        tile = lambda im: cv2.resize(im[Y0:Y1, X0:X1], (256, 256), interpolation=cv2.INTER_AREA)
        matte = runs[1]["mattes"][index]
        front = np.zeros(frames[index].shape[:2], np.float32)
        for value in matte.values():
            front = np.maximum(front, value)
        panels = [tile(frames[index]), tile(runs[0]["outputs"][index]), tile(runs[1]["outputs"][index]),
                  tile(cv2.cvtColor((np.clip(front, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))]
        for panel, name in zip(panels, ("original", "standard", "robust", "robust occluder")):
            cv2.putText(panel, f"{name} f{index}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))
    return bool(rows)


def run(out_dir: Path, *, provider: str = "CPUExecutionProvider", names: tuple[str, ...] | None = None,
        frames_per_clip: int | None = None, robust_options: RobustOptions | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (RuntimeConfig(cpu_threads=4) if provider == "CPUExecutionProvider"
              else RuntimeConfig(provider=provider, hardware_profile="12gb"))
    catalog = ModelCatalog()
    tracker = FaceTracker(catalog, config=config)
    standard = CompositeOptions(color_strength=0, feather_px=5)
    robust = CompositeOptions(color_strength=0, feather_px=5, occlusion_mode="robust",
                              robust=robust_options or RobustOptions(local_illumination=0.0, texture_strength=0.0))
    summary = {"provider": provider, "probe": "target's own aligned crop, green tint, colour matching off",
               "paint_delta": PAINT_DELTA, "scenarios": {}}
    reference_compositor = FrameCompositor(catalog, config)
    for scenario in SCENARIOS:
        if names and scenario.name not in names:
            continue
        count = frames_per_clip or scenario.frames
        entry = {"description": scenario.description}
        if not scenario.candidates:
            entry["status"] = "NOT_COVERED"
            entry["reason"] = "no real footage of this event was found in the local media; synthetic unit tests only"
            summary["scenarios"][scenario.name] = entry
            continue
        chosen = None
        starts = []
        for prefix, start in scenario.candidates:
            try:
                path = _video(prefix)
            except FileNotFoundError:
                continue
            if start < 0:                        # a real shot change must sit in the middle of the clip
                starts.extend((prefix, max(0, cut - count // 2), cut) for cut in find_scene_cuts(path, 0, 12000))
            else:
                starts.append((prefix, start, None))
        for prefix, start, cut in starts:
            path = _video(prefix)
            clip = read_clip(path, start, count)
            if len(clip) < count:
                continue
            tracker.mark_scene_cut()
            observed, cuts = track_clip(tracker, clip)
            if not any(observed):
                continue
            if cut is not None and not (observed[cut - start - 1] and observed[cut - start]):
                continue                         # need a face on both sides of the shot change
            score = {"first": 0.0, "fastest": clip_speed(observed), "darkest": -clip_darkness(clip, observed)}[scenario.select]
            candidate = (score, prefix, start, clip, observed, cuts)
            if chosen is None or candidate[0] > chosen[0]:
                chosen = candidate
            if scenario.select == "first":
                break
        if chosen is None:
            entry["status"] = "NOT_COVERED"
            entry["reason"] = "candidate clips contained no trackable face"
            summary["scenarios"][scenario.name] = entry
            continue
        _, prefix, start, clip, observed, cuts = chosen
        runs = [run_mode("standard", clip, observed, cuts, standard, catalog, config),
                run_mode("robust", clip, observed, cuts, robust, catalog, config)]
        metrics, _ = evaluate(clip, observed, runs, reference_compositor)
        tiers: dict[str, int] = {}
        reasons: dict[str, int] = {}
        resets: dict[str, int] = {}
        for per_frame in runs[1]["reports"]:
            for report in per_frame.values():
                tiers[report["tier"]] = tiers.get(report["tier"], 0) + 1
                for reason in report["reasons"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
                if report["temporal"]["reset"]:
                    resets[report["temporal"]["reset"]] = resets.get(report["temporal"]["reset"], 0) + 1
        picture = out_dir / f"{scenario.name}.png"
        natural = out_dir / f"{scenario.name}_natural.png"
        nat_std = natural_run(clip, observed, cuts, CompositeOptions(color_strength=0.65, feather_px=5), catalog, config)
        nat_rob = natural_run(clip, observed, cuts, CompositeOptions(color_strength=0.65, feather_px=5, occlusion_mode="robust"), catalog, config)
        natural_ok = natural_montage(natural, clip, observed, nat_std, nat_rob, sorted({0, count // 2, count - 1}))
        entry.update({"status": "TESTED_ON_REAL_FOOTAGE", "clip": {"media": prefix, "first_frame": start, "frames": count},
                      "faces_per_frame": [len(r) for r in observed], "tracker_scene_cuts": [i for i, c in enumerate(cuts) if c],
                      "mean_face_speed_px": round(clip_speed(observed), 2), "metrics": metrics,
                      "robust_tiers": tiers, "robust_reasons": reasons, "robust_temporal_resets": resets,
                      "montage": picture.name if montage(picture, clip, observed, runs, sorted({0, count // 2, count - 1})) else None,
                      "natural_montage": natural.name if natural_ok else None})
        summary["scenarios"][scenario.name] = entry
        print(scenario.name, json.dumps(entry["metrics"]), flush=True)
    reference_compositor.release()
    tracker.release()
    (out_dir / "occlusion_acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"))
    parser.add_argument("--scenario", action="append", help="limit to a scenario name (repeatable)")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()
    run(args.out, provider=args.provider, names=tuple(args.scenario) if args.scenario else None, frames_per_clip=args.frames)


if __name__ == "__main__":
    main()
