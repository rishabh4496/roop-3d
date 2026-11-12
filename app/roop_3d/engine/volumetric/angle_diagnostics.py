"""Per-frame head-angle stability diagnostics for the dense 3D swap.

Pure metric functions plus a bounded recorder. Cheap summary metrics are always collected; image metrics on the composited
frame and per-frame JSONL are produced only at level ``"full"``, selected by the job's ``angle_diagnostics`` field.
The swap stage and composite stage run on different threads, so a record is assembled from two halves keyed by
``(frame, track)``.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import cv2
import numpy as np

LEVELS = ("summary", "full")
YAW_BUCKET_DEGREES = 10
SERIES_LIMIT = 20000            # samples per metric kept for the summary statistics
DARK_LUMA, PLATE_BRIGHT_LUMA = 16.0, 40.0
_LUMA = np.array((0.114, 0.587, 0.299), np.float32)   # BGR




def angle_jsonl_name(job_id: str) -> str:
    """File name of a job's per-frame diagnostics, inside the output directory."""
    return f"roop3d_{job_id}.angle.jsonl"




# -- pure metrics -----------------------------------------------------------------------------------------------
def decompose_similarity(matrix) -> tuple[float, float, float, float]:
    """``(scale, theta_degrees, tx, ty)`` of the similarity part of a 2x3 affine."""
    m = np.asarray(matrix, np.float64)
    return (float(math.hypot(m[0, 0], m[1, 0])), float(math.degrees(math.atan2(m[1, 0], m[0, 0]))),
            float(m[0, 2]), float(m[1, 2]))


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def inter_ocular(landmarks5) -> float:
    points = np.asarray(landmarks5, np.float64)
    return float(np.linalg.norm(points[1] - points[0]))


def landmark_jitter(l0, l1, l2) -> float:
    """RMS second difference of three consecutive landmark sets, divided by the current inter-ocular distance.

    The second difference removes constant-velocity motion, leaving only the frame-to-frame noise.
    """
    a, b, c = (np.asarray(v, np.float64) for v in (l0, l1, l2))
    return float(np.sqrt(np.mean(np.sum((c - 2 * b + a) ** 2, axis=-1))) / max(inter_ocular(c), 1.0))


def affine_jitter(d0, d1, d2, iod: float) -> dict:
    """Second-difference jitter of three consecutive decomposed crop affines ``(s, theta, tx, ty)``."""
    s0, s1, s2 = d0[0], d1[0], d2[0]
    dtheta = wrap_degrees(wrap_degrees(d2[1] - d1[1]) - wrap_degrees(d1[1] - d0[1]))
    dt = np.hypot(d2[2] - 2 * d1[2] + d0[2], d2[3] - 2 * d1[3] + d0[3])
    return {"ds_rel": abs(s2 - 2 * s1 + s0) / max(s1, 1e-9), "dtheta_deg": abs(float(dtheta)),
            "dt_rel": float(dt) / max(s2 * iod, 1.0)}


def mask_stats(mask) -> dict | None:
    """Area (pixels with value > .5, as a fraction of the crop) and centroid (crop pixels) of a coverage mask.

    Measured on a stride-decimated copy (at most ~256 px a side) so a 1024 px mask costs the same as a small one.
    """
    mask = np.asarray(mask)
    step = max(1, max(mask.shape[:2]) // 256)
    inside = mask[::step, ::step] > 0.5
    count = int(inside.sum())
    if count == 0:
        return None
    ys, xs = np.nonzero(inside)
    return {"area": count / inside.size, "cx": float(xs.mean() * step + (step - 1) / 2),
            "cy": float(ys.mean() * step + (step - 1) / 2)}


def luma(image_bgr) -> np.ndarray:
    return np.asarray(image_bgr, np.float32) @ _LUMA


def ring_luma_stats(alpha, composite_bgr, plate_bgr) -> dict | None:
    """Black-edge / halo detector on a frame-space ROI.

    ``R_out = {.02 < alpha < .5}`` is the outer feather ring, ``R_in = {.5 < alpha < .98}`` the inner one.
    ``dark_px`` counts outer-ring pixels darker than 16 where the plate was brighter than 40 (a dark fringe the plate
    did not have).
    """
    alpha = np.asarray(alpha, np.float32)
    outer = (alpha > 0.02) & (alpha < 0.5)
    inner = (alpha >= 0.5) & (alpha < 0.98)
    if not outer.any() and not inner.any():
        return None
    out_luma, plate_luma = luma(composite_bgr), luma(plate_bgr)
    result = {"ring_px": int(outer.sum()), "dark_px": int((outer & (out_luma < DARK_LUMA) & (plate_luma > PLATE_BRIGHT_LUMA)).sum())}
    if outer.any():
        result.update(out_mean=float(out_luma[outer].mean()), out_min=float(out_luma[outer].min()),
                      plate_out_mean=float(plate_luma[outer].mean()))
    if inner.any():
        result["in_luma_step"] = float(abs(out_luma[inner].mean() - plate_luma[inner].mean()))
    return result


def sharpness(image_bgr, side: int = 160) -> float:
    """Laplacian variance of a ``side`` x ``side`` gray resize (the proxy Roop Ultimate uses for face sharpness)."""
    gray = cv2.cvtColor(np.asarray(image_bgr, np.uint8), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_metrics(plate_bgr, output_bgr, alpha, inverse) -> dict:
    """Image metrics of one replaced face in frame space.

    ``alpha`` is the crop-space blend alpha and ``inverse`` the crop -> frame affine it was composited with; both are
    mapped into the face's frame ROI, where the composited frame is compared with the untouched plate.
    """
    alpha = np.asarray(alpha, np.float32)
    height, width = np.asarray(plate_bgr).shape[:2]
    side_y, side_x = alpha.shape[:2]
    corners = np.array(((0, 0), (side_x, 0), (0, side_y), (side_x, side_y)), np.float64) @ np.asarray(inverse, np.float64)[:, :2].T \
        + np.asarray(inverse, np.float64)[:, 2]
    x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int), (width, height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {}
    shift = np.asarray(inverse, np.float64).copy()
    shift[:, 2] -= (x0, y0)
    roi = cv2.warpAffine(alpha, shift, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    plate, final = (np.asarray(image)[y0:y1, x0:x1] for image in (plate_bgr, output_bgr))
    result = {}
    ring = ring_luma_stats(roi, final, plate)
    if ring is not None:
        result["ring"] = ring
    inside = np.nonzero(roi > 0.5)
    if len(inside[0]) >= 64:
        ya, yb, xa, xb = inside[0].min(), inside[0].max() + 1, inside[1].min(), inside[1].max() + 1
        if yb - ya >= 8 and xb - xa >= 8:
            result["sharpness_ratio"] = sharpness(final[ya:yb, xa:xb]) / max(sharpness(plate[ya:yb, xa:xb]), 1e-6)
    return result


def yaw_bucket(yaw: float) -> int:
    """Lower edge of the ``|yaw|`` 10-degree bin the frame belongs to (0..90)."""
    return int(min(max(abs(float(yaw)), 0.0), 99.0) // YAW_BUCKET_DEGREES * YAW_BUCKET_DEGREES)


def _stats(values) -> dict:
    values = np.asarray(values, np.float64)
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


class AngleDiagnostics:
    """Bounded recorder; one instance per render session.

    ``record_swap`` is called by the swap stage with the dense engine's ``last_diag`` (and the tracker pose),
    ``record_composite`` by the composite stage with the frame-space metrics. ``summary`` is a compact JSON object for
    the job manifest.
    """

    def __init__(self, level: str = "summary", *, max_records: int = 512, max_pending: int = 64):
        if level not in LEVELS:
            raise ValueError(f"Diagnostics level must be one of {LEVELS}")
        self.level, self.level_source = level, "job"
        self._lock = threading.Lock()
        self._pending: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._records: deque[dict] = deque(maxlen=max_records)
        self._max_pending = max_pending
        self._history: dict[int, deque] = {}
        self._views: dict[int, dict] = {}
        self._sink = None
        self._partial = self._final = None
        self._reset_totals()

    # -- configuration -----------------------------------------------------------------------------------------
    @property
    def full(self) -> bool:
        return self.level == "full"

    def configure(self, level: str, jsonl_path: str | Path | None = None) -> None:
        """Start collection at the job-selected level; full output is atomically published by :meth:`finalize`."""
        if level not in LEVELS:
            raise ValueError(f"Diagnostics level must be one of {LEVELS}")
        with self._lock:
            self._close_sink()
            self._remove_files()
            self.level, self.level_source = level, "job"
            self._pending.clear()
            self._records.clear()
            self._history.clear()
            self._views.clear()
            self._reset_totals()
            if self.full and jsonl_path is not None:
                self._final = Path(jsonl_path)
                self._partial = self._final.with_name(self._final.name + ".partial")
                self._final.parent.mkdir(parents=True, exist_ok=True)
                self._sink = self._partial.open("w", encoding="utf-8")

    def _reset_totals(self) -> None:
        self.artifact = None
        self.frames = 0
        series = lambda: deque(maxlen=SERIES_LIMIT)      # summary statistics cover the most recent SERIES_LIMIT frames
        self._lm5, self._ds, self._dtheta, self._dt = series(), series(), series(), series()
        self._d_centroid, self._d_area = series(), series()
        self._switches, self._weight_delta = 0, series()
        self._dark_px, self._dark_frames, self._ring_frames = 0, 0, 0
        self._luma_step = series()
        self._sharp: dict[int, deque] = {}

    def _close_sink(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def close(self) -> None:
        with self._lock:
            for key in list(self._pending):
                self._finish(key)
            self._close_sink()

    def _remove_files(self) -> None:
        for path in (self._partial, self._final):
            if path is not None:
                path.unlink(missing_ok=True)
        self._partial = self._final = None

    def finalize(self) -> dict | None:
        """Close the stream and publish it atomically; ``{"path", "sha256", "rows"}`` (``path`` is the file name inside
        the output directory), or ``None`` when nothing was streamed (summary level, or no dense face was swapped)."""
        self.close()
        with self._lock:
            if self._partial is None or not self._partial.exists():
                return None
            digest, rows = hashlib.sha256(), 0
            with self._partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
                    rows += chunk.count(b"\n")
            if rows == 0:
                self._partial.unlink()
                return None
            os.replace(self._partial, self._final)
            self.artifact = {"path": self._final.name, "sha256": digest.hexdigest(), "rows": rows}
            self._partial = None
            return dict(self.artifact)

    def discard(self) -> None:
        """Forget the collection and delete the partial and the published JSONL (a cancelled or failed job)."""
        with self._lock:
            self._close_sink()
            self._remove_files()
            self.artifact = None

    # -- swap stage --------------------------------------------------------------------------------------------
    def record_swap(self, frame: int, track_id: int, diag: dict, *, pose) -> None:
        """``diag``: ``crop_matrix`` (2x3), ``landmarks5`` (frame px) and optional ``coverage`` mask (with
        ``coverage_scale`` = mask side / swap-crop side) / ``views`` list."""
        record = {"frame": int(frame), "track": int(track_id),
                  "pose": [round(float(v), 2) for v in pose[:3]]}
        decomposed = decompose_similarity(diag["crop_matrix"])
        landmarks = np.asarray(diag["landmarks5"], np.float64)
        iod = inter_ocular(landmarks)
        if diag.get("presence") is not None:
            record["presence"] = round(float(diag["presence"]), 4)
        record["crop"] = {"s": round(decomposed[0], 5), "theta_deg": round(decomposed[1], 4),
                          "tx": round(decomposed[2], 3), "ty": round(decomposed[3], 3)}
        stats = mask_stats(diag["coverage"]) if diag.get("coverage") is not None else None
        if stats is not None:
            # Centroids are reported in swap-crop pixels whatever resolution the enhancer returned.
            scale = float(diag.get("coverage_scale", 1.0))
            stats["cx"], stats["cy"] = stats["cx"] / scale, stats["cy"] / scale
            record["coverage"] = {key: round(value, 5) for key, value in stats.items()}
        with self._lock:
            history = self._history.setdefault(int(track_id), deque(maxlen=2))
            if history and history[-1]["frame"] != frame - 1:
                history.clear()
            current = {"frame": int(frame), "lm": landmarks, "affine": decomposed, "iod": iod, "mask": stats}
            if history:
                previous = history[-1]
                if stats is not None and previous["mask"] is not None:
                    shift = math.hypot(stats["cx"] - previous["mask"]["cx"], stats["cy"] - previous["mask"]["cy"])
                    change = (stats["area"] - previous["mask"]["area"]) / max(previous["mask"]["area"], 1e-9)
                    record["coverage"].update(d_centroid_px=round(shift, 4), d_area_rel=round(change, 5))
                    self._d_centroid.append(shift)
                    self._d_area.append(abs(change))
            if len(history) == 2:
                older, newer = history
                record["jitter"] = {"lm5": round(landmark_jitter(older["lm"], newer["lm"], landmarks), 6),
                                    **{key: round(value, 6) for key, value in
                                       affine_jitter(older["affine"], newer["affine"], decomposed, iod).items()}}
                self._lm5.append(record["jitter"]["lm5"])
                self._ds.append(record["jitter"]["ds_rel"])
                self._dtheta.append(record["jitter"]["dtheta_deg"])
                self._dt.append(record["jitter"]["dt_rel"])
            history.append(current)
            views = diag.get("views")
            if views is not None:
                record["views"] = self._view_log(int(track_id), int(frame), views)
            self.frames += 1
            self._pending[(int(frame), int(track_id))] = record
            while len(self._pending) > self._max_pending:
                self._finish(next(iter(self._pending)))

    def _view_log(self, track_id: int, frame: int, views: list[dict]) -> dict:
        indices = tuple(v["index"] for v in views)
        weights = {v["index"]: float(v["weight"]) for v in views}
        state = self._views.get(track_id)
        switch = state is not None and state["frame"] == frame - 1 and set(state["indices"]) != set(indices)
        dwell = 1 if state is None or state["frame"] != frame - 1 or switch else state["dwell"] + 1
        if state is not None and state["frame"] == frame - 1:
            delta = max((abs(weights.get(i, 0.0) - state["weights"].get(i, 0.0))
                         for i in set(weights) | set(state["weights"])), default=0.0)
            self._weight_delta.append(delta)
        else:
            delta = None
        self._switches += int(switch)
        self._views[track_id] = {"frame": frame, "indices": indices, "weights": weights, "dwell": dwell}
        entry = {"selected": [{"index": v["index"], "yaw": round(float(v["yaw"]), 1), "pitch": round(float(v["pitch"]), 1),
                               "raw_score": round(float(v["raw"]), 4), "weight": round(float(v["weight"]), 4)} for v in views],
                 "switch": bool(switch), "dwell": dwell}
        if delta is not None:
            entry["max_weight_delta"] = round(delta, 4)
        return entry

    # -- composite stage ---------------------------------------------------------------------------------------
    def record_composite(self, frame: int, track_id: int, metrics: dict) -> None:
        with self._lock:
            record = self._pending.get((int(frame), int(track_id)))
            if record is None:
                return
            ring = metrics.get("ring")
            if ring is not None:
                record["ring"] = {key: round(value, 3) if isinstance(value, float) else value for key, value in ring.items()}
                self._ring_frames += 1
                self._dark_px += ring["dark_px"]
                self._dark_frames += int(ring["dark_px"] > 0)
                if "in_luma_step" in ring:
                    self._luma_step.append(ring["in_luma_step"])
            if metrics.get("sharpness_ratio") is not None:
                bucket = yaw_bucket(record["pose"][0])
                record["sharpness"] = {"ratio": round(metrics["sharpness_ratio"], 4), "yaw_bucket": bucket}
                self._sharp.setdefault(bucket, deque(maxlen=SERIES_LIMIT)).append(metrics["sharpness_ratio"])
            self._finish((int(frame), int(track_id)))

    def _finish(self, key) -> None:
        record = self._pending.pop(key, None)
        if record is None:
            return
        self._records.append(record)
        if self._sink is not None:
            self._sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self._sink.flush()


    # -- reporting ---------------------------------------------------------------------------------------------
    def records(self) -> list[dict]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict:
        """Compact JSON-safe section for the job manifest (``stage12.angle_diagnostics``)."""
        with self._lock:
            result = {"level": self.level, "level_source": self.level_source, "frames": self.frames,
                      "landmarks5_second_difference_over_iod": _stats(self._lm5),
                      "crop_affine_second_difference": {"scale_relative": _stats(self._ds), "theta_degrees": _stats(self._dtheta),
                                                        "translation_over_scaled_iod": _stats(self._dt)},
                      "coverage_delta": {"centroid_px": _stats(self._d_centroid), "area_relative": _stats(self._d_area)},
                      "source_views": {"switches": self._switches,
                                       "switches_per_100_frames": round(100.0 * self._switches / max(self.frames, 1), 3),
                                       "max_weight_delta": float(max(self._weight_delta, default=0.0))}}
            if self.full:
                result["mask_edge"] = {"frames_measured": self._ring_frames, "frames_with_dark_px": self._dark_frames,
                                       "dark_px_total": self._dark_px, "inner_luma_step": _stats(self._luma_step)}
                result["sharpness_ratio_by_abs_yaw"] = {
                    f"{bucket}-{bucket + YAW_BUCKET_DEGREES}": {"n": len(values), "mean": float(np.mean(values))}
                    for bucket, values in sorted(self._sharp.items())}
                result["jsonl"] = self.artifact
            return json.loads(json.dumps(result, allow_nan=False))
