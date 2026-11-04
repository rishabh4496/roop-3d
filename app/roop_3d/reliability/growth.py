"""Bounded-growth analysis: decide whether a resource series is a plateau (healthy) or a leak.

A single "end minus start" delta is a poor leak test: allocators ratchet up during warm-up, garbage collection makes
series saw-toothed, and a one-off spike looks like growth. This module uses two independent checks that must agree:

* **Theil-Sen slope** over the post-warm-up window. It is the median of all pairwise slopes, so isolated spikes and
  GC dips cannot move it, and a genuine linear leak always does.
* **High-water growth**: the largest value in the final third of the window against the largest in the middle third.
  A bounded profile stops setting new highs; a leak keeps setting them.
* **Net growth**: the lowest of the final three samples against the median of the window's head. A leak is never given
  back; a burst during one phase of a job (decoder threads while muxing) that has been released by the end is not growth.

A metric is flagged only when the projected growth over the window exceeds that metric's allowance *and* all three checks
show a rising trend. Absolute ceilings are checked separately and always fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Any, Iterable, Mapping, Sequence


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Median pairwise slope. O(n^2); callers cap the series (see ``MAX_POINTS``) so this stays instant."""
    slopes = sorted((ys[j] - ys[i]) / (xs[j] - xs[i])
                    for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i])
    if not slopes:
        return 0.0
    middle = len(slopes) // 2
    return slopes[middle] if len(slopes) % 2 else (slopes[middle - 1] + slopes[middle]) / 2


def positive_fraction(ys: Sequence[float]) -> float:
    """Share of consecutive-window comparisons that rise (Mann-Kendall flavoured, tolerant of plateaus)."""
    pairs = [(ys[i], ys[j]) for i in range(len(ys)) for j in range(i + 1, len(ys)) if ys[j] != ys[i]]
    return sum(b > a for a, b in pairs) / len(pairs) if pairs else 0.0


MAX_POINTS = 240


def _thin(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    if len(xs) <= MAX_POINTS:
        return xs, ys
    step = len(xs) / MAX_POINTS
    keep = [int(i * step) for i in range(MAX_POINTS)]
    return [xs[i] for i in keep], [ys[i] for i in keep]


@dataclass(frozen=True)
class MetricBound:
    """Allowed behaviour for one metric.

    ``allowance`` is the growth, in the metric's own unit, tolerated across the steady-state window (after warm-up).
    ``ceiling`` is an absolute value that must never be exceeded. ``settle_to`` (for live-object counts) is the value
    the metric must return to when the workload is idle.
    """
    allowance: float
    ceiling: float | None = None
    settle_to: float | None = None


@dataclass(frozen=True)
class BoundedProfile:
    name: str
    bounds: Mapping[str, MetricBound]
    warmup_fraction: float = 0.25
    min_samples: int = 8
    default: MetricBound | None = None

    def bound_for(self, metric: str) -> MetricBound | None:
        if metric in self.bounds:
            return self.bounds[metric]
        # ``live.<kind>`` gauges fall back to a strict "no growth" bound.
        return self.default if self.default is not None and metric.startswith("live.") else None


@dataclass
class MetricVerdict:
    metric: str
    status: str                       # "ok" | "growing" | "ceiling" | "insufficient"
    first: float | None = None
    last: float | None = None
    peak: float | None = None
    slope_per_unit: float = 0.0
    projected_growth: float = 0.0
    high_water_growth: float = 0.0
    net_growth: float = 0.0
    rising_fraction: float = 0.0
    allowance: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in self.__dict__.items()}


@dataclass
class GrowthReport:
    profile: str
    samples: int
    window: tuple[float, float] | None
    verdicts: dict[str, MetricVerdict] = field(default_factory=dict)

    @property
    def violations(self) -> list[MetricVerdict]:
        return [item for item in self.verdicts.values() if item.status in ("growing", "ceiling")]

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "samples": self.samples, "ok": self.ok, "window": self.window,
                "violations": [item.metric for item in self.violations],
                "metrics": {name: item.as_dict() for name, item in sorted(self.verdicts.items())}}


def analyse_series(metric: str, xs: Sequence[float], ys: Sequence[float], bound: MetricBound,
                   profile: BoundedProfile) -> MetricVerdict:
    cut = int(len(xs) * profile.warmup_fraction)
    wx, wy = list(xs[cut:]), list(ys[cut:])
    if len(wy) < profile.min_samples:
        return MetricVerdict(metric, "insufficient", detail=f"{len(wy)} steady-state samples < {profile.min_samples}",
                             allowance=bound.allowance)
    peak = max(ys)
    verdict = MetricVerdict(metric, "ok", first=wy[0], last=wy[-1], peak=peak, allowance=bound.allowance)
    if bound.ceiling is not None and peak > bound.ceiling:
        verdict.status, verdict.detail = "ceiling", f"peak {peak:.2f} exceeds ceiling {bound.ceiling:.2f}"
        return verdict
    tx, ty = _thin(wx, wy)
    verdict.slope_per_unit = theil_sen(tx, ty)
    verdict.projected_growth = verdict.slope_per_unit * (wx[-1] - wx[0])
    verdict.rising_fraction = positive_fraction(ty)
    third = len(wy) // 3
    middle, final = wy[third:2 * third], wy[2 * third:]
    verdict.high_water_growth = max(final) - max(middle) if middle and final else 0.0
    # A leak never gives the memory back: even its lowest recent reading must sit above where the window started;
    # a phase change that raises a metric transiently (render -> mux spawns more decoder threads) and then releases it
    # ends where it began and is not growth.
    verdict.net_growth = min(wy[-3:]) - statistics.median(wy[:max(3, len(wy) // 10)])
    if (verdict.projected_growth > bound.allowance and verdict.high_water_growth > bound.allowance * 0.25
            and verdict.rising_fraction >= 0.6 and verdict.net_growth > bound.allowance * 0.5):
        verdict.status = "growing"
        verdict.detail = (f"projected +{verdict.projected_growth:.2f} over the window (allowance "
                          f"{bound.allowance:.2f}); still setting new highs (+{verdict.high_water_growth:.2f})")
    elif bound.settle_to is not None and wy[-1] > bound.settle_to + bound.allowance:
        verdict.status = "growing"
        verdict.detail = f"final {wy[-1]:.2f} did not settle to {bound.settle_to:.2f}"
    return verdict


def analyse(profile: BoundedProfile, series: Mapping[str, Iterable[tuple[float, float]]]) -> GrowthReport:
    """``series[metric]`` is a sequence of ``(x, value)``; x is time or iteration number."""
    report = GrowthReport(profile.name, 0, None)
    for metric, points in series.items():
        bound = profile.bound_for(metric)
        pts = sorted(points)
        report.samples = max(report.samples, len(pts))
        if bound is None or not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        report.window = (xs[0], xs[-1]) if report.window is None else (min(report.window[0], xs[0]), max(report.window[1], xs[-1]))
        report.verdicts[metric] = analyse_series(metric, xs, ys, bound, profile)
    return report


# The profile every soak run is judged against. Values are *allowances over the steady-state window*, in the
# metric's own unit, and were established from the measured soak runs recorded in docs/RELIABILITY_STAGE18.md.
DEFAULT_PROFILE = BoundedProfile(
    name="roop3d-stage18-v1",
    bounds={
        "rss_mib": MetricBound(allowance=48.0),
        "private_mib": MetricBound(allowance=48.0),
        "handles": MetricBound(allowance=12.0),
        # Inside one job the OS thread count legitimately swings by ~40 (libav frame threads and ONNX Runtime pools appear
        # in the mux phase and retire afterwards; measured 32..76 on a 1080p job). Per-job thread *leaks* are judged
        # strictly at idle points by IDLE_PROFILE below.
        "os_threads": MetricBound(allowance=32.0),
        "py_threads": MetricBound(allowance=4.0),
        # Direct child processes include short-lived helpers the hardware probe starts (nvidia-smi, powershell, an ffmpeg
        # encoder probe: ~1 s each). Orphans are caught exactly instead: `live.ffmpeg_process` (registry gauge, strict) and
        # the post-drain child check in the soak invariants.
        "children": MetricBound(allowance=3.0),
        "gpu_used_mib": MetricBound(allowance=96.0),
        "gpu_process_mib": MetricBound(allowance=96.0),
        "ort_sessions": MetricBound(allowance=0.0),
        "py_heap_blocks": MetricBound(allowance=200_000.0),
    },
    default=MetricBound(allowance=0.0),
)


# Sampled once per finished job / model switch, i.e. at idle points, where a thread, handle or session that is still
# there after the work is done is a leak by definition. Same memory allowances; strict on everything that is a count.
IDLE_PROFILE = BoundedProfile(
    name="roop3d-stage18-v1-idle",
    bounds={**DEFAULT_PROFILE.bounds,
            "os_threads": MetricBound(allowance=6.0),
            "py_threads": MetricBound(allowance=2.0),
            "children": MetricBound(allowance=1.0)},
    default=MetricBound(allowance=0.0),
)
