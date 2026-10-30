"""Integer tick arithmetic. Rounding is nearest, with ties away from zero."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from roop_3d.core.errors import DomainError


def round_ticks(value: Fraction) -> int:
    sign = -1 if value < 0 else 1
    value = abs(value)
    return sign * ((2 * value.numerator + value.denominator) // (2 * value.denominator))


def rescale(ticks: int, source: Fraction, target: Fraction) -> int:
    if type(ticks) is not int or not isinstance(source, Fraction) or not isinstance(target, Fraction):
        raise ValueError("Rescaling requires integer ticks and rational time bases")
    if source <= 0 or target <= 0:
        raise ValueError("Time bases must be positive")
    return round_ticks(ticks * source / target)


@dataclass(frozen=True)
class Timestamp:
    pts: int
    time_base: Fraction

    def __post_init__(self) -> None:
        if type(self.pts) is not int or not isinstance(self.time_base, Fraction) or self.time_base <= 0:
            raise ValueError("Timestamp requires integer ticks and a positive time base")

    @property
    def seconds(self) -> Fraction:
        return self.pts * self.time_base

    def in_base(self, base: Fraction) -> int:
        return rescale(self.pts, self.time_base, base)


class TimestampEngine:
    """Reject missing or non-increasing source PTS rather than invent timing."""

    def __init__(self) -> None:
        self.previous: Fraction | None = None

    def accept(self, pts: int | None, base: Fraction) -> Timestamp:
        if pts is None:
            raise DomainError("TIMESTAMP_MISSING", "A decoded frame has no source PTS", 422)
        stamp = Timestamp(pts, base)
        if self.previous is not None and stamp.seconds <= self.previous:
            raise DomainError("TIMESTAMP_ORDER", "Source frame PTS must strictly increase", 422)
        self.previous = stamp.seconds
        return stamp
