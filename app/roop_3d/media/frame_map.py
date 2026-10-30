"""Streaming CFR sample-and-hold selection, including every drop and duplicate."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Iterable, Iterator, Protocol, TypeVar

from roop_3d.media.timestamps import Timestamp


class TimedFrame(Protocol):
    index: int
    timestamp: Timestamp
    duration: int


T = TypeVar("T", bound=TimedFrame)


@dataclass(frozen=True)
class FrameMapEntry:
    source_index: int
    output_index: int | None
    action: str
    source_pts: int
    source_time_base: str
    output_pts: int | None
    output_time_base: str | None


class FrameMap:
    def __init__(self, rate: Fraction, origin: Fraction = Fraction(0)):
        if not isinstance(rate, Fraction) or not isinstance(origin, Fraction) or rate <= 0:
            raise ValueError("CFR rate must be positive")
        self.rate, self.origin = rate, origin
        self.time_base = Fraction(1, lcm(rate.numerator, origin.denominator))
        self.entries: list[FrameMapEntry] = []

    def select(self, frames: Iterable[T]) -> Iterator[tuple[T, Timestamp]]:
        iterator = iter(frames)
        current = next(iterator, None)
        if current is None:
            return
        output = 0
        while current is not None:
            following = next(iterator, None)
            end = (following.timestamp.seconds if following is not None else
                   current.timestamp.seconds + current.duration * current.timestamp.time_base)
            if current.duration <= 0 and following is None:
                raise ValueError("Final CFR source frame requires an explicit duration")
            uses = 0
            while self.origin + Fraction(output, 1) / self.rate < end:
                at = self.origin + Fraction(output, 1) / self.rate
                if at >= current.timestamp.seconds:
                    stamp = Timestamp(int(at / self.time_base), self.time_base)
                    self.entries.append(FrameMapEntry(current.index, output, "keep" if not uses else "duplicate",
                                                      current.timestamp.pts, str(current.timestamp.time_base),
                                                      stamp.pts, str(stamp.time_base)))
                    yield current, stamp
                    uses += 1
                output += 1
            if not uses:
                self.entries.append(FrameMapEntry(current.index, None, "drop", current.timestamp.pts,
                                                  str(current.timestamp.time_base), None, None))
            current = following
