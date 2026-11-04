"""8-bit BGR -> encoder frames with an explicit YCbCr matrix and range.

PyAV's ``reformat`` can only request BT.709, BT.601 (470BG / 170M), FCC and SMPTE 240M. BT.2020 is not in
its enum, so asking for it silently produced a BT.601 conversion that was then *tagged* BT.2020 (a
visible colour shift). Those matrices still go through ``reformat`` (identical output to before); BT.2020
and RGB-native output use a libavfilter ``scale`` graph, whose ``out_color_matrix`` supports them and
which is verified against the float oracle in the tests.
"""
from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from roop_3d.core.errors import DomainError

#: Matrix code -> ``scale`` filter ``out_color_matrix`` name.
_SCALE_MATRIX = {1: "bt709", 4: "fcc", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m", 9: "bt2020"}
_REFORMAT_MATRICES = (1, 4, 5, 6, 7)


class RgbFrameConverter:
    def __init__(self, pix_fmt: str, *, matrix: int, full_range: bool, width: int, height: int):
        self.pix_fmt, self.matrix, self.full_range = pix_fmt, matrix, full_range
        self.width, self.height = width, height
        traits = av.VideoFormat(pix_fmt)
        self._rgb_output = traits.is_rgb
        if self._rgb_output:
            self.method = "reformat-rgb"
        elif matrix in _REFORMAT_MATRICES:
            self.method = "reformat"
        elif matrix in _SCALE_MATRIX:
            self.method = "filter-graph"
        else:
            raise DomainError("ENCODE_MATRIX_UNSUPPORTED",
                              f"No verified RGB to YCbCr conversion for matrix code {matrix}", 422)
        self._graph = self._source = self._sink = None
        if self.method == "filter-graph":
            self._build_graph()

    def _build_graph(self) -> None:
        graph = av.filter.Graph()
        source = graph.add_buffer(width=self.width, height=self.height, format=av.VideoFormat("bgr24"),
                                  time_base=Fraction(1, 1000))
        scale = graph.add("scale", f"out_color_matrix={_SCALE_MATRIX[self.matrix]}:"
                                   f"out_range={'pc' if self.full_range else 'tv'}:flags=accurate_rnd+full_chroma_int")
        fmt = graph.add("format", f"pix_fmts={self.pix_fmt}")
        sink = graph.add("buffersink")
        source.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()
        self._graph, self._source, self._sink = graph, source, sink

    def __call__(self, bgr: np.ndarray) -> av.VideoFrame:
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        if self.method == "reformat-rgb":
            return frame.reformat(format=self.pix_fmt)
        if self.method == "reformat":
            return frame.reformat(format=self.pix_fmt, dst_colorspace=self.matrix,
                                  dst_color_range=2 if self.full_range else 1)
        frame.pts, frame.time_base = 0, Fraction(1, 1000)
        self._source.push(frame)
        return self._sink.pull()


def clone_frame(frame: av.VideoFrame) -> av.VideoFrame:
    """Independent copy of a planar frame (pixels only), so a reused decoded frame is never re-stamped."""
    copy = av.VideoFrame(frame.width, frame.height, frame.format.name)
    for source, target in zip(frame.planes, copy.planes):
        rows = np.ndarray((source.height, source.line_size), np.uint8, buffer=source)
        out = np.ndarray((target.height, target.line_size), np.uint8, buffer=target)
        span = min(source.line_size, target.line_size)
        out[:, :span] = rows[:, :span]
    copy.colorspace, copy.color_range = frame.colorspace, frame.color_range
    copy.color_primaries, copy.color_trc = frame.color_primaries, frame.color_trc
    return copy


def promote_planes(frame: av.VideoFrame, pix_fmt: str, *, full_range: bool) -> av.VideoFrame:
    """Re-express a decoded frame in another YCbCr layout without touching matrix or range.

    Used by the native path (8-bit -> 10-bit for ProRes, 4:2:0 -> 4:2:2 for 422-only encoders). Source and
    destination range are passed explicitly so swscale cannot apply an implicit limited/full conversion.
    """
    if frame.format.name == pix_fmt:
        return frame
    rng = 2 if full_range else 1
    return frame.reformat(format=pix_fmt, src_color_range=rng, dst_color_range=rng)
