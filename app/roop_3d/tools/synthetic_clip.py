"""Synthetic 10-frame test clip writer (BT.709 limited range, optional AAC audio). Model-free."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def create_clip(path: Path, size: int = 64, *, audio: bool = False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
        stream.codec_context.max_b_frames = 0
        # A properly tagged BT.709 limited-range source: untagged matrix/range is routed to the colour-managed reference path.
        context = stream.codec_context
        context.colorspace = context.color_primaries = context.color_trc = 1
        context.color_range = 1
        audio_stream = container.add_stream("aac", rate=48000) if audio else None
        if audio_stream:
            audio_stream.layout = "mono"
        for index in range(10):
            image = np.full((size, size, 3), (40, 80, 20), np.uint8)
            if 2 <= index <= 6:
                image[8:44, :30] = (240, 240, 240)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p", dst_colorspace=1, dst_color_range=1)
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        if audio_stream:
            samples = (np.sin(np.arange(19200) * (2 * np.pi * 440 / 48000)) * 0.2).astype(np.float32)[None]
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
            for packet in audio_stream.encode(None):
                container.mux(packet)
