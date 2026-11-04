"""Real-codec fixtures for Stage 15. No mocks: every file is encoded by libx264 / libx265 / FFV1 and
decoded again, and every expected pixel comes from the float oracle in ``roop_3d.color.gamut``, which
shares no code with swscale or libavfilter."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from roop_3d.color import gamut
from roop_3d.color.description import HdrStaticMetadata

#: Eight flat stripes: black, white, the three primaries, mid grey, yellow, a muted blue. Gamma-encoded 0..1.
STRIPES = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, .5], [1, 1, 0], [.2, .4, .8]], float)
STRIPE_W = 8


def stripe_rgb(width: int = 64, height: int = 32, stripes: np.ndarray = STRIPES, shift: int = 0) -> np.ndarray:
    """Float RGB image of vertical stripes, optionally rolled by ``shift`` stripes to vary frames."""
    rgb = np.zeros((height, width, 3))
    for i, color in enumerate(np.roll(stripes, shift, axis=0)):
        rgb[:, i * (width // len(stripes)):(i + 1) * (width // len(stripes))] = color
    return rgb


def stripe_centres(width: int = 64, count: int = 8) -> list[int]:
    step = width // count
    return [i * step + step // 2 for i in range(count)]


def expected_bgr8(shift: int = 0, stripes: np.ndarray = STRIPES) -> np.ndarray:
    """What a colour-correct SDR decode must return at each stripe centre (8 x BGR)."""
    return np.rint(np.roll(stripes, shift, axis=0)[:, ::-1] * 255).astype(int)


def _planes(rgb: np.ndarray, matrix: int, full: bool, bits: int, chroma: str):
    ycc = np.clip(np.rint(gamut.rgb_to_ycbcr(rgb, matrix, full, bits)), 0, (1 << bits) - 1)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    if chroma == "420":
        cb, cr = cb[::2, ::2], cr[::2, ::2]
    elif chroma == "422":
        cb, cr = cb[:, ::2], cr[:, ::2]
    return y, cb, cr


def _fill(frame: av.VideoFrame, planes, dtype) -> None:
    for plane, data in zip(frame.planes, planes):
        view = np.ndarray((data.shape[0], plane.line_size // np.dtype(dtype).itemsize), dtype, buffer=plane)
        view[:, :data.shape[1]] = data.astype(dtype)


def make_color_video(path: Path, *, matrix: int = 1, full: bool = False, bits: int = 8, primaries: int = 1,
                     trc: int = 1, chroma: str = "420", width: int = 64, height: int = 32, frames: int = 3,
                     rate: Fraction = Fraction(25), tagged: bool = True, hdr: HdrStaticMetadata | None = None,
                     rotation: int = 0, pts: tuple[int, ...] | None = None, time_base: Fraction | None = None,
                     audio: int = 0, subtitle: bool = False, lossless: bool = True,
                     vcodec: str | None = None, acodec: str = "pcm_s16le") -> Path:
    """Write a video whose YUV code values are computed by the oracle from known stripe colours."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = f"yuv{chroma}p" + (f"{bits}le" if bits > 8 else "")
    if full and bits == 8 and chroma == "420":
        pix = "yuvj420p"
    dtype = np.uint8 if bits == 8 else np.uint16
    base = time_base or (Fraction(1, 1000) if rate == 25 else Fraction(1, rate.numerator))
    with av.open(str(path), "w") as out:
        encoder = vcodec or ("libx264" if bits == 8 else "libx265")
        video = out.add_stream(encoder, rate=rate)
        video.width, video.height, video.pix_fmt = width, height, pix
        ctx = video.codec_context
        ctx.time_base = base
        ctx.max_b_frames = 0
        ctx.gop_size = 2
        if tagged:
            ctx.colorspace, ctx.color_primaries, ctx.color_trc = matrix, primaries, trc
            ctx.color_range = 2 if full else 1
        if rotation:
            video.set_display_rotation(rotation)
        options = {}
        if encoder == "libvpx-vp9":
            options = {"lossless": "1"} if bits == 8 else {"crf": "4", "b:v": "0"}  # VP9 lossless needs 8-bit 4:2:0
        elif lossless:
            options = {"crf": "0"} if encoder == "libx264" else {"x265-params": "lossless=1:log-level=none"}
        if hdr is not None:
            extra = hdr.to_x265_params() or ""
            current = options.get("x265-params", "log-level=none")
            options["x265-params"] = f"{current}:repeat-headers=1:hdr10=1" + (f":{extra}" if extra else "")
        video.options = options
        audio_streams = []
        for index in range(audio):
            stream = out.add_stream(acodec, rate=48000)
            stream.layout = "mono" if index == 0 else "stereo"
            stream.metadata["language"] = ("eng", "hin", "fra")[index % 3]
            audio_streams.append(stream)
        sub = None
        if subtitle:
            sub = out.add_stream("ass")
            sub.time_base = sub.codec_context.time_base = Fraction(1, 1000)
            sub.codec_context.extradata = (b"[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
                                           b"Style: Default,Arial,20\n[Events]\nFormat: Layer, Start, End, Style, Text\n")
            sub.metadata["language"] = "eng"
        stamps = pts if pts is not None else tuple(int(Fraction(i) / rate / base) for i in range(frames))
        duration = int(Fraction(1) / rate / base)
        for index, stamp in enumerate(stamps):
            frame = av.VideoFrame(width, height, pix)
            _fill(frame, _planes(stripe_rgb(width, height, shift=index), matrix, full, bits, chroma), dtype)
            frame.pts, frame.time_base, frame.duration = stamp, base, duration
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        total = (stamps[-1] + duration) * base
        for index, stream in enumerate(audio_streams):
            write_click_audio(out, stream, float(total), click_at=0.5 + 0.25 * index)
        if sub is not None:
            for k in range(2):
                packet = av.Packet(f"{k},0,Default,,Hello {k}".encode())
                packet.stream, packet.pts, packet.dts = sub, 80 * k, 80 * k
                packet.duration, packet.time_base = 40, sub.time_base
                out.mux(packet)
    return path


def write_click_audio(out, stream, seconds: float, click_at: float, rate: int = 48000) -> None:
    """Silence with one full-scale click at ``click_at`` seconds: the marker used to measure A/V sync."""
    count = int(seconds * rate)
    signal = np.zeros(count, np.int16)
    at = int(click_at * rate)
    if at < count:
        signal[at:at + 48] = 20000
    channels = len(stream.layout.channels)
    data = np.repeat(signal, channels).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout=stream.layout.name)
    frame.sample_rate, frame.pts, frame.time_base = rate, 0, Fraction(1, rate)
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)


def read_stripes(image_bgr: np.ndarray, width: int = 64, count: int = 8) -> np.ndarray:
    row = image_bgr.shape[0] // 2
    return np.array([image_bgr[row, x] for x in stripe_centres(width, count)]).astype(int)
