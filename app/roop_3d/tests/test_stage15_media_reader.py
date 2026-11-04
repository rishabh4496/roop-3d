"""Stage 15 decode: colour-correct, explicit about HDR, untagged material resolved by recorded convention."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from roop_3d.color import transfer
from roop_3d.color.hdr import HdrToSdr, ToneMapParams
from roop_3d.core.errors import DomainError
from roop_3d.media.probe import probe
from roop_3d.media.video_reader import VideoReader
from roop_3d.tests.stage15_media import (
    STRIPES, expected_bgr8, make_color_video, read_stripes, stripe_rgb)


def first_image(path, **kwargs):
    reader = VideoReader(path, hardware="software", **kwargs)
    image = next(iter(reader.frames())).image
    return reader, image


@pytest.mark.parametrize("matrix,primaries,trc,full,bits,chroma", [
    (1, 1, 1, False, 8, "420"), (1, 1, 1, True, 8, "420"), (5, 5, 6, False, 8, "420"), (6, 6, 6, False, 8, "420"),
    (9, 9, 14, False, 10, "420"), (9, 9, 14, True, 10, "420"), (1, 1, 1, False, 10, "422"), (1, 1, 1, False, 10, "444"),
    (9, 9, 14, False, 10, "444"), (1, 1, 1, True, 8, "444")])
def test_sdr_decode_is_colour_correct_for_every_matrix_range_and_bit_depth(
        tmp_path, matrix, primaries, trc, full, bits, chroma):
    # Chroma-444 variants have no subsampling error, so they hold a tighter bound.
    path = make_color_video(tmp_path / "s.mkv", matrix=matrix, primaries=primaries, trc=trc, full=full, bits=bits,
                            chroma=chroma, frames=1)
    reader, image = first_image(path)
    error = np.abs(read_stripes(image) - expected_bgr8()).max()
    assert error <= (3 if chroma == "444" else 4), (matrix, full, bits, chroma, error)
    assert reader.working.conversion is None and not reader.source_color.assumed_fields
    assert reader.working.tags == {"primaries": primaries, "transfer": trc, "matrix": matrix, "range": 2 if full else 1}


def test_a_range_mismatch_would_be_caught_by_this_oracle(tmp_path):
    # Guard against a vacuous test: a decoder that squeezed full-range pixels into studio range
    # (the classic accidental range change) must exceed the tolerance by a wide margin.
    path = make_color_video(tmp_path / "full.mkv", full=True, frames=1)
    correct = read_stripes(first_image(path)[1])
    squeezed = correct * 219 / 255 + 16
    assert np.abs(correct - expected_bgr8()).max() <= 4
    assert np.abs(squeezed - expected_bgr8()).max() >= 16


@pytest.mark.parametrize("width,height,true_matrix", [(1280, 720, 1), (640, 480, 6), (720, 576, 5)])
def test_untagged_material_decodes_with_the_conventional_matrix_not_swscales_601(tmp_path, width, height, true_matrix):
    path = make_color_video(tmp_path / "u.mkv", matrix=true_matrix, primaries=1, trc=1, tagged=False, width=width,
                            height=height, frames=1)
    info = probe(path).videos[0]
    assert info.color.matrix == true_matrix and "matrix" in info.color.assumed_fields
    reader, image = first_image(path)
    assert reader.source_color.matrix == true_matrix
    error = np.abs(read_stripes(image, width) - expected_bgr8()).max()
    assert error <= 4, error


def hdr_expected(path_matrix_trc):
    """Oracle: tone-map the PQ/HLG-coded stripe RGB with the (separately tested) converter."""
    trc, peak = path_matrix_trc
    converter = HdrToSdr(trc, 9, ToneMapParams(source_peak_nits=peak))
    rgb16 = np.rint(stripe_rgb() * 65535).astype(np.uint16)
    return converter.convert(rgb16)


@pytest.mark.parametrize("trc", [16, 18])
def test_hdr_input_is_tone_mapped_by_default_never_treated_as_sdr(tmp_path, trc):
    path = make_color_video(tmp_path / "h.mkv", matrix=9, primaries=9, trc=trc, bits=10, frames=1)
    reader, image = first_image(path)
    assert reader.source_color.dynamic_range == ("HDR_PQ" if trc == 16 else "HDR_HLG")
    assert reader.working.tags == {"primaries": 1, "transfer": 1, "matrix": 1, "range": 1}
    assert reader.working.conversion.startswith("hdr_")
    expected = hdr_expected((trc, 1000.0))[..., ::-1]
    got = read_stripes(image)
    want = np.array([expected[16, x] for x in range(4, 64, 8)]).astype(int)
    assert np.abs(got - want).max() <= 4
    # And it genuinely differs from the legacy reinterpretation, so the test cannot pass by accident.
    _, raw = first_image(path, hdr_policy="raw")
    assert np.abs(read_stripes(raw) - got).max() > 10


def test_hdr_policies_are_enforced_at_the_first_frame(tmp_path):
    path = make_color_video(tmp_path / "h.mkv", matrix=9, primaries=9, trc=16, bits=10, frames=2)
    with pytest.raises(DomainError) as error:
        list(VideoReader(path, hardware="software", hdr_policy="reject").frames())
    assert error.value.code == "HDR_INPUT_REJECTED"
    # The hardware-capable default must reach the same decision (the reference decode runs first).
    with pytest.raises(DomainError):
        list(VideoReader(path, hardware="auto", hdr_policy="reject").frames())
    with pytest.raises(ValueError):
        VideoReader(path, hdr_policy="keep")
    reader = VideoReader(path, hardware="software", hdr_policy="raw")
    list(reader.frames())
    assert reader.working.conversion is None and any("not a valid HDR" in n for n in reader.working.notes)


def test_hdr_tone_map_uses_declared_mastering_peak_and_keeps_rotation(tmp_path):
    from roop_3d.color.description import HdrStaticMetadata
    meta = HdrStaticMetadata(max_cll=4000, max_fall=400, primaries=((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)),
                             white_point=(0.3127, 0.329), min_luminance=0.0001, max_luminance=4000.0)
    path = make_color_video(tmp_path / "h.mkv", matrix=9, primaries=9, trc=16, bits=10, frames=1, hdr=meta,
                            width=64, height=32, rotation=90)
    reader, image = first_image(path)
    assert reader.working.tone_map["source_peak_nits"] == 4000.0
    assert image.shape[:2] == (64, 32)  # upright: 90 degrees swaps the sides
    assert reader.source_color.hdr.max_cll == 4000


def test_hdr_decoding_is_deterministic_and_per_frame(tmp_path):
    path = make_color_video(tmp_path / "h.mkv", matrix=9, primaries=9, trc=16, bits=10, frames=3)
    first = [f.image.copy() for f in VideoReader(path, hardware="software").frames()]
    second = [f.image.copy() for f in VideoReader(path, hardware="software").frames()]
    assert len(first) == 3 and all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(first[0], first[1])  # stripes move between frames


def test_sdr_decode_is_bit_identical_to_the_pre_stage15_path(tmp_path):
    # SDR input must never be converted: compare with a direct swscale decode of the same frame.
    import av
    path = make_color_video(tmp_path / "s.mkv", matrix=1, frames=1)
    with av.open(str(path)) as container:
        direct = next(container.decode(video=0)).to_ndarray(format="bgr24")
    assert np.array_equal(first_image(path)[1], direct)
