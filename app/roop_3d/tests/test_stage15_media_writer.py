"""Stage 15 encode: colour-correct output, honest HDR handling and verified export profiles (real codecs)."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
import pytest

from roop_3d.color.description import HdrStaticMetadata
from roop_3d.core.errors import DomainError
from roop_3d.media import fidelity
from roop_3d.media.export_profiles import PROFILES
from roop_3d.media.probe import probe
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.video_writer import encode_video
from roop_3d.tests.stage15_media import expected_bgr8, make_color_video, read_stripes, stripe_rgb

# Lossy 8-bit codec error on saturated stripe edges is up to ~8 code values; the defects fixed in Stage 15
# (wrong matrix, range squeeze) produce 16-33, so this bound still separates them cleanly.
LOSSY = 16  # max over stripes
LOSSY_MEAN = 5  # mean over stripes: the old BT.2020 bug measured ~12, a healthy lossy encode ~3
HDR_META = HdrStaticMetadata(max_cll=1000, max_fall=400, primaries=((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)),
                             white_point=(0.3127, 0.329), min_luminance=0.0001, max_luminance=1000.0)


@pytest.fixture()
def root(tmp_path):
    """encode_video may only write inside the owned runtime; tmp_path is under it (pytest --basetemp)."""
    (tmp_path / "roop_3d").mkdir(exist_ok=True)
    return tmp_path


def encode(root, source, name, **kwargs):
    kwargs.setdefault("hardware", "software")
    return encode_video(source, name, root=root, **kwargs)


def out_path(root, manifest):
    return root / manifest["output_path"]


def decode_stripes(path, shift=0):
    image = next(iter(VideoReader(path, hardware="software").frames())).image
    return read_stripes(image), expected_bgr8(shift)


def tags(path):
    v = probe(path).videos[0]
    return {"primaries": v.color_primaries, "transfer": v.color_trc, "matrix": v.colorspace, "range": v.color_range}


# --- colour correctness across the write side ------------------------------------------------------
@pytest.mark.parametrize("matrix,primaries,trc,full,bits,chroma", [
    (1, 1, 1, False, 8, "420"), (1, 1, 1, True, 8, "420"), (5, 5, 6, False, 8, "420"), (6, 6, 6, False, 8, "420"),
    (9, 9, 14, False, 10, "420"), (9, 9, 14, True, 10, "420"), (9, 9, 14, False, 10, "444")])
def test_encode_round_trip_preserves_matrix_range_and_tags(root, matrix, primaries, trc, full, bits, chroma):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=matrix, primaries=primaries, trc=trc, full=full,
                              bits=bits, chroma=chroma, frames=2)
    manifest = encode(root, source, "out.mp4")
    out = out_path(root, manifest)
    got, want = decode_stripes(out)
    assert tags(out) == {"primaries": primaries, "transfer": trc, "matrix": matrix, "range": 2 if full else 1}
    # BT.2020 was previously encoded with the BT.601 matrix while tagged BT.2020: ~33/255 error on green.
    assert np.abs(got - want).max() <= LOSSY and np.abs(got - want).mean() <= LOSSY_MEAN, (matrix, full, bits, np.abs(got - want).mean())
    assert manifest["stage15"]["color"]["path"] == "rgb8" and manifest["stage15"]["color"]["processing_bit_depth"] == 8


def test_a_601_encode_of_bt2020_material_would_have_been_caught(root):
    # Proves the colour assertion has teeth: re-tag a correct BT.2020 output as BT.601 and it must differ visibly.
    source = make_color_video(root / "roop_3d/in.mkv", matrix=9, primaries=9, trc=14, bits=10, frames=1)
    out = out_path(root, encode(root, source, "out.mp4"))
    with av.open(str(out)) as container:
        frame = next(container.decode(video=0))
        frame.colorspace = 5
        wrong = read_stripes(frame.to_ndarray(format="bgr24"))
    assert np.abs(wrong - expected_bgr8()).max() > 15


@pytest.mark.parametrize("width,height,true_matrix", [(1280, 720, 1), (640, 480, 6)])
def test_untagged_source_gets_explicit_tags_and_correct_colour(root, width, height, true_matrix):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=true_matrix, primaries=1, trc=1, tagged=False,
                              width=width, height=height, frames=1)
    manifest = encode(root, source, "out.mp4")
    out = out_path(root, manifest)
    t = tags(out)
    assert t["matrix"] == true_matrix and t["range"] == 1 and t["primaries"] != 2 and t["transfer"] != 2
    image = next(iter(VideoReader(out, hardware="software").frames())).image
    error = np.abs(read_stripes(image, width) - expected_bgr8())
    assert error.max() <= LOSSY and error.mean() <= LOSSY_MEAN
    assert set(manifest["stage15"]["color"]["assumptions"]) == {"matrix", "primaries", "transfer", "range"}


def test_full_range_source_stays_full_range_and_limited_stays_limited(root):
    for full in (True, False):
        source = make_color_video(root / f"roop_3d/in{int(full)}.mkv", full=full, frames=1)
        out = out_path(root, encode(root, source, f"r{int(full)}.mp4"))
        assert tags(out)["range"] == (2 if full else 1)
        got = decode_stripes(out)[0]
        assert got[1].min() >= 249 and got[0].max() <= 6  # white stays white, black stays black


# --- HDR policy ---------------------------------------------------------------------------------------
def hdr_source(root, **kwargs):
    kwargs.setdefault("hdr", HDR_META)
    return make_color_video(root / "roop_3d/hdr.mkv", matrix=9, primaries=9, trc=16, bits=10, **kwargs)


def test_hdr_with_a_transform_is_tone_mapped_to_sdr_and_labelled_as_such(root):
    source = hdr_source(root, frames=2)
    manifest = encode(root, source, "sdr.mp4", transform=lambda frame: frame.image)
    out = out_path(root, manifest)
    assert tags(out) == {"primaries": 1, "transfer": 1, "matrix": 1, "range": 1}
    assert probe(out).videos[0].pixel_format == "yuv420p" and probe(out).videos[0].color.dynamic_range == "SDR"
    color = manifest["stage15"]["color"]
    assert color["claims"]["hdr_output"] == "NOT_PRODUCED_TONE_MAPPED_TO_SDR"
    assert color["claims"]["high_bit_depth_processing"].startswith("UNSUPPORTED")
    assert color["conversion"].startswith("hdr_pq_to_sdr_bt709") and color["working"]["tone_map"]["source_peak_nits"] == 1000.0
    assert any("not HDR" in n for n in color["notes"])


def test_hdr_reject_policy_refuses_before_encoding(root):
    source = hdr_source(root, frames=1)
    with pytest.raises(DomainError) as error:
        encode(root, source, "no.mp4", hdr_policy="reject")
    assert error.value.code == "HDR_INPUT_REJECTED"
    assert not list((root / "roop_3d").rglob("no.mp4"))


@pytest.mark.parametrize("kwargs,why", [
    ({"transform": lambda frame: frame.image, "profile": "color_preserving"}, "transform"),
    ({"profile": "standard_mp4"}, "profile"), ({"profile": "mov_workflow"}, "profile")])
def test_hdr_passthrough_is_refused_wherever_it_cannot_be_honest(root, kwargs, why):
    source = hdr_source(root, frames=1)
    name = "x.mov" if kwargs.get("profile") in ("color_preserving", "mov_workflow") else "x.mp4"
    with pytest.raises(DomainError) as error:
        encode(root, source, name, hdr_policy="passthrough", **kwargs)
    assert error.value.code == "HDR_PASSTHROUGH_UNSUPPORTED"


def y_planes(path):
    with av.open(str(path)) as container:
        return [fidelity.luma_plane(frame) for frame in container.decode(video=0)]


def test_lossless_hdr_passthrough_is_bit_exact_and_keeps_pq_tags(root):
    source = hdr_source(root, frames=3, lossless=True)
    manifest = encode(root, source, "pq.mkv", profile="lossless_intermediate", hdr_policy="passthrough")
    out = out_path(root, manifest)
    assert tags(out) == {"primaries": 9, "transfer": 16, "matrix": 9, "range": 1}
    info = probe(out).videos[0]
    assert info.pixel_format == "yuv420p10le" and info.bit_depth == 10 and info.color.dynamic_range == "HDR_PQ"
    assert all(np.array_equal(a, b) for a, b in zip(y_planes(source), y_planes(out))) and len(y_planes(out)) == 3
    claims = manifest["stage15"]["color"]["claims"]
    assert claims["hdr_output"] == "PASSTHROUGH_VERIFIED" and manifest["stage15"]["color"]["path"] == "native_planes"
    assert all(item.get("exact") for item in manifest["stage15"]["fidelity"])
    # FFV1 cannot carry MaxCLL / mastering display; that limit is stated rather than hidden.
    assert any("cannot carry HDR static metadata" in n for n in manifest["stage15"]["color"]["notes"])


def test_hdr_passthrough_to_hevc_keeps_the_static_metadata(root):
    source = hdr_source(root, frames=2, lossless=True)
    manifest = encode(root, source, "pq.mov", profile="color_preserving", hdr_policy="passthrough")
    out = out_path(root, manifest)
    info = probe(out).videos[0]
    assert info.codec == "hevc" and info.pixel_format == "yuv420p10le"
    assert tags(out) == {"primaries": 9, "transfer": 16, "matrix": 9, "range": 1}
    assert info.hdr is not None and info.hdr.max_cll == 1000 and info.hdr.max_luminance == 1000.0


def test_sdr_passthrough_policy_changes_nothing_for_sdr_input(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=1)
    manifest = encode(root, source, "o.mp4", hdr_policy="passthrough")
    assert manifest["stage15"]["color"]["claims"]["hdr_output"] == "NOT_APPLICABLE"


# --- export profiles: every preset really encodes and decodes --------------------------------------------
def render(frame):
    """A real transform: a simple deterministic edit so the RGB path is exercised."""
    image = frame.image.copy()
    image[:8] = 255 - image[:8]
    return image


CASES = [("standard_mp4", ".mp4", "h264", "yuv420p", 8), ("high_quality_mp4", ".mp4", "h264", "yuv420p", 8),
         ("archival_hevc10", ".mkv", "hevc", "yuv420p10le", 10), ("mov_workflow", ".mov", "prores", "yuv422p10le", 10),
         ("lossless_intermediate", ".mkv", "ffv1", "bgr0", 8), ("color_preserving", ".mov", "prores", "yuv422p10le", 10)]


@pytest.mark.parametrize("profile,suffix,codec,pix,bits", CASES)
def test_every_profile_encodes_a_transformed_render_and_the_output_decodes(root, profile, suffix, codec, pix, bits):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=1, frames=4, width=96, height=64)
    manifest = encode(root, source, "out" + suffix, profile=profile, transform=render)
    out = out_path(root, manifest)
    info = probe(out).videos[0]
    assert (info.codec, info.pixel_format, info.bit_depth) == (codec, pix, bits)
    assert manifest["verified_frames"] == manifest["encoded_frames"] == 4
    assert manifest["stage15"]["profile"] == profile and manifest["stage15"]["fidelity"]
    assert tags(out)["transfer"] == 1 and tags(out)["primaries"] == 1
    # The container bit depth never turns into a claim about processing precision.
    color = manifest["stage15"]["color"]
    assert color["processing_bit_depth"] == 8 and color["output_bit_depth"] == bits


def test_lossless_intermediate_decodes_to_the_exact_encoder_input(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=3, width=96, height=64)
    marks = []

    def capture(frame):
        image = render(frame)
        marks.append(image.copy())
        return image
    manifest = encode(root, source, "l.mkv", profile="lossless_intermediate", transform=capture)
    decoded = [f.image for f in VideoReader(out_path(root, manifest), hardware="software").frames()]
    assert len(decoded) == 3 and all(np.array_equal(a, b) for a, b in zip(marks, decoded))
    assert tags(out_path(root, manifest))["matrix"] == 0  # RGB storage: no matrix to get wrong


def test_color_preserving_profile_keeps_source_tags_for_non_709_material(root):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=9, primaries=9, trc=14, bits=10, full=False, frames=2)
    manifest = encode(root, source, "c.mov", profile="color_preserving", transform=render)
    out = out_path(root, manifest)
    assert tags(out) == {"primaries": 9, "transfer": 14, "matrix": 9, "range": 1}
    got = next(iter(VideoReader(out, hardware="software").frames())).image
    assert got.shape[:2] == (32, 64)


def test_native_path_reencodes_planes_without_an_rgb_round_trip(root):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=9, primaries=9, trc=14, bits=10, chroma="422", frames=3)
    manifest = encode(root, source, "n.mkv", profile="lossless_intermediate")
    out = out_path(root, manifest)
    assert manifest["stage15"]["color"]["path"] == "native_planes"
    assert probe(out).videos[0].pixel_format == "yuv422p10le" and tags(out)["matrix"] == 9
    assert all(np.array_equal(a, b) for a, b in zip(y_planes(source), y_planes(out)))


def test_profile_and_filename_must_agree_and_unknown_profiles_are_rejected(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=1)
    with pytest.raises(ValueError):
        encode(root, source, "wrong.mkv", profile="standard_mp4")
    with pytest.raises(ValueError):
        encode(root, source, "x.mp4", profile="uncompressed_dream")
    assert set(PROFILES) == {"standard_mp4", "high_quality_mp4", "archival_hevc10", "mov_workflow",
                             "lossless_intermediate", "image_sequence", "color_preserving"}


def test_standard_profile_is_byte_equivalent_to_the_pre_stage15_default(root):
    source = make_color_video(root / "roop_3d/in.mkv", matrix=1, frames=3)
    legacy = encode(root, source, "legacy.mp4")
    profiled = encode(root, source, "standard.mp4", profile="standard_mp4")
    a, b = out_path(root, legacy), out_path(root, profiled)
    assert [bytes(p) for p in av.open(str(a)).demux(video=0) if p.size] == [bytes(p) for p in av.open(str(b)).demux(video=0) if p.size]


# --- fidelity guard -----------------------------------------------------------------------------------------
def test_fidelity_check_detects_range_squeeze_and_matrix_swap_but_not_codec_noise():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    image[:4] = 0
    image[4:8] = 255
    fidelity.compare_frames(0, image, image, bias_tolerance=3, tail_tolerance=10)
    jitter = np.clip(image.astype(int) + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
    fidelity.compare_frames(0, image, jitter, bias_tolerance=3, tail_tolerance=10)
    squeezed = (image.astype(np.float32) * 219 / 255 + 16).astype(np.uint8)
    with pytest.raises(DomainError) as error:
        fidelity.compare_frames(0, image, squeezed, bias_tolerance=3, tail_tolerance=10)
    assert error.value.code == "OUTPUT_COLOR_DRIFT"
    with pytest.raises(DomainError):
        fidelity.compare_frames(0, image, image[::-1].copy()[:, :, ::-1][..., :], bias_tolerance=0, tail_tolerance=0)
