"""Stage 15 colour science: every number is checked against a published reference value or an
independent derivation, never against the implementation's own output."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.color import codes, gamut, transfer
from roop_3d.color.description import (
    HdrStaticMetadata, merge_hdr_metadata, parse_content_light, parse_mastering_display,
    pixel_format_traits, resolve_color, working_color)
from roop_3d.color.hdr import HdrToSdr, ToneMapParams, bt2390_eetf
from roop_3d.core.errors import DomainError


# --- transfer functions -----------------------------------------------------------------------------
@pytest.mark.parametrize("nits,signal", [(0.0, 0.0), (100.0, 0.5081), (203.0, 0.5807), (1000.0, 0.7518), (10000.0, 1.0)])
def test_pq_matches_published_reference_levels(nits, signal):
    # ITU-R BT.2408 Table 1 / SMPTE ST 2084 reference levels.
    assert float(transfer.pq_inverse_eotf(nits)) == pytest.approx(signal, abs=5e-4)
    assert float(transfer.pq_eotf(signal)) == pytest.approx(nits, rel=2e-3, abs=0.05)


def test_pq_round_trip_is_exact_over_the_whole_range():
    codes_ = np.linspace(0, 1, 4097)
    back = transfer.pq_inverse_eotf(transfer.pq_eotf(codes_))
    assert np.allclose(back[1:], codes_[1:], atol=1e-9)
    # ST 2084's inverse EOTF is c1**m2 (~7.3e-7) at zero luminance, so black does not return to exactly 0.
    assert back[0] == pytest.approx(transfer.PQ_C1 ** transfer.PQ_M2) and back[0] < 1e-6


def test_hlg_oetf_matches_bt2100_anchor_points_and_round_trips():
    assert float(transfer.hlg_oetf(1 / 12)) == pytest.approx(0.5, abs=1e-12)
    assert float(transfer.hlg_oetf(1.0)) == pytest.approx(1.0, abs=1e-6)
    scene = np.linspace(0, 1, 2049)
    assert np.allclose(transfer.hlg_inverse_oetf(transfer.hlg_oetf(scene)), scene, atol=1e-9)


def test_hlg_reference_white_is_203_nits_on_a_1000_nit_display():
    # BT.2408: 75% HLG signal is reference white and is displayed at 203 cd/m^2 with the 1.2 system gamma.
    scene = float(transfer.hlg_inverse_oetf(0.75))
    nits = transfer.hlg_ootf(np.full(3, scene), 1000.0)
    assert nits[0] == pytest.approx(203.0, abs=0.5)
    assert transfer.hlg_system_gamma(1000.0) == pytest.approx(1.2)
    assert transfer.hlg_system_gamma(2000.0) > 1.2 > transfer.hlg_system_gamma(500.0)


def test_sdr_transfers_are_inverse_pairs_and_monotonic():
    values = np.linspace(0, 1, 1025)
    for name in transfer.BLEND_TRANSFERS:
        linear = transfer.to_linear(name, values)
        assert np.all(np.diff(linear) >= -1e-12), name
        assert np.allclose(transfer.from_linear(name, linear), values, atol=1e-6), name
    assert float(transfer.to_linear("srgb", 0.5)) == pytest.approx(0.21404, abs=1e-5)
    assert float(transfer.to_linear("bt1886", 0.5)) == pytest.approx(0.5 ** 2.4, abs=1e-12)
    assert float(transfer.bt709_oetf(0.18)) == pytest.approx(0.4090, abs=1e-3)
    with pytest.raises(ValueError):
        transfer.to_linear("pq", 0.5)


def test_blend_transfer_selection_follows_the_tagged_transfer():
    assert transfer.blend_transfer_for(codes.TRC_SRGB) == "srgb"
    assert transfer.blend_transfer_for(codes.TRC_BT709) == "bt1886"
    assert transfer.blend_transfer_for(codes.TRC_BT2020_10) == "bt1886"
    assert transfer.blend_transfer_for(8) == "linear"
    assert transfer.blend_transfer_for(4) == "gamma22"


# --- gamut and YCbCr --------------------------------------------------------------------------------
def test_bt709_rgb_to_xyz_matches_the_published_matrix():
    published = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
    assert np.allclose(gamut.rgb_to_xyz_matrix(1), published, atol=6e-4)


def test_bt2020_to_bt709_matrix_matches_itu_r_bt2087():
    published = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
    matrix = gamut.conversion_matrix(9, 1)
    assert np.allclose(matrix, published, atol=6e-4)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9)  # white maps to white
    assert np.allclose(gamut.conversion_matrix(1, 9) @ matrix, np.eye(3), atol=1e-9)
    assert np.allclose(gamut.conversion_matrix(1, 9), [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114],
                                                       [0.0164, 0.0880, 0.8956]], atol=6e-4)


def test_luma_coefficients_match_the_standards():
    assert np.allclose(gamut.luma_coefficients(1), [0.2126, 0.7152, 0.0722], atol=6e-4)
    assert np.allclose(gamut.luma_coefficients(9), [0.2627, 0.6780, 0.0593], atol=6e-4)
    assert gamut.matrix_weights(9) == (0.2627, 0.0593)
    with pytest.raises(ValueError):
        gamut.rgb_to_xyz_matrix(11)  # DCI-P3 needs chromatic adaptation and is refused, not guessed


@pytest.mark.parametrize("matrix,red", [(1, (63, 102, 240)), (5, (81, 90, 240)), (9, (74, 97, 240))])
def test_limited_range_primary_red_matches_known_code_values(matrix, red):
    # Well known 8-bit studio-range codes for 100% red under each matrix.
    got = gamut.rgb_to_ycbcr([1, 0, 0], matrix, False, 8)
    assert [int(round(v)) for v in got] == list(red)


def test_white_and_black_quantisation_by_range_and_depth():
    assert [round(v) for v in gamut.rgb_to_ycbcr([1, 1, 1], 1, False, 8)] == [235, 128, 128]
    assert [round(v) for v in gamut.rgb_to_ycbcr([0, 0, 0], 1, False, 8)] == [16, 128, 128]
    assert [round(v) for v in gamut.rgb_to_ycbcr([1, 1, 1], 1, True, 8)] == [255, 128, 128]
    assert [round(v) for v in gamut.rgb_to_ycbcr([1, 1, 1], 9, False, 10)] == [940, 512, 512]
    assert [round(v) for v in gamut.rgb_to_ycbcr([1, 1, 1], 9, True, 10)] == [1023, 512, 512]


def test_ycbcr_round_trip_every_supported_matrix():
    rng = np.random.default_rng(3)
    rgb = rng.random((200, 3))
    for matrix in gamut.SUPPORTED_MATRICES:
        for full in (False, True):
            for bits in (8, 10, 12):
                back = gamut.ycbcr_to_rgb(gamut.rgb_to_ycbcr(rgb, matrix, full, bits), matrix, full, bits)
                assert np.allclose(back, rgb, atol=1e-9)


# --- BT.2390 EETF and HDR -> SDR --------------------------------------------------------------------
def test_eetf_is_identity_below_the_knee_continuous_monotonic_and_reaches_the_target_peak():
    signal = np.linspace(0, 1, 4001) * float(transfer.pq_inverse_eotf(1000))
    out = bt2390_eetf(signal, 1000, 100)
    peak = float(transfer.pq_inverse_eotf(100))
    # Knee = 1.5*maxLum - 0.5 in normalised PQ (0.5137 of the 1000-nit range, i.e. signal ~0.386).
    low = signal < 0.38
    assert np.allclose(out[low], signal[low], atol=1e-6)  # PQ(0) is c1**m2 = 7.3e-7, not 0
    assert np.all(np.diff(out) >= -1e-12)
    assert out.max() == pytest.approx(peak, abs=1e-9)
    assert np.abs(np.diff(out)).max() < 1e-3  # no jump at the knee
    # Nothing to compress when the target peak is not below the source peak.
    assert np.allclose(bt2390_eetf(signal, 1000, 1000), signal, atol=1e-6)


@pytest.fixture(scope="module")
def pq_converter():
    return HdrToSdr(codes.TRC_PQ, codes.PRI_BT2020, ToneMapParams(source_peak_nits=1000))


def _pq_gray(nits):
    code = np.rint(transfer.pq_inverse_eotf(np.asarray(nits, float)) * 65535).astype(np.uint16)
    return np.repeat(code[:, None, None], 3, axis=2)


def test_hdr_to_sdr_preserves_midtone_exposure_and_orders_the_highlights(pq_converter):
    nits = np.array([0.0, 5.0, 18.0, 36.5, 100.0, 203.0, 400.0, 1000.0, 4000.0])
    linear = pq_converter.to_relative_linear(_pq_gray(nits))[:, 0, 0]
    assert linear[0] == pytest.approx(0.0, abs=1e-4)
    # 18% grey of a 203-nit reference white (36.5 nit) must land on SDR 18% grey: exposure preserved.
    assert linear[3] == pytest.approx(0.18, abs=0.005)
    assert np.all(np.diff(linear) >= 0)
    assert 0.70 < linear[5] < 0.90          # diffuse white sits on the broadcast ~90% signal level
    assert linear[7] == pytest.approx(1.0, abs=0.01) and linear[8] == pytest.approx(1.0, abs=0.01)
    eight = pq_converter.convert(_pq_gray(nits))[:, 0, 0]
    assert eight[0] == 0 and eight[-1] == 255
    assert eight[5] == round(255 * float(linear[5]) ** (1 / 2.4))


def test_hdr_to_sdr_lut_agrees_with_the_direct_formula(pq_converter):
    rng = np.random.default_rng(5)
    code = rng.integers(0, 65536, 500).astype(np.uint16)
    scale = 100 / 203
    nits = transfer.pq_eotf(code / 65535.0) * scale
    peak = max(1000 * scale, 100)
    expected = transfer.pq_eotf(bt2390_eetf(transfer.pq_inverse_eotf(nits), peak, 100)) / 100
    got = pq_converter._code_lut[code]
    assert np.allclose(got, expected, atol=2e-4)


def test_hdr_to_sdr_converts_wide_gamut_primaries_in_linear_light(pq_converter):
    # A saturated BT.2020 green is outside BT.709: the negative R/B must be clipped, not wrapped.
    pixel = np.zeros((1, 1, 3), np.uint16)
    pixel[0, 0] = (0, int(transfer.pq_inverse_eotf(150) * 65535), 0)
    rgb = pq_converter.convert(pixel)[0, 0]
    assert rgb[1] > 150 and rgb[0] == 0 and rgb[2] == 0
    assert pq_converter.to_relative_linear(pixel)[0, 0, 0] < 0  # unclipped value shows the gamut excursion


def test_hlg_path_renders_reference_white_like_pq_does():
    hlg = HdrToSdr(codes.TRC_HLG, codes.PRI_BT2020, ToneMapParams(source_peak_nits=1000))
    white = np.full((1, 1, 3), int(round(transfer.hlg_oetf(transfer.hlg_inverse_oetf(0.75)) * 65535)), np.uint16)
    pq = HdrToSdr(codes.TRC_PQ, codes.PRI_BT2020, ToneMapParams(source_peak_nits=1000))
    pq_white = _pq_gray([203.15]).reshape(1, 1, 3)
    assert hlg.to_relative_linear(white)[0, 0, 1] == pytest.approx(pq.to_relative_linear(pq_white)[0, 0, 1], abs=0.01)
    black = np.zeros((1, 1, 3), np.uint16)
    assert hlg.convert(black).max() == 0
    assert hlg.convert(np.full((1, 1, 3), 65535, np.uint16)).min() >= 250


def test_hdr_converter_validates_its_inputs(pq_converter):
    with pytest.raises(ValueError):
        HdrToSdr(codes.TRC_BT709, codes.PRI_BT2020)
    with pytest.raises(ValueError):
        HdrToSdr(codes.TRC_PQ, 11)
    with pytest.raises(ValueError):
        pq_converter.convert(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(ValueError):
        ToneMapParams(sdr_peak_nits=300, reference_white_nits=203)
    assert ToneMapParams().fingerprint() != ToneMapParams(source_peak_nits=4000).fingerprint()


# --- description and resolution ---------------------------------------------------------------------
@pytest.mark.parametrize("name,bits,chroma,family,full", [
    ("yuv420p", 8, "420", "yuv", False), ("yuvj420p", 8, "420", "yuv", True), ("yuv422p10le", 10, "422", "yuv", False),
    ("yuv444p12le", 12, "444", "yuv", False), ("p010le", 10, "420", "yuv", False), ("nv12", 8, "420", "yuv", False),
    ("gbrp", 8, "rgb", "rgb", True), ("gbrp10le", 10, "rgb", "rgb", True), ("bgr24", 8, "rgb", "rgb", True),
    ("rgb48le", 16, "rgb", "rgb", True), ("bgr0", 8, "rgb", "rgb", True), ("gray", 8, "gray", "gray", True),
    ("yuva420p10le", 10, "420", "yuv", False)])
def test_pixel_format_traits(name, bits, chroma, family, full):
    traits = pixel_format_traits(name)
    assert (traits.bit_depth, traits.chroma, traits.family, traits.implies_full_range) == (bits, chroma, family, full)


def resolve(**tags):
    values = dict(primaries=2, transfer=2, matrix=2, range_=0, width=1920, height=1080, pix_fmt="yuv420p")
    values.update(tags)
    return resolve_color(**values)


def test_untagged_material_is_resolved_by_convention_and_every_assumption_is_recorded():
    hd = resolve()
    assert (hd.matrix, hd.primaries, hd.transfer, hd.range) == (1, 1, 1, 1)
    assert set(hd.assumed_fields) == {"matrix", "primaries", "transfer", "range"}
    assert resolve(width=720, height=576).matrix == codes.MAT_BT470BG
    assert resolve(width=720, height=480).matrix == codes.MAT_SMPTE170M
    assert resolve(width=640, height=480).primaries == codes.PRI_SMPTE170M
    assert resolve(pix_fmt="yuvj420p").range == codes.RANGE_FULL


def test_tagged_fields_are_never_overridden_and_say_so():
    pq = resolve(primaries=9, transfer=16, matrix=9, range_=1, pix_fmt="yuv420p10le")
    assert (pq.primaries, pq.transfer, pq.matrix, pq.range, pq.bit_depth) == (9, 16, 9, 1, 10)
    assert pq.dynamic_range == "HDR_PQ" and pq.assumed_fields == () and pq.wide_gamut
    assert resolve(transfer=18, primaries=9, matrix=9, range_=1).dynamic_range == "HDR_HLG"
    assert resolve(matrix=1, primaries=1, transfer=1, range_=2).range == codes.RANGE_FULL


def test_untagged_hdr_signalling_is_never_inferred():
    # 4K 10-bit BT.2020 with no transfer tag: could be anything, so it is SDR + a recorded assumption.
    uhd = resolve(width=3840, height=2160, matrix=9, primaries=9, pix_fmt="yuv420p10le", range_=1)
    assert uhd.dynamic_range == "SDR" and "transfer" in uhd.assumed_fields
    # An HDR transfer with no matrix tag implies BT.2020.
    assert resolve(transfer=16, pix_fmt="yuv420p10le").matrix == codes.MAT_BT2020NC


def test_rgb_pixel_formats_carry_the_identity_matrix_and_full_range():
    rgb = resolve(pix_fmt="gbrp")
    assert rgb.matrix == codes.MAT_RGB and rgb.range == codes.RANGE_FULL and rgb.chroma == "rgb"


def test_hdr_policy_is_explicit_and_never_silent():
    sdr = resolve(primaries=9, transfer=14, matrix=9, range_=1, pix_fmt="yuv420p10le")
    wc = working_color(sdr)
    assert wc.tags == {"primaries": 9, "transfer": 14, "matrix": 9, "range": 1} and wc.conversion is None
    pq = resolve(primaries=9, transfer=16, matrix=9, range_=1, pix_fmt="yuv420p10le")
    mapped = working_color(pq)
    assert mapped.tags == {"primaries": 1, "transfer": 1, "matrix": 1, "range": 1}
    assert mapped.conversion.startswith("hdr_pq_to_sdr_bt709") and mapped.tone_map["source_peak_nits"] == 1000.0
    assert any("tone-mapped" in note for note in mapped.notes) and any("assumed 1000" in n for n in mapped.notes)
    with pytest.raises(DomainError) as error:
        working_color(pq, "reject")
    assert error.value.code == "HDR_INPUT_REJECTED"
    raw = working_color(pq, "raw")
    assert raw.tags["transfer"] == 16 and any("not a valid HDR" in note for note in raw.notes)
    with pytest.raises(ValueError):
        working_color(pq, "keep")


def test_mastering_metadata_drives_the_tone_map_peak_and_is_parsed_exactly():
    import struct
    blob = struct.pack("<20i2i", 34000, 50000, 16000, 50000, 13250, 50000, 34500, 50000, 7500, 50000, 3000, 50000,
                       15635, 50000, 16450, 50000, 1, 10000, 40000000, 10000, 1, 1)
    meta = parse_mastering_display(blob)
    assert meta.primaries[0] == pytest.approx((0.68, 0.32)) and meta.primaries[1] == pytest.approx((0.265, 0.69))
    assert meta.white_point == pytest.approx((0.3127, 0.329)) and meta.max_luminance == 4000 and meta.min_luminance == 0.0001
    both = merge_hdr_metadata(meta, parse_content_light(struct.pack("<II", 1000, 400)))
    assert both.to_x265_params() == ("master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                                     "L(40000000,1):max-cll=1000,400")
    assert parse_mastering_display(b"short") is None and parse_content_light(b"") is None
    assert merge_hdr_metadata(None, None) is None and HdrStaticMetadata().to_x265_params() is None
    pq = resolve(primaries=9, transfer=16, matrix=9, range_=1, pix_fmt="yuv420p10le", hdr=both)
    assert working_color(pq).tone_map["source_peak_nits"] == 4000.0
