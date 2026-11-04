"""ITU-T H.273 / FFmpeg colour code points, with names and explicit "unspecified" handling.

FFmpeg and H.273 share numeric values: 1 is BT.709 in every table, 9 is BT.2020 for primaries and
for the non-constant-luminance matrix, 16 is SMPTE ST 2084 (PQ) and 18 is ARIB STD-B67 (HLG). Both 0
(reserved) and 2 (unspecified) mean "the file does not say"; range uses 0 for that.
"""
from __future__ import annotations

PRIMARIES = {1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m", 8: "film",
             9: "bt2020", 10: "smpte428", 11: "smpte431", 12: "smpte432", 22: "ebu3213"}
TRANSFER = {1: "bt709", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 7: "smpte240m", 8: "linear",
            13: "iec61966-2-1", 14: "bt2020-10", 15: "bt2020-12", 16: "smpte2084", 17: "smpte428",
            18: "arib-std-b67"}
MATRIX = {0: "rgb", 1: "bt709", 4: "fcc", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
          8: "ycgco", 9: "bt2020nc", 10: "bt2020c"}
RANGE = {0: "unspecified", 1: "limited", 2: "full"}

PRI_BT709, PRI_BT470BG, PRI_SMPTE170M, PRI_BT2020, PRI_P3D65 = 1, 5, 6, 9, 12
TRC_BT709, TRC_SMPTE170M, TRC_SRGB, TRC_BT2020_10, TRC_PQ, TRC_HLG = 1, 6, 13, 14, 16, 18
MAT_RGB, MAT_BT709, MAT_BT470BG, MAT_SMPTE170M, MAT_BT2020NC = 0, 1, 5, 6, 9
RANGE_LIMITED, RANGE_FULL = 1, 2

#: Value reported by FFmpeg for an untagged field. 0 is also accepted as "not specified".
UNSPECIFIED_CODES = (0, 2)


def is_unspecified(field: str, value: int | None) -> bool:
    if value is None:
        return True
    if field == "range":
        return value == 0
    return value in UNSPECIFIED_CODES


def name_of(field: str, value: int | None) -> str:
    table = {"primaries": PRIMARIES, "transfer": TRANSFER, "matrix": MATRIX, "range": RANGE}[field]
    if is_unspecified(field, value):
        return "unspecified"
    return table.get(int(value), f"code-{int(value)}")
