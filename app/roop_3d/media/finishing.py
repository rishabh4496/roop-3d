"""Local display-referred finishing. Data passes never enter this pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import cv2
import numpy as np
from pydantic import Field, model_validator

from roop_3d.core.settings import Contract


class GrainOptions(Contract):
    size: float = Field(default=1, ge=0.5, le=8)
    intensity: float = Field(default=0, ge=0, le=0.2)
    shadows: float = Field(default=1, ge=0, le=2)
    highlights: float = Field(default=0.5, ge=0, le=2)
    seed: int = Field(default=0, ge=0, le=2147483647, strict=True)


class FinishingOptions(Contract):
    cube: str | None = Field(default=None, max_length=1048576)
    grain: GrainOptions = GrainOptions()
    provenance: str = Field(default='none', pattern=r'^(none|sha256|hmac-sha256|c2pa)$')

    @model_validator(mode='after')
    def lut_valid(self):
        if self.cube is not None:
            parse_cube(self.cube)
        return self


class ExportOptions(Contract):
    baked_video: bool = True
    passes: tuple[str, ...] = Field(default=(), max_length=5)
    finishing: FinishingOptions = FinishingOptions()

    @model_validator(mode='after')
    def pass_contract(self):
        if len(set(self.passes)) != len(self.passes) or set(self.passes) - {'alpha', 'occluder', 'uv', 'normal', 'depth'}:
            raise ValueError('Select unique supported compositing passes')
        if not self.baked_video and not self.passes:
            raise ValueError('Select baked video or at least one pass')
        if not self.baked_video and (self.finishing.cube or self.finishing.grain.intensity or self.finishing.provenance != 'none'):
            raise ValueError('Finishing and embedded provenance require a baked video')
        return self


@dataclass(frozen=True)
class CubeLut:
    size: int
    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        p = np.clip((rgb - self.domain_min) / (self.domain_max - self.domain_min), 0, 1) * (self.size - 1)
        lo = np.floor(p).astype(np.int32)
        hi = np.minimum(lo + 1, self.size - 1)
        f = p - lo
        out = np.zeros_like(rgb, dtype=np.float32)
        for b in (0, 1):
            for g in (0, 1):
                for r in (0, 1):
                    idx = np.where(np.array([r, g, b]), hi, lo)
                    weight = np.prod(np.where(np.array([r, g, b]), f, 1 - f), axis=-1)
                    out += self.values[idx[..., 2], idx[..., 1], idx[..., 0]] * weight[..., None]
        return out


def parse_cube(text: str) -> CubeLut:
    if len(text.encode('utf-8')) > 1048576:
        raise ValueError('LUT exceeds 1 MiB')
    size, rows = None, []
    domains = {'DOMAIN_MIN': np.zeros(3, np.float32), 'DOMAIN_MAX': np.ones(3, np.float32)}
    seen = set()
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if re.fullmatch(r'TITLE\s+"[^"\r\n]*"', line):
            continue
        parts = line.split()
        if parts[0] in {'LUT_3D_SIZE', 'DOMAIN_MIN', 'DOMAIN_MAX'}:
            key = parts[0]
            if key in seen or rows:
                raise ValueError('Duplicate or misplaced CUBE header')
            seen.add(key)
            if key == 'LUT_3D_SIZE':
                if len(parts) != 2 or not parts[1].isdigit() or not 2 <= int(parts[1]) <= 33:
                    raise ValueError('3D LUT size must be 2..33')
                size = int(parts[1])
            else:
                if len(parts) != 4:
                    raise ValueError('Invalid CUBE domain')
                domains[key] = np.array(parts[1:], np.float32)
        else:
            if size is None or len(parts) != 3 or len(rows) >= size ** 3:
                raise ValueError('Expected a bounded 3D CUBE LUT (1D LUTs are unsupported)')
            rows.append(tuple(float(value) for value in parts))
    if size is None or len(rows) != size ** 3:
        raise ValueError('CUBE sample count differs from LUT_3D_SIZE cubed')
    values = np.array(rows, np.float32).reshape(size, size, size, 3)
    low, high = domains['DOMAIN_MIN'], domains['DOMAIN_MAX']
    if not np.isfinite(values).all() or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError('LUT values and domains must be finite with increasing domain bounds')
    return CubeLut(size, values, low, high)


class Finisher:
    def __init__(self, options: FinishingOptions):
        self.options = options
        self.lut = parse_cube(options.cube) if options.cube else None

    def apply(self, bgr: np.ndarray, index: int) -> np.ndarray:
        if self.lut is None and not self.options.grain.intensity:
            return bgr
        rgb = bgr[..., ::-1].astype(np.float32) / 255
        if self.lut:
            rgb = self.lut.apply(rgb)
        grain = self.options.grain
        if grain.intensity:
            h, w = rgb.shape[:2]
            rng = np.random.default_rng(np.random.SeedSequence([grain.seed, index]))
            noise = rng.standard_normal((max(1, round(h / grain.size)), max(1, round(w / grain.size)))).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            deviation = float(noise.std())
            if deviation > 1e-6:
                noise /= deviation
            luma = np.clip(rgb @ np.array([.2126, .7152, .0722], np.float32), 0, 1)
            amplitude = grain.shadows * (1 - luma) + grain.highlights * luma
            rgb += (noise * amplitude * grain.intensity)[..., None]
        return np.rint(np.clip(rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)

    def metadata(self) -> dict:
        return {'lut_sha256': hashlib.sha256(self.options.cube.encode()).hexdigest() if self.options.cube else None,
                'lut_interpolation': 'trilinear', 'color_domain': 'display-referred RGB; no automatic log/HDR conversion',
                'grain': self.options.grain.model_dump(), 'order': ['LUT', 'grain']}
