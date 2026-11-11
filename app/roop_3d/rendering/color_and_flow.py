# SPDX-License-Identifier: MIT
"""Local `.cube` 3D LUTs, histogram matching and tangent-aligned hair shading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


def _image_rgb_float(image: np.ndarray) -> tuple[np.ndarray, np.dtype, bool]:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("image must be a non-empty BGR image with shape (H, W, 3)")
    integer = np.issubdtype(value.dtype, np.integer)
    scale = float(np.iinfo(value.dtype).max) if integer else 1.0
    rgb = value[..., ::-1].astype(np.float32) / scale
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("image values must be finite and lie in the dtype's normalized range")
    return rgb, value.dtype, integer


@dataclass(frozen=True)
class CubeLUT:
    """A 3D RGB lookup table, indexed as ``data[blue, green, red]``."""

    data: np.ndarray
    title: str = "Roop 3D LUT"
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        table = np.asarray(self.data, np.float32)
        if table.ndim != 4 or table.shape[0] != table.shape[1] or table.shape[1] != table.shape[2] or table.shape[3] != 3:
            raise ValueError("3D LUT data must have shape (N, N, N, 3)")
        if not 2 <= table.shape[0] <= 65 or not np.isfinite(table).all():
            raise ValueError("3D LUT size must be 2..65 and all entries finite")
        lo, hi = np.asarray(self.domain_min, np.float32), np.asarray(self.domain_max, np.float32)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
            raise ValueError("LUT domain bounds must contain three finite increasing values")
        object.__setattr__(self, "data", table)
        object.__setattr__(self, "title", str(self.title).replace("\n", " ").replace("\r", " "))
        object.__setattr__(self, "domain_min", tuple(float(v) for v in lo))
        object.__setattr__(self, "domain_max", tuple(float(v) for v in hi))

    @property
    def size(self) -> int:
        return int(self.data.shape[0])

    @classmethod
    def parse(cls, text: str) -> "CubeLUT":
        if not isinstance(text, str) or len(text.encode("utf-8")) > 64 * 1024 * 1024:
            raise ValueError(".cube text must be a string no larger than 64 MiB")
        size: int | None = None
        title = "Roop 3D LUT"
        domain_min, domain_max = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        rows: list[tuple[float, float, float]] = []
        for line_no, raw in enumerate(text.splitlines(), 1):
            line = raw.partition("#")[0].strip()
            if not line:
                continue
            if line.startswith("TITLE"):
                parts = line.split(None, 1)
                title = parts[1].strip().strip('"') if len(parts) > 1 else "Roop 3D LUT"
            elif line.startswith("LUT_1D_SIZE"):
                raise ValueError("1D LUTs are unsupported; use a .cube file with LUT_3D_SIZE")
            elif line.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"invalid LUT_3D_SIZE on line {line_no}")
                size = int(parts[1])
                if not 2 <= size <= 65:
                    raise ValueError("LUT_3D_SIZE must be within 2..65")
            elif line.startswith("DOMAIN_MIN") or line.startswith("DOMAIN_MAX"):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"invalid LUT domain on line {line_no}")
                values = tuple(float(v) for v in parts[1:])
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite LUT domain on line {line_no}")
                if parts[0] == "DOMAIN_MIN":
                    domain_min = values
                else:
                    domain_max = values
            else:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"unsupported .cube directive or malformed RGB row on line {line_no}")
                row = tuple(float(v) for v in parts)
                if not np.isfinite(row).all():
                    raise ValueError(f"non-finite LUT value on line {line_no}")
                rows.append(row)
        if size is None or len(rows) != size ** 3:
            raise ValueError(".cube file must define exactly LUT_3D_SIZE cubed RGB rows")
        # The cube format lists red fastest, followed by green, then blue.
        data = np.asarray(rows, np.float32).reshape(size, size, size, 3)
        return cls(data=data, title=title, domain_min=domain_min, domain_max=domain_max)

    @classmethod
    def load(cls, path: str | Path) -> "CubeLUT":
        source = Path(path)
        if (source.suffix.casefold() != ".cube" or not source.is_file() or source.is_symlink()
                or source.stat().st_size > 64 * 1024 * 1024):
            raise ValueError("LUT import requires a regular local .cube file")
        return cls.parse(source.read_text(encoding="utf-8-sig"))

    def to_text(self, *, precision: int = 8) -> str:
        if not 1 <= precision <= 12:
            raise ValueError("precision must be within 1..12")
        title = self.title.replace('"', "'")
        lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {self.size}",
                 "DOMAIN_MIN " + " ".join(f"{v:.{precision}f}" for v in self.domain_min),
                 "DOMAIN_MAX " + " ".join(f"{v:.{precision}f}" for v in self.domain_max)]
        lines.extend(" ".join(f"{float(channel):.{precision}f}" for channel in row)
                     for row in self.data.reshape(-1, 3))
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path, *, precision: int = 8) -> Path:
        target = Path(path)
        if target.suffix.casefold() != ".cube":
            raise ValueError("LUT export path must end in .cube")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_text(precision=precision), encoding="utf-8", newline="\n")
        return target

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        """Trilinearly sample the RGB cube while preserving BGR input/output."""
        rgb, dtype, integer = _image_rgb_float(image_bgr)
        low = np.asarray(self.domain_min, np.float32)
        high = np.asarray(self.domain_max, np.float32)
        p = np.clip((rgb - low) / (high - low), 0.0, 1.0) * (self.size - 1)
        r, g, b = p[..., 0], p[..., 1], p[..., 2]
        r0, g0, b0 = np.floor(r).astype(np.int32), np.floor(g).astype(np.int32), np.floor(b).astype(np.int32)
        r1, g1, b1 = np.minimum(r0 + 1, self.size - 1), np.minimum(g0 + 1, self.size - 1), np.minimum(b0 + 1, self.size - 1)
        fr, fg, fb = (r - r0)[..., None], (g - g0)[..., None], (b - b0)[..., None]
        d = self.data
        c000, c100 = d[b0, g0, r0], d[b0, g0, r1]
        c010, c110 = d[b0, g1, r0], d[b0, g1, r1]
        c001, c101 = d[b1, g0, r0], d[b1, g0, r1]
        c011, c111 = d[b1, g1, r0], d[b1, g1, r1]
        c00, c10 = c000 * (1 - fr) + c100 * fr, c010 * (1 - fr) + c110 * fr
        c01, c11 = c001 * (1 - fr) + c101 * fr, c011 * (1 - fr) + c111 * fr
        c0, c1 = c00 * (1 - fg) + c10 * fg, c01 * (1 - fg) + c11 * fg
        out = np.clip(c0 * (1 - fb) + c1 * fb, 0.0, 1.0)[..., ::-1]
        if integer:
            out = np.rint(out * np.iinfo(dtype).max).astype(dtype)
        else:
            out = out.astype(dtype)
        return out


class CubeLUTColorMatcher:
    """Read/write/apply .cube LUTs and build a deterministic histogram-match LUT."""

    @staticmethod
    def import_cube(path: str | Path) -> CubeLUT:
        return CubeLUT.load(path)

    @staticmethod
    def export_cube(lut: CubeLUT, path: str | Path, *, precision: int = 8) -> Path:
        return lut.save(path, precision=precision)

    @staticmethod
    def match_histograms(source_bgr: np.ndarray, target_bgr: np.ndarray, *, size: int = 17,
                         source_mask: np.ndarray | None = None, target_mask: np.ndarray | None = None,
                         title: str = "Roop 3D Histogram Match") -> CubeLUT:
        """Build a monotone per-channel quantile LUT from locally supplied footage.

        Source and target may be different sizes. Masks select the graded skin or
        scene region; absent masks use all pixels. The transform is an empirical
        RGB histogram match, not a learned look or a color-space inference.
        """
        if type(size) is not int or not 2 <= size <= 65:
            raise ValueError("size must be an integer within 2..65")
        source, _, _ = _image_rgb_float(source_bgr)
        target, _, _ = _image_rgb_float(target_bgr)

        def samples(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
            if mask is None:
                return image.reshape(-1, 3)
            selected = np.asarray(mask) > 0
            if selected.shape != image.shape[:2]:
                raise ValueError("histogram mask must match the image height and width")
            pixels = image[selected]
            if len(pixels) < 16:
                raise ValueError("histogram mask must select at least 16 pixels")
            return pixels

        src, dst = samples(source, source_mask), samples(target, target_mask)
        axis = np.linspace(0, 1, size, dtype=np.float32)
        grid_b, grid_g, grid_r = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack((grid_r, grid_g, grid_b), axis=-1)
        out = np.empty_like(points)
        for channel in range(3):
            src_sorted = np.sort(src[:, channel])
            dst_sorted = np.sort(dst[:, channel])
            ranks = np.searchsorted(src_sorted, points[..., channel], side="right") / max(1, len(src_sorted))
            out[..., channel] = np.quantile(dst_sorted, np.clip(ranks, 0, 1)).astype(np.float32)
        return CubeLUT(out, title=title)


class FlowMapAnisotropicShader:
    """Estimate local hair tangents and shade with an orientation-aligned lobe."""

    REGIONS = ("eyebrows", "hairline", "beard")

    @staticmethod
    def tangent_flow(image_bgr: np.ndarray, mask: np.ndarray | None = None, *, smooth_sigma: float = 2.0,
                     minimum_energy: float = 1e-5) -> np.ndarray:
        """Return HxWx3 float flow: signed tangent x/y and strand confidence."""
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("image must be a BGR image with shape (H, W, 3)")
        if not np.isfinite(smooth_sigma) or smooth_sigma < 0 or smooth_sigma > 64:
            raise ValueError("smooth_sigma must be within [0, 64]")
        if np.issubdtype(image.dtype, np.integer):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.iinfo(image.dtype).max
        else:
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("floating images must be finite and within [0, 1]")
            gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        sigma = max(float(smooth_sigma), 0.5)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + np.pi * 0.5
        tangent_x, tangent_y = np.cos(angle), np.sin(angle)
        anisotropy = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / np.maximum(jxx + jyy, minimum_energy)
        energy = np.sqrt(np.maximum(jxx + jyy, 0))
        energy_scale = max(float(np.quantile(energy, 0.98)), minimum_energy)
        confidence = np.clip(anisotropy * energy / energy_scale, 0, 1)
        if mask is not None:
            selected = np.asarray(mask, np.float32)
            if selected.shape != gray.shape or not np.isfinite(selected).all():
                raise ValueError("flow mask must be a finite image-sized array")
            confidence *= np.clip(selected, 0, 1)
        return np.stack((tangent_x, tangent_y, confidence), axis=-1).astype(np.float32)

    @classmethod
    def region_flows(cls, image_bgr: np.ndarray, masks: Mapping[str, np.ndarray], *,
                     smooth_sigma: float = 2.0) -> dict[str, np.ndarray]:
        unknown = set(masks) - set(cls.REGIONS)
        if unknown:
            raise ValueError(f"unknown hair flow regions: {sorted(unknown)}")
        return {name: cls.tangent_flow(image_bgr, mask, smooth_sigma=smooth_sigma) for name, mask in masks.items()}

    @staticmethod
    def shade(base_bgr: np.ndarray, flow_map: np.ndarray, *, light_direction: tuple[float, float, float] = (-0.3, -0.4, 0.86),
              view_direction: tuple[float, float, float] = (0.0, 0.0, 1.0), roughness: float = 0.28,
              strength: float = 0.25, light_color_bgr: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """Add an anisotropic specular lobe whose long axis follows the flow map."""
        base = np.asarray(base_bgr)
        flow = np.asarray(flow_map, np.float32)
        if (base.ndim != 3 or base.shape[2] != 3 or flow.shape != (*base.shape[:2], 3)
                or not np.isfinite(flow).all()):
            raise ValueError("base and flow_map must be matching HxWx3 arrays")
        if not 0.02 <= roughness <= 1 or not 0 <= strength <= 4:
            raise ValueError("roughness must be .02..1 and strength must be 0..4")
        light = np.asarray(light_direction, np.float32)
        view = np.asarray(view_direction, np.float32)
        color = np.asarray(light_color_bgr, np.float32)
        if (light.shape != (3,) or view.shape != (3,) or color.shape != (3,) or not np.isfinite(light).all()
                or not np.isfinite(view).all() or not np.isfinite(color).all()
                or np.linalg.norm(light) <= 1e-6 or np.linalg.norm(view) <= 1e-6 or np.any(color < 0)):
            raise ValueError("light_direction and view_direction must be nonzero 3-vectors")
        light /= np.linalg.norm(light)
        view /= np.linalg.norm(view)
        half = light + view
        half /= max(float(np.linalg.norm(half)), 1e-6)
        tx, ty = flow[..., 0], flow[..., 1]
        norm = np.sqrt(tx * tx + ty * ty)
        tx, ty = tx / np.maximum(norm, 1e-6), ty / np.maximum(norm, 1e-6)
        dot_t = tx * half[0] + ty * half[1]
        dot_b = -ty * half[0] + tx * half[1]
        dot_n = max(float(half[2]), 1e-4)
        alpha_t = max(0.015, roughness * roughness * 0.55)
        alpha_b = max(alpha_t, roughness * roughness * 1.8)
        exponent = -((dot_t / alpha_t) ** 2 + (dot_b / alpha_b) ** 2) / (dot_n * dot_n)
        lobe = np.exp(np.clip(exponent, -60, 0)) / max(4 * np.pi * alpha_t * alpha_b * dot_n ** 4, 1e-4)
        highlight = np.clip(lobe * 0.004 * flow[..., 2] * max(float(light[2]), 0) * strength, 0, 1)
        if np.issubdtype(base.dtype, np.integer):
            scale = float(np.iinfo(base.dtype).max)
            output = base.astype(np.float32) / scale
        else:
            if not np.isfinite(base).all() or base.min() < 0 or base.max() > 1:
                raise ValueError("floating base image must be finite and within [0, 1]")
            scale = 1.0
            output = base.astype(np.float32)
        output = output + highlight[..., None] * color[None, None, :]
        output = np.clip(output, 0, 1)
        return np.rint(output * scale).astype(base.dtype) if np.issubdtype(base.dtype, np.integer) else output
