"""Lazy NVIDIA codec adapters. Pixel buffers stay on CUDA; packets/metadata do not.

The owner and consuming stream must outlive a borrowed decoded surface. These
adapters intentionally do not silently download pixels to a host fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import importlib
from pathlib import Path
import subprocess

from roop_3d.models.owned_files import owned_path


class GPUBackendUnavailable(RuntimeError):
    pass


def optional_module(name):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise GPUBackendUnavailable(f"Optional {name} backend unavailable: {error}") from error


def configure_gpu_runtime():
    from roop_3d.audit.project_paths import application_root
    from roop_3d.models.trt_builder import configure_native_paths
    configure_native_paths(application_root())


def cuda_tensor(surface):
    """Borrow DLPack storage with producer/consumer stream negotiation."""
    if not hasattr(surface, "__dlpack__"):
        raise GPUBackendUnavailable("Backend must export the stream-aware DLPack protocol")
    if surface.__dlpack_device__()[0] != 2:
        raise ValueError("Host surfaces are forbidden in the CUDA path")
    import torch
    tensor = torch.utils.dlpack.from_dlpack(surface)
    if not tensor.is_cuda:
        raise ValueError("Decoder returned a non-CUDA tensor")
    return tensor


@dataclass
class DeviceFrame:
    tensor: object
    owner: object
    index: int
    statistics: dict | None = None

    @property
    def motion(self):
        return motion_summary(self.statistics)


def motion_summary(statistics: dict | None, threshold_pixels=2.0):
    """Decode metadata is host-side, codec-dependent, and never pixel telemetry."""
    if not statistics:
        return {"available": False, "high_motion": None, "mean_qp": None}
    import numpy as np
    qp = np.asarray(statistics.get("qp_luma", []), dtype=float)
    magnitudes = []
    for reference in (0, 1):
        x = np.asarray(statistics.get(f"mv{reference}_x", []), dtype=float)
        y = np.asarray(statistics.get(f"mv{reference}_y", []), dtype=float)
        if x.size and x.shape == y.shape:
            magnitudes.append(np.hypot(x, y) / 4.0)
    motion = float(max(v.mean() for v in magnitudes)) if magnitudes else None
    return {"available": motion is not None, "mean_motion_pixels": motion,
            "high_motion": motion >= threshold_pixels if motion is not None else None,
            "mean_qp": float(qp.mean()) if qp.size else None}


class NvVideoDecoder:
    """Sequential RGB HWC uint8 decode; use inside the consuming CUDA stream.

    A yield is a lease: finish GPU use before advancing the iterator. The stream
    wait in finally prevents decoder surface reuse while kernels still read it.
    """
    backend = "PYNVVIDEOCODEC"

    def __init__(self, path: Path, *, device=0, statistics=False):
        self.path = owned_path(path)
        if not self.path.is_file():
            raise ValueError("Decoder input must be a local owned file")
        self.device, self.statistics = device, statistics

    def frames(self):
        import torch
        configure_gpu_runtime()
        nvc = optional_module("PyNvVideoCodec")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            decoder = nvc.SimpleDecoder(str(self.path), gpu_id=self.device,
                cuda_stream=stream.cuda_stream, use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
                enableDecodeStats=self.statistics)
            index = 0
            try:
                while True:
                    batch = decoder.get_batch_frames(1)
                    if not batch:
                        break
                    surface = batch[0]
                    pixels = cuda_tensor(surface)
                    if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != torch.uint8:
                        raise ValueError("Expected decoded RGB HWC uint8")
                    stats = (surface.ParseDecodeStats() if self.statistics
                             and getattr(surface, "decode_stats_size", 0) > 0 else None)
                    try:
                        yield DeviceFrame(pixels, surface, index, stats)
                    finally:
                        stream.synchronize()
                    index += 1
            finally:
                del decoder


def decode_image(path: Path, *, device=0):
    """nvImageCodec image decode; no host fallback disguised as zero-copy."""
    import torch
    configure_gpu_runtime()
    codec = optional_module("nvidia.nvimgcodec")
    path = owned_path(path)
    stream = torch.cuda.current_stream(device)
    decoder = codec.Decoder(device_id=device)
    image = decoder.read(str(path), cuda_stream=stream.cuda_stream)
    pixels = cuda_tensor(image)
    return DeviceFrame(pixels, (image, decoder), 0)


class NvVideoEncoder:
    """NVENC accepts RGBA (ABGR byte order on little endian), then stream-copy mux.

    Compressed packets cross host memory as required by the codec API. No raw
    frame crosses host memory. SEI objects follow the installed NVIDIA API.
    Only exact integer CFR rates are currently admitted by this encoder path.
    """
    def __init__(self, destination: Path, shape, rate: Fraction, *, codec="h264", device=0):
        import torch
        from roop_3d.hardware_streamer import _binary, _process_environment
        self.destination = owned_path(destination)
        if (self.destination.exists() or self.destination.suffix.lower() not in (".mp4", ".mov")
                or codec not in ("h264", "hevc") or len(shape) != 3 or shape[2] != 4
                or min(shape) < 1 or rate <= 0 or rate.denominator != 1):
            raise ValueError("Fresh MP4/MOV, RGBA shape, H264/HEVC and integer CFR required")
        configure_gpu_runtime()
        nvc = optional_module("PyNvVideoCodec")
        self.binary, self.environment = _binary("ffmpeg"), _process_environment()
        self.device, self.shape, self.rate = device, tuple(shape), rate
        self.stream = torch.cuda.current_stream(device)
        self.encoder = nvc.CreateEncoder(shape[1], shape[0], "ABGR", False,
            codec=codec, gpu_id=device, cudastream=self.stream.cuda_stream,
            fps=rate.numerator, bf=0)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting an unrelated elementary stream.
        self.raw_path = owned_path(self.destination.with_suffix(self.destination.suffix + "." + codec))
        self.raw = self.raw_path.open("xb")
        self.closed, self.count, self.codec = False, 0, codec

    def write(self, rgba, *, sei_messages=None):
        import torch
        if (self.closed or not rgba.is_cuda or rgba.device.index != self.device
                or tuple(rgba.shape) != self.shape or rgba.dtype != torch.uint8 or not rgba.is_contiguous()):
            raise ValueError("Expected contiguous RGBA uint8 on the encoder device")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise ValueError("Encode must use its owning compute stream")
        packet = self.encoder.Encode(rgba, 0, sei_messages) if sei_messages else self.encoder.Encode(rgba)
        self.stream.synchronize()  # Keep input allocation alive through native consumption.
        self._write_packets(packet)
        self.count += 1

    def _write_packets(self, packets):
        # 2.2 returns packet dictionaries (data/PTS/DTS); older APIs return bytes.
        if isinstance(packets, (bytes, bytearray, memoryview)):
            self.raw.write(packets)
        elif isinstance(packets, list):
            for packet in packets:
                self.raw.write(packet["data"])
        else:
            raise GPUBackendUnavailable("Unknown PyNvVideoCodec packet representation")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._write_packets(self.encoder.EndEncode())
        finally:
            self.raw.close()
        if not self.count:
            self.raw_path.unlink(missing_ok=True)
            raise ValueError("Cannot mux an empty encode")
        result = subprocess.run([self.binary, "-nostdin", "-v", "error", "-n", "-r", str(self.rate),
            "-f", self.codec, "-i", str(self.raw_path), "-c:v", "copy", "-movflags", "+faststart",
            str(self.destination)], capture_output=True, timeout=120, env=self.environment)
        if result.returncode:
            self.destination.unlink(missing_ok=True)
            raise RuntimeError("NVENC stream-copy mux failed: " + result.stderr.decode(errors="replace")[:2048])
        self.raw_path.unlink()

    def abort(self):
        if not self.closed:
            self.closed = True
            self.raw.close()
        self.raw_path.unlink(missing_ok=True)
        self.destination.unlink(missing_ok=True)
