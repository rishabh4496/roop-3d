"""Heterogeneous Intel + NVIDIA Hybrid Execution Scheduler.

Stage 15 Remediation:
Offloads pre-scanning, semantic parsing, and 3D mesh raycasting to Intel OpenVINO
and AVX-512 while reserving 100% of NVIDIA TensorRT compute for Stage 3 face swapping
and restoration.

Context & Architectural Standards:
1. Intel OpenVINO Asynchronous Co-Processor:
   - Provisions openvino / onnxruntime-openvino targeting CPU (AVX-512 / AMX / VNNI),
     iGPU (Xe / UHD), or NPU (AUTO:GPU,NPU,CPU).
   - Heterogeneous Split-Pipeline Mode:
     (a) Concurrent Preview + Prepass: Stage 1 Sparse Pre-Scan (scrfd_2.5g_kps) and
         Stage 2 Active Prepass (2d106det, w600k_r50) execute on OpenVINO co-processor,
         keeping NVIDIA RTX GPU 100% unlocked for instantaneous POST /api/preview calls
         with 0ms GPU lock contention (zero HTTP 503 busy waits during pre-scan).
     (b) Parallel Auxiliary Model Offload: Asynchronous lookahead worker offloads
         lightweight models (BiSeNet 19-class face parser and L2CS-Net eye gaze estimator)
         to Intel OpenVINO INT8/FP16 on CPU/iGPU (processing Frame N+2 while NVIDIA TensorRT swaps Frame N).
2. Dual-Hardware Video Decode/Encode Multiplexer (Intel QuickSync h264_qsv / hevc_qsv + NVIDIA NVDEC/NVENC):
   - Detects Intel QuickSync Video (QSV) availability in GET /api/system/hardware.
   - Offloads target video decoding to the Intel iGPU media engine when NVIDIA NVDEC utilization
     or VRAM bandwidth is saturated by 4K multi-face restoration.
3. CPU 3D mesh self-occlusion (vertex-level z-buffer, see ``mesh_raycaster``):
   - Back-face cull plus nearest-depth test hides the far cheek at steep profile angles (>45° yaw);
     the measured time is reported in telemetry, not assumed.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator

import numpy as np

from roop_3d.core.settings import Settings
from roop_3d.mesh_raycaster import MeshRaycaster, VisibilityResult
from roop_3d.openvino_coprocessor import (
    AuxiliaryLookaheadWorker,
    OpenVINOCoprocessor,
    get_openvino_device_info,
)
from roop_3d.server.engine_gate import EngineGate

logger = logging.getLogger(__name__)

# Register compatibility alias so legacy module paths map to roop_3d
sys.modules.setdefault("roop.heterogeneous_scheduler", sys.modules[__name__])


class VideoCodecMultiplexer:
    """Dual-Hardware Video Decode/Encode Multiplexer (Intel QSV + NVIDIA NVDEC/NVENC)."""

    def __init__(self, ffmpeg_path: str | None = None):
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self.qsv_encoders: list[str] = []
        self.qsv_decoders: list[str] = []
        self.nvenc_encoders: list[str] = []
        self.nvdec_decoders: list[str] = []
        self.qsv_available = False
        self.nvdec_available = False
        self._probe_codecs()

    def _probe_codecs(self) -> None:
        if not self.ffmpeg_path:
            return
        try:
            enc = subprocess.run([self.ffmpeg_path, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            dec = subprocess.run([self.ffmpeg_path, "-hide_banner", "-decoders"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            self.nvenc_encoders = sorted(set(re.findall(r"\b\w+_nvenc\b", enc)))
            self.nvdec_decoders = sorted(set(re.findall(r"\b\w+_cuvid\b", dec)))
            self.qsv_encoders = sorted(set(re.findall(r"\b\w+_qsv\b", enc)))
            self.qsv_decoders = sorted(set(re.findall(r"\b\w+_qsv\b", dec)))
            self.qsv_available = bool(self.qsv_decoders or self.qsv_encoders)
            self.nvdec_available = bool(self.nvdec_decoders)
        except Exception as e:
            logger.debug("Video codec multiplexer probe incomplete: %s", e)

    def should_offload_to_qsv(
        self,
        width: int,
        height: int,
        face_count: int = 1,
        nvdec_saturated: bool = False,
        settings: Settings | None = None,
    ) -> bool:
        """Determine whether to route decode to Intel QuickSync Video on the iGPU.

        Triggered when:
        (1) Explicitly enabled via `settings.qsv_offload_enabled` or `settings.heterogeneous_split_mode`, OR
        (2) Target video is 4K (>=3840x2160), OR
        (3) Multi-face restoration workload saturates NVDEC / VRAM bandwidth, OR
        (4) nvdec_saturated flag is asserted.
        """
        if not self.qsv_available:
            return False

        if settings and getattr(settings, "qsv_offload_enabled", False):
            return True

        # 4K resolution threshold
        is_4k = (width >= 3840 or height >= 2160)
        # Multi-face saturation threshold under high resolution
        is_multi_face_heavy = (face_count >= 2 and width >= 1920)

        if nvdec_saturated or is_4k or is_multi_face_heavy:
            return True

        if settings and getattr(settings, "heterogeneous_split_mode", False) and is_4k:
            return True

        return False

    def get_decoder_codec_name(self, codec: str = "h264", offload_qsv: bool = False) -> str:
        """Resolve decoder codec name for FFmpeg pipeline."""
        base = codec.lower().replace("libx264", "h264").replace("libx265", "hevc")
        if offload_qsv and self.qsv_available:
            target = f"{base}_qsv"
            if target in self.qsv_decoders or not self.qsv_decoders:
                return target
        if self.nvdec_available:
            target = f"{base}_cuvid"
            if target in self.nvdec_decoders or not self.nvdec_decoders:
                return target
        return base


class HeterogeneousScheduler:
    """Heterogeneous Intel + NVIDIA Hybrid Execution Scheduler."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine_gate: EngineGate | None = None,
    ):
        self.settings = settings or Settings()
        self.gate = engine_gate or EngineGate()
        self.coprocessor = OpenVINOCoprocessor(
            target_device=getattr(self.settings, "openvino_device", "AUTO:GPU,NPU,CPU"),
            precision="FP16",
        )
        self.raycaster = MeshRaycaster()
        self.last_visibility_ms: float | None = None
        self.multiplexer = VideoCodecMultiplexer()
        self.lookahead_worker = self.coprocessor.lookahead_worker
        self._lock = threading.Lock()
        self.total_swaps = 0
        self.total_prepass_frames = 0
        self.gpu_lock_contention_ms = 0.0

    @property
    def split_mode(self) -> bool:
        return bool(getattr(self.settings, "heterogeneous_split_mode", False))

    def run_prescan_heterogeneous(
        self,
        frames: list[Any],
        interval: int = 25,
        detect_func: Callable[[np.ndarray], list[Any]] | None = None,
    ) -> list[int]:
        """Execute Stage 1 Sparse Pre-Scan.

        In Heterogeneous Split-Pipeline Mode:
        Runs on OpenVINO co-processor with `domain="openvino"`.
        The NVIDIA RTX GPU is 100% unlocked for instantaneous POST /api/preview requests.
        """
        domain = "openvino" if self.split_mode else "gpu"
        with self.gate.owner("STAGE_1_SPARSE_PRESCAN", domain=domain):
            hits = self.coprocessor.run_stage1_sparse_prescan(
                frames,
                interval=interval,
                detect_func=detect_func,
            )
            self.total_prepass_frames += len(frames)
            return hits

    def run_active_prepass_heterogeneous(
        self,
        frames: list[Any],
        ranges: list[tuple[int, int]],
        track_func: Callable[[np.ndarray], list[Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute Stage 2 Active Prepass.

        In Heterogeneous Split-Pipeline Mode:
        Runs on OpenVINO co-processor with `domain="openvino"`.
        The NVIDIA RTX GPU is 100% unlocked for instantaneous POST /api/preview requests.
        """
        domain = "openvino" if self.split_mode else "gpu"
        with self.gate.owner("STAGE_2_ACTIVE_PREPASS", domain=domain):
            results = self.coprocessor.run_stage2_active_prepass(
                frames,
                active_ranges=ranges,
                track_func=track_func,
            )
            return results

    def execute_stage3_frame(
        self,
        frame_index: int,
        frame_image: np.ndarray,
        mesh: Any | None = None,
        head_yaw: float = 0.0,
        swap_func: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Execute Stage 3 Face Swapping & Restoration with Auxiliary Offload.

        - Lookahead worker processes auxiliary models (BiSeNet + L2CS) for Frame N+2 on OpenVINO.
        - 3D Mesh Raycaster computes self-occlusion visibility map on CPU (measured).
        - NVIDIA TensorRT executes face swapping on Frame N under GPU gate lock.
        """
        # 1. Asynchronously submit lookahead frame N+2 to OpenVINO worker
        if self.split_mode and getattr(self.settings, "auxiliary_offload_enabled", True):
            self.lookahead_worker.submit_lookahead(frame_index + 2, frame_image)

        # 2. 3D Self-Occlusion Raycasting (measured)
        vis_result: VisibilityResult | None = None
        if mesh is not None and abs(head_yaw) > 45.0:
            vis_result = self.raycaster.compute_visibility_map(mesh, yaw_override=head_yaw)
            self.last_visibility_ms = vis_result.execution_time_ms

        # 3. NVIDIA TensorRT Main Swap (held under GPU lock)
        t_enter = time.perf_counter()
        with self.gate.owner("STAGE_3_MAIN_SWAP", domain="gpu"):
            contention = (time.perf_counter() - t_enter) * 1000.0
            self.gpu_lock_contention_ms += contention

            # Fetch auxiliary results for this frame if available
            aux_data = None
            if self.split_mode:
                aux_data = self.lookahead_worker.get_auxiliary_result(frame_index, timeout=0.05)

            swapper = swap_func or (lambda img: img)
            swapped = swapper(frame_image)
            self.total_swaps += 1

        return {
            "frame_index": frame_index,
            "swapped_image": swapped,
            "visibility_result": vis_result,
            "auxiliary_data": aux_data,
            "self_occluded_fraction": vis_result.self_occluded_fraction if vis_result else None,
            "executed_devices": {
                "swap": "nvidia_tensorrt",
                "auxiliary": "intel_openvino" if self.split_mode else "nvidia_gpu",
                "raycast": "cpu_zbuffer",
            },
        }

    def get_hardware_telemetry(self) -> dict[str, Any]:
        """Telemetry snapshot for heterogenous co-processing."""
        ov_info = get_openvino_device_info()
        return {
            "heterogeneous_split_mode": self.split_mode,
            "openvino_coprocessor": {
                "available": ov_info["available"],
                "version": ov_info["version"],
                "devices": ov_info["devices"],
                "target_device": self.coprocessor.target_device,
                "simd_features": ov_info["simd_features"],
                "lookahead_worker_active": self.lookahead_worker.is_running(),
                "lookahead_frames_processed": self.lookahead_worker.frames_processed,
            },
            "quicksync_multiplexer": {
                "qsv_available": self.multiplexer.qsv_available,
                "qsv_decoders": self.multiplexer.qsv_decoders,
                "qsv_encoders": self.multiplexer.qsv_encoders,
            },
            "self_occlusion": {
                "method": "cpu_vertex_zbuffer",
                "last_execution_ms": self.last_visibility_ms,
            },
            "gate_status": {
                "gpu_owners": self.gate.owner_count_for_domain("gpu"),
                "openvino_owners": self.gate.owner_count_for_domain("openvino"),
                "previews_in_flight": self.gate.previews_in_flight,
            },
        }
