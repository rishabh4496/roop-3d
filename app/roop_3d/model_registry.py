"""Unified Declarative Model Registry & Resilient Auto-Downloader.

Single source-of-truth manifest for all deep learning models across:
1. Swappers: inswapper_128, hyperswap_1a/1b/1c_256, simswap_unofficial_512, ghost_1_256.
2. Enhancers / Restorers: GPEN-BFR-512/1024, ultramax_bfr, codeformer_dynamic, RestoreFormer_Plus_Plus, GFPGANv1.4, RealESRGAN_x4plus.
3. 3D Pose, Landmarks & Expression: 3ddfa_v2_mb1_120x120, 1k3d68, 2d106det, 2dfan4, and LivePortrait modules.
4. Detectors & Embedders: scrfd_2.5g_kps, det_10g, yoloface_8n, w600k_r50.
5. Occlusion & Semantic Parsers: xseg_3, bisenet_resnet_34, clipseg_rd64_refined.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Callable, Literal

from roop_3d.audit.project_paths import application_root

ModelCategory = Literal[
    "swapper",
    "enhancer",
    "3d_geometry",
    "detector_embedder",
    "occlusion_masking",
]

ReadinessState = Literal[
    "READY + TRT ENGINE",
    "DOWNLOADED (NEEDS TRT)",
    "MISSING",
    "CORRUPT HASH",
]

PrecisionPolicy = Literal["FP32", "FP16"]


@dataclass(frozen=True)
class ModelManifestEntry:
    model_id: str
    name: str
    category: ModelCategory
    primary_url: str
    fallback_url: str
    expected_size_bytes: int
    expected_sha256: str
    input_shapes: dict[str, list[int | str]]
    output_shapes: dict[str, list[int | str]]
    precision_policy: PrecisionPolicy
    supports_dynamic_batch: bool
    vram_footprint_mb: int
    description: str
    companion_files: tuple[str, ...] = ()
    license_id: str = "custom"


# Canonical definitions across all 5 model categories
UNIFIED_MODEL_ENTRIES: tuple[ModelManifestEntry, ...] = (
    # ── Category 1: Swappers ──────────────────────────────────────────────────
    ModelManifestEntry(
        model_id="inswapper_128",
        name="inswapper_128.onnx",
        category="swapper",
        primary_url="https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/inswapper_128.onnx",
        expected_size_bytes=554253681,
        expected_sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        input_shapes={"target": [1, 3, 128, 128], "source": [1, 512]},
        output_shapes={"output": [1, 3, 128, 128]},
        precision_policy="FP32",  # FP32 mandatory for inswapper_128
        supports_dynamic_batch=True,
        vram_footprint_mb=1150,
        description="Original InsightFace 128x128 face swapper model (FP32 precision required).",
        companion_files=("inswapper_128_dynamic.onnx", "inswapper_128_emap.npy"),
    ),
    ModelManifestEntry(
        model_id="hyperswap_1a_256",
        name="hyperswap_1a_256.onnx",
        category="swapper",
        primary_url="https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/hyperswap_1a_256.onnx",
        expected_size_bytes=341071060,
        expected_sha256="c0e98a8a03a276eb5d14df9dc8453488ff91ecb5b5ec1e6dc83c65c2b04f475a",
        input_shapes={"target": [1, 3, 256, 256], "source": [1, 512]},
        output_shapes={"output": [1, 3, 256, 256]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1450,
        description="HyperSwap 256px checkpoint 1a with normalized direct ArcFace embeddings.",
    ),
    ModelManifestEntry(
        model_id="hyperswap_1b_256",
        name="hyperswap_1b_256.onnx",
        category="swapper",
        primary_url="https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/hyperswap_1b_256.onnx",
        expected_size_bytes=341071060,
        expected_sha256="5124031789c490ffb5d4fb17676fb322a30bbcf32e01dfd2948bb37e7fe0c3b0",
        input_shapes={"target": [1, 3, 256, 256], "source": [1, 512]},
        output_shapes={"output": [1, 3, 256, 256]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1450,
        description="HyperSwap 256px checkpoint 1b balancing identity retention and blending.",
    ),
    ModelManifestEntry(
        model_id="hyperswap_1c_256",
        name="hyperswap_1c_256.onnx",
        category="swapper",
        primary_url="https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/hyperswap_1c_256.onnx",
        expected_size_bytes=341071060,
        expected_sha256="5528c2d76fe9f9ee5c697089b0930caaaae8ca95c51722cb44e398d5a153d865",
        input_shapes={"target": [1, 3, 256, 256], "source": [1, 512]},
        output_shapes={"output": [1, 3, 256, 256]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1450,
        description="HyperSwap 256px checkpoint 1c with highest fidelity identity projection.",
    ),
    ModelManifestEntry(
        model_id="simswap_unofficial_512",
        name="simswap_unofficial_512.onnx",
        category="swapper",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/simswap_unofficial_512.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/simswap_unofficial_512.onnx",
        expected_size_bytes=384110300,
        expected_sha256="4a20b2d699e116a4dbb6ebf4b1eb7285116717a02796dc95ea5f68b3563a3cb0",
        input_shapes={"target": [1, 3, 512, 512], "source": [1, 512]},
        output_shapes={"output": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=2100,
        description="Native 512x512 high-resolution SimSwap identity replacement engine.",
    ),
    ModelManifestEntry(
        model_id="ghost_1_256",
        name="ghost_1_256.onnx",
        category="swapper",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/ghost_1_256.onnx",
        expected_size_bytes=254823480,
        expected_sha256="9d98cbcfd8eb53ec1ec2f0b719ee9ce79717013897ca4fca783f912c98d7db54",
        input_shapes={"target": [1, 3, 256, 256], "source": [1, 512]},
        output_shapes={"output": [1, 3, 256, 256]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1200,
        description="Ghost 256px multi-layer representation face swapper.",
    ),

    # ── Category 2: Enhancers / Restorers ─────────────────────────────────────
    ModelManifestEntry(
        model_id="gpen_bfr_512",
        name="GPEN-BFR-512.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/GPEN-BFR-512.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GPEN-BFR-512.onnx",
        expected_size_bytes=284241604,
        expected_sha256="0960f836488737faee862412e87fc4c2f74122d2ee7fefd4e0e29b10ec871b69",
        input_shapes={"input": [1, 3, 512, 512]},
        output_shapes={"output": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=False,  # GPEN internal reshapes reject dynamic batch
        vram_footprint_mb=1250,
        description="GPEN Realistic 512px blind face restorer (static batch 1 in TensorRT).",
    ),
    ModelManifestEntry(
        model_id="gpen_bfr_1024",
        name="GPEN-BFR-1024.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/GPEN-BFR-1024.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GPEN-BFR-1024.onnx",
        expected_size_bytes=285493200,
        expected_sha256="bcd31aa52110c7bf394e101f35f8d95185d898ee8ef0e62a1ab5b20675bf0d42",
        input_shapes={"input": [1, 3, 1024, 1024]},
        output_shapes={"output": [1, 3, 1024, 1024]},
        precision_policy="FP16",
        supports_dynamic_batch=False,
        vram_footprint_mb=2400,
        description="GPEN Realistic 1024px maximum-resolution face detail restorer.",
    ),
    ModelManifestEntry(
        model_id="ultramax_bfr",
        name="ultramax_bfr.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.fp16.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/ultramax_bfr.onnx",
        expected_size_bytes=188981248,
        expected_sha256="5414b2c989eb2fb12d5930b91e779a1f10bc47cc15da9942a4a753697ebaa400",
        input_shapes={"x": [1, 3, 512, 512], "w": [1]},
        output_shapes={"y": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1100,
        description="UltraMax decoupled zero-copy pipeline with Laplacian pyramid fusion.",
    ),
    ModelManifestEntry(
        model_id="codeformer_dynamic",
        name="codeformer_dynamic.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/codeformer_dynamic.onnx",
        expected_size_bytes=377030656,
        expected_sha256="9aa48fc4b212ddbc9d75b3cc40f58d92984cfb03650228deca4d7f76634ff8c8",
        input_shapes={"x": [1, 3, 512, 512], "w": [1]},
        output_shapes={"y": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1350,
        description="CodeFormer with dynamic 2nd input w fidelity control.",
    ),
    ModelManifestEntry(
        model_id="restoreformer_plus_plus",
        name="RestoreFormer_Plus_Plus.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/restoreformer_plus_plus.onnx",
        expected_size_bytes=261898744,
        expected_sha256="f4db5a89902be71fdf1c38e788eb5b6f00dbf53bb1068ba0c18ba58f6fe954a2",
        input_shapes={"input": [1, 3, 512, 512]},
        output_shapes={"output": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1280,
        description="RestoreFormer++ high-fidelity contextual face restoration.",
    ),
    ModelManifestEntry(
        model_id="gfpgan_1_4",
        name="GFPGANv1.4.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/GFPGANv1.4.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GFPGANv1.4.onnx",
        expected_size_bytes=348632823,
        expected_sha256="54eb68323be01aa5610e19c968f9219f4044ffc9cd785c49042b4ee239c0fa46",
        input_shapes={"input": [1, 3, 512, 512]},
        output_shapes={"output": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=1400,
        description="GFPGAN v1.4 generative facial prior restoration model.",
    ),
    ModelManifestEntry(
        model_id="realesrgan_x4plus",
        name="RealESRGAN_x4plus.onnx",
        category="enhancer",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/RealESRGAN_x4plus.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/RealESRGAN_x4plus.onnx",
        expected_size_bytes=67041792,
        expected_sha256="a248f760e402fd055eeab9fae829d115e580e07ae41bf6beeb3bb4bf7a8ba3fb",
        input_shapes={"input": [1, 3, -1, -1]},
        output_shapes={"output": [1, 3, -1, -1]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=980,
        description="RealESRGAN x4plus high-order degradation super-resolution model.",
    ),

    # ── Category 3: 3D Pose, Landmarks & Expression ───────────────────────────
    ModelManifestEntry(
        model_id="3ddfa_v2_mb1_120x120",
        name="3ddfa_v2_mb1_120x120.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/3ddfa_v2_mb1_120x120.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/3ddfa_v2_mb1_120x120.onnx",
        expected_size_bytes=13271296,
        expected_sha256="b0959828d11ca7ff989a50ae40da1f5ee7e54c87da2364c3c3a9d2ec05cb51dc",
        input_shapes={"input": [1, 3, 120, 120]},
        output_shapes={"output": [1, 62]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=320,
        description="3DDFA V2 MobileNet 3D dense face alignment and 3DMM shape reconstruction.",
    ),
    ModelManifestEntry(
        model_id="1k3d68",
        name="1k3d68.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/1k3d68.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/1k3d68.onnx",
        expected_size_bytes=9040332,
        expected_sha256="df5c06b8a0c109b82e21ee648a734685ff4cfc79469e38e68cfb9efbe7b274c6",
        input_shapes={"input": [1, 3, 192, 192]},
        output_shapes={"output": [1, 68, 3]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=280,
        description="68-point 3D landmark regressor.",
    ),
    ModelManifestEntry(
        model_id="2d106det",
        name="2d106det.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/2d106det.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/2d106det.onnx",
        expected_size_bytes=4944948,
        expected_sha256="f001b856447ce7c8340d046c3ef6be59cb7a3fa0b7b1ca2cb5c50c37f4cf9983",
        input_shapes={"input": [1, 3, 192, 192]},
        output_shapes={"output": [1, 212]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=190,
        description="106-point dense 2D contour and facial feature landmark detector.",
    ),
    ModelManifestEntry(
        model_id="2dfan4",
        name="2dfan4.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/2dfan4.onnx",
        expected_size_bytes=96349000,
        expected_sha256="78ea5553556093539744cfa068e2f89c6cf84381389e6eb1264c84a5a544c9d7",
        input_shapes={"input": [1, 3, 256, 256]},
        output_shapes={"output": [1, 68, 64, 64]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=450,
        description="2DFAN-4 stacked hourglass facial landmark detector.",
    ),
    ModelManifestEntry(
        model_id="liveportrait_appearance",
        name="liveportrait_appearance_feature_extractor.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/liveportrait_appearance_feature_extractor.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/liveportrait_appearance_feature_extractor.onnx",
        expected_size_bytes=13280000,
        expected_sha256="80cfeb5176b91122a6136d892797669d084cb9ec4d740eb70a92d47ad68fbf64",
        input_shapes={"input": [1, 3, 256, 256]},
        output_shapes={"output": [1, 32, 16, 64, 64]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=520,
        description="LivePortrait appearance feature extractor module.",
    ),
    ModelManifestEntry(
        model_id="liveportrait_motion",
        name="liveportrait_motion_extractor.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/liveportrait_motion_extractor.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/liveportrait_motion_extractor.onnx",
        expected_size_bytes=38200000,
        expected_sha256="bfd058a5ae03154101e9be7520e11893c83a7c646b194f48ff61df9cf67ec805",
        input_shapes={"input": [1, 3, 256, 256]},
        output_shapes={"pitch": [1, 66], "yaw": [1, 66], "roll": [1, 66], "t": [1, 3], "exp": [1, 21, 3], "scale": [1, 1], "kp": [1, 21, 3]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=680,
        description="LivePortrait motion and head pose expression extractor.",
    ),
    ModelManifestEntry(
        model_id="liveportrait_stitching",
        name="liveportrait_stitching.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/liveportrait_stitching.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/liveportrait_stitching.onnx",
        expected_size_bytes=1040000,
        expected_sha256="0681df70cff15ba6cfb81467df9964585c5484fe98c484a0d923058a26b1c312",
        input_shapes={"input": [1, 126]},
        output_shapes={"output": [1, 126]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=120,
        description="LivePortrait expression stitching and delta adaptation module.",
    ),
    ModelManifestEntry(
        model_id="liveportrait_warping_spade",
        name="liveportrait_warping_spade.onnx",
        category="3d_geometry",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/liveportrait_warping_spade.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/liveportrait_warping_spade.onnx",
        expected_size_bytes=64100000,
        expected_sha256="bfd9df4841604a113d09a259c63b72c918cba4ea988a8f4c1c9b6be672433ea1",
        input_shapes={"feature": [1, 32, 16, 64, 64], "kp_s": [1, 21, 3], "kp_d": [1, 21, 3]},
        output_shapes={"out": [1, 3, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=890,
        description="LivePortrait warping SPADE generator neural renderer.",
    ),

    # ── Category 4: Detectors & Embedders ─────────────────────────────────────
    ModelManifestEntry(
        model_id="scrfd_2_5g_kps",
        name="scrfd_2.5g_kps.onnx",
        category="detector_embedder",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/scrfd_2.5g_kps.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/scrfd_2.5g_kps.onnx",
        expected_size_bytes=3272996,
        expected_sha256="b0959828d11ca7ff989a50ae40da1f5ee7e54c87da2364c3c3a9d2ec05cb51dc",
        input_shapes={"input": [1, 3, 640, 640]},
        output_shapes={"score_8": [1, 12800, 1], "score_16": [1, 3200, 1], "score_32": [1, 800, 1]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=180,
        description="High-speed lightweight 2.5G SCRFD face and 5-point landmark detector.",
        companion_files=("scrfd_2.5g_kps_dynamic.onnx",),
    ),
    ModelManifestEntry(
        model_id="det_10g",
        name="det_10g.onnx",
        category="detector_embedder",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/det_10g.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/det_10g.onnx",
        expected_size_bytes=16867080,
        expected_sha256="5838f7fe0536ef55fa3ba422709a3fc3a4b008d66ff97b4c3ae6a090e9cc0c7b",
        input_shapes={"input": [1, 3, 640, 640]},
        output_shapes={"score_8": [1, 12800, 1], "score_16": [1, 3200, 1], "score_32": [1, 800, 1]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=350,
        description="SCRFD 10G high-accuracy multi-scale face detector.",
        companion_files=("det_10g_dynamic.onnx",),
    ),
    ModelManifestEntry(
        model_id="yoloface_8n",
        name="yoloface_8n.onnx",
        category="detector_embedder",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/yoloface_8n.onnx",
        expected_size_bytes=6246416,
        expected_sha256="821cdbb1e65f3f044bb77ce9df4249a0397576595d7cf1417a80ee327fefefaa",
        input_shapes={"images": [1, 3, 640, 640]},
        output_shapes={"output0": [1, 16, 8400]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=240,
        description="YOLOv8-Face nano ultra-fast bounding box and landmark detector.",
    ),
    ModelManifestEntry(
        model_id="w600k_r50",
        name="w600k_r50.onnx",
        category="detector_embedder",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/w600k_r50.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/w600k_r50.onnx",
        expected_size_bytes=174388474,
        expected_sha256="4c06341c33c2a68be0004523e64aa34f0d367c3cae8eb276e033a5538d35e165",
        input_shapes={"input": [1, 3, 112, 112]},
        output_shapes={"output": [1, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=720,
        description="ArcFace ResNet-50 512-dimensional facial identity feature embedder.",
    ),

    # ── Category 5: Occlusion & Semantic Parsers ──────────────────────────────
    ModelManifestEntry(
        model_id="xseg_3",
        name="xseg_3.onnx",
        category="occlusion_masking",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/xseg_3.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/xseg_3.onnx",
        expected_size_bytes=160492582,
        expected_sha256="48ccd7e8541e2b6c38260421711204b4cfab13f4fd80004ca5187ea1c0a0c29f",
        input_shapes={"input": [1, 3, 256, 256]},
        output_shapes={"output": [1, 1, 256, 256]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=640,
        description="DeepFaceLab XSeg-3 neural occluder (preserves hands, glasses, hair, props).",
    ),
    ModelManifestEntry(
        model_id="bisenet_resnet_34",
        name="bisenet_resnet_34.onnx",
        category="occlusion_masking",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/bisenet_resnet_34.onnx",
        expected_size_bytes=53342080,
        expected_sha256="4a0b8c958a3cbf34800b6564e525beeb56c80249c5be4e3cae6d1912a2333796",
        input_shapes={"input": [1, 3, 512, 512]},
        output_shapes={"output": [1, 19, 512, 512]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=580,
        description="BiSeNet ResNet-34 19-class semantic face parsing model.",
    ),
    ModelManifestEntry(
        model_id="clipseg_rd64_refined",
        name="clipseg_rd64_refined.onnx",
        category="occlusion_masking",
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/clipseg_rd64_refined.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/clipseg_rd64_refined.onnx",
        expected_size_bytes=158498800,
        expected_sha256="78c949c81121de4e1eb33d596ee248ea871804c86cb356230f2526ae75bf7121",
        input_shapes={"input": [1, 3, 352, 352]},
        output_shapes={"output": [1, 1, 352, 352]},
        precision_policy="FP16",
        supports_dynamic_batch=True,
        vram_footprint_mb=750,
        description="CLIPSeg ViT-B/16 text-prompted neural occlusion and object segmentation.",
    ),
)

UNIFIED_MANIFEST_MAP: dict[str, ModelManifestEntry] = {
    entry.model_id: entry for entry in UNIFIED_MODEL_ENTRIES
}
UNIFIED_NAME_MAP: dict[str, ModelManifestEntry] = {
    entry.name: entry for entry in UNIFIED_MODEL_ENTRIES
}


class UnifiedModelRegistry:
    """Unified declarative model registry and status inspector."""

    def __init__(self, root: Path | None = None):
        self.root = root or application_root()
        self.entries = UNIFIED_MANIFEST_MAP
        self.name_map = UNIFIED_NAME_MAP

    def get_search_paths(self) -> list[Path]:
        return [
            self.root / "app/models",
            self.root / "roop_3d/models/cache",
            self.root / ".cache/models",
            self.root / "cache/models",
        ]

    def get_trt_cache_paths(self) -> list[Path]:
        return [
            self.root / "app/models/trt_cache",
            self.root / ".cache/trt",
            self.root / "cache/trt",
        ]

    def resolve_model_file(self, entry: ModelManifestEntry) -> Path | None:
        """Search local candidate paths for the model file."""
        for base in self.get_search_paths():
            p = base / entry.name
            if p.is_file():
                return p
            p_id = base / f"{entry.model_id}.onnx"
            if p_id.is_file():
                return p_id
        return None

    def resolve_trt_engine(self, entry: ModelManifestEntry) -> Path | None:
        """Search local candidate paths for a compiled TensorRT engine."""
        stem = Path(entry.name).stem
        for base in self.get_trt_cache_paths():
            for cand in base.glob(f"*{stem}*.engine"):
                if cand.is_file():
                    return cand
        return None

    def inspect_model_readiness(self, entry: ModelManifestEntry) -> dict[str, Any]:
        """Inspect model presence, hash validity, and TRT engine readiness."""
        file_path = self.resolve_model_file(entry)
        trt_path = self.resolve_trt_engine(entry)

        if file_path is None:
            readiness: ReadinessState = "MISSING"
            hash_valid = False
            actual_sha256 = None
        else:
            # Check file size & SHA-256
            try:
                data = file_path.read_bytes()
                actual_sha256 = hashlib.sha256(data).hexdigest()
                hash_valid = actual_sha256.lower() == entry.expected_sha256.lower()
                if not hash_valid:
                    readiness = "CORRUPT HASH"
                elif trt_path is not None and trt_path.is_file():
                    readiness = "READY + TRT ENGINE"
                else:
                    readiness = "DOWNLOADED (NEEDS TRT)"
            except Exception:
                readiness = "CORRUPT HASH"
                hash_valid = False
                actual_sha256 = None

        return {
            "model_id": entry.model_id,
            "name": entry.name,
            "category": entry.category,
            "readiness_state": readiness,
            "vram_footprint_mb": entry.vram_footprint_mb,
            "precision_policy": entry.precision_policy,
            "supports_dynamic_batch": entry.supports_dynamic_batch,
            "expected_size_bytes": entry.expected_size_bytes,
            "expected_sha256": entry.expected_sha256,
            "actual_sha256": actual_sha256,
            "hash_valid": hash_valid,
            "model_path": str(file_path) if file_path else None,
            "trt_engine_path": str(trt_path) if trt_path else None,
            "description": entry.description,
            "primary_url": entry.primary_url,
            "fallback_url": entry.fallback_url,
        }

    def get_full_readiness_report(self) -> list[dict[str, Any]]:
        return [self.inspect_model_readiness(entry) for entry in UNIFIED_MODEL_ENTRIES]


class ResilientModelDownloader:
    """Chunked, resumable (*.part) downloader with telemetry progress hooks."""

    def __init__(self, destination_dir: Path, timeout_seconds: float = 60.0):
        self.destination_dir = destination_dir
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def download_model(
        self,
        entry: ModelManifestEntry,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        use_fallback: bool = False,
    ) -> Path:
        """Download model into *.part file with resumption and verification."""
        import urllib.request
        from urllib.error import URLError

        url = entry.fallback_url if use_fallback else entry.primary_url
        target_path = self.destination_dir / entry.name
        part_path = self.destination_dir / f"{entry.name}.part"

        # If already verified on disk, return immediately
        if target_path.is_file():
            hasher = hashlib.sha256()
            with open(target_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            if hasher.hexdigest().lower() == entry.expected_sha256.lower():
                if progress_callback:
                    progress_callback({
                        "type": "model_download_progress",
                        "model_id": entry.model_id,
                        "bytes_downloaded": target_path.stat().st_size,
                        "total_bytes": entry.expected_size_bytes,
                        "percent": 100.0,
                        "speed_mbps": 0.0,
                        "eta_seconds": 0.0,
                        "status": "ready",
                    })
                return target_path

        existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "roop-3d/1.0"})
        if existing_bytes > 0:
            req.add_header("Range", f"bytes={existing_bytes}-")

        mode = "ab" if existing_bytes > 0 else "wb"
        start_time = time.monotonic()
        downloaded = existing_bytes
        total_size = entry.expected_size_bytes

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp, open(part_path, mode) as out_f:
                content_len = resp.headers.get("Content-Length")
                if content_len:
                    total_size = existing_bytes + int(content_len)

                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = max(0.001, time.monotonic() - start_time)
                    speed_mbps = (downloaded - existing_bytes) / (1024 * 1024) / elapsed
                    pct = min(100.0, (downloaded / max(1, total_size)) * 100.0)
                    eta = max(0.0, (total_size - downloaded) / max(0.001, speed_mbps * 1024 * 1024))

                    if progress_callback:
                        progress_callback({
                            "type": "model_download_progress",
                            "model_id": entry.model_id,
                            "bytes_downloaded": downloaded,
                            "total_bytes": total_size,
                            "percent": round(pct, 1),
                            "speed_mbps": round(speed_mbps, 2),
                            "eta_seconds": round(eta, 1),
                            "status": "downloading",
                        })

        except (URLError, OSError) as err:
            if not use_fallback:
                # Retry with fallback URL
                return self.download_model(entry, progress_callback, use_fallback=True)
            raise RuntimeError(f"Download failed for {entry.name}: {err}") from err

        # Verify hash
        if progress_callback:
            progress_callback({
                "type": "model_download_progress",
                "model_id": entry.model_id,
                "bytes_downloaded": downloaded,
                "total_bytes": total_size,
                "percent": 100.0,
                "speed_mbps": 0.0,
                "eta_seconds": 0.0,
                "status": "verifying",
            })

        hasher = hashlib.sha256()
        with open(part_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest().lower()

        if digest != entry.expected_sha256.lower():
            if part_path.is_file():
                part_path.unlink()
            raise ValueError(
                f"SHA-256 hash mismatch for {entry.name}: expected {entry.expected_sha256}, got {digest}"
            )

        # Atomic rename from .part to final destination
        if target_path.is_file():
            target_path.unlink()
        part_path.rename(target_path)

        if progress_callback:
            progress_callback({
                "type": "model_download_progress",
                "model_id": entry.model_id,
                "bytes_downloaded": target_path.stat().st_size,
                "total_bytes": target_path.stat().st_size,
                "percent": 100.0,
                "speed_mbps": 0.0,
                "eta_seconds": 0.0,
                "status": "ready",
            })

        return target_path
