"""GPU execution helpers for mixed precision inference and bounded frame processing."""

from roop_3d.execution.trt_mixed_precision import (
    FP16_TENSOR_OPS,
    FP32_PINNED_OPS,
    OptimizationProfiles,
    ResilientORTSession,
    TensorRTMixedPrecisionSessionManager,
    derive_optimization_profiles,
)
from roop_3d.execution.zero_copy_pipeline import (
    GPU_PIPELINE_STAGES,
    GPUExecutionFallback,
    OutputBufferSpec,
    ZeroCopyGPUExecutionRing,
)
from roop_3d.execution.worker_pools import (
    DecodePreprocessPool,
    DeviceFrameLease,
    encode_nvenc_chunk,
    GopChunk,
    PostEncodeWorkerPool,
    stitch_gop_chunks,
)

__all__ = [
    "DecodePreprocessPool",
    "DeviceFrameLease",
    "FP16_TENSOR_OPS",
    "FP32_PINNED_OPS",
    "GPU_PIPELINE_STAGES",
    "GPUExecutionFallback",
    "GopChunk",
    "OptimizationProfiles",
    "OutputBufferSpec",
    "PostEncodeWorkerPool",
    "ResilientORTSession",
    "TensorRTMixedPrecisionSessionManager",
    "ZeroCopyGPUExecutionRing",
    "derive_optimization_profiles",
    "encode_nvenc_chunk",
    "stitch_gop_chunks",
]
