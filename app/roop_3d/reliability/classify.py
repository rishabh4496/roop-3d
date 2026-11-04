"""Turn an unexpected exception into a stable, actionable, *non-silent* job failure.

The worker used to flatten every non-``DomainError`` into ``WORKER_FAILED: a local runtime error``. That is a silent
failure from the user's point of view: it names no cause and offers no next step. ``classify`` keeps the privacy
rule (no raw exception text, which can contain paths and media names, ever reaches a job record or a response) while
naming the *kind* of failure and whether the job can be resumed once the cause is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

from roop_3d.core.errors import DomainError
from roop_3d.engine.vram_governor import is_cuda_oom
from roop_3d.reliability.disk import is_disk_full


@dataclass(frozen=True)
class Classified:
    code: str
    message: str
    recoverable: bool          # True: the checkpoint is intact and the job should become `interrupted`, not `failed`
    warning: str               # one sanitized line for the diagnostics buffer (exception *type* only)


_ORT_LOAD_MARKERS = ("protobuf parsing failed", "load model", "invalid_protobuf", "invalid model", "model_load",
                     "invalidgraph", "invalid_graph", "nosuchfile", "no_suchfile", "failed to load model")
_MEDIA_MARKERS = ("invalid data found", "moov atom not found", "invalid argument", "end of file",
                  "decoder", "demuxer", "avformat", "ffmpeg")


def _chain(error: BaseException):
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def classify(error: BaseException) -> Classified:
    kind = type(error).__name__
    if is_disk_full(error):
        return Classified("DISK_FULL",
                          "The output volume ran out of space. Completed work is saved; free disk space and resume the job.",
                          True, f"disk full ({kind})")
    for item in _chain(error):
        name = f"{type(item).__module__}.{type(item).__name__}".lower()
        text = str(item).lower()
        if "onnxruntime" in name and any(marker in text or marker in name for marker in _ORT_LOAD_MARKERS):
            return Classified("MODEL_CORRUPT",
                              "A model file could not be loaded and appears damaged. Re-verify or re-import the model, then retry.",
                              False, f"model load failure ({type(item).__name__})")
    if is_cuda_oom(error):
        text = " ".join(str(item).lower() for item in _chain(error))
        if any(marker in text for marker in ("cuda", "cudnn", "vram", "gpu")):
            return Classified("GPU_MEMORY_EXHAUSTED",
                              "The GPU ran out of memory. Close other GPU programs or lower the preview/batch size, then resume the job.",
                              True, "gpu memory exhausted")
        if any(isinstance(item, MemoryError) for item in _chain(error)) or "bad alloc" in text or "bad_alloc" in text:
            return Classified("HOST_MEMORY_EXHAUSTED",
                              "The system ran out of RAM. Close other programs or lower the thread count, then resume the job.",
                              True, "host memory exhausted")
        # ONNX Runtime's arena reports CPU and CUDA allocation failures with identical text; do not guess which.
        return Classified("MEMORY_EXHAUSTED",
                          "The GPU or system ran out of memory. Close other programs, then resume the job.",
                          True, "memory exhausted (device not identifiable)")
    for item in _chain(error):
        module = type(item).__module__.split(".")[0]
        if module == "av" or (isinstance(item, (OSError, ValueError, RuntimeError))
                              and any(marker in str(item).lower() for marker in _MEDIA_MARKERS)):
            return Classified("MEDIA_CORRUPT",
                              "The target video could not be decoded and may be corrupt or truncated. Re-export or repair the file.",
                              False, f"media decode failure ({type(item).__name__})")
    if isinstance(error, PermissionError):
        return Classified("FILE_ACCESS_DENIED",
                          "A file or folder needed by the job is locked or not writable. Close programs using it, then retry.",
                          True, "file access denied")
    if isinstance(error, (BrokenPipeError, ConnectionResetError)):
        return Classified("ENCODER_TERMINATED",
                          "The video encoder process ended unexpectedly. Completed frames are saved; resume the job.",
                          True, f"encoder pipe closed ({kind})")
    return Classified("WORKER_FAILED", "Job processing failed due to a local runtime error.", False,
                      f"unclassified runtime error ({kind})")


def classify_domain(error: DomainError) -> Classified | None:
    """DomainErrors that mean 'resumable after the user acts' rather than 'this job is wrong'."""
    if error.code in {"DISK_FULL", "DISK_SPACE_LOW"}:
        return Classified(error.code, error.message[:512], True, error.code.lower())
    if error.code == "PIPELINE_OOM":
        # Every memory profile (GPU, then CPU) ran out. The checkpoint is intact and the cause is environmental, so
        # the job waits for the user instead of dying: close other programs / lower the thread count, then resume.
        return Classified("PIPELINE_OOM",
                          "Rendering ran out of memory on every available profile. Completed work is saved; close other "
                          "programs or lower the thread count, then resume the job.", True, "memory profiles exhausted")
    return None
