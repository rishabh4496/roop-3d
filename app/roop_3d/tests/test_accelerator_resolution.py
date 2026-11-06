from roop_3d.core.hardware import resolve_accelerator

TRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"


def test_auto_prefers_tensorrt_then_cuda_then_cpu():
    assert resolve_accelerator("auto", [TRT, CUDA, CPU]) == TRT
    assert resolve_accelerator("auto", [CUDA, CPU]) == CUDA
    assert resolve_accelerator("auto", [CPU]) == CPU
    assert resolve_accelerator("auto", None) == CPU


def test_explicit_choice_is_kept_and_cpu_wins_without_gpu():
    assert resolve_accelerator(CUDA, [TRT, CUDA, CPU]) == CUDA
    assert resolve_accelerator(TRT, [TRT, CUDA, CPU]) == TRT
    assert resolve_accelerator(CPU, [TRT, CUDA, CPU]) == CPU
    assert resolve_accelerator("auto", [TRT, CUDA, CPU], has_gpu=False) == CPU
