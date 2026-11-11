from __future__ import annotations

import numpy as np

from roop_3d.execution.zero_copy_pipeline import (
    GPU_PIPELINE_STAGES,
    OutputBufferSpec,
    ZeroCopyGPUExecutionRing,
)


class _OrtValue:
    def __init__(self, device="cuda"):
        self._device = device

    def device_name(self):
        return self._device


class _FakeOrt:
    class OrtValue:
        @staticmethod
        def ortvalue_from_shape_and_type(shape, dtype, device, device_id):
            return _OrtValue(device)


class _Output:
    name = "features"


class _Binding:
    def __init__(self):
        self.outputs = []

    def clear_binding_inputs(self):
        pass

    def clear_binding_outputs(self):
        self.outputs.clear()

    def bind_ortvalue_input(self, name, value):
        assert value.device_name() == "cuda"

    def bind_ortvalue_output(self, name, value):
        self.outputs.append((name, value))

    def synchronize_outputs(self):
        pass

    def get_outputs(self):
        return []


class _Session:
    def __init__(self):
        self.last_binding = None

    def get_outputs(self):
        return [_Output()]

    def io_binding(self):
        self.last_binding = _Binding()
        return self.last_binding

    def run_with_iobinding(self, binding):
        assert binding.outputs


def test_ring_reuses_device_ortvalue_and_tracks_zero_intermediate_copies():
    session = _Session()
    ring = ZeroCopyGPUExecutionRing(
        slots=2,
        stage_order=("detection",),
        torch_module=object(),
        ort_module=_FakeOrt,
        require_cuda=False,
    )
    ring.register_ort_stage(
        "detection", session,
        {"features": OutputBufferSpec((1, 512), np.float32, "ortvalue")},
    )
    input_value = _OrtValue("cuda")

    lease = ring.acquire()
    output = lease.run_ort_stage("detection", {"image": input_value})
    output_buffer = output["features"]
    lease.release()

    assert output_buffer.device_name() == "cuda"
    assert ring.intermediate_host_copies == 0
    assert ring.host_fallback_copies == 0
    ring.close()


def test_ring_requires_the_declared_stage_order():
    ring = ZeroCopyGPUExecutionRing(
        slots=2,
        stage_order=GPU_PIPELINE_STAGES,
        torch_module=object(),
        ort_module=_FakeOrt,
        require_cuda=False,
    )
    lease = ring.acquire()
    try:
        try:
            lease.run_torch_stage("relit_swap", lambda: None)
        except ValueError as error:
            assert "expected 'detection'" in str(error)
        else:
            raise AssertionError("out-of-order GPU stage was accepted")
    finally:
        lease.release()
        ring.close()
