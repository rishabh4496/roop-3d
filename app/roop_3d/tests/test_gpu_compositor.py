"""Stage 10 geometry, semantic exclusions, real CUDA binding and slot ownership."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from roop_3d.gpu_compositor import (CompositorOptions, GpuCompositor, TripleBufferedCudaPipeline,
    TorchIOBinding, color_transfer, invert_affine, laplacian_blend, rgb_to_lab, lab_to_rgb, soft_matte, warp)


def test_lab_matches_opencv_float_contract_and_roundtrips():
    image = np.random.default_rng(10).random((32, 32, 3), dtype=np.float32)
    value = torch.from_numpy(image.transpose(2, 0, 1)[None])
    lab = rgb_to_lab(value)
    reference = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # OpenCV's float implementation uses a quantized lookup table.
    assert np.abs(lab[0].permute(1, 2, 0).numpy() - reference).max() < .5
    assert torch.max(torch.abs(lab_to_rgb(lab) - value)) < 2e-5


def test_semantic_exclusions_empty_masks_and_coarse_lighting():
    source = torch.rand((1, 3, 32, 32))
    target = source * .7 + .1
    skin = torch.ones((1, 1, 32, 32))
    skin[..., 4:9, 8:24] = 0
    skin[..., 22:27, 12:20] = 0
    visibility = torch.ones_like(skin)
    visibility[..., :7] = 0
    protected = skin * visibility == 0
    for regional in (False, True):
        matched = color_transfer(source, target, skin * visibility, regional=regional)
        assert torch.equal(matched.masked_select(protected), source.masked_select(protected))
        assert torch.equal(color_transfer(source, target, torch.zeros_like(skin), regional=regional), source)
    alpha = soft_matte(skin, visibility)
    assert torch.count_nonzero(alpha[visibility == 0]) == 0
    blended = laplacian_blend(torch.ones_like(source) * .9, torch.ones_like(source) * .2, torch.ones_like(skin))
    assert torch.allclose(blended, torch.ones_like(source) * .2, atol=1e-6)


def test_no_geometric_skin_claim_and_invalid_options():
    with pytest.raises(ValueError, match='semantic'):
        GpuCompositor(lambda *_: None, options=CompositorOptions(color_strength=1))
    for kwargs in ({'color_strength': float('nan')}, {'erosion': 65}, {'laplacian_blend_weight': -1}):
        with pytest.raises(ValueError):
            CompositorOptions(**kwargs)


def test_affine_out_of_frame_and_inverse_composition():
    value = torch.arange(7 * 9, dtype=torch.float32).reshape(1, 1, 7, 9)
    matrix = torch.tensor([[[1., 0, -2], [0, 1, 3]]])
    actual = warp(value, matrix, (7, 9))[0, 0].numpy()
    reference = cv2.warpAffine(value[0, 0].numpy(), matrix[0].numpy(), (9, 7), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    np.testing.assert_allclose(actual, reference, atol=1e-5)
    np.testing.assert_allclose(invert_affine(matrix)[0].numpy(), cv2.invertAffineTransform(matrix[0].numpy()), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_triple_slots_order_ownership_shape_change_and_failure_recovery():
    failing = [False]
    def factory(stream):
        def compute(frame, faces):
            if failing[0]:
                raise RuntimeError('injected compute failure')
            return frame
        return compute
    pipeline = TripleBufferedCudaPipeline(factory)
    frames = [np.full((32 + i % 2, 48, 3), i, np.uint8) for i in range(8)]
    output = pipeline.process_batch(frames, [[]] * len(frames))
    for a, b in zip(frames, output):
        np.testing.assert_array_equal(a, b)
    assert len({pipeline.upload.cuda_stream, pipeline.compute.cuda_stream, pipeline.download.cuda_stream}) == 3
    assert all(slot['host_in'].is_pinned() and slot['host_out'].is_pinned() for slot in pipeline.slots)
    with torch.inference_mode():
        for slot in pipeline.slots:
            slot['host_out'].zero_()
    np.testing.assert_array_equal(output[-1], frames[-1])
    failing[0] = True
    with pytest.raises(RuntimeError, match='injected'):
        pipeline.process(frames[0], [])
    failing[0] = False
    np.testing.assert_array_equal(pipeline.process(frames[-1], []), frames[-1])
    with pytest.raises(ValueError, match='budget'):
        pipeline.process(np.zeros((2160, 3840, 3), np.uint8), [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_real_ort_device_pointer_binding_and_stream_validation():
    import onnx
    import onnxruntime as ort
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(h.make_graph([h.make_node('Mul', ['x', 'x'], ['y'])], 'cuda_pointer_test',
        [h.make_tensor_value_info('x', t.FLOAT, [1, 3, 16, 16])],
        [h.make_tensor_value_info('y', t.FLOAT, [1, 3, 16, 16])]), opset_imports=[h.make_opsetid('', 17)])
    stream = torch.cuda.Stream()
    options = ort.SessionOptions()
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
        providers=[('CUDAExecutionProvider', {'user_compute_stream': str(stream.cuda_stream)})])
    runner = TorchIOBinding(session, stream)
    with pytest.raises(ValueError, match='owning'):
        runner.run({}, {})
    with torch.cuda.stream(stream):
        x = torch.rand((1, 3, 16, 16), device='cuda')
        actual = runner.run({'x': x}, {'y': (tuple(x.shape), torch.float32)})['y']
        expected = x * x
    stream.synchronize()
    assert actual.is_cuda and torch.equal(actual, expected)
    with pytest.raises(ValueError, match='compute stream'):
        TorchIOBinding(session, torch.cuda.Stream())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_opt_in_stage3_dispatch_uses_cloned_stream_session(tmp_path, monkeypatch):
    import onnx
    import onnxruntime as ort
    from roop_3d.vectorized_pipeline import GpuBatchPipeline
    from roop_3d.optimized_prepass import LockedFace
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(h.make_graph([
        h.make_node('ReduceSum', ['source'], ['sum'], keepdims=0),
        h.make_node('Add', ['target', 'sum'], ['output'])], 'stage3_device_swap',
        [h.make_tensor_value_info('target', t.FLOAT, [1, 3, 128, 128]),
         h.make_tensor_value_info('source', t.FLOAT, [1, 512])],
        [h.make_tensor_value_info('output', t.FLOAT, [1, 3, 128, 128])]), opset_imports=[h.make_opsetid('', 17)])
    path = tmp_path / 'synthetic_swap.onnx'
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=['CUDAExecutionProvider'])
    original_options = session.get_provider_options()
    monkeypatch.setenv('ROOP_GPU_COMPOSITOR', '1')
    pipeline = GpuBatchPipeline(session, {0: np.zeros(512, np.float32)})
    assert pipeline.backend == 'CUDA_RESIDENT_TRIPLE_BUFFER'
    frame = np.full((160, 160, 3), 93, np.uint8)
    face = LockedFace(0, 0, np.array((0, 0, 128, 128)), (ARCFACE_REFERENCE_5PTS + [8, 0]).astype(np.float32))
    with monkeypatch.context() as patcher:
        patcher.setattr(cv2, 'warpAffine', lambda *_a, **_k: pytest.fail('CPU warp in CUDA path'))
        outputs = pipeline.process_batch([frame] * 4, [[face]] * 4)
    for value in outputs:
        np.testing.assert_array_equal(value, frame)
    assert session.get_provider_options() == original_options
    worker = pipeline.fork_worker()
    assert worker._resident.compute.cuda_stream != pipeline._resident.compute.cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_nonfinite_device_output_rejected_at_final_download():
    pipeline = TripleBufferedCudaPipeline(lambda stream: lambda image, faces: image * float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        pipeline.process(np.zeros((16, 16, 3), np.uint8), [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_semantic_adapter_uses_labels_to_exclude_eyes_teeth_and_occluders():
    from roop_3d.gpu_compositor import SemanticMaskAdapter
    logits = torch.zeros((1, 5, 512, 512), device='cuda')
    logits[:, 1] = 1
    for class_id, start in ((2, 100), (3, 200), (4, 300)):
        logits[:, class_id, start:start + 32, 100:400] = 9
    adapter = SemanticMaskAdapter(lambda *_: logits, face_ids=(1, 2, 3), skin_ids=(1,),
        protected_ids=(2, 3), occluder_ids=(4,), evidence='synthetic five-class test contract')
    alpha, skin, visible = adapter(torch.zeros((1, 3, 512, 512), device='cuda'), None)
    for start in (100, 200, 300):
        assert torch.count_nonzero(skin[..., start:start + 32, 100:400]) == 0
    assert torch.count_nonzero(alpha[..., 300:332, 100:400]) == 0
    assert torch.count_nonzero(visible[..., 300:332, 100:400]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_complete_device_chain_keeps_512_restoration_and_semantics_on_compute_stream(monkeypatch):
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.optimized_prepass import LockedFace
    calls = []
    def factory(stream):
        def observed(name, image):
            assert image.is_cuda and torch.cuda.current_stream() == stream
            calls.append((name, tuple(image.shape)))
        def swap(image, face):
            observed('swap', image)
            return (image + .08).clamp(0, 1)
        def restore(image, face):
            observed('restore', image)
            return image
        def mask(image, face):
            observed('mask', image)
            alpha = torch.ones_like(image[:, :1])
            skin, visibility = alpha.clone(), alpha.clone()
            skin[..., 130:200, 100:400] = 0
            skin[..., 300:360, 180:300] = 0
            visibility[..., 220:400, 200:300] = 0
            return alpha, skin, visibility
        return GpuCompositor(swap, restorer=restore, masker=mask,
            options=CompositorOptions(color_strength=.7, regional_color=True,
                laplacian_blend_weight=.8, erosion=2, feather_sigma=2, jaw_falloff=.1))
    frame = np.full((512, 512, 3), 100, np.uint8)
    face = LockedFace(0, 0, np.array((0, 0, 512, 512)),
                      ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 4).astype(np.float32))
    pipeline = TripleBufferedCudaPipeline(factory)
    with monkeypatch.context() as patcher:
        for name in ('warpAffine', 'GaussianBlur'):
            patcher.setattr(cv2, name, lambda *_a, **_k: pytest.fail('CPU operation in device chain'))
        output = pipeline.process(frame, [face])
    assert calls == [('swap', (1, 3, 128, 128)), ('restore', (1, 3, 512, 512)), ('mask', (1, 3, 512, 512))]
    # Exact interior of the protected foreground remains target-owned.
    np.testing.assert_array_equal(output[230:390, 210:290], frame[230:390, 210:290])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_device_coordinate_and_color_math_leave_no_stream_workspace():
    import gc
    stream = torch.cuda.Stream()
    baseline = torch.cuda.memory_allocated()
    with torch.cuda.stream(stream):
        image = torch.rand((1, 3, 32, 32), device='cuda')
        matrix = torch.tensor([[[1., 0, 0], [0, 1, 0]]], device='cuda')
        sampled = warp(image, matrix, (32, 32))
        converted = lab_to_rgb(rgb_to_lab(sampled))
    stream.synchronize()
    del image, matrix, sampled, converted
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == baseline
