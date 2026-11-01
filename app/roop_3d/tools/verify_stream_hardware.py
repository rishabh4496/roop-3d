"""Opt-in native synthetic acceptance; no production neural fidelity claims."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tempfile

import av
import cv2
import numpy as np

from roop_3d.render_policy import CudaWorkerContext
from roop_3d.vectorized_pipeline import GpuBatchPipeline, affine_graph, crop_matrix, normalized_matrix
from roop_3d.optimized_prepass import LockedFace
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.core.storage import owned
from roop_3d.audit.project_paths import application_root
from roop_3d.models.owned_files import owned_path
from roop_3d.tools.verify_stream_integration import create_clip
from roop_3d.hardware_streamer import NvdecFrameSource, NVHardwareVideoWriter


def packet_digest(path, kind):
    with av.open(str(path)) as container:
        stream = (container.streams.video if kind == "video" else container.streams.audio)[0]
        digest = hashlib.sha256()
        for packet in container.demux(stream):
            if packet.size:
                digest.update(bytes(packet))
        return digest.hexdigest()


def verify_hardware():
    import onnx
    import onnxruntime as ort
    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_model(h.make_graph([
        h.make_node("ReduceSum", ["source"], ["sum"], keepdims=0),
        h.make_node("Mul", ["sum", "zero"], ["z"]), h.make_node("Add", ["target", "z"], ["output"])
    ], "synthetic_gpu_identity", [h.make_tensor_value_info("target", t.FLOAT, ["batch", 3, 128, 128]),
        h.make_tensor_value_info("source", t.FLOAT, ["batch", 512])],
        [h.make_tensor_value_info("output", t.FLOAT, ["batch", 3, 128, 128])],
        [onnx.numpy_helper.from_array(np.array(0, np.float32), "zero")]), opset_imports=[h.make_opsetid("", 17)])
    with CudaWorkerContext().activate():
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=["CUDAExecutionProvider"])
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("Native acceptance forbids CPU fallback")
        engine = GpuBatchPipeline(session, {0: np.ones(512, np.float32), 1: np.ones(512, np.float32)}, execution_threads=3)
    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.35).astype(np.float32)
    faces = [LockedFace(0, 0, np.array([-10, -5, 40, 45]), points + [-10, -5]),
             LockedFace(0, 1, np.array([35, 30, 85, 80]), points + [35, 30])]
    matrices = np.stack([crop_matrix(v.landmarks5) for v in faces])
    image = frame[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    cpu = ort.SessionOptions()
    cpu.intra_op_num_threads = 1
    crops = ort.InferenceSession(affine_graph(64, 64, 2).SerializeToString(), sess_options=cpu,
        providers=["CPUExecutionProvider"]).run(None, {"image": image, "matrix": np.stack([
            normalized_matrix(cv2.invertAffineTransform(m), 64, 64) for m in matrices])})[0]
    reference = ort.InferenceSession(affine_graph(64, 64, 2, compose=True).SerializeToString(), sess_options=cpu,
        providers=["CPUExecutionProvider"]).run(None, {"image": image, "crops": crops,
            "matrix": np.stack([normalized_matrix(m, 128, 128) for m in matrices])})[0]
    reference = np.clip(reference[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).round().astype(np.uint8)
    def execute(_):
        worker = engine.fork_worker()
        errors = []
        for _ in range(3):
            actual = worker.process(frame, faces)
            delta = int(np.abs(actual.astype(np.int16) - reference.astype(np.int16)).max())
            if delta > 1:
                raise AssertionError(f"CUDA/CPU affine fixture differs by {delta} uint8 levels")
            errors.append(delta)
        return max(errors)
    with ThreadPoolExecutor(max_workers=3) as executor:
        errors = list(executor.map(execute, range(3)))
    base = owned(application_root(), "roop_3d/.runtime/stage3-native-acceptance")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        folder = owned_path(Path(temporary))
        source = folder / "input.mp4"
        create_clip(source, size=256, audio=True)
        decoder = NvdecFrameSource(source, strict=True)
        lut = folder / "identity.cube"
        lut.write_text("LUT_3D_SIZE 2\n" + "\n".join(f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1)) + "\n")
        writer = NVHardwareVideoWriter(folder / "encoded.mp4", decoder.shape, decoder.rate,
            hardware=True, strict=True, lut_path=lut, lut_bake=True)
        try:
            for image, metadata in decoder.frames():
                writer.write(image)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        destination = folder / "remuxed.mp4"
        NVHardwareVideoWriter.remux_audio(folder / "encoded.mp4", source, destination)
        assert packet_digest(folder / "encoded.mp4", "video") == packet_digest(destination, "video")
        assert packet_digest(source, "audio") == packet_digest(destination, "audio")
        from roop_3d.optimized_processor import StreamingProcessor
        from roop_3d.render_policy import RenderPolicy
        def detect(image):
            return [1] if image[10:40, :25].mean() > 180 else []
        def track(image):
            return [(0, np.array([-10, 0, 42, 52]), points + [-10, 0])] if detect(image) else []
        coupled = StreamingProcessor(folder, "native_pipeline", source, detect=detect, track=track, pipeline=engine,
            policy=RenderPolicy(hardware_profile="6gb", strict_nvdec=True, strict_nvenc=True,
                sparse_interval=2, range_padding=2, execution_threads=3, ring_slots=3, segment_frames=3))
        coupled_path = folder / "native_pipeline.mp4"
        coupled_result = coupled.run(coupled_path)
        assert coupled_result["frames"] == 10 and coupled_result["verified_target_frames"] == 5
        assert packet_digest(source, "audio") == packet_digest(coupled_path, "audio")
        return {"cuda_affine_composite": "PASS", "execution_threads": 3, "max_uint8_pixel_error": max(errors),
                "decoder": decoder.backend, "encoder": writer.codec, "frames": writer.frames,
                "lut_bake": "PASS", "remux_video_packets": "BYTE_IDENTICAL", "remux_audio_packets": "BYTE_IDENTICAL",
                "three_stage_native_pipeline": "PASS", "native_pipeline_frames": coupled_result["frames"],
                "scope": "synthetic geometry/identity workload; production-model fidelity and FPS unmeasured"}
