"""Studio contracts independent of physical cameras and proprietary weights."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.swapper_engine import SwapperEngineAdapter
from roop_3d.engine.audio_sync.audio_features import AudioFeatures, AudioSignal, LipSyncer
from roop_3d.engine.audio_sync.lip_sync_adapter import LipSyncAdapter, LipSyncFrameInput
from roop_3d.server.live_studio import StudioConfig, LiveStudio


def node(name, shape, dtype='tensor(float)'):
    return SimpleNamespace(name=name, shape=shape, type=dtype)


class DfmSession:
    def get_inputs(self):
        return [node('in_face:0', [1, 64, 64, 3]), node('morph_value:0', [1])]

    def get_outputs(self):
        return [node('mask', [1, 64, 64, 1]), node('celeb', [1, 64, 64, 3]), node('celeb_mask', [1, 64, 64, 1])]


def test_dfm_bgr_morph_and_masks():
    catalog = SimpleNamespace(swapper_spec=lambda _: {'category': 'swapper', 'native_resolution': 0, 'identity_mode': 'dfm'})
    def run(feed):
        assert feed['in_face:0'].shape == (1, 64, 64, 3)
        assert feed['in_face:0'][0, 0, 0, 0] == 1
        assert feed['morph_value:0'][0] == np.float32(.3)
        mask = np.ones((1, 64, 64, 1), np.float32)
        mask[:, :32] = 0
        return {'mask': mask, 'celeb': np.zeros((1, 64, 64, 3), np.float32), 'celeb_mask': np.ones_like(mask)}
    adapter = SwapperEngineAdapter(catalog, 'local_identity', session=DfmSession(), runner=run)
    crop = np.zeros((64, 64, 3), np.uint8)
    crop[:, :, 0] = 255
    result = adapter.infer(crop, np.zeros(512), morph_factor=.3)
    np.testing.assert_array_equal(result[:32], crop[:32])
    assert not result[32:].any()
    assert adapter.resolution == 64 and adapter.executions == 1
    with pytest.raises(DomainError):
        adapter.infer(crop, np.zeros(512), morph_factor=1.2)


def test_dfm_rejects_unknown_inputs():
    class Invalid(DfmSession):
        def get_inputs(self):
            return [node('in_face:0', [1, 64, 64, 3]), node('unrecognized', [1])]
    catalog = SimpleNamespace(swapper_spec=lambda _: {'category': 'swapper', 'native_resolution': 0, 'identity_mode': 'dfm'})
    with pytest.raises(DomainError):
        SwapperEngineAdapter(catalog, 'local_identity', session=Invalid(), runner=lambda _: {})


def test_audio_pts_and_silence_padding():
    syncer = object.__new__(LipSyncer)
    signal = AudioSignal(np.zeros(16000, np.float32), start_time=Fraction(1, 10))
    syncer.features = AudioFeatures(signal, np.tile(np.arange(80, dtype=np.float32), (80, 1)))
    frame = syncer.frame_input(Fraction(3, 10))
    assert frame.mel_window[0, 0] == 16
    assert frame.timestamp == Fraction(3, 10)
    assert np.all(syncer.frame_input(Fraction(2)).mel_window == -4)


def test_fp16_lipsync_preserves_silent_closures():
    adapter = LipSyncAdapter(SimpleNamespace(registry=SimpleNamespace(records=lambda: [])))
    adapter.available = True
    adapter.image_node = node('face', [1, 6, 96, 96], 'tensor(float16)')
    adapter.mel_node = node('mel', [1, 1, 80, 16], 'tensor(float16)')
    def run(_, feeds):
        assert feeds['face'].dtype == np.float16 and feeds['mel'].dtype == np.float16
        assert not feeds['face'][:, :3, 48:].any()
        return [np.ones((1, 3, 96, 96), np.float16)]
    adapter.session = SimpleNamespace(run=run)
    result = adapter.process_frame(np.zeros((64, 64, 3), np.uint8), 0,
                                   LipSyncFrameInput(np.full((80, 16), -4, np.float32), 0, 1))
    assert result.applied and result.blend == 1 and result.image.min() == 255
    adapter.release()
    assert not adapter.available and adapter.session is None


def test_live_studio_reports_measured_scope():
    status = LiveStudio(SimpleNamespace()).snapshot()
    assert status['state'] == 'stopped'
    assert status['zero_copy'] is False and status['end_to_end_verified'] is False
    assert status['capture_to_publish_p95_ms'] is None
    with pytest.raises(ValueError):
        StudioConfig(source_faceset_id='a', fps=120)


def test_scanned_dfm_runs_real_onnx_session(tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from roop_3d.models.catalog import ModelCatalog
    from roop_3d.models.registry import Manifest, ModelRegistry
    models = tmp_path / 'models'
    models.mkdir()
    mask = np.ones((1, 64, 64, 1), np.float32)
    graph = helper.make_graph([
        helper.make_node('Identity', ['in_face:0'], ['celeb']),
        helper.make_node('Constant', [], ['mask'], value=numpy_helper.from_array(mask)),
        helper.make_node('Constant', [], ['celeb_mask'], value=numpy_helper.from_array(mask)),
    ], 'dfm', [helper.make_tensor_value_info('in_face:0', TensorProto.FLOAT, [1, 64, 64, 3])], [
        helper.make_tensor_value_info('mask', TensorProto.FLOAT, [1, 64, 64, 1]),
        helper.make_tensor_value_info('celeb', TensorProto.FLOAT, [1, 64, 64, 3]),
        helper.make_tensor_value_info('celeb_mask', TensorProto.FLOAT, [1, 64, 64, 1]),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 9
    onnx.save(model, str(models / 'identity.dfm'))
    registry = ModelRegistry(tmp_path, manifest=Manifest(evidence_scope='test', models=()))
    catalog = ModelCatalog(root=tmp_path, registry=registry)
    row = next(r for r in catalog.public() if r.get('local_format') == 'dfm')
    assert row['category'] == 'swapper'
    adapter = SwapperEngineAdapter(catalog, row['id'])
    try:
        image = np.full((64, 64, 3), (10, 20, 240), np.uint8)
        np.testing.assert_array_equal(adapter.infer(image, np.zeros(512)), image)
    finally:
        adapter.release()


def test_voiceover_is_muxed_and_verified_against_replacement(tmp_path):
    import av
    from roop_3d.tests.stage15_media import make_color_video
    from roop_3d.media.video_writer import encode_video
    (tmp_path / 'roop_3d').mkdir(exist_ok=True)
    source = make_color_video(tmp_path / 'roop_3d/source.mkv', frames=25, audio=0, width=96, height=64)
    voiceover = tmp_path / 'roop_3d/voice.wav'
    with av.open(str(voiceover), 'w') as container:
        stream = container.add_stream('pcm_s16le', rate=16000)
        stream.layout = 'mono'
        samples = (np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000) * 10000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples[None], format='s16', layout='mono')
        frame.sample_rate = 16000
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    manifest = encode_video(source, 'replaced.mkv', root=tmp_path, hardware='software', audio_source=voiceover)
    assert len(manifest['stage15']['audio_sync']) == 1
    assert manifest['stage15']['audio_sync'][0]['ok']


def test_live_failure_releases_capture_models_and_admission(monkeypatch):
    from roop_3d.core.settings import Settings
    from roop_3d.server.engine_gate import EngineGate
    from roop_3d.engine import pipeline_runner
    from roop_3d.engine.volumetric import session
    from roop_3d.server import live_studio
    released = []
    reader = SimpleNamespace(read=lambda *a, **kw: object(), release=lambda: released.append('reader'))
    monkeypatch.setattr(pipeline_runner, 'MasterPipelineRunner', lambda *a: SimpleNamespace(
        faceset_reader=reader, _faceset_path=lambda value: value))
    class Pipeline:
        swapper = SimpleNamespace(model=None)
        def __init__(self, *a): pass
        def select_swapper(self, *a): pass
        def prepare_session(self, *a, **kw): pass
        def release_session(self): released.append('session')
    monkeypatch.setattr(session, 'ReferencePipeline', Pipeline)
    monkeypatch.setattr(live_studio.cv2, 'VideoCapture', lambda *a: SimpleNamespace(
        isOpened=lambda: False, release=lambda: released.append('camera')))
    scheduler = SimpleNamespace(try_admit=lambda *a: SimpleNamespace(provider='CPUExecutionProvider', profile='cpu'),
                                release=lambda key: released.append('admission'))
    gate = EngineGate()
    state = SimpleNamespace(settings=Settings(), previews=SimpleNamespace(scheduler=scheduler),
                            engine_gate=gate, root=None, models=None, media=None)
    studio = LiveStudio(state)
    studio._run(StudioConfig(source_faceset_id='alice'))
    assert studio.snapshot()['state'] == 'error'
    assert released == ['camera', 'session', 'reader', 'admission']
    assert gate.previews_in_flight == 0


def test_wav2lip_frontend_is_finite_and_normalized():
    pytest.importorskip("librosa", reason="Wav2Lip audio features require librosa")
    from roop_3d.engine.audio_sync.audio_features import extract_wav2lip_mel
    samples = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000).astype(np.float32)
    features = extract_wav2lip_mel(AudioSignal(samples))
    assert features.mel.shape == (80, 81)
    assert np.isfinite(features.mel).all()
    assert -4 <= features.mel.min() <= features.mel.max() <= 4
