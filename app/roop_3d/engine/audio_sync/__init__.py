"""Local audio feature, rational alignment and evidence-gated lip-sync contracts."""

from roop_3d.engine.audio_sync.audio_alignment import AudioWindow, frame_sample_index, window_for_frame
from roop_3d.engine.audio_sync.audio_features import AudioFeatures, AudioSignal, decode_audio, extract_log_mel
from roop_3d.engine.audio_sync.lip_sync_adapter import LipSyncAdapter, LipSyncFrameInput, LipSyncResult
from roop_3d.engine.audio_sync.track_selection import AudioTrackChoice, choose_track, decode_selected

__all__ = [
    "AudioFeatures",
    "AudioSignal",
    "AudioTrackChoice",
    "AudioWindow",
    "LipSyncAdapter",
    "LipSyncFrameInput",
    "LipSyncResult",
    "choose_track",
    "decode_audio",
    "decode_selected",
    "extract_log_mel",
    "frame_sample_index",
    "window_for_frame",
]
