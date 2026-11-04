"""Explicit audio-track selection for speech-driven features.

A file can carry several audio tracks (languages, commentary, stems). A feature that *reads* speech must
never guess which one: with more than one track the caller has to name it by stream index or language.
Selection only reads; no track is ever modified, muted, replaced or re-timed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class AudioTrackChoice:
    stream_index: int
    language: str | None
    codec: str
    reason: str


def choose_track(tracks: Sequence, *, stream_index: int | None = None, language: str | None = None) -> AudioTrackChoice:
    """Pick one of ``tracks`` (anything with ``index``, ``codec`` and ``language``) by explicit request."""
    if not tracks:
        raise DomainError("AUDIO_STREAM_MISSING", "Media contains no audio stream", 422)
    if stream_index is not None and language is not None:
        raise DomainError("AUDIO_TRACK_SELECTION_INVALID", "Select by stream index or by language, not both", 422)
    if stream_index is not None:
        match = [t for t in tracks if t.index == stream_index]
        if not match:
            raise DomainError("AUDIO_STREAM_MISSING", "Requested audio stream index is unavailable", 422)
        reason = "selected by stream index"
    elif language is not None:
        match = [t for t in tracks if (t.language or "").casefold() == language.casefold()]
        if len(match) != 1:
            raise DomainError("AUDIO_TRACK_AMBIGUOUS" if match else "AUDIO_STREAM_MISSING",
                              f"{len(match)} audio tracks match language {language!r}", 422)
        reason = "selected by language"
    elif len(tracks) == 1:
        match, reason = [tracks[0]], "the only audio track"
    else:
        raise DomainError("AUDIO_TRACK_AMBIGUOUS",
                          f"{len(tracks)} audio tracks: choose one explicitly by stream index or language", 422)
    chosen = match[0]
    return AudioTrackChoice(chosen.index, chosen.language, chosen.codec, reason)


def decode_selected(path: Path, *, stream_index: int | None = None, language: str | None = None, target_rate: int = 16_000):
    """Probe the file's audio tracks, require an explicit choice when there are several, and decode that one."""
    from roop_3d.engine.audio_sync.audio_features import decode_audio
    from roop_3d.media.probe import probe
    choice = choose_track(probe(Path(path)).audio, stream_index=stream_index, language=language)
    return choice, decode_audio(Path(path), stream_index=choice.stream_index, target_rate=target_rate)
