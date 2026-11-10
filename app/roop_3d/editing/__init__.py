# SPDX-License-Identifier: MIT
"""Pre-render editing of the tracked target performance: emotion offsets and audio-locked lip/jaw motion."""

from roop_3d.editing.expression_audio_director import (
    AudioViseme,
    AudioVisemeTrack,
    DirectedExpression,
    DirectorSettings,
    EMOTIONS,
    EmotionBasis,
    EmotionOffsets,
    ExpressionAndAudioDirector,
)

__all__ = [
    "AudioViseme",
    "AudioVisemeTrack",
    "DirectedExpression",
    "DirectorSettings",
    "EMOTIONS",
    "EmotionBasis",
    "EmotionOffsets",
    "ExpressionAndAudioDirector",
]
