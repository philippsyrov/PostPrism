"""Frozen configuration and trust policy for Qwen ASR."""

from __future__ import annotations

import re
import zlib

from postprism.asr.audio_class import (
    AUDIO_CLASS_INCOMPLETE,
    AUDIO_CLASS_MIXED,
    AUDIO_CLASS_MUSIC,
    AUDIO_CLASS_SPEECH,
)

DEFAULT_QWEN_MODEL_NAME: str = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_QWEN_MODEL_REVISION: str = "bda30b99af3446a954339e28465acf530464e52e"
DEFAULT_QWEN_DEVICE: str = "cuda:0"
DEFAULT_QWEN_COMPUTE_TYPE: str = "bfloat16"
DEFAULT_QWEN_MAX_NEW_TOKENS: int = 512
DEFAULT_QWEN_MAX_BATCH_SIZE: int = 8
# qwen-v5: long-media probes use the container origin, so non-zero-start
# media can yield a different transcript than under qwen-v4. These are
# different ASR extractions; the Qwen embedding identity is unaffected.
DEFAULT_QWEN_MODEL_VERSION: str = "qwen3-asr-0.6b-bfloat16-cuda0-vadgate2-astgate1-qwen-v5"

# Media up to this long is decoded in full and selected on the active-time
# axis (the accepted vadgate2 behaviour). Longer media is probed instead:
# `speech_windows` equal spans, each decoded in seek-based
# TIMELINE_PROBE_SECONDS chunks up to TIMELINE_PROBE_MAX_SECONDS until the
# per-window activity budget is met. The beginning, middle, and end are
# sampled without decoding all of the media; speech between the probes can
# still be missed.
FULL_DECODE_MAX_SECONDS: float = 600.0
TIMELINE_PROBE_SECONDS: float = 60.0
TIMELINE_PROBE_MAX_SECONDS: float = 120.0
MIN_AUDIO_SECONDS: float = 0.5

# Calibrated launch constants for the one supported service policy.
DEFAULT_MIN_SPEECH_RATIO: float = 0.30
DEFAULT_MAX_COMPRESSION_RATIO: float = 2.4
NO_SPEECH_RATIO_FLOOR: float = 0.05

# The 30 s activity cap bounds inference while deterministic windows preserve
# beginning/middle/end coverage.
DEFAULT_QWEN_MAX_ACTIVE_SECONDS: float = 30.0
DEFAULT_QWEN_SPEECH_WINDOWS: int = 3
# Density thresholds fail closed on long sparse music-shaped transcripts and
# physically implausible decoder output.
DEFAULT_MAX_CHARS_PER_SPEECH_SECOND: float = 40.0
DEFAULT_QWEN_LONG_CLIP_SECONDS: float = 20.0
DEFAULT_MIN_LONG_CLIP_DENSITY: float = 5.0
QWEN_REPETITION_MIN_TOKENS: int = 12
QWEN_REPETITION_DOMINANT_SHARE: float = 0.8
QWEN_LOW_INFORMATION_MIN_SECONDS: float = 8.0
QWEN_LOW_INFORMATION_MIN_TOKENS: int = 3
QWEN_LOW_INFORMATION_MIN_LETTERS: int = 16

_TRANSCRIPT_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _transcript_compression_ratio(transcript: str) -> float | None:
    """Return whole-transcript zlib compression ratio, or `None` if empty."""
    encoded = transcript.encode("utf-8")
    if not encoded:
        return None
    return len(encoded) / len(zlib.compress(encoded))


def _is_dominant_token_repetition(transcript: str) -> bool:
    """Whether one token dominates a long-enough transcript."""
    tokens = [t.casefold() for t in _TRANSCRIPT_TOKEN_RE.findall(transcript)]
    if len(tokens) < QWEN_REPETITION_MIN_TOKENS:
        return False
    top_count = max(tokens.count(token) for token in set(tokens))
    return top_count / len(tokens) >= QWEN_REPETITION_DOMINANT_SHARE


def _is_low_information_transcript(transcript: str, sent_seconds: float) -> bool:
    """Whether sustained speech-shaped audio yielded too little lexical evidence."""
    if sent_seconds < QWEN_LOW_INFORMATION_MIN_SECONDS:
        return False
    tokens = _TRANSCRIPT_TOKEN_RE.findall(transcript)
    letters = sum(character.isalpha() for character in transcript)
    return (
        len(tokens) < QWEN_LOW_INFORMATION_MIN_TOKENS and letters < QWEN_LOW_INFORMATION_MIN_LETTERS
    )


def derive_qwen_trust_decision(
    transcript: str,
    speech_ratio: float | None,
    sent_seconds: float,
    min_speech_ratio: float = DEFAULT_MIN_SPEECH_RATIO,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    max_chars_per_speech_second: float = DEFAULT_MAX_CHARS_PER_SPEECH_SECOND,
    long_clip_seconds: float = DEFAULT_QWEN_LONG_CLIP_SECONDS,
    min_long_clip_density: float = DEFAULT_MIN_LONG_CLIP_DENSITY,
    audio_class: str | None = None,
    enforce_audio_class: bool = False,
) -> tuple[bool, str | None]:
    """Return the first-failure qwen-v5 fusion decision and reason.

    All inputs are persisted or versioned. Audio-class failures precede the
    transcript gates, and only an explicit `speech` verdict may pass when the
    detector is enforced.
    """
    if enforce_audio_class:
        if audio_class == AUDIO_CLASS_MUSIC:
            return False, "music_audio"
        if audio_class == AUDIO_CLASS_MIXED:
            return False, "mixed_audio"
        if audio_class == AUDIO_CLASS_INCOMPLETE:
            return False, "incomplete_audio"
        if audio_class != AUDIO_CLASS_SPEECH:
            return False, "uncertain_audio"
    if speech_ratio is None or speech_ratio < min_speech_ratio:
        return False, "low_speech_ratio"
    compression_ratio = _transcript_compression_ratio(transcript)
    if compression_ratio is not None and compression_ratio >= max_compression_ratio:
        return False, "repetitive_transcript"
    if _is_dominant_token_repetition(transcript):
        return False, "repetitive_transcript"
    density = len(transcript) / sent_seconds if sent_seconds > 0.0 else None
    if (
        density is not None
        and sent_seconds >= long_clip_seconds
        and density < min_long_clip_density
    ):
        return False, "sparse_long_audio"
    if density is not None and density > max_chars_per_speech_second:
        return False, "implausible_density"
    if _is_low_information_transcript(transcript, sent_seconds):
        return False, "low_information_transcript"
    return True, None
