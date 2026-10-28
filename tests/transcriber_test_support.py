"""Shared deterministic fakes and media helpers for transcriber tests."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from postprism.asr.audio_class import (
    DEFAULT_AUDIO_CLASS_REVISION,
    MUSIC_LABELS,
    SINGING_LABELS,
    SPEECH_LABELS,
)

requires_av = pytest.mark.skipif(
    importlib.util.find_spec("av") is None,
    reason="PyAV (`av` extra) not installed",
)


class FakeQwenResult:
    """Stand-in for one qwen-asr transcription result."""

    def __init__(
        self,
        text: str = "привет мир",
        language: str = "Russian",
        duration_seconds: float = 3.5,
    ) -> None:
        self.text = text
        self.language = language
        self.duration_seconds = duration_seconds


class FakeQwenModel:
    """Deterministic stand-in for `qwen_asr.Qwen3ASRModel`."""

    def __init__(
        self,
        result: FakeQwenResult | dict[str, object] | None = None,
        raise_on_transcribe: BaseException | None = None,
    ) -> None:
        self.result = result if result is not None else FakeQwenResult()
        self.raise_on_transcribe = raise_on_transcribe
        self.transcribe_calls: list[dict[str, Any]] = []

    def transcribe(self, audio: str, **kwargs: Any) -> list[object]:
        self.transcribe_calls.append({"audio": audio, **kwargs})
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        return [self.result]


class SnoopQwenModel(FakeQwenModel):
    """Fake Qwen that records the duration and path of its WAV input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sent_wav_path: Path | None = None
        self.sent_wav_seconds: float | None = None

    def transcribe(self, audio: str, **kwargs: Any) -> list[object]:
        self.sent_wav_path = Path(audio)
        with wave.open(str(audio)) as wav_file:
            self.sent_wav_seconds = wav_file.getnframes() / wav_file.getframerate()
        return super().transcribe(audio, **kwargs)


def patch_qwen_loader(monkeypatch: pytest.MonkeyPatch, model: FakeQwenModel) -> None:
    """Replace the model loader so warmup never touches HF or a GPU."""
    monkeypatch.setattr(
        "postprism.asr.transcriber._load_qwen_asr_model",
        lambda model_name, device, compute_type, max_new_tokens, max_batch_size, model_revision: (
            model
        ),
    )


def write_wav_media(
    path: Path,
    active_seconds: float,
    silence_seconds: float = 0.0,
    amplitude: float = 0.3,
) -> None:
    """Synthesize a mono 16 kHz WAV: a tone burst followed by silence."""
    sample_rate = 16000
    timeline = np.arange(int(active_seconds * sample_rate)) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 440.0 * timeline) * 32767).astype(np.int16)
    silence = np.zeros(int(silence_seconds * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.concatenate([tone, silence]).tobytes())


class FakeAudioTagger:
    """Stand-in for the AST AudioSet tagger with fixed per-chunk scores."""

    def __init__(
        self,
        label_scores: dict[str, float] | None = None,
        raise_on_scores: BaseException | None = None,
    ) -> None:
        self.label_scores = label_scores or {}
        self.raise_on_scores = raise_on_scores
        self.id2label = dict(enumerate(sorted(SPEECH_LABELS | SINGING_LABELS | MUSIC_LABELS)))
        self.resolved_revision: str | None = DEFAULT_AUDIO_CLASS_REVISION
        self.dtype: str | None = "torch.float32"
        self.scored_seconds: float | None = None
        self.scored_chunks: list[np.ndarray] = []
        self.calls = 0

    def scores(self, chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
        self.calls += 1
        self.scored_seconds = sum(len(chunk) for chunk in chunks) / sample_rate
        self.scored_chunks = [np.array(chunk, copy=True) for chunk in chunks]
        if self.raise_on_scores is not None:
            raise self.raise_on_scores
        index_of = {label: index for index, label in self.id2label.items()}
        matrix = np.zeros((len(chunks), len(self.id2label)), dtype=np.float64)
        for label, value in self.label_scores.items():
            matrix[:, index_of[label]] = value
        return matrix
