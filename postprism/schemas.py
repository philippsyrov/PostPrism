"""Response contracts emitted by the live enrichment workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

MULTIMODAL_EMBEDDING_SUCCESS_STATUSES = frozenset({"ok", "text_only_ok"})
TRANSCRIPTION_SUCCESS_STATUSES = frozenset({"ok", "no_speech", "no_audio", "too_short"})


def _round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class MetadataIdentity:
    model: str
    model_revision: str
    method_version: str
    dtype: str
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        self.health_identity()

    def health_identity(self) -> dict[str, str]:
        if self.artifacts.keys() - {
            "checkpoint_sha256",
            "vocabulary_sha256",
            "thresholds_sha256",
        }:
            raise ValueError("invalid visual metadata artifact key")
        return {
            "model": self.model,
            "model_revision": self.model_revision,
            "method_version": self.method_version,
            "dtype": self.dtype,
            **self.artifacts,
        }


@dataclass
class VisualMetadataResult:
    identity: MetadataIdentity
    items: list[dict[str, object]]
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, object]:
        statuses = [item["status"] for item in self.items if item["status"] != "skipped_duplicate"]
        if any(
            status not in {"ok", "media_load_failed", "inference_failed"} for status in statuses
        ):
            raise ValueError("invalid visual metadata item status")
        status = (
            "skipped_no_media"
            if not statuses
            else "ok"
            if all(s == "ok" for s in statuses)
            else "partial"
            if "ok" in statuses
            else "failed"
        )
        return {
            **self.identity.health_identity(),
            "status": status,
            "generated_at": self.generated_at,
            "items": self.items,
        }


@dataclass
class MultimodalEmbeddingResult:
    multimodal_embedding: list[float]
    embedding_dim: int
    multimodal_embedding_model: str
    multimodal_embedding_model_version: str
    multimodal_embedding_model_revision: str
    multimodal_embedding_method_version: str
    multimodal_embedding_status: str = "ok"
    input_evidence: list[dict[str, object]] = field(default_factory=list)
    multimodal_embedded_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "multimodal_embedding": [round(value, 6) for value in self.multimodal_embedding],
            "embedding_dim": self.embedding_dim,
            "multimodal_embedding_model": self.multimodal_embedding_model,
            "multimodal_embedding_model_version": self.multimodal_embedding_model_version,
            "multimodal_embedding_model_revision": self.multimodal_embedding_model_revision,
            "multimodal_embedding_method_version": self.multimodal_embedding_method_version,
            "multimodal_embedded_at": self.multimodal_embedded_at,
            "multimodal_embedding_status": self.multimodal_embedding_status,
            "input_evidence": self.input_evidence,
        }


@dataclass
class TranscriptionResult:
    asr_transcript: str
    language: str
    duration_seconds: float
    asr_model: str = ""
    asr_model_version: str = ""
    asr_used_for_embedding: bool = False
    duration_after_vad: float | None = None
    speech_ratio: float | None = None
    # Seconds of audio actually handed to Qwen after the activity cap. The
    # transcript-density trust rules are judged against this value (not the
    # capped activity, which can exceed it under timeline probes), so the row
    # re-derives the live decision exactly.
    duration_sent_to_asr: float | None = None
    max_compression_ratio: float | None = None
    asr_trust_reason: str | None = None
    audio_class: str | None = None
    audio_class_confidence: float | None = None
    audio_class_model: str | None = None
    audio_class_revision: str | None = None
    audio_class_coverage: float | None = None
    # `full-decode`: the whole clip was decoded and selected on the active-time
    # axis. `timeline-probes`: media longer than the full-decode ceiling was
    # sampled at bounded real-time probes, so `duration_after_vad` and
    # `speech_ratio` describe the probed audio, not the full timeline.
    asr_audio_selection: str = "full-decode"
    asr_transcribed_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    status: str = "ok"
    asr_model_revision: str = "unpinned"

    def to_dict(self) -> dict[str, object]:
        return {
            "asr_transcript": self.asr_transcript,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 3),
            "asr_model": self.asr_model,
            "asr_model_version": self.asr_model_version,
            "asr_model_revision": self.asr_model_revision,
            "asr_used_for_embedding": self.asr_used_for_embedding,
            "duration_after_vad": _round_or_none(self.duration_after_vad, 3),
            "speech_ratio": _round_or_none(self.speech_ratio, 4),
            "duration_sent_to_asr": _round_or_none(self.duration_sent_to_asr, 3),
            "max_compression_ratio": _round_or_none(self.max_compression_ratio, 4),
            "asr_trust_reason": self.asr_trust_reason,
            "audio_class": self.audio_class,
            "audio_class_confidence": _round_or_none(self.audio_class_confidence, 4),
            "audio_class_model": self.audio_class_model,
            "audio_class_revision": self.audio_class_revision,
            "audio_class_coverage": _round_or_none(self.audio_class_coverage, 4),
            "asr_audio_selection": self.asr_audio_selection,
            "asr_transcribed_at": self.asr_transcribed_at,
            "status": self.status,
        }
