from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from postprism.multimodal.input import assemble_multimodal_text
from postprism.schemas import (
    MultimodalEmbeddingResult,
    TranscriptionResult,
    VisualMetadataResult,
)


class FakeMultimodalEmbedder:
    def __init__(self, status: str = "ok") -> None:
        self.status = status
        self.calls: list[dict[str, object]] = []

    def embed(
        self,
        bundle: object | None,
        paths: dict[int, Path],
        post_text: str | None,
        asr_results: list[dict[str, object]],
        timings: dict[str, float] | None = None,
    ) -> MultimodalEmbeddingResult:
        distinct_positions = tuple(getattr(bundle, "distinct_positions", ()))
        if timings is not None:
            timings["video_decode"] = 0.0
            timings["qwen_processing"] = 0.1
        self.calls.append(
            {
                "assembled_text": assemble_multimodal_text(post_text, asr_results),
                "asr_results": list(asr_results),
                "paths_exist": all(path.is_file() for path in paths.values()),
                "source_positions": list(distinct_positions),
            }
        )
        return MultimodalEmbeddingResult(
            multimodal_embedding=[1.0, 0.0] if self.status in {"ok", "text_only_ok"} else [],
            embedding_dim=1024,
            multimodal_embedding_model="fake-qwen",
            multimodal_embedding_model_version="fake-qwen-v1",
            multimodal_embedding_model_revision="a" * 40,
            multimodal_embedding_method_version="fake-method-v1",
            multimodal_embedding_status=self.status,
            input_evidence=[{"source_position": p, "kind": "image"} for p in distinct_positions],
        )


class FakeTranscriber:
    """Per-call verdicts let one fake describe several videos in one post."""

    def __init__(
        self,
        *,
        trusted: bool | Sequence[bool] = True,
        statuses: Sequence[str] = (),
        transcripts: Sequence[str] = (),
    ) -> None:
        self.trusted = trusted
        self.statuses = tuple(statuses)
        self.transcripts = tuple(transcripts)
        self.calls: list[str] = []

    def transcribe(self, path: str) -> TranscriptionResult:
        index = len(self.calls)
        self.calls.append(path)
        trusted = self.trusted if isinstance(self.trusted, bool) else self.trusted[index]
        status = self.statuses[index] if index < len(self.statuses) else "ok"
        transcript = self.transcripts[index] if index < len(self.transcripts) else "trusted speech"
        return TranscriptionResult(
            asr_transcript=transcript if status == "ok" else "",
            language="en",
            duration_seconds=1.0,
            asr_model="fake-asr",
            asr_model_version="fake-asr-v1",
            status=status,
            asr_used_for_embedding=trusted and status == "ok",
        )


class RecordingMetadataWorker:
    def __init__(self, identity, *, caption=False, status="ok", on_call=None):
        self.identity = identity
        self.caption = caption
        self.status = status
        self.on_call = on_call
        self.calls = []

    def enrich(self, bundle, paths):
        self.calls.append((bundle, dict(paths)))
        if self.on_call is not None:
            self.on_call(bundle, paths)
        items = []
        for item in getattr(bundle, "items", ()):
            if item.duplicate_of_position is not None:
                items.append(
                    {
                        "source_position": item.position,
                        "kind": "duplicate",
                        "status": "skipped_duplicate",
                        "duplicate_of_position": item.duplicate_of_position,
                    }
                )
                continue
            success = self.status == "ok" or (self.status == "partial" and not items)
            output = (
                {"caption": "CAPTION_SENTINEL" if success else ""}
                if self.caption
                else {
                    "native_tags": ["TAG_SENTINEL"] if success else [],
                    "display_tags": ["TAG_SENTINEL"] if success else [],
                }
            )
            items.append(
                {
                    "source_position": item.position,
                    "kind": item.upload.media_type,
                    "status": "ok" if success else "inference_failed",
                    **output,
                }
            )
        return VisualMetadataResult(self.identity, items)
