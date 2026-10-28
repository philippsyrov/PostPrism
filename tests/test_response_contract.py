import pytest
from fastapi.testclient import TestClient

from app import app
from postprism.api import enrich as enrich_api
from postprism.schemas import TranscriptionResult
from tests.enrich_endpoint_support import FakeMultimodalEmbedder


def test_unspecified_asr_trust_defaults_to_not_used_for_embedding() -> None:
    result = TranscriptionResult(
        asr_transcript="unchecked speech",
        language="en",
        duration_seconds=1.0,
    )

    assert result.asr_used_for_embedding is False


def test_text_only_response_contains_only_live_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: FakeMultimodalEmbedder(status="text_only_ok"),
    )

    payload = TestClient(app).post("/enrich", data={"text": "hello"}).json()

    assert set(payload) == {
        "content_enrichment_status",
        "multimodal_result",
        "asr_results",
        "visual_tag_result",
        "caption_result",
        "request_timings_ms",
        "leg_timings_ms",
    }
    assert "text_truncated" not in payload["multimodal_result"]
    assert "topic" not in payload["leg_timings_ms"]
    assert payload["asr_results"] == []
    assert payload["visual_tag_result"] is None
    assert payload["caption_result"] is None


def test_media_response_adds_only_media_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: FakeMultimodalEmbedder(),
    )

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            files={"files": ("image.png", b"not-real-png", "image/png")},
        )
        .json()
    )

    assert set(payload) == {
        "content_enrichment_status",
        "multimodal_result",
        "asr_results",
        "visual_tag_result",
        "caption_result",
        "media_items",
        "media_summary",
        "request_timings_ms",
        "leg_timings_ms",
    }
    assert "topic" not in payload["leg_timings_ms"]
    assert payload["visual_tag_result"] is None
    assert payload["caption_result"] is None
