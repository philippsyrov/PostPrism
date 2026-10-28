"""Optional metadata must not change retrieval inputs, rollups, or serialization."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from postprism.api import enrich as enrich_api
from postprism.captioning.identity import CAPTION_IDENTITY
from postprism.schemas import TranscriptionResult
from postprism.visual_tags.identity import RAM_IDENTITY
from tests.enrich_endpoint_support import (
    FakeMultimodalEmbedder,
    FakeTranscriber,
    RecordingMetadataWorker,
)


def video_files():
    return [
        ("files", ("first.mp4", b"first video", "video/mp4")),
        ("files", ("second.mp4", b"second video", "video/mp4")),
        ("files", ("duplicate.mp4", b"first video", "video/mp4")),
    ]


def install_workers(monkeypatch, embedder, ram=None, captioner=None, asr=None):
    for name, worker in (
        ("get_multimodal_embedder", embedder),
        ("get_visual_tagger", ram),
        ("get_captioner", captioner),
        ("get_transcriber", asr),
    ):
        monkeypatch.setattr(enrich_api, name, lambda _, worker=worker: worker, raising=False)


@pytest.mark.parametrize("ram_status", ["ok", "partial", "failed"])
@pytest.mark.parametrize("caption_status", ["ok", "partial", "failed"])
@pytest.mark.parametrize("trusted", [True, False, "true"])
@pytest.mark.parametrize("second_status", ["ok", "transcription_failed"])
def test_metadata_statuses_leave_canonical_inputs_and_rollup_unchanged(
    monkeypatch, ram_status, caption_status, trusted, second_status
):
    class Transcriber:
        def __init__(self):
            self.paths = []

        def transcribe(self, path):
            index = len(self.paths)
            self.paths.append(path)
            return TranscriptionResult(
                asr_transcript="accepted speech" if index == 0 else "rejected speech",
                language="en",
                duration_seconds=900.0,
                asr_model_version="qwen-v5",
                asr_used_for_embedding=trusted if index == 0 else False,
                status="ok" if index == 0 else second_status,
                duration_sent_to_asr=27.125,
                asr_audio_selection="timeline-probes",
            )

    baseline_embedder = FakeMultimodalEmbedder()
    install_workers(monkeypatch, baseline_embedder, asr=Transcriber())
    client = TestClient(app)
    baseline = client.post("/enrich", data={"text": "author"}, files=video_files()).json()

    embedder, asr = FakeMultimodalEmbedder(), Transcriber()
    ram = RecordingMetadataWorker(RAM_IDENTITY, status=ram_status)
    captioner = RecordingMetadataWorker(CAPTION_IDENTITY, caption=True, status=caption_status)
    install_workers(monkeypatch, embedder, ram, captioner, asr)
    response = client.post("/enrich", data={"text": "author"}, files=video_files())
    assert response.status_code == 200
    payload = response.json()
    assert payload["visual_tag_result"] is not None
    assert payload["caption_result"] is not None
    assert payload["visual_tag_result"]["status"] == ram_status
    assert payload["caption_result"]["status"] == caption_status
    for key in ("content_enrichment_status", "media_items", "media_summary"):
        assert payload[key] == baseline[key]
    assert payload["content_enrichment_status"] == ("ok" if second_status == "ok" else "partial")
    assert {
        key: value
        for key, value in payload["multimodal_result"].items()
        if key != "multimodal_embedded_at"
    } == {
        key: value
        for key, value in baseline["multimodal_result"].items()
        if key != "multimodal_embedded_at"
    }
    expected_text = (
        "Post text: author\n\nTrusted speech transcript: accepted speech"
        if trusted is True
        else "Post text: author"
    )
    assert embedder.calls[0]["assembled_text"] == expected_text
    assert embedder.calls[0]["assembled_text"] == baseline_embedder.calls[0]["assembled_text"]
    assert "CAPTION_SENTINEL" not in embedder.calls[0]["assembled_text"]
    assert "TAG_SENTINEL" not in embedder.calls[0]["assembled_text"]
    assert len(asr.paths) == 2
    assert [row["source_position"] for row in payload["asr_results"]] == [0, 1]
    assert payload["asr_results"][0]["asr_used_for_embedding"] == trusted
    for row in payload["asr_results"]:
        assert row["duration_sent_to_asr"] == 27.125
        assert row["asr_audio_selection"] == "timeline-probes"
    for key in ("visual_tag_result", "caption_result"):
        assert payload[key]["items"][-1] == {
            "source_position": 2,
            "kind": "duplicate",
            "status": "skipped_duplicate",
            "duplicate_of_position": 0,
        }
    assert len(ram.calls) == len(captioner.calls) == 1
    assert set(payload["leg_timings_ms"]) == set(baseline["leg_timings_ms"]) | {
        "visual_tags",
        "captions",
    }
    assert payload["leg_timings_ms"]["video_decode"] == 0.0
    assert payload["leg_timings_ms"]["qwen_processing"] == 0.1
    assert "lock_wait" in payload["leg_timings_ms"]


@pytest.mark.parametrize(
    "ram_enabled,caption_enabled", [(True, True), (True, False), (False, True)]
)
def test_text_only_metadata_is_skipped_and_workers_are_independent(
    monkeypatch, ram_enabled, caption_enabled
):
    ram = RecordingMetadataWorker(RAM_IDENTITY) if ram_enabled else None
    captioner = RecordingMetadataWorker(CAPTION_IDENTITY, caption=True) if caption_enabled else None
    install_workers(monkeypatch, FakeMultimodalEmbedder("text_only_ok"), ram, captioner)
    payload = TestClient(app).post("/enrich", data={"text": "hello"}).json()
    for key, worker in (("visual_tag_result", ram), ("caption_result", captioner)):
        if worker is None:
            assert payload[key] is None
        else:
            assert payload[key] is not None
            assert payload[key]["status"] == "skipped_no_media"
            assert payload[key]["items"] == []
            assert worker.calls == [(None, {})]


def test_entire_optional_media_path_runs_in_dispatch_under_existing_lock(monkeypatch):
    events = []
    dispatch_active = False
    lock_active = False
    materialized_paths = []

    class RecordingLock:
        def __enter__(self):
            nonlocal lock_active
            assert dispatch_active
            lock_active = True

        def __exit__(self, *args):
            nonlocal lock_active
            lock_active = False

    def check_model(name, paths):
        assert dispatch_active and lock_active
        assert paths and all(Path(path).is_file() for path in paths)
        events.append(name)

    class Embedder(FakeMultimodalEmbedder):
        def embed(self, bundle, paths, text, asr_results, timings=None):
            check_model("embedding", paths.values())
            return super().embed(bundle, paths, text, asr_results, timings)

    class Transcriber(FakeTranscriber):
        def transcribe(self, path):
            check_model("asr", [path])
            return super().transcribe(path)

    original_materialized = enrich_api.materialized_post_media

    @contextmanager
    def materialized(bundle):
        assert dispatch_active
        with original_materialized(bundle) as paths:
            materialized_paths.extend(paths.values())
            yield paths
        assert dispatch_active

    async def dispatch(function, /, *args, **kwargs):
        nonlocal dispatch_active
        assert function is enrich_api.run_enrichment
        events.append("dispatch")
        dispatch_active = True
        try:
            return function(*args, **kwargs)
        finally:
            dispatch_active = False

    monkeypatch.setattr(enrich_api, "_MODEL_INFERENCE_LOCK", RecordingLock())
    monkeypatch.setattr(enrich_api, "materialized_post_media", materialized)
    monkeypatch.setattr(enrich_api, "run_in_threadpool", dispatch)
    ram = RecordingMetadataWorker(
        RAM_IDENTITY, on_call=lambda bundle, paths: check_model("ram", paths.values())
    )
    captioner = RecordingMetadataWorker(
        CAPTION_IDENTITY,
        caption=True,
        on_call=lambda bundle, paths: check_model("caption", paths.values()),
    )
    install_workers(monkeypatch, Embedder(), ram, captioner, Transcriber())
    response = TestClient(app).post("/enrich", files=video_files())
    assert response.status_code == 200
    assert events == ["dispatch", "asr", "asr", "embedding", "ram", "caption"]
    assert materialized_paths and not any(path.exists() for path in materialized_paths)


def test_blocked_caption_prevents_second_request_entering_any_model_leg(monkeypatch):
    caption_entered, release_caption, second_attempt = (threading.Event() for _ in range(3))
    actual_lock = threading.Lock()
    events = []

    class ObservedLock:
        attempts = 0

        def __enter__(self):
            self.attempts += 1
            if self.attempts == 2:
                second_attempt.set()
            actual_lock.acquire()

        def __exit__(self, *args):
            actual_lock.release()

    class Embedder(FakeMultimodalEmbedder):
        def embed(self, bundle, paths, text, asr_results, timings=None):
            events.append("embedding")
            return super().embed(bundle, paths, text, asr_results, timings)

    class Transcriber(FakeTranscriber):
        def transcribe(self, path):
            events.append("asr")
            return super().transcribe(path)

    def caption_call(bundle, paths):
        events.append("caption")
        if not caption_entered.is_set():
            caption_entered.set()
            assert release_caption.wait(timeout=5)

    ram = RecordingMetadataWorker(RAM_IDENTITY, on_call=lambda *_: events.append("ram"))
    captioner = RecordingMetadataWorker(CAPTION_IDENTITY, caption=True, on_call=caption_call)
    install_workers(monkeypatch, Embedder(), ram, captioner, Transcriber())
    monkeypatch.setattr(enrich_api, "_MODEL_INFERENCE_LOCK", ObservedLock())

    def post():
        return TestClient(app).post("/enrich", files=video_files()[:1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(post)
        try:
            assert caption_entered.wait(timeout=2)
            assert actual_lock.locked()
            second = pool.submit(post)
            assert second_attempt.wait(timeout=2)
            assert events == ["asr", "embedding", "ram", "caption"]
        finally:
            release_caption.set()
        assert first.result(timeout=5).status_code == 200
        assert second.result(timeout=5).status_code == 200
    assert events == ["asr", "embedding", "ram", "caption"] * 2


def test_optional_programming_errors_are_not_silenced(monkeypatch):
    def fail(*args):
        raise RuntimeError("programming error")

    ram = RecordingMetadataWorker(RAM_IDENTITY, on_call=fail)
    install_workers(monkeypatch, FakeMultimodalEmbedder(), ram)
    with pytest.raises(RuntimeError, match="programming error"):
        TestClient(app).post("/enrich", files=video_files())


def test_optional_workers_do_not_add_routes_or_http_configuration():
    assert {(route.path, frozenset(route.methods)) for route in app.routes} == {
        ("/enrich", frozenset({"POST"})),
        ("/health", frozenset({"GET"})),
    }
    route = next(route for route in app.routes if route.path == "/enrich")
    assert {field.name for field in route.dependant.body_params} == {"files", "text"}
