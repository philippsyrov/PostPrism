import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app import app
from postprism.api import application as application_api
from postprism.api import enrich as enrich_api
from postprism.schemas import MultimodalEmbeddingResult
from tests.enrich_endpoint_support import (
    FakeMultimodalEmbedder,
    FakeTranscriber,
)


def test_concurrent_requests_do_not_overlap_shared_model_inference() -> None:
    start = threading.Barrier(2)
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0

    class OverlapTrackingEmbedder:
        def embed(self, *_args: object) -> MultimodalEmbeddingResult:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return MultimodalEmbeddingResult(
                multimodal_embedding=[1.0, 0.0],
                embedding_dim=1024,
                multimodal_embedding_model="fake-qwen",
                multimodal_embedding_model_version="fake-qwen-v1",
                multimodal_embedding_model_revision="a" * 40,
                multimodal_embedding_method_version="fake-method-v1",
                multimodal_embedding_status="text_only_ok",
            )

    embedder = OverlapTrackingEmbedder()

    def run_one() -> None:
        start.wait()
        enrich_api.run_enrichment(
            uploads=(),
            text="hello",
            multimodal_embedder=embedder,
            transcriber=None,
            threadpool_dispatched_at=time.perf_counter(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one) for _ in range(2)]
        for future in futures:
            future.result()

    assert maximum_active == 1


def test_only_supported_routes_are_registered() -> None:
    routes = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }

    assert routes == {("GET", "/health"), ("POST", "/enrich")}


def test_interactive_api_documentation_is_disabled() -> None:
    assert app.openapi_url is None
    assert app.docs_url is None
    assert app.redoc_url is None


def test_single_media_uses_the_ordered_bundle_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FakeMultimodalEmbedder()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: worker)

    response = TestClient(app).post(
        "/enrich",
        data={"text": "beach day"},
        files={"files": ("beach.png", b"not-real-png", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["media_items"][0]["position"] == 0
    assert payload["media_summary"]["submitted_count"] == 1
    assert payload["media_summary"]["successful_positions"] == [0]
    assert worker.calls[0]["source_positions"] == [0]


def test_legacy_singular_file_field_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: pytest.fail("worker lookup ran"),
    )

    response = TestClient(app).post(
        "/enrich",
        files={"file": ("legacy.png", b"not-real-png", "image/png")},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("data", "files"),
    [
        # Valid text plus a legacy field: previously became a text-only 200
        # and the media vanished without provenance.
        ({"text": "beach day"}, [("file", ("clip.mp4", b"not-real-mp4", "video/mp4"))]),
        # A valid `files` upload plus a stray upload under another name.
        (
            {},
            [
                ("files", ("slide.png", b"not-real-png", "image/png")),
                ("attachment", ("extra.png", b"not-real-png", "image/png")),
            ],
        ),
        ({"text": "beach day"}, [("Files", ("clip.mp4", b"not-real-mp4", "video/mp4"))]),
    ],
)
def test_uploads_under_any_field_other_than_files_are_rejected_before_worker_lookup(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: pytest.fail("worker lookup ran"),
    )

    response = TestClient(app).post("/enrich", data=data, files=files)

    assert response.status_code == 400
    assert "files" in response.json()["detail"]


@pytest.mark.parametrize(
    "files",
    [
        [("files", (f"{index}.jpg", b"x", "image/jpeg")) for index in range(11)],
        [("files", (f"{index}.mp4", b"x", "video/mp4")) for index in range(11)],
    ],
)
def test_oversized_media_bundle_is_rejected_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    files: list[tuple[str, tuple[str, bytes, str]]],
) -> None:
    embedder = FakeMultimodalEmbedder()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: embedder)

    response = TestClient(app).post("/enrich", files=files)

    assert response.status_code == 400
    assert embedder.calls == []


def test_multi_video_post_transcribes_each_distinct_video_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = FakeMultimodalEmbedder()
    transcriber = FakeTranscriber(trusted=[True, False], transcripts=["alpha", "beta"])
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: embedder)
    monkeypatch.setattr(enrich_api, "get_transcriber", lambda _: transcriber)

    response = TestClient(app).post(
        "/enrich",
        data={"text": "two clips"},
        files=[
            ("files", ("cover.jpg", b"cover", "image/jpeg")),
            ("files", ("first.mp4", b"first", "video/mp4")),
            ("files", ("middle.jpg", b"middle", "image/jpeg")),
            ("files", ("second.mp4", b"second", "video/mp4")),
            ("files", ("first-again.mp4", b"first", "video/mp4")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_enrichment_status"] == "ok"
    assert [entry["source_position"] for entry in payload["asr_results"]] == [1, 3]
    assert [entry["asr_transcript"] for entry in payload["asr_results"]] == ["alpha", "beta"]
    assert [entry["asr_used_for_embedding"] for entry in payload["asr_results"]] == [
        True,
        False,
    ]
    assert len(transcriber.calls) == 2
    assert len(set(transcriber.calls)) == 2
    assert payload["media_items"][4]["duplicate_of_position"] == 1
    assert payload["media_summary"]["video_positions"] == [1, 3]
    assert payload["media_summary"]["contract_version"] == "post-media-v2"
    assert embedder.calls[0]["assembled_text"] == (
        "Post text: two clips\n\nTrusted speech transcript: alpha"
    )
    assert embedder.calls[0]["asr_results"] == payload["asr_results"]
    assert embedder.calls[0]["source_positions"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["ok", "ok"], "ok"),
        (["no_speech", "too_short"], "ok"),
        (["ok", "transcription_failed"], "partial"),
        (["audio_load_failed", "ok"], "partial"),
    ],
)
def test_multi_video_rollup_requires_every_video_asr_to_succeed(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[str],
    expected: str,
) -> None:
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: FakeMultimodalEmbedder())
    monkeypatch.setattr(
        enrich_api,
        "get_transcriber",
        lambda _: FakeTranscriber(statuses=statuses),
    )

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            files=[
                ("files", ("one.mp4", b"one", "video/mp4")),
                ("files", ("two.mp4", b"two", "video/mp4")),
            ],
        )
        .json()
    )

    assert payload["content_enrichment_status"] == expected
    assert [entry["status"] for entry in payload["asr_results"]] == statuses


def test_ten_video_post_is_accepted_and_fully_transcribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber = FakeTranscriber()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: FakeMultimodalEmbedder())
    monkeypatch.setattr(enrich_api, "get_transcriber", lambda _: transcriber)

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            files=[
                ("files", (f"{index}.mp4", f"v{index}".encode(), "video/mp4"))
                for index in range(10)
            ],
        )
        .json()
    )

    assert payload["content_enrichment_status"] == "ok"
    assert [entry["source_position"] for entry in payload["asr_results"]] == list(range(10))
    assert len(transcriber.calls) == 10


def test_video_asr_feeds_the_same_ordered_post_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = FakeMultimodalEmbedder()
    transcriber = FakeTranscriber()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: embedder)
    monkeypatch.setattr(enrich_api, "get_transcriber", lambda _: transcriber)

    response = TestClient(app).post(
        "/enrich",
        data={"text": "robot parade"},
        files=[
            ("files", ("one.jpg", b"one", "image/jpeg")),
            ("files", ("clip.mp4", b"video", "video/mp4")),
            ("files", ("again.jpg", b"one", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_enrichment_status"] == "ok"
    assert payload["asr_results"][0]["source_position"] == 1
    assert payload["asr_results"][0]["asr_used_for_embedding"] is True
    assert payload["media_items"][2]["duplicate_of_position"] == 0
    assert embedder.calls == [
        {
            "assembled_text": (
                "Post text: robot parade\n\nTrusted speech transcript: trusted speech"
            ),
            "asr_results": payload["asr_results"],
            "paths_exist": True,
            "source_positions": [0, 1],
        }
    ]
    assert len(transcriber.calls) == 1


def test_enrichment_model_work_is_dispatched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_active = False
    dispatch_count = 0

    class DispatchCheckingEmbedder(FakeMultimodalEmbedder):
        def embed(self, *args, **kwargs) -> MultimodalEmbeddingResult:
            assert dispatch_active
            return super().embed(*args, **kwargs)

    async def dispatch(function, /, *args, **kwargs):
        nonlocal dispatch_active, dispatch_count
        dispatch_count += 1
        dispatch_active = True
        try:
            return function(*args, **kwargs)
        finally:
            dispatch_active = False

    monkeypatch.setattr(enrich_api, "run_in_threadpool", dispatch)
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: DispatchCheckingEmbedder(status="text_only_ok"),
    )

    response = TestClient(app).post("/enrich", data={"text": "hello"})

    assert response.status_code == 200
    assert dispatch_count == 1


def test_content_length_limit_rejects_before_worker_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: pytest.fail("worker lookup ran"),
    )

    response = TestClient(app).post(
        "/enrich",
        headers={"content-length": str(enrich_api.MAX_REQUEST_BODY_BYTES + 1)},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": f"Request body too large (max {enrich_api.MAX_REQUEST_BODY_BYTES})"
    }


def test_chunked_limit_rejects_before_worker_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_api, "MAX_REQUEST_BODY_BYTES", 64)
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: pytest.fail("worker lookup ran"),
    )
    limited_app = application_api.build_application()

    def body():
        yield b"x" * 128
        yield b"x" * 128

    response = TestClient(limited_app).post(
        "/enrich",
        content=body(),
        headers={"content-type": "multipart/form-data; boundary=----x"},
    )

    assert "content-length" not in {key.lower() for key in response.request.headers}
    assert response.status_code == 413


def test_unloaded_required_worker_returns_503() -> None:
    unloaded_app = application_api.build_application()

    response = TestClient(unloaded_app).post("/enrich", data={"text": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Multimodal model not loaded yet"}


def test_video_without_available_asr_is_partial(
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
            files={"files": ("clip.mp4", b"video", "video/mp4")},
        )
        .json()
    )

    assert payload["content_enrichment_status"] == "partial"
    assert payload["asr_results"] == []
    assert payload["media_summary"]["video_positions"] == [0]


def test_untrusted_asr_is_returned_but_excluded_from_qwen_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = FakeMultimodalEmbedder()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: embedder)
    monkeypatch.setattr(enrich_api, "get_transcriber", lambda _: FakeTranscriber(trusted=False))

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            data={"text": "author text"},
            files={"files": ("clip.mp4", b"video", "video/mp4")},
        )
        .json()
    )

    assert payload["asr_results"][0]["asr_transcript"] == "trusted speech"
    assert payload["asr_results"][0]["asr_used_for_embedding"] is False
    assert embedder.calls[0]["assembled_text"] == "Post text: author text"


@pytest.mark.parametrize("status", ["text_only_ok", "inference_failed", "input_too_long"])
def test_enrichment_never_emits_retired_topic_output(monkeypatch, status):
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: FakeMultimodalEmbedder(status=status),
    )
    response = TestClient(app).post("/enrich", data={"text": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert "topic_result" not in payload
    assert "topic" not in payload["leg_timings_ms"]
    assert payload["multimodal_result"]["multimodal_embedding_status"] == status
    assert payload["content_enrichment_status"] == ("ok" if status == "text_only_ok" else "failed")


def test_context_overflow_does_not_blame_media_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich_api,
        "get_multimodal_embedder",
        lambda _: FakeMultimodalEmbedder(status="input_too_long"),
    )

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            files=[
                ("files", ("one.png", b"one", "image/png")),
                ("files", ("two.mp4", b"two", "video/mp4")),
            ],
        )
        .json()
    )

    assert payload["content_enrichment_status"] == "failed"
    assert payload["media_summary"]["status"] == "failed"
    assert payload["media_summary"]["failed_positions"] == []
    assert payload["media_summary"]["successful_positions"] == []


def test_failed_media_position_is_preserved_in_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedPositionEmbedder:
        def embed(self, *_args, **_kwargs) -> MultimodalEmbeddingResult:
            return MultimodalEmbeddingResult(
                multimodal_embedding=[],
                embedding_dim=1024,
                multimodal_embedding_model="fake-qwen",
                multimodal_embedding_model_version="fake-qwen-v1",
                multimodal_embedding_model_revision="a" * 40,
                multimodal_embedding_method_version="fake-method-v1",
                multimodal_embedding_status="media_load_failed",
                input_evidence=[{"source_position": 1, "kind": "media_load_failed"}],
            )

    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: FailedPositionEmbedder())

    payload = (
        TestClient(app)
        .post(
            "/enrich",
            files=[
                ("files", ("one.png", b"one", "image/png")),
                ("files", ("two.png", b"two", "image/png")),
            ],
        )
        .json()
    )

    assert payload["content_enrichment_status"] == "failed"
    assert payload["media_summary"]["failed_positions"] == [1]


def test_leg_timings_separate_lock_wait_video_decode_and_qwen_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedEmbedder(FakeMultimodalEmbedder):
        def embed(self, bundle, paths, post_text, asr_results, timings):
            started = time.perf_counter()
            result = super().embed(bundle, paths, post_text, asr_results, timings)
            # Use elapsed work so the inner duration fits inside the real outer timer.
            self.processing_ms = (time.perf_counter() - started) * 1000.0
            timings["qwen_processing"] = self.processing_ms
            return result

    worker = TimedEmbedder()
    monkeypatch.setattr(enrich_api, "get_multimodal_embedder", lambda _: worker)
    client = TestClient(app)
    inference_lock = threading.Lock()
    lock_requested = threading.Event()

    class ObservedLock:
        def __enter__(self):
            lock_requested.set()
            inference_lock.acquire()

        def __exit__(self, *args):
            inference_lock.release()

    monkeypatch.setattr(enrich_api, "_MODEL_INFERENCE_LOCK", ObservedLock())
    with ThreadPoolExecutor(max_workers=1) as executor:
        inference_lock.acquire()
        try:
            response = executor.submit(client.post, "/enrich", data={"text": "hello"})
            assert lock_requested.wait(timeout=5), "request did not reach the inference lock"
            time.sleep(0.2)
        finally:
            inference_lock.release()
        payload = response.result(timeout=10).json()

    timings = payload["leg_timings_ms"]
    assert timings["lock_wait"] >= 150.0
    assert timings["video_decode"] == 0.0
    assert timings["qwen_processing"] == round(worker.processing_ms, 1)
    assert 0.0 <= timings["qwen_processing"] <= timings["multimodal"]
    assert {"media_bundle", "lock_wait", "multimodal", "video_decode", "qwen_processing"} <= set(
        timings
    )
