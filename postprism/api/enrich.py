"""The single HTTP and synchronous execution path for post enrichment."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from postprism.api.transport import (
    normalized_optional_text,
    read_validated_upload,
    read_validated_uploads,
    reject_uploads_outside_files_field,
    require_text_or_files,
)
from postprism.api.workers import (
    get_captioner,
    get_multimodal_embedder,
    get_transcriber,
    get_visual_tagger,
    optional_worker,
)
from postprism.media import (
    MAX_POST_MEDIA_BYTES,
    PostMediaBundle,
    PostMediaContractError,
    UploadPayload,
    build_post_media_bundle,
    materialized_post_media,
)
from postprism.schemas import (
    MULTIMODAL_EMBEDDING_SUCCESS_STATUSES,
    TRANSCRIPTION_SUCCESS_STATUSES,
)

MAX_REQUEST_BODY_BYTES = MAX_POST_MEDIA_BYTES
MAX_ENCODE_TEXT_CHARS = 64 * 1024
_MODEL_INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class EnrichmentExecution:
    payload: dict[str, object]
    temp_file_write_ms: float
    threadpool_dispatch_to_sync_start_ms: float
    sync_start_to_first_ml_leg_ms: float


def rollup_enrichment_status(
    multimodal_result: Mapping[str, object],
    *,
    video_count: int,
    asr_results: Sequence[Mapping[str, object]],
) -> Literal["ok", "partial", "failed"]:
    """The vector decides usability; ASR must succeed for every distinct video."""
    qwen_ok = (
        str(multimodal_result.get("multimodal_embedding_status", ""))
        in MULTIMODAL_EMBEDDING_SUCCESS_STATUSES
    )
    if not qwen_ok:
        return "failed"
    if video_count == 0:
        return "ok"
    asr_ok = len(asr_results) == video_count and all(
        str(asr_result.get("status", "")) in TRANSCRIPTION_SUCCESS_STATUSES
        for asr_result in asr_results
    )
    return "ok" if asr_ok else "partial"


def _timed(
    timings: dict[str, float],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        timings[name] = round((time.perf_counter() - started) * 1000.0, 1)


def _media_provenance(
    bundle: PostMediaBundle,
    multimodal_result: Mapping[str, object],
    video_positions: tuple[int, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    media_items = [
        {
            "position": item.position,
            "filename": item.filename,
            "sha256": item.sha256,
            "duplicate_of_position": item.duplicate_of_position,
        }
        for item in bundle.items
    ]
    qwen_ok = (
        str(multimodal_result.get("multimodal_embedding_status", ""))
        in MULTIMODAL_EMBEDDING_SUCCESS_STATUSES
    )
    reported_failed_positions = sorted(
        {
            int(evidence["source_position"])
            for evidence in multimodal_result.get("input_evidence", [])
            if isinstance(evidence, Mapping)
            and evidence.get("kind") == "media_load_failed"
            and isinstance(evidence.get("source_position"), int)
        }
    )
    # Only an undecodable item names media positions; context overflow or a
    # model fault is a whole-post failure the backend cannot repair per item.
    media_failed = str(multimodal_result.get("multimodal_embedding_status")) == "media_load_failed"
    failed_positions = (
        (reported_failed_positions or list(bundle.distinct_positions)) if media_failed else []
    )
    summary: dict[str, object] = {
        "contract_version": bundle.contract_version,
        "submitted_count": len(bundle.items),
        "distinct_count": len(bundle.distinct_positions),
        "status": "ok" if qwen_ok else "failed",
        "video_positions": list(video_positions),
        "successful_positions": list(bundle.distinct_positions) if qwen_ok else [],
        "failed_positions": failed_positions,
        "duplicate_map": {
            str(item.position): item.duplicate_of_position
            for item in bundle.items
            if item.duplicate_of_position is not None
        },
    }
    return media_items, summary


def run_enrichment(
    *,
    uploads: tuple[UploadPayload, ...],
    text: str | None,
    multimodal_embedder: Any,
    transcriber: Any | None,
    threadpool_dispatched_at: float,
    visual_tagger: Any | None = None,
    captioner: Any | None = None,
) -> EnrichmentExecution:
    """Run one post synchronously; FastAPI dispatches this through a threadpool."""
    timings: dict[str, float] = {}
    sync_started = time.perf_counter()
    dispatch_to_sync_ms = round(
        (sync_started - threadpool_dispatched_at) * 1000.0,
        1,
    )
    bundle = build_post_media_bundle(uploads) if uploads else None
    timings["media_bundle"] = round((time.perf_counter() - sync_started) * 1000.0, 1)
    materialization_started = time.perf_counter()
    materialized = materialized_post_media(bundle) if bundle is not None else nullcontext({})

    with materialized as paths:
        temp_file_write_ms = round(
            (time.perf_counter() - materialization_started) * 1000.0,
            1,
        )
        video_positions = (
            tuple(
                position
                for position in bundle.distinct_positions
                if bundle.items[position].upload.media_type == "video"
            )
            if bundle is not None
            else ()
        )
        asr_results: list[dict[str, object]] = []

        def transcribe_each_video() -> None:
            for position in video_positions:
                result = transcriber.transcribe(str(paths[position])).to_dict()
                asr_results.append({"source_position": position, **result})

        lock_requested = time.perf_counter()
        with _MODEL_INFERENCE_LOCK:
            timings["lock_wait"] = round((time.perf_counter() - lock_requested) * 1000.0, 1)
            first_ml_ms = round((time.perf_counter() - sync_started) * 1000.0, 1)
            if video_positions and transcriber is not None:
                _timed(timings, "asr", transcribe_each_video)
            embed_timings: dict[str, float] = {}
            multimodal_result = _timed(
                timings,
                "multimodal",
                lambda: multimodal_embedder.embed(
                    bundle,
                    paths,
                    text,
                    asr_results,
                    embed_timings,
                ).to_dict(),
            )
            # Decode and model time are reported beside the `multimodal` total
            # so a slow post can be attributed to frames, the model, or the lock.
            for key in ("video_decode", "qwen_processing"):
                if key in embed_timings:
                    timings[key] = round(embed_timings[key], 1)
            visual_tag_result = (
                _timed(
                    timings, "visual_tags", lambda: visual_tagger.enrich(bundle, paths).to_dict()
                )
                if visual_tagger is not None
                else None
            )
            caption_result = (
                _timed(timings, "captions", lambda: captioner.enrich(bundle, paths).to_dict())
                if captioner is not None
                else None
            )

    payload: dict[str, object] = {
        "content_enrichment_status": rollup_enrichment_status(
            multimodal_result,
            video_count=len(video_positions),
            asr_results=asr_results,
        ),
        "multimodal_result": multimodal_result,
        "asr_results": asr_results,
        "visual_tag_result": visual_tag_result,
        "caption_result": caption_result,
        "leg_timings_ms": timings,
    }
    if bundle is not None:
        media_items, media_summary = _media_provenance(
            bundle,
            multimodal_result,
            video_positions,
        )
        payload["media_items"] = media_items
        payload["media_summary"] = media_summary
    return EnrichmentExecution(
        payload,
        temp_file_write_ms,
        dispatch_to_sync_ms,
        first_ml_ms,
    )


async def _read_uploads(
    files: list[UploadFile],
    timing: dict[str, float],
) -> tuple[UploadPayload, ...] | JSONResponse:
    return await read_validated_uploads(
        files,
        timing,
        max_request_body_bytes=MAX_REQUEST_BODY_BYTES,
        read_one=read_validated_upload,
    )


async def enrich_endpoint(
    request: Request,
    files: list[UploadFile] | None = File(default=None),
    text: str | None = Form(default=None, max_length=MAX_ENCODE_TEXT_CHARS),
) -> dict[str, object] | JSONResponse:
    incoming_files = list(files or ())
    await reject_uploads_outside_files_field(request)
    require_text_or_files(incoming_files, text)
    endpoint_entered = time.perf_counter()
    request_received = getattr(request.state, "request_received_at", endpoint_entered)
    request_timings: dict[str, float] = {
        "request_receipt_to_endpoint_entry_ms": round(
            (endpoint_entered - request_received) * 1000.0,
            1,
        )
    }
    uploads_or_error = await _read_uploads(incoming_files, request_timings)
    if isinstance(uploads_or_error, JSONResponse):
        return uploads_or_error
    setup_started = time.perf_counter()
    transcriber = (
        optional_worker(get_transcriber, request)
        if any(upload.media_type == "video" for upload in uploads_or_error)
        else None
    )
    multimodal_embedder = get_multimodal_embedder(request)
    visual_tagger = optional_worker(get_visual_tagger, request)
    captioner = optional_worker(get_captioner, request)
    request_timings["endpoint_pre_threadpool_setup_ms"] = round(
        (time.perf_counter() - setup_started) * 1000.0,
        1,
    )

    dispatched = time.perf_counter()
    try:
        execution = await run_in_threadpool(
            run_enrichment,
            uploads=uploads_or_error,
            text=normalized_optional_text(text),
            multimodal_embedder=multimodal_embedder,
            transcriber=transcriber,
            threadpool_dispatched_at=dispatched,
            visual_tagger=visual_tagger,
            captioner=captioner,
        )
    except PostMediaContractError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})
    request_timings["threadpool_dispatch_to_sync_start_ms"] = (
        execution.threadpool_dispatch_to_sync_start_ms
    )
    request_timings["temp_file_write_ms"] = execution.temp_file_write_ms
    request_timings["sync_start_to_first_ml_leg_ms"] = execution.sync_start_to_first_ml_leg_ms
    request_timings["request_receipt_to_first_ml_leg_ms"] = round(
        (dispatched - request_received) * 1000.0
        + request_timings["threadpool_dispatch_to_sync_start_ms"]
        + execution.sync_start_to_first_ml_leg_ms,
        1,
    )
    execution.payload["request_timings_ms"] = request_timings
    return execution.payload
