"""HTTP middleware, multipart validation, and request-boundary helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from postprism.media import (
    UploadPayload,
    UploadValidationError,
    read_upload_content,
    validate_upload,
)

logger = logging.getLogger("app")

OVERSIZED_UPLOAD_ROUTES: frozenset[str] = frozenset({"/enrich"})
TIMED_ROUTES: frozenset[str] = frozenset({"/enrich"})


def request_body_too_large(
    content_length: str | None,
    *,
    max_request_body_bytes: int,
) -> bool:
    """Check a request Content-Length header against the upload cap."""
    if content_length is None:
        return False
    try:
        return int(content_length) > max_request_body_bytes
    except ValueError:
        return False


def oversized_body_response(max_request_body_bytes: int) -> JSONResponse:
    """Stable 413 payload shared by header, stream, and handler-level caps."""
    return JSONResponse(
        status_code=413,
        content={"error": f"Request body too large (max {max_request_body_bytes})"},
    )


def _content_length_from_scope(scope: Scope) -> str | None:
    for key, value in scope.get("headers") or []:
        if key == b"content-length":
            return value.decode("latin-1")
    return None


class OversizedUploadLimitMiddleware:
    """Count streamed POST bodies on media-upload routes and fail closed at 413.

    Content-Length is rejected immediately. Chunked uploads omit that header,
    so the receive wrapper counts bytes and stops before the full body is
    consumed or a worker is looked up.

    Oversized chunks are not raised into FastAPI: a generic receive exception is
    rewritten as 400 ``error parsing the body``. Disconnect the stream instead
    and replace the inner response with the stable 413 JSON.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_request_body_bytes: int | Callable[[], int],
        routes: frozenset[str] = OVERSIZED_UPLOAD_ROUTES,
    ) -> None:
        self.app = app
        self._max_request_body_bytes = max_request_body_bytes
        self.routes = routes

    def _limit(self) -> int:
        max_bytes = self._max_request_body_bytes
        return max_bytes() if callable(max_bytes) else max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self.routes
        ):
            await self.app(scope, receive, send)
            return

        max_bytes = self._limit()
        if request_body_too_large(
            _content_length_from_scope(scope),
            max_request_body_bytes=max_bytes,
        ):
            await oversized_body_response(max_bytes)(scope, receive, send)
            return

        received = 0
        too_large = False

        async def receive_with_limit() -> Message:
            nonlocal received, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b"") or b"")
                if received > max_bytes:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        async def send_with_limit(message: Message) -> None:
            if too_large:
                return
            await send(message)

        try:
            await self.app(scope, receive_with_limit, send_with_limit)
        except Exception:
            if not too_large:
                raise
        if too_large:
            await oversized_body_response(max_bytes)(scope, receive, send)


async def log_endpoint_timing(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log latency for model endpoints and mark request receipt time."""
    start = time.perf_counter()
    request.state.request_received_at = start
    response = await call_next(request)
    if request.url.path in TIMED_ROUTES:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "endpoint timing: method=%s path=%s status=%d elapsed_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


async def read_validated_upload(
    file: UploadFile,
    timing: dict[str, float] | None = None,
) -> UploadPayload | JSONResponse:
    """Read and validate one upload, returning the stable 400 response."""
    try:
        read_started = time.perf_counter()
        content = await read_upload_content(file)
        if timing is not None:
            timing["upload_spooled_file_read_ms"] = round(
                (time.perf_counter() - read_started) * 1000.0,
                1,
            )
        validation_started = time.perf_counter()
        upload = validate_upload(file.filename, content)
        if timing is not None:
            timing["upload_validation_ms"] = round(
                (time.perf_counter() - validation_started) * 1000.0,
                1,
            )
        return upload
    except UploadValidationError as err:
        return JSONResponse(status_code=400, content=err.to_dict())


async def read_validated_uploads(
    files: Sequence[UploadFile],
    timing: dict[str, float],
    *,
    max_request_body_bytes: int,
    read_one: Callable[
        [UploadFile, dict[str, float] | None],
        Awaitable[UploadPayload | JSONResponse],
    ],
) -> tuple[UploadPayload, ...] | JSONResponse:
    """Read an ordered upload sequence fully before any worker lookup."""
    validated: list[UploadPayload] = []
    total_bytes = 0
    read_ms = 0.0
    validation_ms = 0.0
    for file in files:
        item_timing: dict[str, float] = {}
        result = await read_one(file, item_timing)
        if isinstance(result, JSONResponse):
            return result
        read_ms += item_timing.get("upload_spooled_file_read_ms", 0.0)
        validation_ms += item_timing.get("upload_validation_ms", 0.0)
        total_bytes += len(result.content)
        if total_bytes > max_request_body_bytes:
            return oversized_body_response(max_request_body_bytes)
        validated.append(result)

    if validated:
        timing["upload_spooled_file_read_ms"] = round(read_ms, 1)
        timing["upload_validation_ms"] = round(validation_ms, 1)
    return tuple(validated)


def normalized_optional_text(text: str | None) -> str | None:
    """Strip optional form text once and collapse blank text to ``None``."""
    stripped = text.strip() if text else ""
    return stripped or None


async def reject_uploads_outside_files_field(request: Request) -> None:
    """Refuse uploads under any multipart field other than `files`.

    FastAPI binds only the declared `files` field, so a legacy `file=` or a
    misspelt field would otherwise be dropped silently and the post embedded
    without that media. Starlette caches the parsed form, so this reads no
    body twice.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        return
    form = await request.form()
    foreign = sorted(
        {
            name
            for name, value in form.multi_items()
            if name != "files" and hasattr(value, "filename")
        }
    )
    if foreign:
        raise HTTPException(
            status_code=400,
            detail="Media is accepted only under the repeatable 'files' field; "
            f"unexpected upload fields: {', '.join(foreign)}",
        )


def require_text_or_files(files: Sequence[UploadFile], text: str | None) -> None:
    """Reject a post carrying neither submitted media nor real text."""
    if not files and normalized_optional_text(text) is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one file or text field",
        )
