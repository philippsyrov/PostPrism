"""Construct the enrichment service."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from postprism.api.enrich import MAX_REQUEST_BODY_BYTES, enrich_endpoint
from postprism.api.transport import (
    OversizedUploadLimitMiddleware,
    log_endpoint_timing,
)
from postprism.api.workers import health, lifespan
from postprism.branding import SERVICE_NAME

SERVICE_LOGGERS: tuple[str, ...] = ("app", "postprism")
SERVICE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_service_logging(level: int = logging.INFO) -> None:
    """Emit this service's INFO logs without raising third-party loggers to INFO.

    Uvicorn configures only its own loggers, so without this the service's INFO
    lines (endpoint timing, ASR trust decisions) are dropped. The root logger is
    left alone because Hugging Face Hub, httpx, and Transformers are noisy at
    INFO. Propagation is off and repeat calls are no-ops, so lines never double.
    """
    for name in SERVICE_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        if not any(getattr(handler, "_postprism_service", False) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(SERVICE_LOG_FORMAT))
            handler._postprism_service = True  # type: ignore[attr-defined]
            logger.addHandler(handler)


def build_application() -> FastAPI:
    """Build the service with its exact middleware and routes."""
    app = FastAPI(
        title=SERVICE_NAME,
        version="1.0",
        lifespan=lifespan,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(
        OversizedUploadLimitMiddleware,
        max_request_body_bytes=MAX_REQUEST_BODY_BYTES,
    )
    app.middleware("http")(log_endpoint_timing)

    app.post("/enrich", response_model=None)(enrich_endpoint)
    app.get("/health", response_model=None)(health)
    return app
