"""Worker registry, lifecycle, getters, and health rendering."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from postprism import MultimodalEmbedder, Transcriber
from postprism.captioning.worker import Captioner
from postprism.multimodal.identity import (
    DEFAULT_MODEL_NAME as MULTIMODAL_MODEL_NAME,
)
from postprism.multimodal.identity import (
    DEFAULT_MODEL_REVISION as MULTIMODAL_MODEL_REVISION,
)
from postprism.startup_checks import (
    RuntimeProfile,
    installed_runtime_versions,
    validate_runtime_configuration,
)
from postprism.visual_tags.worker import VisualTagger

logger = logging.getLogger("app")


@dataclass(frozen=True)
class WorkerSpec:
    attr: str
    worker_type: type
    not_ready_detail: str
    env_flag: str | None = None
    label: str = ""
    model_env: str | None = None
    build: Callable[[str | None], object] | None = None
    version: Callable[[object], str] | None = None


TRANSCRIBER = WorkerSpec(
    attr="transcriber",
    worker_type=Transcriber,
    not_ready_detail="ASR not enabled (set USE_ASR=true)",
    env_flag="USE_ASR",
    label="Transcriber",
    build=lambda _: Transcriber(),
    version=lambda worker: worker.model_version,
)

MULTIMODAL_EMBEDDER = WorkerSpec(
    attr="multimodal_embedder",
    worker_type=MultimodalEmbedder,
    not_ready_detail="Multimodal model not loaded yet",
    label="Multimodal embedder",
    version=lambda worker: worker.model_version,
)

VISUAL_TAGGER = WorkerSpec(
    attr="visual_tagger",
    worker_type=VisualTagger,
    not_ready_detail="Visual tags not enabled (set USE_VISUAL_TAGS=true)",
    env_flag="USE_VISUAL_TAGS",
    label="Visual tagger",
    build=lambda _: VisualTagger(),
    version=lambda worker: worker.model_version,
)

CAPTIONER = WorkerSpec(
    attr="captioner",
    worker_type=Captioner,
    not_ready_detail="Captions not enabled (set USE_CAPTIONS=true)",
    env_flag="USE_CAPTIONS",
    label="Captioner",
    build=lambda _: Captioner(),
    version=lambda worker: worker.model_version,
)

OPTIONAL_WORKERS: tuple[WorkerSpec, ...] = (
    TRANSCRIBER,
    VISUAL_TAGGER,
    CAPTIONER,
)


def env_flag(
    name: str,
    default: bool = False,
    *,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Read a boolean environment flag."""
    value = environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def warmup_or_bust(worker: Any, env_flag_name: str, label: str) -> None:
    """Warm an optional worker, turning missing extras into a loud failure."""
    try:
        worker.warmup()
    except ImportError as err:
        message = (
            f"{label} requested but optional deps missing (%s). "
            f"Install requirements.txt or unset {env_flag_name}."
        )
        logger.error(message, err)
        raise RuntimeError(message % err) from err
    except Exception:
        logger.exception("%s warmup failed", label)
        raise


def maybe_build(
    spec: WorkerSpec,
    *,
    flag_reader: Callable[[str], bool],
    warmup: Callable[[Any, str, str], None],
    environ: Mapping[str, str] = os.environ,
) -> object | None:
    """Construct and warm one optional worker when its env flag is set."""
    if spec.env_flag is None or spec.build is None or spec.version is None:
        return None
    if not flag_reader(spec.env_flag):
        return None

    model_name = environ.get(spec.model_env) if spec.model_env else None
    logger.info(
        "%s=true; loading %s: %s",
        spec.env_flag,
        spec.label.lower(),
        model_name or "default",
    )
    worker = spec.build(model_name)
    warmup(worker, spec.env_flag, spec.label)
    logger.info("%s ready: %s", spec.label, spec.version(worker))
    return worker


def initialize_worker_state(
    app: FastAPI,
    *,
    required_worker: WorkerSpec,
    build_required: Callable[[WorkerSpec], object],
    build_optional: Callable[[WorkerSpec], object | None],
    optional_workers: Sequence[WorkerSpec],
) -> None:
    """Construct and warm startup workers outside the event loop."""
    logger.info("Loading required worker: %s", required_worker.label)
    worker = build_required(required_worker)
    worker.warmup()
    setattr(app.state, required_worker.attr, worker)
    model_name = getattr(worker, "model_name", getattr(worker, "model_id", "unknown"))
    logger.info("Required worker ready: %s", model_name)

    for spec in optional_workers:
        setattr(app.state, spec.attr, build_optional(spec))


def require_worker(request: Request, spec: WorkerSpec) -> object:
    """Return a loaded worker from app state, or raise its stable 503."""
    worker = getattr(request.app.state, spec.attr, None)
    if not isinstance(worker, spec.worker_type):
        raise HTTPException(status_code=503, detail=spec.not_ready_detail)
    return worker


def optional_worker(
    get_worker: Callable[[Request], object],
    request: Request,
) -> object | None:
    """Fetch an optional worker, or ``None`` when its 503 gate says disabled."""
    try:
        return get_worker(request)
    except HTTPException as err:
        if err.status_code == 503:
            return None
        raise


def worker_health(spec: WorkerSpec, worker: object | None) -> dict[str, object]:
    """Render one optional worker's stable health state."""
    if not isinstance(worker, spec.worker_type) or spec.version is None:
        return {"enabled": False}
    if spec.attr in {"visual_tagger", "captioner"}:
        return {"enabled": True, "available": True, **worker.identity.health_identity()}
    status: dict[str, object] = {
        "enabled": True,
        "model": getattr(
            worker,
            "model_name",
            getattr(worker, "model_id", "unknown"),
        ),
        "model_version": spec.version(worker),
    }
    if hasattr(worker, "model_revision"):
        status["model_revision"] = worker.model_revision
    return status


def health_response(
    request: Request,
    *,
    required_worker: WorkerSpec,
    optional_workers: Sequence[WorkerSpec],
) -> dict[str, object] | JSONResponse:
    """Render required retrieval readiness and optional-worker health blocks."""
    worker = getattr(request.app.state, required_worker.attr, None)
    if not isinstance(worker, required_worker.worker_type):
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model": "not loaded"},
        )

    model_name = getattr(worker, "model_name", getattr(worker, "model_id", "unknown"))
    model_version = (
        required_worker.version(worker)
        if required_worker.version is not None
        else getattr(worker, "model_version", "unknown")
    )
    model_revision = getattr(worker, "model_revision", "unpinned")
    profile = getattr(request.app.state, "runtime_profile", None)
    payload: dict[str, object] = {
        "status": "ok",
        "service_commit": os.environ.get("SERVICE_COMMIT", "unknown"),
        "runtime_profile": getattr(profile, "name", "unknown"),
        "multimodal": {
            "model": model_name,
            "model_version": model_version,
            "model_revision": model_revision,
            "method_version": worker.method_version,
            "dtype": worker.dtype,
            "embedding_dim": worker.output_dim,
        },
    }
    health_names = {
        "transcriber": "asr",
        "visual_tagger": "visual_tags",
        "captioner": "captions",
    }
    for spec in optional_workers:
        worker = getattr(request.app.state, spec.attr, None)
        payload[health_names.get(spec.attr, spec.attr)] = worker_health(spec, worker)
    return payload


def build_multimodal_embedder(profile: RuntimeProfile) -> MultimodalEmbedder:
    return MultimodalEmbedder(
        model_name=MULTIMODAL_MODEL_NAME,
        model_revision=MULTIMODAL_MODEL_REVISION,
        device=profile.device,
        dtype=profile.dtype,
    )


def build_optional(spec: WorkerSpec) -> object | None:
    return maybe_build(
        spec,
        flag_reader=env_flag,
        warmup=warmup_or_bust,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(
        initialize_service_state,
        app,
    )
    yield


def initialize_service_state(app: FastAPI) -> None:
    """Validate the measured runtime, then construct all enabled workers."""
    profile = validate_runtime_configuration(os.environ, installed_runtime_versions())
    app.state.runtime_profile = profile
    initialize_worker_state(
        app,
        required_worker=MULTIMODAL_EMBEDDER,
        build_required=lambda _: build_multimodal_embedder(profile),
        build_optional=build_optional,
        optional_workers=OPTIONAL_WORKERS,
    )


def get_multimodal_embedder(request: Request) -> MultimodalEmbedder:
    return require_worker(request, MULTIMODAL_EMBEDDER)


def get_transcriber(request: Request) -> Transcriber:
    return require_worker(request, TRANSCRIBER)


def get_visual_tagger(request: Request) -> VisualTagger:
    return require_worker(request, VISUAL_TAGGER)


def get_captioner(request: Request) -> Captioner:
    return require_worker(request, CAPTIONER)


def health(request: Request) -> dict[str, object] | JSONResponse:
    return health_response(
        request,
        required_worker=MULTIMODAL_EMBEDDER,
        optional_workers=OPTIONAL_WORKERS,
    )
