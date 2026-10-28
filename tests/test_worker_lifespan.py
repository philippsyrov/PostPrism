"""Guards startup worker loading against blocking the FastAPI event loop."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from postprism.api import workers
from postprism.captioning.identity import CAPTION_IDENTITY
from postprism.captioning.worker import Captioner
from postprism.startup_checks import RuntimeProfile
from postprism.visual_tags.identity import RAM_IDENTITY
from postprism.visual_tags.worker import VisualTagger
from tests.test_startup_checks import approved_environment, approved_versions


def test_service_lifespan_validates_runtime_before_worker_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dispatch_active = False
    profile = RuntimeProfile(
        name="lab-mps",
        production=False,
        device="mps:0",
        dtype="bfloat16",
        asr_enabled=False,
    )

    class FakeEmbedder:
        model_name = "fake-qwen"

        def warmup(self) -> None:
            assert dispatch_active
            events.append("warmup")

    monkeypatch.setattr(
        workers,
        "installed_runtime_versions",
        lambda: {"runtime": "versions"},
        raising=False,
    )

    def validate(environ, versions):
        assert dispatch_active
        assert versions == {"runtime": "versions"}
        events.append("validate")
        return profile

    monkeypatch.setattr(
        workers,
        "validate_runtime_configuration",
        validate,
        raising=False,
    )

    def build_embedder(received: RuntimeProfile) -> FakeEmbedder:
        assert dispatch_active
        events.append(f"build:{received.device}")
        return FakeEmbedder()

    monkeypatch.setattr(workers, "build_multimodal_embedder", build_embedder)

    async def dispatch(function, /, *args, **kwargs):
        nonlocal dispatch_active
        events.append("dispatch")
        dispatch_active = True
        try:
            return function(*args, **kwargs)
        finally:
            dispatch_active = False

    monkeypatch.setattr(workers, "run_in_threadpool", dispatch)

    async def run() -> None:
        application = FastAPI()
        async with workers.lifespan(application):
            assert application.state.multimodal_embedder.model_name == "fake-qwen"

    asyncio.run(run())

    assert events == ["dispatch", "validate", "build:mps:0", "warmup"]


def test_health_exposes_one_identity_block_and_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RequiredWorker:
        model_name = "qwen"
        model_version = "qwen-v1"
        model_revision = "a" * 40
        method_version = "qwen-method-v1"
        dtype = "bfloat16"
        output_dim = 1024

    class RequiredSpec:
        attr = "multimodal_embedder"
        worker_type = RequiredWorker
        version = staticmethod(lambda worker: worker.model_version)

    class OptionalWorker:
        pass

    class OptionalSpec:
        attr = "transcriber"
        worker_type = OptionalWorker
        version = staticmethod(lambda worker: "asr-v1")

    application = FastAPI()
    application.state.multimodal_embedder = RequiredWorker()
    application.state.transcriber = None
    application.state.runtime_profile = RuntimeProfile(
        name="lab-mps",
        production=False,
        device="mps:0",
        dtype="bfloat16",
        asr_enabled=False,
    )
    request = Request({"type": "http", "app": application})
    monkeypatch.setenv("SERVICE_COMMIT", "abc123")

    payload = workers.health_response(
        request,
        required_worker=RequiredSpec(),
        optional_workers=[OptionalSpec()],
    )

    assert payload == {
        "status": "ok",
        "service_commit": "abc123",
        "runtime_profile": "lab-mps",
        "multimodal": {
            "model": "qwen",
            "model_version": "qwen-v1",
            "model_revision": "a" * 40,
            "method_version": "qwen-method-v1",
            "dtype": "bfloat16",
            "embedding_dim": 1024,
        },
        "asr": {"enabled": False},
    }


@pytest.fixture
def visual_startup(monkeypatch):
    for flag in (
        "USE_ASR",
        "USE_TOPICS",
        "TOPIC_DEFINITIONS_FILE",
        "USE_VISUAL_TAGS",
        "USE_CAPTIONS",
    ):
        monkeypatch.delenv(flag, raising=False)
    for key, value in approved_environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(workers, "installed_runtime_versions", approved_versions)
    monkeypatch.setattr(workers.MultimodalEmbedder, "warmup", lambda self: None)
    return FastAPI()


@pytest.mark.parametrize("key", ["USE_TOPICS", "TOPIC_DEFINITIONS_FILE"])
@pytest.mark.parametrize("value", ["", "false", "true"])
def test_retired_topic_environment_fails_before_worker_construction(
    monkeypatch, visual_startup, key, value
):
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        workers,
        "build_multimodal_embedder",
        lambda _: pytest.fail("worker construction ran before retired configuration rejection"),
    )
    with pytest.raises(
        RuntimeError, match="semantic topics were removed; remove retired variables"
    ):
        workers.initialize_service_state(visual_startup)


@pytest.mark.parametrize("enabled", [False, True])
def test_service_health_and_state_have_no_topics(monkeypatch, visual_startup, enabled):
    if enabled:
        monkeypatch.setenv("USE_VISUAL_TAGS", "true")
        monkeypatch.setenv("USE_CAPTIONS", "true")
        monkeypatch.setattr(VisualTagger, "warmup", lambda self: None)
        monkeypatch.setattr(Captioner, "warmup", lambda self: None)
    workers.initialize_service_state(visual_startup)
    payload = workers.health(Request({"type": "http", "app": visual_startup}))
    assert set(payload) == {
        "status",
        "service_commit",
        "runtime_profile",
        "multimodal",
        "asr",
        "visual_tags",
        "captions",
    }
    assert not hasattr(visual_startup.state, "topic_matcher")


@pytest.mark.parametrize(
    "flag,attr,worker_type",
    [
        ("USE_VISUAL_TAGS", "visual_tagger", VisualTagger),
        ("USE_CAPTIONS", "captioner", Captioner),
    ],
)
def test_enabled_visual_worker_warms_once_at_startup(
    monkeypatch, visual_startup, flag, attr, worker_type
):
    warmups = []
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(worker_type, "warmup", lambda self: warmups.append(self))
    workers.initialize_service_state(visual_startup)
    worker = getattr(visual_startup.state, attr, None)
    assert isinstance(worker, worker_type)
    assert warmups == [worker]
    request = Request({"type": "http", "app": visual_startup})
    getter = getattr(workers, "get_" + attr)
    assert getter(request) is worker
    assert getter(request) is worker
    assert warmups == [worker]


@pytest.mark.parametrize(
    "flag,worker_type",
    [("USE_VISUAL_TAGS", VisualTagger), ("USE_CAPTIONS", Captioner)],
)
@pytest.mark.parametrize("failure", [ImportError("missing backend"), ValueError("integrity")])
def test_enabled_visual_worker_startup_failure_is_fatal(
    monkeypatch, visual_startup, flag, worker_type, failure
):
    monkeypatch.setenv(flag, "true")

    def fail(self):
        raise failure

    monkeypatch.setattr(worker_type, "warmup", fail)
    expected = RuntimeError if isinstance(failure, ImportError) else ValueError
    with pytest.raises(expected, match="missing backend|integrity"):
        workers.initialize_service_state(visual_startup)


def test_disabled_visual_workers_are_not_constructed(monkeypatch, visual_startup):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("disabled worker was constructed")

    monkeypatch.setattr(VisualTagger, "__init__", forbidden)
    monkeypatch.setattr(Captioner, "__init__", forbidden)
    workers.initialize_service_state(visual_startup)
    request = Request({"type": "http", "app": visual_startup})
    payload = workers.health(request)
    assert payload.get("visual_tags") == {"enabled": False}
    assert payload.get("captions") == {"enabled": False}


def test_importing_service_does_not_import_optional_model_backends():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'transformers', 'ram'}:
        raise AssertionError('unexpected model backend import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from postprism.api.application import build_application
build_application()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_visual_worker_health_contains_only_readiness_and_identity(visual_startup):
    workers.initialize_service_state(visual_startup)
    visual_startup.state.visual_tagger = VisualTagger(SimpleNamespace(identity=RAM_IDENTITY))
    visual_startup.state.captioner = Captioner(SimpleNamespace(identity=CAPTION_IDENTITY))
    payload = workers.health(Request({"type": "http", "app": visual_startup}))
    assert payload.get("visual_tags") == {
        "enabled": True,
        "available": True,
        **RAM_IDENTITY.health_identity(),
    }
    assert payload.get("captions") == {
        "enabled": True,
        "available": True,
        **CAPTION_IDENTITY.health_identity(),
    }
