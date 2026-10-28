"""Frozen Qwen ASR identity and lazy-loading contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from postprism import Transcriber
from postprism.asr.qwen_policy import (
    FULL_DECODE_MAX_SECONDS,
    TIMELINE_PROBE_MAX_SECONDS,
    TIMELINE_PROBE_SECONDS,
)
from tests.transcriber_test_support import FakeQwenModel, patch_qwen_loader


def test_qwen_runtime_owner_imports_without_optional_model_stacks() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import postprism.asr.qwen_runtime

blocked = ("torch", "qwen_asr")
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)
)))
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout) == []


def test_runtime_owner_collects_and_skips_without_pyav() -> None:
    probe_source = """
import builtins
import importlib.util
import pytest

real_import = builtins.__import__
real_find_spec = importlib.util.find_spec

def blocked_import(name, *args, **kwargs):
    if name == "av" or name.startswith("av."):
        raise ModuleNotFoundError("No module named 'av'")
    return real_import(name, *args, **kwargs)

def blocked_find_spec(name, *args, **kwargs):
    if name == "av":
        return None
    return real_find_spec(name, *args, **kwargs)

builtins.__import__ = blocked_import
importlib.util.find_spec = blocked_find_spec
raise SystemExit(pytest.main(ARGS))
"""
    root = Path(__file__).resolve().parents[1]
    collect = subprocess.run(
        [
            sys.executable,
            "-c",
            "ARGS = ['--collect-only', '-q', 'tests/test_transcriber_runtime.py']\n" + probe_source,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    selected = subprocess.run(
        [
            sys.executable,
            "-c",
            "ARGS = ['-q', 'tests/test_transcriber_runtime.py::"
            "TestQwenGatedPregate::test_silence_skips_qwen_and_returns_no_speech']\n"
            + probe_source,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert collect.returncode == 0, collect.stderr
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert selected.stdout.lstrip().startswith("s")


def test_transcriber_uses_the_frozen_service_policy() -> None:
    transcriber = Transcriber()

    assert transcriber.model_name == "Qwen/Qwen3-ASR-0.6B"
    assert transcriber.model_revision == "bda30b99af3446a954339e28465acf530464e52e"
    assert transcriber.device == "cuda:0"
    assert transcriber.compute_type == "bfloat16"
    assert transcriber.max_audio_seconds == 30.0
    # Media up to this length decodes fully (unchanged path); longer media is
    # sampled at its beginning, middle, and end with bounded seek-decoded probes.
    assert FULL_DECODE_MAX_SECONDS == 600.0
    assert TIMELINE_PROBE_SECONDS == 60.0
    assert TIMELINE_PROBE_MAX_SECONDS == 120.0
    assert transcriber.audio_classifier is not None
    assert transcriber.model_version == ("qwen3-asr-0.6b-bfloat16-cuda0-vadgate2-astgate1-qwen-v5")


def test_qwen_warmup_loads_the_frozen_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    model = FakeQwenModel()

    def fake_loader(
        model_name: str,
        device: str,
        compute_type: str,
        max_new_tokens: int,
        max_batch_size: int,
        model_revision: str,
    ) -> FakeQwenModel:
        calls.append(
            (
                model_name,
                device,
                compute_type,
                max_new_tokens,
                max_batch_size,
                model_revision,
            )
        )
        return model

    monkeypatch.setattr("postprism.asr.transcriber._load_qwen_asr_model", fake_loader)
    monkeypatch.setattr(
        "postprism.asr.transcriber.ensure_audio_prep_available",
        lambda: None,
    )
    monkeypatch.setattr(
        "postprism.asr.audio_class.AudioClassifier.warmup",
        lambda _: None,
    )
    transcriber = Transcriber()

    transcriber.warmup()
    transcriber.warmup()

    assert transcriber.is_warm is True
    assert calls == [
        (
            "Qwen/Qwen3-ASR-0.6B",
            "cuda:0",
            "bfloat16",
            512,
            8,
            "bda30b99af3446a954339e28465acf530464e52e",
        )
    ]


def test_qwen_warmup_fails_loudly_when_audio_prep_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_qwen_loader(monkeypatch, FakeQwenModel())

    def raise_missing() -> None:
        raise RuntimeError("PyAV missing")

    monkeypatch.setattr(
        "postprism.asr.transcriber.ensure_audio_prep_available",
        raise_missing,
    )
    transcriber = Transcriber()

    with pytest.raises(RuntimeError, match="PyAV missing"):
        transcriber.warmup()

    assert transcriber.is_warm is False
