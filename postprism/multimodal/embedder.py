"""Synchronous canonical Qwen multimodal embedding worker."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from postprism.media import PostMediaBundle
from postprism.multimodal.identity import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_METHOD_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_OUTPUT_DIM,
)
from postprism.multimodal.input import assemble_multimodal_text
from postprism.multimodal.sampling import (
    PreparedVisual,
    VisualPreparationError,
    prepare_ordered_visuals,
)
from postprism.schemas import MultimodalEmbeddingResult

logger = logging.getLogger(__name__)

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTEXT_LIMIT_ERROR_PREFIX = "multimodal input exceeds Qwen context limit:"


class EmbeddingAdapter(Protocol):
    def process(
        self,
        text: str,
        visuals: tuple[PreparedVisual, ...],
    ) -> tuple[np.ndarray, bool]: ...


def normalized_mrl_prefix(
    vector: np.ndarray,
    size: int = DEFAULT_OUTPUT_DIM,
) -> np.ndarray:
    """Take the frozen MRL prefix and normalize it as one retrieval vector."""
    prefix = np.asarray(vector, dtype=np.float32).reshape(-1)[:size]
    norm = float(np.linalg.norm(prefix))
    if (
        prefix.shape != (size,)
        or not np.isfinite(prefix).all()
        or not np.isfinite(norm)
        or norm == 0.0
    ):
        raise ValueError("invalid native multimodal embedding")
    return np.asarray(prefix / norm, dtype=np.float32)


def _model_version(model_name: str, dtype: str, output_dim: int) -> str:
    if (
        model_name == DEFAULT_MODEL_NAME
        and dtype == DEFAULT_DTYPE
        and output_dim == DEFAULT_OUTPUT_DIM
    ):
        return DEFAULT_MODEL_VERSION
    slug = model_name.rsplit("/", 1)[-1].lower().replace("_", "-")
    return f"{slug}-{dtype}-{output_dim}-v1"


def _empty_result(
    worker: MultimodalEmbedder,
    status: str,
    *,
    input_evidence: list[dict[str, object]] | None = None,
) -> MultimodalEmbeddingResult:
    return MultimodalEmbeddingResult(
        multimodal_embedding=[],
        embedding_dim=worker.output_dim,
        multimodal_embedding_model=worker.model_name,
        multimodal_embedding_model_version=worker.model_version,
        multimodal_embedding_model_revision=worker.model_revision,
        multimodal_embedding_method_version=worker.method_version,
        multimodal_embedding_status=status,
        input_evidence=input_evidence or [],
    )


def _ordered_input_evidence(
    bundle: PostMediaBundle | None,
    visuals: tuple[PreparedVisual, ...],
) -> list[dict[str, object]]:
    if bundle is None:
        return []
    visuals_by_position: dict[int, list[dict[str, object]]] = {}
    for visual in visuals:
        visuals_by_position.setdefault(visual.evidence.source_position, []).append(
            visual.evidence.to_dict()
        )

    evidence: list[dict[str, object]] = []
    for item in bundle.items:
        if item.duplicate_of_position is not None:
            evidence.append(
                {
                    "source_position": item.position,
                    "kind": "duplicate",
                    "duplicate_of_position": item.duplicate_of_position,
                }
            )
        else:
            evidence.extend(visuals_by_position.get(item.position, ()))
    return evidence


class MultimodalEmbedder:
    """Embed ordered visuals plus trusted text into one versioned Qwen space."""

    def __init__(
        self,
        model_name: str | None = None,
        model_revision: str | None = None,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        device: str | None = None,
        dtype: str | None = None,
        adapter: EmbeddingAdapter | None = None,
    ) -> None:
        self.model_name = (
            model_name or os.environ.get("MULTIMODAL_EMBEDDER_MODEL") or DEFAULT_MODEL_NAME
        )
        if model_revision is not None:
            resolved_revision = model_revision
        elif self.model_name == DEFAULT_MODEL_NAME:
            resolved_revision = DEFAULT_MODEL_REVISION
        else:
            resolved_revision = "unpinned"
        if _IMMUTABLE_REVISION.fullmatch(resolved_revision) is None:
            raise ValueError(
                "multimodal model revision must be a 40-character lowercase hex commit"
            )
        if output_dim <= 0:
            raise ValueError("multimodal output dimension must be positive")

        self.model_revision = resolved_revision
        self.output_dim = output_dim
        self.device = device or os.environ.get("MULTIMODAL_EMBEDDER_DEVICE") or DEFAULT_DEVICE
        self.dtype = dtype or os.environ.get("MULTIMODAL_EMBEDDER_DTYPE") or DEFAULT_DTYPE
        self.model_version = _model_version(
            self.model_name,
            self.dtype,
            self.output_dim,
        )
        self.method_version = DEFAULT_METHOD_VERSION
        self._adapter = adapter

    @property
    def is_warm(self) -> bool:
        return self._adapter is not None

    def warmup(self) -> None:
        """Load the pinned checkpoint once and fail loudly if unavailable."""
        if self.is_warm:
            return
        from postprism.multimodal.qwen_model import (  # noqa: PLC0415
            load_qwen_embedding_adapter,
        )

        self._adapter = load_qwen_embedding_adapter(
            self.model_name,
            self.model_revision,
            self.device,
            self.dtype,
        )
        logger.info(
            "Multimodal embedder ready: model=%s revision=%s version=%s",
            self.model_name,
            self.model_revision,
            self.model_version,
        )

    def embed(
        self,
        bundle: PostMediaBundle | None,
        paths: Mapping[int, Path],
        post_text: str | None,
        asr_results: Sequence[Mapping[str, object]] | None,
        timings: dict[str, float] | None = None,
    ) -> MultimodalEmbeddingResult:
        """Return a status-tagged embedding without changing other pipeline legs.

        `timings` receives `video_decode` and `qwen_processing` milliseconds.
        `qwen_processing` spans the whole adapter call: chat-template assembly,
        visual preprocessing, processor tokenisation, device transfer, the
        forward pass, and pooling, so it is the model-side cost after frames
        exist, not the forward pass alone.
        """
        text = assemble_multimodal_text(post_text, asr_results)
        if bundle is None and not text:
            return _empty_result(self, "no_input")
        if not self.is_warm:
            raise RuntimeError("MultimodalEmbedder not warmed up; call .warmup() before .embed()")

        if timings is not None:
            timings["video_decode"] = 0.0
        try:
            visuals = prepare_ordered_visuals(bundle, paths, timings) if bundle is not None else ()
        except VisualPreparationError as err:
            logger.warning("Multimodal media preparation failed: %s", err)
            evidence = (
                [
                    {
                        "source_position": err.source_position,
                        "kind": "media_load_failed",
                    }
                ]
                if err.source_position is not None
                else []
            )
            return _empty_result(
                self,
                "media_load_failed",
                input_evidence=evidence,
            )

        evidence = _ordered_input_evidence(bundle, visuals)
        if not visuals and not text:
            return _empty_result(self, "no_input", input_evidence=evidence)

        assert self._adapter is not None
        inference_started = time.perf_counter()
        try:
            native, _ = self._adapter.process(text, visuals)
            embedding = normalized_mrl_prefix(native, self.output_dim)
        except ValueError as err:
            if str(err).startswith(_CONTEXT_LIMIT_ERROR_PREFIX):
                logger.warning("Multimodal input rejected: %s", err)
                return _empty_result(self, "input_too_long", input_evidence=evidence)
            logger.exception("Multimodal embedding failed: %s", err)
            return _empty_result(self, "inference_failed", input_evidence=evidence)
        except Exception as err:  # noqa: BLE001 -- model faults stay in-band
            logger.exception("Multimodal embedding failed: %s", err)
            return _empty_result(self, "inference_failed", input_evidence=evidence)
        finally:
            if timings is not None:
                timings["qwen_processing"] = (time.perf_counter() - inference_started) * 1000.0

        return MultimodalEmbeddingResult(
            multimodal_embedding=embedding.tolist(),
            embedding_dim=self.output_dim,
            multimodal_embedding_model=self.model_name,
            multimodal_embedding_model_version=self.model_version,
            multimodal_embedding_model_revision=self.model_revision,
            multimodal_embedding_method_version=self.method_version,
            multimodal_embedding_status="ok" if visuals else "text_only_ok",
            input_evidence=evidence,
        )
