"""Synchronous per-image/frame RAM metadata, independent of embedding inputs."""

import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from postprism.media import IMAGE_FORMATS_BY_EXTENSION, MAX_IMAGE_PIXELS, PostMediaBundle
from postprism.multimodal.sampling import (
    VIDEO_FRAME_FRACTIONS,
    PreparedVisual,
    VisualInputEvidence,
    prepared_visual_sha256,
    sample_video_frames,
)
from postprism.schemas import VisualMetadataResult
from postprism.visual_tags.adapter import load_ram_adapter
from postprism.visual_tags.identity import RAM_IDENTITY
from postprism.visual_tags.policy import compact_native_tags

logger = logging.getLogger(__name__)


def _failure(position: int, kind: str, status: str, error: Exception) -> dict[str, object]:
    logger.warning("RAM position %d: %s", position, type(error).__name__)
    return {
        "source_position": position,
        "kind": kind,
        "status": status,
        "native_tags": [],
        "display_tags": [],
    }


class VisualTagger:
    def __init__(self, adapter=None):
        self._adapter = adapter

    @property
    def is_warm(self) -> bool:
        return self._adapter is not None

    @property
    def identity(self):
        return self._adapter.identity if self._adapter is not None else RAM_IDENTITY

    @property
    def model_name(self) -> str:
        return self.identity.model

    @property
    def model_revision(self) -> str:
        return self.identity.model_revision

    @property
    def model_version(self) -> str:
        return self.identity.method_version

    def warmup(self) -> None:
        if self.is_warm:
            return
        for name in ("RAM_CHECKPOINT_FILE", "RAM_TOKENIZER_DIR"):
            if not os.environ.get(name):
                raise RuntimeError(f"{name} is required for RAM warmup")
        adapter = load_ram_adapter(
            Path(os.environ["RAM_CHECKPOINT_FILE"]),
            Path(os.environ["RAM_TOKENIZER_DIR"]),
            device="cuda:0",
        )
        adapter.predict(Image.new("RGB", (384, 384)))
        self._adapter = adapter

    def _infer(self, visual: PreparedVisual) -> dict[str, object]:
        evidence = visual.evidence.to_dict()
        try:
            prediction = self._adapter.predict(visual.image)
            return {
                **evidence,
                "status": "ok",
                "native_tags": list(prediction.native_tags),
                "display_tags": compact_native_tags(
                    prediction.native_tags, prediction.vocabulary, prediction.logits
                ),
            }
        except Exception as error:
            return {
                **evidence,
                **_failure(
                    visual.evidence.source_position, visual.evidence.kind, "inference_failed", error
                ),
            }

    def enrich(
        self, bundle: PostMediaBundle | None, paths: Mapping[int, Path]
    ) -> VisualMetadataResult:
        items = []
        if bundle is None:
            return VisualMetadataResult(self.identity, items)
        if not self.is_warm:
            raise RuntimeError("RAM worker must be warm before media enrichment")
        for item in bundle.items:
            if item.duplicate_of_position is not None:
                items.append(
                    {
                        "source_position": item.position,
                        "kind": "duplicate",
                        "status": "skipped_duplicate",
                        "duplicate_of_position": item.duplicate_of_position,
                    }
                )
                continue
            try:
                if item.upload.media_type == "video":
                    visuals = sample_video_frames(paths[item.position], VIDEO_FRAME_FRACTIONS)
                    if not visuals:
                        raise ValueError("video has no decoded frames")
                else:
                    with Image.open(
                        paths[item.position], formats=IMAGE_FORMATS_BY_EXTENSION[item.upload.suffix]
                    ) as source:
                        if source.width * source.height > MAX_IMAGE_PIXELS:
                            raise ValueError("image exceeds pixel limit")
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    visuals = (
                        PreparedVisual(
                            image,
                            VisualInputEvidence(
                                item.position, "image", sha256=prepared_visual_sha256(image)
                            ),
                        ),
                    )
            except Exception as error:
                items.append(
                    _failure(item.position, item.upload.media_type, "media_load_failed", error)
                )
                continue
            items.extend(
                self._infer(
                    replace(
                        visual, evidence=replace(visual.evidence, source_position=item.position)
                    )
                )
                for visual in visuals
            )
        return VisualMetadataResult(self.identity, items)
