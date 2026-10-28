"""Synchronous per-media semantic captions, independent of retrieval inputs."""

import logging
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from postprism.captioning.adapter import load_caption_adapter
from postprism.captioning.identity import CAPTION_IDENTITY
from postprism.captioning.media import bound_frame_pixels, load_caption_visual
from postprism.captioning.policy import generate_caption
from postprism.media import PostMediaBundle
from postprism.multimodal.sampling import prepared_visual_sha256
from postprism.schemas import VisualMetadataResult

logger = logging.getLogger(__name__)


class Captioner:
    def __init__(self, adapter=None):
        self._adapter = adapter

    @property
    def is_warm(self) -> bool:
        return self._adapter is not None

    @property
    def identity(self):
        return self._adapter.identity if self._adapter is not None else CAPTION_IDENTITY

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
        adapter = load_caption_adapter()
        if generate_caption(adapter, [Image.new("RGB", (384, 384))]) is None:
            raise RuntimeError("Caption warmup did not produce a valid caption")
        self._adapter = adapter

    def enrich(
        self, bundle: PostMediaBundle | None, paths: Mapping[int, Path]
    ) -> VisualMetadataResult:
        items = []
        if bundle is None:
            return VisualMetadataResult(self.identity, items)
        if not self.is_warm:
            raise RuntimeError("Caption worker must be warm before media enrichment")
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
                visual = load_caption_visual(paths[item.position], item.upload.media_type)
            except Exception as error:
                logger.warning(
                    "Caption position %d media load: %s",
                    item.position,
                    type(error).__name__,
                )
                items.append(
                    {
                        "source_position": item.position,
                        "kind": item.upload.media_type,
                        "status": "media_load_failed",
                        "caption": "",
                    }
                )
                continue
            evidence = {
                "source_position": item.position,
                "kind": item.upload.media_type,
                "sha256": prepared_visual_sha256(visual.image),
            }
            if item.upload.media_type == "video":
                evidence.update(
                    seconds=visual.seconds, seek_fallback_used=visual.seek_fallback_used
                )
            try:
                caption = generate_caption(self._adapter, [bound_frame_pixels(visual.image)])
            except Exception as error:
                logger.warning(
                    "Caption position %d inference: %s",
                    item.position,
                    type(error).__name__,
                )
                caption = None
            items.append(
                {
                    **evidence,
                    "status": "ok" if caption is not None else "inference_failed",
                    "caption": caption or "",
                }
            )
        return VisualMetadataResult(self.identity, items)
