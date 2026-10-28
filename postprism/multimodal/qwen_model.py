"""Reviewed local adapter for Qwen3-VL embedding checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from postprism.multimodal.sampling import PreparedVisual

MAX_LENGTH = 8192
IMAGE_PATCH_SIZE = 16
IMAGE_FACTOR = IMAGE_PATCH_SIZE * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
# qwen-vl-utils caps each frame at min(FRAME_MAX_PIXELS, total_pixels / frames * 2),
# so a block's total_pixels is really a budget per temporal pair. One accepted
# video spends at most four pairs at the frame cap; multi-video posts share
# that same spend across every real-video block instead of multiplying it.
SHARED_VIDEO_PAIR_PIXELS = 4 * FRAME_MAX_PIXELS

DOCUMENT_INSTRUCTION = (
    "Represent this social media post for semantic retrieval. Capture its overall "
    "meaning, including relationships between chronological frames, visible text, "
    "post text, and trusted speech. Treat watermarks as provenance rather than the "
    "post's main meaning."
)


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    attention_mask: torch.Tensor | None = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Native Transformers backbone with the official embedding output surface."""

    _checkpoint_conversion_mapping: dict[str, str] = {}
    accepts_loss_kwargs = False
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self) -> Any:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: Any) -> None:
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _embedding_model_class() -> type[Qwen3VLForEmbedding]:
    return Qwen3VLForEmbedding


def _processor_class() -> type[Qwen3VLProcessor]:
    return Qwen3VLProcessor


def _append_shape_safe_blocks(
    content: list[dict[str, object]],
    run: Sequence[PreparedVisual],
    *,
    lone_image_max_pixels: int,
) -> None:
    """Pack only equal-size neighbors because Qwen stacks video frames first."""
    start = 0
    while start < len(run):
        end = start + 1
        while end < len(run) and run[end].image.size == run[start].image.size:
            end += 1
        same_shape = run[start:end]
        paired_count = len(same_shape) - (len(same_shape) % 2)
        if paired_count:
            content.append(
                {
                    "type": "video",
                    "video": [item.image for item in same_shape[:paired_count]],
                    "total_pixels": MAX_TOTAL_PIXELS,
                }
            )
        if paired_count != len(same_shape):
            content.append(
                {
                    "type": "image",
                    "image": same_shape[-1].image,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": lone_image_max_pixels,
                }
            )
        start = end


def _share_video_pixel_budget(video_run_blocks: list[dict[str, object]]) -> None:
    """Split one accepted video's pair budget across every block cut from real videos.

    A lone frame left by short-video deduplication costs Qwen as much as a pair,
    so it takes one pair's share through ``max_pixels`` instead of escaping the
    budget at the full frame cap.
    """
    units_by_block = [
        len(block["video"]) // 2 if block["type"] == "video" else 1 for block in video_run_blocks
    ]
    total_units = sum(units_by_block)
    for block, units in zip(video_run_blocks, units_by_block, strict=True):
        share = SHARED_VIDEO_PAIR_PIXELS * units // total_units
        if block["type"] == "video":
            block["total_pixels"] = share
        else:
            block["max_pixels"] = max(MIN_PIXELS, min(FRAME_MAX_PIXELS, share))


def build_document_conversation(
    text: str,
    visuals: Sequence[PreparedVisual],
    *,
    instruction: str = DOCUMENT_INSTRUCTION,
) -> list[dict[str, object]]:
    """Represent each prepared visual exactly once without helper-side padding."""
    content: list[dict[str, object]] = []
    video_run_blocks: list[dict[str, object]] = []
    video_sources: set[int] = set()
    index = 0
    while index < len(visuals):
        visual = visuals[index]
        if visual.evidence.kind == "image":
            end = index
            while end < len(visuals) and visuals[end].evidence.kind == "image":
                end += 1
            run = visuals[index:end]
            _append_shape_safe_blocks(
                content,
                run,
                lone_image_max_pixels=(MAX_PIXELS if len(run) == 1 else FRAME_MAX_PIXELS),
            )
            index = end
            continue
        if visual.evidence.kind != "video_frame":
            raise ValueError(f"unsupported prepared visual kind: {visual.evidence.kind!r}")

        source_position = visual.evidence.source_position
        end = index
        while (
            end < len(visuals)
            and visuals[end].evidence.kind == "video_frame"
            and visuals[end].evidence.source_position == source_position
        ):
            end += 1
        run = visuals[index:end]
        video_sources.add(source_position)
        first_new_block = len(content)
        _append_shape_safe_blocks(
            content,
            run,
            lone_image_max_pixels=FRAME_MAX_PIXELS,
        )
        video_run_blocks.extend(content[first_new_block:])
        index = end
    if len(video_sources) > 1:
        _share_video_pixel_budget(video_run_blocks)
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        raise ValueError("multimodal model input must not be empty")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {"role": "user", "content": content},
    ]


def pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last attended token, matching Qwen's official implementation."""
    last_one_positions = attention_mask.flip(dims=[1]).argmax(dim=1)
    columns = attention_mask.shape[1] - last_one_positions - 1
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[rows, columns]


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as err:
        raise ValueError("MULTIMODAL_EMBEDDER_DTYPE must be bfloat16, float16, or float32") from err


def _require_qwen_vl_utils() -> Any:
    """Resolve the external visual preprocessor before model loading starts."""
    from qwen_vl_utils.vision_process import process_vision_info  # noqa: PLC0415

    return process_vision_info


class QwenEmbeddingAdapter:
    """Preprocess one post and return its native embedding vector."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_length = max_length

    def process(
        self,
        text: str,
        visuals: tuple[PreparedVisual, ...],
    ) -> tuple[np.ndarray, bool]:
        return self.process_with_instruction(text, visuals, DOCUMENT_INSTRUCTION)

    def process_with_instruction(
        self,
        text: str,
        visuals: tuple[PreparedVisual, ...],
        instruction: str,
    ) -> tuple[np.ndarray, bool]:
        conversations = [build_document_conversation(text, visuals, instruction=instruction)]
        prompt = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = _require_qwen_vl_utils()(
            conversations,
            image_patch_size=IMAGE_PATCH_SIZE,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs, strict=True)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        video_kwargs = dict(video_kwargs)
        video_kwargs["do_sample_frames"] = False
        video_kwargs.pop("max_frames", None)

        inputs = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=False,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"multimodal input exceeds Qwen context limit: "
                f"{input_ids.shape[1]} > {self.max_length} tokens"
            )
        model_inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            embedding = pool_last(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
            )[0]
        vector = embedding.detach().float().cpu().numpy()
        return np.asarray(vector, dtype=np.float32), False


def load_qwen_embedding_adapter(
    repo_id: str,
    revision: str,
    device: str,
    dtype: str,
) -> QwenEmbeddingAdapter:
    """Load exact weights into locally reviewed native Transformers code."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    _require_qwen_vl_utils()
    snapshot = snapshot_download(repo_id=repo_id, revision=revision)
    model, loading_info = _embedding_model_class().from_pretrained(
        snapshot,
        torch_dtype=_resolve_torch_dtype(dtype),
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    incomplete = {
        key: loading_info.get(key, [])
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(key)
    }
    if incomplete:
        raise RuntimeError(f"incomplete Qwen checkpoint load: {incomplete}")
    model = model.to(device)
    model.eval()
    processor = _processor_class().from_pretrained(snapshot, padding_side="right")
    return QwenEmbeddingAdapter(model, processor)
