"""Lazy Qwen-ASR loading and result normalization.

Model imports stay inside the loader so importing the package does not load weights.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class QwenAsrModelProtocol(Protocol):
    """Minimal interface exposed by `qwen_asr.Qwen3ASRModel`."""

    def transcribe(self, audio: str, **kwargs: Any) -> Any: ...


def load_qwen_asr_model(
    model_name: str,
    device: str,
    compute_type: str,
    max_new_tokens: int,
    max_batch_size: int,
    model_revision: str,
) -> QwenAsrModelProtocol:
    """Import `qwen-asr` lazily and load a Qwen3-ASR model."""
    import torch  # noqa: PLC0415
    from qwen_asr import Qwen3ASRModel  # noqa: PLC0415

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(compute_type.lower())
    if dtype is None:
        raise ValueError(
            f"Qwen ASR compute type must be bfloat16, float16, or float32; got {compute_type!r}"
        )

    device_map = "cuda:0" if device in {"cuda", "gpu"} else device
    logger.info(
        "Loading Qwen ASR: model=%s device_map=%s dtype=%s max_new_tokens=%d max_batch_size=%d",
        model_name,
        device_map,
        compute_type,
        max_new_tokens,
        max_batch_size,
    )
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    model_source = snapshot_download(repo_id=model_name, revision=model_revision)
    return Qwen3ASRModel.from_pretrained(
        model_source,
        dtype=dtype,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        max_inference_batch_size=max_batch_size,
    )


_QWEN_LANGUAGE_MAP: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}
_QWEN_LANGUAGE_REVERSE: dict[str, str] = {
    value.lower(): key for key, value in _QWEN_LANGUAGE_MAP.items()
}


def normalize_language_output(language: Any) -> str:
    """Normalize ASR language output to the short codes stored by the schema."""
    if language is None:
        return ""
    value = str(language).strip()
    if not value:
        return ""
    lowered = value.lower()
    return _QWEN_LANGUAGE_REVERSE.get(lowered, lowered)


def qwen_result_value(result: Any, name: str) -> Any:
    """Read a field from either an object-style or dict-style Qwen result."""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def first_qwen_result(results: Any) -> Any:
    """Normalize Qwen's batch-shaped return to a single result object."""
    if isinstance(results, (list, tuple)):
        return results[0] if results else {}
    return results
