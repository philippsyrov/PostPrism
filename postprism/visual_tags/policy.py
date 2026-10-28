"""Compact display ordering within RAM's native emitted tags."""

import math
from collections.abc import Sequence


def compact_native_tags(
    native_tags: Sequence[str], vocabulary: Sequence[str], logits: Sequence[float]
) -> list[str]:
    index = {tag: i for i, tag in enumerate(vocabulary)}
    if len(index) != len(vocabulary) or len(vocabulary) != len(logits):
        raise ValueError("invalid RAM vocabulary/logit shape")
    if len(set(native_tags)) != len(native_tags) or any(t not in index for t in native_tags):
        raise ValueError("invalid native RAM tags")
    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("non-finite RAM logits")
    return sorted(native_tags, key=lambda tag: (-float(logits[index[tag]]), index[tag]))[:6]
