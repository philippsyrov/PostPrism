"""Pinned caption-only metadata identity, independent of model imports."""

from postprism.schemas import MetadataIdentity

CAPTION_IDENTITY = MetadataIdentity(
    model="Qwen/Qwen3-VL-2B-Instruct",
    model_revision="89644892e4d85e24eaac8bacfd4f463576704203",
    method_version=(
        "qwen-caption-promptv4-bfloat16-greedy-beam1-tok512-imagecap4m-middle-frame-v1-retry1"
    ),
    dtype="bfloat16",
)
