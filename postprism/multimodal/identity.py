"""Model-free identity of the frozen primary Qwen retrieval space."""

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
DEFAULT_OUTPUT_DIM = 1024
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MODEL_VERSION = "qwen3-vl-embedding-2b-bfloat16-1024-v1"
# `tsorigin`: video frame targets are measured from the stream's first
# presentation timestamp instead of absolute zero, which changes frame selection
# only for media whose timestamps do not start at 0 (zero-start media keeps
# byte-identical inputs). Bumped before initial deployment, so no index exists
# under the previous string.
DEFAULT_METHOD_VERSION = (
    "qwen-vl-embed-v5-pixelcaps-shapesafeblocks-noinputtrunc-frames8-"
    "p10-20p7143-31p4286-42p1429-52p8571-63p5714-74p2857-85-"
    "posttext-trustedasr-tsorigin"
)
