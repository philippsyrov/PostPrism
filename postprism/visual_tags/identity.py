"""Pinned RAM++ metadata identity, independent of model imports."""

from postprism.schemas import MetadataIdentity

RAM_IDENTITY = MetadataIdentity(
    model="RAM++-Swin-L-14M",
    model_revision="7cb804a8609e9f4b1a50b7f31436d2df40bb9481",
    method_version=(
        "ram-native-compact-display-v1-exif-rgb-transform384-float32-frames8-per-video-tsorigin"
    ),
    dtype="float32",
    artifacts={
        "checkpoint_sha256": "497c178836ba66698ca226c7895317e6e800034be986452dbd2593298d50e87d",
        "vocabulary_sha256": "db241b78f0691cc0291a2c876a5eba562653526d34a732b3f48a525f0ce2384f",
        "thresholds_sha256": "43a429712680f4daabb3937e86ab11c268fc479a74f563d58a521b3e1f870844",
    },
)
