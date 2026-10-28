from __future__ import annotations

import struct
import sys
import types
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from postprism.media import (
    UploadPayload,
    build_post_media_bundle,
    materialized_post_media,
)
from postprism.multimodal import sampling
from postprism.multimodal.embedder import (
    DEFAULT_MODEL_REVISION,
    MultimodalEmbedder,
    normalized_mrl_prefix,
)
from postprism.multimodal.qwen_model import (
    DOCUMENT_INSTRUCTION,
    MAX_TOTAL_PIXELS,
    SHARED_VIDEO_PAIR_PIXELS,
    QwenEmbeddingAdapter,
    build_document_conversation,
    load_qwen_embedding_adapter,
    pool_last,
)
from postprism.multimodal.sampling import (
    PreparedVisual,
    VisualInputEvidence,
    prepared_visual_sha256,
)

FIXTURE = Path(__file__).parent / "fixtures" / "generated" / "carousel_green.png"


class _FakeAdapter:
    def __init__(self, vector: np.ndarray | None = None) -> None:
        self.vector = vector if vector is not None else np.arange(1, 2049, dtype=np.float32)
        self.calls: list[tuple[str, tuple[PreparedVisual, ...]]] = []

    def process(
        self,
        text: str,
        visuals: tuple[PreparedVisual, ...],
    ) -> tuple[np.ndarray, bool]:
        self.calls.append((text, visuals))
        return self.vector, False


class _FailingAdapter(_FakeAdapter):
    def process(
        self,
        text: str,
        visuals: tuple[PreparedVisual, ...],
    ) -> tuple[np.ndarray, bool]:
        raise RuntimeError("model failed")


def _upload(filename: str, content: bytes) -> UploadPayload:
    return UploadPayload(
        filename=filename,
        suffix=".png",
        media_type="image",
        content=content,
    )


def test_prepared_visual_hash_identifies_exact_model_pixels() -> None:
    image = Image.new("RGB", (2, 2), color=(1, 2, 3))

    assert prepared_visual_sha256(image) == (
        "dc553683fef6520c4fb631880bbe36c3c27d6bc8f826aa79c1ccad22e1387fe8"
    )


def test_normalized_mrl_prefix_uses_first_1024_dimensions() -> None:
    native = np.arange(1, 2049, dtype=np.float32)

    actual = normalized_mrl_prefix(native, 1024)

    assert actual.shape == (1024,)
    assert np.linalg.norm(actual) == pytest.approx(1.0)
    np.testing.assert_allclose(
        actual,
        native[:1024] / np.linalg.norm(native[:1024]),
    )


@pytest.mark.parametrize(
    "native",
    [
        np.ones(1023, dtype=np.float32),
        np.zeros(2048, dtype=np.float32),
        np.full(2048, np.nan, dtype=np.float32),
    ],
)
def test_normalized_mrl_prefix_rejects_invalid_native_vectors(
    native: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="invalid native multimodal embedding"):
        normalized_mrl_prefix(native, 1024)


def test_worker_does_not_run_model_for_empty_input() -> None:
    adapter = _FakeAdapter()

    result = MultimodalEmbedder(adapter=adapter).embed(None, {}, None, None)

    assert result.multimodal_embedding_status == "no_input"
    assert adapter.calls == []


def test_worker_versions_the_exact_block_and_fail_closed_input_policy() -> None:
    worker = MultimodalEmbedder(adapter=_FakeAdapter())

    assert worker.method_version == (
        "qwen-vl-embed-v5-pixelcaps-shapesafeblocks-noinputtrunc-frames8-"
        "p10-20p7143-31p4286-42p1429-52p8571-63p5714-74p2857-85-"
        "posttext-trustedasr-tsorigin"
    )


def test_worker_embeds_text_only_with_trusted_asr() -> None:
    adapter = _FakeAdapter()
    worker = MultimodalEmbedder(adapter=adapter)

    result = worker.embed(
        None,
        {},
        "post",
        [
            {"asr_transcript": "speech", "asr_used_for_embedding": True},
            {"asr_transcript": "lyrics", "asr_used_for_embedding": False},
            {"asr_transcript": "more speech", "asr_used_for_embedding": True},
        ],
    )

    assert result.multimodal_embedding_status == "text_only_ok"
    assert len(result.multimodal_embedding) == 1024
    assert adapter.calls[0][0] == (
        "Post text: post\n\nTrusted speech transcript: speech"
        "\n\nTrusted speech transcript: more speech"
    )
    assert adapter.calls[0][1] == ()


def test_normal_post_keeps_document_instruction() -> None:
    conversation = build_document_conversation("ordinary post", ())
    assert conversation[0]["content"] == [{"type": "text", "text": DOCUMENT_INSTRUCTION}]


def test_worker_records_duplicate_provenance_without_duplicate_model_weight() -> None:
    first = _upload("first.png", FIXTURE.read_bytes())
    second = _upload("second.png", FIXTURE.read_bytes() + b"\x00")
    bundle = build_post_media_bundle((first, first, second))
    adapter = _FakeAdapter()
    worker = MultimodalEmbedder(adapter=adapter)

    with materialized_post_media(bundle) as paths:
        result = worker.embed(bundle, paths, None, None)

    assert [visual.evidence.source_position for visual in adapter.calls[0][1]] == [
        0,
        2,
    ]
    assert result.input_evidence == [
        {
            "source_position": 0,
            "kind": "image",
            "sha256": "6badd014fd098521fd67e8b47877340a29ec4fe08c2d680248d018ff47604604",
        },
        {
            "source_position": 1,
            "kind": "duplicate",
            "duplicate_of_position": 0,
        },
        {
            "source_position": 2,
            "kind": "image",
            "sha256": "6badd014fd098521fd67e8b47877340a29ec4fe08c2d680248d018ff47604604",
        },
    ]


def test_worker_maps_corrupt_media_to_media_load_failed() -> None:
    bundle = build_post_media_bundle((_upload("broken.png", b"not an image"),))
    adapter = _FakeAdapter()
    worker = MultimodalEmbedder(adapter=adapter)

    with materialized_post_media(bundle) as paths:
        result = worker.embed(bundle, paths, None, None)

    assert result.multimodal_embedding_status == "media_load_failed"
    assert result.multimodal_embedding == []
    assert result.input_evidence == [{"source_position": 0, "kind": "media_load_failed"}]
    assert adapter.calls == []


@pytest.mark.parametrize(
    "chunk_type,payload,parser_error",
    [
        (b"zTXt", b"label\x00\x01bad", SyntaxError),
        (b"gAMA", b"", struct.error),
        (b"iCCP", b"", IndexError),
    ],
)
def test_worker_maps_recognized_image_parser_errors_to_media_load_failed(
    chunk_type, payload, parser_error
) -> None:
    encoded = BytesIO()
    Image.new("RGB", (2, 2), "red").save(encoded, format="PNG")
    png = encoded.getvalue()
    chunk = (
        len(payload).to_bytes(4, "big")
        + chunk_type
        + payload
        + zlib.crc32(chunk_type + payload).to_bytes(4, "big")
    )
    # A valid chunk envelope after IDAT reaches Pillow's lazy pixel decode.
    malformed = png[:-12] + chunk + png[-12:]
    bundle = build_post_media_bundle(
        (_upload("valid.png", png), _upload("malformed.png", malformed))
    )
    adapter = _FakeAdapter()
    worker = MultimodalEmbedder(adapter=adapter)

    with materialized_post_media(bundle) as paths:
        with pytest.raises(sampling.VisualPreparationError) as failure:
            sampling.prepare_ordered_visuals(bundle, paths)
        assert failure.value.source_position == 1
        assert isinstance(failure.value.__cause__, parser_error)
        result = worker.embed(bundle, paths, None, None)

    assert result.multimodal_embedding_status == "media_load_failed"
    assert result.multimodal_embedding == []
    assert result.input_evidence == [{"source_position": 1, "kind": "media_load_failed"}]
    assert adapter.calls == []


@pytest.mark.parametrize(
    "error_type", [ValueError, KeyError, IndexError, SyntaxError, struct.error, RuntimeError]
)
def test_visual_provenance_programming_errors_are_not_media_failures(
    monkeypatch, error_type
) -> None:
    failure = error_type("provenance bug")

    def fail_hash(image):
        raise failure

    monkeypatch.setattr(sampling, "prepared_visual_sha256", fail_hash)
    bundle = build_post_media_bundle((_upload("valid.png", FIXTURE.read_bytes()),))
    worker = MultimodalEmbedder(adapter=_FakeAdapter())
    with materialized_post_media(bundle) as paths, pytest.raises(error_type) as raised:
        worker.embed(bundle, paths, None, None)
    assert raised.value is failure


def test_worker_maps_model_failure_to_independent_status() -> None:
    worker = MultimodalEmbedder(adapter=_FailingAdapter())

    result = worker.embed(None, {}, "text", None)

    assert result.multimodal_embedding_status == "inference_failed"
    assert result.multimodal_embedding == []


def test_worker_distinguishes_context_overflow_from_model_failure() -> None:
    class _OverLimitAdapter:
        def process(self, *_args: object) -> object:
            raise ValueError("multimodal input exceeds Qwen context limit: 9000 > 8192 tokens")

    result = MultimodalEmbedder(adapter=_OverLimitAdapter()).embed(None, {}, "x" * 20_000, None)

    assert result.multimodal_embedding_status == "input_too_long"
    assert result.multimodal_embedding == []


def test_non_default_model_requires_exact_revision() -> None:
    with pytest.raises(ValueError, match="40-character lowercase hex"):
        MultimodalEmbedder(model_name="someone/custom", model_revision="unpinned")


def test_adapter_preserves_a_single_post_image_as_an_image_block() -> None:
    visual = PreparedVisual(
        image=Image.new("RGB", (2, 2), color=(1, 0, 0)),
        evidence=VisualInputEvidence(
            source_position=0,
            kind="image",
        ),
    )

    conversation = build_document_conversation("Post text: hi", (visual,))

    visual_block = conversation[1]["content"][0]
    assert visual_block["type"] == "image"
    assert visual_block["image"] is visual.image
    assert visual_block["min_pixels"] == 4_096
    assert visual_block["max_pixels"] == 1_843_200
    assert conversation[1]["content"][1] == {
        "type": "text",
        "text": "Post text: hi",
    }


@pytest.mark.parametrize("count", [2, 3, 4, 5, 10])
def test_adapter_packs_carousel_images_without_padding_the_last_slide(
    count: int,
) -> None:
    visuals = tuple(
        PreparedVisual(
            image=Image.new("RGB", (2, 2), color=(index, 0, 0)),
            evidence=VisualInputEvidence(source_position=index, kind="image"),
        )
        for index in range(count)
    )

    conversation = build_document_conversation("", visuals)

    blocks = conversation[1]["content"]
    assert all(block["type"] != "video" or len(block["video"]) % 2 == 0 for block in blocks)
    actual_images = []
    for block in blocks:
        if block["type"] == "video":
            actual_images.extend(block["video"])
        else:
            actual_images.append(block["image"])
    assert actual_images == [visual.image for visual in visuals]
    if count % 2:
        assert blocks[-1]["min_pixels"] == 4_096
        assert blocks[-1]["max_pixels"] == 786_432


@pytest.mark.parametrize("kind", ["image", "video_frame"])
def test_adapter_never_packs_different_shapes_into_one_video_block(
    kind: str,
) -> None:
    visuals = (
        PreparedVisual(
            image=Image.new("RGB", (1080, 1440), color=(1, 0, 0)),
            evidence=VisualInputEvidence(
                source_position=0,
                kind=kind,
                frame_index=0 if kind == "video_frame" else None,
            ),
        ),
        PreparedVisual(
            image=Image.new("RGB", (1080, 1080), color=(2, 0, 0)),
            evidence=VisualInputEvidence(
                source_position=0 if kind == "video_frame" else 1,
                kind=kind,
                frame_index=1 if kind == "video_frame" else None,
            ),
        ),
    )

    conversation = build_document_conversation("", visuals)

    blocks = conversation[1]["content"]
    assert [block["type"] for block in blocks] == ["image", "image"]
    assert [block["image"] for block in blocks] == [visual.image for visual in visuals]
    assert all(block["max_pixels"] == 786_432 for block in blocks)


def test_adapter_does_not_feed_an_odd_video_block_to_qwen() -> None:
    visuals = tuple(
        PreparedVisual(
            image=Image.new("RGB", (2, 2), color=(index, 0, 0)),
            evidence=VisualInputEvidence(
                source_position=0,
                kind="video_frame",
                frame_index=index,
            ),
        )
        for index in range(3)
    )

    conversation = build_document_conversation("", visuals)

    blocks = conversation[1]["content"]
    assert [block["type"] for block in blocks] == ["video", "image"]
    assert blocks[0]["video"] == [visuals[0].image, visuals[1].image]
    assert blocks[1]["image"] is visuals[2].image
    assert blocks[1]["min_pixels"] == 4_096
    assert blocks[1]["max_pixels"] == 786_432


def test_adapter_preserves_mixed_media_order_without_padding() -> None:
    kinds = (
        (0, "image"),
        (1, "video_frame"),
        (1, "video_frame"),
        (1, "video_frame"),
        (2, "image"),
    )
    visuals = tuple(
        PreparedVisual(
            image=Image.new("RGB", (2, 2), color=(index, 0, 0)),
            evidence=VisualInputEvidence(
                source_position=position,
                kind=kind,
                frame_index=index if kind == "video_frame" else None,
            ),
        )
        for index, (position, kind) in enumerate(kinds)
    )

    conversation = build_document_conversation("", visuals)

    blocks = conversation[1]["content"]
    assert [block["type"] for block in blocks] == [
        "image",
        "video",
        "image",
        "image",
    ]
    assert blocks[0]["image"] is visuals[0].image
    assert blocks[1]["video"] == [visuals[1].image, visuals[2].image]
    assert blocks[2]["image"] is visuals[3].image
    assert blocks[3]["image"] is visuals[4].image


def _video_frames(
    position: int, count: int, size: tuple[int, int] = (2, 2)
) -> list[PreparedVisual]:
    return [
        PreparedVisual(
            image=Image.new("RGB", size, color=(position, index, 0)),
            evidence=VisualInputEvidence(
                source_position=position,
                kind="video_frame",
                frame_index=index,
            ),
        )
        for index in range(count)
    ]


def _video_blocks(conversation: list[dict[str, object]]) -> list[dict[str, object]]:
    return [block for block in conversation[1]["content"] if block["type"] == "video"]


def test_single_video_block_keeps_the_accepted_pixel_budget_exactly() -> None:
    visuals = tuple(_video_frames(0, 8))

    blocks = _video_blocks(build_document_conversation("", visuals))

    assert len(blocks) == 1
    assert blocks[0]["total_pixels"] == MAX_TOTAL_PIXELS == 7_864_320
    assert SHARED_VIDEO_PAIR_PIXELS == 3_145_728


@pytest.mark.parametrize(
    ("allocation", "expected_budgets"),
    [
        ((4, 4), [1_572_864, 1_572_864]),
        ((4, 2, 2), [1_572_864, 786_432, 786_432]),
        ((2, 2, 2, 2), [786_432] * 4),
        ((2,) * 8, [393_216] * 8),
        ((2,) * 10, [314_572] * 10),
    ],
)
def test_multi_video_blocks_share_one_video_pair_budget(
    allocation: tuple[int, ...],
    expected_budgets: list[int],
) -> None:
    visuals = tuple(
        frame
        for position, count in enumerate(allocation)
        for frame in _video_frames(position, count)
    )

    blocks = _video_blocks(build_document_conversation("", visuals))

    assert [block["total_pixels"] for block in blocks] == expected_budgets
    assert [len(block["video"]) for block in blocks] == list(allocation)
    assert sum(block["total_pixels"] for block in blocks) <= SHARED_VIDEO_PAIR_PIXELS


def test_shared_video_budget_leaves_image_runs_alone_and_charges_lone_frames() -> None:
    visuals = (
        *_video_frames(0, 4),
        PreparedVisual(
            image=Image.new("RGB", (2, 2), color=(9, 0, 0)),
            evidence=VisualInputEvidence(source_position=1, kind="image"),
        ),
        PreparedVisual(
            image=Image.new("RGB", (2, 2), color=(9, 1, 0)),
            evidence=VisualInputEvidence(source_position=2, kind="image"),
        ),
        *_video_frames(3, 1),
    )

    blocks = build_document_conversation("", visuals)[1]["content"]

    assert [block["type"] for block in blocks] == ["video", "video", "image"]
    # Two pairs plus one deduplicated lone frame share three units; a share
    # above the per-frame cap is still clamped to that cap.
    assert blocks[0]["total_pixels"] == SHARED_VIDEO_PAIR_PIXELS * 2 // 3
    assert blocks[1]["total_pixels"] == MAX_TOTAL_PIXELS
    assert blocks[2]["max_pixels"] == min(786_432, SHARED_VIDEO_PAIR_PIXELS // 3) == 786_432


def test_ten_deduplicated_lone_frames_cannot_escape_the_shared_budget() -> None:
    visuals = tuple(frame for position in range(10) for frame in _video_frames(position, 1))

    blocks = build_document_conversation("", visuals)[1]["content"]

    assert all(block["type"] == "image" for block in blocks)
    assert all(block["max_pixels"] == SHARED_VIDEO_PAIR_PIXELS // 10 for block in blocks)


def test_single_video_lone_frame_keeps_the_accepted_frame_cap() -> None:
    blocks = build_document_conversation("", tuple(_video_frames(0, 3)))[1]["content"]

    assert [block["type"] for block in blocks] == ["video", "image"]
    assert blocks[0]["total_pixels"] == MAX_TOTAL_PIXELS
    assert blocks[1]["max_pixels"] == 786_432


def test_adapter_rejects_unknown_visual_evidence_kind() -> None:
    visual = PreparedVisual(
        image=Image.new("RGB", (2, 2)),
        evidence=VisualInputEvidence(source_position=0, kind="unknown"),
    )

    with pytest.raises(ValueError, match="unsupported prepared visual kind"):
        build_document_conversation("", (visual,))


def test_adapter_fails_before_pooling_when_input_exceeds_model_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Processor:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            assert kwargs["truncation"] is False
            assert "max_length" not in kwargs
            return {
                "input_ids": torch.ones((1, 5), dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            }

    class _Model:
        device = "cpu"

        def __call__(self, **_kwargs: object) -> object:
            pytest.fail("model must not run after an overlength input")

    vision_process = types.ModuleType("qwen_vl_utils.vision_process")
    vision_process.process_vision_info = lambda *_args, **_kwargs: (None, None, {})
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_process)

    adapter = QwenEmbeddingAdapter(_Model(), _Processor(), max_length=4)

    with pytest.raises(ValueError, match="exceeds Qwen context limit"):
        adapter.process("meaningful text", ())


def test_adapter_processes_images_and_video_metadata_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_calls: list[dict[str, object]] = []
    seen_conversations: list[object] = []
    image_input = object()
    video_tensor = object()
    video_metadata = {"fps": 1.0}

    class _Processor:
        def apply_chat_template(self, conversations: object, **_kwargs: object) -> str:
            seen_conversations.append(conversations)
            return "prompt"

        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            processor_calls.append(kwargs)
            return {
                "input_ids": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    class _Model:
        device = "cpu"

        def __call__(self, **_kwargs: object) -> object:
            hidden = torch.arange(3 * 2048, dtype=torch.float32).reshape(1, 3, 2048)
            return types.SimpleNamespace(last_hidden_state=hidden)

    def process_vision_info(
        conversations: object, **kwargs: object
    ) -> tuple[list[object], list[tuple[object, object]], dict[str, object]]:
        assert conversations is seen_conversations[0]
        assert kwargs == {
            "image_patch_size": 16,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        return [image_input], [(video_tensor, video_metadata)], {"max_frames": 8}

    vision_process = types.ModuleType("qwen_vl_utils.vision_process")
    vision_process.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_process)
    visuals = (
        PreparedVisual(
            image=Image.new("RGB", (2, 2)),
            evidence=VisualInputEvidence(source_position=0, kind="image"),
        ),
        PreparedVisual(
            image=Image.new("RGB", (2, 2)),
            evidence=VisualInputEvidence(source_position=1, kind="video_frame", frame_index=1),
        ),
        PreparedVisual(
            image=Image.new("RGB", (2, 2)),
            evidence=VisualInputEvidence(source_position=1, kind="video_frame", frame_index=2),
        ),
    )

    vector, text_truncated = QwenEmbeddingAdapter(_Model(), _Processor()).process("post", visuals)

    assert vector.shape == (2048,)
    assert text_truncated is False
    assert processor_calls[0]["images"] == [image_input]
    assert processor_calls[0]["videos"] == [video_tensor]
    assert processor_calls[0]["video_metadata"] == [video_metadata]
    assert processor_calls[0]["do_sample_frames"] is False
    assert "max_frames" not in processor_calls[0]


def test_adapter_pools_last_attended_token() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [9.0]], [[3.0], [8.0], [0.0]]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    assert pool_last(hidden, mask).tolist() == [[9.0], [8.0]]


def test_loader_snapshots_exact_revision_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from postprism.multimodal import qwen_model

    snapshot_calls: list[dict[str, object]] = []
    model_calls: list[tuple[object, dict[str, object]]] = []
    processor_calls: list[tuple[object, dict[str, object]]] = []

    class _LoadedModel:
        device = "cuda:0"

        @classmethod
        def from_pretrained(
            cls, source: object, **kwargs: object
        ) -> tuple[_LoadedModel, dict[str, list[object]]]:
            model_calls.append((source, kwargs))
            return cls(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

        def to(self, device: str) -> _LoadedModel:
            assert device == "cuda:0"
            return self

        def eval(self) -> None:
            return None

    class _Processor:
        @classmethod
        def from_pretrained(cls, source: object, **kwargs: object) -> _Processor:
            processor_calls.append((source, kwargs))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **kwargs: (
                snapshot_calls.append(kwargs) or "/private/cache/qwen"
            )
        ),
    )
    monkeypatch.setattr(qwen_model, "_embedding_model_class", lambda: _LoadedModel)
    monkeypatch.setattr(qwen_model, "_processor_class", lambda: _Processor)
    monkeypatch.setattr(qwen_model, "_require_qwen_vl_utils", lambda: object())

    load_qwen_embedding_adapter(
        "Qwen/Qwen3-VL-Embedding-2B",
        DEFAULT_MODEL_REVISION,
        "cuda:0",
        "bfloat16",
    )

    assert snapshot_calls == [
        {
            "repo_id": "Qwen/Qwen3-VL-Embedding-2B",
            "revision": DEFAULT_MODEL_REVISION,
        }
    ]
    assert model_calls[0][0] == "/private/cache/qwen"
    assert "trust_remote_code" not in model_calls[0][1]
    assert model_calls[0][1]["output_loading_info"] is True
    assert processor_calls == [("/private/cache/qwen", {"padding_side": "right"})]


def test_loader_checks_vision_runtime_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from postprism.multimodal import qwen_model

    monkeypatch.setattr(
        qwen_model,
        "_require_qwen_vl_utils",
        lambda: (_ for _ in ()).throw(ImportError("qwen-vl-utils missing")),
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **_kwargs: pytest.fail(
                "model download must not start before dependency validation"
            )
        ),
    )

    with pytest.raises(ImportError, match="qwen-vl-utils missing"):
        load_qwen_embedding_adapter(
            "Qwen/Qwen3-VL-Embedding-2B",
            DEFAULT_MODEL_REVISION,
            "cuda:0",
            "bfloat16",
        )


def test_embedding_model_exposes_a_loadable_transformers_config_class() -> None:
    from postprism.multimodal import qwen_model

    config = qwen_model.Qwen3VLForEmbedding.config_class.from_dict({})

    assert isinstance(config, qwen_model.Qwen3VLConfig)


@pytest.mark.parametrize(
    "loading_field",
    ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"],
)
def test_loader_rejects_incomplete_checkpoint_weights(
    loading_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from postprism.multimodal import qwen_model

    class _LoadedModel:
        @classmethod
        def from_pretrained(
            cls, _source: object, **_kwargs: object
        ) -> tuple[_LoadedModel, dict[str, list[str]]]:
            loading_info = {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
            loading_info[loading_field] = ["model.visual.patch_embed.proj.weight"]
            return cls(), loading_info

        def to(self, _device: str) -> _LoadedModel:
            return self

        def eval(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: "/cache/qwen"),
    )
    monkeypatch.setattr(qwen_model, "_embedding_model_class", lambda: _LoadedModel)
    monkeypatch.setattr(qwen_model, "_require_qwen_vl_utils", lambda: object())
    monkeypatch.setattr(
        qwen_model,
        "_processor_class",
        lambda: pytest.fail("processor must not load after incomplete weights"),
    )

    with pytest.raises(RuntimeError, match="incomplete Qwen checkpoint load"):
        load_qwen_embedding_adapter(
            "Qwen/Qwen3-VL-Embedding-2B",
            DEFAULT_MODEL_REVISION,
            "cuda:0",
            "bfloat16",
        )


def test_embed_reports_video_decode_and_qwen_processing_timings() -> None:
    adapter = _FakeAdapter()
    worker = MultimodalEmbedder(adapter=adapter)
    upload = UploadPayload(
        filename="slide.png",
        suffix=".png",
        media_type="image",
        content=FIXTURE.read_bytes(),
    )
    bundle = build_post_media_bundle((upload,))
    timings: dict[str, float] = {}

    with materialized_post_media(bundle) as paths:
        result = worker.embed(bundle, paths, "text", None, timings)

    assert result.multimodal_embedding_status == "ok"
    assert timings["video_decode"] == 0.0  # image-only post decoded no video
    assert timings["qwen_processing"] >= 0.0
    assert set(timings) == {"video_decode", "qwen_processing"}
