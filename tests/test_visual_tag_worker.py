import io
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from postprism.media import UploadPayload, build_post_media_bundle
from postprism.multimodal.sampling import (
    VIDEO_FRAME_FRACTIONS,
    PreparedVisual,
    VisualInputEvidence,
    prepared_visual_sha256,
)
from postprism.schemas import MetadataIdentity
from postprism.visual_tags import worker as worker_module
from postprism.visual_tags.adapter import RamPrediction
from postprism.visual_tags.worker import VisualTagger


class StubRam:
    identity = MetadataIdentity("RAM", "a" * 40, "test", "float32")

    def __init__(self):
        self.images = []

    def predict(self, image):
        self.images.append(image.copy())
        return RamPrediction(("red", "truck"), ("red", "truck", "dog"), (1.0, 2.0, 99.0))


def image_bytes(*, colour="red", format="PNG", orientation=None):
    image = Image.new("RGB", (12, 7), colour)
    buffer = io.BytesIO()
    kwargs = {}
    if orientation:
        exif = image.getexif()
        exif[274] = orientation
        kwargs["exif"] = exif
    image.save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def bundle_for_files(tmp_path, files):
    uploads = []
    paths = {}
    for index, (name, content) in enumerate(files):
        suffix = Path(name).suffix
        media_type = "video" if suffix == ".mp4" else "image"
        uploads.append(UploadPayload(name, suffix, media_type, content))
        paths[index] = tmp_path / f"{index}{suffix}"
        paths[index].write_bytes(content)
    return build_post_media_bundle(uploads), paths


def test_text_only_never_invokes_ram():
    adapter = StubRam()
    worker = VisualTagger(adapter=adapter)
    result = worker.enrich(None, {}).to_dict()
    assert result["status"] == "skipped_no_media"
    assert adapter.images == []
    assert worker.is_warm
    assert worker.identity == adapter.identity
    assert (worker.model_name, worker.model_revision, worker.model_version) == (
        "RAM",
        "a" * 40,
        "test",
    )


def test_exif_rotation_precedes_rgb_pixels_hash_and_inference(tmp_path):
    content = image_bytes(format="JPEG", orientation=6)
    bundle, paths = bundle_for_files(tmp_path, [("one.jpg", content)])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    with Image.open(io.BytesIO(content)) as source:
        expected = ImageOps.exif_transpose(source).convert("RGB")
    assert result["status"] == "ok"
    assert len(adapter.images) == 1
    assert adapter.images[0].size == (7, 12)
    assert adapter.images[0].tobytes() == expected.tobytes()
    assert result["items"] == [
        {
            "source_position": 0,
            "kind": "image",
            "status": "ok",
            "sha256": prepared_visual_sha256(expected),
            "native_tags": ["red", "truck"],
            "display_tags": ["truck", "red"],
        }
    ]


def test_duplicate_bytes_infer_once_and_preserve_original_position(tmp_path):
    content = image_bytes()
    bundle, paths = bundle_for_files(tmp_path, [("one.png", content), ("copy.png", content)])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, {0: paths[0]}).to_dict()
    assert result["status"] == "ok"
    assert len(adapter.images) == 1
    assert result["items"][1] == {
        "source_position": 1,
        "kind": "duplicate",
        "status": "skipped_duplicate",
        "duplicate_of_position": 0,
    }


def test_corrupt_image_returns_empty_decode_failure_without_private_error(tmp_path, caplog):
    bundle, paths = bundle_for_files(tmp_path, [("private-name.png", b"private-corrupt-pixels")])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"] == [
        {
            "source_position": 0,
            "kind": "image",
            "status": "media_load_failed",
            "native_tags": [],
            "display_tags": [],
        }
    ]
    assert adapter.images == []
    assert "private" not in str(result) + caplog.text
    assert str(paths[0]) not in caplog.text


def test_image_format_mismatch_is_rejected_before_ram_inference(tmp_path):
    content = image_bytes(format="JPEG")
    bundle, paths = bundle_for_files(tmp_path, [("mismatch.png", content)])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"][0]["status"] == "media_load_failed"
    assert adapter.images == []


def test_adapter_exception_is_an_empty_inference_failure(tmp_path, caplog):
    class BrokenRam(StubRam):
        def predict(self, image):
            raise RuntimeError("private inference response")

    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    result = VisualTagger(BrokenRam()).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"][0]["status"] == "inference_failed"
    assert result["items"][0]["native_tags"] == result["items"][0]["display_tags"] == []
    assert "private inference" not in str(result) + caplog.text
    assert "RuntimeError" in caplog.text


def test_bad_first_image_keeps_later_success_and_aggregates_partial(tmp_path):
    bundle, paths = bundle_for_files(tmp_path, [("bad.png", b"bad"), ("good.png", image_bytes())])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "partial"
    assert [(row["source_position"], row["status"]) for row in result["items"]] == [
        (0, "media_load_failed"),
        (1, "ok"),
    ]
    assert len(adapter.images) == 1


def test_native_empty_output_is_success_not_a_failure_or_extra_candidate(tmp_path):
    class EmptyRam(StubRam):
        def predict(self, image):
            return RamPrediction((), ("red", "truck"), (1.0, 99.0))

    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    result = VisualTagger(EmptyRam()).enrich(bundle, paths).to_dict()
    assert result["status"] == result["items"][0]["status"] == "ok"
    assert result["items"][0]["native_tags"] == result["items"][0]["display_tags"] == []


def sampled_frames():
    return tuple(
        PreparedVisual(
            Image.new("RGB", (10, 6), colour),
            VisualInputEvidence(
                0,
                "video_frame",
                fraction=fraction,
                seconds=seconds,
                frame_index=index,
                sha256=prepared_visual_sha256(Image.new("RGB", (10, 6), colour)),
            ),
        )
        for colour, fraction, seconds, index in [("red", 0.1, 1.0, 30), ("blue", 0.85, 8.5, 255)]
    )


def test_video_uses_pinned_fractions_and_returns_each_frame_in_order(tmp_path, monkeypatch):
    frames = sampled_frames()
    calls = []

    def sample(path, fractions):
        calls.append((path, fractions))
        return frames

    monkeypatch.setattr(worker_module, "sample_video_frames", sample)
    bundle, paths = bundle_for_files(tmp_path, [("clip.mp4", b"fake-video")])
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "ok"
    assert calls == [(paths[0], VIDEO_FRAME_FRACTIONS)]
    assert len(adapter.images) == 2
    assert result["items"] == [
        {
            **frame.evidence.to_dict(),
            "status": "ok",
            "native_tags": ["red", "truck"],
            "display_tags": ["truck", "red"],
        }
        for frame in frames
    ]
    assert "native_tags" not in result


def test_each_distinct_video_keeps_eight_targets_and_actual_source_positions(tmp_path, monkeypatch):
    calls = []
    frames = sampled_frames()

    def sample(path, fractions):
        calls.append((path, fractions))
        return frames

    monkeypatch.setattr(worker_module, "sample_video_frames", sample)
    bundle, paths = bundle_for_files(
        tmp_path,
        [
            ("image.png", image_bytes()),
            ("one.mp4", b"one"),
            ("two.mp4", b"two"),
            ("duplicate.mp4", b"one"),
        ],
    )
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert calls == [(paths[1], VIDEO_FRAME_FRACTIONS), (paths[2], VIDEO_FRAME_FRACTIONS)]
    assert [len(fractions) for _, fractions in calls] == [8, 8]
    assert len(adapter.images) == 5
    assert [row["source_position"] for row in result["items"]] == [0, 1, 1, 2, 2, 3]
    assert result["items"][-1] == {
        "source_position": 3,
        "kind": "duplicate",
        "status": "skipped_duplicate",
        "duplicate_of_position": 1,
    }
    assert frames[0].evidence.source_position == 0


def test_video_decoder_exception_returns_one_failure_at_video_position(
    tmp_path, monkeypatch, caplog
):
    def sample(path, fractions):
        raise ValueError("private decoder error")

    monkeypatch.setattr(worker_module, "sample_video_frames", sample)
    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes()), ("video.mp4", b"one")])
    result = VisualTagger(StubRam()).enrich(bundle, paths).to_dict()
    assert result["status"] == "partial"
    assert result["items"][1] == {
        "source_position": 1,
        "kind": "video",
        "status": "media_load_failed",
        "native_tags": [],
        "display_tags": [],
    }
    assert "private decoder" not in str(result) + caplog.text


def test_one_frame_prediction_failure_preserves_other_frame_and_provenance(tmp_path, monkeypatch):
    class OneFailure(StubRam):
        def predict(self, image):
            if image.getpixel((0, 0)) == (255, 0, 0):
                self.images.append(image.copy())
                raise RuntimeError("failed frame")
            return super().predict(image)

    frames = sampled_frames()
    monkeypatch.setattr(worker_module, "sample_video_frames", lambda *a: frames)
    bundle, paths = bundle_for_files(tmp_path, [("video.mp4", b"one")])
    adapter = OneFailure()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "partial"
    assert len(adapter.images) == 2
    assert result["items"][0] == {
        **frames[0].evidence.to_dict(),
        "status": "inference_failed",
        "native_tags": [],
        "display_tags": [],
    }
    assert result["items"][1] == {
        **frames[1].evidence.to_dict(),
        "status": "ok",
        "native_tags": ["red", "truck"],
        "display_tags": ["truck", "red"],
    }


def test_video_with_no_decoded_frames_is_one_decode_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module, "sample_video_frames", lambda *a: ())
    bundle, paths = bundle_for_files(tmp_path, [("empty.mp4", b"one")])
    result = VisualTagger(StubRam()).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"] == [
        {
            "source_position": 0,
            "kind": "video",
            "status": "media_load_failed",
            "native_tags": [],
            "display_tags": [],
        }
    ]


def test_unwarmed_worker_skips_text_but_rejects_media_without_loading(tmp_path):
    worker = VisualTagger()
    assert not worker.is_warm
    assert worker.enrich(None, {}).to_dict()["status"] == "skipped_no_media"
    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    with pytest.raises(RuntimeError, match="warm"):
        worker.enrich(bundle, paths)


def test_warmup_loads_local_artifacts_once_and_checks_inference_before_readiness(
    tmp_path, monkeypatch
):
    calls = []
    adapter = StubRam()
    monkeypatch.setenv("RAM_CHECKPOINT_FILE", str(tmp_path / "ram.pth"))
    monkeypatch.setenv("RAM_TOKENIZER_DIR", str(tmp_path / "tokenizer"))

    def load(checkpoint, tokenizer_dir, device):
        calls.append((checkpoint, tokenizer_dir, device))
        return adapter

    monkeypatch.setattr(worker_module, "load_ram_adapter", load)
    worker = VisualTagger()
    worker.warmup()
    worker.warmup()
    assert worker.is_warm
    assert calls == [(tmp_path / "ram.pth", tmp_path / "tokenizer", "cuda:0")]
    assert len(adapter.images) == 1
    assert adapter.images[0].mode == "RGB"
    assert adapter.images[0].size == (384, 384)


def test_injected_adapter_warmup_never_loads_or_infers():
    adapter = StubRam()
    VisualTagger(adapter).warmup()
    assert adapter.images == []


@pytest.mark.parametrize("failure", [ImportError("missing dependency"), ValueError("integrity")])
def test_load_failure_propagates_and_worker_stays_cold(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("RAM_CHECKPOINT_FILE", str(tmp_path / "ram.pth"))
    monkeypatch.setenv("RAM_TOKENIZER_DIR", str(tmp_path / "tokenizer"))

    def load(*a, **k):
        raise failure

    monkeypatch.setattr(worker_module, "load_ram_adapter", load)
    worker = VisualTagger()
    with pytest.raises(type(failure)):
        worker.warmup()
    assert not worker.is_warm


def test_inference_warmup_failure_does_not_mark_worker_ready(tmp_path, monkeypatch):
    class BrokenRam(StubRam):
        def predict(self, image):
            raise RuntimeError("warmup failed")

    monkeypatch.setenv("RAM_CHECKPOINT_FILE", str(tmp_path / "ram.pth"))
    monkeypatch.setenv("RAM_TOKENIZER_DIR", str(tmp_path / "tokenizer"))
    monkeypatch.setattr(worker_module, "load_ram_adapter", lambda *a, **k: BrokenRam())
    worker = VisualTagger()
    with pytest.raises(RuntimeError, match="warmup"):
        worker.warmup()
    assert not worker.is_warm


@pytest.mark.parametrize("missing", ["RAM_CHECKPOINT_FILE", "RAM_TOKENIZER_DIR"])
def test_missing_operator_artifact_configuration_fails_clearly_before_loading(monkeypatch, missing):
    monkeypatch.setenv("RAM_CHECKPOINT_FILE", "/operator/ram.pth")
    monkeypatch.setenv("RAM_TOKENIZER_DIR", "/operator/tokenizer")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        VisualTagger().warmup()


def test_image_pixel_limit_is_checked_before_preparing_rgb(tmp_path, monkeypatch):
    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    monkeypatch.setattr(worker_module, "MAX_IMAGE_PIXELS", 80)
    adapter = StubRam()
    result = VisualTagger(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"][0]["status"] == "media_load_failed"
    assert adapter.images == []
