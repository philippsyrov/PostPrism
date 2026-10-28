import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from postprism.captioning import worker as worker_module
from postprism.captioning.identity import CAPTION_IDENTITY
from postprism.captioning.media import CaptionVisual
from postprism.captioning.policy import CAPTION_PROMPT, RETRY_PROMPT
from postprism.captioning.worker import Captioner
from postprism.media import UploadPayload, build_post_media_bundle
from postprism.multimodal.sampling import prepared_visual_sha256


class RecordingAdapter:
    identity = CAPTION_IDENTITY

    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = iter(outputs) if outputs is not None else None

    def generate(self, frames, prompt):
        self.calls.append(([image.copy() for image in frames], prompt))
        value = (
            next(self.outputs) if self.outputs else '{"caption":"A truck.","ocr_text":"private"}'
        )
        if isinstance(value, Exception):
            raise value
        return value


def image_bytes(size=(12, 7), colour="red"):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def bundle_for_files(tmp_path, files):
    uploads, paths = [], {}
    for index, (name, content) in enumerate(files):
        suffix = Path(name).suffix
        media_type = "video" if suffix == ".mp4" else "image"
        uploads.append(UploadPayload(name, suffix, media_type, content))
        paths[index] = tmp_path / f"{index}{suffix}"
        paths[index].write_bytes(content)
    return build_post_media_bundle(uploads), paths


def test_text_only_skips_without_loading_or_inference():
    adapter = RecordingAdapter()
    worker = Captioner(adapter)
    result = worker.enrich(None, {}).to_dict()
    assert result["status"] == "skipped_no_media"
    assert result["items"] == []
    assert adapter.calls == []
    assert worker.is_warm
    assert worker.identity == CAPTION_IDENTITY
    assert worker.model_name == CAPTION_IDENTITY.model
    assert worker.model_revision == CAPTION_IDENTITY.model_revision
    assert worker.model_version == CAPTION_IDENTITY.method_version
    assert Captioner().enrich(None, {}).to_dict()["status"] == "skipped_no_media"


def test_source_rgb_hash_precedes_resize_and_only_caption_is_published(tmp_path):
    bundle, paths = bundle_for_files(tmp_path, [("large.png", image_bytes((3001, 2003)))])
    adapter = RecordingAdapter()
    result = Captioner(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "ok"
    assert len(adapter.calls) == 1
    frames, prompt = adapter.calls[0]
    assert len(frames) == 1
    assert frames[0].size == (2448, 1633)
    assert prompt == CAPTION_PROMPT
    expected = Image.new("RGB", (3001, 2003), "red")
    assert result["items"] == [
        {
            "source_position": 0,
            "kind": "image",
            "status": "ok",
            "sha256": prepared_visual_sha256(expected),
            "caption": "A truck.",
        }
    ]
    assert "ocr" not in json.dumps(result)
    assert "private" not in json.dumps(result)


def test_distinct_mixed_media_and_duplicates_stay_in_original_order(tmp_path, monkeypatch):
    image = image_bytes()
    bundle, paths = bundle_for_files(
        tmp_path,
        [
            ("one.png", image),
            ("first.mp4", b"video-one"),
            ("copy.png", image),
            ("second.mp4", b"video-two"),
            ("copy.mp4", b"video-one"),
        ],
    )
    decoded = []

    def load(path, media_type):
        position = int(path.stem)
        decoded.append((position, media_type))
        return CaptionVisual(
            Image.new("RGB", (12, 7)),
            seconds=3.5 if position == 1 else None,
            seek_fallback_used=position == 3,
        )

    monkeypatch.setattr(worker_module, "load_caption_visual", load)
    adapter = RecordingAdapter()
    result = Captioner(adapter).enrich(bundle, {i: paths[i] for i in [0, 1, 3]}).to_dict()
    assert result["status"] == "ok"
    assert decoded == [(0, "image"), (1, "video"), (3, "video")]
    assert len(adapter.calls) == 3
    assert all(len(frames) == 1 and prompt == CAPTION_PROMPT for frames, prompt in adapter.calls)
    assert [item["source_position"] for item in result["items"]] == [0, 1, 2, 3, 4]
    for duplicate, original in [(2, 0), (4, 1)]:
        assert result["items"][duplicate] == {
            "source_position": duplicate,
            "kind": "duplicate",
            "status": "skipped_duplicate",
            "duplicate_of_position": original,
        }
    for position, seconds, fallback in [(1, 3.5, False), (3, None, True)]:
        item = result["items"][position]
        assert item["kind"] == "video"
        assert item["seconds"] == seconds
        assert item["seek_fallback_used"] is fallback
        assert "frame_index" not in item
        assert "fraction" not in item


def test_corrupt_image_fails_only_its_item_without_private_error(tmp_path, caplog):
    bundle, paths = bundle_for_files(
        tmp_path,
        [
            ("private.png", b"private-corrupt-pixels"),
            ("good.png", image_bytes()),
        ],
    )
    adapter = RecordingAdapter()
    result = Captioner(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "partial"
    assert result["items"][0] == {
        "source_position": 0,
        "kind": "image",
        "status": "media_load_failed",
        "caption": "",
    }
    assert result["items"][1]["status"] == "ok"
    assert len(adapter.calls) == 1
    assert "private" not in str(result) + caplog.text
    assert str(paths[0]) not in caplog.text
    assert "UnidentifiedImageError" in caplog.text


def test_unwarmed_media_request_raises_without_loading(tmp_path):
    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    with pytest.raises(RuntimeError, match="warm"):
        Captioner().enrich(bundle, paths)


def test_warmup_loads_and_generates_once_then_is_idempotent(monkeypatch):
    loads = []
    adapter = RecordingAdapter()

    def load():
        loads.append("load")
        return adapter

    monkeypatch.setattr(worker_module, "load_caption_adapter", load, raising=False)
    worker = Captioner()
    assert not worker.is_warm
    worker.warmup()
    worker.warmup()
    assert worker.is_warm
    assert loads == ["load"]
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1] == CAPTION_PROMPT
    Captioner(adapter).warmup()
    assert len(adapter.calls) == 1


@pytest.mark.parametrize("mode", ["load_exception", "generate_exception", "invalid_twice"])
def test_failed_warmup_remains_unavailable(monkeypatch, mode):
    adapter = RecordingAdapter(
        [RuntimeError("private")] if mode == "generate_exception" else ["bad", "bad"]
    )

    def load():
        if mode == "load_exception":
            raise ImportError("missing backend")
        return adapter

    monkeypatch.setattr(worker_module, "load_caption_adapter", load, raising=False)
    worker = Captioner()
    with pytest.raises((ImportError, RuntimeError)):
        worker.warmup()
    assert not worker.is_warm


@pytest.mark.parametrize(
    "failed_outputs",
    [
        ["private invalid output", "still private invalid output"],
        [RuntimeError("private model failure")],
    ],
)
def test_inference_failure_retains_other_items_and_hides_private_output(
    tmp_path, caplog, failed_outputs
):
    bundle, paths = bundle_for_files(
        tmp_path,
        [
            ("one.png", image_bytes()),
            ("two.png", image_bytes(colour="blue")),
        ],
    )
    adapter = RecordingAdapter([*failed_outputs, '{"caption":"good"}'])
    result = Captioner(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "partial"
    assert result["items"][0]["status"] == "inference_failed"
    assert result["items"][0]["caption"] == ""
    assert result["items"][0]["sha256"] == prepared_visual_sha256(Image.new("RGB", (12, 7), "red"))
    assert result["items"][1]["caption"] == "good"
    assert [prompt for _, prompt in adapter.calls] == (
        [CAPTION_PROMPT, RETRY_PROMPT, CAPTION_PROMPT]
        if len(failed_outputs) == 2
        else [CAPTION_PROMPT, CAPTION_PROMPT]
    )
    assert "private" not in str(result) + caplog.text


def test_inference_exception_logs_type_without_private_details(tmp_path, caplog):
    bundle, paths = bundle_for_files(tmp_path, [("one.png", image_bytes())])
    adapter = RecordingAdapter([RuntimeError("private model failure")])
    result = Captioner(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"][0]["status"] == "inference_failed"
    assert "RuntimeError" in caplog.text
    assert "private model failure" not in caplog.text
    assert str(paths[0]) not in caplog.text


def test_no_decoded_video_frame_is_one_media_failure_without_inference(tmp_path, monkeypatch):
    class EmptyVideo:
        streams = SimpleNamespace(video=[SimpleNamespace(duration=None)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            return iter(())

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: EmptyVideo()))
    bundle, paths = bundle_for_files(tmp_path, [("empty.mp4", b"empty-video")])
    adapter = RecordingAdapter()
    result = Captioner(adapter).enrich(bundle, paths).to_dict()
    assert result["status"] == "failed"
    assert result["items"] == [
        {
            "source_position": 0,
            "kind": "video",
            "status": "media_load_failed",
            "caption": "",
        }
    ]
    assert adapter.calls == []
