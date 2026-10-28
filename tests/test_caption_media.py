import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from postprism.captioning import media


def test_image_converts_rgb_without_exif_transpose_and_survives_close(tmp_path):
    path = tmp_path / "orientation.jpg"
    original = Image.new("RGB", (12, 7), "red")
    exif = original.getexif()
    exif[274] = 6
    original.save(path, exif=exif)
    with Image.open(path) as source:
        expected = source.convert("RGB").tobytes()
    visual = media.load_caption_visual(path, "image")
    assert visual.image.mode == "RGB"
    assert visual.image.size == (12, 7)
    assert visual.image.tobytes() == expected
    assert visual.seconds is None
    assert visual.seek_fallback_used is False


def test_palette_image_converts_rgb(tmp_path):
    path = tmp_path / "palette.png"
    Image.new("P", (9, 5)).save(path)
    assert media.load_caption_visual(path, "image").image.mode == "RGB"


def test_image_format_mismatch_is_rejected(tmp_path):
    path = tmp_path / "mismatch.png"
    Image.new("RGB", (12, 7), "red").save(path, format="JPEG")
    with pytest.raises(OSError):
        media.load_caption_visual(path, "image")


def test_oversized_image_is_rejected_before_conversion(monkeypatch, tmp_path):
    path = tmp_path / "oversized.png"
    Image.new("RGB", (10, 9), "red").save(path)
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 80)
    conversions = []
    original_convert = Image.Image.convert

    def record_conversion(self, *args, **kwargs):
        conversions.append(self.size)
        return original_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", record_conversion)
    with pytest.raises(ValueError, match="too many pixels"):
        media.load_caption_visual(path, "image")
    assert conversions == []


def test_image_at_shared_pixel_limit_is_accepted(monkeypatch, tmp_path):
    path = tmp_path / "at-limit.png"
    Image.new("RGB", (10, 8), "red").save(path)
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 80)
    visual = media.load_caption_visual(path, "image")
    assert visual.image.size == (10, 8)
    assert visual.image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("size", [(12, 7), (2000, 2000)])
def test_frame_at_or_under_cap_is_unchanged(size):
    frame = Image.new("RGB", size)
    assert media.bound_frame_pixels(frame) is frame


@pytest.mark.parametrize("size", [(3001, 2003), (2003, 3001), (4_000_001, 1), (1, 4_000_001)])
def test_resize_keeps_historical_rounding_and_pixel_cap(size):
    frame = Image.new("RGB", size)
    bounded = media.bound_frame_pixels(frame)
    assert bounded.width * bounded.height <= 4_000_000
    if min(size) > 1:
        assert abs(bounded.width / bounded.height - size[0] / size[1]) < 0.002
        expected_size = (2448, 1633) if size[0] > size[1] else (1633, 2448)
    else:
        expected_size = (4_000_000, 1) if size[0] > size[1] else (1, 4_000_000)
    assert bounded.size == expected_size


@pytest.mark.parametrize(
    "duration,seek_fails,timestamp,expected_seek,expected_fallback",
    [
        (101, False, 3.5, [50], False),
        (None, False, 0.0, [], False),
        (100, True, 0.0, [50], True),
        (100, False, None, [50], False),
    ],
)
def test_video_preserves_first_frame_after_historical_seek(
    monkeypatch, duration, seek_fails, timestamp, expected_seek, expected_fallback
):
    stream = SimpleNamespace(duration=duration)
    events = {"seeks": [], "decoded": 0, "closed": False}

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __init__(self):
            self.source_image = Image.new("RGB", (13, 7), "blue")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events["closed"] = True
            self.source_image.close()

        def seek(self, target, **kwargs):
            assert kwargs == {"stream": stream}
            events["seeks"].append(target)
            if seek_fails:
                raise RuntimeError("private seek error")

        def decode(self, selected_stream):
            assert selected_stream is stream
            events["decoded"] += 1
            yield SimpleNamespace(time=timestamp, to_image=lambda: self.source_image)
            raise AssertionError("must consume only the first decoded frame")

    def open_video(path):
        assert path == "sample.mp4"
        return Container()

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video))
    visual = media.load_caption_visual(Path("sample.mp4"), "video")
    assert events == {"seeks": expected_seek, "decoded": 1, "closed": True}
    assert visual.seconds == timestamp
    assert visual.seek_fallback_used is expected_fallback
    assert visual.image.size == (13, 7)
    assert visual.image.getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize(
    "has_stream,message", [(False, "no video stream"), (True, "no decodable video frame")]
)
def test_empty_video_raises_decode_failure_and_closes(monkeypatch, has_stream, message):
    events = []

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(duration=None)] if has_stream else [])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def decode(self, stream):
            return iter(())

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: Container()))
    with pytest.raises(ValueError, match=message):
        media.load_caption_visual(Path("empty.mp4"), "video")
    assert events == ["closed"]


def test_oversized_video_metadata_is_rejected_before_seek_or_decode(monkeypatch):
    stream = SimpleNamespace(duration=100, width=10, height=10)
    events = []

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def seek(self, *args, **kwargs):
            events.append("seek")

        def decode(self, selected_stream):
            events.append("decode")
            return iter(())

    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 80)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: Container()))
    with pytest.raises(ValueError, match="too many pixels"):
        media.load_caption_visual(Path("oversized.mp4"), "video")
    assert events == ["closed"]


def test_failed_seek_reopens_before_decoding_the_true_head(monkeypatch):
    containers = []

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(duration=100)])
            self.moved = False
            self.closed = False
            self.decode_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def seek(self, target, *, stream):
            assert target == 50
            assert stream is self.streams.video[0]
            self.moved = True
            raise RuntimeError("seek failed after moving the demuxer")

        def decode(self, stream):
            assert stream is self.streams.video[0]
            self.decode_calls += 1
            yield SimpleNamespace(
                time=9.0 if self.moved else 0.25,
                to_image=lambda: Image.new("RGB", (13, 7), "blue" if self.moved else "red"),
            )

    def open_video(path):
        assert path == "sample.mp4"
        if containers:
            assert containers[-1].closed
        container = Container()
        containers.append(container)
        return container

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video))
    visual = media.load_caption_visual(Path("sample.mp4"), "video")
    assert visual.image.getpixel((0, 0)) == (255, 0, 0)
    assert visual.seconds == 0.25
    assert visual.seek_fallback_used is True
    assert len(containers) == 2
    assert [container.decode_calls for container in containers] == [0, 1]
    assert all(container.closed for container in containers)
