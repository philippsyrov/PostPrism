from __future__ import annotations

import asyncio

import pytest

from postprism.media import (
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    UploadValidationError,
    read_upload_content,
    validate_upload,
)


class FakeUploadFile:
    def __init__(self, filename: str | None, chunks: list[bytes]) -> None:
        self.filename = filename
        self._chunks = chunks
        self.read_calls: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        return self._chunks.pop(0) if self._chunks else b""


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_suffix"),
    [("cat.png", "image", ".png"), ("clip.mp4", "video", ".mp4")],
)
def test_validate_upload_accepts_supported_media(
    filename: str,
    expected_type: str,
    expected_suffix: str,
) -> None:
    upload = validate_upload(filename, b"bytes")
    assert upload.media_type == expected_type
    assert upload.suffix == expected_suffix


def test_validate_upload_rejects_unknown_extension() -> None:
    with pytest.raises(UploadValidationError, match="Unsupported file type") as error:
        validate_upload("notes.txt", b"hello")
    assert error.value.supported is not None
    assert ".png" in error.value.supported["images"]


@pytest.mark.parametrize(
    ("filename", "limit"),
    [("huge.png", MAX_IMAGE_BYTES), ("huge.mp4", MAX_VIDEO_BYTES)],
)
def test_validate_upload_enforces_per_type_byte_limit(filename: str, limit: int) -> None:
    with pytest.raises(UploadValidationError, match="File too large"):
        validate_upload(filename, b"x" * (limit + 1))


def test_read_upload_content_rejects_extension_before_reading() -> None:
    upload = FakeUploadFile("notes.txt", [b"secret"])
    with pytest.raises(UploadValidationError, match="Unsupported file type"):
        asyncio.run(read_upload_content(upload))
    assert upload.read_calls == []


def test_read_upload_content_stops_at_limit() -> None:
    upload = FakeUploadFile("huge.png", [b"x" * MAX_IMAGE_BYTES, b"y"])
    with pytest.raises(UploadValidationError, match="File too large"):
        asyncio.run(read_upload_content(upload))
    assert len(upload.read_calls) == 2
