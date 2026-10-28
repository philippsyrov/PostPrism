"""Validated uploads and ordered media provenance."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
MAX_POST_MEDIA_ITEMS = 10
MAX_POST_MEDIA_BYTES = MAX_VIDEO_BYTES + (5 * 1024 * 1024)
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
# v2: any image/video mix within the item cap and ordered plural asr_results.
POST_MEDIA_CONTRACT_VERSION = "post-media-v2"

MediaType = Literal["image", "video"]

IMAGE_FORMATS_BY_EXTENSION: dict[str, list[str]] = {
    ".jpg": ["JPEG"],
    ".jpeg": ["JPEG"],
    ".png": ["PNG"],
    ".gif": ["GIF"],
    ".webp": ["WEBP"],
    ".bmp": ["BMP"],
    ".tiff": ["TIFF"],
}


class AsyncUploadFile(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class UploadSpec:
    filename: str
    suffix: str
    media_type: MediaType
    max_upload_bytes: int


@dataclass(frozen=True)
class UploadValidationError(Exception):
    message: str
    supported: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"error": self.message}
        if self.supported is not None:
            payload["supported"] = self.supported
        return payload


@dataclass(frozen=True)
class UploadPayload:
    filename: str
    suffix: str
    media_type: MediaType
    content: bytes


class PostMediaContractError(ValueError):
    """Invalid post-media input rejected before worker execution."""


@dataclass(frozen=True)
class PostMediaItem:
    position: int
    upload: UploadPayload
    sha256: str
    duplicate_of_position: int | None

    @property
    def filename(self) -> str:
        return self.upload.filename


@dataclass(frozen=True)
class PostMediaBundle:
    items: tuple[PostMediaItem, ...]
    distinct_positions: tuple[int, ...]
    total_bytes: int
    contract_version: str = POST_MEDIA_CONTRACT_VERSION


def supported_media_types() -> dict[str, list[str]]:
    return {
        "images": sorted(IMAGE_EXTENSIONS),
        "videos": sorted(VIDEO_EXTENSIONS),
    }


def _upload_spec_for_filename(filename: str | None) -> UploadSpec:
    if not filename:
        raise UploadValidationError("Missing filename")
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_type: MediaType = "image"
        max_upload_bytes = MAX_IMAGE_BYTES
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
        max_upload_bytes = MAX_VIDEO_BYTES
    else:
        raise UploadValidationError(
            f"Unsupported file type: {suffix}",
            supported=supported_media_types(),
        )
    return UploadSpec(filename, suffix, media_type, max_upload_bytes)


async def read_upload_content(file: AsyncUploadFile) -> bytes:
    spec = _upload_spec_for_filename(file.filename)
    chunks: list[bytes] = []
    bytes_read = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        bytes_read += len(chunk)
        if bytes_read > spec.max_upload_bytes:
            raise UploadValidationError(
                f"File too large: more than {spec.max_upload_bytes} bytes "
                f"(max {spec.max_upload_bytes})"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_upload(filename: str | None, content: bytes) -> UploadPayload:
    spec = _upload_spec_for_filename(filename)
    if len(content) > spec.max_upload_bytes:
        raise UploadValidationError(
            f"File too large: {len(content)} bytes (max {spec.max_upload_bytes})"
        )
    return UploadPayload(spec.filename, spec.suffix, spec.media_type, content)


def build_post_media_bundle(uploads: Sequence[UploadPayload]) -> PostMediaBundle:
    if not uploads:
        raise PostMediaContractError("post media requires at least one upload")
    if len(uploads) > MAX_POST_MEDIA_ITEMS:
        raise PostMediaContractError(f"post media accepts at most {MAX_POST_MEDIA_ITEMS} uploads")
    total_bytes = sum(len(upload.content) for upload in uploads)
    if total_bytes > MAX_POST_MEDIA_BYTES:
        raise PostMediaContractError(f"total media bytes exceed {MAX_POST_MEDIA_BYTES}")

    first_position_by_sha: dict[str, int] = {}
    items: list[PostMediaItem] = []
    distinct_positions: list[int] = []
    for position, upload in enumerate(uploads):
        sha256 = hashlib.sha256(upload.content).hexdigest()
        duplicate_of = first_position_by_sha.get(sha256)
        if duplicate_of is None:
            first_position_by_sha[sha256] = position
            distinct_positions.append(position)
        items.append(PostMediaItem(position, upload, sha256, duplicate_of))
    return PostMediaBundle(tuple(items), tuple(distinct_positions), total_bytes)


@contextmanager
def materialized_post_media(bundle: PostMediaBundle) -> Iterator[dict[int, Path]]:
    distinct_paths: dict[int, Path] = {}
    created_paths: list[Path] = []
    try:
        for position in bundle.distinct_positions:
            upload = bundle.items[position].upload
            with tempfile.NamedTemporaryFile(suffix=upload.suffix, delete=False) as temporary:
                path = Path(temporary.name)
                created_paths.append(path)
                temporary.write(upload.content)
            distinct_paths[position] = path
        yield {
            item.position: distinct_paths[
                item.duplicate_of_position
                if item.duplicate_of_position is not None
                else item.position
            ]
            for item in bundle.items
        }
    finally:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("Could not remove temporary media file %s: %s", path, error)
