from __future__ import annotations

import logging
from pathlib import Path

import pytest

from postprism import media
from postprism.media import (
    PostMediaContractError,
    UploadPayload,
    build_post_media_bundle,
    materialized_post_media,
)


def _upload(filename: str, content: bytes, media_type: str = "image") -> UploadPayload:
    return UploadPayload(
        filename=filename,
        suffix=".mp4" if media_type == "video" else ".jpg",
        media_type=media_type,
        content=content,
    )


def test_bundle_preserves_order_hashes_and_exact_duplicates() -> None:
    bundle = build_post_media_bundle(
        (
            _upload("one.jpg", b"one"),
            _upload("two.jpg", b"two"),
            _upload("again.jpg", b"one"),
        )
    )
    assert [item.filename for item in bundle.items] == [
        "one.jpg",
        "two.jpg",
        "again.jpg",
    ]
    assert bundle.items[2].duplicate_of_position == 0
    assert bundle.distinct_positions == (0, 1)
    assert bundle.items[0].sha256 == bundle.items[2].sha256


@pytest.mark.parametrize("video_count", [1, 2, 4, 10])
def test_bundle_accepts_any_video_mix_within_the_item_cap(video_count: int) -> None:
    uploads = tuple(
        _upload(f"clip-{index}.mp4", f"video-{index}".encode(), "video")
        for index in range(video_count)
    )
    if video_count < 10:
        uploads = (_upload("one.jpg", b"one"), *uploads)

    bundle = build_post_media_bundle(uploads)

    assert sum(item.upload.media_type == "video" for item in bundle.items) == video_count
    assert bundle.distinct_positions == tuple(range(len(uploads)))
    assert bundle.contract_version == "post-media-v2"


def test_bundle_rejects_empty_too_many_or_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(PostMediaContractError, match="at least one"):
        build_post_media_bundle(())
    with pytest.raises(PostMediaContractError, match="at most 10"):
        build_post_media_bundle(tuple(_upload(f"{index}.jpg", b"x") for index in range(11)))

    monkeypatch.setattr(media, "MAX_POST_MEDIA_BYTES", 5)
    with pytest.raises(PostMediaContractError, match="total media bytes"):
        build_post_media_bundle((_upload("one.jpg", b"123"), _upload("two.jpg", b"456")))


def test_materialized_media_shares_duplicates_and_always_cleans_up() -> None:
    bundle = build_post_media_bundle((_upload("one.jpg", b"one"), _upload("again.jpg", b"one")))
    yielded: tuple[Path, ...]

    with materialized_post_media(bundle) as paths:
        yielded = tuple(paths.values())
        assert paths[1] == paths[0]
        assert paths[0].read_bytes() == b"one"

    assert all(not path.exists() for path in yielded)


def test_materialized_media_cleans_up_after_consumer_failure() -> None:
    bundle = build_post_media_bundle((_upload("one.jpg", b"one"),))
    yielded: Path | None = None

    with pytest.raises(RuntimeError), materialized_post_media(bundle) as paths:
        yielded = paths[0]
        raise RuntimeError("worker failed")

    assert yielded is not None and not yielded.exists()


def test_materialized_media_cleanup_failure_does_not_replace_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bundle = build_post_media_bundle((_upload("one.jpg", b"one"),))
    original_unlink = Path.unlink
    yielded: Path | None = None

    def fail_unlink(path: Path, *, missing_ok: bool = False) -> None:
        raise OSError("cleanup denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with caplog.at_level(logging.WARNING), materialized_post_media(bundle) as paths:
        yielded = paths[0]
        assert yielded.read_bytes() == b"one"

    assert "Could not remove temporary media file" in caplog.text
    assert yielded is not None
    original_unlink(yielded, missing_ok=True)
