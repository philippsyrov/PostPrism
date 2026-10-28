from __future__ import annotations

import sys
import time
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from postprism.media import (
    UploadPayload,
    build_post_media_bundle,
    materialized_post_media,
)
from postprism.multimodal import sampling
from postprism.multimodal.sampling import (
    FOUR_FRAME_FRACTIONS,
    TWO_FRAME_FRACTIONS,
    VIDEO_FRAME_FRACTIONS,
    DecodedVideoFrame,
    PreparedVisual,
    VisualInputEvidence,
    VisualPreparationError,
    frame_fractions,
    prepare_ordered_visuals,
    sample_video_frames,
    video_frame_allocation,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "generated"


def _image_upload(position: int) -> UploadPayload:
    source = FIXTURE_DIR / ("carousel_green.png" if position % 2 == 0 else "carousel_blue.png")
    return UploadPayload(
        filename=f"slide-{position}.png",
        suffix=".png",
        media_type="image",
        content=source.read_bytes() + bytes([position]),
    )


def _valid_image_upload(position: int) -> UploadPayload:
    source = FIXTURE_DIR / ("carousel_green.png" if position % 2 == 0 else "carousel_blue.png")
    return UploadPayload(
        filename=f"slide-{position}.png",
        suffix=".png",
        media_type="image",
        content=source.read_bytes(),
    )


def _video_upload(name: str = "clip") -> UploadPayload:
    return UploadPayload(
        filename=f"{name}.mp4",
        suffix=".mp4",
        media_type="video",
        content=f"{name} video bytes".encode(),
    )


def _fake_frames(path: Path, fractions: tuple[float, ...]) -> tuple[PreparedVisual, ...]:
    return tuple(
        PreparedVisual(
            image=Image.new("RGB", (2, 2)),
            evidence=VisualInputEvidence(
                source_position=0,
                kind="video_frame",
                fraction=fraction,
                seconds=fraction * 10.0,
                frame_index=index,
            ),
        )
        for index, fraction in enumerate(fractions)
    )


def _decoded(index: int, fraction: float, seconds: float) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        image=Image.new("RGB", (2, 2), color=(index, 0, 0)),
        fraction=fraction,
        seconds=seconds,
        frame_index=index,
    )


def test_video_targets_use_the_measured_eight_frame_policy() -> None:
    assert VIDEO_FRAME_FRACTIONS == (
        0.10,
        0.207143,
        0.314286,
        0.421429,
        0.528571,
        0.635714,
        0.742857,
        0.85,
    )


@pytest.mark.parametrize(
    ("video_count", "allocation"),
    [
        (0, ()),
        (1, (8,)),
        (2, (4, 4)),
        (3, (4, 2, 2)),
        (4, (2, 2, 2, 2)),
        (8, (2,) * 8),
        (10, (2,) * 10),
    ],
)
def test_every_distinct_video_receives_at_least_one_temporal_pair(
    video_count: int,
    allocation: tuple[int, ...],
) -> None:
    assert video_frame_allocation(video_count) == allocation
    assert all(count % 2 == 0 and count >= 2 for count in allocation)


def test_frame_fractions_are_pinned_literals() -> None:
    assert frame_fractions(8) is VIDEO_FRAME_FRACTIONS
    assert frame_fractions(4) == FOUR_FRAME_FRACTIONS == (0.10, 0.35, 0.60, 0.85)
    assert frame_fractions(2) == TWO_FRAME_FRACTIONS == (0.10, 0.85)
    with pytest.raises(ValueError, match="no pinned frame policy"):
        frame_fractions(3)


def test_single_video_default_still_decodes_the_legacy_eight_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[float, ...]] = []

    def capture(_path: Path, fractions: tuple[float, ...]) -> tuple[DecodedVideoFrame, ...]:
        requested.append(fractions)
        return ()

    monkeypatch.setattr(sampling, "_decode_targets", capture)

    sample_video_frames(Path("clip.mp4"))
    sample_video_frames(Path("clip.mp4"), TWO_FRAME_FRACTIONS)

    assert requested == [VIDEO_FRAME_FRACTIONS, TWO_FRAME_FRACTIONS]
    assert requested[0] is VIDEO_FRAME_FRACTIONS


def test_short_video_deduplicates_decoded_frames_without_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sampling,
        "_decode_targets",
        lambda *_: (
            _decoded(0, 0.10, 0.0),
            _decoded(0, 0.35, 0.0),
            _decoded(2, 0.60, 0.2),
            _decoded(2, 0.85, 0.2),
        ),
    )

    prepared = sample_video_frames(Path("short.mp4"))

    assert [item.evidence.frame_index for item in prepared] == [0, 2]
    assert [item.evidence.fraction for item in prepared] == [0.10, 0.60]


def test_video_stream_dimensions_are_rejected_before_frame_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_called = False
    stream = types.SimpleNamespace(width=2, height=2)

    class Container:
        streams = types.SimpleNamespace(video=[stream])

        def __enter__(self) -> Container:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def decode(self, _stream: object):
            nonlocal decode_called
            decode_called = True
            return iter(())

    fake_av = types.SimpleNamespace(open=lambda _path: Container(), time_base=1_000_000)
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(sampling, "MAX_IMAGE_PIXELS", 3)

    with pytest.raises(ValueError, match="video has too many pixels"):
        sampling._decode_targets(Path("huge.mp4"))

    assert decode_called is False


def test_decoded_video_frame_enforces_the_shared_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampling, "MAX_IMAGE_PIXELS", 3)
    monkeypatch.setattr(sampling, "_decode_targets", lambda *_: (_decoded(0, 0.1, 0.0),))

    with pytest.raises(VisualPreparationError, match="video frame has too many pixels"):
        sample_video_frames(Path("huge.mp4"))


def test_real_pyav_decode_returns_ordered_distinct_target_frames(
    tmp_path: Path,
) -> None:
    av = pytest.importorskip("av")

    path = tmp_path / "sequence.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(20):
            frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), color=(index * 10, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    prepared = sample_video_frames(path)

    assert len(prepared) == 8
    assert all(item.evidence.kind == "video_frame" for item in prepared)
    assert [item.evidence.frame_index for item in prepared] == sorted(
        item.evidence.frame_index for item in prepared
    )
    assert len({item.evidence.frame_index for item in prepared}) == 8

    paired = sample_video_frames(path, TWO_FRAME_FRACTIONS)

    assert [item.evidence.fraction for item in paired] == [0.10, 0.85]
    assert [item.evidence.frame_index for item in paired] == [
        prepared[0].evidence.frame_index,
        prepared[-1].evidence.frame_index,
    ]


@pytest.mark.parametrize("count", [2, 3, 4, 5, 10])
def test_carousel_keeps_each_distinct_slide_once_in_order(count: int) -> None:
    uploads = tuple(_image_upload(position) for position in range(count))
    bundle = build_post_media_bundle(uploads)

    with materialized_post_media(bundle) as paths:
        prepared = prepare_ordered_visuals(bundle, paths)

    assert [item.evidence.source_position for item in prepared] == list(range(count))
    assert all(item.evidence.kind == "image" for item in prepared)


def test_duplicate_slide_keeps_provenance_without_model_weight() -> None:
    first = _valid_image_upload(0)
    second = _valid_image_upload(1)
    bundle = build_post_media_bundle((first, first, second))

    with materialized_post_media(bundle) as paths:
        prepared = prepare_ordered_visuals(bundle, paths)

    assert [item.evidence.source_position for item in prepared] == [0, 2]
    assert bundle.items[1].duplicate_of_position == 0


def test_mixed_media_expands_video_frames_at_the_video_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sampling,
        "sample_video_frames",
        lambda _path, _fractions: (
            PreparedVisual(
                image=Image.new("RGB", (2, 2)),
                evidence=VisualInputEvidence(
                    source_position=0,
                    kind="video_frame",
                    fraction=0.10,
                    seconds=1.0,
                    frame_index=1,
                ),
            ),
            PreparedVisual(
                image=Image.new("RGB", (2, 2)),
                evidence=VisualInputEvidence(
                    source_position=0,
                    kind="video_frame",
                    fraction=0.60,
                    seconds=6.0,
                    frame_index=6,
                ),
            ),
        ),
    )
    bundle = build_post_media_bundle(
        (_valid_image_upload(0), _video_upload(), _valid_image_upload(1))
    )

    with materialized_post_media(bundle) as paths:
        prepared = prepare_ordered_visuals(bundle, paths)

    assert [item.evidence.source_position for item in prepared] == [0, 1, 1, 2]
    assert [item.evidence.kind for item in prepared] == [
        "image",
        "video_frame",
        "video_frame",
        "image",
    ]


def test_single_video_in_a_carousel_keeps_the_legacy_eight_fractions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[float, ...]] = []

    def capture(path: Path, fractions: tuple[float, ...]) -> tuple[PreparedVisual, ...]:
        requested.append(fractions)
        return _fake_frames(path, fractions)

    monkeypatch.setattr(sampling, "sample_video_frames", capture)
    bundle = build_post_media_bundle(
        (_valid_image_upload(0), _video_upload(), _valid_image_upload(1))
    )

    with materialized_post_media(bundle) as paths:
        prepare_ordered_visuals(bundle, paths)

    assert requested == [VIDEO_FRAME_FRACTIONS]
    assert requested[0] is VIDEO_FRAME_FRACTIONS


@pytest.mark.parametrize(
    ("video_count", "expected"),
    [
        (2, [FOUR_FRAME_FRACTIONS, FOUR_FRAME_FRACTIONS]),
        (3, [FOUR_FRAME_FRACTIONS, TWO_FRAME_FRACTIONS, TWO_FRAME_FRACTIONS]),
        (4, [TWO_FRAME_FRACTIONS] * 4),
        (10, [TWO_FRAME_FRACTIONS] * 10),
    ],
)
def test_multi_video_posts_allocate_paired_frames_in_media_order(
    monkeypatch: pytest.MonkeyPatch,
    video_count: int,
    expected: list[tuple[float, ...]],
) -> None:
    requested: list[tuple[Path, tuple[float, ...]]] = []

    def capture(path: Path, fractions: tuple[float, ...]) -> tuple[PreparedVisual, ...]:
        requested.append((path, fractions))
        return _fake_frames(path, fractions)

    monkeypatch.setattr(sampling, "sample_video_frames", capture)
    uploads = tuple(_video_upload(f"clip-{index}") for index in range(video_count))
    bundle = build_post_media_bundle(uploads)

    with materialized_post_media(bundle) as paths:
        prepared = prepare_ordered_visuals(bundle, paths)
        assert [path for path, _ in requested] == [paths[p] for p in range(video_count)]

    assert [fractions for _, fractions in requested] == expected
    assert [item.evidence.source_position for item in prepared] == [
        position for position, fractions in enumerate(expected) for _ in fractions
    ]
    assert [item.evidence.fraction for item in prepared] == [
        fraction for fractions in expected for fraction in fractions
    ]


def test_duplicate_video_bytes_count_once_toward_the_frame_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[float, ...]] = []

    def capture(path: Path, fractions: tuple[float, ...]) -> tuple[PreparedVisual, ...]:
        requested.append(fractions)
        return _fake_frames(path, fractions)

    monkeypatch.setattr(sampling, "sample_video_frames", capture)
    first = _video_upload("first")
    bundle = build_post_media_bundle((first, first, _video_upload("second")))

    with materialized_post_media(bundle) as paths:
        prepared = prepare_ordered_visuals(bundle, paths)

    assert requested == [FOUR_FRAME_FRACTIONS, FOUR_FRAME_FRACTIONS]
    assert sorted({item.evidence.source_position for item in prepared}) == [0, 2]


def test_corrupt_media_raises_the_worker_facing_preparation_error() -> None:
    corrupt = UploadPayload(
        filename="broken.png",
        suffix=".png",
        media_type="image",
        content=b"not an image",
    )
    bundle = build_post_media_bundle((corrupt,))

    with (
        materialized_post_media(bundle) as paths,
        pytest.raises(VisualPreparationError, match="position 0"),
    ):
        prepare_ordered_visuals(bundle, paths)


def test_decompression_bomb_raises_the_worker_facing_preparation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_post_media_bundle((_valid_image_upload(0),))
    monkeypatch.setattr(
        sampling.Image,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            Image.DecompressionBombError("too many pixels")
        ),
    )

    with (
        materialized_post_media(bundle) as paths,
        pytest.raises(VisualPreparationError, match="position 0"),
    ):
        prepare_ordered_visuals(bundle, paths)


def test_image_decoder_rejects_content_that_does_not_match_its_extension() -> None:
    jpeg_bytes = Image.new("RGB", (2, 2), color="red")
    encoded = BytesIO()
    jpeg_bytes.save(encoded, format="JPEG")
    disguised = UploadPayload(
        filename="disguised.png",
        suffix=".png",
        media_type="image",
        content=encoded.getvalue(),
    )
    bundle = build_post_media_bundle((disguised,))

    with (
        materialized_post_media(bundle) as paths,
        pytest.raises(VisualPreparationError, match="position 0"),
    ):
        prepare_ordered_visuals(bundle, paths)


def test_image_decoder_enforces_the_shared_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampling, "MAX_IMAGE_PIXELS", 3)
    bundle = build_post_media_bundle((_valid_image_upload(0),))

    with (
        materialized_post_media(bundle) as paths,
        pytest.raises(VisualPreparationError, match="position 0"),
    ):
        prepare_ordered_visuals(bundle, paths)


def _write_offset_video(path: Path, *, frames: int, rate: int, start_seconds: float) -> None:
    """Encode a clip whose first presentation timestamp is `start_seconds`, not 0."""
    av = pytest.importorskip("av")
    from fractions import Fraction  # noqa: PLC0415

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, rate)
        offset = int(round(start_seconds * rate))
        for index in range(frames):
            frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), color=(index * 4, 0, 0)))
            frame.pts = offset + index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# --- Seek-based decoding must reproduce the sequential decoder exactly --------


def _write_gop_video(path: Path, *, frames: int, rate: int = 10, gop: int = 5) -> None:
    """Encode a short mpeg4 clip with small GOPs so seeks land on real keyframes."""
    av = pytest.importorskip("av")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop
        for index in range(frames):
            color = ((index * 9) % 256, (index * 5) % 256, 60)
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color=color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_targets_follow_the_stream_origin_for_non_zero_start_timestamps(tmp_path: Path) -> None:
    # Regression: targets were `duration * fraction` compared with absolute
    # frame times, so a clip starting at 5 s satisfied seven of eight targets
    # with its very first frame and kept only two distinct frames.
    av = pytest.importorskip("av")
    path = tmp_path / "offset.mp4"
    _write_offset_video(path, frames=60, rate=10, start_seconds=5.0)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert float(stream.start_time * stream.time_base) == pytest.approx(5.0)

    prepared = sample_video_frames(path)

    assert len(prepared) == 8
    seconds = [item.evidence.seconds for item in prepared]
    assert seconds == sorted(seconds)
    assert min(seconds) >= 5.0 + 0.10 * 6.0 - 0.2
    assert max(seconds) <= 5.0 + 6.0
    assert max(seconds) - min(seconds) > 3.0  # spread across the clip, not its head
    assert len({item.evidence.frame_index for item in prepared}) == 8


def test_zero_start_video_selection_is_unchanged_by_the_origin_rule(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "zero.mp4"
    _write_offset_video(path, frames=60, rate=10, start_seconds=0.0)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)

    prepared = sample_video_frames(path)

    # First frame at or after duration * fraction, exactly the accepted rule.
    for item, fraction in zip(prepared, VIDEO_FRAME_FRACTIONS, strict=True):
        assert item.evidence.seconds >= duration * fraction
        assert item.evidence.seconds - duration * fraction < 0.1 + 1e-9


def _write_audio_first_video(path: Path, *, frames: int, rate: int = 10, gop: int = 5) -> None:
    """Like `_write_gop_video` but with an audio stream added BEFORE the video one,
    so the video stream's global index is 1 while it is video ordinal 0."""
    av = pytest.importorskip("av")
    import numpy as np  # noqa: PLC0415

    with av.open(str(path), mode="w") as container:
        audio = container.add_stream("aac", rate=16000, layout="mono")
        video = container.add_stream("mpeg4", rate=rate)
        video.width = 32
        video.height = 32
        video.pix_fmt = "yuv420p"
        video.codec_context.gop_size = gop
        assert audio.index == 0 and video.index == 1
        silence = av.AudioFrame.from_ndarray(
            np.zeros((1, 1024), dtype=np.int16), format="s16", layout="mono"
        )
        silence.sample_rate = 16000
        for index in range(frames):
            color = ((index * 9) % 256, (index * 5) % 256, 60)
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color=color))
            for packet in video.encode(frame):
                container.mux(packet)
            silence.pts = index * 1024
            for packet in audio.encode(silence):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def _identity(decoded: tuple[DecodedVideoFrame, ...]) -> list[tuple[float, float, int, str]]:
    return [
        (
            frame.fraction,
            frame.seconds,
            frame.frame_index,
            sampling.prepared_visual_sha256(frame.image),
        )
        for frame in decoded
    ]


@pytest.mark.parametrize(
    "fractions", [VIDEO_FRAME_FRACTIONS, FOUR_FRAME_FRACTIONS, TWO_FRAME_FRACTIONS]
)
def test_seeking_decoder_matches_sequential_timestamps_indices_and_pixels(
    tmp_path: Path, fractions: tuple[float, ...]
) -> None:
    path = tmp_path / "gops.mp4"
    _write_gop_video(path, frames=60)

    sequential = sampling._decode_targets(path, fractions, allow_seek=False)
    seeking = sampling._decode_targets(path, fractions, allow_seek=True)

    assert len(sequential) == len(fractions)
    assert _identity(seeking) == _identity(sequential)
    # The clip spans several GOPs, so this is not a single-keyframe accident.
    assert len({frame.frame_index for frame in sequential}) == len(fractions)


def test_audio_first_container_seeks_and_matches_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the pts table was indexed with the video stream's GLOBAL
    # index into the video-only collection, which raised IndexError (and so a
    # VisualPreparationError) for every audio-first MP4.
    path = tmp_path / "audio_first.mp4"
    _write_audio_first_video(path, frames=60)
    strategies: list[str] = []
    real_seek = sampling._decode_targets_by_seeking

    def seek_spy(*args: object, **kwargs: object) -> object:
        strategies.append("seek")
        return real_seek(*args, **kwargs)

    monkeypatch.setattr(sampling, "_decode_targets_by_seeking", seek_spy)

    sequential = sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=False)
    seeking = sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=True)

    assert strategies == ["seek"]
    assert _identity(seeking) == _identity(sequential)
    assert len(sample_video_frames(path)) == 8


def test_frame_pts_missing_or_ambiguous_in_the_packet_table_falls_back() -> None:
    table = [0, 512, 1024, 1024, 2048]

    assert sampling._presentation_frame_index(512, table) == 1
    with pytest.raises(sampling.SeekDecodeFallback, match="absent"):
        sampling._presentation_frame_index(700, table)
    with pytest.raises(sampling.SeekDecodeFallback, match="ambiguous"):
        sampling._presentation_frame_index(1024, table)
    with pytest.raises(sampling.SeekDecodeFallback, match="absent"):
        sampling._presentation_frame_index(4096, table)
    with pytest.raises(sampling.SeekDecodeFallback):
        sampling._presentation_frame_index(None, table)


def _write_preroll_video(path: Path, *, frames: int, rate: int = 10) -> None:
    """Encode a clip whose first packets carry negative timestamps.

    The MP4 muxer writes an edit list for them, and on demux they come back
    flagged for discard, exactly like the pre-roll a stream-copy trim leaves.
    Which negative-pts frames survive muxing depends on the encoder's keyframe
    pattern, so the colour ramp and offset are pinned to a recipe that keeps
    two pre-roll packets under the pinned PyAV build; the test asserts that
    precondition rather than assuming it.
    """
    av = pytest.importorskip("av")
    from fractions import Fraction  # noqa: PLC0415

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 5
        stream.codec_context.time_base = Fraction(1, rate)
        for index in range(frames):
            color = ((index * 8) % 256, 0, 0)
            frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), color=color))
            frame.pts = index - 5
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_discard_flagged_preroll_forces_the_sequential_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a stream-copy trim keeps pre-roll packets the decoder drops.
    # The packet table counted them, so the seeking decoder reported frame
    # indices shifted by the pre-roll while pixels and timestamps matched, and
    # the uniqueness check could not notice. Such streams decode sequentially.
    av = pytest.importorskip("av")
    path = tmp_path / "preroll.mp4"
    _write_preroll_video(path, frames=40)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packets = [packet for packet in container.demux(stream) if packet.size]
        assert any(packet.is_discard for packet in packets)
    with av.open(str(path)) as container:
        decoded_count = sum(1 for _ in container.decode(container.streams.video[0]))
    assert decoded_count < len(packets)  # the fixture reproduces the hazard

    with pytest.raises(sampling.SeekDecodeFallback, match="discard"):
        sampling._presentation_pts_table(path, 0, av)

    strategies: list[str] = []
    real_sequential = sampling._decode_targets_sequentially

    def sequential_spy(*args: object, **kwargs: object) -> object:
        strategies.append("sequential")
        return real_sequential(*args, **kwargs)

    monkeypatch.setattr(sampling, "_decode_targets_sequentially", sequential_spy)
    expected = _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=False))

    assert _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS)) == expected
    assert strategies == ["sequential", "sequential"]
    assert max(frame.frame_index for frame in sampling._decode_targets(path)) < decoded_count


def test_seeking_decoder_matches_sequential_on_a_clip_shorter_than_its_targets(
    tmp_path: Path,
) -> None:
    # Three frames, eight targets: several targets share a frame and the last
    # targets fall past the final frame, which sequential decoding reuses.
    path = tmp_path / "tiny.mp4"
    _write_gop_video(path, frames=3)

    sequential = sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=False)
    seeking = sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=True)

    assert _identity(seeking) == _identity(sequential)
    # Frames at 0.0, 0.1 and 0.2 s: no target precedes 0.1 s, so two survive.
    assert [item.evidence.frame_index for item in sample_video_frames(path)] == [1, 2]


def test_default_decode_prefers_seeking_and_falls_back_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "gops.mp4"
    _write_gop_video(path, frames=60)
    expected = _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=False))
    strategies: list[str] = []
    real_seek = sampling._decode_targets_by_seeking
    real_sequential = sampling._decode_targets_sequentially

    def seek_spy(*args: object, **kwargs: object) -> object:
        strategies.append("seek")
        return real_seek(*args, **kwargs)

    def sequential_spy(*args: object, **kwargs: object) -> object:
        strategies.append("sequential")
        return real_sequential(*args, **kwargs)

    monkeypatch.setattr(sampling, "_decode_targets_by_seeking", seek_spy)
    monkeypatch.setattr(sampling, "_decode_targets_sequentially", sequential_spy)

    assert _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS)) == expected
    assert strategies == ["seek"]

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise sampling.SeekDecodeFallback("container is not seekable")

    monkeypatch.setattr(sampling, "_decode_targets_by_seeking", refuse)
    strategies.clear()
    with caplog.at_level("INFO", logger="postprism.multimodal.sampling"):
        assert _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS)) == expected

    assert strategies == ["sequential"]
    assert any("decoding sequentially" in record.message for record in caplog.records)


def test_seek_that_lands_after_its_target_falls_back_instead_of_drifting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A broken index could make `seek` land past the target. The seeking
    # decoder must notice (first frame already beyond the target) and fall
    # back rather than return a later frame under the same evidence.
    path = tmp_path / "gops.mp4"
    _write_gop_video(path, frames=60)
    expected = _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS, allow_seek=False))
    import av  # noqa: PLC0415

    real_open = av.open

    class LateSeekingContainer:
        """Proxy over the real container whose seeks land 10 s late."""

        def __init__(self, container: object) -> None:
            self._container = container

        def __getattr__(self, name: str) -> object:
            return getattr(self._container, name)

        def __enter__(self) -> LateSeekingContainer:
            return self

        def __exit__(self, *_args: object) -> None:
            self._container.close()

        def seek(self, offset: int, *, stream: object, **kwargs: object) -> None:
            late = offset + int(10 / stream.time_base)
            self._container.seek(late, stream=stream, **kwargs)

    monkeypatch.setattr(
        av, "open", lambda *args, **kwargs: LateSeekingContainer(real_open(*args, **kwargs))
    )

    assert _identity(sampling._decode_targets(path, VIDEO_FRAME_FRACTIONS)) == expected


def test_prepare_ordered_visuals_accumulates_video_decode_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_sample(path: Path, fractions: tuple[float, ...]) -> tuple[PreparedVisual, ...]:
        time.sleep(0.01)
        return _fake_frames(path, fractions)

    monkeypatch.setattr(sampling, "sample_video_frames", slow_sample)
    bundle = build_post_media_bundle(
        (_valid_image_upload(0), _video_upload("a"), _video_upload("b"))
    )
    timings: dict[str, float] = {}

    with materialized_post_media(bundle) as paths:
        prepare_ordered_visuals(bundle, paths, timings=timings)

    assert timings["video_decode"] >= 20.0  # two videos, milliseconds

    image_only = build_post_media_bundle((_valid_image_upload(0),))
    timings = {}
    with materialized_post_media(image_only) as paths:
        prepare_ordered_visuals(image_only, paths, timings=timings)

    assert timings["video_decode"] == 0.0
