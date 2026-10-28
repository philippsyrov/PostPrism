"""Exact ordered visual preparation for the unified embedding worker."""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from postprism.media import (
    IMAGE_FORMATS_BY_EXTENSION,
    MAX_IMAGE_PIXELS,
    PostMediaBundle,
)

logger = logging.getLogger(__name__)

# The accepted single-video policy. These literals are part of the measured
# embedding method; never regenerate them from floating-point arithmetic.
VIDEO_FRAME_FRACTIONS: tuple[float, ...] = (
    0.10,
    0.207143,
    0.314286,
    0.421429,
    0.528571,
    0.635714,
    0.742857,
    0.85,
)
# Qwen consumes frames in temporal pairs, so every allocation is even and every
# distinct video receives at least one pair. Two or three videos split the eight
# single-video frames; five to ten videos carry 10-20 frames, and all real-video
# frames together share one single-video-equivalent visual pixel/token budget
# (see qwen_model.SHARED_VIDEO_PAIR_PIXELS).
SINGLE_VIDEO_FRAME_COUNT = 8
FOUR_FRAME_FRACTIONS: tuple[float, ...] = (0.10, 0.35, 0.60, 0.85)
TWO_FRAME_FRACTIONS: tuple[float, ...] = (0.10, 0.85)
_FRACTIONS_BY_FRAME_COUNT: dict[int, tuple[float, ...]] = {
    8: VIDEO_FRAME_FRACTIONS,
    4: FOUR_FRAME_FRACTIONS,
    2: TWO_FRAME_FRACTIONS,
}


def video_frame_allocation(video_count: int) -> tuple[int, ...]:
    """Return even per-video frame counts in media order: 8, 4+4, 4+2+2, then one pair each."""
    if video_count <= 0:
        return ()
    if video_count == 1:
        return (SINGLE_VIDEO_FRAME_COUNT,)
    if video_count == 2:
        return (4, 4)
    if video_count == 3:
        return (4, 2, 2)
    return (2,) * video_count


def frame_fractions(frame_count: int) -> tuple[float, ...]:
    """Return the pinned proportional targets for one allocated frame count."""
    try:
        return _FRACTIONS_BY_FRAME_COUNT[frame_count]
    except KeyError as err:
        raise ValueError(f"no pinned frame policy for {frame_count} frames") from err


class VisualPreparationError(ValueError):
    """Submitted media could not be decoded into model-ready visuals."""

    def __init__(self, message: str, *, source_position: int | None = None) -> None:
        super().__init__(message)
        self.source_position = source_position


@dataclass(frozen=True)
class VisualInputEvidence:
    source_position: int
    kind: str
    duplicate_of_position: int | None = None
    fraction: float | None = None
    seconds: float | None = None
    frame_index: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class PreparedVisual:
    image: Image.Image
    evidence: VisualInputEvidence


@dataclass(frozen=True)
class DecodedVideoFrame:
    image: Image.Image
    fraction: float
    seconds: float
    frame_index: int


def prepared_visual_sha256(image: Image.Image) -> str:
    """Hash the exact decoded RGB pixels passed to Qwen with their shape."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    width, height = rgb.size
    payload = b"RGB\0" + width.to_bytes(8, "big") + height.to_bytes(8, "big") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _stream_duration_seconds(container: object, stream: object, av_module: object) -> float:
    stream_duration = getattr(stream, "duration", None)
    stream_time_base = getattr(stream, "time_base", None)
    if stream_duration is not None and stream_time_base is not None:
        duration = float(stream_duration * stream_time_base)
        if duration > 0:
            return duration

    container_duration = getattr(container, "duration", None)
    if container_duration is not None:
        duration = float(container_duration) / float(av_module.time_base)
        if duration > 0:
            return duration

    frame_count = int(getattr(stream, "frames", 0) or 0)
    average_rate = getattr(stream, "average_rate", None)
    if frame_count > 0 and average_rate:
        duration = frame_count / float(average_rate)
        if duration > 0:
            return duration

    raise ValueError("video duration is unavailable")


def _stream_origin_seconds(stream: object) -> float | None:
    """First presentation timestamp of the stream, or None when unknown."""
    start_time = getattr(stream, "start_time", None)
    time_base = getattr(stream, "time_base", None)
    if start_time is None or time_base is None:
        return None
    return float(start_time * time_base)


def _targets_from_origin(
    origin: float, duration: float, fractions: tuple[float, ...]
) -> tuple[float, ...]:
    """Pinned fractions as target seconds in the stream's own timestamp coordinates.

    A clip whose timestamps start at 5 s is sampled across 5 s..end, not compared
    against absolute 0 (which kept only its head): the `tsorigin` method element.
    """
    return tuple(origin + duration * fraction for fraction in fractions)


def _frame_seconds(frame: object, stream: object, frame_index: int) -> float:
    frame_time = getattr(frame, "time", None)
    if frame_time is not None:
        return float(frame_time)

    pts = getattr(frame, "pts", None)
    time_base = getattr(stream, "time_base", None)
    if pts is not None and time_base is not None:
        return float(pts * time_base)

    average_rate = getattr(stream, "average_rate", None)
    if average_rate:
        return frame_index / float(average_rate)

    raise ValueError("video frame timestamp is unavailable")


# Below this gap the next target is decoded by continuing forward from the
# current frame instead of seeking: a seek re-decodes from the preceding
# keyframe, which costs more than a few dozen frames on short clips. Either
# route selects the same first-frame-at-or-after-target, so this only affects
# speed. Calibrated on the 2026-09-04 acceptance videos (phone MP4s, ~1-2 s GOPs).
SEEK_MIN_GAP_SECONDS: float = 3.0


class SeekDecodeFallback(Exception):
    """Seeking cannot reproduce the sequential decode here; decode sequentially."""


def _open_video_targets(
    container: object,
    av_module: object,
) -> tuple[object, float, float | None]:
    """Validate the video stream and return it with its duration and origin.

    The origin (first presentation timestamp) is None when the container does
    not declare one; the sequential decoder then takes it from the first frame.
    """
    if not container.streams.video:
        raise ValueError("video container has no video stream")
    stream = container.streams.video[0]
    width = int(getattr(stream, "width", 0) or 0)
    height = int(getattr(stream, "height", 0) or 0)
    if width > 0 and height > 0 and width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"video has too many pixels: {width * height} (max {MAX_IMAGE_PIXELS})")
    duration = _stream_duration_seconds(container, stream, av_module)
    return stream, duration, _stream_origin_seconds(stream)


def _decode_targets_sequentially(
    container: object,
    stream: object,
    duration: float,
    origin: float | None,
    fractions: tuple[float, ...],
) -> tuple[DecodedVideoFrame, ...]:
    """Decode once from the start and select the first frame at or after each target.

    This is the accepted method; the seeking decoder must reproduce it exactly.
    """
    targets: tuple[float, ...] = ()
    selected: list[DecodedVideoFrame] = []
    next_target = 0
    last_frame: tuple[object, float, int] | None = None

    for frame_index, frame in enumerate(container.decode(stream)):
        seconds = _frame_seconds(frame, stream, frame_index)
        if last_frame is None:
            targets = _targets_from_origin(
                seconds if origin is None else origin, duration, fractions
            )
        last_frame = (frame, seconds, frame_index)
        image: Image.Image | None = None
        while next_target < len(targets) and seconds >= targets[next_target]:
            if image is None:
                image = frame.to_image().convert("RGB")
            selected.append(
                DecodedVideoFrame(
                    image=image,
                    fraction=fractions[next_target],
                    seconds=seconds,
                    frame_index=frame_index,
                )
            )
            next_target += 1
        if next_target == len(targets):
            break

    if last_frame is None:
        raise ValueError("video contains no decodable frames")

    frame, seconds, frame_index = last_frame
    image = frame.to_image().convert("RGB")
    while next_target < len(targets):
        selected.append(
            DecodedVideoFrame(
                image=image,
                fraction=fractions[next_target],
                seconds=seconds,
                frame_index=frame_index,
            )
        )
        next_target += 1
    return tuple(selected)


def _presentation_pts_table(path: Path, stream_index: int, av_module: object) -> list[int]:
    """Sorted presentation timestamps of every packet in the video stream.

    Demuxing reads packets without decoding them, so this costs I/O only. It
    lets the seeking decoder report the same presentation-order `frame_index`
    the sequential decoder counts, exactly, for variable-frame-rate media too.

    A stream-copy trim (`ffmpeg -ss ... -c copy`) keeps the pre-roll packets
    before the cut, flagged for discard with negative timestamps; the decoder
    drops those frames, so counting them would shift every index. Rather than
    model the decoder's dropping rule, any such packet forces the sequential
    fallback, which is exact by construction and only costs speed on trimmed
    clips.
    """
    pts_values: list[int] = []
    with av_module.open(str(path)) as container:
        # `stream_index` is the container-global index (an audio-first MP4 has
        # its video stream at 1), so look it up in the global collection.
        stream = container.streams[stream_index]
        for packet in container.demux(stream):
            if packet.pts is None:
                if packet.size == 0:
                    continue  # the flush packet at end of stream carries no frame
                raise SeekDecodeFallback("packet without presentation timestamp")
            if getattr(packet, "is_discard", False):
                raise SeekDecodeFallback("stream carries discard-flagged pre-roll packets")
            if packet.pts < 0:
                raise SeekDecodeFallback("stream carries packets with negative timestamps")
            pts_values.append(int(packet.pts))
    pts_values.sort()
    return pts_values


def _presentation_frame_index(pts: int | None, pts_table: list[int]) -> int:
    """Index of the frame with presentation timestamp `pts` in display order.

    The timestamp must appear exactly once in the packet table; a missing or
    duplicated timestamp means the table cannot name this frame, so the caller
    falls back to the sequential decoder rather than guess.
    """
    if pts is None:
        raise SeekDecodeFallback("frame without presentation timestamp")
    position = bisect_left(pts_table, int(pts))
    if position >= len(pts_table) or pts_table[position] != int(pts):
        raise SeekDecodeFallback("frame timestamp absent from the packet table")
    if position + 1 < len(pts_table) and pts_table[position + 1] == int(pts):
        raise SeekDecodeFallback("frame timestamp ambiguous in the packet table")
    return position


def _is_stream_head(frame: object, stream: object) -> bool:
    """Whether `frame` is the first presentation frame of `stream`."""
    start_time = getattr(stream, "start_time", None)
    pts = getattr(frame, "pts", None)
    return start_time is not None and pts is not None and pts <= start_time


def _decode_targets_by_seeking(
    container: object,
    stream: object,
    duration: float,
    origin: float | None,
    fractions: tuple[float, ...],
    av_module: object,
    path: Path,
) -> tuple[DecodedVideoFrame, ...]:
    """Seek to the keyframe before each target, then decode forward to it.

    Selects the same first-frame-at-or-after-target as the sequential decoder,
    because decoding forward from a preceding keyframe yields every frame up to
    the target in presentation order. Any sign that the seek cannot guarantee
    that (no time base or origin, a seek landing after its target, decoder
    errors) raises `SeekDecodeFallback` so the caller repeats the accepted
    sequential method.
    """
    time_base = getattr(stream, "time_base", None)
    if time_base is None:
        raise SeekDecodeFallback("stream has no time base")
    if origin is None:
        # Targets depend on the first frame's timestamp, which only a decode
        # from the start can establish.
        raise SeekDecodeFallback("stream does not declare its start time")
    targets = _targets_from_origin(origin, duration, fractions)
    pts_table = _presentation_pts_table(path, stream.index, av_module)
    selected: list[DecodedVideoFrame] = []
    current: tuple[Image.Image, float, int] | None = None
    decoder = None  # the forward decode run that produced `current`
    exhausted = False  # that run reached the end of the stream

    for target, fraction in zip(targets, fractions, strict=True):
        if current is not None and (current[1] >= target or exhausted):
            image, seconds, frame_index = current
            selected.append(DecodedVideoFrame(image, fraction, seconds, frame_index))
            continue
        if decoder is None or target - current[1] > SEEK_MIN_GAP_SECONDS:
            try:
                container.seek(
                    int(target / time_base),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            except (av_module.error.FFmpegError, OSError, ValueError) as err:
                raise SeekDecodeFallback(f"seek failed: {err}") from err
            decoder = enumerate(container.decode(stream))
            landed = True
        else:
            landed = False

        found: tuple[object, float] | None = None
        last: tuple[object, float] | None = None
        try:
            for decoded_index, frame in decoder:
                seconds = _frame_seconds(frame, stream, decoded_index)
                if (
                    landed
                    and decoded_index == 0
                    and seconds > target
                    and not _is_stream_head(frame, stream)
                ):
                    # Only the stream's own first frame may legitimately start
                    # past a target; anything else means the seek overshot.
                    raise SeekDecodeFallback("seek landed after its target")
                last = (frame, seconds)
                if seconds >= target:
                    found = last
                    break
        except av_module.error.FFmpegError as err:
            raise SeekDecodeFallback(f"decode after seek failed: {err}") from err
        if found is None:
            # The run ended before the target: reuse its final frame, exactly as
            # the sequential decoder reuses the last frame of the stream.
            exhausted = True
            if last is not None:
                found = last
            elif current is not None:
                image, seconds, frame_index = current
                selected.append(DecodedVideoFrame(image, fraction, seconds, frame_index))
                continue
            else:
                raise SeekDecodeFallback("no frames decoded after seek")
        frame, seconds = found
        image = frame.to_image().convert("RGB")
        current = (image, seconds, _presentation_frame_index(frame.pts, pts_table))
        selected.append(DecodedVideoFrame(image, fraction, seconds, current[2]))
    return tuple(selected)


def _decode_targets(
    path: Path,
    fractions: tuple[float, ...] = VIDEO_FRAME_FRACTIONS,
    *,
    allow_seek: bool = True,
) -> tuple[DecodedVideoFrame, ...]:
    """Decode the pinned target frames, seeking when the container allows it."""
    import av  # noqa: PLC0415 -- optional dependency stays off the base import path

    if allow_seek:
        with av.open(str(path)) as container:
            stream, duration, origin = _open_video_targets(container, av)
            try:
                return _decode_targets_by_seeking(
                    container, stream, duration, origin, fractions, av, path
                )
            except SeekDecodeFallback as err:
                logger.info(
                    "Video seek decode unavailable (%s); decoding sequentially: %s", err, path.name
                )

    with av.open(str(path)) as container:
        stream, duration, origin = _open_video_targets(container, av)
        return _decode_targets_sequentially(container, stream, duration, origin, fractions)


def sample_video_frames(
    path: Path,
    fractions: tuple[float, ...] = VIDEO_FRAME_FRACTIONS,
) -> tuple[PreparedVisual, ...]:
    """Return distinct decoded frames without padding the requested quota."""
    prepared: list[PreparedVisual] = []
    seen_frame_indices: set[int] = set()
    try:
        decoded_frames = _decode_targets(path, fractions)
    except Exception as err:  # noqa: BLE001 -- PyAV exposes decoder-specific errors
        raise VisualPreparationError(f"cannot decode video: {path.name}") from err

    for decoded in decoded_frames:
        if decoded.frame_index in seen_frame_indices:
            continue
        width, height = decoded.image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise VisualPreparationError(
                f"video frame has too many pixels: {width * height} (max {MAX_IMAGE_PIXELS})"
            )
        seen_frame_indices.add(decoded.frame_index)
        prepared.append(
            PreparedVisual(
                image=decoded.image,
                evidence=VisualInputEvidence(
                    source_position=0,
                    kind="video_frame",
                    fraction=decoded.fraction,
                    seconds=decoded.seconds,
                    frame_index=decoded.frame_index,
                    sha256=prepared_visual_sha256(decoded.image),
                ),
            )
        )
    return tuple(prepared)


def prepare_ordered_visuals(
    bundle: PostMediaBundle,
    paths: Mapping[int, Path],
    timings: dict[str, float] | None = None,
) -> tuple[PreparedVisual, ...]:
    """Prepare each distinct media item once at its original post position.

    When `timings` is given, `video_decode` accumulates the milliseconds spent
    sampling video frames so slow requests can be attributed to decoding.
    """
    if timings is not None:
        timings["video_decode"] = 0.0
    video_positions = [
        position
        for position in bundle.distinct_positions
        if bundle.items[position].upload.media_type == "video"
    ]
    fractions_by_position = {
        position: frame_fractions(frame_count)
        for position, frame_count in zip(
            video_positions,
            video_frame_allocation(len(video_positions)),
            strict=True,
        )
    }
    prepared: list[PreparedVisual] = []
    for position in bundle.distinct_positions:
        item = bundle.items[position]
        if item.upload.media_type == "video":
            decode_started = time.perf_counter()
            try:
                samples = sample_video_frames(paths[position], fractions_by_position[position])
            except VisualPreparationError as err:
                raise VisualPreparationError(
                    f"cannot decode media at position {position}",
                    source_position=position,
                ) from err
            finally:
                if timings is not None:
                    timings["video_decode"] += (time.perf_counter() - decode_started) * 1000.0
            for sampled in samples:
                prepared.append(
                    replace(
                        sampled,
                        evidence=replace(sampled.evidence, source_position=position),
                    )
                )
            continue

        allowed_formats = IMAGE_FORMATS_BY_EXTENSION[item.upload.suffix]
        path = paths[position]
        try:
            with Image.open(path, formats=allowed_formats) as image:
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"image has too many pixels: {width * height} (max {MAX_IMAGE_PIXELS})"
                    )
                converted = image.convert("RGB")
        except (
            Image.DecompressionBombError,
            UnidentifiedImageError,
            IndexError,
            KeyError,
            OSError,
            SyntaxError,
            ValueError,
            struct.error,
        ) as err:
            # Pillow plugins can raise parser errors during lazy pixel decoding.
            raise VisualPreparationError(
                f"cannot decode media at position {position}",
                source_position=position,
            ) from err
        prepared.append(
            PreparedVisual(
                image=converted,
                evidence=VisualInputEvidence(
                    source_position=position,
                    kind="image",
                    sha256=prepared_visual_sha256(converted),
                ),
            )
        )
    return tuple(prepared)
