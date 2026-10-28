"""Historical RGB and midpoint-seek caption inputs with sampled-view provenance."""

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from postprism.media import IMAGE_FORMATS_BY_EXTENSION, MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class CaptionVisual:
    image: Image.Image
    seconds: float | None = None
    seek_fallback_used: bool = False


def bound_frame_pixels(frame: Image.Image, max_pixels: int = 4_000_000) -> Image.Image:
    area = frame.width * frame.height
    if area <= max_pixels:
        return frame
    scale = math.sqrt(max_pixels / area)
    size = (
        max(1, int(frame.width * scale)),
        max(1, int(frame.height * scale)),
    )
    if size[0] * size[1] > max_pixels:
        if frame.width >= frame.height:
            size = (max(1, max_pixels // size[1]), size[1])
        else:
            size = (size[0], max(1, max_pixels // size[0]))
    return frame.resize(size, Image.Resampling.LANCZOS)


def load_caption_visual(path: Path, media_type: str) -> CaptionVisual:
    if media_type == "video":
        return _extract_middle_frame(path)
    with Image.open(path, formats=IMAGE_FORMATS_BY_EXTENSION[path.suffix.lower()]) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"image has too many pixels: {source.width}x{source.height}")
        return CaptionVisual(source.convert("RGB").copy())


def _extract_middle_frame(path: Path) -> CaptionVisual:
    import av

    for seek_fallback_used in (False, True):
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("video container has no video stream")
            stream = container.streams.video[0]
            width = int(getattr(stream, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or 0)
            if width > 0 and height > 0 and width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"video has too many pixels: {width}x{height}")
            if stream.duration and not seek_fallback_used:
                try:
                    container.seek(stream.duration // 2, stream=stream)
                except Exception:
                    # A failed seek can move the demuxer; reopen to guarantee head decode.
                    continue
            # Preserve the first frame decoded from the midpoint seek, which may be a
            # preceding keyframe; changing this would invalidate caption evidence.
            for frame in container.decode(stream):
                return CaptionVisual(
                    frame.to_image().convert("RGB").copy(),
                    seconds=frame.time,
                    seek_fallback_used=seek_fallback_used,
                )
        raise ValueError("no decodable video frame")
