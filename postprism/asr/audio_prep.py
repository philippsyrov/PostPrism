"""Normalize media audio to capped 16 kHz mono PCM for Qwen ASR.

PyAV performs one structured decode; Qwen receives a plain WAV. Temporary WAVs
are always unlinked. Missing streams and decode failures stay distinct so the
transcriber can map them to its stable statuses.

"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: int = 16000

# Folded into the Qwen `asr_model_version` (qwen-v5): the decode/resample
# method affects the audio Qwen hears, so rows quarantine if it changes.
AUDIO_PREP_METHOD_VERSION: str = "pyav-16k-mono-v1"


class AudioDecodeError(Exception):
    """The media container could not be opened or its audio decoded."""


class NoAudioStreamError(AudioDecodeError):
    """The container decoded fine but carries no audio stream."""


def ensure_audio_prep_available() -> None:
    """Raise `RuntimeError` if PyAV is missing. Called from `warmup()` so a
    misconfigured deploy fails loudly at startup, never on the request path."""
    try:
        import av  # noqa: F401, PLC0415
    except ImportError as err:
        raise RuntimeError(
            "The Qwen ASR VAD pre-gate needs PyAV (`av`) for audio decode. "
            "Install requirements.txt."
        ) from err


def _frame_to_mono_int16(frame_ndarray: np.ndarray) -> np.ndarray:
    """Flatten a resampled s16 frame array to 1-D int16 samples."""
    return np.asarray(frame_ndarray, dtype=np.int16).reshape(-1)


def _resample_audio(resampler, frame):
    try:
        return resampler.resample(frame)
    except ValueError as err:
        # PyAV uses ValueError when decoded frames change rate, layout, or format.
        raise AudioDecodeError(f"audio resample failed: {err}") from err


def decode_audio_16k_mono(
    media_path: str | Path,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Decode a media file's audio track to int16 mono samples at 16 kHz.

    `max_seconds` bounds decode work (the transcriber passes the raw audio
    ceiling): decoding stops once that many output samples exist. Raises
    `NoAudioStreamError` / `AudioDecodeError`; never returns partial garbage.
    """
    import av  # noqa: PLC0415 — optional extra, lazy per module discipline

    max_samples = int(max_seconds * TARGET_SAMPLE_RATE) if max_seconds is not None else None
    try:
        container = av.open(str(media_path))
    except av.error.FFmpegError as err:
        raise AudioDecodeError(f"cannot open media container: {err}") from err
    except (FileNotFoundError, PermissionError, OSError) as err:
        raise AudioDecodeError(f"cannot open media file: {err}") from err

    try:
        if not container.streams.audio:
            raise NoAudioStreamError("media has no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        collected = 0
        try:
            for frame in container.decode(audio=0):
                for resampled in _resample_audio(resampler, frame):
                    chunk = _frame_to_mono_int16(resampled.to_ndarray())
                    chunks.append(chunk)
                    collected += len(chunk)
                if max_samples is not None and collected >= max_samples:
                    break
            else:
                # Only flush the resampler on full decode; a capped break
                # already has more than enough samples.
                for resampled in _resample_audio(resampler, None):
                    chunks.append(_frame_to_mono_int16(resampled.to_ndarray()))
        except av.error.FFmpegError as err:
            raise AudioDecodeError(f"audio decode failed: {err}") from err
    finally:
        container.close()

    samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def write_wav_16k_mono(samples: np.ndarray, wav_path: str | Path) -> None:
    """Write int16 mono samples as a 16 kHz PCM WAV (stdlib `wave`)."""
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


@contextlib.contextmanager
def temporary_wav_path() -> Iterator[Path]:
    """Yield a temp `.wav` path and guarantee it is removed afterwards."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        yield wav_path
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError as err:
            logger.warning("Could not remove temp WAV %s: %s", wav_path, err)


def media_timeline_seconds(media_path: str | Path) -> tuple[float, float]:
    """Return `(origin, elapsed duration)` in the decoder's timestamp clock."""
    import av  # noqa: PLC0415 — optional extra, lazy per module discipline

    try:
        container = av.open(str(media_path))
    except av.error.FFmpegError as err:
        raise AudioDecodeError(f"cannot open media container: {err}") from err
    except (FileNotFoundError, PermissionError, OSError) as err:
        raise AudioDecodeError(f"cannot open media file: {err}") from err
    try:
        origin = max(0.0, float(container.start_time or 0) / float(av.time_base))
        if container.duration is not None and container.duration > 0:
            return origin, float(container.duration) / float(av.time_base)
        for stream in container.streams.audio:
            if stream.duration is not None and stream.duration > 0 and stream.time_base is not None:
                stream_origin = (
                    float(stream.start_time * stream.time_base)
                    if stream.start_time is not None
                    else origin
                )
                # A delayed audio stream's duration excludes its leading gap.
                # Express its end relative to the same origin used for probes.
                duration = stream_origin + float(stream.duration * stream.time_base) - origin
                if duration > 0:
                    return origin, duration
    finally:
        container.close()
    raise AudioDecodeError("media duration is unavailable")


def decode_audio_16k_mono_span(
    media_path: str | Path,
    start_seconds: float,
    seconds: float,
) -> tuple[np.ndarray, float]:
    """Decode `[start_seconds, start_seconds + seconds)` to int16 mono 16 kHz.

    Returns `(samples, first_sample_seconds)`. Seeks the audio stream to the
    packet at or before `start_seconds`, then keeps only audio whose timestamps
    fall inside the window: frames ending before it are skipped, the first
    frame that starts at or past its end stops the decode, and both edges are
    trimmed to the sample. Cost scales with the window rather than the media.
    A stream that begins after the window yields an empty array, and one that
    ends inside it is truncated, never padded, so the returned start is the
    only trustworthy placement of the samples (fail closed: a frame without a
    timestamp raises `AudioDecodeError` instead of guessing).
    """
    import av  # noqa: PLC0415 — optional extra, lazy per module discipline

    if seconds <= 0.0 or start_seconds < 0.0:
        raise ValueError("start_seconds must be non-negative and seconds positive")
    end_seconds = start_seconds + seconds
    try:
        container = av.open(str(media_path))
    except av.error.FFmpegError as err:
        raise AudioDecodeError(f"cannot open media container: {err}") from err
    except (FileNotFoundError, PermissionError, OSError) as err:
        raise AudioDecodeError(f"cannot open media file: {err}") from err

    try:
        if not container.streams.audio:
            raise NoAudioStreamError("media has no audio stream")
        stream = container.streams.audio[0]
        time_base = stream.time_base
        if time_base is None:
            raise AudioDecodeError("audio stream has no time base")
        try:
            container.seek(
                int(start_seconds / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        except av.error.FFmpegError as err:
            raise AudioDecodeError(f"audio seek failed: {err}") from err

        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        first_kept_time: float | None = None
        try:
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None:
                    frame_time = float(frame.pts * time_base)
                if frame_time is None:
                    raise AudioDecodeError("audio frame carries no presentation timestamp")
                frame_end = frame_time + frame.samples / float(frame.sample_rate)
                if frame_end <= start_seconds:
                    continue
                if frame_time >= end_seconds:
                    break
                if first_kept_time is None:
                    first_kept_time = frame_time
                for resampled in _resample_audio(resampler, frame):
                    chunks.append(_frame_to_mono_int16(resampled.to_ndarray()))
            for resampled in _resample_audio(resampler, None):
                chunks.append(_frame_to_mono_int16(resampled.to_ndarray()))
        except av.error.FFmpegError as err:
            raise AudioDecodeError(f"audio decode failed: {err}") from err
    finally:
        container.close()

    if first_kept_time is None:
        return np.empty(0, dtype=np.int16), start_seconds
    samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    # The first kept frame may straddle the window start and the last one its
    # end; cut both to the sample so nothing outside the window is returned.
    actual_start = max(start_seconds, first_kept_time)
    skip = int(round((actual_start - first_kept_time) * TARGET_SAMPLE_RATE))
    keep = int(round((end_seconds - actual_start) * TARGET_SAMPLE_RATE))
    return samples[skip : skip + keep], actual_start
