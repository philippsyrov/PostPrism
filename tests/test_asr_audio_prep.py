"""Unit tests for the Qwen-path audio normalization (audio_prep.py).

Exercises the REAL PyAV decode against synthesized WAV fixtures (no network,
no model): resample-to-16k-mono correctness, the decode-work cap, the failure
taxonomy, and guaranteed temp-WAV cleanup on success and failure. Skipped
when the optional `av` extra is not installed (default install has no ASR).
"""

from __future__ import annotations

import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

av = pytest.importorskip("av")

from postprism.asr import audio_prep  # noqa: E402 — after importorskip
from postprism.asr.audio_prep import (  # noqa: E402 — after importorskip
    TARGET_SAMPLE_RATE,
    AudioDecodeError,
    decode_audio_16k_mono,
    decode_audio_16k_mono_span,
    temporary_wav_path,
    write_wav_16k_mono,
)


def _write_stereo_wav(path: Path, seconds: float, sample_rate: int = 44100) -> None:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class TestDecodeAudio:
    def test_resamples_stereo_44k_to_16k_mono(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.wav"
        _write_stereo_wav(src, seconds=3.0)
        samples = decode_audio_16k_mono(src)
        assert samples.dtype == np.int16
        assert samples.ndim == 1
        assert len(samples) == pytest.approx(3.0 * TARGET_SAMPLE_RATE, rel=0.01)

    def test_max_seconds_caps_decode_output(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.wav"
        _write_stereo_wav(src, seconds=5.0)
        samples = decode_audio_16k_mono(src, max_seconds=2.0)
        assert len(samples) == int(2.0 * TARGET_SAMPLE_RATE)

    def test_garbage_container_raises_decode_error(self, tmp_path: Path) -> None:
        src = tmp_path / "garbage.mp4"
        src.write_bytes(b"NOT A REAL CONTAINER")
        with pytest.raises(AudioDecodeError):
            decode_audio_16k_mono(src)

    def test_missing_file_raises_decode_error(self, tmp_path: Path) -> None:
        with pytest.raises(AudioDecodeError):
            decode_audio_16k_mono(tmp_path / "missing.mp4")


@pytest.mark.parametrize(
    "decoder,kwargs",
    [
        (decode_audio_16k_mono, {}),
        (decode_audio_16k_mono_span, {"start_seconds": 0.0, "seconds": 1.0}),
    ],
    ids=["full", "span"],
)
class TestResamplerFailures:
    def test_changing_audio_rate_is_a_decode_error(self, monkeypatch, decoder, kwargs):
        frames = []
        for index, rate in enumerate((44100, 22050)):
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 1024), dtype=np.int16), format="s16", layout="mono"
            )
            frame.sample_rate = rate
            frame.time_base = Fraction(1, rate)
            frame.pts = index * 1024
            frames.append(frame)

        class Container:
            streams = SimpleNamespace(audio=[SimpleNamespace(time_base=Fraction(1, 44100))])
            closed = False

            def decode(self, *args, **kwargs):
                return iter(frames)

            def seek(self, *args, **kwargs):
                pass

            def close(self):
                self.closed = True

        container = Container()
        monkeypatch.setattr(av, "open", lambda path: container)
        with pytest.raises(AudioDecodeError, match="resampl") as failure:
            decoder("changing-rate.aac", **kwargs)
        assert isinstance(failure.value.__cause__, ValueError)
        assert "Frame does not match AudioResampler setup" in str(failure.value.__cause__)
        assert container.closed

    def test_resampler_flush_value_error_is_a_decode_error(
        self, monkeypatch, tmp_path, decoder, kwargs
    ):
        path = tmp_path / "clip.wav"
        _write_stereo_wav(path, seconds=0.1)
        original_resampler = av.AudioResampler
        failure = ValueError("invalid buffered audio")

        class Resampler:
            def __init__(self, **kwargs):
                self.delegate = original_resampler(**kwargs)

            def resample(self, frame):
                if frame is None:
                    raise failure
                return self.delegate.resample(frame)

        monkeypatch.setattr(av, "AudioResampler", Resampler)
        with pytest.raises(AudioDecodeError) as raised:
            decoder(path, **kwargs)
        assert raised.value.__cause__ is failure

    @pytest.mark.parametrize("failure_at", ["configuration", "sample_conversion"])
    def test_programming_value_errors_are_not_decode_errors(
        self, monkeypatch, tmp_path, decoder, kwargs, failure_at
    ):
        path = tmp_path / "clip.wav"
        _write_stereo_wav(path, seconds=0.1)
        failure = ValueError("programming bug")

        def fail(*args, **kwargs):
            raise failure

        if failure_at == "configuration":
            monkeypatch.setattr(av, "AudioResampler", fail)
        else:
            monkeypatch.setattr(audio_prep, "_frame_to_mono_int16", fail)
        with pytest.raises(ValueError) as raised:
            decoder(path, **kwargs)
        assert raised.value is failure


class TestWavRoundTrip:
    def test_write_wav_16k_mono_round_trips(self, tmp_path: Path) -> None:
        samples = (
            0.2
            * np.sin(2 * np.pi * 300.0 * np.arange(TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE)
            * 32767
        ).astype(np.int16)
        out = tmp_path / "out.wav"
        write_wav_16k_mono(samples, out)
        with wave.open(str(out)) as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == TARGET_SAMPLE_RATE
            assert wav_file.getsampwidth() == 2
            read_back = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        assert np.array_equal(read_back, samples)


class TestTemporaryWavCleanup:
    def test_removed_on_success(self) -> None:
        with temporary_wav_path() as wav_path:
            wav_path.write_bytes(b"RIFF")
            assert wav_path.exists()
        assert not wav_path.exists()

    def test_removed_on_failure(self) -> None:
        captured: Path | None = None
        with pytest.raises(RuntimeError, match="boom"), temporary_wav_path() as wav_path:
            captured = wav_path
            raise RuntimeError("boom")
        assert captured is not None
        assert not captured.exists()


def _write_mono_wav_with_tone_at(
    path: Path, seconds: float, tone_span: tuple[float, float]
) -> None:
    rate = 16000
    samples = np.zeros(int(seconds * rate), dtype=np.int16)
    start, end = (int(edge * rate) for edge in tone_span)
    t = np.arange(end - start) / rate
    samples[start:end] = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())


def _write_aac_starting_at(path: Path, *, start_seconds: float, seconds: float) -> None:
    """Encode a mono AAC MP4 whose audio timestamps begin at `start_seconds`."""
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415

    rate = 16000
    frame_size = 1024
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    offset = int(round(start_seconds * rate))
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=rate, layout="mono")
        stream.codec_context.time_base = Fraction(1, rate)
        for first in range(0, len(tone) - frame_size + 1, frame_size):
            frame = av.AudioFrame.from_ndarray(
                tone[first : first + frame_size].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = rate
            frame.pts = offset + first
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TestTimelineSpans:
    """Seek-based partial decode for the long-media ASR probes."""

    def test_media_timeline_comes_from_container_metadata(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.wav"
        _write_mono_wav_with_tone_at(src, seconds=60.0, tone_span=(30.0, 40.0))
        assert audio_prep.media_timeline_seconds(src) == pytest.approx((0.0, 60.0), abs=0.05)

    def test_container_origin_is_not_elapsed_duration(self, tmp_path: Path) -> None:
        import av  # noqa: PLC0415

        src = tmp_path / "shifted.mp4"
        _write_aac_starting_at(src, start_seconds=1000.0, seconds=8.0)
        with av.open(str(src)) as container:
            expected_origin = container.start_time / av.time_base
            expected_duration = container.duration / av.time_base
        origin, duration = audio_prep.media_timeline_seconds(src)
        assert origin == expected_origin
        assert duration == expected_duration
        assert origin > 999.0
        assert duration == pytest.approx(8.0, abs=0.1)

    @pytest.mark.parametrize("start_time", [None, -5_000_000, 0])
    def test_missing_or_negative_container_origin_is_clamped_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, start_time: int | None
    ) -> None:
        container = SimpleNamespace(duration=700_000_000, start_time=start_time, close=Mock())
        monkeypatch.setattr("av.open", lambda _path: container)

        assert audio_prep.media_timeline_seconds("metadata-only") == (0.0, 700.0)
        container.close.assert_called_once_with()

    @pytest.mark.parametrize(
        ("container_start", "audio_start", "audio_duration", "expected"),
        [(0, 30, 670, (0.0, 700.0)), (1000, 1030, 670, (1000.0, 700.0))],
    )
    def test_duration_fallback_is_elapsed_from_the_container_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_start: int,
        audio_start: int,
        audio_duration: int,
        expected: tuple[float, float],
    ) -> None:
        from fractions import Fraction  # noqa: PLC0415

        stream = SimpleNamespace(
            duration=audio_duration * 16000,
            start_time=audio_start * 16000,
            time_base=Fraction(1, 16000),
        )
        container = SimpleNamespace(
            duration=None,
            start_time=container_start * 1_000_000,
            streams=SimpleNamespace(audio=[stream]),
            close=Mock(),
        )
        monkeypatch.setattr("av.open", lambda _path: container)

        assert audio_prep.media_timeline_seconds("metadata-only") == expected
        container.close.assert_called_once_with()

    def test_span_decode_returns_exactly_the_requested_window(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.wav"
        _write_mono_wav_with_tone_at(src, seconds=60.0, tone_span=(30.0, 40.0))

        window, start = decode_audio_16k_mono_span(src, start_seconds=28.0, seconds=6.0)

        assert start == pytest.approx(28.0, abs=0.01)
        assert window.dtype == np.int16
        assert len(window) == pytest.approx(6.0 * TARGET_SAMPLE_RATE, abs=0.02 * TARGET_SAMPLE_RATE)
        quiet = window[: int(1.5 * TARGET_SAMPLE_RATE)].astype(np.float64)
        loud = window[int(3.0 * TARGET_SAMPLE_RATE) : int(5.0 * TARGET_SAMPLE_RATE)].astype(
            np.float64
        )
        assert np.sqrt(np.mean(quiet**2)) < 50.0
        assert np.sqrt(np.mean(loud**2)) > 3000.0

    def test_span_past_the_end_is_truncated_not_padded(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.wav"
        _write_mono_wav_with_tone_at(src, seconds=60.0, tone_span=(30.0, 40.0))

        window, start = decode_audio_16k_mono_span(src, start_seconds=58.0, seconds=5.0)

        assert start == pytest.approx(58.0, abs=0.01)
        assert len(window) == pytest.approx(2.0 * TARGET_SAMPLE_RATE, abs=0.02 * TARGET_SAMPLE_RATE)

    def test_span_never_returns_audio_from_outside_its_window(self, tmp_path: Path) -> None:
        # Regression: the decoder seeked to the packet before the window and
        # then kept the first `seconds` of whatever followed, so a stream whose
        # audio starts at 10 s answered a 0-5 s request with audio from 10-15 s.
        src = tmp_path / "late_audio.mp4"
        _write_aac_starting_at(src, start_seconds=10.0, seconds=8.0)

        before_stream, before_start = decode_audio_16k_mono_span(
            src, start_seconds=0.0, seconds=5.0
        )
        straddling, straddling_start = decode_audio_16k_mono_span(
            src, start_seconds=8.0, seconds=4.0
        )
        inside, inside_start = decode_audio_16k_mono_span(src, start_seconds=12.0, seconds=3.0)

        assert len(before_stream) == 0
        assert before_start == 0.0
        # The stream begins inside the window: the span starts where the audio
        # does (the AAC encoder's priming frame sits just before 10 s) and
        # covers only up to the window end.
        assert 9.8 <= straddling_start < 10.05
        assert len(straddling) == pytest.approx(
            (12.0 - straddling_start) * TARGET_SAMPLE_RATE, abs=0.02 * TARGET_SAMPLE_RATE
        )
        assert inside_start == pytest.approx(12.0, abs=0.01)
        assert len(inside) == pytest.approx(3.0 * TARGET_SAMPLE_RATE, abs=0.02 * TARGET_SAMPLE_RATE)
        assert np.sqrt(np.mean(inside.astype(np.float64) ** 2)) > 3000.0

    def test_span_decode_failures_use_the_audio_error_taxonomy(self, tmp_path: Path) -> None:
        src = tmp_path / "garbage.mp4"
        src.write_bytes(b"NOT A REAL CONTAINER")
        with pytest.raises(AudioDecodeError):
            decode_audio_16k_mono_span(src, start_seconds=0.0, seconds=1.0)
        with pytest.raises(AudioDecodeError):
            audio_prep.media_timeline_seconds(src)
