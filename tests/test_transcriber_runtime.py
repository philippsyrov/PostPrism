"""PyAV, VAD, runtime status, cap, cleanup, and detector integration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from postprism import Transcriber
from postprism.asr import transcriber as transcriber_module
from postprism.asr.audio_class import (
    AUDIO_CLASS_INCOMPLETE,
    AUDIO_CLASS_MIXED,
    AUDIO_CLASS_MUSIC,
    AUDIO_CLASS_SPEECH,
    AUDIO_CLASS_UNCERTAIN,
    DEFAULT_AUDIO_CLASS_MODEL,
    DEFAULT_AUDIO_CLASS_REVISION,
)
from postprism.asr.qwen_policy import derive_qwen_trust_decision
from postprism.multimodal.input import assemble_multimodal_text
from tests.transcriber_test_support import (
    FakeAudioTagger as _FakeAudioTagger,
)
from tests.transcriber_test_support import (
    FakeQwenModel as _FakeQwenModel,
)
from tests.transcriber_test_support import (
    FakeQwenResult as _FakeQwenResult,
)
from tests.transcriber_test_support import (
    SnoopQwenModel as _SnoopQwenModel,
)
from tests.transcriber_test_support import (
    patch_qwen_loader as _patch_qwen_loader,
)
from tests.transcriber_test_support import requires_av
from tests.transcriber_test_support import (
    write_wav_media as _write_wav_media,
)


@requires_av
class TestQwenGatedPregate:
    """The qwen-v5 pre-gate in isolation from the default AST audio gate.

    Real PyAV decode over synthesized WAV media; only the Qwen model itself is
    faked. Pins the acceptance-relevant behaviors: silence skips the model,
    speech reaches it as a clean 16 kHz WAV (no librosa fallback inside
    qwen-asr), untrusted transcripts are stored but never fused, long audio is
    bounded, and every failure path cleans up the temp WAV.
    """

    SPEECH_TEXT = "нормальная человеческая речь про путешествия и еду"

    def _warmed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: _FakeQwenModel,
    ) -> Transcriber:
        _patch_qwen_loader(monkeypatch, model)
        transcriber = Transcriber()
        transcriber.audio_classifier = None
        transcriber.warmup()
        return transcriber

    def test_silence_skips_qwen_and_returns_no_speech(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "silent.wav"
        _write_wav_media(media, active_seconds=0.0, silence_seconds=5.0)
        model = _FakeQwenModel()
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "no_speech"
        assert result.asr_used_for_embedding is False
        assert model.transcribe_calls == []  # the latency win: model skipped
        # Measured — not degenerate — clip stats survive the skip.
        assert result.duration_seconds == pytest.approx(5.0, rel=0.01)
        assert result.speech_ratio == pytest.approx(0.0)
        assert result.duration_after_vad == pytest.approx(0.0)

    def test_speech_reaches_qwen_as_clean_wav_and_fuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0, silence_seconds=1.0)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_transcript == self.SPEECH_TEXT
        assert result.asr_used_for_embedding is True
        # Qwen received a NORMALIZED temp WAV, not the original media path.
        assert model.sent_wav_path is not None
        assert model.sent_wav_path.suffix == ".wav"
        assert model.sent_wav_path != media
        # ... and the temp WAV is gone after the call (success cleanup).
        assert not model.sent_wav_path.exists()
        # Gate signals are measured, ready for GPU threshold calibration.
        assert result.duration_seconds == pytest.approx(5.0, rel=0.01)
        assert 0.6 < result.speech_ratio < 1.0
        assert result.duration_after_vad is not None
        assert result.max_compression_ratio is not None

    def test_repetitive_transcript_is_stored_but_not_fused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        looped = "ла ла ла " * 20  # chorus-loop hallucination shape
        model = _FakeQwenModel(_FakeQwenResult(text=looped))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"  # stored for audit
        assert result.asr_transcript == looped.strip()
        assert result.asr_used_for_embedding is False  # never fused
        assert result.max_compression_ratio is not None
        assert result.max_compression_ratio >= transcriber.max_compression_ratio

    def test_implausibly_dense_transcript_is_not_fused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        # ~4.5 s of audible audio cannot support ~500 chars of unique text
        # (>40 chars/s ceiling) — the "insufficiently supported" tell.
        dense = " ".join(f"слово{i}" for i in range(70))
        model = _FakeQwenModel(_FakeQwenResult(text=dense))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_transcript == dense
        assert result.asr_used_for_embedding is False

    def test_low_activity_ratio_is_not_fused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 2 s of tone in a 30 s clip: above the no_speech floor (0.05) but
        # below the trust ratio (0.30) — transcript stored, not fused.
        media = tmp_path / "mostly_silent.wav"
        _write_wav_media(media, active_seconds=2.0, silence_seconds=28.0)
        model = _FakeQwenModel(_FakeQwenResult(text="короткая фраза"))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_used_for_embedding is False
        assert result.speech_ratio is not None
        assert result.speech_ratio < transcriber.min_speech_ratio

    def test_long_audio_is_capped_deterministically(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "long.wav"
        _write_wav_media(media, active_seconds=90.0)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        first = transcriber.transcribe(media)
        first_sent = model.sent_wav_seconds
        second = transcriber.transcribe(media)

        assert first.status == "ok"
        assert first_sent == pytest.approx(30.0, abs=0.5)  # default cap binds
        assert model.sent_wav_seconds == first_sent  # deterministic re-run
        assert first.duration_seconds == pytest.approx(90.0, rel=0.01)
        assert second.asr_transcript == first.asr_transcript
        # 30 s of capped audio yielding only ~50 chars is the incident shape:
        # the long-audio sparse-transcript rule stores but never fuses it.
        assert first.asr_used_for_embedding is False

    def test_empty_transcript_after_qwen_is_no_speech(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        model = _FakeQwenModel(_FakeQwenResult(text="   "))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "no_speech"
        assert result.asr_used_for_embedding is False
        assert model.transcribe_calls != []  # model ran; VAD saw activity
        assert result.max_compression_ratio is None  # nothing to measure

    def test_undecodable_media_is_audio_load_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "garbage.mp4"
        media.write_bytes(b"NOT A REAL CONTAINER")
        model = _FakeQwenModel()
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "audio_load_failed"
        assert model.transcribe_calls == []

    def test_no_audio_stream_maps_to_no_audio(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from postprism.asr.audio_prep import NoAudioStreamError  # noqa: PLC0415

        media = tmp_path / "video_only.mp4"
        media.write_bytes(b"FAKE")

        def raise_no_audio(path: Any, max_seconds: Any = None) -> Any:
            raise NoAudioStreamError("media has no audio stream")

        monkeypatch.setattr("postprism.asr.transcriber.decode_audio_16k_mono", raise_no_audio)
        model = _FakeQwenModel()
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "no_audio"
        assert model.transcribe_calls == []

    def test_too_short_clip_is_too_short(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "blip.wav"
        _write_wav_media(media, active_seconds=0.3)
        model = _FakeQwenModel()
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "too_short"
        assert model.transcribe_calls == []

    def test_temp_wav_cleaned_up_when_qwen_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        model = _SnoopQwenModel(raise_on_transcribe=RuntimeError("CUDA OOM"))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "transcription_failed"
        # The snoop recorded the WAV path before raising; it must be gone.
        assert model.sent_wav_path is not None
        assert not model.sent_wav_path.exists()

    def test_live_decision_matches_pure_rederivation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The auditability contract: for a CAPPED clip (where measured sent
        # audio and persisted activity differ), the live gate decision must be
        # exactly reproducible from the persisted row alone via
        # derive_qwen_trust_decision + the default policy.
        media = tmp_path / "long.wav"
        _write_wav_media(media, active_seconds=90.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.duration_sent_to_asr == pytest.approx(30.0, abs=0.5)
        derived_used, derived_reason = derive_qwen_trust_decision(
            result.asr_transcript,
            result.speech_ratio,
            result.duration_sent_to_asr,
        )
        assert derived_used == result.asr_used_for_embedding
        assert derived_reason == "sparse_long_audio"

    def test_result_contract_contains_only_live_qwen_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        payload = transcriber.transcribe(media).to_dict()

        assert set(payload.keys()) == {
            "asr_transcript",
            "language",
            "duration_seconds",
            "asr_model",
            "asr_model_version",
            "asr_model_revision",
            "asr_used_for_embedding",
            "duration_after_vad",
            "speech_ratio",
            "duration_sent_to_asr",
            "max_compression_ratio",
            "asr_trust_reason",
            "audio_class",
            "audio_class_confidence",
            "audio_class_model",
            "audio_class_revision",
            "audio_class_coverage",
            "asr_audio_selection",
            "asr_transcribed_at",
            "status",
        }
        assert payload["asr_audio_selection"] == "full-decode"
        # Under the cap, everything active was sent.
        assert payload["duration_sent_to_asr"] == pytest.approx(payload["duration_after_vad"])
        # This isolated pre-gate fixture skips AST, so detector fields are null.
        assert payload["audio_class"] is None
        assert payload["audio_class_confidence"] is None
        assert payload["audio_class_model"] is None
        assert payload["audio_class_coverage"] is None
        # A trusted transcript has no failing check to name.
        assert payload["asr_used_for_embedding"] is True
        assert payload["asr_trust_reason"] is None

    def _write_long_wav(
        self, path: Path, seconds: float, tone_spans: list[tuple[float, float]]
    ) -> None:
        import wave  # noqa: PLC0415

        rate = 16000
        samples = np.zeros(int(seconds * rate), dtype=np.int16)
        for start, end in tone_spans:
            first, last = int(start * rate), int(end * rate)
            t = np.arange(last - first) / rate
            samples[first:last] = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(samples.tobytes())

    def _spy_span_decoder(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
        from postprism.asr import audio_prep  # noqa: PLC0415

        requested: list[tuple[float, float]] = []
        real = audio_prep.decode_audio_16k_mono_span

        def spy(path: Any, start_seconds: float, seconds: float) -> Any:
            requested.append((start_seconds, seconds))
            return real(path, start_seconds, seconds)

        monkeypatch.setattr(transcriber_module, "decode_audio_16k_mono_span", spy)
        return requested

    def test_long_media_is_probed_across_the_whole_timeline_without_full_decode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 700 s of media with speech-shaped activity at 100-110 s (head span,
        # past its first probe), 330-350 s (around the middle-span centre) and
        # 660-680 s (inside the end-anchored probe). The old 600 s ceiling
        # truncated the timeline; every third must now contribute while only
        # bounded windows are decoded.
        media = tmp_path / "long.wav"
        self._write_long_wav(media, 700.0, [(100.0, 110.0), (330.0, 350.0), (660.0, 680.0)])

        def refuse_full_decode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("long media must not be decoded in full")

        monkeypatch.setattr(transcriber_module, "decode_audio_16k_mono", refuse_full_decode)
        requested = self._spy_span_decoder(monkeypatch)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        assert result.duration_seconds == pytest.approx(700.0, abs=0.05)
        # Every timeline third contributed activity, bounded by the 30 s cap.
        assert model.sent_wav_seconds is not None
        assert 28.0 <= model.sent_wav_seconds <= 30.5
        assert any(start + seconds >= 700.0 - 0.5 for start, seconds in requested)  # end probed
        decoded = sum(seconds for _, seconds in requested)
        assert decoded <= 3 * 120.0 + 1.0  # bounded, not the 700 s clip
        assert all(start >= 0.0 and start + seconds <= 700.0 + 1.0 for start, seconds in requested)
        assert result.duration_after_vad is not None and result.duration_after_vad > 0.0
        assert result.speech_ratio is not None and 0.0 < result.speech_ratio <= 1.0

    def test_long_media_extends_a_silent_probe_before_giving_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Activity sits at 100-110 s: past the first 60 s probe, inside the
        # bounded extension. Nothing elsewhere, so the clip is as sparse as a
        # 600 s full decode with 10 s of speech and stays `no_speech` — but the
        # extension must have found the burst rather than stopping at 60 s.
        media = tmp_path / "late_start.wav"
        self._write_long_wav(media, 700.0, [(100.0, 110.0)])
        requested = self._spy_span_decoder(monkeypatch)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "no_speech"
        assert result.duration_after_vad == pytest.approx(10.0, abs=1.0)
        assert result.speech_ratio is not None and result.speech_ratio < 0.05
        assert model.transcribe_calls == []
        first_third = [(s, n) for s, n in requested if s < 233.0]
        assert sum(n for _, n in first_third) <= 120.0 + 1.0
        assert max(s + n for s, n in first_third) >= 110.0

    def test_silent_long_media_skips_qwen_with_bounded_probing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "silent_long.wav"
        self._write_long_wav(media, 700.0, [])
        requested = self._spy_span_decoder(monkeypatch)
        model = _FakeQwenModel()
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "no_speech"
        assert result.asr_audio_selection == "timeline-probes"
        assert model.transcribe_calls == []
        assert sum(seconds for _, seconds in requested) <= 3 * 120.0 + 1.0
        assert result.duration_seconds == pytest.approx(700.0, abs=0.05)

    def test_positive_container_origin_beyond_duration_still_probes_all_three_spans(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import av  # noqa: PLC0415

        media = tmp_path / "shifted_long.mp4"
        _write_aac_media_starting_at(
            media,
            start_seconds=1000.0,
            seconds=700.0,
            tone_spans=[(1010.0, 1030.0), (1340.0, 1360.0), (1680.0, 1700.0)],
        )
        with av.open(str(media)) as container:
            origin = container.start_time / av.time_base
            duration = container.duration / av.time_base
        assert origin > duration > 600.0
        requested = self._spy_span_decoder(monkeypatch)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        assert result.duration_seconds == pytest.approx(duration)
        assert model.sent_wav_seconds == pytest.approx(30.0, abs=0.1)
        assert requested == pytest.approx(
            [(origin, 60.0), (origin + duration / 2 - 30.0, 60.0), (origin + duration - 60.0, 60.0)]
        )
        assert max(start + seconds for start, seconds in requested) == pytest.approx(
            origin + duration
        )

    def test_zero_origin_keeps_exact_probe_timestamps_and_selected_pcm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import wave  # noqa: PLC0415

        class PCMRecorder(_SnoopQwenModel):
            sent_pcm: np.ndarray

            def transcribe(self, audio: str, **kwargs: Any) -> list[object]:
                with wave.open(audio) as wav_file:
                    self.sent_pcm = np.frombuffer(
                        wav_file.readframes(wav_file.getnframes()), dtype=np.int16
                    ).copy()
                return super().transcribe(audio, **kwargs)

        media = tmp_path / "zero_origin.wav"
        self._write_long_wav(media, 700.0, [(0.0, 700.0)])
        with wave.open(str(media)) as wav_file:
            source_pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        # The accepted zero-start policy takes each anchor's first 10 s.
        expected = np.concatenate(
            [source_pcm[start * 16000 : (start + 10) * 16000] for start in (0, 320, 640)]
        )
        requested = self._spy_span_decoder(monkeypatch)
        model = PCMRecorder(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.duration_seconds == 700.0
        assert requested == [(0.0, 60.0), (320.0, 60.0), (640.0, 60.0)]
        np.testing.assert_array_equal(model.sent_pcm, expected)

    def test_media_within_the_full_decode_ceiling_keeps_the_accepted_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0, silence_seconds=1.0)

        def refuse_spans(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("short media must use the full-decode path")

        monkeypatch.setattr(transcriber_module, "decode_audio_16k_mono_span", refuse_spans)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "full-decode"
        assert result.asr_used_for_embedding is True

    def test_speech_only_in_the_final_seconds_is_found_by_the_end_probe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The end-anchored probe must see minute 11+ of an 11.5-minute clip;
        # a forward-only scheme left the last minutes unseen.
        media = tmp_path / "tail_speech.wav"
        self._write_long_wav(media, 700.0, [(670.0, 700.0)])
        requested = self._spy_span_decoder(monkeypatch)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        assert result.duration_after_vad == pytest.approx(30.0, abs=1.0)
        assert model.sent_wav_seconds == pytest.approx(10.0, abs=1.0)  # one window's share
        assert max(start + seconds for start, seconds in requested) == pytest.approx(700.0, abs=0.5)
        assert sum(seconds for _, seconds in requested) <= 3 * 120.0 + 1.0

    def test_long_media_failures_keep_the_probe_selection_provenance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from postprism.asr.audio_prep import AudioDecodeError  # noqa: PLC0415

        media = tmp_path / "long.wav"
        self._write_long_wav(media, 700.0, [(0.0, 30.0)])

        def broken_span(*_args: Any, **_kwargs: Any) -> Any:
            raise AudioDecodeError("truncated packet")

        monkeypatch.setattr(transcriber_module, "decode_audio_16k_mono_span", broken_span)
        transcriber = self._warmed(monkeypatch, _FakeQwenModel())
        failed_decode = transcriber.transcribe(media)

        assert failed_decode.status == "audio_load_failed"
        assert failed_decode.asr_audio_selection == "timeline-probes"

        monkeypatch.undo()
        model = _FakeQwenModel(raise_on_transcribe=RuntimeError("CUDA OOM"))
        transcriber = self._warmed(monkeypatch, model)
        failed_model = transcriber.transcribe(media)

        assert failed_model.status == "transcription_failed"
        assert failed_model.asr_audio_selection == "timeline-probes"

    def test_long_media_rows_carry_the_bumped_asr_policy_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "long.wav"
        self._write_long_wav(media, 700.0, [(0.0, 30.0)])
        transcriber = self._warmed(
            monkeypatch, _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        )

        result = transcriber.transcribe(media)

        assert result.asr_model_version.endswith("-qwen-v5")

    def test_probe_density_is_judged_on_the_audio_sent_not_the_capped_activity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Regression: the density rules received min(activity, 30 s). Under
        # probes a span's activity is counted in full but only its 10 s share
        # is sent, so 70 s of activity with 20 s sent was judged as 30 s and a
        # normal transcript failed `sparse_long_audio`. Probes are bounded to
        # one chunk here so the silent middle span keeps the ratio above 0.30.
        monkeypatch.setattr(transcriber_module, "TIMELINE_PROBE_MAX_SECONDS", 60.0)
        media = tmp_path / "front_and_tail.wav"
        self._write_long_wav(media, 700.0, [(0.0, 60.0), (690.0, 700.0)])
        # ~125 chars of ordinary speech: 6 chars/s over the 20 s sent (trusted)
        # but 4 chars/s over the 30 s the old input claimed (sparse_long_audio).
        transcript = (
            "мы приехали на побережье рано утром, купили свежий хлеб и сыр на рынке, "
            "потом долго гуляли вдоль набережной и смотрели на лодки"
        )
        model = _SnoopQwenModel(_FakeQwenResult(text=transcript))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        assert result.duration_after_vad == pytest.approx(70.0, abs=1.0)
        assert result.duration_sent_to_asr == pytest.approx(20.0, abs=0.1)
        assert model.sent_wav_seconds == pytest.approx(result.duration_sent_to_asr, abs=0.01)
        assert result.asr_trust_reason is None
        assert result.asr_used_for_embedding is True
        # The persisted row re-derives the live decision; the old input would not.
        assert derive_qwen_trust_decision(
            result.asr_transcript, result.speech_ratio, result.duration_sent_to_asr
        ) == (True, None)
        assert derive_qwen_trust_decision(
            result.asr_transcript,
            result.speech_ratio,
            min(result.duration_after_vad, transcriber.max_audio_seconds),
        ) == (False, "sparse_long_audio")

    def test_end_span_budget_keeps_the_tail_chunk_over_its_earlier_extension(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Regression: the last span probes the end chunk first and extends
        # backward only because the tail was short. Selecting the span's 10 s
        # chronologically then spent the whole budget on the extension's
        # 600-610 s and dropped the tail speech at 695-700 s entirely.
        media = tmp_path / "tail_and_extension.wav"
        self._write_long_wav(media, 700.0, [(0.0, 30.0), (600.0, 610.0), (695.0, 700.0)])
        sliced: list[list[tuple[float, float]]] = []
        real_slice = transcriber_module.slice_decoded_spans

        def slice_spy(spans: Any, sample_rate: int, windows: Any) -> Any:
            sliced.append(list(windows))
            return real_slice(spans, sample_rate, windows)

        monkeypatch.setattr(transcriber_module, "slice_decoded_spans", slice_spy)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        last_span = sliced[-1]
        assert len(last_span) == 2
        (extension_start, extension_end), (tail_start, tail_end) = last_span  # chronological
        assert tail_start == pytest.approx(695.0, abs=0.3)  # whole tail survives
        assert tail_end == pytest.approx(700.0, abs=0.05)
        assert extension_start == pytest.approx(600.0, abs=0.3)
        assert extension_end - extension_start == pytest.approx(
            10.0 - (tail_end - tail_start), abs=0.05
        )  # only the budget remainder (~5 s) of the extension
        assert sum(end - start for start, end in last_span) == pytest.approx(10.0, abs=0.05)
        assert model.sent_wav_seconds == pytest.approx(20.0, abs=0.1)  # head span + end span

    def test_probe_chunks_keep_only_audio_inside_their_window(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Audio timestamps start at 30 s. The 0-60 s probe used to answer
        # with 30-90 s of audio, so speech at 62-72 s was "found" by the first
        # chunk and mis-placed; now that chunk yields 30-60 s and the second
        # chunk must be requested to reach the speech.
        media = tmp_path / "late_audio_long.mp4"
        _write_aac_media_starting_at(
            media,
            start_seconds=30.0,
            seconds=670.0,
            tone_spans=[(62.0, 72.0), (335.0, 345.0)],
            video_start_seconds=0.0,
        )
        requested = self._spy_span_decoder(monkeypatch)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_audio_selection == "timeline-probes"
        first_span = [(start, seconds) for start, seconds in requested if start < 233.0]
        assert first_span[:2] == [(0.0, 60.0), (60.0, 60.0)]
        assert result.duration_after_vad == pytest.approx(20.0, abs=1.5)
        assert model.sent_wav_seconds == pytest.approx(20.0, abs=0.5)

    def test_audio_delayed_behind_video_keeps_the_overall_media_tail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import av  # noqa: PLC0415

        media = tmp_path / "delayed_tail.mp4"
        _write_aac_media_starting_at(
            media,
            start_seconds=30.0,
            seconds=670.0,
            tone_spans=[(670.0, 700.0)],
            video_start_seconds=0.0,
        )
        with av.open(str(media)) as container:
            assert container.start_time == 0
            assert container.streams.audio[0].start_time * container.streams.audio[0].time_base > 29
            duration = container.duration / av.time_base
        requested = self._spy_span_decoder(monkeypatch)
        model = _SnoopQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        transcriber = self._warmed(monkeypatch, model)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.duration_seconds == pytest.approx(700.0, abs=0.1)
        assert requested[0] == (0.0, 60.0)
        assert max(start + seconds for start, seconds in requested) == pytest.approx(duration)
        assert result.duration_after_vad == pytest.approx(30.0, abs=0.5)
        assert model.sent_wav_seconds == pytest.approx(10.0, abs=0.1)


def _write_aac_media_starting_at(
    path: Path,
    *,
    start_seconds: float,
    seconds: float,
    tone_spans: list[tuple[float, float]],
    video_start_seconds: float | None = None,
) -> None:
    """Mono AAC MP4 whose audio timestamps begin at `start_seconds`; tones are
    placed at absolute timeline seconds."""
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415

    rate = 16000
    frame_size = 1024
    samples = np.zeros(int(seconds * rate), dtype=np.int16)
    for tone_start, tone_end in tone_spans:
        first = int((tone_start - start_seconds) * rate)
        last = int((tone_end - start_seconds) * rate)
        t = np.arange(last - first) / rate
        samples[first:last] = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    offset = int(round(start_seconds * rate))
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=rate, layout="mono")
        stream.codec_context.time_base = Fraction(1, rate)
        if video_start_seconds is not None:
            video = container.add_stream("mpeg4", rate=1)
            video.width = video.height = 16
            video.pix_fmt = "yuv420p"
            video_frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), "rgb24")
            video_frame.pts = int(video_start_seconds)
            video_frame.time_base = Fraction(1, 1)
            for packet in video.encode(video_frame):
                container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        for first in range(0, len(samples) - frame_size + 1, frame_size):
            frame = av.AudioFrame.from_ndarray(
                samples[first : first + frame_size].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = rate
            frame.pts = offset + first
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@requires_av
class TestQwenAudioGateIntegration:
    """End-to-end wiring of the fixed gate through the real qwen-v5 path.

    Real PyAV decode + real energy VAD + real `AudioClassifier`; only the Qwen
    model and the AudioSet tagger are faked, so no weights are downloaded.
    """

    SPEECH_TEXT = "нормальная человеческая речь про путешествия и еду"
    PLAUSIBLE_LYRICS = (
        "I don't wanna wait no more baby take my hand tonight we can run "
        "into the light everything will be alright"
    )
    MUSIC_SCORES = {"Music": 0.96, "Singing": 0.63, "Speech": 0.06}
    SPEECH_SCORES = {"Speech": 0.94, "Music": 0.08, "Singing": 0.01}
    MIXED_SCORES = {"Speech": 0.85, "Music": 0.93, "Singing": 0.47}

    def _warmed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: _FakeQwenModel,
        tagger: _FakeAudioTagger | None = None,
        max_chunks: int | None = None,
    ) -> Transcriber:
        _patch_qwen_loader(monkeypatch, model)
        if tagger is not None:
            monkeypatch.setattr(
                "postprism.asr.audio_class._load_audio_class_tagger",
                lambda model_name, device, revision=None: tagger,
            )
        transcriber = Transcriber()
        if max_chunks is not None:
            assert transcriber.audio_classifier is not None
            transcriber.audio_classifier.max_chunks = max_chunks
        transcriber.warmup()
        return transcriber

    def test_gate_is_always_active(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=4.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        assert transcriber.audio_classifier is not None
        assert "astgate1" in transcriber.model_version
        assert result.audio_class == AUDIO_CLASS_SPEECH
        assert result.asr_used_for_embedding is True

    def test_music_verdict_stores_the_transcript_but_never_fuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "song.wav"
        _write_wav_media(media, active_seconds=12.0)
        lyrics = self.PLAUSIBLE_LYRICS
        model = _FakeQwenModel(_FakeQwenResult(text=lyrics))
        tagger = _FakeAudioTagger(self.MUSIC_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_transcript == lyrics  # retained for audit
        assert result.asr_used_for_embedding is False
        assert result.audio_class == AUDIO_CLASS_MUSIC
        assert result.audio_class_confidence == pytest.approx(0.96)
        assert result.asr_trust_reason == "music_audio"
        assert result.audio_class_model == DEFAULT_AUDIO_CLASS_MODEL
        assert result.audio_class_coverage == pytest.approx(1.0)
        # The RESOLVED sha, so a withheld row says which weights withheld it.
        assert result.audio_class_revision == DEFAULT_AUDIO_CLASS_REVISION
        assert result.to_dict()["audio_class_revision"] == DEFAULT_AUDIO_CLASS_REVISION
        # And the serialized payload is refused by the fail-closed assemblers.
        assert assemble_multimodal_text(None, [result.to_dict()]) == ""

    def test_speech_verdict_still_fuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=6.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_used_for_embedding is True
        assert result.audio_class == AUDIO_CLASS_SPEECH
        assert result.asr_trust_reason is None
        assert result.audio_class_coverage == pytest.approx(1.0)
        assert "Trusted speech transcript" in assemble_multimodal_text(None, [result.to_dict()])

    def test_mixed_verdict_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech_over_song.wav"
        _write_wav_media(media, active_seconds=8.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(self.MIXED_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        assert result.audio_class == AUDIO_CLASS_MIXED
        assert result.asr_used_for_embedding is False
        assert result.asr_trust_reason == "mixed_audio"

    def test_incomplete_coverage_fails_closed_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Force incomplete coverage to prove the detector cannot approve a
        # transcript drawn from audio it only partly heard.
        media = tmp_path / "long.wav"
        _write_wav_media(media, active_seconds=30.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger, max_chunks=1)

        result = transcriber.transcribe(media)

        assert result.status == "ok"
        assert result.asr_transcript == self.SPEECH_TEXT  # still stored
        assert result.audio_class == AUDIO_CLASS_INCOMPLETE
        assert result.asr_used_for_embedding is False
        assert result.asr_trust_reason == "incomplete_audio"
        assert result.audio_class_coverage is not None
        assert result.audio_class_coverage < 1.0
        assert tagger.calls == 0  # no partial verdict is even computed
        assert assemble_multimodal_text(None, [result.to_dict()]) == ""

    def test_undersized_chunk_budget_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A budget that cannot cover the ASR cap makes every long clip
        # `incomplete`, so the mismatch must be visible during warmup.
        model = _FakeQwenModel()
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        with caplog.at_level("WARNING", logger="postprism.asr.transcriber"):
            self._warmed(monkeypatch, model, tagger, max_chunks=1)

        assert any("ASR audio gate covers only" in record.message for record in caplog.records)

    def test_default_chunk_budget_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = _FakeQwenModel()
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        with caplog.at_level("WARNING", logger="postprism.asr.transcriber"):
            self._warmed(monkeypatch, model, tagger)

        assert not any("ASR audio gate covers only" in record.message for record in caplog.records)

    def test_detector_failure_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "speech.wav"
        _write_wav_media(media, active_seconds=6.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(raise_on_scores=RuntimeError("CUDA OOM"))
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        # ASR still ran and the transcript is stored — only fusion is denied.
        assert result.status == "ok"
        assert result.asr_transcript == self.SPEECH_TEXT
        assert result.audio_class == AUDIO_CLASS_UNCERTAIN
        assert result.asr_used_for_embedding is False
        assert result.asr_trust_reason == "uncertain_audio"

    def test_silence_still_skips_asr_and_the_detector(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "silent.wav"
        _write_wav_media(media, active_seconds=0.0, silence_seconds=5.0)
        model = _FakeQwenModel()
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        assert result.status == "no_speech"
        assert result.asr_used_for_embedding is False
        assert model.transcribe_calls == []  # unchanged pre-gate skip
        assert tagger.calls == 0  # the detector costs nothing on silence
        assert result.audio_class is None

    def test_detector_scores_exactly_the_samples_handed_to_qwen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The binding the whole gate rests on: the detector's windows and the
        # ASR input are the SAME samples. Capture what gets serialized to the
        # WAV Qwen reads and compare it to the concatenated scored windows.
        written: list[np.ndarray] = []
        real_write = transcriber_module.write_wav_16k_mono

        def _spy(samples: np.ndarray, path: Path) -> None:
            written.append(np.array(samples, copy=True))
            real_write(samples, path)

        monkeypatch.setattr(transcriber_module, "write_wav_16k_mono", _spy)

        media = tmp_path / "long.wav"
        _write_wav_media(media, active_seconds=90.0)
        model = _FakeQwenModel(_FakeQwenResult(text=self.SPEECH_TEXT))
        tagger = _FakeAudioTagger(self.SPEECH_SCORES)
        transcriber = self._warmed(monkeypatch, model, tagger)

        result = transcriber.transcribe(media)

        # The same 30 s the cap hands to Qwen — not the 90 s clip.
        assert tagger.scored_seconds == pytest.approx(30.0, abs=0.5)
        assert len(written) == 1
        # Windows are contiguous at the default chunking, so concatenating them
        # must reproduce the ASR input sample-for-sample.
        assert np.array_equal(np.concatenate(tagger.scored_chunks), written[0])
        assert result.audio_class_coverage == pytest.approx(1.0)


def test_pyav_runtime_classes_follow_dependency_availability() -> None:
    expected_skip = importlib.util.find_spec("av") is None
    for owner in (TestQwenGatedPregate, TestQwenAudioGateIntegration):
        skip_marks = [mark for mark in getattr(owner, "pytestmark", []) if mark.name == "skipif"]
        assert len(skip_marks) == 1
        assert skip_marks[0].args[0] is expected_skip
