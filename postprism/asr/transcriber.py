"""Synchronous Qwen ASR worker with VAD and audio trust gating."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from postprism.asr.audio_class import (
    AUDIO_CLASS_INCOMPLETE,
    AudioClassifier,
    AudioClassResult,
)
from postprism.asr.audio_prep import (
    AUDIO_PREP_METHOD_VERSION,
    TARGET_SAMPLE_RATE,
    AudioDecodeError,
    NoAudioStreamError,
    decode_audio_16k_mono,
    decode_audio_16k_mono_span,
    ensure_audio_prep_available,
    media_timeline_seconds,
    temporary_wav_path,
    write_wav_16k_mono,
)
from postprism.asr.qwen_policy import (
    DEFAULT_MAX_CHARS_PER_SPEECH_SECOND,
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MIN_LONG_CLIP_DENSITY,
    DEFAULT_MIN_SPEECH_RATIO,
    DEFAULT_QWEN_COMPUTE_TYPE,
    DEFAULT_QWEN_DEVICE,
    DEFAULT_QWEN_LONG_CLIP_SECONDS,
    DEFAULT_QWEN_MAX_ACTIVE_SECONDS,
    DEFAULT_QWEN_MAX_BATCH_SIZE,
    DEFAULT_QWEN_MAX_NEW_TOKENS,
    DEFAULT_QWEN_MODEL_NAME,
    DEFAULT_QWEN_MODEL_REVISION,
    DEFAULT_QWEN_MODEL_VERSION,
    DEFAULT_QWEN_SPEECH_WINDOWS,
    FULL_DECODE_MAX_SECONDS,
    MIN_AUDIO_SECONDS,
    NO_SPEECH_RATIO_FLOOR,
    TIMELINE_PROBE_MAX_SECONDS,
    TIMELINE_PROBE_SECONDS,
    _transcript_compression_ratio,
    derive_qwen_trust_decision,
)
from postprism.asr.qwen_runtime import (
    QwenAsrModelProtocol as _QwenAsrModelProtocol,
)
from postprism.asr.qwen_runtime import (
    first_qwen_result as _first_qwen_result,
)
from postprism.asr.qwen_runtime import (
    load_qwen_asr_model as _load_qwen_asr_model,
)
from postprism.asr.qwen_runtime import (
    normalize_language_output as _normalize_language_output,
)
from postprism.asr.qwen_runtime import (
    qwen_result_value as _qwen_result_value,
)
from postprism.asr.vad import (
    VAD_METHOD_VERSION,
    VadConfig,
    detect_active_intervals,
    merge_touching_intervals,
    plan_timeline_probes,
    select_speech_windows,
    select_windows_by_chunk_priority,
    slice_decoded_spans,
    slice_samples,
    total_interval_seconds,
)
from postprism.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


def _empty_result(
    model_name: str,
    model_version: str,
    status: str,
    model_revision: str,
    selection: str = "full-decode",
) -> TranscriptionResult:
    """Build the stable empty result for a failure or skip status.

    `selection` keeps the audio-selection provenance on failures too, so a
    long-media row never claims the full-decode path it did not take.
    """
    return TranscriptionResult(
        asr_transcript="",
        language="",
        duration_seconds=0.0,
        asr_model=model_name,
        asr_model_version=model_version,
        asr_model_revision=model_revision,
        asr_used_for_embedding=False,
        asr_audio_selection=selection,
        status=status,
    )


class Transcriber:
    """Transcribe one audio/video file into a `TranscriptionResult`.

    Sync and lazy. User-data failures collapse to stable statuses; calling
    `transcribe()` before `warmup()` remains the loud programmer-error path.
    """

    def __init__(self) -> None:
        self.model_name = DEFAULT_QWEN_MODEL_NAME
        self.model_revision = DEFAULT_QWEN_MODEL_REVISION
        self.device = DEFAULT_QWEN_DEVICE
        self.compute_type = DEFAULT_QWEN_COMPUTE_TYPE
        self.qwen_max_new_tokens = DEFAULT_QWEN_MAX_NEW_TOKENS
        self.qwen_max_batch_size = DEFAULT_QWEN_MAX_BATCH_SIZE
        self.max_audio_seconds = DEFAULT_QWEN_MAX_ACTIVE_SECONDS
        self.speech_windows = DEFAULT_QWEN_SPEECH_WINDOWS
        self.vad_energy_threshold_dbfs = VadConfig().energy_threshold_dbfs
        self.min_speech_ratio = DEFAULT_MIN_SPEECH_RATIO
        self.max_compression_ratio = DEFAULT_MAX_COMPRESSION_RATIO
        self.max_chars_per_speech_second = DEFAULT_MAX_CHARS_PER_SPEECH_SECOND
        self.long_clip_seconds = DEFAULT_QWEN_LONG_CLIP_SECONDS
        self.min_long_clip_density = DEFAULT_MIN_LONG_CLIP_DENSITY
        self.audio_classifier: AudioClassifier | None = AudioClassifier()
        self.model_version = DEFAULT_QWEN_MODEL_VERSION
        self._model: _QwenAsrModelProtocol | None = None

    @property
    def is_warm(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Load ASR weights. Idempotent and optional-dependency-lazy."""
        if self.is_warm:
            return
        ensure_audio_prep_available()
        logger.info(
            "Qwen ASR pre-gate active: vad=%s prep=%s cap=%.0fs windows=%d",
            VAD_METHOD_VERSION,
            AUDIO_PREP_METHOD_VERSION,
            self.max_audio_seconds,
            self.speech_windows,
        )
        if self.audio_classifier is not None:
            self.audio_classifier.warmup()
            coverable = self.audio_classifier.chunk_seconds * self.audio_classifier.max_chunks
            if coverable < self.max_audio_seconds:
                logger.warning(
                    "ASR audio gate covers only %.0fs (%d x %.1fs) "
                    "of the %.0fs ASR cap; clips longer than that will be "
                    "classified %s and will NOT fuse.",
                    coverable,
                    self.audio_classifier.max_chunks,
                    self.audio_classifier.chunk_seconds,
                    self.max_audio_seconds,
                    AUDIO_CLASS_INCOMPLETE,
                )
        self._model = _load_qwen_asr_model(
            self.model_name,
            self.device,
            self.compute_type,
            self.qwen_max_new_tokens,
            self.qwen_max_batch_size,
            model_revision=self.model_revision,
        )
        logger.info(
            "Transcriber ready: model=%s version=%s",
            self.model_name,
            self.model_version,
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe one audio/video file through the gated qwen-v5 path."""
        if not self.is_warm:
            raise RuntimeError("Transcriber not warmed up; call .warmup() before .transcribe()")
        assert self._model is not None

        path_obj = Path(audio_path)
        if not path_obj.exists():
            logger.warning("ASR audio missing: %s", path_obj)
            return _empty_result(
                self.model_name, self.model_version, "audio_load_failed", self.model_revision
            )
        started = time.perf_counter()
        return self._transcribe_qwen_gated(path_obj, started)

    def _transcribe_qwen_gated(self, path_obj: Path, started: float) -> TranscriptionResult:
        """Decode, select activity, classify, transcribe, and gate fusion."""
        assert self._model is not None

        # Media longer than the full-decode ceiling is probed across its whole
        # timeline; an unreadable duration falls through to the accepted path,
        # which reports the decode problem with the usual statuses.
        try:
            timeline_hint = media_timeline_seconds(path_obj)
        except AudioDecodeError:
            timeline_hint = None
        if timeline_hint is not None and timeline_hint[1] > FULL_DECODE_MAX_SECONDS:
            origin_seconds, duration_seconds = timeline_hint
            return self._transcribe_by_timeline_probes(
                path_obj, started, origin_seconds, duration_seconds
            )

        decode_started = time.perf_counter()
        try:
            samples = decode_audio_16k_mono(path_obj, max_seconds=FULL_DECODE_MAX_SECONDS)
        except NoAudioStreamError as err:
            logger.warning("Qwen ASR no_audio for %s: %s", path_obj, err)
            return _empty_result(
                self.model_name, self.model_version, "no_audio", self.model_revision
            )
        except AudioDecodeError as err:
            logger.warning("Qwen ASR audio_load_failed for %s: %s", path_obj, err)
            return _empty_result(
                self.model_name, self.model_version, "audio_load_failed", self.model_revision
            )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        duration_seconds = len(samples) / TARGET_SAMPLE_RATE
        if duration_seconds <= 0.0:
            return _empty_result(
                self.model_name, self.model_version, "no_audio", self.model_revision
            )
        if duration_seconds < MIN_AUDIO_SECONDS:
            return _empty_result(
                self.model_name, self.model_version, "too_short", self.model_revision
            )

        vad_started = time.perf_counter()
        intervals = detect_active_intervals(samples, TARGET_SAMPLE_RATE, self._vad_config())
        active_seconds = min(total_interval_seconds(intervals), duration_seconds)
        speech_ratio = max(0.0, min(1.0, active_seconds / duration_seconds))
        vad_ms = (time.perf_counter() - vad_started) * 1000.0

        if active_seconds < MIN_AUDIO_SECONDS or speech_ratio < NO_SPEECH_RATIO_FLOOR:
            return self._no_speech(
                path_obj,
                started,
                duration_seconds,
                active_seconds,
                speech_ratio,
                "full-decode",
                decode_ms,
                vad_ms,
            )

        windows = select_speech_windows(intervals, self.max_audio_seconds, self.speech_windows)
        clipped = slice_samples(samples, TARGET_SAMPLE_RATE, windows)
        return self._transcribe_selected(
            path_obj,
            started,
            clipped,
            duration_seconds,
            active_seconds,
            speech_ratio,
            "full-decode",
            decode_ms,
            vad_ms,
        )

    def _vad_config(self) -> VadConfig:
        return VadConfig(energy_threshold_dbfs=self.vad_energy_threshold_dbfs)

    def _transcribe_by_timeline_probes(
        self,
        path_obj: Path,
        started: float,
        origin_seconds: float,
        duration_seconds: float,
    ) -> TranscriptionResult:
        """Select bounded speech windows across long media without decoding it all.

        The timeline is split into `speech_windows` equal spans probed at the
        beginning, middle, and end (see `plan_timeline_probes`); each span
        decodes its planned chunks in order until it has found its share of the
        activity cap or its TIMELINE_PROBE_MAX_SECONDS bound. `duration_after_vad`
        and `speech_ratio` describe the probed audio, so the trust policy judges
        the evidence it actually heard.
        """
        selection = "timeline-probes"
        budget = self.max_audio_seconds / self.speech_windows
        plan = plan_timeline_probes(
            duration_seconds,
            self.speech_windows,
            TIMELINE_PROBE_SECONDS,
            TIMELINE_PROBE_MAX_SECONDS,
        )
        pieces: list[np.ndarray] = []
        probed_seconds = 0.0
        active_seconds = 0.0
        decode_ms = 0.0
        vad_ms = 0.0
        for chunks in plan:
            # Decoded spans are placed at the start the decoder actually
            # returned, never at the planned chunk start: a stream that begins
            # or ends inside a chunk leaves a gap, so spans of one probe are
            # not assumed adjacent and activity is detected per span.
            decoded: list[tuple[float, np.ndarray]] = []
            intervals: list[tuple[float, float]] = []
            # Activity per chunk in decode order: the span budget is allocated
            # chunk by chunk (anchor first), not chronologically.
            found: list[list[tuple[float, float]]] = []
            for chunk_start, chunk_end in chunks:
                decode_started = time.perf_counter()
                try:
                    chunk, chunk_origin = decode_audio_16k_mono_span(
                        # Planning is relative; seeking uses absolute media PTS.
                        path_obj,
                        origin_seconds + chunk_start,
                        chunk_end - chunk_start,
                    )
                except NoAudioStreamError as err:
                    logger.warning("Qwen ASR no_audio for %s: %s", path_obj, err)
                    return _empty_result(
                        self.model_name,
                        self.model_version,
                        "no_audio",
                        self.model_revision,
                        selection,
                    )
                except AudioDecodeError as err:
                    logger.warning("Qwen ASR audio_load_failed for %s: %s", path_obj, err)
                    return _empty_result(
                        self.model_name,
                        self.model_version,
                        "audio_load_failed",
                        self.model_revision,
                        selection,
                    )
                decode_ms += (time.perf_counter() - decode_started) * 1000.0
                if len(chunk) == 0:
                    continue
                decoded.append((chunk_origin, chunk))
                vad_started = time.perf_counter()
                chunk_intervals = [
                    (chunk_origin + interval_start, chunk_origin + interval_end)
                    for interval_start, interval_end in detect_active_intervals(
                        chunk, TARGET_SAMPLE_RATE, self._vad_config()
                    )
                ]
                found.append(chunk_intervals)
                intervals = merge_touching_intervals(intervals + chunk_intervals)
                vad_ms += (time.perf_counter() - vad_started) * 1000.0
                if total_interval_seconds(intervals) >= budget:
                    break
            probed_seconds += sum(len(samples) for _, samples in decoded) / TARGET_SAMPLE_RATE
            active_seconds += total_interval_seconds(intervals)
            if intervals:
                windows = select_windows_by_chunk_priority(found, budget)
                pieces.append(slice_decoded_spans(decoded, TARGET_SAMPLE_RATE, windows))

        speech_ratio = (
            max(0.0, min(1.0, active_seconds / probed_seconds)) if probed_seconds > 0 else 0.0
        )
        if active_seconds < MIN_AUDIO_SECONDS or speech_ratio < NO_SPEECH_RATIO_FLOOR:
            return self._no_speech(
                path_obj,
                started,
                duration_seconds,
                active_seconds,
                speech_ratio,
                selection,
                decode_ms,
                vad_ms,
            )
        clipped = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int16)
        return self._transcribe_selected(
            path_obj,
            started,
            clipped,
            duration_seconds,
            active_seconds,
            speech_ratio,
            selection,
            decode_ms,
            vad_ms,
        )

    def _no_speech(
        self,
        path_obj: Path,
        started: float,
        duration_seconds: float,
        active_seconds: float,
        speech_ratio: float,
        selection: str,
        decode_ms: float,
        vad_ms: float,
    ) -> TranscriptionResult:
        logger.info(
            "Qwen ASR no_speech (pre-gate, model skipped): path=%s selection=%s "
            "dur=%.2fs active=%.2fs ratio=%.2f decode_ms=%.0f vad_ms=%.0f elapsed=%.2fs",
            path_obj,
            selection,
            duration_seconds,
            active_seconds,
            speech_ratio,
            decode_ms,
            vad_ms,
            time.perf_counter() - started,
        )
        return TranscriptionResult(
            asr_transcript="",
            language="",
            duration_seconds=duration_seconds,
            asr_model=self.model_name,
            asr_model_version=self.model_version,
            asr_model_revision=self.model_revision,
            asr_used_for_embedding=False,
            duration_after_vad=active_seconds,
            speech_ratio=speech_ratio,
            duration_sent_to_asr=0.0,
            asr_audio_selection=selection,
            status="no_speech",
        )

    def _transcribe_selected(
        self,
        path_obj: Path,
        started: float,
        clipped: np.ndarray,
        duration_seconds: float,
        active_seconds: float,
        speech_ratio: float,
        selection: str,
        decode_ms: float,
        vad_ms: float,
    ) -> TranscriptionResult:
        """Classify and transcribe the selected samples, then gate fusion."""
        assert self._model is not None
        sent_seconds = len(clipped) / TARGET_SAMPLE_RATE

        # Detector and Qwen must judge exactly the same capped samples. The
        # detector fails closed if it cannot cover all of this array.
        audio_class_started = time.perf_counter()
        audio_class_result: AudioClassResult | None = (
            self.audio_classifier.classify(clipped, TARGET_SAMPLE_RATE)
            if self.audio_classifier is not None
            else None
        )
        audio_class_ms = (time.perf_counter() - audio_class_started) * 1000.0

        with temporary_wav_path() as wav_path:
            write_wav_16k_mono(clipped, wav_path)
            model_started = time.perf_counter()
            try:
                results = self._model.transcribe(
                    audio=str(wav_path),
                    language=None,
                )
            except Exception as err:  # noqa: BLE001 — stable status boundary
                logger.warning(
                    "Qwen ASR transcription_failed for %s: %s: %s",
                    path_obj,
                    type(err).__name__,
                    err,
                )
                return _empty_result(
                    self.model_name,
                    self.model_version,
                    "transcription_failed",
                    self.model_revision,
                    selection,
                )
            model_ms = (time.perf_counter() - model_started) * 1000.0

        result = _first_qwen_result(results)
        transcript = str(_qwen_result_value(result, "text") or "").strip()
        language = _normalize_language_output(_qwen_result_value(result, "language"))

        compression_ratio = _transcript_compression_ratio(transcript)
        if not transcript:
            status = "no_speech"
            used_for_embedding: bool = False
            gate_reason: str | None = None
        else:
            status = "ok"
            # Density rules judge the audio Qwen actually heard. Under timeline
            # probes the capped activity can exceed the sent seconds (a span's
            # activity is counted in full but only its budget share is sent),
            # which under-stated density both ways.
            used_for_embedding, gate_reason = derive_qwen_trust_decision(
                transcript,
                speech_ratio,
                sent_seconds,
                min_speech_ratio=self.min_speech_ratio,
                max_compression_ratio=self.max_compression_ratio,
                max_chars_per_speech_second=self.max_chars_per_speech_second,
                long_clip_seconds=self.long_clip_seconds,
                min_long_clip_density=self.min_long_clip_density,
                audio_class=(
                    audio_class_result.audio_class if audio_class_result is not None else None
                ),
                enforce_audio_class=self.audio_classifier is not None,
            )

        elapsed = time.perf_counter() - started
        logger.info(
            "Qwen ASR %s: path=%s selection=%s dur=%.2fs active=%.2fs sent=%.2fs ratio=%.2f "
            "compress=%s audio_class=%s audio_cov=%s audio_detail=%s used=%s "
            "reason=%s lang=%s chars=%d decode_ms=%.0f vad_ms=%.0f "
            "audio_class_ms=%.0f model_ms=%.0f elapsed=%.2fs",
            status,
            path_obj,
            selection,
            duration_seconds,
            active_seconds,
            sent_seconds,
            speech_ratio,
            f"{compression_ratio:.2f}" if compression_ratio is not None else "n/a",
            audio_class_result.audio_class if audio_class_result else "-",
            f"{audio_class_result.coverage_ratio:.3f}" if audio_class_result else "-",
            (audio_class_result.detail or "-") if audio_class_result else "-",
            used_for_embedding,
            gate_reason or "-",
            language,
            len(transcript),
            decode_ms,
            vad_ms,
            audio_class_ms,
            model_ms,
            elapsed,
        )

        return TranscriptionResult(
            asr_transcript=transcript,
            language=language,
            duration_seconds=duration_seconds,
            asr_model=self.model_name,
            asr_model_version=self.model_version,
            asr_model_revision=self.model_revision,
            asr_used_for_embedding=used_for_embedding,
            duration_after_vad=active_seconds,
            speech_ratio=speech_ratio,
            duration_sent_to_asr=sent_seconds,
            max_compression_ratio=compression_ratio,
            asr_trust_reason=gate_reason,
            audio_class=(audio_class_result.audio_class if audio_class_result else None),
            audio_class_confidence=(audio_class_result.confidence if audio_class_result else None),
            audio_class_model=(audio_class_result.model_name if audio_class_result else None),
            audio_class_revision=(
                audio_class_result.resolved_revision if audio_class_result else None
            ),
            audio_class_coverage=(
                audio_class_result.coverage_ratio if audio_class_result else None
            ),
            asr_audio_selection=selection,
            status=status,
        )
