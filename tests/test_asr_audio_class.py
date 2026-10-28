"""Tests for the opt-in AudioSet speech/music discriminator (astgate-v1).

The AST checkpoint is never downloaded here: `_load_audio_class_tagger` is
monkeypatched with a fake tagger that returns a plain numpy score matrix, so
these tests need neither torch nor transformers. What they pin:

- the pure `derive_audio_class` truth table (speech / music / mixed /
  uncertain), including the "instrumental backing track is still speech" case
  that keeps ordinary talking-head reels eligible;
- the COVERAGE rule: a verdict is only valid when the scored windows cover
  every sample the transcript came from, and a shortfall fails closed to
  `incomplete` instead of approving the clip from a prefix of it;
- `AudioClassifier` aggregation and its fail-closed behaviour when the
  detector is missing, mislabelled, or raises;
- that the runtime class set and trust-reason set stay identical to the
  backend CHECK constraints.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from postprism.asr.audio_class import (
    AUDIO_CLASS_COVERAGE_POLICY,
    AUDIO_CLASS_INCOMPLETE,
    AUDIO_CLASS_METHOD_VERSION,
    AUDIO_CLASS_MIXED,
    AUDIO_CLASS_MUSIC,
    AUDIO_CLASS_SPEECH,
    AUDIO_CLASS_UNCERTAIN,
    AUDIO_CLASSES,
    AUDIO_CLASSES_WITHHELD,
    DEFAULT_AUDIO_CLASS_CHUNK_SECONDS,
    DEFAULT_AUDIO_CLASS_MAX_CHUNKS,
    DEFAULT_AUDIO_CLASS_MODEL,
    DEFAULT_AUDIO_CLASS_REVISION,
    DEFAULT_MUSIC_FLOOR,
    DEFAULT_SINGING_FLOOR,
    DEFAULT_SPEECH_FLOOR,
    MIN_CHUNK_SECONDS,
    MUSIC_LABELS,
    SINGING_LABELS,
    SPEECH_LABELS,
    AudioClassifier,
    derive_audio_class,
    plan_chunk_spans,
)

TARGET_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Fake tagger — stands in for the AST checkpoint + feature extractor
# ---------------------------------------------------------------------------


class _FakeTagger:
    """Returns a caller-supplied per-chunk score matrix over a fake AudioSet.

    `chunk_scores` is a list of `{label: score}` dicts, one per chunk the
    classifier is expected to submit. The recorded `chunk_lengths` prove which
    windows it actually scored.
    """

    def __init__(
        self,
        chunk_scores: list[dict[str, float]] | None = None,
        raise_on_scores: BaseException | None = None,
        id2label: dict[int, str] | None = None,
        resolved_revision: str | None = DEFAULT_AUDIO_CLASS_REVISION,
    ) -> None:
        self.chunk_scores = chunk_scores if chunk_scores is not None else [{}]
        self.raise_on_scores = raise_on_scores
        self.id2label = (
            id2label
            if id2label is not None
            else dict(enumerate(sorted(SPEECH_LABELS | SINGING_LABELS | MUSIC_LABELS)))
        )
        self.resolved_revision = resolved_revision
        self.dtype = "torch.float32"
        self.chunk_lengths: list[int] = []
        self.calls = 0

    def scores(self, chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
        self.calls += 1
        self.chunk_lengths = [len(chunk) for chunk in chunks]
        if self.raise_on_scores is not None:
            raise self.raise_on_scores
        label_to_index = {label: index for index, label in self.id2label.items()}
        matrix = np.zeros((len(chunks), len(self.id2label)), dtype=np.float64)
        for row, chunk_score in enumerate(self.chunk_scores[: len(chunks)]):
            for label, value in chunk_score.items():
                matrix[row, label_to_index[label]] = value
        return matrix


def _install(monkeypatch: pytest.MonkeyPatch, tagger: _FakeTagger) -> list[tuple]:
    """Patch the loader and return the list of (model, device, revision) calls."""
    calls: list[tuple] = []

    def _loader(model_name: str, device: str, revision: str | None = None):
        calls.append((model_name, device, revision))
        return tagger

    monkeypatch.setattr("postprism.asr.audio_class._load_audio_class_tagger", _loader)
    return calls


def _warmed(
    monkeypatch: pytest.MonkeyPatch,
    tagger: _FakeTagger,
    *,
    chunk_seconds: float | None = None,
    max_chunks: int | None = None,
) -> AudioClassifier:
    """Warm the fixed classifier with optional test-only coverage limits."""
    _install(monkeypatch, tagger)
    classifier = AudioClassifier()
    if chunk_seconds is not None:
        classifier.chunk_seconds = chunk_seconds
    if max_chunks is not None:
        classifier.max_chunks = max_chunks
    classifier.warmup()
    return classifier


def _tone(seconds: float) -> np.ndarray:
    """int16 mono 16 kHz samples, matching what `audio_prep` produces."""
    t = np.arange(int(seconds * TARGET_SAMPLE_RATE)) / TARGET_SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


class TestDeriveAudioClass:
    def test_clear_speech_is_speech(self) -> None:
        audio_class, confidence = derive_audio_class(speech=0.93, music=0.05, singing=0.01)
        assert audio_class == AUDIO_CLASS_SPEECH
        assert confidence == pytest.approx(0.93)

    def test_instrumental_backing_track_under_speech_is_still_speech(self) -> None:
        # The common social-video shape: a talking head over a music bed. The
        # transcript comes from the speech, so it stays eligible — keying
        # `mixed` on VOCAL music is what keeps this case from being lost.
        audio_class, _ = derive_audio_class(speech=0.88, music=0.79, singing=0.04)
        assert audio_class == AUDIO_CLASS_SPEECH

    def test_sung_vocals_without_speech_are_music(self) -> None:
        audio_class, confidence = derive_audio_class(speech=0.08, music=0.96, singing=0.61)
        assert audio_class == AUDIO_CLASS_MUSIC
        assert confidence == pytest.approx(0.96)

    def test_instrumental_only_is_music(self) -> None:
        audio_class, _ = derive_audio_class(speech=0.04, music=0.91, singing=0.02)
        assert audio_class == AUDIO_CLASS_MUSIC

    def test_singing_alone_is_music_even_below_the_music_floor(self) -> None:
        audio_class, confidence = derive_audio_class(speech=0.10, music=0.20, singing=0.55)
        assert audio_class == AUDIO_CLASS_MUSIC
        assert confidence == pytest.approx(0.55)

    def test_speech_over_sung_vocals_is_mixed(self) -> None:
        audio_class, confidence = derive_audio_class(speech=0.81, music=0.90, singing=0.44)
        assert audio_class == AUDIO_CLASS_MIXED
        assert confidence == pytest.approx(0.44)

    def test_nothing_above_any_floor_is_uncertain(self) -> None:
        audio_class, confidence = derive_audio_class(speech=0.11, music=0.09, singing=0.02)
        assert audio_class == AUDIO_CLASS_UNCERTAIN
        assert confidence == pytest.approx(0.11)

    def test_floors_are_inclusive_and_overridable(self) -> None:
        at_floor, _ = derive_audio_class(speech=DEFAULT_SPEECH_FLOOR, music=0.0, singing=0.0)
        assert at_floor == AUDIO_CLASS_SPEECH
        # A stricter speech floor demotes the same scores to uncertain.
        stricter, _ = derive_audio_class(
            speech=DEFAULT_SPEECH_FLOOR,
            music=0.0,
            singing=0.0,
            speech_floor=DEFAULT_SPEECH_FLOOR + 0.1,
        )
        assert stricter == AUDIO_CLASS_UNCERTAIN

    def test_label_groups_are_disjoint_and_non_empty(self) -> None:
        assert SPEECH_LABELS and SINGING_LABELS and MUSIC_LABELS
        assert not SPEECH_LABELS & SINGING_LABELS
        assert not SPEECH_LABELS & MUSIC_LABELS
        assert not SINGING_LABELS & MUSIC_LABELS

    def test_speech_is_the_only_class_that_is_not_withheld(self) -> None:
        assert AUDIO_CLASSES - {AUDIO_CLASS_SPEECH} == AUDIO_CLASSES_WITHHELD
        assert AUDIO_CLASS_INCOMPLETE in AUDIO_CLASSES_WITHHELD


# ---------------------------------------------------------------------------
# Coverage planning — pure, integer-exact, no audio needed
# ---------------------------------------------------------------------------


class TestPlanChunkSpans:
    def test_exact_multiple_covers_everything(self) -> None:
        spans, covered = plan_chunk_spans(30, size=10, min_size=1, max_chunks=6)
        assert spans == [(0, 10), (10, 20), (20, 30)]
        assert covered == 30

    def test_long_remainder_is_scored_as_its_own_window(self) -> None:
        spans, covered = plan_chunk_spans(25, size=10, min_size=1, max_chunks=6)
        assert spans == [(0, 10), (10, 20), (20, 25)]
        assert covered == 25

    def test_short_remainder_is_covered_by_a_backward_anchored_window(self) -> None:
        # The 0.5-sample-equivalent tail is NOT dropped (that would be a
        # coverage hole) and NOT padded (that would dilute the scores): the
        # last window slides back so it ends at the end of the audio.
        spans, covered = plan_chunk_spans(205, size=100, min_size=10, max_chunks=6)
        assert spans == [(0, 100), (100, 200), (105, 205)]
        assert covered == 205

    def test_clip_shorter_than_one_window_is_a_single_full_span(self) -> None:
        spans, covered = plan_chunk_spans(3, size=10, min_size=5, max_chunks=6)
        assert spans == [(0, 3)]
        assert covered == 3

    def test_max_chunks_shortfall_is_reported_not_hidden(self) -> None:
        spans, covered = plan_chunk_spans(90, size=10, min_size=1, max_chunks=2)
        assert spans == [(0, 10), (10, 20)]
        assert covered == 20  # < 90 ⇒ the caller must fail closed

    @pytest.mark.parametrize("total", [1, 7, 99, 100, 101, 199, 200, 201, 999])
    def test_spans_never_exceed_the_window_and_never_leave_a_gap(self, total: int) -> None:
        spans, covered = plan_chunk_spans(total, size=100, min_size=10, max_chunks=64)
        assert spans, "a non-empty clip must produce at least one span"
        assert all(0 <= start < end <= total for start, end in spans)
        # No window may exceed one mel window — AST would truncate its tail.
        assert all(end - start <= 100 for start, end in spans)
        # Contiguous or overlapping, so the union really is [0, covered).
        assert spans[0][0] == 0
        assert all(spans[index + 1][0] <= spans[index][1] for index in range(len(spans) - 1))
        assert covered == total

    def test_degenerate_inputs_are_empty(self) -> None:
        assert plan_chunk_spans(0, 10, 1, 6) == ([], 0)
        assert plan_chunk_spans(10, 0, 1, 6) == ([], 0)
        assert plan_chunk_spans(10, 10, 1, 0) == ([], 0)


# ---------------------------------------------------------------------------
# AudioClassifier
# ---------------------------------------------------------------------------


class TestAudioClassifier:
    def test_torchaudio_feature_backend_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "postprism.asr.audio_class.detect_feature_backend",
            lambda: "torchaudio",
        )

        with pytest.raises(RuntimeError, match="numpy feature backend"):
            AudioClassifier()

    def test_music_scores_classify_as_music(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tagger = _FakeTagger([{"Music": 0.97, "Singing": 0.66, "Speech": 0.07}])
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(_tone(6.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_MUSIC
        assert result.scores is not None
        assert result.scores.music == pytest.approx(0.97)
        assert result.scores.singing == pytest.approx(0.66)
        assert result.model_name == DEFAULT_AUDIO_CLASS_MODEL
        assert result.method_version == AUDIO_CLASS_METHOD_VERSION
        assert result.coverage_policy == AUDIO_CLASS_COVERAGE_POLICY
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.detail is None

    def test_group_score_is_the_max_over_member_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "Female speech, woman speaking" alone must raise the speech group.
        tagger = _FakeTagger(
            [{"Speech": 0.10, "Female speech, woman speaking": 0.88, "Music": 0.02}]
        )
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(_tone(4.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_SPEECH
        assert result.scores is not None
        assert result.scores.speech == pytest.approx(0.88)

    def test_singing_in_any_chunk_makes_a_talking_clip_mixed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Chunk 1 is speech, chunk 2 is a sung hook. Aggregating by MAX over
        # chunks is what turns this into the fail-closed `mixed` verdict.
        tagger = _FakeTagger(
            [
                {"Speech": 0.92, "Music": 0.10, "Singing": 0.01},
                {"Speech": 0.20, "Music": 0.95, "Singing": 0.58},
            ]
        )
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(_tone(20.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_MIXED
        assert result.scores is not None
        assert result.scores.speech == pytest.approx(0.92)
        assert result.scores.singing == pytest.approx(0.58)

    def test_audio_is_chunked_at_the_configured_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tagger = _FakeTagger([{"Speech": 0.9}] * 4)
        classifier = _warmed(monkeypatch, tagger, chunk_seconds=10.0)

        result = classifier.classify(_tone(25.0), TARGET_SAMPLE_RATE)

        # 25 s → 10 s + 10 s + a 5 s tail, all of it scored.
        assert [round(n / TARGET_SAMPLE_RATE, 2) for n in tagger.chunk_lengths] == [
            10.0,
            10.0,
            5.0,
        ]
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.scored_seconds == pytest.approx(25.0)

    def test_short_tail_is_covered_by_a_backward_anchored_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tagger = _FakeTagger([{"Speech": 0.9}] * 3)
        classifier = _warmed(monkeypatch, tagger, chunk_seconds=10.0)

        result = classifier.classify(_tone(10.2), TARGET_SAMPLE_RATE)

        # Two full windows, the second ending at 10.2 s: the 0.2 s tail is
        # scored (overlapped), never dropped and never zero-padded.
        assert [round(n / TARGET_SAMPLE_RATE, 2) for n in tagger.chunk_lengths] == [
            10.0,
            10.0,
        ]
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.audio_class == AUDIO_CLASS_SPEECH

    def test_incomplete_coverage_fails_closed_when_music_comes_last(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The dangerous shape: 10 s of talking, then 80 s of song. A detector
        # bounded to the first window would call the whole clip `speech` and
        # let the sung transcript fuse.
        tagger = _FakeTagger(
            [
                {"Speech": 0.95, "Music": 0.05, "Singing": 0.01},
                {"Speech": 0.04, "Music": 0.98, "Singing": 0.83},
            ]
        )
        classifier = _warmed(monkeypatch, tagger, chunk_seconds=10.0, max_chunks=1)

        with caplog.at_level("WARNING", logger="postprism.asr.audio_class"):
            result = classifier.classify(_tone(90.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_INCOMPLETE
        assert result.detail == "incomplete_coverage"
        assert result.coverage_ratio == pytest.approx(10.0 / 90.0)
        assert result.scores is None
        # It must not even guess: no partial verdict is produced.
        assert tagger.calls == 0
        assert any("failing closed" in record.message for record in caplog.records)

    def test_incomplete_coverage_fails_closed_when_music_comes_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The mirror shape: song first, talking later. `incomplete` is not a
        # dressed-up `music` verdict — it is "no verdict", in both directions.
        tagger = _FakeTagger(
            [
                {"Speech": 0.03, "Music": 0.99, "Singing": 0.88},
                {"Speech": 0.94, "Music": 0.06, "Singing": 0.02},
            ]
        )
        classifier = _warmed(monkeypatch, tagger, chunk_seconds=10.0, max_chunks=1)

        result = classifier.classify(_tone(90.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_INCOMPLETE
        assert result.scores is None
        assert tagger.calls == 0

    def test_full_coverage_within_max_chunks_still_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The shipped configuration: a 30 s ASR cap is 3 of 6 allowed windows,
        # so the default path is never `incomplete`.
        tagger = _FakeTagger([{"Speech": 0.9}] * 3)
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(_tone(30.0), TARGET_SAMPLE_RATE)

        assert len(tagger.chunk_lengths) == 3
        assert result.audio_class == AUDIO_CLASS_SPEECH
        assert result.coverage_ratio == pytest.approx(1.0)

    def test_detector_exception_fails_closed_as_uncertain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tagger = _FakeTagger(raise_on_scores=RuntimeError("CUDA OOM"))
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(_tone(4.0), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_UNCERTAIN
        assert result.detail == "detector_error"
        assert result.scores is None
        assert result.confidence == pytest.approx(0.0)

    def test_empty_audio_is_uncertain_without_calling_the_detector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tagger = _FakeTagger()
        classifier = _warmed(monkeypatch, tagger)

        result = classifier.classify(np.empty(0, dtype=np.int16), TARGET_SAMPLE_RATE)

        assert result.audio_class == AUDIO_CLASS_UNCERTAIN
        assert result.detail == "empty_audio"
        assert tagger.calls == 0

    def test_classify_before_warmup_raises(self) -> None:
        classifier = AudioClassifier()
        with pytest.raises(RuntimeError, match="warmup"):
            classifier.classify(_tone(4.0), TARGET_SAMPLE_RATE)

    def test_warmup_rejects_a_non_audioset_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A checkpoint whose labels are not the AudioSet ontology cannot be
        # scored by this policy — fail loudly at startup, never per-request.
        tagger = _FakeTagger(id2label={0: "LABEL_0", 1: "LABEL_1"})
        _install(monkeypatch, tagger)
        classifier = AudioClassifier()
        with pytest.raises(RuntimeError, match="AudioSet"):
            classifier.warmup()

    def test_warmup_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tagger = _FakeTagger([{"Speech": 0.9}])
        calls = _install(monkeypatch, tagger)
        classifier = AudioClassifier()
        classifier.warmup()
        classifier.warmup()

        assert [call[0] for call in calls] == [DEFAULT_AUDIO_CLASS_MODEL]
        assert classifier.is_warm

    def test_defaults_are_the_documented_constants(self) -> None:
        classifier = AudioClassifier()
        assert classifier.model_name == DEFAULT_AUDIO_CLASS_MODEL
        # Pinned to an exact commit by default — NOT the default branch.
        assert classifier.revision == DEFAULT_AUDIO_CLASS_REVISION
        assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_AUDIO_CLASS_REVISION or "")
        assert classifier.speech_floor == pytest.approx(DEFAULT_SPEECH_FLOOR)
        assert classifier.music_floor == pytest.approx(DEFAULT_MUSIC_FLOOR)
        assert classifier.singing_floor == pytest.approx(DEFAULT_SINGING_FLOOR)
        assert classifier.chunk_seconds == pytest.approx(DEFAULT_AUDIO_CLASS_CHUNK_SECONDS)
        assert classifier.max_chunks == DEFAULT_AUDIO_CLASS_MAX_CHUNKS
        assert classifier.coverage_policy == AUDIO_CLASS_COVERAGE_POLICY

    def test_default_chunking_covers_the_default_asr_cap(self) -> None:
        # A configuration where the detector cannot cover the ASR cap would
        # make every clip `incomplete`; this is the arithmetic that prevents it.
        from postprism.asr.transcriber import DEFAULT_QWEN_MAX_ACTIVE_SECONDS

        reachable = DEFAULT_AUDIO_CLASS_CHUNK_SECONDS * DEFAULT_AUDIO_CLASS_MAX_CHUNKS
        assert reachable >= DEFAULT_QWEN_MAX_ACTIVE_SECONDS
        assert MIN_CHUNK_SECONDS < DEFAULT_AUDIO_CLASS_CHUNK_SECONDS


# ---------------------------------------------------------------------------
# Revision pinning — the weights must BE the approved commit
# ---------------------------------------------------------------------------


class TestRevisionPin:
    def test_warmup_refuses_weights_that_are_not_the_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this closes: HF serves a different commit, every score
        shifts, and the rows still claim the approved policy."""
        tagger = _FakeTagger(resolved_revision="0" * 40)
        _install(monkeypatch, tagger)
        classifier = AudioClassifier()
        with pytest.raises(RuntimeError, match="not the fixed pin"):
            classifier.warmup()
        assert not classifier.is_warm

    def test_warmup_refuses_an_unverifiable_load_while_pinned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No resolved sha means the pin is decoration; fail at startup rather
        # than write rows that claim a verified checkpoint.
        tagger = _FakeTagger(resolved_revision=None)
        _install(monkeypatch, tagger)
        classifier = AudioClassifier()
        with pytest.raises(RuntimeError, match="cannot be verified"):
            classifier.warmup()

    def test_matching_pin_warms_and_records_the_resolved_sha(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved = DEFAULT_AUDIO_CLASS_REVISION
        classifier = _warmed(monkeypatch, _FakeTagger(resolved_revision=resolved))
        assert classifier.resolved_revision == resolved

    def test_resolved_revision_reaches_every_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolved = DEFAULT_AUDIO_CLASS_REVISION
        classifier = _warmed(
            monkeypatch,
            _FakeTagger([{"Speech": 0.9}], resolved_revision=resolved),
            max_chunks=1,
        )
        scored = classifier.classify(_tone(5.0), TARGET_SAMPLE_RATE)
        withheld = classifier.classify(np.zeros(0, dtype=np.int16), TARGET_SAMPLE_RATE)
        # Fail-closed rows are the ones an auditor most needs to attribute.
        assert scored.resolved_revision == resolved
        assert withheld.resolved_revision == resolved
