"""Unit tests for the standalone energy VAD pre-gate (postprism/asr/vad.py).

Pure numpy — no model, no PyAV, no network. Pins the properties the Qwen
pre-gate relies on: silence yields nothing, activity detection is padded and
merged, and the speech-cap windows are deterministic and spread across the
timeline (never just the head of the clip).
"""

from __future__ import annotations

import numpy as np
import pytest

from postprism.asr.vad import (
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

SAMPLE_RATE = 16000


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


class TestDetectActiveIntervals:
    def test_silence_yields_no_intervals(self) -> None:
        assert detect_active_intervals(_silence(3.0), SAMPLE_RATE) == []

    def test_empty_input_yields_no_intervals(self) -> None:
        assert detect_active_intervals(_silence(0.0), SAMPLE_RATE) == []

    def test_tone_burst_is_detected_with_padding(self) -> None:
        samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
        intervals = detect_active_intervals(samples, SAMPLE_RATE)
        assert len(intervals) == 1
        start, end = intervals[0]
        # Padded outward by pad_seconds (0.25) but clamped to the clip.
        assert start == pytest.approx(0.75, abs=0.1)
        assert end == pytest.approx(3.25, abs=0.1)

    def test_nearby_bursts_merge_across_small_gaps(self) -> None:
        # 0.2 s gap < merge tolerance (0.3 s) ⇒ one interval, not two.
        samples = np.concatenate([_tone(1.0), _silence(0.2), _tone(1.0)])
        intervals = detect_active_intervals(samples, SAMPLE_RATE)
        assert len(intervals) == 1

    def test_short_blip_is_dropped(self) -> None:
        # 0.1 s blip < min_interval_seconds (0.2 s) ⇒ ignored as noise.
        samples = np.concatenate([_silence(2.0), _tone(0.1), _silence(2.0)])
        assert detect_active_intervals(samples, SAMPLE_RATE) == []

    def test_quiet_audio_below_threshold_is_inactive(self) -> None:
        # -55 dBFS default: a signal at ~-70 dBFS must not register.
        samples = _tone(2.0, amplitude=0.0003)
        assert detect_active_intervals(samples, SAMPLE_RATE) == []

    def test_threshold_is_configurable(self) -> None:
        quiet = _tone(2.0, amplitude=0.0003)
        config = VadConfig(energy_threshold_dbfs=-90.0)
        assert detect_active_intervals(quiet, SAMPLE_RATE, config) != []


class TestSelectSpeechWindows:
    def test_under_cap_passes_through_unchanged(self) -> None:
        intervals = [(0.0, 10.0), (20.0, 25.0)]
        assert select_speech_windows(intervals, 60.0, 3) == intervals

    def test_over_cap_total_equals_cap(self) -> None:
        windows = select_speech_windows([(0.0, 90.0)], 60.0, 3)
        assert total_interval_seconds(windows) == pytest.approx(60.0)

    def test_over_cap_samples_across_the_timeline_not_just_the_head(self) -> None:
        # 90 s of continuous activity, 30 s cap: a head-truncating policy would
        # keep only [0, 30). The windows must reach the final third.
        windows = select_speech_windows([(0.0, 90.0)], 30.0, 3)
        assert total_interval_seconds(windows) == pytest.approx(30.0)
        assert any(start >= 60.0 for start, _ in windows)
        assert any(30.0 <= start < 60.0 for start, _ in windows)

    def test_windows_map_back_through_fragmented_intervals(self) -> None:
        # Active time is fragmented; selection budgets on the ACTIVE axis and
        # maps back to real time, so late fragments are still represented.
        intervals = [(0.0, 20.0), (40.0, 60.0), (80.0, 100.0)]  # 60 s active
        windows = select_speech_windows(intervals, 30.0, 3)
        assert total_interval_seconds(windows) == pytest.approx(30.0)
        assert any(start >= 80.0 for start, _ in windows)

    def test_selection_is_deterministic(self) -> None:
        intervals = [(0.0, 45.0), (50.0, 95.0)]
        first = select_speech_windows(intervals, 20.0, 4)
        second = select_speech_windows(intervals, 20.0, 4)
        assert first == second

    def test_rejects_non_positive_cap_or_windows(self) -> None:
        with pytest.raises(ValueError):
            select_speech_windows([(0.0, 10.0)], 0.0, 3)
        with pytest.raises(ValueError):
            select_speech_windows([(0.0, 10.0)], 30.0, 0)


class TestSliceSamples:
    def test_concatenates_window_ranges(self) -> None:
        samples = np.arange(SAMPLE_RATE * 10, dtype=np.int16)
        windows = [(0.0, 1.0), (5.0, 6.5)]
        clipped = slice_samples(samples, SAMPLE_RATE, windows)
        assert len(clipped) == int(2.5 * SAMPLE_RATE)
        # First selected sample is timeline start; the tail comes from t=5 s on.
        assert clipped[0] == samples[0]
        assert clipped[SAMPLE_RATE] == samples[5 * SAMPLE_RATE]

    def test_no_windows_yields_empty(self) -> None:
        samples = np.arange(SAMPLE_RATE, dtype=np.int16)
        assert len(slice_samples(samples, SAMPLE_RATE, [])) == 0


class TestPlacedSpans:
    """Probe chunks are placed at their decoded start and need not be adjacent."""

    def test_touching_and_overlapping_intervals_merge_across_spans(self) -> None:
        merged = merge_touching_intervals([(60.0, 70.0), (10.0, 20.0), (20.0, 25.0), (68.0, 75.0)])
        assert merged == [(10.0, 25.0), (60.0, 75.0)]

    def test_windows_are_cut_from_every_span_they_overlap(self) -> None:
        # Span A covers 30-60 s (the stream began at 30 s inside a 0-60 s
        # chunk); span B covers 60-120 s. Sample values encode absolute time.
        span_a = (30.0, np.arange(30 * SAMPLE_RATE, dtype=np.int16))
        span_b = (60.0, np.arange(60 * SAMPLE_RATE, dtype=np.int16) + 1000)
        windows = [(58.0, 63.0), (100.0, 101.0)]

        clipped = slice_decoded_spans([span_b, span_a], SAMPLE_RATE, windows)

        assert len(clipped) == 6 * SAMPLE_RATE
        assert clipped[0] == span_a[1][28 * SAMPLE_RATE]  # 58 s inside span A
        assert clipped[2 * SAMPLE_RATE] == span_b[1][0]  # 60 s is span B's first sample
        assert clipped[5 * SAMPLE_RATE] == span_b[1][40 * SAMPLE_RATE]

    def test_window_in_a_gap_between_spans_yields_nothing(self) -> None:
        spans = [(0.0, _silence(10.0)), (30.0, _silence(10.0))]
        assert len(slice_decoded_spans(spans, SAMPLE_RATE, [(12.0, 20.0)])) == 0
        assert len(slice_decoded_spans(spans, SAMPLE_RATE, [])) == 0

    def test_budget_follows_chunk_priority_then_returns_chronological_windows(self) -> None:
        # Last span: the end chunk (probed first) found 5 s at 695-700, the
        # earlier extension chunk found 10 s at 600-610. A chronological pick
        # would spend the whole 10 s budget on 600-610 and drop the tail.
        windows = select_windows_by_chunk_priority(
            [[(695.0, 700.0)], [(600.0, 610.0)]], cap_seconds=10.0
        )
        assert windows == [(600.0, 605.0), (695.0, 700.0)]

    def test_chunk_priority_passes_everything_through_under_the_cap(self) -> None:
        assert select_windows_by_chunk_priority([[(50.0, 52.0)], [], [(10.0, 13.0)]], 10.0) == [
            (10.0, 13.0),
            (50.0, 52.0),
        ]
        assert select_windows_by_chunk_priority([[], []], 10.0) == []
        with pytest.raises(ValueError):
            select_windows_by_chunk_priority([[(0.0, 1.0)]], 0.0)


class TestPlanTimelineProbes:
    """Long media is probed at the beginning, middle, and end, never fully decoded."""

    def test_first_span_grows_forward_middle_around_its_centre_last_backward_from_the_end(
        self,
    ) -> None:
        plan = plan_timeline_probes(900.0, 3, 60.0, 120.0)
        assert plan == [
            [(0.0, 60.0), (60.0, 120.0)],
            [(420.0, 480.0), (480.0, 540.0)],
            [(840.0, 900.0), (780.0, 840.0)],
        ]

    def test_the_final_seconds_are_always_inside_the_last_probe(self) -> None:
        for duration in (601.0, 700.0, 900.0, 3600.0):
            plan = plan_timeline_probes(duration, 3, 60.0, 120.0)
            first_chunk_start, first_chunk_end = plan[-1][0]
            assert first_chunk_end == pytest.approx(duration)
            assert first_chunk_start == pytest.approx(duration - 60.0)

    def test_probes_never_leave_their_span_and_stay_within_the_bound(self) -> None:
        plan = plan_timeline_probes(100.0, 3, 60.0, 120.0)
        span = 100.0 / 3
        for index, chunks in enumerate(plan):
            assert sum(end - start for start, end in chunks) <= 120.0 + 1e-9
            for start, end in chunks:
                assert start >= index * span - 1e-9
                assert end <= (index + 1) * span + 1e-9
                assert end > start

    def test_single_and_double_window_plans_keep_the_anchors(self) -> None:
        assert plan_timeline_probes(900.0, 1, 60.0, 120.0) == [[(0.0, 60.0), (60.0, 120.0)]]
        assert plan_timeline_probes(900.0, 2, 60.0, 120.0) == [
            [(0.0, 60.0), (60.0, 120.0)],
            [(840.0, 900.0), (780.0, 840.0)],
        ]

    def test_rejects_non_positive_inputs(self) -> None:
        with pytest.raises(ValueError):
            plan_timeline_probes(0.0, 3, 60.0, 120.0)
        with pytest.raises(ValueError):
            plan_timeline_probes(900.0, 0, 60.0, 120.0)
        with pytest.raises(ValueError):
            plan_timeline_probes(900.0, 3, 0.0, 120.0)
        with pytest.raises(ValueError):
            plan_timeline_probes(900.0, 3, 60.0, 30.0)
