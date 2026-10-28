"""Deterministic energy-activity pre-gate for the Qwen ASR path.

This NumPy-only VAD detects audible activity, not verified speech. Its
conservative threshold skips only obvious silence; the audio classifier and
transcript policy decide trust. Long clips use bounded windows spread across
the active timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Folded into the Qwen `asr_model_version` (qwen-v5) so backend rows can
# quarantine when the activity-detection method itself changes.
VAD_METHOD_VERSION: str = "energy-vad-v1"

# Calibrated conservatively to near-digital silence. Retune only with fresh
# evidence and a method-version bump so quiet speech is never silently skipped.
DEFAULT_ENERGY_THRESHOLD_DBFS: float = -55.0
DEFAULT_FRAME_SECONDS: float = 0.03
DEFAULT_MERGE_GAP_SECONDS: float = 0.30
DEFAULT_MIN_INTERVAL_SECONDS: float = 0.20
DEFAULT_PAD_SECONDS: float = 0.25

# dBFS floor reported for an all-zero frame (log of zero is undefined).
_SILENCE_FLOOR_DBFS: float = -120.0


@dataclass(frozen=True)
class VadConfig:
    """Settings for detecting audible activity.

    The service uses the frozen defaults; changing them changes the audio
    selected for ASR.
    """

    energy_threshold_dbfs: float = DEFAULT_ENERGY_THRESHOLD_DBFS
    frame_seconds: float = DEFAULT_FRAME_SECONDS
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    pad_seconds: float = DEFAULT_PAD_SECONDS


def _frame_dbfs(samples: np.ndarray, frame_length: int) -> np.ndarray:
    """Per-frame RMS energy in dBFS for int16 mono samples."""
    if len(samples) == 0:
        return np.empty(0, dtype=np.float64)
    frame_count = max(1, int(np.ceil(len(samples) / frame_length)))
    padded = np.zeros(frame_count * frame_length, dtype=np.float64)
    padded[: len(samples)] = samples.astype(np.float64) / 32768.0
    frames = padded.reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    dbfs = np.full(frame_count, _SILENCE_FLOOR_DBFS, dtype=np.float64)
    nonzero = rms > 0.0
    dbfs[nonzero] = 20.0 * np.log10(rms[nonzero])
    return dbfs


def _merge_intervals(
    intervals: list[tuple[float, float]], max_gap: float
) -> list[tuple[float, float]]:
    """Merge sorted intervals whose gap is at most `max_gap` seconds."""
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detect_active_intervals(
    samples: np.ndarray,
    sample_rate: int,
    config: VadConfig | None = None,
) -> list[tuple[float, float]]:
    """Detect audible-activity intervals `(start_s, end_s)` in int16 mono audio.

    Frames above the energy threshold are merged (gap tolerance), short blips
    are dropped, and survivors are padded so interval edges never clip a word.
    Returns [] for silence.
    """
    config = config or VadConfig()
    if len(samples) == 0 or sample_rate <= 0:
        return []
    frame_length = max(1, int(round(config.frame_seconds * sample_rate)))
    dbfs = _frame_dbfs(samples, frame_length)
    active = dbfs >= config.energy_threshold_dbfs
    if not bool(active.any()):
        return []

    duration = len(samples) / sample_rate
    frame_seconds = frame_length / sample_rate
    raw: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and run_start is None:
            run_start = index
        elif not is_active and run_start is not None:
            raw.append((run_start * frame_seconds, index * frame_seconds))
            run_start = None
    if run_start is not None:
        raw.append((run_start * frame_seconds, min(len(active) * frame_seconds, duration)))

    merged = _merge_intervals(raw, config.merge_gap_seconds)
    kept = [(start, end) for start, end in merged if end - start >= config.min_interval_seconds]
    padded = [
        (max(0.0, start - config.pad_seconds), min(duration, end + config.pad_seconds))
        for start, end in kept
    ]
    # Padding can overlap neighbors; re-merge with zero tolerance.
    return _merge_intervals(padded, 0.0)


def total_interval_seconds(intervals: list[tuple[float, float]]) -> float:
    """Sum of interval durations in seconds."""
    return sum(end - start for start, end in intervals)


def select_speech_windows(
    intervals: list[tuple[float, float]],
    cap_seconds: float,
    num_windows: int,
) -> list[tuple[float, float]]:
    """Bound total active audio at `cap_seconds`, sampled across the timeline.

    When total activity fits the cap, intervals pass through unchanged. When it
    does not, the concatenated *active* timeline is partitioned into
    `num_windows` equal spans and the first `cap / num_windows` active seconds
    of each span are kept — deterministic coverage of beginning, middle, and
    end instead of blindly truncating to the head of the clip. Returned windows
    are real-time `(start_s, end_s)` ranges, merged where adjacent.
    """
    if cap_seconds <= 0.0 or num_windows <= 0:
        raise ValueError("cap_seconds and num_windows must be positive")
    total = total_interval_seconds(intervals)
    if total <= cap_seconds:
        return list(intervals)

    span_length = total / num_windows
    budget_per_window = cap_seconds / num_windows
    selected: list[tuple[float, float]] = []
    for window_index in range(num_windows):
        window_start = window_index * span_length
        window_end = window_start + min(budget_per_window, span_length)
        # Map the [window_start, window_end) range on the active-time axis
        # back to real-time sub-intervals.
        active_cursor = 0.0
        for start, end in intervals:
            length = end - start
            take_from = max(window_start, active_cursor)
            take_to = min(window_end, active_cursor + length)
            if take_to > take_from:
                selected.append(
                    (start + (take_from - active_cursor), start + (take_to - active_cursor))
                )
            active_cursor += length
            if active_cursor >= window_end:
                break
    return _merge_intervals(sorted(selected), 0.0)


def slice_samples(
    samples: np.ndarray,
    sample_rate: int,
    windows: list[tuple[float, float]],
) -> np.ndarray:
    """Concatenate the sample ranges covered by `windows` (real-time seconds)."""
    if not windows:
        return samples[:0]
    pieces = [
        samples[int(round(start * sample_rate)) : int(round(end * sample_rate))]
        for start, end in windows
    ]
    return np.concatenate(pieces) if pieces else samples[:0]


def merge_touching_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort intervals and merge those that overlap or touch (zero gap)."""
    return _merge_intervals(sorted(intervals), 0.0)


def select_windows_by_chunk_priority(
    chunk_intervals: list[list[tuple[float, float]]],
    cap_seconds: float,
) -> list[tuple[float, float]]:
    """Allocate `cap_seconds` of activity to probe chunks in decode order.

    `chunk_intervals` holds one interval list per decoded chunk, in the order
    the chunks were probed: the probe plan visits its anchor first (the end of
    the media for the last span, the centre for a middle span) and extends
    away from it only when the anchor is quiet, so activity in an earlier
    chunk outranks activity in a later extension chunk even when the
    extension lies earlier on the timeline. A chronological pick would have
    spent the whole budget on the extension and dropped the anchor. Within
    one chunk activity is taken chronologically. The selected ranges are
    returned merged and in timeline order, ready to be sliced for the model.
    """
    if cap_seconds <= 0.0:
        raise ValueError("cap_seconds must be positive")
    remaining = cap_seconds
    selected: list[tuple[float, float]] = []
    for intervals in chunk_intervals:
        if remaining <= 1e-6:
            break
        if not intervals:
            continue
        taken = select_speech_windows(sorted(intervals), remaining, 1)
        selected.extend(taken)
        remaining -= total_interval_seconds(taken)
    return merge_touching_intervals(selected)


def slice_decoded_spans(
    spans: list[tuple[float, np.ndarray]],
    sample_rate: int,
    windows: list[tuple[float, float]],
) -> np.ndarray:
    """Concatenate the sample ranges covered by `windows` from placed spans.

    Each span is `(start_seconds, samples)` on the real-time axis; spans need
    not be adjacent or ordered, so a window is cut from every span it overlaps
    and the pieces are emitted in window order, then span order.
    """
    pieces: list[np.ndarray] = []
    ordered = sorted(spans, key=lambda span: span[0])
    for window_start, window_end in windows:
        for span_start, samples in ordered:
            span_end = span_start + len(samples) / sample_rate
            cut_start = max(window_start, span_start)
            cut_end = min(window_end, span_end)
            if cut_end <= cut_start:
                continue
            first = int(round((cut_start - span_start) * sample_rate))
            last = int(round((cut_end - span_start) * sample_rate))
            pieces.append(samples[first:last])
    if not pieces:
        return np.empty(0, dtype=np.int16)
    return np.concatenate(pieces)


def plan_timeline_probes(
    duration_seconds: float,
    num_windows: int,
    probe_seconds: float,
    max_probe_seconds: float,
) -> list[list[tuple[float, float]]]:
    """Plan bounded decode chunks for each equal span of long media.

    Returns one ordered chunk list per span. The first span grows forward from
    the start of the media, the last span grows backward from its end, and any
    middle span grows outward from its centre (first the chunk around the
    centre, then alternately after and before it). Each list stays inside its
    span and totals at most `max_probe_seconds`, so the beginning, middle, and
    end are sampled without decoding the whole timeline; speech between the
    probes can still be missed. The transcriber
    decodes chunks in the returned order and stops early once a span has
    yielded its activity budget.
    """
    if duration_seconds <= 0.0 or num_windows <= 0 or probe_seconds <= 0.0:
        raise ValueError("duration_seconds, num_windows and probe_seconds must be positive")
    if max_probe_seconds < probe_seconds:
        raise ValueError("max_probe_seconds must be at least probe_seconds")
    span = duration_seconds / num_windows
    plan: list[list[tuple[float, float]]] = []
    for index in range(num_windows):
        span_start = index * span
        span_end = min(duration_seconds, (index + 1) * span)
        chunks: list[tuple[float, float]] = []
        if index == num_windows - 1 and num_windows > 1:
            cursor = span_end
            while cursor > span_start and _planned(chunks) < max_probe_seconds:
                start = max(span_start, cursor - probe_seconds)
                chunks.append((start, cursor))
                cursor = start
        elif index == 0:
            cursor = span_start
            while cursor < span_end and _planned(chunks) < max_probe_seconds:
                end = min(span_end, cursor + probe_seconds)
                chunks.append((cursor, end))
                cursor = end
        else:
            centre = (span_start + span_end) / 2.0
            after = min(span_end, centre + probe_seconds / 2.0)
            before = max(span_start, centre - probe_seconds / 2.0)
            chunks.append((before, after))
            grow_after = True
            while (after < span_end or before > span_start) and _planned(
                chunks
            ) < max_probe_seconds:
                if grow_after and after < span_end:
                    end = min(span_end, after + probe_seconds)
                    chunks.append((after, end))
                    after = end
                elif before > span_start:
                    start = max(span_start, before - probe_seconds)
                    chunks.append((start, before))
                    before = start
                grow_after = not grow_after
        plan.append(_clamp_to_budget(chunks, max_probe_seconds))
    return plan


def _planned(chunks: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in chunks)


def _clamp_to_budget(
    chunks: list[tuple[float, float]], max_probe_seconds: float
) -> list[tuple[float, float]]:
    """Trim the last chunk so a span never plans more than its bound."""
    total = 0.0
    kept: list[tuple[float, float]] = []
    for start, end in chunks:
        remaining = max_probe_seconds - total
        if remaining <= 0.0:
            break
        length = end - start
        if length > remaining:
            # Shorten toward the chunk's growth direction: forward chunks keep
            # their start, backward chunks keep their end.
            if kept and start < kept[-1][0]:
                start = end - remaining
            else:
                end = start + remaining
            length = remaining
        kept.append((start, end))
        total += length
    return kept
