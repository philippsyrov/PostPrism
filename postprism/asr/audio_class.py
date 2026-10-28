"""Speech/music discriminator for the Qwen ASR trust gate.

The energy VAD detects activity, not speech. This AST gate only admits verified
speech; music, sung vocals, mixed, incomplete, and uncertain evidence fail
closed. It scores the exact capped samples sent to Qwen and requires full
coverage. The worker is synchronous and lazily loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Recorded in the fixed Qwen `asr_model_version` so incompatible rows remain
# distinguishable if this discriminator changes.
AUDIO_CLASS_METHOD_VERSION: str = "astgate-v1"

DEFAULT_AUDIO_CLASS_MODEL: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
# The approved Hugging Face commit. Warmup verifies the resolved SHA so
# upstream changes cannot alter the fixed policy under the same version.
DEFAULT_AUDIO_CLASS_REVISION: str = "f826b80d28226b62986cc218e5cec390b1096902"

# AST's fixed input window is 1024 mel frames ≈ 10.24 s at 16 kHz, so 10 s
# chunks fit one window each with no truncation. `max_chunks` bounds the work
# for long clips; the Qwen path caps activity at 30 s by default, which is
# 3 chunks — 6 leaves headroom while keeping pathological clips bounded.
DEFAULT_AUDIO_CLASS_CHUNK_SECONDS: float = 10.0
DEFAULT_AUDIO_CLASS_MAX_CHUNKS: int = 6
# A trailing remainder shorter than this is not scored as its own window —
# padding a fragment out to a full mel window dilutes the scores. Instead the
# final window is anchored to the END of the audio and extended backwards, so
# the remainder is still covered (see `plan_chunk_spans`).
MIN_CHUNK_SECONDS: float = 1.0

# The coverage rule recorded in `asr_model_version`.
AUDIO_CLASS_COVERAGE_POLICY: str = "full-window-v1"

AUDIO_CLASS_SPEECH: str = "speech"
AUDIO_CLASS_MUSIC: str = "music"
AUDIO_CLASS_MIXED: str = "mixed"
AUDIO_CLASS_INCOMPLETE: str = "incomplete"
AUDIO_CLASS_UNCERTAIN: str = "uncertain"
AUDIO_CLASSES: frozenset[str] = frozenset(
    {
        AUDIO_CLASS_SPEECH,
        AUDIO_CLASS_MUSIC,
        AUDIO_CLASS_MIXED,
        AUDIO_CLASS_INCOMPLETE,
        AUDIO_CLASS_UNCERTAIN,
    }
)
# Everything that is not an explicit `speech` verdict fails closed; this set
# exists so consumers can assert the complement rather than re-listing it.
AUDIO_CLASSES_WITHHELD: frozenset[str] = AUDIO_CLASSES - {AUDIO_CLASS_SPEECH}

# --- Frozen launch floors ----------------------------------------------------
# AST emits independent per-label sigmoids, not a softmax, so these are
# presence thresholds rather than shares and do not need to sum to anything.
# These are frozen as part of the accepted ASR policy.
DEFAULT_SPEECH_FLOOR: float = 0.50
DEFAULT_MUSIC_FLOOR: float = 0.50
# Sung vocals score lower than the `Speech` / `Music` parents on AudioSet
# heads, and this floor is the fail-closed direction, so it sits well below
# the other two.
DEFAULT_SINGING_FLOOR: float = 0.20

# --- AudioSet label groups (exact display names from the AST config) ---------
# Verified against MIT/ast-finetuned-audioset-10-10-0.4593's `id2label`.
SPEECH_LABELS: frozenset[str] = frozenset(
    {
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Speech synthesizer",
    }
)
# Vocal music only — the shape that yields plausible-looking lyrics.
SINGING_LABELS: frozenset[str] = frozenset(
    {
        "Singing",
        "Male singing",
        "Female singing",
        "Child singing",
        "Synthetic singing",
        "Choir",
        "Chant",
        "Mantra",
        "Rapping",
        "Humming",
        "Yodeling",
        "A capella",
    }
)
# Music parents, deliberately compact: the top-level nodes an AudioSet head
# scores reliably on any musical audio. Enumerating the ~200 genre/instrument
# leaves would add noise without adding separation.
MUSIC_LABELS: frozenset[str] = frozenset(
    {
        "Music",
        "Musical instrument",
        "Background music",
        "Theme music",
        "Soundtrack music",
        "Jingle (music)",
        "Song",
    }
)
# Anchors that must exist for a checkpoint to be scoreable by this policy.
REQUIRED_LABELS: frozenset[str] = frozenset({"Speech", "Music", "Singing"})


FEATURE_BACKEND_NUMPY: str = "numpy"
FEATURE_BACKEND_TORCHAUDIO: str = "torchaudio"


def detect_feature_backend() -> str:
    """Return the audio feature backend Transformers would use.

    The frozen ASR policy accepts only NumPy; AudioClassifier rejects torchaudio
    before loading weights. If Transformers is unavailable, return NumPy here
    and let warmup report the missing dependency.
    """
    try:
        from transformers.utils import is_speech_available  # noqa: PLC0415
    except ImportError:
        return FEATURE_BACKEND_NUMPY
    return FEATURE_BACKEND_TORCHAUDIO if is_speech_available() else FEATURE_BACKEND_NUMPY


@dataclass(frozen=True)
class AudioClassScores:
    """Max per-group AudioSet sigmoid scores over the submitted chunks."""

    speech: float
    music: float
    singing: float


@dataclass(frozen=True)
class AudioClassResult:
    """One clip's audio-class verdict plus the provenance to audit it."""

    audio_class: str
    confidence: float
    model_name: str
    method_version: str
    # Fraction of the offered samples the verdict actually looked at. The
    # verdict is only valid at 1.0; anything less is `incomplete`.
    coverage_ratio: float = 0.0
    scored_seconds: float = 0.0
    offered_seconds: float = 0.0
    feature_backend: str = FEATURE_BACKEND_NUMPY
    coverage_policy: str = AUDIO_CLASS_COVERAGE_POLICY
    # Configured revision and the sha transformers actually resolved.
    model_revision: str | None = None
    resolved_revision: str | None = None
    # Why a fail-closed verdict happened: `empty_audio`, `incomplete_coverage`,
    # `detector_error`, or None for a scored verdict.
    detail: str | None = None
    # `None` when the detector never produced scores (empty audio, incomplete
    # coverage, or it raised) — every fail-closed path.
    scores: AudioClassScores | None = None


def derive_audio_class(
    speech: float,
    music: float,
    singing: float,
    speech_floor: float = DEFAULT_SPEECH_FLOOR,
    music_floor: float = DEFAULT_MUSIC_FLOOR,
    singing_floor: float = DEFAULT_SINGING_FLOOR,
) -> tuple[str, float]:
    """Map three group scores to `(audio_class, confidence)`. Pure.

    Check order is part of the versioned policy:

    1. speech + sung vocals ⇒ `mixed` (fail closed; no source separation).
    2. speech alone ⇒ `speech` — instrumental backing music does NOT demote
       it, because the transcript still comes from the speech.
    3. music and/or sung vocals without speech ⇒ `music`.
    4. anything else ⇒ `uncertain`.

    `confidence` is the score that drove the verdict (the closest-to-the-line
    group score for `uncertain`), recorded so floors can be recalibrated from
    stored rows without re-running the detector.
    """
    speech_present = speech >= speech_floor
    music_present = music >= music_floor
    singing_present = singing >= singing_floor

    if speech_present and singing_present:
        return AUDIO_CLASS_MIXED, singing
    if speech_present:
        return AUDIO_CLASS_SPEECH, speech
    if music_present or singing_present:
        return AUDIO_CLASS_MUSIC, max(
            music if music_present else 0.0,
            singing if singing_present else 0.0,
        )
    return AUDIO_CLASS_UNCERTAIN, max(speech, music, singing)


def plan_chunk_spans(
    total_samples: int,
    size: int,
    min_size: int,
    max_chunks: int,
) -> tuple[list[tuple[int, int]], int]:
    """Plan `[start, end)` scoring windows and report how much they cover.

    Pure and integer-exact, so the coverage rule can be tested without audio.
    Returns `(spans, covered_samples)` where `covered_samples` counts the
    union of the spans (they are contiguous from 0, so it is the last end).

    Two properties the coverage policy depends on:

    1. **No window exceeds `size`.** AST truncates its input at one mel window,
       so an over-long chunk would silently drop its own tail.
    2. **A short trailing remainder is still covered.** Rather than dropping it
       (a coverage hole) or padding it (score dilution), the final window is
       anchored to the end of the audio and extended backwards, overlapping its
       predecessor. Overlap is harmless under the max-over-chunks aggregation.

    Together these mean the spans cover every sample whenever the audio fits in
    `max_chunks * size`; beyond that the shortfall is reported, never hidden.
    """
    if total_samples <= 0 or size <= 0 or max_chunks <= 0:
        return [], 0
    spans: list[tuple[int, int]] = []
    for start in range(0, total_samples, size):
        if len(spans) >= max_chunks:
            break
        end = min(start + size, total_samples)
        if end - start < min_size and spans:
            # Short remainder: add a full window anchored to the END of the
            # audio. It overlaps its predecessor (harmless under max-over-
            # chunks) and leaves no gap, because the remainder is shorter than
            # one window by construction.
            spans.append((max(0, total_samples - size), total_samples))
            break
        spans.append((start, end))
    if not spans:
        return [], 0
    # Spans are emitted in non-decreasing start order and are contiguous or
    # overlapping, so the union is exactly `[0, last_end)`.
    return spans, spans[-1][1]


def chunk_samples(
    samples: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    max_chunks: int,
) -> tuple[list[np.ndarray], int]:
    """Slice `samples` per `plan_chunk_spans`. Returns `(chunks, covered)`."""
    if len(samples) == 0 or sample_rate <= 0:
        return [], 0
    size = max(1, int(round(chunk_seconds * sample_rate)))
    min_size = max(1, int(round(MIN_CHUNK_SECONDS * sample_rate)))
    spans, covered = plan_chunk_spans(len(samples), size, min_size, max_chunks)
    return [samples[start:end] for start, end in spans], covered


class _AudioTaggerProtocol(Protocol):
    """Minimal surface the classifier needs from an AudioSet tagger."""

    @property
    def id2label(self) -> dict[int, str]: ...

    @property
    def resolved_revision(self) -> str | None: ...

    @property
    def dtype(self) -> str | None: ...

    def scores(self, chunks: list[np.ndarray], sample_rate: int) -> np.ndarray: ...


class _AstAudioTagger:
    """Adapter over the HF AST checkpoint: int16 chunks in, sigmoids out.

    AST is a multi-label head trained with BCE, so the per-label outputs are
    independent sigmoids — never softmaxed across the 527 classes.
    """

    def __init__(self, model: Any, feature_extractor: Any, device: str) -> None:
        self._model = model
        self._feature_extractor = feature_extractor
        self._device = device

    @property
    def id2label(self) -> dict[int, str]:
        return {int(key): str(value) for key, value in self._model.config.id2label.items()}

    @property
    def resolved_revision(self) -> str | None:
        """The commit sha transformers loaded, when it records one.

        `_commit_hash` is set by the HF cache layer on a downloaded config; a
        local path or an offline load may not have it, hence the `None`.
        """
        commit = getattr(self._model.config, "_commit_hash", None)
        return str(commit) if commit else None

    @property
    def dtype(self) -> str | None:
        """Parameter dtype actually loaded — provenance for the cost probe."""
        try:
            return str(next(self._model.parameters()).dtype)
        except (StopIteration, AttributeError):  # pragma: no cover — stub models
            return None

    def scores(self, chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import torch  # noqa: PLC0415 — optional extra, lazy per module discipline

        waveforms = [np.asarray(chunk, dtype=np.float32) / 32768.0 for chunk in chunks]
        inputs = self._feature_extractor(waveforms, sampling_rate=sample_rate, return_tensors="pt")
        tensors = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**tensors).logits
        return logits.sigmoid().float().cpu().numpy()


def _load_audio_class_tagger(model_name: str, device: str, revision: str) -> _AudioTaggerProtocol:
    """Import transformers lazily and load the AudioSet tagger.

    Module-level so tests can monkeypatch it without touching Hugging Face.
    Weights load in float32: at 86.6 M params that is ~350 MB — negligible
    beside the ASR backbone — and it keeps one fewer output-affecting knob.

    The sha transformers resolves is read back off the config and verified
    against `revision` by `warmup()`.
    """
    from transformers import (  # noqa: PLC0415
        ASTFeatureExtractor,
        ASTForAudioClassification,
    )

    logger.info(
        "Loading ASR audio-class detector: model=%s revision=%s device=%s",
        model_name,
        revision,
        device,
    )
    kwargs: dict[str, Any] = {"revision": revision}
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, **kwargs)
    model = ASTForAudioClassification.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return _AstAudioTagger(model, feature_extractor, device)


class AudioClassifier:
    """Classify one clip's audio as speech / music / mixed / uncertain.

    Sync, one instance per process, same shape as `Transcriber`. Construction
    is cheap (config only); `warmup()` loads the weights.
    """

    def __init__(self) -> None:
        self.model_name = DEFAULT_AUDIO_CLASS_MODEL
        self.revision = DEFAULT_AUDIO_CLASS_REVISION
        self.device = "cuda:0"
        self.speech_floor = DEFAULT_SPEECH_FLOOR
        self.music_floor = DEFAULT_MUSIC_FLOOR
        self.singing_floor = DEFAULT_SINGING_FLOOR
        self.chunk_seconds = DEFAULT_AUDIO_CLASS_CHUNK_SECONDS
        self.max_chunks = DEFAULT_AUDIO_CLASS_MAX_CHUNKS
        self.method_version = AUDIO_CLASS_METHOD_VERSION
        self.coverage_policy = AUDIO_CLASS_COVERAGE_POLICY
        self.feature_backend = detect_feature_backend()
        if self.feature_backend != FEATURE_BACKEND_NUMPY:
            raise RuntimeError("The frozen ASR policy requires the numpy feature backend")
        self.resolved_revision: str | None = None
        # Recorded at warmup for the GPU cost probe. Not a knob and not folded
        # into the version: weights always load float32 (see the loader).
        self.resolved_dtype: str | None = None
        self._tagger: _AudioTaggerProtocol | None = None
        self._group_indices: dict[str, list[int]] = {}

    @property
    def is_warm(self) -> bool:
        return self._tagger is not None

    def warmup(self) -> None:
        """Load the detector, verify its revision, resolve its label groups.

        Two startup failures, both loud and both before any request is served:
        a checkpoint whose labels are not the AudioSet ontology cannot be
        scored by this policy at all, and weights that are not the pinned
        commit are not the policy `asr_model_version` claims.
        """
        if self.is_warm:
            return
        tagger = _load_audio_class_tagger(self.model_name, self.device, self.revision)
        self._verify_revision(tagger.resolved_revision)
        id2label = tagger.id2label
        present = set(id2label.values())
        missing_anchors = sorted(REQUIRED_LABELS - present)
        if missing_anchors:
            raise RuntimeError(
                f"ASR audio-class model {self.model_name!r} does not expose the "
                f"AudioSet labels this gate scores (missing: {missing_anchors}). "
                "Use the approved AudioSet-labelled checkpoint."
            )
        self._group_indices = {
            "speech": _resolve_indices(id2label, SPEECH_LABELS),
            "music": _resolve_indices(id2label, MUSIC_LABELS),
            "singing": _resolve_indices(id2label, SINGING_LABELS),
        }
        self._tagger = tagger
        self.resolved_revision = tagger.resolved_revision
        self.resolved_dtype = tagger.dtype
        logger.info(
            "ASR audio-class detector ready: model=%s revision=%s resolved=%s "
            "device=%s method=%s features=%s coverage=%s chunks=%.1fsx%d "
            "floors speech=%.2f music=%.2f singing=%.2f",
            self.model_name,
            self.revision,
            self.resolved_revision or "<unknown>",
            self.device,
            self.method_version,
            self.feature_backend,
            self.coverage_policy,
            self.chunk_seconds,
            self.max_chunks,
            self.speech_floor,
            self.music_floor,
            self.singing_floor,
        )

    def _verify_revision(self, resolved: str | None) -> None:
        """Fail closed unless the loaded weights match the fixed revision.

        Both failure directions raise, because both mean the rows this process
        would write are labelled with a policy it is not running:

        - a resolved sha that differs from the pin — different weights;
        - no resolved sha at all — the load cannot be verified. `warmup()` is startup, not
          the request path, so raising here costs a boot rather than a batch.
        """
        if not resolved:
            raise RuntimeError(
                f"ASR audio-class model {self.model_name!r} is pinned to "
                f"revision {self.revision!r} but the load exposed no commit "
                "sha, so the pin cannot be verified. Load from the Hugging "
                "Face cache that contains the approved revision."
            )
        if self.revision.casefold() != resolved.casefold():
            raise RuntimeError(
                f"ASR audio-class model {self.model_name!r} resolved to "
                f"revision {resolved!r}, which is not the fixed pin "
                f"{self.revision!r}. Refusing to classify: the trust policy "
                "recorded in asr_model_version would not be the one running."
            )

    def classify(self, samples: np.ndarray, sample_rate: int) -> AudioClassResult:
        """Classify int16 mono samples. Never raises on data problems.

        `samples` MUST be the exact array handed to the ASR backbone: the
        verdict is only meaningful for the audio the transcript came from.
        Empty audio, incomplete coverage, and any detector failure all collapse
        to a fail-closed verdict the trust gate refuses to fuse.
        """
        if not self.is_warm:
            raise RuntimeError("AudioClassifier not warmed up; call .warmup() before .classify()")
        assert self._tagger is not None  # for type-checkers; is_warm guards this

        offered_samples = len(samples)
        offered_seconds = offered_samples / sample_rate if sample_rate > 0 else 0.0
        chunks, covered_samples = chunk_samples(
            samples, sample_rate, self.chunk_seconds, self.max_chunks
        )
        coverage_ratio = covered_samples / offered_samples if offered_samples > 0 else 0.0
        scored_seconds = covered_samples / sample_rate if sample_rate > 0 else 0.0
        if not chunks:
            return self._withheld(
                AUDIO_CLASS_UNCERTAIN,
                "empty_audio",
                coverage_ratio=0.0,
                scored_seconds=0.0,
                offered_seconds=offered_seconds,
            )
        if covered_samples < offered_samples:
            # A verdict over part of the audio may NOT approve a transcript
            # produced from all of it — the unscored tail is precisely where
            # the music would be. Reachable only when the ASR cap is raised
            # past `max_chunks * chunk_seconds`; fail closed and say so.
            logger.warning(
                "ASR audio-class coverage %.1fs of %.1fs (max_chunks=%d x %.1fs); "
                "failing closed to %s",
                scored_seconds,
                offered_seconds,
                self.max_chunks,
                self.chunk_seconds,
                AUDIO_CLASS_INCOMPLETE,
            )
            return self._withheld(
                AUDIO_CLASS_INCOMPLETE,
                "incomplete_coverage",
                coverage_ratio=coverage_ratio,
                scored_seconds=scored_seconds,
                offered_seconds=offered_seconds,
            )

        try:
            matrix = self._tagger.scores(chunks, sample_rate)
        except Exception as err:  # noqa: BLE001 — fail closed, never propagate
            logger.warning(
                "ASR audio-class detection failed (%s: %s); failing closed to %s",
                type(err).__name__,
                err,
                AUDIO_CLASS_UNCERTAIN,
            )
            return self._withheld(
                AUDIO_CLASS_UNCERTAIN,
                "detector_error",
                coverage_ratio=coverage_ratio,
                scored_seconds=scored_seconds,
                offered_seconds=offered_seconds,
            )

        scores = AudioClassScores(
            speech=_group_max(matrix, self._group_indices["speech"]),
            music=_group_max(matrix, self._group_indices["music"]),
            singing=_group_max(matrix, self._group_indices["singing"]),
        )
        audio_class, confidence = derive_audio_class(
            scores.speech,
            scores.music,
            scores.singing,
            speech_floor=self.speech_floor,
            music_floor=self.music_floor,
            singing_floor=self.singing_floor,
        )
        return AudioClassResult(
            audio_class=audio_class,
            confidence=confidence,
            model_name=self.model_name,
            method_version=self.method_version,
            coverage_ratio=coverage_ratio,
            scored_seconds=scored_seconds,
            offered_seconds=offered_seconds,
            feature_backend=self.feature_backend,
            coverage_policy=self.coverage_policy,
            model_revision=self.revision,
            resolved_revision=self.resolved_revision,
            detail=None,
            scores=scores,
        )

    def _withheld(
        self,
        audio_class: str,
        detail: str,
        coverage_ratio: float,
        scored_seconds: float,
        offered_seconds: float,
    ) -> AudioClassResult:
        """A fail-closed verdict with no scores, carrying full provenance."""
        return AudioClassResult(
            audio_class=audio_class,
            confidence=0.0,
            model_name=self.model_name,
            method_version=self.method_version,
            coverage_ratio=coverage_ratio,
            scored_seconds=scored_seconds,
            offered_seconds=offered_seconds,
            feature_backend=self.feature_backend,
            coverage_policy=self.coverage_policy,
            model_revision=self.revision,
            resolved_revision=self.resolved_revision,
            detail=detail,
            scores=None,
        )


def _resolve_indices(id2label: dict[int, str], labels: frozenset[str]) -> list[int]:
    """Indices of `labels` in the checkpoint's label map, logging any misses."""
    index_of = {label: index for index, label in id2label.items()}
    resolved = sorted(index_of[label] for label in labels if label in index_of)
    missing = sorted(label for label in labels if label not in index_of)
    if missing:
        logger.info("ASR audio-class labels absent from checkpoint: %s", missing)
    return resolved


def _group_max(matrix: np.ndarray, indices: list[int]) -> float:
    """Max score for a label group across every chunk (presence semantics).

    Max, not mean: one sung chunk in an otherwise-spoken clip is exactly the
    `mixed` case that must fail closed, and averaging would bury it.
    """
    if not indices or matrix.size == 0:
        return 0.0
    return float(np.max(matrix[:, indices]))
