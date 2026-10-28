"""Pure Qwen trust-ordering and failure-shape tests."""

from __future__ import annotations

import pytest

from postprism.asr.audio_class import (
    AUDIO_CLASS_INCOMPLETE,
    AUDIO_CLASS_MIXED,
    AUDIO_CLASS_MUSIC,
    AUDIO_CLASS_SPEECH,
    AUDIO_CLASS_UNCERTAIN,
)
from postprism.asr.qwen_policy import (
    _is_dominant_token_repetition,
    derive_qwen_trust_decision,
)


class TestTranscriptRepetitionPolicy:
    SPEECH_TEXT = "нормальная человеческая речь про путешествия и еду"

    def test_dominant_token_repetition_gate_branch(self) -> None:
        # Unit-pins the dominance helper plus the repetition reason label of
        # the pure policy function.
        assert _is_dominant_token_repetition("ой " * 15) is True
        assert _is_dominant_token_repetition(self.SPEECH_TEXT) is False
        assert _is_dominant_token_repetition("ой ой") is False  # too few tokens
        used, reason = derive_qwen_trust_decision("ой " * 15, speech_ratio=0.9, sent_seconds=10.0)
        assert used is False
        assert reason == "repetitive_transcript"


class TestQwenIncidentRegression:
    """A long, sparse, song-shaped transcript must be withheld."""

    INCIDENT_SPEECH_RATIO = 0.9517
    INCIDENT_PRODUCED_FROM_SECONDS = 59.910
    INCIDENT_SHAPE_TRANSCRIPT = (
        "гора река ветер поле дым звезда камень туман ночь свет "
        "берег волна лодка парус небо дождь трава мост город сад "
        "зима осень утро вечер путь голос эхо тень крыло птица "
        "остров залив маяк скала прибой"
    )

    def test_incident_transcript_is_withheld(self) -> None:
        used, reason = derive_qwen_trust_decision(
            self.INCIDENT_SHAPE_TRANSCRIPT,
            self.INCIDENT_SPEECH_RATIO,
            self.INCIDENT_PRODUCED_FROM_SECONDS,
        )
        assert used is False
        assert reason == "sparse_long_audio"

    def test_incident_shape_survives_the_new_default_cap(self) -> None:
        # Under the new 30 s cap the pod hears less audio, but fragmentary
        # music transcripts stay roughly density-invariant — the rule must
        # still withhold at the capped operating point (sent == cap ≥ the
        # 20 s long-clip trigger).
        capped_transcript = self.INCIDENT_SHAPE_TRANSCRIPT[:101]
        used, reason = derive_qwen_trust_decision(
            capped_transcript, self.INCIDENT_SPEECH_RATIO, 30.0
        )
        assert used is False
        assert reason == "sparse_long_audio"

    def test_short_sparse_clip_still_fuses(self) -> None:
        # Clip 3 of the same run: 51 chars over 12.8 s of activity
        # (3.97 chars/s) is a GOOD transcript — short clips with one sentence
        # must not be caught by the long-audio rule.
        used, reason = derive_qwen_trust_decision(
            "короткая осмысленная фраза сказанная в коротком ролике"[:51],
            0.946,
            12.8,
        )
        assert used is True
        assert reason is None

    def test_long_dense_speech_still_fuses(self) -> None:
        # A long-form talking video: 30 s of capped audio at natural
        # speech density (~10 chars/s) passes — the rule only catches
        # sparse-for-their-length transcripts.
        dense = (
            "сегодня расскажу про наши путешествия по горным тропам где мы "
            "видели водопады и озёра потом спустились в долину к реке разбили "
            "лагерь приготовили ужин на костре обсудили планы на завтра решили "
            "идти к перевалу утром собрали палатку и двинулись дальше по "
            "маршруту через лес мимо скал и ручьёв к вершине"
        )
        used, reason = derive_qwen_trust_decision(dense, 0.95, 30.0)
        assert used is True
        assert reason is None


class TestLowInformationTranscriptPolicy:
    """Reject long speech-shaped audio that yields only a tiny fragment."""

    def test_tiny_fragment_is_withheld(self) -> None:
        used, reason = derive_qwen_trust_decision(
            "Ну, ладно!",
            speech_ratio=0.9491,
            sent_seconds=14.17,
            audio_class=AUDIO_CLASS_SPEECH,
            enforce_audio_class=True,
        )

        assert used is False
        assert reason == "low_information_transcript"

    @pytest.mark.parametrize(
        ("transcript", "sent_seconds"),
        [
            ("иди сюда", 3.0),
            ("позвони мне пожалуйста", 14.17),
            ("восемнадцать символов", 14.17),
        ],
    )
    def test_short_commands_and_meaningful_fragments_remain_eligible(
        self,
        transcript: str,
        sent_seconds: float,
    ) -> None:
        used, reason = derive_qwen_trust_decision(
            transcript,
            speech_ratio=0.95,
            sent_seconds=sent_seconds,
            audio_class=AUDIO_CLASS_SPEECH,
            enforce_audio_class=True,
        )

        assert used is True
        assert reason is None


class TestAudioClassTrustPolicy:
    """The opt-in audio-class check inside the pure trust policy.

    `PLAUSIBLE_LYRICS` is the shape that motivated this gate: a short music
    clip whose hallucinated lyrics are ordinary-looking prose. It clears every
    transcript-level check — ratio 0.97, zlib ratio well under 2.4, no token
    dominance, 7.2 chars/s (above the 5.0 sparse-long floor, far below the
    40 chars/s ceiling) — and, at 15 s, never even arms the long-audio rule.
    """

    PLAUSIBLE_LYRICS = (
        "I don't wanna wait no more baby take my hand tonight we can run "
        "into the light everything will be alright"
    )
    ORDINARY_SPEECH = (
        "сегодня показываю как я готовлю простой завтрак из трёх "
        "ингредиентов и это занимает пять минут"
    )
    DENSE_SPEECH = (
        "сегодня показываю как я готовлю простой завтрак из трёх "
        "ингредиентов и это занимает пять минут потом мы поедем в горы "
        "смотреть водопады а вечером будет ужин у костра с гитарой и "
        "историями завтра планируем подъём на перевал и спуск к озеру "
        "где остановимся на ночлег в палатке"
    )
    SPEECH_RATIO = 0.97
    SENT_SECONDS = 15.0

    def test_lyrics_pass_every_transcript_level_check(self) -> None:
        # The gap, pinned: without the audio gate this transcript fuses.
        used, reason = derive_qwen_trust_decision(
            self.PLAUSIBLE_LYRICS, self.SPEECH_RATIO, self.SENT_SECONDS
        )
        assert used is True
        assert reason is None

    def test_music_verdict_withholds_the_same_lyrics(self) -> None:
        used, reason = derive_qwen_trust_decision(
            self.PLAUSIBLE_LYRICS,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=AUDIO_CLASS_MUSIC,
            enforce_audio_class=True,
        )
        assert used is False
        assert reason == "music_audio"

    def test_mixed_verdict_fails_closed(self) -> None:
        used, reason = derive_qwen_trust_decision(
            self.ORDINARY_SPEECH,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=AUDIO_CLASS_MIXED,
            enforce_audio_class=True,
        )
        assert used is False
        assert reason == "mixed_audio"

    def test_incomplete_coverage_fails_closed_with_its_own_reason(self) -> None:
        # Distinct from `uncertain`: the detector had no opinion because it was
        # not shown the whole clip, which is an operational fault to fix, not a
        # property of the audio. Keeping the reasons separate is what makes
        # that visible in stored rows.
        used, reason = derive_qwen_trust_decision(
            self.ORDINARY_SPEECH,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=AUDIO_CLASS_INCOMPLETE,
            enforce_audio_class=True,
        )
        assert used is False
        assert reason == "incomplete_audio"

    @pytest.mark.parametrize("audio_class", [AUDIO_CLASS_UNCERTAIN, None, "", "banana", "SPEECH"])
    def test_unknown_or_missing_verdict_fails_closed(self, audio_class: str | None) -> None:
        # Missing detector output must never read as a pass — including the
        # case-mismatched value a hand-written row might carry.
        used, reason = derive_qwen_trust_decision(
            self.ORDINARY_SPEECH,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=audio_class,
            enforce_audio_class=True,
        )
        assert used is False
        assert reason == "uncertain_audio"

    def test_speech_verdict_keeps_ordinary_speech_eligible(self) -> None:
        used, reason = derive_qwen_trust_decision(
            self.ORDINARY_SPEECH,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=AUDIO_CLASS_SPEECH,
            enforce_audio_class=True,
        )
        assert used is True
        assert reason is None

    def test_speech_verdict_does_not_bypass_the_existing_checks(self) -> None:
        # The audio gate is additive: a speech verdict still has to survive
        # repetition, sparse-long-audio, ratio, and density.
        for transcript, ratio, seconds, expected in (
            ("ой " * 15, 0.9, 10.0, "repetitive_transcript"),
            (self.ORDINARY_SPEECH[:60], 0.9, 30.0, "sparse_long_audio"),
            (self.ORDINARY_SPEECH, 0.10, 15.0, "low_speech_ratio"),
            # 274 varied chars over 5 s ⇒ 54.8 chars/s, past the 40 ceiling,
            # while staying well under the 2.4 zlib repetition threshold.
            (self.DENSE_SPEECH, 0.9, 5.0, "implausible_density"),
        ):
            used, reason = derive_qwen_trust_decision(
                transcript,
                ratio,
                seconds,
                audio_class=AUDIO_CLASS_SPEECH,
                enforce_audio_class=True,
            )
            assert used is False
            assert reason == expected

    def test_verdict_is_ignored_when_the_gate_is_not_enforced(self) -> None:
        # The shipped default: `audio_class` present on the row changes
        # nothing unless the version says the gate was enforced.
        used, reason = derive_qwen_trust_decision(
            self.PLAUSIBLE_LYRICS,
            self.SPEECH_RATIO,
            self.SENT_SECONDS,
            audio_class=AUDIO_CLASS_MUSIC,
        )
        assert used is True
        assert reason is None
