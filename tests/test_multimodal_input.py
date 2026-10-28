from __future__ import annotations

import pytest

from postprism.multimodal.input import assemble_multimodal_text


def test_multimodal_text_uses_post_text_and_literal_true_asr() -> None:
    assert (
        assemble_multimodal_text(
            " hello ",
            [{"asr_transcript": " spoken ", "asr_used_for_embedding": True}],
        )
        == "Post text: hello\n\nTrusted speech transcript: spoken"
    )


def test_multimodal_text_keeps_every_trusted_transcript_in_media_order() -> None:
    assert (
        assemble_multimodal_text(
            "post",
            [
                {"asr_transcript": "first", "asr_used_for_embedding": True},
                {"asr_transcript": "skipped", "asr_used_for_embedding": False},
                {"asr_transcript": "", "asr_used_for_embedding": True},
                {"asr_transcript": "third", "asr_used_for_embedding": True},
            ],
        )
        == "Post text: post\n\nTrusted speech transcript: first\n\nTrusted speech transcript: third"
    )


@pytest.mark.parametrize("untrusted_flag", ["true", 1, 1.0, None, False])
def test_multimodal_text_rejects_non_literal_true_asr(untrusted_flag: object) -> None:
    assert (
        assemble_multimodal_text(
            None,
            [
                {
                    "asr_transcript": "do not use",
                    "asr_used_for_embedding": untrusted_flag,
                    "caption": "do not use caption",
                    "ocr_text": "do not use OCR",
                }
            ],
        )
        == ""
    )


def test_one_trusted_transcript_never_admits_an_untrusted_sibling() -> None:
    assert (
        assemble_multimodal_text(
            "actual post text",
            [
                {"asr_transcript": "music lyrics", "asr_used_for_embedding": False},
                {"asr_transcript": "trusted", "asr_used_for_embedding": True},
            ],
        )
        == "Post text: actual post text\n\nTrusted speech transcript: trusted"
    )


@pytest.mark.parametrize("asr_results", [None, [], ()])
def test_multimodal_text_without_asr_is_post_text_only(asr_results: object) -> None:
    assert assemble_multimodal_text("text", asr_results) == "Post text: text"


def test_a_bare_result_dict_is_refused_instead_of_silently_dropping_speech() -> None:
    trusted = {"asr_transcript": "speech", "asr_used_for_embedding": True}

    with pytest.raises(TypeError, match="sequence"):
        assemble_multimodal_text("text", trusted)
    with pytest.raises(TypeError, match="mapping"):
        assemble_multimodal_text("text", ["not a mapping"])
