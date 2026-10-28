"""Build the trusted text portion of a unified multimodal post input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def assemble_multimodal_text(
    post_text: str | None,
    asr_results: Sequence[Mapping[str, object]] | None,
) -> str:
    """Return author text plus every transcript whose trust flag is literally ``True``."""
    # A bare result dict iterates as its keys and would silently drop every
    # transcript; refuse that shape loudly instead of failing quietly closed.
    if isinstance(asr_results, Mapping):
        raise TypeError("asr_results must be a sequence of transcription results")
    sections: list[str] = []
    normalized_post = (post_text or "").strip()
    if normalized_post:
        sections.append(f"Post text: {normalized_post}")

    for asr_result in asr_results or ():
        if not isinstance(asr_result, Mapping):
            raise TypeError("each ASR result must be a mapping")
        if asr_result.get("asr_used_for_embedding") is not True:
            continue
        transcript = str(asr_result.get("asr_transcript") or "").strip()
        if transcript:
            sections.append(f"Trusted speech transcript: {transcript}")

    return "\n\n".join(sections)
