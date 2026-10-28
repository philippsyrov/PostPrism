"""Historical promptv4 captions with one bounded JSON retry, without OCR salvage."""

import json

CAPTION_PROMPT = (
    "You are an image-understanding assistant. Look at the provided image "
    "and return a STRICT JSON object with exactly one key:\n"
    '- "caption": describe the post\'s intended meaning in one short '
    "sentence. Incorporate visible text when it materially changes or "
    "explains that meaning; prioritize meaning over copying incidental visible strings.\n"
    "\n"
    "Language rule: if the image's primary text or context is Russian, write "
    '"caption" in Russian. Otherwise write it in English.\n'
    "\n"
    "Output ONLY the JSON object. No markdown fences, no commentary, no text "
    "outside the JSON."
)

RETRY_PROMPT = (
    "The previous response was invalid. Inspect the provided image again and "
    'return ONLY valid JSON with exactly one non-empty string key: "caption". '
    "Describe the post's intended meaning in one short sentence, using Russian "
    "when the image's primary text or context is Russian and English otherwise."
)


def _extract_json_blob(text: str) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    return stripped[first : last + 1]


def parse_caption(raw: str) -> str | None:
    blob = _extract_json_blob(raw)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    caption = parsed.get("caption")
    if not isinstance(caption, str):
        return None
    return caption.strip() or None


def generate_caption(adapter, frames) -> str | None:
    for prompt in (CAPTION_PROMPT, RETRY_PROMPT):
        caption = parse_caption(adapter.generate(frames, prompt))
        if caption is not None:
            return caption
    return None
