import hashlib

import pytest

from postprism.captioning.policy import (
    CAPTION_PROMPT,
    RETRY_PROMPT,
    generate_caption,
    parse_caption,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"caption":"  A truck.  "}', "A truck."),
        ('```json\n{"caption":"Привет"}\n```', "Привет"),
        ('{"caption":"A truck.","ocr_text":"private"}', "A truck."),
        ('{"caption":""}', None),
        ('{"caption":42}', None),
        ('{"caption":"incomplete', None),
        ("[]", None),
        ('{"caption":"  "}', None),
        ('{"other":"value"}', None),
        ('Commentary\n{"caption":"A truck."}\nDone', "A truck."),
    ],
)
def test_caption_parser(raw, expected):
    assert parse_caption(raw) == expected


class Adapter:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def generate(self, frames, prompt):
        self.prompts.append(prompt)
        return next(self.outputs)


def test_retry_once_only_for_invalid_output():
    adapter = Adapter(["bad", '{"caption":"recovered"}'])
    assert generate_caption(adapter, []) == "recovered"
    assert adapter.prompts == [CAPTION_PROMPT, RETRY_PROMPT]


def test_valid_caption_uses_approved_prompt_without_retry():
    adapter = Adapter(['{"caption":"valid"}'])
    assert generate_caption(adapter, []) == "valid"
    assert adapter.prompts == [CAPTION_PROMPT]
    # Frozen promptv4 strings from be81af2; no Git dependency at test/runtime.
    assert hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest() == (
        "57b2aaaf53b1d6a680fba1059fcbc7f09308b5839b305bd3d95105fa42114a65"
    )
    assert hashlib.sha256(RETRY_PROMPT.encode()).hexdigest() == (
        "8ae0a5ef136e7816897711615f007b7a3a5e142d4b0464fe4623222b2077c95e"
    )


def test_two_invalid_outputs_stop_after_two_calls_without_ocr_salvage():
    adapter = Adapter(['{"caption":"valid","ocr_text":"unclosed', "bad"])
    assert generate_caption(adapter, []) is None
    assert adapter.prompts == [CAPTION_PROMPT, RETRY_PROMPT]


def test_generation_exception_propagates_without_retry():
    class BrokenAdapter(Adapter):
        def generate(self, frames, prompt):
            self.prompts.append(prompt)
            raise RuntimeError("private output")

    adapter = BrokenAdapter([])
    with pytest.raises(RuntimeError, match="private output"):
        generate_caption(adapter, [])
    assert adapter.prompts == [CAPTION_PROMPT]
