from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from postprism.captioning import adapter as adapter_module
from postprism.captioning.identity import CAPTION_IDENTITY


@pytest.fixture
def backend(monkeypatch):
    events = {}

    class Inputs(dict):
        def to(self, device):
            events["input_device"] = device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            events["processor_load"] = (name, kwargs)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            events["template"] = (messages, kwargs)
            return Inputs(input_ids=[[1, 2, 3]], pixel_values="pixels")

        def batch_decode(self, sequences, **kwargs):
            events["decode"] = (sequences, kwargs)
            return events.get("decoded", ['{"caption":"A truck."}'])

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            events["model_load"] = (name, kwargs)
            return cls()

        def eval(self):
            events["eval"] = True
            return self

        def generate(self, **kwargs):
            events["generate"] = kwargs
            events["inference_during_generate"] = events.get("inference_active", False)
            return [[1, 2, 3, 91, 92]]

    @contextmanager
    def inference_mode():
        events["inference_active"] = True
        try:
            yield
        finally:
            events["inference_active"] = False

    torch = SimpleNamespace(bfloat16=object(), inference_mode=inference_mode)
    monkeypatch.setattr(adapter_module, "_load_backend", lambda: (torch, Processor, Model))
    return events, torch


def test_loader_pins_model_processor_bf16_cuda_and_eval(backend):
    events, torch = backend
    adapter = adapter_module.load_caption_adapter()
    assert adapter.identity == CAPTION_IDENTITY
    assert events["model_load"] == (
        "Qwen/Qwen3-VL-2B-Instruct",
        {
            "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
        },
    )
    assert events["processor_load"] == (
        "Qwen/Qwen3-VL-2B-Instruct",
        {"revision": "89644892e4d85e24eaac8bacfd4f463576704203"},
    )
    assert events["eval"] is True


@pytest.mark.parametrize(
    "decoded,expected", [(['{"caption":"A truck."}'], '{"caption":"A truck."}'), ([], "")]
)
def test_generation_uses_images_then_prompt_greedy_inference_and_suffix_decode(
    backend, decoded, expected
):
    events, _ = backend
    events["decoded"] = decoded
    frames = [Image.new("RGB", (8, 5)), Image.new("RGB", (5, 8))]
    adapter = adapter_module.load_caption_adapter()
    assert adapter.generate(frames, "prompt") == expected
    assert events["template"] == (
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frames[0]},
                    {"type": "image", "image": frames[1]},
                    {"type": "text", "text": "prompt"},
                ],
            }
        ],
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        },
    )
    assert events["input_device"] == "cuda:0"
    assert events["generate"] == {
        "input_ids": [[1, 2, 3]],
        "pixel_values": "pixels",
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
    }
    assert events["inference_during_generate"] is True
    assert events["inference_active"] is False
    assert events["decode"] == (
        [[91, 92]],
        {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
    )
