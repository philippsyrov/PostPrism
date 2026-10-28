"""Lazy, pinned Qwen3-VL caption generation backend."""

from PIL import Image

from postprism.captioning.identity import CAPTION_IDENTITY


def _load_backend():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    return torch, AutoProcessor, Qwen3VLForConditionalGeneration


class CaptionAdapter:
    identity = CAPTION_IDENTITY

    def __init__(self, model, processor, torch):
        self._model = model
        self._processor = processor
        self._torch = torch

    def generate(self, frames: list[Image.Image], prompt: str) -> str:
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": prompt})
        inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False, num_beams=1
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated, strict=False)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0] if decoded else ""


def load_caption_adapter() -> CaptionAdapter:
    torch, processor_class, model_class = _load_backend()
    model = model_class.from_pretrained(
        CAPTION_IDENTITY.model,
        revision=CAPTION_IDENTITY.model_revision,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    processor = processor_class.from_pretrained(
        CAPTION_IDENTITY.model, revision=CAPTION_IDENTITY.model_revision
    )
    return CaptionAdapter(model, processor, torch)
