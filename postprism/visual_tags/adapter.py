"""Pinned RAM++ inference and artifact integrity, with lazy model imports."""

import hashlib
import importlib.util
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from postprism.visual_tags.artifacts import validate_package, validate_tokenizer
from postprism.visual_tags.identity import RAM_IDENTITY


class RamAdapter:
    identity = RAM_IDENTITY

    def __init__(self, model, transform, torch_module, device: str):
        self._model = model
        self._transform = transform
        self._torch = torch_module
        self._device = device

    def predict(self, image: Image.Image) -> "RamPrediction":
        with self._torch.inference_mode():
            tensor = (
                self._transform(image)
                .unsqueeze(0)
                .to(device=self._device, dtype=self._torch.float32)
            )
            return capture_native_logits(self._model, tensor)


@dataclass(frozen=True)
class RamPrediction:
    native_tags: tuple[str, ...]
    vocabulary: tuple[str, ...]
    logits: tuple[float, ...]


def capture_native_logits(model, tensor) -> RamPrediction:
    captured = []
    handle = model.fc.register_forward_hook(
        lambda _module, _inputs, out: captured.append(out.squeeze(-1).detach().float().cpu())
    )
    try:
        tags, _ = model.generate_tag(tensor)
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (1, len(model.tag_list)):
        raise ValueError("RAM output shape mismatch")
    native = tuple(tag.strip() for tag in tags[0].split("|") if tag.strip())
    logits = tuple(float(v) for v in captured[0][0])
    if not all(math.isfinite(value) for value in logits):
        raise ValueError("non-finite RAM logits")
    return RamPrediction(native, tuple(str(tag) for tag in model.tag_list), logits)


def canonical_array_sha256(values: Sequence) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_file(path: Path, *, expected_sha256: str, expected_bytes: int | None = None) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ValueError("RAM checkpoint artifact unreadable") from error
    if digest.hexdigest() != expected_sha256 or (
        expected_bytes is not None and size != expected_bytes
    ):
        raise ValueError("RAM checkpoint artifact integrity mismatch")


def _verify_runtime_artifacts(tokenizer_dir: Path) -> None:
    validate_tokenizer(tokenizer_dir)
    spec = importlib.util.find_spec("ram")
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise ValueError("validated RAM package is not installed")
    validate_package(Path(spec.origin).parent)


def _import_runtime():
    import torch
    from ram import get_transform
    from ram.models import ram_plus

    return torch, ram_plus, get_transform


def load_ram_adapter(checkpoint: Path, tokenizer_dir: Path, device: str = "cuda:0") -> RamAdapter:
    verify_file(
        checkpoint,
        expected_sha256=RAM_IDENTITY.artifacts["checkpoint_sha256"],
        expected_bytes=3_010_210_801,
    )
    _verify_runtime_artifacts(tokenizer_dir)
    torch_module, constructor, get_transform = _import_runtime()
    model = (
        constructor(
            pretrained=str(checkpoint),
            image_size=384,
            vit="swin_l",
            text_encoder_type=str(tokenizer_dir),
        )
        .to(device=device, dtype=torch_module.float32)
        .eval()
    )
    if len(model.tag_list) != 4585 or len(model.class_threshold) != 4585:
        raise ValueError("RAM requires 4585 vocabulary and threshold entries")
    for key, values in (
        ("vocabulary_sha256", [str(tag) for tag in model.tag_list]),
        (
            "thresholds_sha256",
            [
                struct.unpack("f", struct.pack("f", float(value)))[0]
                for value in model.class_threshold
            ],
        ),
    ):
        if canonical_array_sha256(values) != RAM_IDENTITY.artifacts[key]:
            raise ValueError(f"RAM {key} mismatch")
    return RamAdapter(model, get_transform(image_size=384), torch_module, device)
