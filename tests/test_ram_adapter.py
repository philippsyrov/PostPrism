import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from postprism.visual_tags import adapter


def test_array_hash_uses_compact_utf8_json_and_python_float_values():
    assert (
        adapter.canonical_array_sha256(["red", "truck"])
        == hashlib.sha256(b'["red","truck"]').hexdigest()
    )
    assert (
        adapter.canonical_array_sha256(["красный", 0.10000000149011612])
        == hashlib.sha256('["красный",0.10000000149011612]'.encode()).hexdigest()
    )


def test_checkpoint_mismatch_fails_before_deserialization(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checkpoint"):
        adapter.verify_file(checkpoint, expected_sha256="0" * 64, expected_bytes=5)


def test_missing_checkpoint_fails_as_integrity_error(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        adapter.verify_file(tmp_path / "missing", expected_sha256="0" * 64)


def test_correct_hash_with_wrong_byte_count_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"valid")
    digest = hashlib.sha256(b"valid").hexdigest()
    with pytest.raises(ValueError, match="checkpoint"):
        adapter.verify_file(checkpoint, expected_sha256=digest, expected_bytes=6)
    adapter.verify_file(checkpoint, expected_sha256=digest, expected_bytes=5)


class Logits:
    def __init__(self, values=(1.0, 2.0), shape=None):
        self.values = values
        self.shape = shape or (1, len(values))
        self.operations = []

    def squeeze(self, axis):
        self.operations.append(("squeeze", axis))
        return self

    def detach(self):
        self.operations.append("detach")
        return self

    def float(self):
        self.operations.append("float")
        return self

    def cpu(self):
        self.operations.append("cpu")
        return self

    def __getitem__(self, index):
        assert index == 0
        return self.values


class NativeModel:
    def __init__(self, *, output=None, error=None, calls=1):
        self.tag_list = ("red", "truck")
        self.output = output or Logits()
        self.error = error
        self.hook_calls = calls
        self.generate_calls = 0
        self.removed = False
        self.fc = SimpleNamespace(register_forward_hook=self.register_hook)

    def register_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: setattr(self, "removed", True))

    def generate_tag(self, tensor):
        self.generate_calls += 1
        for _ in range(self.hook_calls):
            self.hook(self.fc, (tensor,), self.output)
        if self.error:
            raise self.error
        return [" red | truck "], ["unused"]


def test_capture_uses_one_native_forward_and_fp32_cpu_logits_in_native_order():
    model = NativeModel()
    prediction = adapter.capture_native_logits(model, object())
    assert prediction is not None
    assert prediction.native_tags == ("red", "truck")
    assert prediction.vocabulary == ("red", "truck")
    assert prediction.logits == (1.0, 2.0)
    assert model.generate_calls == 1
    assert model.output.operations == [("squeeze", -1), "detach", "float", "cpu"]


@pytest.mark.parametrize("error", [None, RuntimeError("private model error")])
def test_capture_always_removes_hook(error):
    model = NativeModel(error=error)
    if error:
        with pytest.raises(RuntimeError):
            adapter.capture_native_logits(model, object())
    else:
        adapter.capture_native_logits(model, object())
    assert model.removed


def test_predict_transforms_and_transfers_fp32_under_inference_mode():
    from PIL import Image

    events = []

    @contextmanager
    def inference_mode():
        events.append("enter")
        yield
        events.append("exit")

    class InputTensor:
        def unsqueeze(self, axis):
            events.append(("unsqueeze", axis))
            return self

        def to(self, **kwargs):
            events.append(("transfer", kwargs))
            return self

    def transform(image):
        events.append(("transform", image.mode, image.size))
        return InputTensor()

    model = NativeModel()
    ram = adapter.RamAdapter(
        model,
        transform,
        SimpleNamespace(inference_mode=inference_mode, float32="float32"),
        "cuda:0",
    )
    result = ram.predict(Image.new("RGB", (12, 7)))
    assert result.native_tags == ("red", "truck")
    assert events == [
        "enter",
        ("transform", "RGB", (12, 7)),
        ("unsqueeze", 0),
        ("transfer", {"device": "cuda:0", "dtype": "float32"}),
        "exit",
    ]
    assert model.generate_calls == 1


@pytest.mark.parametrize("shape,calls", [((2, 2), 1), ((1, 3), 1), ((1, 2), 0), ((1, 2), 2)])
def test_capture_rejects_shape_or_forward_count_mismatch(shape, calls):
    model = NativeModel(output=Logits(shape=shape), calls=calls)
    with pytest.raises(ValueError, match="shape"):
        adapter.capture_native_logits(model, object())
    assert model.removed


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_capture_rejects_nonfinite_logits(bad_value):
    model = NativeModel(output=Logits((1.0, bad_value)))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.capture_native_logits(model, object())
    assert model.removed


def test_real_loader_checks_checkpoint_before_importing_any_runtime(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__
    imported = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ram", "transformers"}:
            imported.append(name)
            raise AssertionError("runtime imported before integrity check")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checkpoint"):
        adapter.load_ram_adapter(checkpoint, tmp_path)
    assert imported == []


def test_real_loader_fails_closed_without_validated_package_and_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "verify_file", lambda *a, **k: None)
    with pytest.raises(ValueError, match="manifest"):
        adapter.load_ram_adapter(tmp_path / "checkpoint.pth", tmp_path)


def fake_runtime(monkeypatch):
    events = []
    model = NativeModel()
    model.tag_list = tuple(str(i) for i in range(4585))
    model.class_threshold = [0.10000000149011612] * 4585

    def to(**kwargs):
        events.append(("model_to", kwargs))
        return model

    def evaluate():
        events.append("eval")
        return model

    model.to = to
    model.eval = evaluate
    transform = object()

    def constructor(**kwargs):
        events.append(("construct", kwargs))
        return model

    def get_transform(**kwargs):
        events.append(("transform", kwargs))
        return transform

    runtime = (SimpleNamespace(float32="float32"), constructor, get_transform)
    monkeypatch.setattr(adapter, "verify_file", lambda *a, **k: events.append(("file", a, k)))
    monkeypatch.setattr(
        adapter, "_verify_runtime_artifacts", lambda p: events.append(("artifacts", p))
    )
    monkeypatch.setattr(adapter, "_import_runtime", lambda: events.append("import") or runtime)
    return model, events


def test_loader_constructs_only_after_integrity_and_uses_local_tokenizer_fp32_eval(
    tmp_path, monkeypatch
):
    model, events = fake_runtime(monkeypatch)
    hashes = iter(
        [
            "db241b78f0691cc0291a2c876a5eba562653526d34a732b3f48a525f0ce2384f",
            "43a429712680f4daabb3937e86ab11c268fc479a74f563d58a521b3e1f870844",
        ]
    )
    monkeypatch.setattr(adapter, "canonical_array_sha256", lambda values: next(hashes))
    ram = adapter.load_ram_adapter(tmp_path / "checkpoint.pth", tmp_path / "tokenizer")
    assert isinstance(ram, adapter.RamAdapter)
    assert events == [
        (
            "file",
            (tmp_path / "checkpoint.pth",),
            {
                "expected_sha256": "497c178836ba66698ca226c7895317e6e800034be986452dbd2593298d50e87d",
                "expected_bytes": 3010210801,
            },
        ),
        ("artifacts", tmp_path / "tokenizer"),
        "import",
        (
            "construct",
            {
                "pretrained": str(tmp_path / "checkpoint.pth"),
                "image_size": 384,
                "vit": "swin_l",
                "text_encoder_type": str(tmp_path / "tokenizer"),
            },
        ),
        ("model_to", {"device": "cuda:0", "dtype": "float32"}),
        "eval",
        ("transform", {"image_size": 384}),
    ]


@pytest.mark.parametrize("vocab_count,threshold_count", [(2, 4585), (4585, 2)])
def test_loader_rejects_nonproduction_vocabulary_or_threshold_shape(
    tmp_path, monkeypatch, vocab_count, threshold_count
):
    model, _ = fake_runtime(monkeypatch)
    model.tag_list = tuple(str(i) for i in range(vocab_count))
    model.class_threshold = [0.1] * threshold_count
    with pytest.raises(ValueError, match="4585"):
        adapter.load_ram_adapter(tmp_path / "checkpoint.pth", tmp_path)


@pytest.mark.parametrize("bad_hash", ["vocabulary_sha256", "thresholds_sha256"])
def test_loader_rejects_changed_loaded_vocabulary_or_thresholds(tmp_path, monkeypatch, bad_hash):
    from postprism.visual_tags.identity import RAM_IDENTITY

    model, _ = fake_runtime(monkeypatch)
    model.class_threshold = [0.1] * 4585
    received = []

    def digest(values):
        received.append(list(values))
        key = "vocabulary_sha256" if len(received) == 1 else "thresholds_sha256"
        return "0" * 64 if key == bad_hash else RAM_IDENTITY.artifacts[key]

    monkeypatch.setattr(adapter, "canonical_array_sha256", digest)
    with pytest.raises(ValueError, match=bad_hash):
        adapter.load_ram_adapter(tmp_path / "checkpoint.pth", tmp_path)
    assert received[0] == list(model.tag_list)
    if bad_hash == "thresholds_sha256":
        assert received[1] == [0.10000000149011612] * 4585
