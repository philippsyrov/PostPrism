import pytest

from postprism.multimodal.identity import (
    DEFAULT_DTYPE,
    DEFAULT_METHOD_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_OUTPUT_DIM,
)
from postprism.startup_checks import validate_runtime_configuration


def approved_environment() -> dict[str, str]:
    return {
        "POSTPRISM_RUNTIME_PROFILE": "production",
        "MULTIMODAL_EMBEDDER_MODEL": DEFAULT_MODEL_NAME,
        "MULTIMODAL_EMBEDDER_MODEL_REVISION": DEFAULT_MODEL_REVISION,
        "MULTIMODAL_EMBEDDER_DTYPE": DEFAULT_DTYPE,
        "MULTIMODAL_EMBEDDER_METHOD_VERSION": DEFAULT_METHOD_VERSION,
        "MULTIMODAL_EMBEDDER_OUTPUT_DIM": str(DEFAULT_OUTPUT_DIM),
        "MULTIMODAL_EMBEDDER_DEVICE": "cuda:0",
    }


def approved_versions() -> dict[str, str]:
    return {
        "python": "3.11.16",
        "torch": "2.13.0+cu129",
        "torchvision": "0.28.0+cu129",
        "transformers": "4.57.6",
        "huggingface-hub": "0.36.2",
        "qwen-vl-utils": "0.0.14",
        "qwen-asr": "0.0.6",
        "accelerate": "1.12.0",
        "av": "18.0.0",
        "pillow": "12.3.0",
        "numpy": "2.4.6",
        "fastapi": "0.141.1",
        "uvicorn": "0.52.0",
        "python-multipart": "0.0.32",
        "timm": "0.4.12",
        "fairscale": "0.4.4",
    }


def approved_asr_environment() -> dict[str, str]:
    return approved_environment() | {"USE_ASR": "true"}


def test_production_accepts_the_approved_identity() -> None:
    profile = validate_runtime_configuration(
        approved_environment(),
        approved_versions(),
    )

    assert profile.name == "production"
    assert profile.production is True
    assert profile.device == "cuda:0"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("MULTIMODAL_EMBEDDER_MODEL", "another/model"),
        ("MULTIMODAL_EMBEDDER_MODEL_REVISION", "0" * 40),
        ("MULTIMODAL_EMBEDDER_DTYPE", "float32"),
        ("MULTIMODAL_EMBEDDER_METHOD_VERSION", "old-method"),
        ("MULTIMODAL_EMBEDDER_OUTPUT_DIM", "768"),
    ],
)
def test_production_refuses_identity_drift(field: str, bad_value: str) -> None:
    environment = approved_environment() | {field: bad_value}

    with pytest.raises(RuntimeError, match=field):
        validate_runtime_configuration(environment, approved_versions())


@pytest.mark.parametrize(
    ("package", "bad_version"),
    [
        ("torch", "2.12.0"),
        ("torchvision", "0.27.0"),
        ("transformers", "4.58.0"),
        ("huggingface-hub", "0.36.3"),
        ("qwen-vl-utils", "0.0.13"),
        ("accelerate", "1.11.0"),
        ("av", "18.1.0"),
        ("pillow", "12.4.0"),
        ("numpy", "2.5.2"),
        ("fastapi", "0.142.0"),
        ("uvicorn", "0.52.4"),
        ("python-multipart", "0.0.33"),
    ],
)
def test_production_refuses_unmeasured_package_lines(
    package: str,
    bad_version: str,
) -> None:
    versions = approved_versions() | {package: bad_version}

    with pytest.raises(RuntimeError, match=package):
        validate_runtime_configuration(approved_environment(), versions)


def test_production_refuses_unmeasured_python() -> None:
    with pytest.raises(RuntimeError, match="Python 3.11"):
        validate_runtime_configuration(
            approved_environment(),
            approved_versions() | {"python": "3.12.11"},
        )


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_production_refuses_the_wrong_cuda_build(package: str) -> None:
    versions = approved_versions() | {
        package: approved_versions()[package].replace("cu129", "cu128")
    }

    with pytest.raises(RuntimeError, match=package):
        validate_runtime_configuration(approved_environment(), versions)


def test_production_refuses_the_unmeasured_mps_profile() -> None:
    with pytest.raises(RuntimeError, match="POSTPRISM_RUNTIME_PROFILE=lab-mps"):
        validate_runtime_configuration(
            approved_environment() | {"MULTIMODAL_EMBEDDER_DEVICE": "mps:0"},
            approved_versions(),
        )


def test_lab_mps_profile_is_explicitly_non_production() -> None:
    profile = validate_runtime_configuration(
        approved_environment()
        | {
            "POSTPRISM_RUNTIME_PROFILE": "lab-mps",
            "MULTIMODAL_EMBEDDER_DEVICE": "mps:0",
        },
        approved_versions(),
    )

    assert profile.name == "lab-mps"
    assert profile.production is False


def test_lab_mps_profile_requires_the_mps_device() -> None:
    with pytest.raises(RuntimeError, match="mps:0"):
        validate_runtime_configuration(
            approved_environment() | {"POSTPRISM_RUNTIME_PROFILE": "lab-mps"},
            approved_versions(),
        )


def test_lab_mps_profile_refuses_unmeasured_asr() -> None:
    with pytest.raises(RuntimeError, match="ASR is not approved"):
        validate_runtime_configuration(
            approved_environment()
            | {
                "POSTPRISM_RUNTIME_PROFILE": "lab-mps",
                "MULTIMODAL_EMBEDDER_DEVICE": "mps:0",
                "USE_ASR": "true",
            },
            approved_versions() | {"qwen-asr": "0.0.6"},
        )


def test_production_refuses_custom_asr_package_build() -> None:
    with pytest.raises(RuntimeError, match="qwen-asr"):
        validate_runtime_configuration(
            approved_asr_environment(),
            approved_versions() | {"qwen-asr": "0.0.6+custom"},
        )


def test_enabled_asr_refuses_retired_configuration() -> None:
    with pytest.raises(RuntimeError, match="remove retired overrides: TRANSCRIBER_LANGUAGE"):
        validate_runtime_configuration(
            approved_asr_environment() | {"TRANSCRIBER_LANGUAGE": "ru"},
            approved_versions(),
        )


def test_enabled_asr_uses_the_pinned_model_and_gate_defaults() -> None:
    environment = approved_environment() | {"USE_ASR": "true"}

    profile = validate_runtime_configuration(
        environment,
        approved_versions() | {"qwen-asr": "0.0.6"},
    )

    assert profile.asr_enabled is True


@pytest.mark.parametrize("key", ["USE_TOPICS", "TOPIC_DEFINITIONS_FILE"])
@pytest.mark.parametrize("value", ["", "false", "0", "true", " /old/catalog.json "])
@pytest.mark.parametrize("profile_name,device", [("production", "cuda:0"), ("lab-mps", "mps:0")])
def test_retired_topic_environment_is_rejected_by_presence(key, value, profile_name, device):
    environment = approved_environment() | {
        "POSTPRISM_RUNTIME_PROFILE": profile_name,
        "MULTIMODAL_EMBEDDER_DEVICE": device,
        key: value,
    }
    with pytest.raises(
        RuntimeError, match="semantic topics were removed; remove retired variables"
    ) as error:
        validate_runtime_configuration(environment, approved_versions())
    assert key in str(error.value)


def test_both_retired_topic_environment_keys_are_reported():
    with pytest.raises(RuntimeError) as error:
        validate_runtime_configuration(
            approved_environment() | {"USE_TOPICS": "false", "TOPIC_DEFINITIONS_FILE": ""},
            approved_versions(),
        )
    assert "USE_TOPICS" in str(error.value)
    assert "TOPIC_DEFINITIONS_FILE" in str(error.value)


def test_runtime_profile_has_no_topic_enablement():
    profile = validate_runtime_configuration(approved_environment(), approved_versions())
    assert not hasattr(profile, "topics_enabled")


@pytest.mark.parametrize(
    "flag,field",
    [("USE_VISUAL_TAGS", "visual_tags_enabled"), ("USE_CAPTIONS", "captions_enabled")],
)
def test_visual_workers_default_off_and_opt_in_on_production(flag, field) -> None:
    disabled = validate_runtime_configuration(approved_environment(), approved_versions())
    assert getattr(disabled, field, None) is False
    enabled = validate_runtime_configuration(
        approved_environment() | {flag: "true"}, approved_versions()
    )
    assert getattr(enabled, field, None) is True


@pytest.mark.parametrize("flag", ["USE_VISUAL_TAGS", "USE_CAPTIONS"])
def test_lab_mps_refuses_unmeasured_visual_workers(flag) -> None:
    with pytest.raises(RuntimeError, match=flag + ".*lab-mps"):
        validate_runtime_configuration(
            approved_environment()
            | {
                "POSTPRISM_RUNTIME_PROFILE": "lab-mps",
                "MULTIMODAL_EMBEDDER_DEVICE": "mps:0",
                flag: "true",
            },
            approved_versions(),
        )


@pytest.mark.parametrize("package", ["timm", "fairscale"])
@pytest.mark.parametrize("actual", ["missing", "0.0.0", "0.4.12+custom"])
def test_enabled_ram_requires_exact_extra_dependency_versions(package, actual):
    versions = approved_versions() | {package: actual}
    validate_runtime_configuration(approved_environment(), versions)
    with pytest.raises(RuntimeError, match=package):
        validate_runtime_configuration(
            approved_environment() | {"USE_VISUAL_TAGS": "true"}, versions
        )
