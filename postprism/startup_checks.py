"""Model-free validation of the measured service runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from platform import python_version

from postprism.branding import RUNTIME_PROFILE_ENV
from postprism.multimodal.identity import (
    DEFAULT_DTYPE,
    DEFAULT_METHOD_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_OUTPUT_DIM,
)

REQUIRED_PACKAGE_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "qwen-vl-utils": "0.0.14",
    "accelerate": "1.12.0",
    "av": "18.0.0",
    "pillow": "13.0.0",
    "numpy": "2.4.6",
    "fastapi": "0.141.1",
    "uvicorn": "0.52.0",
    "python-multipart": "0.0.32",
}
ASR_PACKAGE_VERSION = "0.0.6"
VISUAL_TAG_PACKAGE_VERSIONS = {"timm": "0.4.12", "fairscale": "0.4.4"}
PRODUCTION_PYTHON_MINOR = "3.11"
PRODUCTION_TORCH_VERSIONS = {
    "torch": "2.13.0+cu129",
    "torchvision": "0.28.0+cu129",
}


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    production: bool
    device: str
    dtype: str
    asr_enabled: bool
    visual_tags_enabled: bool = False
    captions_enabled: bool = False


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _base_version(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def installed_runtime_versions() -> dict[str, str]:
    packages = [*REQUIRED_PACKAGE_VERSIONS, "qwen-asr", *VISUAL_TAG_PACKAGE_VERSIONS]
    installed: dict[str, str] = {"python": python_version()}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "missing"
    return installed


def validate_runtime_configuration(
    environ: Mapping[str, str],
    installed_versions: Mapping[str, str],
) -> RuntimeProfile:
    problems: list[str] = []
    retired = [name for name in ("USE_TOPICS", "TOPIC_DEFINITIONS_FILE") if name in environ]
    if retired:
        problems.append(
            "semantic topics were removed; remove retired variables: " + ", ".join(retired)
        )
    profile_name = environ.get(RUNTIME_PROFILE_ENV, "production").strip()
    if profile_name not in {"production", "lab-mps"}:
        problems.append(f"{RUNTIME_PROFILE_ENV} must be production or lab-mps")
    device = environ.get("MULTIMODAL_EMBEDDER_DEVICE", "cuda:0")
    if profile_name == "production" and device != "cuda:0":
        problems.append(f"non-CUDA devices require {RUNTIME_PROFILE_ENV}=lab-mps")
    if profile_name == "lab-mps" and device != "mps:0":
        problems.append(f"{RUNTIME_PROFILE_ENV}=lab-mps requires device mps:0")
    installed_python = installed_versions.get("python", "missing")
    if profile_name == "production" and not installed_python.startswith(
        f"{PRODUCTION_PYTHON_MINOR}."
    ):
        problems.append(f"Python {PRODUCTION_PYTHON_MINOR} is required, found {installed_python}")

    identity = {
        "MULTIMODAL_EMBEDDER_MODEL": (
            environ.get("MULTIMODAL_EMBEDDER_MODEL", DEFAULT_MODEL_NAME),
            DEFAULT_MODEL_NAME,
        ),
        "MULTIMODAL_EMBEDDER_MODEL_REVISION": (
            environ.get(
                "MULTIMODAL_EMBEDDER_MODEL_REVISION",
                DEFAULT_MODEL_REVISION,
            ),
            DEFAULT_MODEL_REVISION,
        ),
        "MULTIMODAL_EMBEDDER_DTYPE": (
            environ.get("MULTIMODAL_EMBEDDER_DTYPE", DEFAULT_DTYPE),
            DEFAULT_DTYPE,
        ),
        "MULTIMODAL_EMBEDDER_METHOD_VERSION": (
            environ.get(
                "MULTIMODAL_EMBEDDER_METHOD_VERSION",
                DEFAULT_METHOD_VERSION,
            ),
            DEFAULT_METHOD_VERSION,
        ),
        "MULTIMODAL_EMBEDDER_OUTPUT_DIM": (
            environ.get("MULTIMODAL_EMBEDDER_OUTPUT_DIM", str(DEFAULT_OUTPUT_DIM)),
            str(DEFAULT_OUTPUT_DIM),
        ),
    }
    for field, (actual, expected) in identity.items():
        if actual != expected:
            problems.append(f"{field} must be {expected}")

    for package, base_expected in REQUIRED_PACKAGE_VERSIONS.items():
        actual = installed_versions.get(package, "missing")
        expected = (
            PRODUCTION_TORCH_VERSIONS.get(package, base_expected)
            if profile_name == "production"
            else base_expected
        )
        comparable = (
            _base_version(actual)
            if profile_name == "lab-mps" and package in PRODUCTION_TORCH_VERSIONS
            else actual
        )
        if comparable != expected:
            problems.append(f"{package} must be {expected}, found {actual}")

    asr_enabled = _enabled(environ.get("USE_ASR"))
    if profile_name == "lab-mps" and asr_enabled:
        problems.append("ASR is not approved for the lab-mps profile")
    if asr_enabled:
        retired_overrides = sorted(name for name in environ if name.startswith("TRANSCRIBER_"))
        if retired_overrides:
            problems.append(
                "ASR policy is fixed; remove retired overrides: " + ", ".join(retired_overrides)
            )
        installed_asr = installed_versions.get("qwen-asr", "missing")
        if installed_asr != ASR_PACKAGE_VERSION:
            problems.append(f"qwen-asr must be {ASR_PACKAGE_VERSION}, found {installed_asr}")

    visual_tags_enabled = _enabled(environ.get("USE_VISUAL_TAGS"))
    if visual_tags_enabled:
        for package, expected in VISUAL_TAG_PACKAGE_VERSIONS.items():
            actual = installed_versions.get(package, "missing")
            if actual != expected:
                problems.append(f"{package} must be {expected}, found {actual}")
    captions_enabled = _enabled(environ.get("USE_CAPTIONS"))
    for flag, enabled in (
        ("USE_VISUAL_TAGS", visual_tags_enabled),
        ("USE_CAPTIONS", captions_enabled),
    ):
        if profile_name == "lab-mps" and enabled:
            problems.append(f"{flag} is not approved for the lab-mps profile")

    if problems:
        raise RuntimeError("invalid runtime configuration: " + "; ".join(problems))
    return RuntimeProfile(
        name=profile_name,
        production=profile_name == "production",
        device=device,
        dtype=DEFAULT_DTYPE,
        asr_enabled=asr_enabled,
        visual_tags_enabled=visual_tags_enabled,
        captions_enabled=captions_enabled,
    )
