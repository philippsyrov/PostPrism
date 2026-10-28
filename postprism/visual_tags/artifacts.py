"""Model-free validation for the packaged RAM runtime and local BERT tokenizer."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from postprism.branding import RAM_BUILD_MANIFEST
from postprism.visual_tags.identity import RAM_IDENTITY

PATCH_SHA256 = "7b4069a11cd2c8d2fa7d53665144291f57e16fcb3e84db0e7b43fcd210695d8d"
REQUIRED_DATA = (
    "configs/med_config.json",
    "configs/q2l_config.json",
    "configs/swin/config_swinL_384.json",
    "data/ram_tag_list.txt",
    "data/ram_tag_list_chinese.txt",
    "data/ram_tag_list_threshold.txt",
)
REQUIRED_PACKAGE_FILES = (
    *REQUIRED_DATA,
    "__init__.py",
    "models/bert.py",
    "LICENSE",
    "UPSTREAM_README.md",
    "transformers-457.patch",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid artifact path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("invalid artifact path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact path missing or escapes root")
    return path


def _read_manifest(root: Path, name: str) -> dict:
    try:
        manifest = json.loads(checked_path(root, name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("artifact manifest missing or invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be an object")
    return manifest


def _validate_files(
    root: Path, manifest: dict, manifest_name: str, required: tuple, *, bytecode: bool = False
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or not set(required).issubset(files):
        raise ValueError("artifact manifest omits required files")
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid artifact SHA-256")
        if file_sha256(checked_path(root, name)) != expected:
            raise ValueError("artifact file hash mismatch")
    actual = set()
    for path in root.rglob("*"):
        if bytecode and "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_file() or path.is_symlink():
            name = path.relative_to(root).as_posix()
            checked_path(root, name)
            if name != manifest_name:
                actual.add(name)
    if actual != set(files):
        raise ValueError("unlisted or missing artifact files")


def validate_package(package_root: Path) -> dict:
    manifest = _read_manifest(package_root, RAM_BUILD_MANIFEST)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_commit") != RAM_IDENTITY.model_revision
        or manifest.get("patch_sha256") != PATCH_SHA256
    ):
        raise ValueError("RAM package build identity mismatch")
    _validate_files(
        package_root, manifest, RAM_BUILD_MANIFEST, REQUIRED_PACKAGE_FILES, bytecode=True
    )
    if manifest["files"]["transformers-457.patch"] != PATCH_SHA256:
        raise ValueError("RAM packaged compatibility patch mismatch")
    return manifest


def validate_tokenizer(tokenizer_root: Path) -> dict:
    manifest = _read_manifest(tokenizer_root, "tokenizer-artifacts.json")
    revision = manifest.get("source_revision")
    if (
        manifest.get("model") != "bert-base-uncased"
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("BERT tokenizer requires model and immutable source revision")
    _validate_files(
        tokenizer_root, manifest, "tokenizer-artifacts.json", ("vocab.txt", "tokenizer_config.json")
    )
    return manifest
