"""Guards that the README download pins agree with the runtime's own RAM identity.

The install steps check downloads against files in packaging/ram and
requirements-visual-tags.txt; if those drift from the startup checks, a developer
would fetch artifacts that only fail later, at worker startup.
"""

import json
import re
from pathlib import Path

from postprism.startup_checks import VISUAL_TAG_PACKAGE_VERSIONS
from postprism.visual_tags.identity import RAM_IDENTITY

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_hash_file_matches_runtime_identity() -> None:
    pinned = (REPO_ROOT / "packaging/ram/checkpoint.sha256").read_text().strip()
    assert pinned == RAM_IDENTITY.artifacts["checkpoint_sha256"]


def test_tokenizer_manifest_pins_an_immutable_revision() -> None:
    manifest = json.loads((REPO_ROOT / "packaging/ram/tokenizer-artifacts.json").read_text())
    assert manifest["model"] == "bert-base-uncased"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"])
    assert set(manifest["files"]) == {"vocab.txt", "tokenizer_config.json", "tokenizer.json"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in manifest["files"].values())


def test_visual_tag_requirements_are_hash_pinned_to_startup_versions() -> None:
    text = (REPO_ROOT / "requirements-visual-tags.txt").read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, flags=re.MULTILINE))
    assert pins == VISUAL_TAG_PACKAGE_VERSIONS
    for name in pins:
        block = re.search(rf"^{re.escape(name)}==.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
        assert block is not None and "--hash=sha256:" in block.group(0)
