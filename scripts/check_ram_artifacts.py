"""Check the installed runtime and RAM++ artifacts before opening the service.

Validates the runtime pins and flags, the installed RAM package's file hashes,
and the local tokenizer manifest, then imports RAM++ and loads the tokenizer
with socket connections blocked. It constructs no model weights and does not
hash the checkpoint; the README's `sha256sum` step and worker startup do that.
Prints the evidence JSON to keep with the deployment record. See docs/DEPLOYMENT.md.
"""

import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from postprism.startup_checks import (
    installed_runtime_versions,
    validate_runtime_configuration,
)
from postprism.visual_tags.adapter import _verify_runtime_artifacts
from postprism.visual_tags.artifacts import validate_tokenizer


def _no_network(*args, **kwargs):
    raise RuntimeError("network access forbidden in RAM artifact check")


def main() -> None:
    versions = installed_runtime_versions()
    validate_runtime_configuration(os.environ, versions)
    tokenizer_dir = Path(os.environ["RAM_TOKENIZER_DIR"]).resolve()
    with (
        patch("socket.socket.connect", _no_network),
        patch("socket.socket.connect_ex", _no_network),
        patch("socket.create_connection", _no_network),
    ):
        _verify_runtime_artifacts(tokenizer_dir)
        importlib.import_module("ram.models.ram_plus")
        from ram.models.utils import init_tokenizer  # noqa: PLC0415

        tokenizer = init_tokenizer(str(tokenizer_dir))
        if len(tokenizer) <= 0:
            raise RuntimeError("RAM tokenizer is empty")
    print(
        json.dumps({"runtime": versions, "tokenizer": validate_tokenizer(tokenizer_dir)}, indent=2)
    )


if __name__ == "__main__":
    main()
