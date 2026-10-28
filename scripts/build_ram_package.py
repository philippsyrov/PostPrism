"""Build the approved RAM source offline without changing the supplied checkout."""

import argparse
import configparser
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

from postprism.branding import RAM_BUILD_MANIFEST
from postprism.visual_tags.artifacts import (
    PATCH_SHA256,
    checked_path,
    file_sha256,
    validate_package,
)
from postprism.visual_tags.identity import RAM_IDENTITY

SOURCE_COMMIT = RAM_IDENTITY.model_revision
PATCH_FILE = Path(__file__).resolve().parents[1] / "packaging/ram/transformers-457.patch"
OLD_IMPORTS = """from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
"""
NEW_IMPORTS = """from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
"""


def patch_bert_imports(source: str) -> str:
    if source.count(OLD_IMPORTS) != 1 or NEW_IMPORTS in source:
        raise ValueError("expected exactly one approved BERT import block, not already patched")
    return source.replace(OLD_IMPORTS, NEW_IMPORTS, 1)


def _git(source: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("RAM source must be an existing Git checkout") from error


def verify_source(source: Path, expected_commit: str) -> None:
    if _git(source, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("RAM source HEAD does not match pinned commit")
    if _git(source, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError("RAM source must be clean, including untracked and ignored files")
    # Git status alone trusts assume-unchanged/skip-worktree flags. Bind copied bytes
    # to the commit's blobs even when an index flag hides a working-tree edit.
    for entry in _git(source, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, kind, expected_blob = metadata.split()
        path = checked_path(source, name)
        if mode not in {"100644", "100755"} or kind != "blob" or path.is_symlink():
            raise ValueError("RAM source commit requires regular files")
        if _git(source, "hash-object", "--no-filters", "--", str(path)) != expected_blob:
            raise ValueError("RAM source file differs from pinned commit")
    target = source / "ram/models/bert.py"
    if not target.is_file() or target.is_symlink():
        raise ValueError("RAM BERT patch target missing or not a regular file")
    patch_bert_imports(target.read_text(encoding="utf-8"))


def validate_wheel(wheel: Path) -> dict:
    # Validate every member before extraction; never import code from a build output.
    with (
        zipfile.ZipFile(wheel) as archive,
        tempfile.TemporaryDirectory(prefix="ram-wheel-") as temp,
    ):
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel members")
        root = Path(temp)
        for member in archive.infolist():
            name = member.filename
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("unsafe wheel member")
            if member.is_dir():
                continue
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(archive.read(member))
        return validate_package(root / "ram")


def build_package(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or output.exists():
        raise ValueError("build output must be a new directory outside the source checkout")
    verify_source(source, SOURCE_COMMIT)
    if file_sha256(PATCH_FILE) != PATCH_SHA256:
        raise ValueError("tracked compatibility patch hash mismatch")
    with tempfile.TemporaryDirectory(prefix="postprism-ram-build-") as temporary:
        staged = Path(temporary) / "source"
        staged.mkdir()
        source_files = {}
        for name in _git(source, "ls-files", "-z").split("\0"):
            if not name:
                continue
            original = checked_path(source, name)
            if original.is_symlink():
                raise ValueError("RAM source symlinks are not supported")
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            source_files[name] = file_sha256(target)
        bert = staged / "ram/models/bert.py"
        bert.write_text(patch_bert_imports(bert.read_text(encoding="utf-8")), encoding="utf-8")
        shutil.copy2(checked_path(staged, "LICENSE"), staged / "ram/LICENSE")
        shutil.copy2(checked_path(staged, "README.md"), staged / "ram/UPSTREAM_README.md")
        shutil.copy2(PATCH_FILE, staged / "ram/transformers-457.patch")
        package_root = staged / "ram"
        files = {
            p.relative_to(package_root).as_posix(): file_sha256(p)
            for p in sorted(package_root.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "schema_version": 1,
            "source_commit": SOURCE_COMMIT,
            "source_url": "https://github.com/xinyu1205/recognize-anything",
            "patch_sha256": PATCH_SHA256,
            "source_files": source_files,
            "files": files,
            "build_tools": {
                "python": sys.version,
                **{name: version(name) for name in ("pip", "setuptools", "wheel")},
            },
        }
        (package_root / RAM_BUILD_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        config = configparser.ConfigParser(interpolation=None)
        config.read(checked_path(staged, "setup.cfg"), encoding="utf-8")
        config["metadata"]["url"] = manifest["source_url"]
        config["metadata"]["license_files"] = "LICENSE"
        config["options"]["install_requires"] = ""
        config["options"]["include_package_data"] = "True"
        config["options.packages.find"] = {"include": "ram\nram.*"}
        config["options.package_data"] = {"ram": "\n".join([*files, RAM_BUILD_MANIFEST])}
        with (staged / "setup.cfg").open("w", encoding="utf-8") as stream:
            config.write(stream)
        validate_package(package_root)
        wheel_dir = Path(temporary) / "wheels"
        wheel_dir.mkdir()
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        } | {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "SOURCE_DATE_EPOCH": _git(source, "show", "-s", "--format=%ct", "HEAD"),
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(wheel_dir),
                str(staged),
            ],
            check=True,
            env=environment,
            stdout=sys.stderr,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("RAM build must produce exactly one wheel")
        validate_wheel(wheels[0])
        verify_source(source, SOURCE_COMMIT)
        if any(
            file_sha256(checked_path(source, name)) != digest
            for name, digest in source_files.items()
        ):
            raise ValueError("RAM source changed during build")
        output.mkdir(parents=True, exist_ok=False)
        destination = output / wheels[0].name
        shutil.copy2(wheels[0], destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(build_package(args.source_dir, args.output_dir))


if __name__ == "__main__":
    main()
