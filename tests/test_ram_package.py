import hashlib
import json
import os
import subprocess
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

from postprism.visual_tags.artifacts import validate_package, validate_tokenizer

PACKAGE_DATA = (
    "configs/med_config.json",
    "configs/q2l_config.json",
    "configs/swin/config_swinL_384.json",
    "data/ram_tag_list.txt",
    "data/ram_tag_list_chinese.txt",
    "data/ram_tag_list_threshold.txt",
)


@pytest.fixture
def package_root(tmp_path):
    root = tmp_path / "ram"
    for name in (
        *PACKAGE_DATA,
        "__init__.py",
        "models/bert.py",
        "LICENSE",
        "UPSTREAM_README.md",
        "transformers-457.patch",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + name)
    patch = Path(__file__).resolve().parents[1] / "packaging/ram/transformers-457.patch"
    (root / "transformers-457.patch").write_bytes(patch.read_bytes())
    manifest = {
        "schema_version": 1,
        "source_commit": "7cb804a8609e9f4b1a50b7f31436d2df40bb9481",
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    (root / "postprism_ram_build.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("missing", [*PACKAGE_DATA, "postprism_ram_build.json", "models/bert.py"])
def test_package_rejects_missing_runtime_data(package_root, missing):
    (package_root / missing).unlink()
    with pytest.raises(ValueError):
        validate_package(package_root)


@pytest.mark.parametrize("change", ["hash", "source", "patch", "extra", "unlisted", "traversal"])
def test_package_rejects_identity_drift_and_unlisted_code(package_root, change):
    manifest_path = package_root / "postprism_ram_build.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "hash":
        (package_root / "models/bert.py").write_text("tampered")
    elif change == "source":
        manifest["source_commit"] = "f" * 40
    elif change == "patch":
        manifest["patch_sha256"] = "f" * 64
    elif change == "extra":
        (package_root / "injected.py").write_text("unlisted")
    elif change == "unlisted":
        del manifest["files"]["configs/med_config.json"]
    else:
        manifest["files"]["../escape"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_package(package_root)


@pytest.fixture
def tokenizer_root(tmp_path):
    root = tmp_path / "tokenizer"
    root.mkdir()
    files = {
        "vocab.txt": b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n",
        "tokenizer_config.json": b'{"do_lower_case":true}',
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {
        "model": "bert-base-uncased",
        "source_revision": "a" * 40,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (root / "tokenizer-artifacts.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "hash",
        "escape",
        "revision_missing",
        "revision_branch",
        "revision_short",
        "model",
        "unlisted",
        "no_vocab",
        "traversal",
    ],
)
def test_tokenizer_rejects_unverified_artifacts(tokenizer_root, change):
    path = tokenizer_root / "tokenizer-artifacts.json"
    manifest = json.loads(path.read_text())
    if change == "missing":
        (tokenizer_root / "vocab.txt").unlink()
    elif change == "hash":
        (tokenizer_root / "vocab.txt").write_text("altered")
    elif change == "escape":
        outside = tokenizer_root.parent / "outside"
        outside.write_bytes((tokenizer_root / "vocab.txt").read_bytes())
        (tokenizer_root / "vocab.txt").unlink()
        (tokenizer_root / "vocab.txt").symlink_to(outside)
    elif change.startswith("revision"):
        manifest.pop("source_revision")
        if change != "revision_missing":
            manifest["source_revision"] = "main" if change == "revision_branch" else "a" * 7
    elif change == "model":
        manifest["model"] = "other-bert"
    elif change == "unlisted":
        (tokenizer_root / "added_tokens.json").write_text("{}")
    elif change == "no_vocab":
        del manifest["files"]["vocab.txt"]
    else:
        manifest["files"]["../outside"] = "f" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_tokenizer(tokenizer_root)


def test_valid_artifacts_return_their_identity_for_acceptance_evidence(
    package_root, tokenizer_root
):
    assert (
        validate_package(package_root)["source_commit"]
        == "7cb804a8609e9f4b1a50b7f31436d2df40bb9481"
    )
    assert validate_tokenizer(tokenizer_root)["source_revision"] == "a" * 40


def synthetic_wheel(root, destination, omit=None):
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.relative_to(root).as_posix() != omit:
                archive.write(path, "ram/" + path.relative_to(root).as_posix())
    return destination


@pytest.mark.parametrize("missing", [*PACKAGE_DATA, "postprism_ram_build.json"])
def test_wheel_rejects_each_missing_required_file(package_root, tmp_path, missing):
    from scripts.build_ram_package import validate_wheel

    wheel = synthetic_wheel(package_root, tmp_path / "ram.whl", missing)
    with pytest.raises(ValueError):
        validate_wheel(wheel)


def test_wheel_rejects_duplicate_and_escaping_members(package_root, tmp_path):
    from scripts.build_ram_package import validate_wheel

    for member in ("ram/__init__.py", "../escape"):
        wheel = synthetic_wheel(package_root, tmp_path / "ram.whl")
        with (
            pytest.warns(UserWarning, match="Duplicate")
            if member.startswith("ram/")
            else nullcontext(),
            zipfile.ZipFile(wheel, "a") as archive,
        ):
            archive.writestr(member, "injected")
        with pytest.raises(ValueError):
            validate_wheel(wheel)


def test_wheel_validates_bytes_without_importing_ram(package_root, tmp_path):
    from scripts.build_ram_package import validate_wheel

    wheel = synthetic_wheel(package_root, tmp_path / "ram.whl")
    assert validate_wheel(wheel)["source_commit"] == "7cb804a8609e9f4b1a50b7f31436d2df40bb9481"


def test_build_stages_only_patch_and_packaging_without_mutating_source(
    source_repo, tmp_path, monkeypatch
):
    import configparser
    import sys

    from postprism.visual_tags import artifacts
    from scripts import build_ram_package as builder

    source, _ = source_repo
    for name in ("__init__.py", *PACKAGE_DATA):
        path = source / "ram" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + name)
    (source / "LICENSE").write_text("upstream license")
    (source / "README.md").write_text("upstream attribution")
    (source / "setup.py").write_text("import setuptools\nsetuptools.setup()\n")
    (source / "setup.cfg").write_text(
        "[metadata]\nname = ram\nversion = 0.0.1\n[options]\npackages = find:\n"
    )
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "package",
    )
    commit = git(source, "rev-parse", "HEAD")
    before = {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    monkeypatch.setattr(builder, "SOURCE_COMMIT", commit, raising=False)
    monkeypatch.setattr(builder, "version", lambda name: "test-build-tool")
    monkeypatch.setattr(artifacts, "RAM_IDENTITY", type("Identity", (), {"model_revision": commit}))
    real_run = subprocess.run
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert not {"clone", "fetch", "install", "pull", "push"}.intersection(command)
        if command[0] == "git":
            return real_run(command, **kwargs)
        assert command[:4] == [sys.executable, "-m", "pip", "wheel"]
        assert {"--no-deps", "--no-build-isolation", "--no-index"}.issubset(command)
        assert kwargs["env"]["PIP_NO_INDEX"] == "1"
        assert int(kwargs["env"]["SOURCE_DATE_EPOCH"]) > 0
        staged = Path(command[-1])
        assert staged != source and not (staged / ".git").exists()
        assert (staged / "ram/models/bert.py").read_text() == NEW_IMPORTS
        assert (staged / "ram/LICENSE").read_text() == "upstream license"
        assert (staged / "ram/UPSTREAM_README.md").read_text() == "upstream attribution"
        config = configparser.ConfigParser()
        config.read(staged / "setup.cfg")
        data = config["options.package_data"]["ram"].split()
        assert set(PACKAGE_DATA).issubset(data)
        assert "postprism_ram_build.json" in data
        assert config["options"]["install_requires"] == ""
        wheel_dir = Path(command[command.index("--wheel-dir") + 1])
        synthetic_wheel(staged / "ram", wheel_dir / "ram-0.0.1-py3-none-any.whl")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builder.subprocess, "run", run)
    output = tmp_path / "built"
    wheel = builder.build_package(source, output)
    assert wheel == output / "ram-0.0.1-py3-none-any.whl"
    assert builder.validate_wheel(wheel)["source_commit"] == commit
    assert before == {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert git(source, "rev-parse", "HEAD") == commit
    assert git(source, "status", "--porcelain") == ""
    assert len([c for c in commands if c[0] != "git"]) == 1


def test_builder_refuses_output_inside_source_or_existing_directory(source_repo, tmp_path):
    from scripts.build_ram_package import build_package

    source, _ = source_repo
    for output in (source / "new", tmp_path):
        with pytest.raises(ValueError):
            build_package(source, output)


def test_adapter_validates_actual_import_location_before_loading_runtime(
    package_root, tokenizer_root, monkeypatch
):
    from importlib.machinery import ModuleSpec

    from postprism.visual_tags import adapter

    spec = ModuleSpec("ram", loader=None, origin=str(package_root / "__init__.py"), is_package=True)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: spec)
    adapter._verify_runtime_artifacts(tokenizer_root)
    (package_root / "models/bert.py").write_text("changed after installation")
    with pytest.raises(ValueError, match="hash"):
        adapter._verify_runtime_artifacts(tokenizer_root)


def test_adapter_rejects_missing_ram_package(tokenizer_root, monkeypatch):
    from postprism.visual_tags import adapter

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ValueError, match="package"):
        adapter._verify_runtime_artifacts(tokenizer_root)


def test_source_rejects_hidden_worktree_edits_even_if_git_status_is_clean(source_repo):
    from scripts.build_ram_package import verify_source

    source, commit = source_repo
    git(source, "update-index", "--assume-unchanged", "ram/models/bert.py")
    (source / "ram/models/bert.py").write_text(OLD_IMPORTS + "# hidden drift\n")
    assert git(source, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="commit"):
        verify_source(source, commit)


def test_source_checks_ignore_inherited_git_hook_repository(source_repo, tmp_path, monkeypatch):
    from scripts.build_ram_package import verify_source

    source, commit = source_repo
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "other"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "other-index"))
    verify_source(source, commit)
    assert git(source, "rev-parse", "HEAD") == commit


def test_package_rejects_wrong_patch_bytes_even_with_updated_file_manifest(package_root):
    patch = package_root / "transformers-457.patch"
    patch.write_text("different patch")
    path = package_root / "postprism_ram_build.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["transformers-457.patch"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="patch"):
        validate_package(package_root)


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


def git(source, *args):
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    ).stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "ram/models").mkdir(parents=True)
    (source / "ram/models/bert.py").write_text(OLD_IMPORTS)
    git(source, "init", "-q")
    git(source, "add", "ram")
    git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return source, git(source, "rev-parse", "HEAD")


def test_patch_rejects_ambiguous_source():
    from scripts.build_ram_package import patch_bert_imports

    with pytest.raises(ValueError, match="exactly one"):
        patch_bert_imports("not the approved source")


def test_patch_changes_only_the_approved_import_block():
    from scripts.build_ram_package import patch_bert_imports

    assert patch_bert_imports("# upstream\n" + OLD_IMPORTS + "# end\n") == (
        "# upstream\n" + NEW_IMPORTS + "# end\n"
    )


@pytest.mark.parametrize("source", [OLD_IMPORTS * 2, NEW_IMPORTS, OLD_IMPORTS + NEW_IMPORTS])
def test_patch_rejects_repeated_or_already_patched_source(source):
    from scripts.build_ram_package import patch_bert_imports

    with pytest.raises(ValueError, match="exactly one|already"):
        patch_bert_imports(source)


@pytest.mark.parametrize(
    "change", ["dirty", "untracked", "ignored", "wrong_head", "missing", "zero", "two", "patched"]
)
def test_verify_source_rejects_invalid_checkout(source_repo, change):
    from scripts.build_ram_package import verify_source

    source, commit = source_repo
    bert = source / "ram/models/bert.py"
    if change == "wrong_head":
        commit = "f" * 40
    elif change == "missing":
        bert.unlink()
    elif change in {"dirty", "zero", "two", "patched"}:
        bert.write_text(
            {
                "dirty": OLD_IMPORTS + "# edit",
                "zero": "# absent",
                "two": OLD_IMPORTS * 2,
                "patched": NEW_IMPORTS,
            }[change]
        )
    elif change == "ignored":
        (source / ".git/info").mkdir(exist_ok=True)
        (source / ".git/info/exclude").write_text("secret\n")
        (source / "secret").write_text("not part of the commit")
    else:
        (source / "extra").write_text("untracked")
    if change in {"missing", "zero", "two", "patched"}:
        git(source, "add", "-u")
        git(
            source,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "change",
        )
        commit = git(source, "rev-parse", "HEAD")
    with pytest.raises(ValueError):
        verify_source(source, commit)


def test_verify_source_accepts_clean_exact_commit_without_changing_it(source_repo):
    from scripts.build_ram_package import verify_source

    source, commit = source_repo
    before = (source / "ram/models/bert.py").read_bytes()
    verify_source(source, commit)
    assert (source / "ram/models/bert.py").read_bytes() == before
    assert git(source, "rev-parse", "HEAD") == commit
    assert git(source, "status", "--porcelain") == ""
