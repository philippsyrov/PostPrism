# Deployment reference

[README.md](../README.md) is the install path. This file records where every pinned
artifact comes from, what verifies it, and what to check on a new machine.
Do not substitute other versions: each identity below is also enforced at startup.

This repository contains no model files. The README steps fetch them from the
public sources below and check each against a pin in this repository.

## Pinned artifacts

**RAM++ source.** [`xinyu1205/recognize-anything` at `7cb804a8609e9f4b1a50b7f31436d2df40bb9481`](https://github.com/xinyu1205/recognize-anything/tree/7cb804a8609e9f4b1a50b7f31436d2df40bb9481).
`scripts/build_ram_package.py` refuses any other `HEAD`, verifies every tracked file
against its Git blob, and refuses dirty, untracked, or ignored files. It stages a
copy, applies the tracked Transformers 4.57 patch
(`packaging/ram/transformers-457.patch`, SHA-256
`7b4069a11cd2c8d2fa7d53665144291f57e16fcb3e84db0e7b43fcd210695d8d`), and builds
with no dependency resolution, build isolation, or package index. The output
directory must be new and outside the checkout. It prints the validated wheel's
path to stdout and progress to stderr, and never clones, fetches, or installs.
The wheel carries `ram/postprism_ram_build.json` with source and packaged-file hashes
and the build-tool versions; `SOURCE_DATE_EPOCH` comes from the source commit.
Startup re-hashes the installed module files against that manifest rather than
trusting the distribution name.

**Build tools.** pip `26.2.1`, setuptools `84.0.0`, and wheel `0.48.0`. The
packaged-file hashes, which startup checks, do not depend on the toolchain. The
wheel file's own SHA-256 also covers the recorded Python build string, so it can
differ between machines. Keep your wheel's SHA-256 with your deployment notes.

**RAM++ checkpoint.** [`xinyu1205/recognize-anything-plus-model`](https://huggingface.co/xinyu1205/recognize-anything-plus-model)
at revision `84d4aee3a0265c4e0df1f714f0572011d1bf2ec3`, file
`ram_plus_swin_large_14m.pth`: SHA-256
`497c178836ba66698ca226c7895317e6e800034be986452dbd2593298d50e87d`, 3,010,210,801
bytes (`packaging/ram/checkpoint.sha256`). Startup re-hashes it and checks the size,
the 4,585-entry vocabulary, and the threshold hashes.

**Tokenizer.** [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
at revision `86b5e0934494bd15c9632b12f734a8a67f723594`: `vocab.txt`,
`tokenizer_config.json`, and `tokenizer.json`. `packaging/ram/tokenizer-artifacts.json` pins their
hashes. Validation rejects any missing, extra, altered, or symlinked file, so the
directory must hold exactly those three files plus the manifest. Download them
directly as the README does; Hugging Face's `--local-dir` downloads add a hidden
cache folder that validation rejects.

**Extra Python packages.** `timm==0.4.12` and `fairscale==0.4.4`, from the
[pinned upstream requirements](https://github.com/xinyu1205/recognize-anything/blob/7cb804a8609e9f4b1a50b7f31436d2df40bb9481/requirements.txt),
are hash-pinned PyPI files in `requirements-visual-tags.txt`. `fairscale` is a
source-only release; `--no-build-isolation` builds it with the pinned setuptools
instead of fetching unpinned build tools. `pip check` and the RAM import pass with
these two pins on the canonical stack (Python 3.11,
Torch `2.13.0+cu129`, torchvision `0.28.0+cu129`, Transformers `4.57.6`); RAM's
`scipy` import is satisfied by the canonical requirements. Do not install
upstream's mutable CLIP URL or unbounded requirements. A missing dependency or
conflict blocks enabling RAM and needs a failing test plus an exact reviewed pin;
never downgrade canonical dependencies.

**Hugging Face models.** Downloaded on first start at these revisions:

- Qwen embedding: `Qwen/Qwen3-VL-Embedding-2B` at
  `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`.
- Qwen ASR: `Qwen/Qwen3-ASR-0.6B` at
  `bda30b99af3446a954339e28465acf530464e52e`.
- Qwen caption: `Qwen/Qwen3-VL-2B-Instruct` at
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Speech/music gate: `MIT/ast-finetuned-audioset-10-10-0.4593` at
  `f826b80d28226b62986cc218e5cec390b1096902`.

Keep deployed artifacts read-only.

## Check before opening the service

`python -m scripts.check_ram_artifacts`, run after the README install with the
environment block exported, validates the runtime pins and flags, the installed RAM package
files, and the tokenizer manifest. It then imports RAM++ and loads the tokenizer
with socket connections blocked. It constructs no model weights and does not hash
the checkpoint; the README's `sha256sum` step and startup do. Save its JSON output
with the wheel SHA-256. A prior GPU run is not proof that a
newly provisioned machine has compatible artifacts or enough capacity.

## Check on a new GPU

Before sending real posts to a new machine:

1. Keep the JSON from `python -m scripts.check_ram_artifacts` and the RAM wheel's
   SHA-256.
2. Confirm `/health` reports your `service_commit`, `"runtime_profile":
   "production"`, the model revisions and method versions pinned in this file and
   [INTEGRATION.md](INTEGRATION.md), and `asr`, `visual_tags`, and `captions`
   enabled.
3. Send the README quickstart request and confirm `content_enrichment_status`,
   `multimodal_embedding_status`, `visual_tag_result.status`, and
   `caption_result.status` are all `ok` and the vector has 1024 numbers.
4. With every worker enabled, send a few representative posts of your own,
   including a ten-video post and a mixed image/video post. Record latency and
   peak GPU memory (`nvidia-smi`) and size backend timeouts and concurrency from
   those numbers.
5. Review tags and captions on a sample of those posts. Tags carry no moderation
   or quality decision.

These checks show the install works on your GPU. They do not show that vectors are
identical to ones made on a different GPU or driver; if you move an existing index
to new hardware, compare vectors for the same posts before mixing them.

## Disconnected machines

Prepare on a connected machine with the same platform and Python. Build a
wheelhouse for the extra packages with
`python -m pip wheel --no-deps --no-build-isolation -r requirements-visual-tags.txt -w wheelhouse`
and install from it with `--no-deps --no-index --find-links wheelhouse`; do the
same for `requirements.txt`. Copy the RAM wheel, checkpoint, and tokenizer
directory, cache all four Hugging Face models with their processors or feature
extractors, then set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Troubleshooting

- Startup fails naming `USE_TOPICS` or `TOPIC_DEFINITIONS_FILE`: semantic topics were
  removed; unset both, even if empty or `false`.
- `build_ram_package` refuses the output directory: it must be new. Delete
  `$POSTPRISM_ARTIFACTS/ram-build` to rebuild.
- `sha256sum` reports `FAILED`, or startup reports an integrity mismatch: delete the
  file and download it again. Never edit a pin to match a download.
- The artifact check or startup names a package version: install exactly the pinned
  requirements in a fresh `.venv`.

## Local Apple Silicon experiment

For a local experiment on Apple Silicon only, set
`POSTPRISM_RUNTIME_PROFILE=lab-mps` and `MULTIMODAL_EMBEDDER_DEVICE=mps:0`, and
leave ASR, RAM++, and captions disabled. That profile is not for production; its
vectors have not been compared with NVIDIA output.
