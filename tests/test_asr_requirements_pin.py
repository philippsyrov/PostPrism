"""Guards the qwen-asr install split: the package is hash-pinned to the startup
version in requirements-asr.txt, and the only broken requirement `pip check` may
report is the demo-only gradio dependency skipped on purpose.
"""

import re
from pathlib import Path

from postprism.startup_checks import ASR_PACKAGE_VERSION
from scripts.pip_check import unexpected_problems

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_asr_requirement_is_hash_pinned_to_startup_version() -> None:
    text = (REPO_ROOT / "requirements-asr.txt").read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, flags=re.MULTILINE))
    assert pins == {"qwen-asr": ASR_PACKAGE_VERSION}
    assert "--hash=sha256:" in text


def test_main_requirements_do_not_pull_qwen_asr_or_gradio() -> None:
    text = (REPO_ROOT / "requirements.txt").read_text().lower()
    assert not re.search(r"^(qwen-asr|gradio)\b", text, flags=re.MULTILINE)


def test_pip_check_allows_only_the_skipped_gradio_requirement() -> None:
    assert unexpected_problems("No broken requirements found.\n") == []
    assert unexpected_problems("qwen-asr 0.0.6 requires gradio, which is not installed.\n") == []
    other = "qwen-asr 0.0.6 requires librosa, which is not installed."
    conflict = "torchvision 0.28.0 has requirement pillow<13, but you have pillow 13.0.0."
    assert unexpected_problems(f"{other}\n{conflict}\n") == [other, conflict]
