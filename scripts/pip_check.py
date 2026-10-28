"""Run `pip check`, allowing only the dependencies skipped on purpose.

qwen-asr installs with --no-deps (see requirements-asr.txt) so its demo-only
gradio requirement can't hold Pillow back. Any other broken requirement fails.
"""

import subprocess
import sys

ALLOWED = frozenset({"qwen-asr 0.0.6 requires gradio, which is not installed."})
CLEAN = "No broken requirements found."


def unexpected_problems(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [line for line in lines if line != CLEAN and line not in ALLOWED]


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    problems = unexpected_problems(result.stdout + result.stderr)
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
