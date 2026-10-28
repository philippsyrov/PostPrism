#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x .venv/bin/ruff && -x .venv/bin/pytest ]]; then
  ruff=.venv/bin/ruff
  pytest=.venv/bin/pytest
else
  ruff=ruff
  pytest=pytest
fi

"$ruff" format --check .
"$ruff" check .
"$pytest" -q
