from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The README exports these in the same terminal that runs ./verify.sh; unit tests
# must never inherit a deployment shell (USE_ASR=true would load real models).
SERVICE_ENV_PREFIXES = ("USE_", "RAM_", "MULTIMODAL_EMBEDDER_", "TRANSCRIBER_")
SERVICE_ENV_NAMES = frozenset(
    {
        "POSTPRISM_ARTIFACTS",
        "POSTPRISM_RUNTIME_PROFILE",
        "SERVICE_COMMIT",
        "TOPIC_DEFINITIONS_FILE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    }
)


@pytest.fixture(autouse=True)
def isolate_service_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name in SERVICE_ENV_NAMES or name.startswith(SERVICE_ENV_PREFIXES):
            monkeypatch.delenv(name)
