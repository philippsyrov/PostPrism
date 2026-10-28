"""Guards that `python app.py` emits service INFO lines once, timestamped, without library noise.

Runs in a subprocess so the logger changes cannot leak into caplog-based tests.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors app.py's main block: configure, then uvicorn applies its dictConfig
# and re-imports `app:app`, which must not add a second handler.
SCRIPT = textwrap.dedent(
    """
    import importlib
    import logging
    import logging.config

    from uvicorn.config import LOGGING_CONFIG

    from postprism.api.application import configure_service_logging

    configure_service_logging()
    logging.config.dictConfig(LOGGING_CONFIG)
    importlib.import_module("app")
    configure_service_logging()

    logging.getLogger("app").info("service-info-line")
    logging.getLogger("postprism.asr.transcriber").info("package-info-line")
    logging.getLogger("httpx").info("library-info-line")
    """
)

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO ")


def test_service_info_lines_are_emitted_once_with_timestamps() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = completed.stderr.splitlines()

    for marker, logger_name in (
        ("service-info-line", "app"),
        ("package-info-line", "postprism.asr.transcriber"),
    ):
        matching = [line for line in lines if marker in line]
        assert len(matching) == 1, completed.stderr
        assert TIMESTAMP.match(matching[0]), matching[0]
        assert f" {logger_name}: " in matching[0]
    assert not any("library-info-line" in line for line in lines)
