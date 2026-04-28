"""Local conversation logging for development."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGGING_DIR = REPO_ROOT / "logging"
# Env-var override that takes precedence over any constructor `category`. Used by the test
# suite to redirect every log emitted during a run into logging/<date>/test/.
LOG_CATEGORY_ENV_VAR = "SHAKESPEARE_LOG_CATEGORY"


class LocalLogging:
    """Persist one conversation to a timestamped JSON file on disk.

    Logs land at ``<logging_root>/<month>_<day>/[category]/<timestamp>_<id>.json``.
    The optional ``category`` slot lets callers separate distinct conversation streams
    (e.g. multimodel dialogues) into their own subdirectory under the date folder.
    """

    def __init__(self, logging_dir: Path | None = None, category: str | None = None):
        self.created_at = datetime.now()
        logging_root = logging_dir or DEFAULT_LOGGING_DIR
        date_dir = logging_root / f"{self.created_at.month}_{self.created_at.day}"

        # Env var overrides the constructor argument so test runs can funnel everything
        # — including category-tagged loggers — into a single directory.
        env_override = os.getenv(LOG_CATEGORY_ENV_VAR)
        resolved_category = (env_override or category or "").strip().strip("/")
        self.logging_dir = date_dir / resolved_category if resolved_category else date_dir

        self.conversation_id = uuid4().hex
        timestamp = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logging_dir / f"{timestamp}_{self.conversation_id}.json"
        self.messages: list[dict[str, Any]] = []

    def append_message(self, message: dict[str, Any]) -> None:
        """Append one chat message and immediately persist the conversation."""
        serialized_message = dict(message)
        if serialized_message.get("role") == "system":
            return

        self.messages.append(serialized_message)
        self._flush()

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace the stored conversation payload and persist it."""
        filtered_messages: list[dict[str, Any]] = []
        for message in messages:
            serialized_message = dict(message)
            if serialized_message.get("role") == "system":
                continue
            filtered_messages.append(serialized_message)

        self.messages = filtered_messages
        self._flush()

    def _flush(self) -> None:
        if not self.messages:
            return

        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
