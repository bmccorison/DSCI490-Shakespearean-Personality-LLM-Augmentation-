"""Local conversation logging for development."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGGING_DIR = REPO_ROOT / "logging"


class LocalLogging:
    """Persist one conversation to a timestamped JSON file on disk."""

    def __init__(self, logging_dir: Path | None = None):
        self.logging_dir = logging_dir or DEFAULT_LOGGING_DIR
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        self.created_at = datetime.now()
        self.conversation_id = uuid4().hex
        timestamp = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logging_dir / f"{timestamp}_{self.conversation_id}.json"
        self.messages: list[dict[str, Any]] = []
        self._flush()

    def append_message(self, message: dict[str, Any]) -> None:
        """Append one chat message and immediately persist the conversation."""
        self.messages.append(dict(message))
        self._flush()

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace the stored conversation payload and persist it."""
        self.messages = [dict(message) for message in messages]
        self._flush()

    def _flush(self) -> None:
        self.log_file.write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

