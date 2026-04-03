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
        self.created_at = datetime.now()
        logging_root = logging_dir or DEFAULT_LOGGING_DIR
        self.logging_dir = logging_root / f"{self.created_at.month}_{self.created_at.day}"
        self.conversation_id = uuid4().hex
        timestamp = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logging_dir / f"{timestamp}_{self.conversation_id}.json"
        self.messages: list[dict[str, Any]] = []
        self.model_name: str = ""
        self.adapter_path: str = ""

    def set_model(self, model_name: str, adapter_path: str) -> None:
        """Record which model and adapter are active for this conversation."""
        self.model_name = model_name.strip()
        self.adapter_path = adapter_path.strip()
        self._flush()

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
        payload: dict[str, Any] = {
            "model": self.model_name,
            "adapter": self.adapter_path,
            "messages": self.messages,
        }
        self.log_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
