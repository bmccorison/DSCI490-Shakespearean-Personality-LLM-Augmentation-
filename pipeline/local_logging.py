"""Local conversation logging for development."""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGGING_DIR = REPO_ROOT / "logging"
# Env-var prefix used by the test suite to redirect logs into logging/<date>/test/.
LOG_CATEGORY_ENV_VAR = "SHAKESPEARE_LOG_CATEGORY"


def _category_path_parts(*categories: str | None) -> list[str]:
    """Return safe relative path parts for optional log categories."""
    path_parts: list[str] = []
    for category in categories:
        normalized_category = (category or "").strip().strip("/")
        if not normalized_category:
            continue

        path_parts.extend(
            path_part
            for path_part in normalized_category.split("/")
            if path_part and path_part not in {".", ".."}
        )

    return path_parts


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

        category_path_parts = _category_path_parts(
            os.getenv(LOG_CATEGORY_ENV_VAR),
            category,
        )
        self.logging_dir = (
            date_dir.joinpath(*category_path_parts)
            if category_path_parts
            else date_dir
        )

        self.conversation_id = uuid4().hex
        timestamp = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logging_dir / f"{timestamp}_{self.conversation_id}.json"
        self.messages: list[dict[str, Any]] = []

    def score_response(self, user_followup: str) -> float:
        user_followup = user_followup.lower()
        negative_signals = [
            "no", "that's wrong", "incorrect", "not what i meant",
            "try again", "wrong", "fix", "doesn't make sense","chud", "dumbass", "idiot", "vile hellspawn", "downvote", "no mistakes btw"
        ]
        positive_signals = [
            "thanks", "ok", "good", "nice", "that works", "thank you", "yes", "upvote", "glaze", "big mcthankies from mcspankies"
        ]
        for phrase in negative_signals:
            if phrase in user_followup:
                return 0.3
        for phrase in positive_signals:
            if phrase in user_followup:
                return 1.5
        return 1.0

    def extract_weighted_examples(self) -> list[dict[str, Any]]:
        """
        Walk the stored messages and assign a reward weight to each
        assistant turn based on the user follow-up that came after it.
        """
        examples = []
        messages = self.messages
        for i in range(len(messages) - 1):
            if messages[i]["role"] != "assistant":
                continue
            response = messages[i]["content"]
            next_msg = messages[i + 1]
            if next_msg["role"] != "user":
                continue
            weight = self.score_response(next_msg["content"])
            examples.append({
                "messages": messages[:i + 1],
                "response": response,
                "weight": weight,
            })
        return examples



    def _flush(self) -> None:
        if len(self.messages) < 2:
            return
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clean_content(self, content: str) -> str:
        """Strip leaked system/role tokens from model output."""
        content = re.sub(r'<\|?\s*/?\s*system\s*\|?\s*>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<\|?\s*(user|assistant|system)\s*\|?\s*>', '', content, flags=re.IGNORECASE)
        return content.strip()

    def append_message(self, message: dict[str, Any]) -> None:
        serialized_message = dict(message)
        if serialized_message.get("role") == "system":
            return
        if "content" in serialized_message:
            serialized_message["content"] = self._clean_content(serialized_message["content"])
        self.messages.append(serialized_message)
        self._flush()

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace the stored conversation payload and persist it."""
        filtered: list[dict[str, Any]] = [
            dict(m) for m in messages if m.get("role") != "system"
        ]
        self.messages = filtered
        self._flush()
