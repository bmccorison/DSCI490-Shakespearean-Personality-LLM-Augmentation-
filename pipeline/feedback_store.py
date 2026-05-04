"""Persist and retrieve per-message feedback alongside conversation logs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def save_feedback(
    log_file: Path,
    feedback: list[dict[str, Any]],
) -> None:
    # Strip suffixes until we get the base name without any .json or .feedback
    base = log_file.parent / log_file.name.replace(".feedback.json", "").replace(".json", "")
    feedback_file = base.with_suffix(".feedback.json")
    feedback_file.write_text(
        json.dumps(feedback, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def load_feedback(log_file: Path) -> list[dict[str, Any]]:
    base = log_file.parent / log_file.name.replace(".feedback.json", "").replace(".json", "")
    feedback_file = base.with_suffix(".feedback.json")
    if not feedback_file.exists():
        return []
    try:
        return json.loads(feedback_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
