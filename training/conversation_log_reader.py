"""
conversation_log_reader.py
Reads all saved conversation logs and yields weighted training examples
compatible with lora_5.py's build_message_style_examples() format.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pipeline.local_logging import LocalLogging, DEFAULT_LOGGING_DIR
from pipeline.feedback_store import load_feedback


def iter_log_files(logging_dir: Path = DEFAULT_LOGGING_DIR):
    """Yield every .json log file found under the logging directory."""
    for path in sorted(logging_dir.rglob("*.json")):
        yield path


def score_from_feedback(
    message_index: int,
    feedback_records: list[dict],
) -> float | None:
    """
    Return a scalar weight from explicit user feedback, or None if
    no feedback exists for this message index.
    """
    for record in feedback_records:
        if record.get("message_index") != message_index:
            continue
        vote = record.get("vote")
        spans = record.get("spans", [])

        base = {"up": 1.5, "down": 0.3}.get(vote, 1.0)

        for span in spans:
            if span.get("polarity") == "good":
                base = min(2.0, base + 0.1)
            elif span.get("polarity") == "bad":
                base = max(0.1, base - 0.15)

        return base
    return None


def load_weighted_examples_from_logs(
    logging_dir: Path = DEFAULT_LOGGING_DIR,
    system_prompt: str = "",
    min_weight: float = 0.0,
) -> list[dict[str, Any]]:
    all_examples: list[dict[str, Any]] = []
    dummy_logger = LocalLogging.__new__(LocalLogging)

    for log_file in iter_log_files(logging_dir):
        # Skip feedback files so we don't try to parse them as conversations
        if log_file.suffix == ".feedback.json" or ".feedback" in log_file.name:
            continue

        try:
            messages: list[dict[str, Any]] = json.loads(
                log_file.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {log_file}: {exc}")
            continue

        if not isinstance(messages, list) or len(messages) < 2:
            continue

        # Always initialize feedback_records before using it
        feedback_records = load_feedback(log_file)

        dummy_logger.messages = messages

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            if i + 1 >= len(messages):
                continue

            response = msg["content"]

            explicit_score = score_from_feedback(i, feedback_records)
            if explicit_score is not None:
                weight = explicit_score
            else:
                weight = dummy_logger.score_response(messages[i + 1]["content"])

            if weight < min_weight:
                continue

            feedback_for_message = next(
                (r for r in feedback_records if r.get("message_index") == i),
                None,
            )
            spans = feedback_for_message.get("spans", []) if feedback_for_message else []

            context_messages = list(messages[:i + 1])
            if system_prompt:
                context_messages = [
                    {"role": "system", "content": system_prompt}
                ] + context_messages

            all_examples.append({
                "messages": context_messages,
                "response": response,
                "token_weights": None,
                "spans": spans,
                "scalar_weight": weight,
                "source_log": str(log_file),
            })

    print(f"Loaded {len(all_examples)} weighted examples from logs.")
    return all_examples
