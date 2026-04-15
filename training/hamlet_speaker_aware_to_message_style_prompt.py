"""
Convert speaker-aware Hamlet context records into message-style training JSON.

Run from repo root:
    python training/hamlet_speaker_aware_to_message_style_prompt.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import full_play_translator
import speaker_aware_context_filtering as context_filter


DEFAULT_SYSTEM_PROMPT = (
    "You are Hamlet: introspective, philosophical, emotionally conflicted."
)


def _resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_json_records(path: Path, encoding: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding=encoding))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}.")

    records: list[dict[str, object]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected record {index} in {path} to be a JSON object, got "
                f"{type(item).__name__}."
            )
        records.append(item)
    return records


def validate_context_records(records: list[dict[str, object]]) -> None:
    required_record_keys = {"speaker", "response", "context_turns"}
    required_context_keys = {"speaker", "text"}

    for index, record in enumerate(records, start=1):
        missing = required_record_keys - record.keys()
        if missing:
            raise ValueError(
                f"Record {index} is missing required keys: {sorted(missing)}"
            )

        context_turns = record["context_turns"]
        if not isinstance(context_turns, list):
            raise ValueError(
                f"Record {index} has non-list context_turns: "
                f"{type(context_turns).__name__}"
            )

        if not isinstance(record["response"], str):
            raise ValueError(
                f"Record {index} response must be a string, got "
                f"{type(record['response']).__name__}."
            )

        for turn_index, context_turn in enumerate(context_turns, start=1):
            if not isinstance(context_turn, dict):
                raise ValueError(
                    f"Record {index} context turn {turn_index} must be a JSON object, got "
                    f"{type(context_turn).__name__}."
                )

            missing_context = required_context_keys - context_turn.keys()
            if missing_context:
                raise ValueError(
                    f"Record {index} context turn {turn_index} is missing keys: "
                    f"{sorted(missing_context)}"
                )


def _context_records_match_requested_settings(
    records: list[dict[str, object]],
    speaker: str,
    k: int,
    include_last_speaker_line: bool,
) -> bool:
    if not records:
        return False

    requested_speaker = full_play_translator.speaker_key(speaker)

    for record in records:
        record_speaker = full_play_translator.speaker_key(str(record.get("speaker", "")))
        record_k = record.get("k")
        record_include_last = record.get("include_last_speaker_line")

        if record_speaker != requested_speaker:
            return False
        if not isinstance(record_k, int) or record_k != k:
            return False
        if not isinstance(record_include_last, bool):
            return False
        if record_include_last != include_last_speaker_line:
            return False

    return True


def _build_context_records_from_full_play(
    repo_root: Path,
    context_output_path: Path,
    full_play_input_file: str,
    speaker: str,
    k: int,
    include_last_speaker_line: bool,
    encoding: str,
) -> list[dict[str, object]]:
    full_play_path = _resolve_repo_relative_path(repo_root, full_play_input_file)
    if not full_play_path.is_file():
        raise FileNotFoundError(
            "Context JSON file was missing or stale and the fallback full-play text "
            f"file was not found: {full_play_path}"
        )

    raw_lines = full_play_path.read_text(encoding=encoding).splitlines()
    blocks = full_play_translator.parse_full_play_blocks(raw_lines)
    turns = full_play_translator.extract_speech_turns(blocks)
    records = context_filter.build_context_filtered_records(
        turns,
        target_speaker=speaker,
        k=k,
        include_last_speaker_line=include_last_speaker_line,
    )
    validate_context_records(records)

    context_output_path.parent.mkdir(parents=True, exist_ok=True)
    context_output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding=encoding,
    )
    return records


def load_or_build_context_records(
    repo_root: Path,
    context_input_file: str,
    full_play_input_file: str,
    speaker: str,
    k: int,
    include_last_speaker_line: bool,
    encoding: str,
) -> tuple[list[dict[str, object]], Path]:
    context_path = _resolve_repo_relative_path(repo_root, context_input_file)
    if context_path.is_file():
        records = _load_json_records(context_path, encoding)
        validate_context_records(records)
        if _context_records_match_requested_settings(
            records,
            speaker,
            k,
            include_last_speaker_line,
        ):
            return records, context_path

        records = _build_context_records_from_full_play(
            repo_root,
            context_path,
            full_play_input_file,
            speaker,
            k,
            include_last_speaker_line,
            encoding,
        )
        print(
            "Source context JSON settings did not match the requested "
            "speaker-aware configuration; rebuilt it from the full play and "
            f"saved it to: {context_path}"
        )
        return records, context_path

    records = _build_context_records_from_full_play(
        repo_root,
        context_path,
        full_play_input_file,
        speaker,
        k,
        include_last_speaker_line,
        encoding,
    )
    print(
        "Source context JSON was missing; built it from the full play and "
        f"saved it to: {context_path}"
    )
    return records, context_path


def build_message_style_records(
    context_records: list[dict[str, object]],
    target_speaker: str,
    k: int,
    include_last_speaker_line: bool,
    system_prompt: str,
) -> list[dict[str, object]]:
    target_key = full_play_translator.speaker_key(target_speaker)
    output_records: list[dict[str, object]] = []

    for record in context_records:
        context_turns = record["context_turns"]
        previous_speaker_turn: dict[str, object] | None = None
        non_target_turns: list[dict[str, object]] = []

        for context_turn in context_turns:
            context_speaker_key = full_play_translator.speaker_key(
                str(context_turn["speaker"])
            )
            if context_speaker_key == target_key:
                previous_speaker_turn = context_turn
            else:
                non_target_turns.append(context_turn)

        selected_non_target_turns = non_target_turns[-k:] if k > 0 else []
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if include_last_speaker_line and previous_speaker_turn is not None:
            assistant_content = str(previous_speaker_turn["text"]).strip()
            if assistant_content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                )

        for context_turn in selected_non_target_turns:
            speaker = str(context_turn["speaker"]).strip()
            text = str(context_turn["text"]).strip()
            if not speaker or not text:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": f"{speaker}: {text}",
                }
            )

        response = str(record["response"]).strip()
        if not response:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        output_records.append(
            {
                "messages": messages,
                "speaker": record.get("speaker", target_speaker),
                "act": record.get("act"),
                "scene": record.get("scene"),
                "source_line": record.get("source_line"),
                "k": k,
                "include_last_speaker_line": include_last_speaker_line,
                "has_previous_speaker_line": previous_speaker_turn is not None,
                "system_prompt": system_prompt,
                "context_message_count": len(messages) - 2,
            }
        )

    return output_records


def validate_message_style_records(records: list[dict[str, object]]) -> None:
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(
                f"Message-style record {index} has non-list messages: "
                f"{type(messages).__name__}"
            )

        if len(messages) < 2:
            raise ValueError(
                f"Message-style record {index} must contain at least system and "
                "assistant messages."
            )

        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Message-style record {index} message {message_index} must be "
                    f"a JSON object, got {type(message).__name__}."
                )

            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Message-style record {index} message {message_index} has "
                    f"invalid role: {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"Message-style record {index} message {message_index} content "
                    f"must be a string, got {type(content).__name__}."
                )

        if messages[0]["role"] != "system":
            raise ValueError(
                f"Message-style record {index} must start with a system message."
            )
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                f"Message-style record {index} must end with an assistant message."
            )


def _message_records_match_requested_settings(
    records: list[dict[str, object]],
    speaker: str,
    k: int,
    include_last_speaker_line: bool,
    system_prompt: str,
) -> bool:
    if not records:
        return False

    requested_speaker = full_play_translator.speaker_key(speaker)

    for record in records:
        record_speaker = full_play_translator.speaker_key(str(record.get("speaker", "")))
        record_k = record.get("k")
        record_include_last = record.get("include_last_speaker_line")
        record_system_prompt = record.get("system_prompt")

        if record_speaker != requested_speaker:
            return False
        if not isinstance(record_k, int) or record_k != k:
            return False
        if not isinstance(record_include_last, bool):
            return False
        if record_include_last != include_last_speaker_line:
            return False
        if record_system_prompt != system_prompt:
            return False

    return True


def load_or_build_message_records(
    repo_root: Path,
    message_input_file: str,
    context_input_file: str,
    full_play_input_file: str,
    speaker: str,
    k: int,
    include_last_speaker_line: bool,
    system_prompt: str,
    encoding: str,
) -> tuple[list[dict[str, object]], Path]:
    message_path = _resolve_repo_relative_path(repo_root, message_input_file)
    if message_path.is_file():
        message_records = _load_json_records(message_path, encoding)
        validate_message_style_records(message_records)
        if _message_records_match_requested_settings(
            message_records,
            speaker,
            k,
            include_last_speaker_line,
            system_prompt,
        ):
            return message_records, message_path

    context_records, context_path = load_or_build_context_records(
        repo_root,
        context_input_file,
        full_play_input_file,
        speaker,
        k,
        include_last_speaker_line,
        encoding,
    )
    message_records = build_message_style_records(
        context_records,
        target_speaker=speaker,
        k=k,
        include_last_speaker_line=include_last_speaker_line,
        system_prompt=system_prompt,
    )
    validate_message_style_records(message_records)

    message_path.parent.mkdir(parents=True, exist_ok=True)
    message_path.write_text(
        json.dumps(message_records, indent=2, ensure_ascii=False) + "\n",
        encoding=encoding,
    )
    print(f"Source context JSON: {context_path}")
    print(f"Message-style JSON: {message_path}")
    print(f"Message-style records written: {len(message_records)}")

    return message_records, message_path


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert speaker-aware Hamlet context records into message-style "
            "JSON for context-aware LoRA training."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "data" / "hamlet_speaker_aware_context.json"),
        help="Path to the source speaker-aware context JSON file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(repo_root / "data" / "hamlet_speaker_aware_messages.json"),
        help="Path to the output message-style JSON file.",
    )
    parser.add_argument(
        "--full-play-input-file",
        default=str(repo_root / "data" / "hamlet_full_play.txt"),
        help="Fallback full-play text used to rebuild the source context JSON if needed.",
    )
    parser.add_argument(
        "--speaker",
        default="Hamlet",
        help="Target speaker to convert for. Default: Hamlet.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of most recent non-target context turns to keep. Default: 4.",
    )
    parser.add_argument(
        "--include-last-speaker-line",
        action="store_true",
        help="Include the most recent prior target-speaker turn as assistant context.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System anchor to prepend to every message-style record.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = full_play_translator.resolve_repo_root()
    args = parse_args(repo_root)

    if args.k < 0:
        raise ValueError("--k must be 0 or greater.")

    records, output_path = load_or_build_message_records(
        repo_root=repo_root,
        message_input_file=args.output_file,
        context_input_file=args.input_file,
        full_play_input_file=args.full_play_input_file,
        speaker=args.speaker,
        k=args.k,
        include_last_speaker_line=args.include_last_speaker_line,
        system_prompt=args.system_prompt,
        encoding=args.encoding,
    )

    print(f"Target speaker: {args.speaker}")
    print(f"Output file: {output_path}")
    print(f"Message-style records written: {len(records)}")
    if records:
        sample_messages = records[0]["messages"]
        print("Sample final assistant message:", sample_messages[-1]["content"][:200] + "...")


if __name__ == "__main__":
    main()
