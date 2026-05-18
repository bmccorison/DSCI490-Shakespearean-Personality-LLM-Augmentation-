"""
Build a speaker-aware context-filtered JSON dataset from the full Hamlet play.

Run from repo root:
    python training/speaker_aware_context_filtering.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import full_play_translator


def _resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the full Hamlet play into speaker-aware context/response "
            "examples for a target speaker."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "data" / "hamlet_full_play.txt"),
        help="Path to the full-play Hamlet text file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(repo_root / "data" / "hamlet_speaker_aware_context.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--speaker",
        default="Hamlet",
        help="Target speaker to filter for. Default: Hamlet.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of most recent non-target turns to keep as context. Default: 3.",
    )
    parser.add_argument(
        "--include-last-speaker-line",
        action="store_true",
        help="Include the most recent prior turn from the target speaker in the context.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    return parser.parse_args()


def build_context_filtered_records(
    turns: list[full_play_translator.SpeechTurn],
    target_speaker: str,
    k: int,
    include_last_speaker_line: bool,
) -> list[dict[str, object]]:
    target_key = full_play_translator.speaker_key(target_speaker)
    records: list[dict[str, object]] = []
    history: list[full_play_translator.SpeechTurn] = []
    last_target_turn: full_play_translator.SpeechTurn | None = None

    for turn in turns:
        is_target_turn = full_play_translator.speaker_key(turn.speaker_raw) == target_key
        if not is_target_turn:
            history.append(turn)
            continue

        context_candidates: dict[int, full_play_translator.SpeechTurn] = {}

        if include_last_speaker_line and last_target_turn is not None:
            context_candidates[last_target_turn.index] = last_target_turn

        if k > 0:
            non_target_count = 0
            for previous_turn in reversed(history):
                if full_play_translator.speaker_key(previous_turn.speaker_raw) == target_key:
                    continue

                context_candidates[previous_turn.index] = previous_turn
                non_target_count += 1
                if non_target_count >= k:
                    break

        context_turns = [
            context_candidates[index]
            for index in sorted(context_candidates)
        ]
        rendered_context_turns = [
            {
                "speaker": context_turn.speaker_display,
                "text": context_turn.text,
                "act": context_turn.act,
                "scene": context_turn.scene,
                "source_line": context_turn.source_line,
            }
            for context_turn in context_turns
        ]
        context_text = "\n".join(
            f"{context_turn['speaker']}: {context_turn['text']}"
            for context_turn in rendered_context_turns
        )

        records.append(
            {
                "speaker": turn.speaker_display,
                "act": turn.act,
                "scene": turn.scene,
                "source_line": turn.source_line,
                "k": k,
                "include_last_speaker_line": include_last_speaker_line,
                "context_turns": rendered_context_turns,
                "context_text": context_text,
                "response": turn.text,
                "response_line": f"{turn.speaker_display}: {turn.text}",
            }
        )

        last_target_turn = turn
        history.append(turn)

    return records


def main() -> None:
    repo_root = full_play_translator.resolve_repo_root()
    args = parse_args(repo_root)

    if args.k < 0:
        raise ValueError("--k must be 0 or greater.")

    input_path = _resolve_repo_relative_path(repo_root, args.input_file)
    output_path = _resolve_repo_relative_path(repo_root, args.output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_lines = input_path.read_text(encoding=args.encoding).splitlines()
    blocks = full_play_translator.parse_full_play_blocks(raw_lines)
    turns = full_play_translator.extract_speech_turns(blocks)
    if not turns:
        raise ValueError("No spoken turns were extracted from the full play file.")

    records = build_context_filtered_records(
        turns,
        target_speaker=args.speaker,
        k=args.k,
        include_last_speaker_line=args.include_last_speaker_line,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding=args.encoding,
    )

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Target speaker: {args.speaker}")
    print(f"Spoken turns parsed: {len(turns)}")
    print(f"Filtered records written: {len(records)}")
    if records:
        print("Sample response:", records[0]["response_line"][:200] + "...")


if __name__ == "__main__":
    main()
