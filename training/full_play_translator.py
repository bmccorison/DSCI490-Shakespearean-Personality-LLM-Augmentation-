"""
Parse and translate the Folger-formatted full text of Hamlet.

Note: this script keeps act/scene headings and stage directions in the output,
but only the spoken turns are sent through the reverse-translation pipeline.

Run from repo root:
    python training/full_play_translator.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ACT_RE = re.compile(r"^ACT\s+\d+\b")
SCENE_RE = re.compile(r"^Scene\s+\d+\b")
SEPARATOR_RE = re.compile(r"^=+$")
SPEAKER_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9' -]*[A-Z0-9]|[A-Z])"
    r"(?:,\s*\[[^\]]+\])?(?:\s{2,}(.+?))?\s*$"
)
WHITESPACE_RE = re.compile(r"\s+")
INLINE_STAGE_DIRECTION_RE = re.compile(r"\[[^\]]+\]")
DISPLAY_SPEAKER_ALIASES = {
    "HAMLET": "Hamlet",
    "KING": "Claudius",
    "QUEEN": "Gertrude",
}
SPEAKER_KEY_ALIASES = {
    "CLAUDIUS": "KING",
    "KING CLAUDIUS": "KING",
    "GERTRUDE": "QUEEN",
    "QUEEN GERTRUDE": "QUEEN",
}


@dataclass(slots=True)
class PlayBlock:
    kind: str
    text: str
    speaker: str | None = None
    act: str | None = None
    scene: str | None = None
    source_line: int | None = None


@dataclass(slots=True)
class SpeechTurn:
    speaker_raw: str
    speaker_display: str
    text: str
    act: str | None
    scene: str | None
    source_line: int
    index: int


def resolve_repo_root() -> Path:
    search_roots = [Path(__file__).resolve().parent, Path.cwd()]
    for start in search_roots:
        for candidate in (start, *start.parents):
            if (candidate / "data" / "hamlet_full_play.txt").is_file():
                return candidate

    raise FileNotFoundError(
        "Could not find repository root. Run from repo root or training/."
    )


def _resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _clean_non_speech_text(text: str) -> str:
    cleaned = text.replace("\u2014", "-").replace("\u2019", "'")
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def _clean_speech_text(lines: list[str]) -> str:
    combined = " ".join(line.strip() for line in lines if line.strip())
    combined = INLINE_STAGE_DIRECTION_RE.sub(" ", combined)
    return _clean_non_speech_text(combined)


def speaker_key(name: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", name.strip().upper())
    return SPEAKER_KEY_ALIASES.get(normalized, normalized)


def format_speaker_name(raw_name: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", raw_name.strip().upper())
    if normalized in DISPLAY_SPEAKER_ALIASES:
        return DISPLAY_SPEAKER_ALIASES[normalized]
    return normalized.title()


def _looks_like_speaker_label(candidate: str) -> bool:
    normalized = WHITESPACE_RE.sub(" ", candidate.strip())
    if not normalized or normalized.endswith(":"):
        return False
    if ACT_RE.match(normalized) or SCENE_RE.match(normalized):
        return False
    return normalized.upper() == normalized


def parse_full_play_blocks(lines: Iterable[str]) -> list[PlayBlock]:
    blocks: list[PlayBlock] = []
    current_act: str | None = None
    current_scene: str | None = None
    saw_first_act = False

    current_speaker: str | None = None
    current_speech_lines: list[str] = []
    current_speech_start_line: int | None = None

    in_stage_direction = False
    stage_lines: list[str] = []
    stage_start_line: int | None = None

    def flush_speech() -> None:
        nonlocal current_speaker, current_speech_lines, current_speech_start_line
        if current_speaker is None:
            return

        text = _clean_speech_text(current_speech_lines)
        if text:
            blocks.append(
                PlayBlock(
                    kind="speech",
                    text=text,
                    speaker=current_speaker,
                    act=current_act,
                    scene=current_scene,
                    source_line=current_speech_start_line,
                )
            )

        current_speaker = None
        current_speech_lines = []
        current_speech_start_line = None

    def flush_stage() -> None:
        nonlocal stage_lines, stage_start_line, in_stage_direction
        if not stage_lines:
            return

        blocks.append(
            PlayBlock(
                kind="stage",
                text=_clean_non_speech_text(" ".join(stage_lines)),
                act=current_act,
                scene=current_scene,
                source_line=stage_start_line,
            )
        )
        stage_lines = []
        stage_start_line = None
        in_stage_direction = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if in_stage_direction:
            if stripped:
                stage_lines.append(stripped)
            if "]" in stripped:
                flush_stage()
            continue

        if not stripped:
            flush_speech()
            continue

        if ACT_RE.match(stripped):
            flush_speech()
            current_act = stripped
            current_scene = None
            saw_first_act = True
            blocks.append(
                PlayBlock(
                    kind="heading",
                    text=stripped,
                    act=current_act,
                    source_line=line_number,
                )
            )
            continue

        if SCENE_RE.match(stripped):
            flush_speech()
            current_scene = stripped
            blocks.append(
                PlayBlock(
                    kind="heading",
                    text=stripped,
                    act=current_act,
                    scene=current_scene,
                    source_line=line_number,
                )
            )
            continue

        if SEPARATOR_RE.match(stripped):
            flush_speech()
            blocks.append(
                PlayBlock(
                    kind="separator",
                    text=stripped,
                    act=current_act,
                    scene=current_scene,
                    source_line=line_number,
                )
            )
            continue

        if stripped.startswith("["):
            flush_speech()
            stage_lines = [stripped]
            stage_start_line = line_number
            if "]" in stripped:
                flush_stage()
            else:
                in_stage_direction = True
            continue

        if not saw_first_act:
            blocks.append(
                PlayBlock(kind="header", text=_clean_non_speech_text(stripped), source_line=line_number)
            )
            continue

        speaker_match = SPEAKER_LINE_RE.match(line)
        if speaker_match and _looks_like_speaker_label(speaker_match.group(1)):
            flush_speech()
            current_speaker = WHITESPACE_RE.sub(" ", speaker_match.group(1).strip())
            current_speech_start_line = line_number
            inline_text = speaker_match.group(2)
            current_speech_lines = [inline_text] if inline_text else []
            continue

        if current_speaker is not None:
            current_speech_lines.append(stripped)
            continue

        blocks.append(
            PlayBlock(
                kind="header",
                text=_clean_non_speech_text(stripped),
                act=current_act,
                scene=current_scene,
                source_line=line_number,
            )
        )

    flush_speech()
    flush_stage()
    return blocks


def extract_speech_turns(blocks: Iterable[PlayBlock]) -> list[SpeechTurn]:
    turns: list[SpeechTurn] = []
    for index, block in enumerate(blocks):
        if block.kind != "speech" or block.speaker is None or block.source_line is None:
            continue

        turns.append(
            SpeechTurn(
                speaker_raw=block.speaker,
                speaker_display=format_speaker_name(block.speaker),
                text=block.text,
                act=block.act,
                scene=block.scene,
                source_line=block.source_line,
                index=index,
            )
        )
    return turns


def _truncate_blocks_by_speech_count(blocks: list[PlayBlock], limit: int) -> list[PlayBlock]:
    if limit <= 0:
        return blocks

    selected: list[PlayBlock] = []
    speech_count = 0
    for block in blocks:
        if block.kind == "speech":
            if speech_count >= limit:
                break
            speech_count += 1
        selected.append(block)

    return selected


def render_play_blocks(blocks: Iterable[PlayBlock], translated_speeches: list[str]) -> str:
    translated_iter = iter(translated_speeches)
    rendered_chunks: list[str] = []

    for block in blocks:
        if block.kind == "speech":
            translated_text = next(translated_iter)
            speaker = format_speaker_name(block.speaker or "")
            rendered_chunks.append(f"{speaker}: {translated_text}".strip())
            continue

        rendered_chunks.append(block.text)

    body = "\n\n".join(chunk for chunk in rendered_chunks if chunk)
    return body + ("\n" if body else "")


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate the Folger full-play Hamlet text into a plain-text file "
            "while preserving play structure."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "data" / "hamlet_full_play.txt"),
        help="Path to the full-play Hamlet text file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(repo_root / "data" / "hamlet_full_play_plain_english.txt"),
        help="Path to the translated output text file.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of spoken turns to translate. Use 0 for the full play.",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = resolve_repo_root()
    args = parse_args(repo_root)

    if args.limit < 0:
        raise ValueError("--limit must be 0 or greater.")

    input_path = _resolve_repo_relative_path(repo_root, args.input_file)
    output_path = _resolve_repo_relative_path(repo_root, args.output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_lines = input_path.read_text(encoding=args.encoding).splitlines()
    all_blocks = parse_full_play_blocks(raw_lines)
    selected_blocks = _truncate_blocks_by_speech_count(all_blocks, args.limit)
    speech_turns = extract_speech_turns(selected_blocks)
    if not speech_turns:
        raise ValueError("No spoken turns were extracted from the full play file.")

    import lora_4 as reverse_training
    import raw_dialouge_translator as raw_translator

    normalized_turns = [
        raw_translator.normalize_hamlet_irregular_words(turn.text) for turn in speech_turns
    ]

    print(f"Repository root: {repo_root}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Spoken turns selected: {len(speech_turns)}")

    reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer = (
        reverse_training.load_reverse_translator()
    )
    try:
        translated_turns = reverse_training.translate_speeches_with_reverse_model(
            normalized_turns,
            reverse_model,
            reverse_inp_tokenizer,
            reverse_tar_tokenizer,
        )
    finally:
        del reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer
        reverse_training.release_reverse_translator()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_play_blocks(selected_blocks, translated_turns),
        encoding=args.encoding,
    )

    print(f"Wrote translated turns: {len(translated_turns)}")
    print(f"Saved translated play to: {output_path}")
    print("Sample original:", speech_turns[0].text[:200] + "...")
    print("Sample translated:", translated_turns[0][:200] + "...")


if __name__ == "__main__":
    main()
