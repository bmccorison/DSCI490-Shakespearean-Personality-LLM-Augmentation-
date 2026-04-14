"""
Standalone raw Hamlet dialogue translator.

Note: this script now applies a small Hamlet-only regex normalization pass
before translation so clipped spellings from `data/hamlet_onlyhamletraw.txt`
are expanded in one place.

Run from repo root:
    python training/raw_dialouge_translator.py

Optional Flags:
    --input-file: Path to the raw Hamlet text file. Default: data/hamlet_onlyhamletraw.txt
    --output-file: Path to the translated output text file.@7 Default: data/hamlet_plain_english.txt
    --limit

This script:
1. Reads the raw Hamlet text file.
2. Extracts Hamlet's speeches.
3. Translates them to modern English with the reverse translator pipeline.
4. Writes one translated speech per paragraph to an output text file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent
TRANSLATIONS_DIR = TRAINING_DIR / "translations"

if str(TRANSLATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATIONS_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(1, str(TRAINING_DIR))

import lora_3 as base_training
import lora_4 as reverse_training


HAMLET_REGEX_RULES: list[tuple[str, str]] = [
    (r"(?<![A-Za-z])God-a-mercy(?![A-Za-z])", "God have mercy"),
    (r"(?<![A-Za-z])God\s+b'\s+wi'\s+ye(?![A-Za-z])", "God be with you"),
    (r"(?<![A-Za-z])by'r(?![A-Za-z])", "by our"),
    (r"(?<![A-Za-z])'swounds(?![A-Za-z])", "God's wounds"),
    (r"(?<![A-Za-z])within\s+'s\s+two\s+hours(?![A-Za-z])", "within these two hours"),
    (r"(?<![A-Za-z])an\s+thou'lt(?![A-Za-z])", "if you will"),
    (r"(?<![A-Za-z])thou'lt(?![A-Za-z])", "you will"),
    (r"(?<![A-Za-z])thou't(?![A-Za-z])", "you will"),
    (r"(?<![A-Za-z])woo't(?![A-Za-z])", "would you"),
    (r"(?<![A-Za-z])th'art(?![A-Za-z])", "you are"),
    (r"(?<![A-Za-z])th's(?![A-Za-z])", "this"),
    (r"(?<![A-Za-z])thyself(?![A-Za-z])", "yourself"),
    (r"(?<![A-Za-z])in's(?![A-Za-z])", "in his"),
    (r"(?<![A-Za-z])for's(?![A-Za-z])", "for his"),
    (r"(?<![A-Za-z])didst(?![A-Za-z])", "did"),
    (r"(?<![A-Za-z])ha't(?![A-Za-z])", "have it"),
    (r"(?<![A-Za-z])do't(?![A-Za-z])", "do it"),
    (r"(?<![A-Za-z])know't(?![A-Za-z])", "know it"),
    (r"(?<![A-Za-z])see't(?![A-Za-z])", "see it"),
    (r"(?<![A-Za-z])gave't(?![A-Za-z])", "gave it"),
    (r"(?<![A-Za-z])pardon't(?![A-Za-z])", "pardon it"),
    (r"(?<![A-Za-z])sworn't(?![A-Za-z])", "sworn it"),
    (r"(?<![A-Za-z])if't(?![A-Za-z])", "if it"),
    (r"(?<![A-Za-z])is't(?![A-Za-z])", "is it"),
    (r"(?<![A-Za-z])was't(?![A-Za-z])", "was it"),
    (r"(?<![A-Za-z])on't(?![A-Za-z])", "on it"),
    (r"(?<![A-Za-z])in't(?![A-Za-z])", "in it"),
    (r"(?<![A-Za-z])to't(?![A-Za-z])", "to it"),
    (r"(?<![A-Za-z])upon't(?![A-Za-z])", "upon it"),
    (r"(?<![A-Za-z])'gainst(?![A-Za-z])", "against"),
    (r"(?<![A-Za-z])'tween(?![A-Za-z])", "between"),
    (r"(?<![A-Za-z])'twixt(?![A-Za-z])", "between"),
    (r"(?<![A-Za-z])'twere(?![A-Za-z])", "it were"),
    (r"(?<![A-Za-z])e'er(?![A-Za-z])", "ever"),
    (r"(?<![A-Za-z])soe'er(?![A-Za-z])", "soever"),
    (r"(?<![A-Za-z])neer(?![A-Za-z])", "never"),
    (r"(?<![A-Za-z])sith(?![A-Za-z])", "since"),
    (r"(?<![A-Za-z])all's(?![A-Za-z])", "all is"),
    (r"(?<![A-Za-z])blest(?![A-Za-z])", "blessed"),
    (r"(?<![A-Za-z])seest(?![A-Za-z])", "see"),
    (r"(?<![A-Za-z])liest(?![A-Za-z])", "lie"),
    (r"(?<![A-Za-z])thinks't(?![A-Za-z])", "think"),
    (r"(?<![A-Za-z])ta'en(?![A-Za-z])", "taken"),
    (r"(?<![A-Za-z])murther'd(?![A-Za-z])", "murdered"),
    (r"(?<![A-Za-z])murtherer(?![A-Za-z])", "murderer"),
    (r"(?<![A-Za-z])murther(?![A-Za-z])", "murder"),
    (r"(?<![A-Za-z])murd'rous(?![A-Za-z])", "murderous"),
    (r"(?<![A-Za-z])coz'nage(?![A-Za-z])", "deception"),
    (r"(?<![A-Za-z])e'il(?![A-Za-z])", "evil"),
]
HAMLET_SHORT_FORM_RULES: list[tuple[str, str]] = [
    (r"(?<![A-Za-z])i'\s+th'(?![A-Za-z])", "in the"),
    (r"(?<![A-Za-z])i'(?![A-Za-z])", "in"),
    (r"(?<![A-Za-z])wi'(?![A-Za-z])", "with"),
    (r"(?<![A-Za-z])t'(?![A-Za-z])", "to"),
    (r"(?<![A-Za-z])th'(?![A-Za-z])", "the"),
    (r"(?<![A-Za-z])o'(?![A-Za-z])", "of"),
    (r"(?<![A-Za-z])'em(?![A-Za-z])", "them"),
]
ED_ELISION_OVERRIDES = {
    "impon": "imposed",
    "klll": "killed",
    "murther": "murdered",
    "op": "opened",
    "wann": "waned",
}
SECOND_PERSON_ELISION_OVERRIDES = {
    "com": "come",
    "liv": "live",
}


def _resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate Hamlet's raw dialogue into modern English using the "
            "reverse translator pipeline."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "data" / "hamlet_onlyhamletraw.txt"),
        help="Path to the raw Hamlet text file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(repo_root / "data" / "hamlet_plain_english.txt"),
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
        help="Optional number of Hamlet speeches to translate. Use 0 for all speeches.",
    )
    return parser.parse_args()


def render_translated_speeches(speeches: list[str]) -> str:
    body = "\n\n".join(speech.strip() for speech in speeches if speech.strip())
    return body + ("\n" if body else "")


def _match_case(source_text: str, replacement: str) -> str:
    letters = [char for char in source_text if char.isalpha()]
    if len(letters) > 1 and all(char.isupper() for char in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _case_aware_regex_sub(text: str, pattern: str, replacement: str) -> str:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    return compiled.sub(
        lambda match: _match_case(match.group(0), replacement),
        text,
    )


def _expand_ed_elision(match: re.Match[str]) -> str:
    stem = match.group(1)
    replacement = ED_ELISION_OVERRIDES.get(stem.lower(), f"{stem}ed")
    return _match_case(match.group(0), replacement)


def _expand_second_person_elision(match: re.Match[str]) -> str:
    stem = match.group(1)
    replacement = SECOND_PERSON_ELISION_OVERRIDES.get(stem.lower(), stem)
    return _match_case(match.group(0), replacement)


def _expand_er_elision(match: re.Match[str]) -> str:
    stem = match.group(1)
    suffix = match.group(2)
    return _match_case(match.group(0), f"{stem}er{suffix}")


def _expand_eth_verb(match: re.Match[str]) -> str:
    stem = match.group(1)
    suffix = "es" if stem.lower().endswith(("s", "sh", "ch", "x", "z", "o")) else "s"
    return _match_case(match.group(0), f"{stem}{suffix}")


# Note: keep the Hamlet-only regex cleanup here so the raw corpus's clipped
# spellings are normalized in one dedicated place before translation.
def normalize_hamlet_irregular_words(text: str) -> str:
    normalized = base_training._clean_text(text)

    for pattern, replacement in HAMLET_REGEX_RULES:
        normalized = _case_aware_regex_sub(normalized, pattern, replacement)

    normalized = _case_aware_regex_sub(
        normalized,
        r"(?<![A-Za-z])o'ermaster't(?![A-Za-z])",
        "overmaster it",
    )
    normalized = re.sub(
        r"(?<![A-Za-z])o'er([A-Za-z]+)(?![A-Za-z])",
        lambda match: _match_case(match.group(0), f"over{match.group(1)}"),
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"(?<![A-Za-z])([A-Za-z]+)'r(ing|ed)(?![A-Za-z])",
        _expand_er_elision,
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![A-Za-z])([A-Za-z]+)'d(?![A-Za-z])",
        _expand_ed_elision,
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![A-Za-z])([A-Za-z]+)'st(?![A-Za-z])",
        _expand_second_person_elision,
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![A-Za-z])([A-Za-z]+)eth(?![A-Za-z])",
        _expand_eth_verb,
        normalized,
        flags=re.IGNORECASE,
    )

    for pattern, replacement in HAMLET_SHORT_FORM_RULES:
        normalized = _case_aware_regex_sub(normalized, pattern, replacement)

    normalized = base_training.shakespeare_to_plain_english(normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return base_training.WHITESPACE_RE.sub(" ", normalized).strip()


def main() -> None:
    repo_root = base_training.find_repo_root()
    args = parse_args(repo_root)

    if args.limit < 0:
        raise ValueError("--limit must be 0 or greater.")

    input_path = _resolve_repo_relative_path(repo_root, args.input_file)
    output_path = _resolve_repo_relative_path(repo_root, args.output_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_lines = input_path.read_text(encoding=args.encoding).splitlines()
    hamlet_speeches = base_training.extract_hamlet_speeches(raw_lines)
    if not hamlet_speeches:
        raise ValueError("No Hamlet speeches were extracted from the input file.")

    if args.limit:
        hamlet_speeches = hamlet_speeches[: args.limit]

    normalized_hamlet_speeches = [
        normalize_hamlet_irregular_words(speech) for speech in hamlet_speeches
    ]

    print(f"Repository root: {repo_root}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Hamlet speeches selected: {len(hamlet_speeches)}")

    reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer = (
        reverse_training.load_reverse_translator()
    )
    try:
        translated_speeches = reverse_training.translate_speeches_with_reverse_model(
            normalized_hamlet_speeches,
            reverse_model,
            reverse_inp_tokenizer,
            reverse_tar_tokenizer,
        )
    finally:
        del reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer
        reverse_training.release_reverse_translator()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_translated_speeches(translated_speeches),
        encoding=args.encoding,
    )

    print(f"Wrote translated speeches: {len(translated_speeches)}")
    print(f"Saved translated dialogue to: {output_path}")
    print("Sample original:", hamlet_speeches[0][:200] + "...")
    print("Sample translated:", translated_speeches[0][:200] + "...")


if __name__ == "__main__":
    main()
