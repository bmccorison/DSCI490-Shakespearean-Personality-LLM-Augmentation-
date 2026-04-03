"""
Text preprocessing utilities for the Hamlet benchmarking pipeline.

Copied verbatim from training/lora_3.ipynb to keep preprocessing parallel
between training and evaluation. Do not modify these functions independently
of lora_3.ipynb — any drift will cause the benchmark to evaluate a different
text distribution than the model was trained on.
"""

from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9'_-]*)\.\s*(.*)$")
STAGE_DIRECTION_RE = re.compile(r"\[[^\]]+\]")
WHITESPACE_RE = re.compile(r"\s+")

MIN_WORDS_PER_SPEECH = 4

SYSTEM_PROMPT_ROLEPLAY = (
    "You are Hamlet, Prince of Denmark. Speak in clear modern English while "
    "preserving Hamlet's introspection, melancholy, philosophical wit, and "
    "moral tension. Stay in character."
)

TRANSLATION_RULES: list[tuple[str, str]] = [
    (r"(?<![A-Za-z])i\s+prithee(?![A-Za-z])", "please"),
    (r"(?<![A-Za-z])prithee(?![A-Za-z])", "please"),
    (r"(?<![A-Za-z])methinks(?![A-Za-z])", "I think"),
    (r"(?<![A-Za-z])wherefore(?![A-Za-z])", "why"),
    (r"(?<![A-Za-z])ere\s+yet(?![A-Za-z])", "before"),
    (r"(?<![A-Za-z])'tis(?![A-Za-z])", "it is"),
    (r"(?<![A-Za-z])'twas(?![A-Za-z])", "it was"),
    (r"(?<![A-Za-z])thou(?![A-Za-z])", "you"),
    (r"(?<![A-Za-z])thee(?![A-Za-z])", "you"),
    (r"(?<![A-Za-z])thy(?![A-Za-z])", "your"),
    (r"(?<![A-Za-z])thine(?![A-Za-z])", "yours"),
    (r"(?<![A-Za-z])art(?![A-Za-z])", "are"),
    (r"(?<![A-Za-z])dost(?![A-Za-z])", "do"),
    (r"(?<![A-Za-z])doth(?![A-Za-z])", "does"),
    (r"(?<![A-Za-z])hast(?![A-Za-z])", "have"),
    (r"(?<![A-Za-z])hath(?![A-Za-z])", "has"),
    (r"(?<![A-Za-z])wilt(?![A-Za-z])", "will"),
    (r"(?<![A-Za-z])shalt(?![A-Za-z])", "shall"),
    (r"(?<![A-Za-z])canst(?![A-Za-z])", "can"),
    (r"(?<![A-Za-z])couldst(?![A-Za-z])", "could"),
    (r"(?<![A-Za-z])wouldst(?![A-Za-z])", "would"),
    (r"(?<![A-Za-z])shouldst(?![A-Za-z])", "should"),
    (r"(?<![A-Za-z])mayst(?![A-Za-z])", "may"),
    (r"(?<![A-Za-z])ere(?![A-Za-z])", "before"),
    (r"(?<![A-Za-z])whilst(?![A-Za-z])", "while"),
    (r"(?<![A-Za-z])ne'er(?![A-Za-z])", "never"),
    (r"(?<![A-Za-z])o'er(?![A-Za-z])", "over"),
    (r"(?<![A-Za-z])e'en(?![A-Za-z])", "even"),
    (r"i'\s*th'", "in the"),
]

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize spacing while preserving Hamlet's wording."""
    text = STAGE_DIRECTION_RE.sub(" ", text)
    text = text.replace("\u2014", "-").replace("\u2019", "'")
    return WHITESPACE_RE.sub(" ", text).strip()


def _match_case(source_text: str, replacement: str) -> str:
    """Keep replacements readable when the source starts with a capital letter."""
    letters = [character for character in source_text if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def replace_case_aware(text: str, pattern: str, replacement: str) -> str:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    return compiled.sub(
        lambda match: _match_case(match.group(0), replacement),
        text,
    )


def shakespeare_to_plain_english(text: str) -> str:
    """Apply transparent heuristic rewrites to approximate plain English."""
    normalized = text
    for pattern, replacement in TRANSLATION_RULES:
        normalized = replace_case_aware(normalized, pattern, replacement)

    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized

# ---------------------------------------------------------------------------
# Speech extraction
# ---------------------------------------------------------------------------

def extract_hamlet_speeches(lines: Iterable[str]) -> list[str]:
    """Extract multi-line speeches that belong only to Hamlet."""
    speeches: list[str] = []
    current_speech: list[str] = []
    collecting_hamlet = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        speaker_match = SPEAKER_PREFIX_RE.match(line)

        if speaker_match:
            if collecting_hamlet and current_speech:
                speeches.append(_clean_text(" ".join(current_speech)))

            speaker = speaker_match.group(1).lower()
            first_text = _clean_text(speaker_match.group(2))
            collecting_hamlet = speaker.startswith("ham")
            current_speech = [first_text] if collecting_hamlet and first_text else []
            continue

        if collecting_hamlet:
            cleaned = _clean_text(line)
            if cleaned:
                current_speech.append(cleaned)

    if collecting_hamlet and current_speech:
        speeches.append(_clean_text(" ".join(current_speech)))

    return [s for s in speeches if len(s.split()) >= MIN_WORDS_PER_SPEECH]

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_roleplay_prompt(instruction: str) -> str:
    """Wrap an instruction in the TinyLlama chat template used during training."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT_ROLEPLAY}</s>\n"
        f"<|user|>\n{instruction}</s>\n"
        "<|assistant|>\n"
    )
