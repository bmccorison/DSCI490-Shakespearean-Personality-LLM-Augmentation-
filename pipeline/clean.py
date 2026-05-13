''' Data cleaning utilities. '''

import re
import json
import copy
from pathlib import Path


_ARTIFACT_RE = re.compile(
    r'\*?'
    r'</?'
    r'[ \t]*'
    r'(?:'
      r'\|[\w\-]*'
      r'(?:\|[>})]?|[>})]|(?=[\s<]|$))'
    r'|think\b[^<>]*>?'
    r')'
)

# **Note:** meta-commentary blocks injected by some model versions after a stop token.
# These appear as a self-annotation paragraph and are never in-character content.
_META_NOTE_RE = re.compile(r'\s*\*\*Note:\*\*.*$', re.DOTALL)


def clean_content(text: str) -> str:
    """
    Remove leaked chat-template tokens from assistant content while
    preserving the text on both sides of each token.

    Each token is replaced with a single space so the segment before and after
    it is joined rather than truncated.  This prevents content loss when the
    model injects a role-separator mid-generation and then continues writing.

    Args:
        text: Raw content string from an assistant message.

    Returns:
        Cleaned string with tokens removed but surrounding text remains.
    """
    if not isinstance(text, str):
        return text

    # Replace each token with a space — preserves text on both sides
    text = _ARTIFACT_RE.sub(' ', text)

    # Strip meta-commentary appended after a stop token by some model versions
    text = _META_NOTE_RE.sub('', text)

    # Collapse runs of spaces and trim trailing whitespace / orphaned asterisks
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[\s\*]+$', '', text)

    return text.strip()


def is_garbled(
    text: str,
    min_word_ratio: float = 0.35,
    max_internal_punct_ratio: float = 0.25,
) -> bool:
    """
    Detect garbled / broken-tokenization output.

    Two checks are applied:

    1. **Alpha-ratio**: real prose has >= `min_word_ratio` of its characters
       inside alphabetic runs of 2+.  Very sparse output fails here.

    2. **Internal-punctuation ratio**: garbled tokenizer output embeds
       punctuation directly before lowercase letters with no space
       (e.g. `I,s`  `.ry`  `?ing`  `come.iss`).  The ratio of such
       matches to 4+-letter word count is compared against
       `max_internal_punct_ratio`.  This catches garbled text that still
       contains enough real words to pass the alpha-ratio check alone.

    Args:
        text: Content string to evaluate.
        min_word_ratio: Minimum fraction of chars inside 2+ alpha runs.
        max_internal_punct_ratio: Max allowed ratio of internal-punct
            hits to 4+-letter word count.

    Returns:
        True if the text appears garbled.
    """
    if not text or len(text) < 20:
        return False

    # Primary: low alphabetic content
    alpha_chars = sum(len(w) for w in re.findall(r'[a-zA-Z]{2,}', text))
    if (alpha_chars / len(text)) < min_word_ratio:
        return True

    # Secondary: punctuation embedded directly before a lowercase letter
    # (no space between).  Normal prose and contractions don't trigger this.
    internal_punct = len(re.findall(r'[,\.!?;][a-z]', text))
    real_words = max(len(re.findall(r'[a-zA-Z]{4,}', text)), 1)
    if (internal_punct / real_words) > max_internal_punct_ratio:
        return True

    return False

''' Batch processing utility functions '''
def clean_conversation(data: "dict | list") -> "dict | list":
    """
    Clean all assistant content fields in a logged conversation.

    Handles both log formats produced by the pipeline:
      - Single Model and Multimodel

    Only assistant messages are cleaned; user messages are left as-is.
    Garbled responses are flagged with ``garbled: True`` rather than dropped
    so downstream steps can decide how to handle them.

    Args:
        data: Parsed JSON from a log file (list or dict).  Modified in-place.

    Returns:
        The same structure with cleaned content fields.
    """
    messages = data if isinstance(data, list) else data.get("messages", [])

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            # Multimodel config turn carries metadata, not plain text — skip
            continue

        if msg.get("role") != "assistant":
            continue

        cleaned = clean_content(content)

        # Evaluate the cleaned text; fall back to original if cleaning emptied it
        if is_garbled(cleaned if cleaned else content):
            msg["garbled"] = True

        msg["content"] = cleaned

def load_and_clean_file(path: Path) -> "dict | list | None":
    """
    Load a single JSON log file and return the cleaned conversation.
    Returns None on parse / IO errors.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ERROR {path.name}: {exc}")
        return None
    return clean_conversation(data)


def batch_clean(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "**/*.json",
) -> dict:
    """
    Clean all JSON log files under `input_dir` and write results to
    `output_dir`, preserving subdirectory structure.

    Args:
        input_dir:    Root directory containing raw log files.
        output_dir:   Destination root for cleaned output files.
        file_pattern: Glob pattern relative to `input_dir`.

    Returns:
        Summary dict: {processed, skipped, garbled_messages}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"processed": 0, "skipped": 0, "garbled_messages": 0}

    for src in sorted(input_dir.glob(file_pattern)):
        rel    = src.relative_to(input_dir)
        dst    = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        result = load_and_clean_file(src)
        if result is None:
            counts["skipped"] += 1
            continue

        msgs       = result if isinstance(result, list) else result.get("messages", [])
        n_garbled  = sum(1 for m in msgs if m.get("garbled"))
        counts["garbled_messages"] += n_garbled

        with open(dst, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        flag = f"  [{n_garbled} garbled]" if n_garbled else ""
        print(f"  {rel}{flag}")
        counts["processed"] += 1

    print(f"\nprocessed={counts['processed']}  skipped={counts['skipped']}  garbled_messages={counts['garbled_messages']}")
    return counts