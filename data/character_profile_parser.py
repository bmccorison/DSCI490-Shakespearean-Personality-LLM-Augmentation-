from __future__ import annotations

"""Character profile text parser.

Usage
-----
1) Basic conversion (writes beside input as .json):
    python pipeline/character_profile_parser.py data/character_profile_hamlet.txt

2) Specify custom output file:
    python pipeline/character_profile_parser.py data/character_profile_hamlet.txt \
         -o data/hamlet_profile.json

What this script does
---------------------
- Reads a structured character profile .txt file.
- Extracts major sections (Background, Core Traits, Conflicts, Relationships, etc.).
- Normalizes wrapped lines and bullet blocks.
- Writes a structured JSON object suitable for downstream LLM/RAG workflows.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


SECTION_HEADINGS = [
    "Background",
    "Core Traits",
    "Key Internal Conflicts",
    "Relationships",
    "Psychological Arc",
    "Why Hamlet Endures",
    "Character Analysis:",
]


def _clean_whitespace(text: str) -> str:
    """Normalize spacing/newlines while preserving paragraph breaks."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _segment_sections(text: str) -> Tuple[str, Dict[str, str]]:
    """Split full document text into heading-indexed section blocks.

    Returns:
        (title, sections)
        - title: first non-empty line in the document.
        - sections: mapping of section heading -> cleaned section text.
    """
    lines = [line.rstrip() for line in text.splitlines()]

    title = ""
    for line in lines:
        if line.strip():
            title = line.strip()
            break

    positions: List[Tuple[str, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in SECTION_HEADINGS:
            positions.append((stripped, index))

    sections: Dict[str, str] = {}
    if not positions:
        sections["body"] = _clean_whitespace("\n".join(lines[1:]))
        return title, sections

    for i, (heading, start_idx) in enumerate(positions):
        content_start = start_idx + 1
        content_end = positions[i + 1][1] if i + 1 < len(positions) else len(lines)
        block = _clean_whitespace("\n".join(lines[content_start:content_end]))
        sections[heading] = block

    return title, sections


def _parse_named_blocks(section_text: str) -> List[Dict[str, str]]:
    """Parse sections shaped like:

    Heading Name
      one-or-more descriptive lines

    into [{"name": ..., "description": ...}, ...].
    """
    entries: List[Dict[str, str]] = []
    current_name = ""
    current_lines: List[str] = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_heading_like = bool(re.match(r"^[A-Z][A-Za-z0-9'\-.,&/ ]{2,60}$", line)) and not line.endswith(".")
        if is_heading_like:
            if current_name:
                entries.append(
                    {
                        "name": current_name,
                        "description": _clean_whitespace(" ".join(current_lines)),
                    }
                )
            current_name = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_name:
        entries.append(
            {
                "name": current_name,
                "description": _clean_whitespace(" ".join(current_lines)),
            }
        )

    return entries


def _parse_relationships(section_text: str) -> List[Dict[str, str]]:
    """Parse relationship lines in the format 'Character → Role'.

    Subsequent wrapped lines are attached to the relationship description.
    """
    relationships: List[Dict[str, str]] = []
    current_name = ""
    current_role = ""
    current_lines: List[str] = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "→" in line:
            if current_name:
                relationships.append(
                    {
                        "character": current_name,
                        "role": current_role,
                        "description": _clean_whitespace(" ".join(current_lines)),
                    }
                )
            left, right = [part.strip() for part in line.split("→", 1)]
            current_name = left
            current_role = right
            current_lines = []
        else:
            current_lines.append(line)

    if current_name:
        relationships.append(
            {
                "character": current_name,
                "role": current_role,
                "description": _clean_whitespace(" ".join(current_lines)),
            }
        )

    return relationships


def _extract_bullets(section_text: str) -> List[str]:
    """Extract bullet items and merge wrapped continuation lines."""
    bullets: List[str] = []
    current = ""

    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(("●", "○", "■", "- ")):
            if current:
                bullets.append(_clean_whitespace(current))
            current = re.sub(r"^[●○■\-]\s*", "", stripped)
            continue

        if current:
            current = f"{current} {stripped}"

    if current:
        bullets.append(_clean_whitespace(current))

    return bullets


def parse_character_profile(text: str, source_file: str = "") -> Dict[str, object]:
    """Convert raw character profile text into normalized JSON-ready dict."""
    title, sections = _segment_sections(text)

    character_name = ""
    match = re.match(r"^Character Profile:\s*(.+)$", title)
    if match:
        character_name = match.group(1).strip()

    core_traits = _parse_named_blocks(sections.get("Core Traits", ""))
    key_conflicts = _parse_named_blocks(sections.get("Key Internal Conflicts", ""))
    relationships = _parse_relationships(sections.get("Relationships", ""))

    analysis_text = sections.get("Character Analysis:", "")
    analysis_bullets = _extract_bullets(analysis_text)

    parsed = {
        "title": title,
        "character": character_name,
        "source_file": source_file,
        "background": sections.get("Background", ""),
        "core_traits": core_traits,
        "key_internal_conflicts": key_conflicts,
        "relationships": relationships,
        "psychological_arc": sections.get("Psychological Arc", ""),
        "why_hamlet_endures": sections.get("Why Hamlet Endures", ""),
        "character_analysis": {
            "summary": _clean_whitespace(re.sub(r"[●○■].*", "", analysis_text, flags=re.S)),
            "bullets": analysis_bullets,
        },
    }

    for key in SECTION_HEADINGS:
        if key not in sections:
            continue
        if key in {"Background", "Core Traits", "Key Internal Conflicts", "Relationships", "Psychological Arc", "Why Hamlet Endures", "Character Analysis:"}:
            continue
        parsed.setdefault("additional_sections", {})[key] = sections[key]

    return parsed


def main() -> None:
    """CLI entry point for text-to-JSON conversion."""
    parser = argparse.ArgumentParser(description="Parse character profile text into JSON.")
    parser.add_argument("input", type=Path, help="Path to input .txt character profile")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: same name as input with .json extension)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_suffix(".json")

    text = input_path.read_text(encoding="utf-8")
    parsed = parse_character_profile(text, source_file=str(input_path))

    output_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote parsed profile JSON to: {output_path}")


if __name__ == "__main__":
    main()
