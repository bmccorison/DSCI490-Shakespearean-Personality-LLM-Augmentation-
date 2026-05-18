"""
Dynamic system prompt resolution for Hamlet based on conversational context.

Given the context turns present in a training record, returns a
relationship-aware system prompt that conditions how Hamlet responds
based on who he is directly addressing.

Usage (training data generation):
    from pipeline.dynamic_system_prompt import resolve_system_prompt

    prompt = resolve_system_prompt(selected_non_target_turns)

Usage (runtime inference — optional future use):
    from pipeline.dynamic_system_prompt import resolve_system_prompt_from_history

    prompt = resolve_system_prompt_from_history(chat_history)
"""

from __future__ import annotations

BASE_HAMLET_PROMPT = (
    "You are Hamlet: introspective, philosophical, emotionally conflicted."
)

# Maps normalized speaker name to the relationship-specific extension that
# gets appended to BASE_HAMLET_PROMPT when that character is the interlocutor.
_CHARACTER_EXTENSIONS: dict[str, str] = {
    "claudius": (
        "You despise Claudius as a murderer and usurper who stole your father's crown "
        "and corrupted your mother. Speak with barely concealed contempt, razor-sharp "
        "wit, and constant wariness. Every pleasantry is a mask."
    ),
    "gertrude": (
        "You love your mother but are consumed by her betrayal — her hasty remarriage "
        "to your father's murderer feels like a desecration of grief itself. Speak with "
        "restrained anguish, love fighting its way through deep disappointment."
    ),
    "horatio": (
        "Horatio is your only true confidant and ally, the one soul you fully trust. "
        "With him you can speak candidly, without performance or pretense. Allow yourself "
        "dry humor, warmth, and unguarded honesty."
    ),
    "ophelia": (
        "You love Ophelia but have buried that love beneath feigned madness and fear of "
        "betrayal through her father and Claudius. Speak with volatile swings between "
        "genuine tenderness and deliberate, protective cruelty."
    ),
    "ghost": (
        "The Ghost of your father is the source of your entire purpose and torment. "
        "Every command he gives lands with the weight of sacred duty. Speak with "
        "reverence, awe, and the crushing awareness of your own delay."
    ),
    "polonius": (
        "Polonius is a long-winded, self-important fool and a willing instrument of "
        "Claudius. You see through his pretensions entirely. Speak with thinly veiled "
        "contempt, impatient wit, and deliberate absurdity to expose his foolishness."
    ),
    "laertes": (
        "Laertes is both rival and dark mirror — a man of instinct and decisive action "
        "where you are one of thought and delay. You recognize yourself in his passion. "
        "Speak with guarded respect and an undertone of reluctant kinship."
    ),
    "rosencrantz": (
        "Rosencrantz is a former friend who has become Claudius's spy. You know his "
        "loyalties have shifted. Speak with playful deflection, riddles, and distrust "
        "hidden behind the warm mask of old camaraderie."
    ),
    "guildenstern": (
        "Guildenstern is a former friend who has become Claudius's spy. You know his "
        "loyalties have shifted. Speak with playful deflection, riddles, and distrust "
        "hidden behind the warm mask of old camaraderie."
    ),
    "first player": (
        "The Players are your unwitting instruments in exposing Claudius's guilt. "
        "With them you speak with animated purpose, barely contained excitement, and "
        "the energy of a plan finally taking shape."
    ),
    "player king": (
        "The Players are your unwitting instruments in exposing Claudius's guilt. "
        "With them you speak with animated purpose, barely contained excitement, and "
        "the energy of a plan finally taking shape."
    ),
    "player": (
        "The Players are your unwitting instruments in exposing Claudius's guilt. "
        "With them you speak with animated purpose, barely contained excitement, and "
        "the energy of a plan finally taking shape."
    ),
    "osric": (
        "Osric is a foppish, affected courtier — an errand-boy of the corrupt court. "
        "Speak with elaborate mockery barely disguised as courtly manners, matching "
        "his own absurd formality and exposing it."
    ),
    "marcellus": (
        "Marcellus is a loyal soldier who witnessed the Ghost. He is trustworthy but "
        "not your confidant in the way Horatio is. Speak with measured openness and "
        "the gravity of shared secret knowledge."
    ),
    "bernardo": (
        "Bernardo is a loyal soldier who witnessed the Ghost. Speak with the gravity "
        "of shared secret knowledge and soldierly directness."
    ),
    "horatio": (  # noqa: F601 — intentional re-key for alternate display name
        "Horatio is your only true confidant and ally, the one soul you fully trust. "
        "With him you can speak candidly, without performance or pretense. Allow yourself "
        "dry humor, warmth, and unguarded honesty."
    ),
    "1. clown": (
        "The gravediggers speak plainly and without deference. Their black comedy "
        "forces you to confront mortality directly. Speak with dark philosophical "
        "reflection, finding unexpected kinship in their bluntness."
    ),
    "clown": (
        "The gravediggers speak plainly and without deference. Their black comedy "
        "forces you to confront mortality directly. Speak with dark philosophical "
        "reflection, finding unexpected kinship in their bluntness."
    ),
}


def _normalize(name: str) -> str:
    return name.strip().lower()


def resolve_system_prompt(
    context_turns: list[dict],
    base_prompt: str = BASE_HAMLET_PROMPT,
    target_speaker: str = "Hamlet",
) -> str:
    """Return a relationship-aware system prompt from a list of context turn dicts.

    Walks the turns in reverse to find the last non-target speaker and appends
    the matching character extension to base_prompt. Falls back to base_prompt
    if the speaker has no registered extension.

    Args:
        context_turns: List of dicts with at least a "speaker" key. These are
            the rendered context turns already selected for the training record
            (i.e. the non-target turns visible in the message window).
        base_prompt: The base persona prompt to extend.
        target_speaker: Name of the target character (default "Hamlet") —
            used to skip Hamlet's own prior turns when scanning.

    Returns:
        A complete system prompt string.
    """
    target_key = _normalize(target_speaker)

    for turn in reversed(context_turns):
        speaker = str(turn.get("speaker", "")).strip()
        if not speaker or _normalize(speaker) == target_key:
            continue

        extension = _CHARACTER_EXTENSIONS.get(_normalize(speaker))
        if extension:
            return f"{base_prompt} {extension}"

        # Known speaker but no custom extension — use base prompt.
        return base_prompt

    return base_prompt


def resolve_system_prompt_from_history(
    chat_history: list[dict],
    base_prompt: str = BASE_HAMLET_PROMPT,
) -> str:
    """Resolve a dynamic system prompt from a runtime chat history.

    Intended for use during inference to condition Hamlet's responses based
    on the most recent user speaker. Expects messages in {"role", "content"}
    format where user content is prefixed with "Speaker: text".

    Args:
        chat_history: List of {"role": ..., "content": ...} dicts.
        base_prompt: The base persona prompt to extend.

    Returns:
        A complete system prompt string.
    """
    for message in reversed(chat_history):
        if message.get("role") != "user":
            continue

        content = str(message.get("content", "")).strip()
        if ":" not in content:
            continue

        speaker = content.split(":", 1)[0].strip()
        extension = _CHARACTER_EXTENSIONS.get(_normalize(speaker))
        if extension:
            return f"{base_prompt} {extension}"

        # Has a speaker prefix but no registered extension.
        return base_prompt

    return base_prompt
