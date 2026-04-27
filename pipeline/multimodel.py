'''In-memory orchestration for model-to-model conversations.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

MIN_PARTICIPANTS = 2
MAX_PARTICIPANTS = 4
DEFAULT_MAX_TURNS = 12
HARD_MAX_TURNS = 20
DEFAULT_CONTEXT_TURNS = 8

ModelLoader = Callable[[str, str], tuple[Any, Any]]


@dataclass(frozen=True)
class MultiModelParticipant:
    '''Configuration for one speaker in a multimodel conversation.'''

    name: str
    character: str
    work: str
    model_name: str
    adapter_path: str

    def __post_init__(self) -> None:
        '''Normalize text fields and reject incomplete participant definitions.'''
        normalized_values = {
            "name": self.name.strip(),
            "character": self.character.strip(),
            "work": self.work.strip(),
            "model_name": self.model_name.strip(),
            "adapter_path": self.adapter_path.strip(),
        }

        missing_fields = [
            field_name
            for field_name, field_value in normalized_values.items()
            if not field_value
        ]
        if missing_fields:
            raise ValueError(
                "Participant is missing required field(s): "
                + ", ".join(missing_fields)
            )

        for field_name, field_value in normalized_values.items():
            object.__setattr__(self, field_name, field_value)

    def to_dict(self) -> dict[str, str]:
        '''Return a JSON-serializable participant payload for API responses.'''
        return {
            "name": self.name,
            "character": self.character,
            "work": self.work,
            "model_name": self.model_name,
            "adapter_path": self.adapter_path,
        }


@dataclass(frozen=True)
class MultiModelTurn:
    '''One generated turn from one multimodel participant.'''

    turn_number: int
    speaker_index: int
    speaker_name: str
    character: str
    content: str

    def to_dict(self) -> dict[str, object]:
        '''Return a JSON-serializable turn payload for API responses.'''
        return {
            "turn_number": self.turn_number,
            "speaker_index": self.speaker_index,
            "speaker_name": self.speaker_name,
            "character": self.character,
            "content": self.content,
        }


class MultiModelConversation:
    '''Coordinate a bounded round-robin conversation between configured models.'''

    def __init__(
        self,
        participants: list[MultiModelParticipant],
        initial_prompt: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        shakespeare_style: bool = False,
        context_turns: int = DEFAULT_CONTEXT_TURNS,
    ) -> None:
        self.session_id = uuid4().hex
        self.participants = self._validate_participants(participants)
        self.initial_prompt = initial_prompt.strip()
        if not self.initial_prompt:
            raise ValueError("Initial prompt is required.")

        self.max_turns = self._validate_max_turns(max_turns)
        self.shakespeare_style = bool(shakespeare_style)
        self.context_turns = max(1, int(context_turns))
        self.turns: list[MultiModelTurn] = []
        self.is_stopped = False

    @staticmethod
    def _validate_participants(
        participants: list[MultiModelParticipant],
    ) -> list[MultiModelParticipant]:
        '''Validate participant count and names before any model loading occurs.'''
        if not isinstance(participants, list):
            raise ValueError("Participants must be provided as a list.")
        if len(participants) < MIN_PARTICIPANTS or len(participants) > MAX_PARTICIPANTS:
            raise ValueError(
                f"Provide between {MIN_PARTICIPANTS} and {MAX_PARTICIPANTS} participants."
            )

        normalized_names = [participant.name.lower() for participant in participants]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("Participant names must be unique.")

        return participants

    @staticmethod
    def _validate_max_turns(max_turns: int) -> int:
        '''Keep automated conversations bounded to protect local compute.'''
        try:
            parsed_max_turns = int(max_turns)
        except (TypeError, ValueError) as exc:
            raise ValueError("Max turns must be an integer.") from exc

        if parsed_max_turns < 1 or parsed_max_turns > HARD_MAX_TURNS:
            raise ValueError(f"Max turns must be between 1 and {HARD_MAX_TURNS}.")
        return parsed_max_turns

    @property
    def status(self) -> str:
        '''Return the current high-level session state.'''
        if self.is_stopped:
            return "stopped"
        if len(self.turns) >= self.max_turns:
            return "complete"
        return "running"

    @property
    def is_complete(self) -> bool:
        '''Return whether no more turns should be generated.'''
        return self.is_stopped or len(self.turns) >= self.max_turns

    def stop(self) -> None:
        '''Mark the conversation as stopped before the next requested turn.'''
        self.is_stopped = True

    def next_participant_index(self) -> int | None:
        '''Return the round-robin speaker index for the next turn.'''
        if self.is_complete:
            return None
        return len(self.turns) % len(self.participants)

    def next_participant(self) -> MultiModelParticipant | None:
        '''Return the participant expected to speak next.'''
        participant_index = self.next_participant_index()
        if participant_index is None:
            return None
        return self.participants[participant_index]

    def build_prompt(self, participant: MultiModelParticipant) -> str:
        '''Build the full prompt for one participant's next line.'''
        prompt_messages = [
            {
                "role": "system",
                "content": self._system_prompt(participant),
            },
            {
                "role": "user",
                "content": self._conversation_prompt(participant),
            },
        ]
        return _render_prompt_messages(prompt_messages)

    def generate_next_turn(
        self,
        model_loader: ModelLoader,
        response_generator: Callable[..., str] | None = None,
    ) -> MultiModelTurn | None:
        '''Load the next speaker's model and append its generated response.'''
        participant_index = self.next_participant_index()
        if participant_index is None:
            return None

        if response_generator is None:
            # Import lazily so tests and non-generation imports do not require
            # heavyweight model dependencies such as torch at module import time.
            from pipeline.lm_generation import generate_response as response_generator

        participant = self.participants[participant_index]
        model, tokenizer = model_loader(participant.model_name, participant.adapter_path)
        prompt = self.build_prompt(participant)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")

        response_text = response_generator(
            tokenized_prompt,
            model,
            tokenizer,
            apply_shakespeare_style=self.shakespeare_style,
        ).strip()
        if self.is_stopped:
            # A stop request may arrive while a heavyweight generation call is
            # still running; do not append stale output after that point.
            return None
        if not response_text:
            response_text = f"{participant.character} falls silent."

        turn = MultiModelTurn(
            turn_number=len(self.turns) + 1,
            speaker_index=participant_index,
            speaker_name=participant.name,
            character=participant.character,
            content=response_text,
        )
        self.turns.append(turn)
        return turn

    def _system_prompt(self, participant: MultiModelParticipant) -> str:
        '''Return instructions that bind the current model to one speaker.'''
        other_speakers = [
            f"{other.name} as {other.character}"
            for other in self.participants
            if other.name != participant.name
        ]
        return (
            f"You are {participant.character}, a character from Shakespeare's "
            f"work {participant.work}. You are participating in a conversation "
            f"with {', '.join(other_speakers)}. Speak only as {participant.character}; "
            "do not write dialogue for any other speaker. Respond in first person, "
            "stay in character, and keep the exchange moving with one concise turn."
        )

    def _conversation_prompt(self, participant: MultiModelParticipant) -> str:
        '''Return the seed prompt plus recent transcript for the next speaker.'''
        recent_turns = self.turns[-self.context_turns :]
        if recent_turns:
            transcript = "\n".join(
                f"{turn.speaker_name} ({turn.character}): {turn.content}"
                for turn in recent_turns
            )
        else:
            transcript = "No one has spoken yet."

        participant_summary = "; ".join(
            f"{speaker.name} as {speaker.character} from {speaker.work}"
            for speaker in self.participants
        )
        return (
            f"Initial prompt: {self.initial_prompt}\n\n"
            f"Participants: {participant_summary}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            f"It is now {participant.name}'s turn. Respond only with "
            f"{participant.character}'s next line."
        )

    def to_dict(self, last_turn: MultiModelTurn | None = None) -> dict[str, object]:
        '''Return a JSON-serializable snapshot of the full session state.'''
        next_speaker = self.next_participant()
        return {
            "active": True,
            "session_id": self.session_id,
            "status": self.status,
            "is_stopped": self.is_stopped,
            "is_complete": self.is_complete,
            "initial_prompt": self.initial_prompt,
            "max_turns": self.max_turns,
            "hard_max_turns": HARD_MAX_TURNS,
            "turn_count": len(self.turns),
            "participants": [participant.to_dict() for participant in self.participants],
            "next_speaker": next_speaker.to_dict() if next_speaker is not None else None,
            "turns": [turn.to_dict() for turn in self.turns],
            "last_turn": last_turn.to_dict() if last_turn is not None else None,
        }


def _render_prompt_messages(prompt_messages: list[dict[str, str]]) -> str:
    '''Render prompts using the same role-tag style as the main chat pipeline.'''
    prompt_sections = []

    for message in prompt_messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        prompt_sections.append(f"<|{role}|>\n{content}</s>\n")

    prompt_sections.append("<|assistant|>\n")
    return "".join(prompt_sections)
