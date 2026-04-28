'''In-memory orchestration for model-to-model conversations.'''

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from pipeline.lm_generation import _render_prompt_messages
from pipeline.local_logging import LocalLogging

MULTIMODEL_LOG_CATEGORY = "multimodel"
MIN_PARTICIPANTS = 2
MAX_PARTICIPANTS = 4
DEFAULT_MAX_TURNS = 12
HARD_MAX_TURNS = 20
DEFAULT_CONTEXT_TURNS = 8

ModelLoader = Callable[[str, str], tuple[Any, Any]]

# Guards generate_next_turn against concurrent /next calls on the same session.
_generation_lock = threading.Lock()


def validate_max_turns(max_turns: int) -> int:
    '''Validate a requested turn count against the hard session limit.'''
    try:
        parsed = int(max_turns)
    except (TypeError, ValueError) as exc:
        raise ValueError("Max turns must be an integer.") from exc

    if parsed < 1 or parsed > HARD_MAX_TURNS:
        raise ValueError(f"Max turns must be between 1 and {HARD_MAX_TURNS}.")
    return parsed


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

        self.max_turns = validate_max_turns(max_turns)
        self.shakespeare_style = bool(shakespeare_style)
        self.context_turns = max(1, int(context_turns))
        self.turns: list[MultiModelTurn] = []
        self.is_stopped = False
        # Lazy: only create the on-disk logger once the first turn is generated, so
        # sessions that never advance leave no artifact behind.
        self._logger: LocalLogging | None = None

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
        '''Load the next speaker's model and append its generated response.

        Acquires a module-level lock so concurrent /next requests cannot double-step
        the round-robin counter or race on self.turns.
        '''
        with _generation_lock:
            participant_index = self.next_participant_index()
            if participant_index is None:
                return None

            if response_generator is None:
                # Imported lazily so the module can be loaded without torch being present.
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

            # A stop request may arrive while generation is running; discard stale output.
            if self.is_stopped:
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
            self._log_turn(turn, participant)
            return turn

    def _ensure_logger(self) -> LocalLogging:
        '''Create the multimodel logger on first use and seed it with session metadata.'''
        if self._logger is None:
            self._logger = LocalLogging(category=MULTIMODEL_LOG_CATEGORY)
            self._logger.append_message({
                "role": "user",
                "session_id": self.session_id,
                "initial_prompt": self.initial_prompt,
                "max_turns": self.max_turns,
                "shakespeare_style": self.shakespeare_style,
                "participants": [participant.to_dict() for participant in self.participants],
            })
        return self._logger

    def _log_turn(self, turn: MultiModelTurn, participant: MultiModelParticipant) -> None:
        '''Persist one generated turn to the multimodel log on disk.'''
        self._ensure_logger().append_message({
            "role": "assistant",
            "turn_number": turn.turn_number,
            "speaker_index": turn.speaker_index,
            "speaker_name": turn.speaker_name,
            "character": turn.character,
            "model_name": participant.model_name,
            "adapter_path": participant.adapter_path,
            "content": turn.content,
        })

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
