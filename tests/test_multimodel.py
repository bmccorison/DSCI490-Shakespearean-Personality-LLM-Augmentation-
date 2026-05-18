'''Tests for pipeline/multimodel.py.'''

import json

import pytest

from pipeline.local_logging import LOG_CATEGORY_ENV_VAR
from pipeline.multimodel import (
    DEFAULT_CONTEXT_TURNS,
    DEFAULT_MAX_TURNS,
    HARD_MAX_TURNS,
    MAX_PARTICIPANTS,
    MIN_PARTICIPANTS,
    MULTIMODEL_LOG_CATEGORY,
    MultiModelConversation,
    MultiModelParticipant,
    MultiModelTurn,
    validate_max_turns,
)
from tests.conftest import (
    FakeTokenizer,
    fake_loader,
    make_conversation,
    make_participant,
    make_response_generator,
)


# ---------------------------------------------------------------------------
# validate_max_turns
# ---------------------------------------------------------------------------

class TestValidateMaxTurns:
    def test_accepts_boundary_values(self):
        assert validate_max_turns(1) == 1
        assert validate_max_turns(HARD_MAX_TURNS) == HARD_MAX_TURNS

    def test_rejects_zero_and_below(self):
        with pytest.raises(ValueError):
            validate_max_turns(0)
        with pytest.raises(ValueError):
            validate_max_turns(-5)

    def test_rejects_above_hard_cap(self):
        with pytest.raises(ValueError):
            validate_max_turns(HARD_MAX_TURNS + 1)

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError):
            validate_max_turns("lots")  # type: ignore[arg-type]

    def test_accepts_integer_strings(self):
        # int("10") works fine — the function casts with int()
        assert validate_max_turns("10") == 10  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MultiModelParticipant
# ---------------------------------------------------------------------------

class TestMultiModelParticipant:
    def test_valid_participant_stores_normalized_values(self):
        p = MultiModelParticipant(
            name="  Hamlet  ",
            character=" Hamlet ",
            work=" Hamlet ",
            model_name=" TinyLlama/TinyLlama-1.1B-Chat-v1.0 ",
            adapter_path=" __base__ ",
        )
        assert p.name == "Hamlet"
        assert p.character == "Hamlet"
        assert p.work == "Hamlet"
        assert p.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert p.adapter_path == "__base__"

    def test_raises_on_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            MultiModelParticipant(
                name="   ", character="Hamlet", work="Hamlet",
                model_name="model", adapter_path="__base__",
            )

    def test_raises_on_multiple_missing_fields(self):
        with pytest.raises(ValueError) as exc_info:
            MultiModelParticipant(
                name="", character="", work="Hamlet",
                model_name="model", adapter_path="__base__",
            )
        # Both missing fields should be named in the error message.
        message = str(exc_info.value)
        assert "name" in message
        assert "character" in message

    def test_to_dict_round_trips_all_fields(self):
        p = make_participant(1, model_name="ModelX", adapter_path="path/to/adapter")
        d = p.to_dict()
        assert d["name"] == p.name
        assert d["character"] == p.character
        assert d["work"] == p.work
        assert d["model_name"] == p.model_name
        assert d["adapter_path"] == p.adapter_path

    def test_frozen_dataclass_rejects_mutation(self):
        p = make_participant(1)
        with pytest.raises((AttributeError, TypeError)):
            p.name = "new name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MultiModelTurn
# ---------------------------------------------------------------------------

class TestMultiModelTurn:
    def test_to_dict_contains_all_fields(self):
        turn = MultiModelTurn(
            turn_number=3,
            speaker_index=1,
            speaker_name="Speaker2",
            character="Ophelia",
            content="There's rosemary, that's for remembrance.",
        )
        d = turn.to_dict()
        assert d["turn_number"] == 3
        assert d["speaker_index"] == 1
        assert d["speaker_name"] == "Speaker2"
        assert d["character"] == "Ophelia"
        assert d["content"] == "There's rosemary, that's for remembrance."


# ---------------------------------------------------------------------------
# MultiModelConversation — construction and validation
# ---------------------------------------------------------------------------

class TestMultiModelConversationInit:
    def test_rejects_single_participant(self):
        with pytest.raises(ValueError):
            MultiModelConversation([make_participant(1)], "Begin.")

    def test_rejects_too_many_participants(self):
        with pytest.raises(ValueError):
            MultiModelConversation(
                [make_participant(i) for i in range(1, MAX_PARTICIPANTS + 2)],
                "Begin.",
            )

    def test_rejects_duplicate_participant_names(self):
        p1 = MultiModelParticipant(
            name="Hamlet", character="Hamlet", work="Hamlet",
            model_name="model", adapter_path="__base__",
        )
        p2 = MultiModelParticipant(
            name="hamlet", character="Ophelia", work="Hamlet",
            model_name="model", adapter_path="__base__",
        )
        with pytest.raises(ValueError, match="unique"):
            MultiModelConversation([p1, p2], "Begin.")

    def test_rejects_empty_initial_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            MultiModelConversation([make_participant(1), make_participant(2)], "   ")

    def test_rejects_max_turns_above_hard_cap(self):
        with pytest.raises(ValueError):
            make_conversation(max_turns=HARD_MAX_TURNS + 1)

    def test_accepts_min_and_max_participant_counts(self):
        make_conversation(n_participants=MIN_PARTICIPANTS)
        make_conversation(n_participants=MAX_PARTICIPANTS)

    def test_stores_shakespeare_style_flag(self):
        conv = make_conversation(shakespeare_style=True)
        assert conv.shakespeare_style is True

    def test_session_id_is_unique_per_instance(self):
        a = make_conversation()
        b = make_conversation()
        assert a.session_id != b.session_id

    def test_initial_status_is_running(self):
        conv = make_conversation()
        assert conv.status == "running"
        assert not conv.is_complete


# ---------------------------------------------------------------------------
# Round-robin mechanics
# ---------------------------------------------------------------------------

class TestRoundRobin:
    def test_next_participant_index_cycles(self):
        conv = make_conversation(n_participants=3, max_turns=6)
        expected = [0, 1, 2, 0, 1, 2]
        for i, expected_index in enumerate(expected):
            assert conv.next_participant_index() == expected_index, f"turn {i}"
            conv.generate_next_turn(fake_loader, make_response_generator())

    def test_next_participant_returns_none_when_complete(self):
        conv = make_conversation(n_participants=2, max_turns=2)
        conv.generate_next_turn(fake_loader, make_response_generator())
        conv.generate_next_turn(fake_loader, make_response_generator())
        assert conv.next_participant_index() is None
        assert conv.next_participant() is None

    def test_speaker_names_follow_round_robin_order(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=3, max_turns=6)
        turns = [conv.generate_next_turn(fake_loader, gen) for _ in range(6)]
        names = [t.speaker_name for t in turns]
        assert names == ["Speaker1", "Speaker2", "Speaker3", "Speaker1", "Speaker2", "Speaker3"]

    def test_turn_numbers_are_sequential(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=4)
        turns = [conv.generate_next_turn(fake_loader, gen) for _ in range(4)]
        assert [t.turn_number for t in turns] == [1, 2, 3, 4]

    def test_status_becomes_complete_at_max_turns(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=2)
        conv.generate_next_turn(fake_loader, gen)
        assert conv.status == "running"
        conv.generate_next_turn(fake_loader, gen)
        assert conv.status == "complete"
        assert conv.is_complete

    def test_generate_next_turn_returns_none_when_already_complete(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=1)
        conv.generate_next_turn(fake_loader, gen)
        assert conv.generate_next_turn(fake_loader, gen) is None
        assert len(conv.turns) == 1


# ---------------------------------------------------------------------------
# Local logging
# ---------------------------------------------------------------------------

class TestMultiModelLogging:
    def test_writes_json_under_dated_multimodel_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pipeline.local_logging.DEFAULT_LOGGING_DIR", tmp_path)
        monkeypatch.delenv(LOG_CATEGORY_ENV_VAR, raising=False)

        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=1)
        conv.generate_next_turn(fake_loader, gen)

        assert conv._logger is not None
        log_file = conv._logger.log_file
        created_at = conv._logger.created_at
        assert log_file.parent == tmp_path / f"{created_at.month}_{created_at.day}" / MULTIMODEL_LOG_CATEGORY
        assert log_file.suffix == ".json"
        assert log_file.exists()

        stored_messages = json.loads(log_file.read_text(encoding="utf-8"))
        assert stored_messages[0]["initial_prompt"] == "Begin."
        assert stored_messages[1]["speaker_name"] == "Speaker1"
        assert stored_messages[1]["content"] == "reply 1"

    def test_log_category_override_preserves_multimodel_subdirectory(self, monkeypatch, tmp_path):
        monkeypatch.setattr("pipeline.local_logging.DEFAULT_LOGGING_DIR", tmp_path)
        monkeypatch.setenv(LOG_CATEGORY_ENV_VAR, "test")

        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=1)
        conv.generate_next_turn(fake_loader, gen)

        assert conv._logger is not None
        log_file = conv._logger.log_file
        assert log_file.parent.name == MULTIMODEL_LOG_CATEGORY
        assert log_file.parent.parent.name == "test"
        assert log_file.exists()


# ---------------------------------------------------------------------------
# Stop logic
# ---------------------------------------------------------------------------

class TestStopLogic:
    def test_stop_before_generation_returns_none(self):
        conv = make_conversation()
        conv.stop()

        def must_not_call(model_name, adapter_path):
            raise AssertionError("loader called after stop")

        assert conv.generate_next_turn(must_not_call) is None
        assert conv.status == "stopped"
        assert conv.is_complete

    def test_stop_during_generation_discards_response(self):
        conv = make_conversation()

        def stopping_response(tokenized, model, tokenizer, apply_shakespeare_style=True):
            conv.stop()
            return "discarded reply"

        result = conv.generate_next_turn(fake_loader, stopping_response)
        assert result is None
        assert conv.turns == []
        assert conv.status == "stopped"

    def test_stop_after_some_turns_preserves_existing_turns(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=4)
        conv.generate_next_turn(fake_loader, gen)
        conv.generate_next_turn(fake_loader, gen)
        conv.stop()
        assert len(conv.turns) == 2
        assert conv.status == "stopped"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_system_prompt_names_character_and_work(self):
        conv = make_conversation(n_participants=2)
        participant = conv.participants[0]
        prompt = conv._system_prompt(participant)
        assert participant.character in prompt
        assert participant.work in prompt

    def test_system_prompt_names_other_speakers(self):
        conv = make_conversation(n_participants=3)
        prompt = conv._system_prompt(conv.participants[0])
        assert "Speaker2" in prompt
        assert "Speaker3" in prompt
        assert "Speaker1" not in prompt

    def test_conversation_prompt_starts_with_initial_prompt(self):
        conv = MultiModelConversation(
            [make_participant(1), make_participant(2)],
            initial_prompt="What is honour?",
            max_turns=4,
        )
        prompt = conv._conversation_prompt(conv.participants[0])
        assert "What is honour?" in prompt

    def test_conversation_prompt_says_no_one_spoken_at_start(self):
        conv = make_conversation()
        prompt = conv._conversation_prompt(conv.participants[0])
        assert "No one has spoken yet" in prompt

    def test_conversation_prompt_includes_prior_turns(self):
        gen = make_response_generator("says")
        conv = make_conversation(n_participants=2, max_turns=4)
        conv.generate_next_turn(fake_loader, gen)
        prompt = conv._conversation_prompt(conv.participants[1])
        assert "Speaker1" in prompt
        assert "says 1" in prompt

    def test_context_window_truncates_old_turns(self):
        gen = make_response_generator()
        # context_turns=2 — only the two most recent turns appear in each prompt.
        conv = MultiModelConversation(
            [make_participant(1), make_participant(2)],
            initial_prompt="Begin.",
            max_turns=8,
            context_turns=2,
        )
        for _ in range(6):
            conv.generate_next_turn(fake_loader, gen)

        prompt = conv._conversation_prompt(conv.participants[0])
        # The first turn should no longer appear.
        assert "reply 1" not in prompt

    def test_build_prompt_returns_string_with_role_tags(self):
        conv = make_conversation()
        result = conv.build_prompt(conv.participants[0])
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result


# ---------------------------------------------------------------------------
# to_dict snapshot
# ---------------------------------------------------------------------------

class TestToDict:
    def test_idle_session_snapshot_shape(self):
        conv = make_conversation(n_participants=2, max_turns=4)
        d = conv.to_dict()
        assert d["active"] is True
        assert d["status"] == "running"
        assert d["is_stopped"] is False
        assert d["is_complete"] is False
        assert d["turn_count"] == 0
        assert d["turns"] == []
        assert d["last_turn"] is None
        assert len(d["participants"]) == 2
        assert d["next_speaker"] is not None

    def test_last_turn_is_included_when_passed(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=4)
        turn = conv.generate_next_turn(fake_loader, gen)
        d = conv.to_dict(last_turn=turn)
        assert d["last_turn"] is not None
        assert d["last_turn"]["turn_number"] == 1

    def test_next_speaker_is_none_when_complete(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=2)
        for _ in range(2):
            conv.generate_next_turn(fake_loader, gen)
        d = conv.to_dict()
        assert d["next_speaker"] is None
        assert d["is_complete"] is True

    def test_turn_dicts_accumulate_in_turns_list(self):
        gen = make_response_generator()
        conv = make_conversation(n_participants=2, max_turns=4)
        for _ in range(3):
            conv.generate_next_turn(fake_loader, gen)
        d = conv.to_dict()
        assert d["turn_count"] == 3
        assert len(d["turns"]) == 3
        for i, turn_dict in enumerate(d["turns"], start=1):
            assert turn_dict["turn_number"] == i


# ---------------------------------------------------------------------------
# Degenerate response fallback
# ---------------------------------------------------------------------------

class TestDegenerateResponseFallback:
    def test_empty_response_is_replaced_with_falls_silent(self):
        conv = make_conversation()

        def blank_response(tokenized, model, tokenizer, apply_shakespeare_style=True):
            return ""

        turn = conv.generate_next_turn(fake_loader, blank_response)
        assert turn is not None
        assert "falls silent" in turn.content
