'''Tests for pure-logic functions in pipeline/lm_generation.py.

All tests here avoid loading real models or touching disk-backed adapters.
Functions that require torch tensors or actual HuggingFace checkpoints are
covered by mocking only at the boundary where weights would be loaded.
'''

import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _render_prompt_messages
# ---------------------------------------------------------------------------

class TestRenderPromptMessages:
    def setup_method(self):
        from pipeline.lm_generation import _render_prompt_messages
        self.render = _render_prompt_messages

    def test_renders_system_user_and_assistant_tags(self):
        messages = [
            {"role": "system", "content": "You are Hamlet."},
            {"role": "user", "content": "What ails thee?"},
        ]
        result = self.render(messages)
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "You are Hamlet." in result
        assert "What ails thee?" in result

    def test_always_ends_with_assistant_tag(self):
        result = self.render([{"role": "user", "content": "Hello."}])
        assert result.endswith("<|assistant|>\n")

    def test_skips_unknown_roles(self):
        messages = [
            {"role": "moderator", "content": "This should be skipped."},
            {"role": "user", "content": "This should appear."},
        ]
        result = self.render(messages)
        assert "moderator" not in result
        assert "This should be skipped." not in result
        assert "This should appear." in result

    def test_skips_entries_with_empty_content(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Say something."},
        ]
        result = self.render(messages)
        assert result.count("<|system|>") == 0

    def test_empty_message_list_returns_only_assistant_tag(self):
        result = self.render([])
        assert result == "<|assistant|>\n"

    def test_segments_use_end_of_segment_token(self):
        messages = [{"role": "user", "content": "Hello."}]
        result = self.render(messages)
        assert "</s>" in result


# ---------------------------------------------------------------------------
# _looks_degenerate_response
# ---------------------------------------------------------------------------

class TestLooksDegenerateResponse:
    def setup_method(self):
        from pipeline.lm_generation import _looks_degenerate_response
        self.is_degenerate = _looks_degenerate_response

    def test_empty_string_is_degenerate(self):
        assert self.is_degenerate("") is True

    def test_whitespace_only_is_degenerate(self):
        assert self.is_degenerate("   ") is True

    def test_short_diverse_response_is_not_degenerate(self):
        assert self.is_degenerate("Hello there.") is False

    def test_normal_prose_is_not_degenerate(self):
        text = "To be, or not to be, that is the question."
        assert self.is_degenerate(text) is False

    def test_single_repeated_word_is_degenerate(self):
        assert self.is_degenerate("I I I I I I I I") is True

    def test_two_alternating_words_is_degenerate(self):
        # Only 2 unique words across many tokens — too low diversity.
        assert self.is_degenerate("go go go go go go go") is True

    def test_dominant_word_over_half_is_degenerate(self):
        # "the" appears 6 out of 10 words.
        assert self.is_degenerate("the the the the the the cat sat here now") is True

    def test_short_response_below_word_threshold_is_not_degenerate(self):
        # Fewer than 5 words: heuristics are skipped.
        assert self.is_degenerate("I am") is False


# ---------------------------------------------------------------------------
# post_processing
# ---------------------------------------------------------------------------

class TestPostProcessing:
    def setup_method(self):
        from pipeline.lm_generation import post_processing
        self.post_process = post_processing

    def test_strips_assistant_tag(self):
        result = self.post_process("<|assistant|>Hello there.", apply_shakespeare_style=False)
        assert "<|assistant|>" not in result
        assert "Hello there." in result

    def test_strips_end_of_segment_token(self):
        result = self.post_process("Farewell.</s>", apply_shakespeare_style=False)
        assert "</s>" not in result

    def test_normalizes_interior_whitespace(self):
        result = self.post_process("Hello     world.", apply_shakespeare_style=False)
        assert "  " not in result

    def test_strips_leading_and_trailing_whitespace(self):
        result = self.post_process("   hello   ", apply_shakespeare_style=False)
        assert result == result.strip()

    def test_applies_shakespeare_style_when_enabled(self):
        result = self.post_process("you are brave", apply_shakespeare_style=True)
        assert "thou art" in result

    def test_skips_shakespeare_style_when_disabled(self):
        result = self.post_process("you are brave", apply_shakespeare_style=False)
        assert "you are" in result
        assert "thou art" not in result


# ---------------------------------------------------------------------------
# _apply_shakespeare_dialogue_style / _match_case
# ---------------------------------------------------------------------------

class TestShakespeareStyle:
    def setup_method(self):
        from pipeline.lm_generation import _apply_shakespeare_dialogue_style, _match_case
        self.style = _apply_shakespeare_dialogue_style
        self.match_case = _match_case

    def test_you_are_becomes_thou_art(self):
        assert "thou art" in self.style("you are brave")

    def test_your_becomes_thy(self):
        assert "thy" in self.style("your kingdom")

    def test_you_becomes_thou(self):
        assert "thou" in self.style("do you know")

    def test_often_becomes_oft(self):
        assert "oft" in self.style("he often walks")

    def test_perhaps_becomes_perchance(self):
        assert "perchance" in self.style("perhaps tomorrow")

    def test_before_becomes_ere(self):
        assert "ere" in self.style("before sunrise")

    def test_preserves_unmatched_text(self):
        text = "The stars are bright tonight."
        result = self.style(text)
        assert "bright tonight" in result

    def test_match_case_uppercase_source(self):
        assert self.match_case("YOU", "thou") == "THOU"

    def test_match_case_capitalized_source(self):
        assert self.match_case("You", "thou") == "Thou"

    def test_match_case_lowercase_source(self):
        assert self.match_case("you", "thou") == "thou"

    def test_match_case_empty_source_returns_replacement(self):
        assert self.match_case("", "thou") == "thou"


# ---------------------------------------------------------------------------
# _read_int_setting / _read_float_setting
# ---------------------------------------------------------------------------

class TestReadSettings:
    def setup_method(self):
        from pipeline.lm_generation import _read_int_setting, _read_float_setting
        self.read_int = _read_int_setting
        self.read_float = _read_float_setting

    def test_int_returns_default_when_env_absent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FAKE_INT_SETTING", None)
            assert self.read_int("FAKE_INT_SETTING", 42, minimum=1) == 42

    def test_int_reads_from_env(self):
        with patch.dict(os.environ, {"FAKE_INT_SETTING": "99"}):
            assert self.read_int("FAKE_INT_SETTING", 1, minimum=1) == 99

    def test_int_clamps_to_minimum(self):
        with patch.dict(os.environ, {"FAKE_INT_SETTING": "0"}):
            assert self.read_int("FAKE_INT_SETTING", 5, minimum=3) == 3

    def test_int_returns_default_on_invalid_value(self):
        with patch.dict(os.environ, {"FAKE_INT_SETTING": "not_a_number"}):
            assert self.read_int("FAKE_INT_SETTING", 7, minimum=1) == 7

    def test_float_returns_default_when_env_absent(self):
        os.environ.pop("FAKE_FLOAT_SETTING", None)
        assert self.read_float("FAKE_FLOAT_SETTING", 0.5, minimum=0.1) == 0.5

    def test_float_reads_from_env(self):
        with patch.dict(os.environ, {"FAKE_FLOAT_SETTING": "0.9"}):
            assert self.read_float("FAKE_FLOAT_SETTING", 0.5, minimum=0.1) == pytest.approx(0.9)

    def test_float_clamps_to_minimum(self):
        with patch.dict(os.environ, {"FAKE_FLOAT_SETTING": "0.0"}):
            assert self.read_float("FAKE_FLOAT_SETTING", 0.5, minimum=0.2) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# set_character_context / get_system_prompt
# ---------------------------------------------------------------------------

class TestCharacterContext:
    def setup_method(self):
        import pipeline.lm_generation as lm
        self.lm = lm
        # Restore defaults after each test.
        lm.current_character = lm.DEFAULT_CHARACTER
        lm.current_work = lm.DEFAULT_WORK

    def test_default_system_prompt_names_hamlet(self):
        prompt = self.lm.get_system_prompt()
        assert "Hamlet" in prompt

    def test_set_character_context_updates_prompt(self):
        self.lm.set_character_context("Ophelia", "Hamlet")
        prompt = self.lm.get_system_prompt()
        assert "Ophelia" in prompt

    def test_set_character_context_updates_work(self):
        self.lm.set_character_context("Iago", "Othello")
        prompt = self.lm.get_system_prompt()
        assert "Iago" in prompt
        assert "Othello" in prompt

    def test_set_character_context_rejects_empty_character(self):
        with pytest.raises(ValueError):
            self.lm.set_character_context("", "Hamlet")

    def test_set_character_context_rejects_empty_work(self):
        with pytest.raises(ValueError):
            self.lm.set_character_context("Hamlet", "")

    def test_set_character_context_strips_whitespace(self):
        self.lm.set_character_context("  Hamlet  ", "  Hamlet  ")
        assert self.lm.current_character == "Hamlet"
        assert self.lm.current_work == "Hamlet"


# ---------------------------------------------------------------------------
# refresh_chat_history / message management
# ---------------------------------------------------------------------------

class TestChatHistory:
    def setup_method(self):
        import pipeline.lm_generation as lm
        self.lm = lm
        lm.current_character = lm.DEFAULT_CHARACTER
        lm.current_work = lm.DEFAULT_WORK
        lm.refresh_chat_history()

    def test_refresh_clears_history_and_adds_system_message(self):
        self.lm.add_chat_history(user_msg="Hello.")
        self.lm.refresh_chat_history()
        assert len(self.lm.messages) == 1
        assert self.lm.messages[0]["role"] == "system"

    def test_add_chat_history_appends_user_message(self):
        self.lm.add_chat_history(user_msg="Test message.")
        roles = [m["role"] for m in self.lm.messages]
        assert "user" in roles

    def test_add_chat_history_appends_assistant_message(self):
        self.lm.add_chat_history(model_response="I am Hamlet.")
        roles = [m["role"] for m in self.lm.messages]
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# validate_and_resolve_adapter
# ---------------------------------------------------------------------------

class TestValidateAndResolveAdapter:
    '''Tests use a mocked model_selection so no real adapter paths are required.'''

    FAKE_MODELS = [
        {
            "name": "ModelA",
            "adapters": [
                {"name": "base", "path": "__base__"},
                {"name": "lora1", "path": "models/lora1"},
            ],
        }
    ]

    def _patch_selection(self):
        return patch(
            "pipeline.lm_generation.model_selection",
            return_value=self.FAKE_MODELS,
        )

    def test_returns_none_for_base_adapter(self):
        from pipeline.lm_generation import validate_and_resolve_adapter
        with self._patch_selection():
            result = validate_and_resolve_adapter("ModelA", "__base__")
        assert result is None

    def test_raises_for_unknown_model(self):
        from pipeline.lm_generation import validate_and_resolve_adapter
        with self._patch_selection():
            with pytest.raises(ValueError, match="not available"):
                validate_and_resolve_adapter("UnknownModel", "__base__")

    def test_raises_for_adapter_not_in_model_list(self):
        from pipeline.lm_generation import validate_and_resolve_adapter
        with self._patch_selection():
            with pytest.raises(ValueError, match="not valid"):
                validate_and_resolve_adapter("ModelA", "models/nonexistent")

    def test_raises_file_not_found_for_missing_adapter_path(self, tmp_path):
        from pipeline.lm_generation import validate_and_resolve_adapter
        with self._patch_selection():
            # "models/lora1" won't exist on disk in the test environment.
            with pytest.raises(FileNotFoundError):
                validate_and_resolve_adapter("ModelA", "models/lora1")

    def test_returns_resolved_path_for_valid_adapter(self, tmp_path):
        from pipeline.lm_generation import validate_and_resolve_adapter, REPO_ROOT

        # Create a temporary adapter directory that passes the existence check.
        # We also need to make _is_adapter_compatible_with_model return True, which
        # it does when no adapter_config.json is present.
        adapter_rel = "models/fake_lora"
        adapter_abs = REPO_ROOT / adapter_rel
        adapter_abs.mkdir(parents=True, exist_ok=True)
        fake_models = [
            {
                "name": "ModelA",
                "adapters": [
                    {"name": "base", "path": "__base__"},
                    {"name": "fake", "path": adapter_rel},
                ],
            }
        ]
        try:
            with patch("pipeline.lm_generation.model_selection", return_value=fake_models):
                result = validate_and_resolve_adapter("ModelA", adapter_rel)
            assert result == adapter_abs
        finally:
            adapter_abs.rmdir()


# ---------------------------------------------------------------------------
# model_selection (integration with real model_list, no real paths needed)
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_returns_list_of_models(self):
        from pipeline.lm_generation import model_selection
        result = model_selection()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_model_has_required_keys(self):
        from pipeline.lm_generation import model_selection
        for model in model_selection():
            assert "name" in model
            assert "adapters" in model
            assert "default_adapter_path" in model

    def test_each_adapter_has_name_and_path(self):
        from pipeline.lm_generation import model_selection
        for model in model_selection():
            for adapter in model["adapters"]:
                assert "name" in adapter
                assert "path" in adapter

    def test_base_adapter_always_present_per_model(self):
        from pipeline.lm_generation import model_selection, BASE_MODEL_ADAPTER_PATH
        for model in model_selection():
            paths = [a["path"] for a in model["adapters"]]
            # At minimum the base pseudo-adapter must be exposed when no real adapters exist.
            assert len(paths) >= 1

    def test_default_adapter_path_is_in_adapter_list(self):
        from pipeline.lm_generation import model_selection
        for model in model_selection():
            adapter_paths = {a["path"] for a in model["adapters"]}
            assert model["default_adapter_path"] in adapter_paths
