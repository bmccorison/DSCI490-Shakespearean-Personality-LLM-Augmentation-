'''Tests for app.py — FastAPI endpoints and model hot-swap state management.

Heavy model loading is stubbed out so tests run without GPU or checkpoints.
'''

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import app as app_module
from app import app
from pipeline import utils as utils_module
from tests.conftest import FakeTokenizer, make_response_generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_app_state():
    '''Reset all mutable app globals before each test.'''
    utils_module._resident_model_name = ""
    utils_module._resident_model = None
    utils_module._resident_tokenizer = None
    utils_module._resident_adapter_slots = {}
    utils_module._next_slot = 0
    app_module.selected_chat_model_name = ""
    app_module.selected_chat_adapter_path = ""
    app_module.active_multimodel_conversation = None
    app_module.multimodel_default_max_turns = app_module.DEFAULT_MULTIMODEL_MAX_TURNS
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _fake_model_and_tokenizer():
    '''Return sentinel objects that satisfy duck-typed checks in app code.'''
    fake_model = MagicMock()
    fake_model.set_adapter = MagicMock()
    fake_tokenizer = FakeTokenizer()
    return fake_model, fake_tokenizer


# ---------------------------------------------------------------------------
# _empty_multimodel_session
# ---------------------------------------------------------------------------

class TestEmptyMultimodelSession:
    def test_shape_matches_to_dict_convention(self):
        from pipeline.utils import empty_multimodel_session
        payload = empty_multimodel_session()
        assert payload["active"] is False
        assert payload["is_complete"] is True
        assert payload["is_stopped"] is False
        assert payload["turn_count"] == 0
        assert payload["turns"] == []
        assert payload["last_turn"] is None
        assert payload["next_speaker"] is None


# ---------------------------------------------------------------------------
# _ensure_loaded_model — hot-swap state
# ---------------------------------------------------------------------------

class TestEnsureLoadedModel:
    def _make_fake_base_loader(self):
        '''Patch load_base_model_and_tokenizer to return sentinels.'''
        model, tokenizer = _fake_model_and_tokenizer()
        return patch("pipeline.utils.load_base_model_and_tokenizer", return_value=(model, tokenizer))

    def test_loads_base_model_on_first_call(self):
        with self._make_fake_base_loader() as mock_load:
            m, t = utils_module.ensure_loaded_model("ModelA", "__base__")
        mock_load.assert_called_once_with("ModelA")
        assert utils_module._resident_model_name == "ModelA"

    def test_same_model_and_adapter_does_not_reload(self):
        # Use spec=object so hasattr(model, "set_adapter") is False, matching a real base model.
        bare_model = MagicMock(spec=object)
        fake_tokenizer = FakeTokenizer()
        with patch("pipeline.utils.load_base_model_and_tokenizer", return_value=(bare_model, fake_tokenizer)) as mock_load:
            utils_module.ensure_loaded_model("ModelA", "__base__")
            utils_module.ensure_loaded_model("ModelA", "__base__")
        # load_base_model_and_tokenizer should only fire once.
        assert mock_load.call_count == 1

    def test_different_model_triggers_full_reload(self):
        with self._make_fake_base_loader() as mock_load:
            utils_module.ensure_loaded_model("ModelA", "__base__")
            utils_module.ensure_loaded_model("ModelB", "__base__")
        assert mock_load.call_count == 2

    def test_hot_swaps_adapter_without_reloading_base(self):
        base_model, base_tokenizer = _fake_model_and_tokenizer()
        patched_adapter = MagicMock()
        patched_adapter.set_adapter = MagicMock()

        with patch("pipeline.utils.load_base_model_and_tokenizer", return_value=(base_model, base_tokenizer)) as mock_load, \
             patch("pipeline.utils.validate_and_resolve_adapter", return_value="/fake/path") as mock_validate, \
             patch("pipeline.utils.attach_named_adapter", return_value=patched_adapter) as mock_attach:

            utils_module.ensure_loaded_model("ModelA", "__base__")
            utils_module.ensure_loaded_model("ModelA", "adapters/lora1")
            utils_module.ensure_loaded_model("ModelA", "adapters/lora2")

        # Base model loaded once; two adapters attached.
        assert mock_load.call_count == 1
        assert mock_attach.call_count == 2

    def test_switching_between_loaded_adapters_uses_set_adapter(self):
        base_model, base_tokenizer = _fake_model_and_tokenizer()
        peft_model = MagicMock()
        peft_model.set_adapter = MagicMock()

        with patch("pipeline.utils.load_base_model_and_tokenizer", return_value=(base_model, base_tokenizer)), \
             patch("pipeline.utils.validate_and_resolve_adapter", return_value="/fake/path"), \
             patch("pipeline.utils.attach_named_adapter", return_value=peft_model):

            utils_module.ensure_loaded_model("ModelA", "adapters/lora1")
            utils_module.ensure_loaded_model("ModelA", "adapters/lora2")
            peft_model.set_adapter.reset_mock()

            # Switch back to lora1 — already in slot, should use set_adapter only.
            with patch("pipeline.utils.attach_named_adapter") as mock_attach_again:
                utils_module.ensure_loaded_model("ModelA", "adapters/lora1")
                mock_attach_again.assert_not_called()

        peft_model.set_adapter.assert_called_once()

    def test_raises_value_error_for_empty_model_name(self):
        with pytest.raises(ValueError):
            utils_module.ensure_loaded_model("", "__base__")

    def test_raises_value_error_for_empty_adapter_path(self):
        with pytest.raises(ValueError):
            utils_module.ensure_loaded_model("ModelA", "")


# ---------------------------------------------------------------------------
# /api/multimodel/config
# ---------------------------------------------------------------------------

class TestMultimodelConfig:
    def test_get_returns_default_config(self, client):
        from pipeline.multimodel import (
            DEFAULT_MAX_TURNS,
            HARD_MAX_TURNS,
            MIN_PARTICIPANTS,
            MAX_PARTICIPANTS,
        )
        resp = client.get("/api/multimodel/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["default_max_turns"] == DEFAULT_MAX_TURNS
        assert data["hard_max_turns"] == HARD_MAX_TURNS
        assert data["min_participants"] == MIN_PARTICIPANTS
        assert data["max_participants"] == MAX_PARTICIPANTS

    def test_post_updates_default_max_turns(self, client):
        resp = client.post("/api/multimodel/config", json={"max_turns": 8})
        assert resp.status_code == 200
        assert resp.json()["default_max_turns"] == 8
        assert app_module.multimodel_default_max_turns == 8

    def test_post_rejects_zero_max_turns(self, client):
        resp = client.post("/api/multimodel/config", json={"max_turns": 0})
        assert resp.status_code == 400

    def test_post_rejects_above_hard_cap(self, client):
        from pipeline.multimodel import HARD_MAX_TURNS
        resp = client.post("/api/multimodel/config", json={"max_turns": HARD_MAX_TURNS + 1})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/multimodel/start
# ---------------------------------------------------------------------------

CONFIGURED_MULTIMODEL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONFIGURED_MULTIMODEL_ADAPTER = "__base__"

VALID_PARTICIPANTS = [
    {
        "name": "A",
        "model_name": CONFIGURED_MULTIMODEL_MODEL,
        "adapter_path": CONFIGURED_MULTIMODEL_ADAPTER,
    },
    {
        "name": "B",
        "model_name": CONFIGURED_MULTIMODEL_MODEL,
        "adapter_path": CONFIGURED_MULTIMODEL_ADAPTER,
    },
]


class TestMultimodelStart:
    def test_creates_session_with_valid_payload(self, client):
        payload = {
            "initial_prompt": "Begin.",
            "participants": VALID_PARTICIPANTS,
        }
        resp = client.post("/api/multimodel/start", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["turn_count"] == 0
        assert len(data["participants"]) == 2
        assert data["participants"][0]["character"] == "Hamlet"
        assert data["participants"][1]["work"] == "Hamlet"

    def test_returns_400_for_unavailable_model(self, client):
        participants = [
            {**VALID_PARTICIPANTS[0], "model_name": "ModelA"},
            VALID_PARTICIPANTS[1],
        ]
        resp = client.post(
            "/api/multimodel/start",
            json={"initial_prompt": "Begin.", "participants": participants},
        )
        assert resp.status_code == 400
        assert "Model is not available" in resp.json()["detail"]

    def test_returns_400_for_unavailable_adapter(self, client):
        participants = [
            {**VALID_PARTICIPANTS[0], "adapter_path": "models/not_configured"},
            VALID_PARTICIPANTS[1],
        ]
        resp = client.post(
            "/api/multimodel/start",
            json={"initial_prompt": "Begin.", "participants": participants},
        )
        assert resp.status_code == 400
        assert "Adapter is not available" in resp.json()["detail"]

    def test_returns_400_for_single_participant(self, client):
        payload = {
            "initial_prompt": "Begin.",
            "participants": [VALID_PARTICIPANTS[0]],
        }
        resp = client.post("/api/multimodel/start", json=payload)
        assert resp.status_code == 400

    def test_returns_400_for_duplicate_participant_names(self, client):
        dupe = [VALID_PARTICIPANTS[0], {**VALID_PARTICIPANTS[0]}]
        resp = client.post("/api/multimodel/start", json={"initial_prompt": "Begin.", "participants": dupe})
        assert resp.status_code == 400

    def test_returns_400_for_empty_initial_prompt(self, client):
        payload = {"initial_prompt": "   ", "participants": VALID_PARTICIPANTS}
        resp = client.post("/api/multimodel/start", json=payload)
        assert resp.status_code == 400

    def test_custom_max_turns_is_respected(self, client):
        payload = {
            "initial_prompt": "Begin.",
            "participants": VALID_PARTICIPANTS,
            "max_turns": 5,
        }
        resp = client.post("/api/multimodel/start", json=payload)
        assert resp.status_code == 200
        assert resp.json()["max_turns"] == 5

    def test_max_turns_defaults_to_server_default_when_absent(self, client):
        payload = {"initial_prompt": "Begin.", "participants": VALID_PARTICIPANTS}
        resp = client.post("/api/multimodel/start", json=payload)
        assert resp.status_code == 200
        assert resp.json()["max_turns"] == app_module.DEFAULT_MULTIMODEL_MAX_TURNS

    def test_replaces_previous_session(self, client):
        payload = {"initial_prompt": "First.", "participants": VALID_PARTICIPANTS}
        client.post("/api/multimodel/start", json=payload)
        first_id = app_module.active_multimodel_conversation.session_id

        client.post("/api/multimodel/start", json={"initial_prompt": "Second.", "participants": VALID_PARTICIPANTS})
        second_id = app_module.active_multimodel_conversation.session_id

        assert first_id != second_id


# ---------------------------------------------------------------------------
# /api/multimodel/session
# ---------------------------------------------------------------------------

class TestMultimodelSession:
    def test_returns_idle_when_no_session(self, client):
        resp = client.get("/api/multimodel/session")
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_returns_active_session_after_start(self, client):
        client.post("/api/multimodel/start", json={
            "initial_prompt": "Begin.", "participants": VALID_PARTICIPANTS
        })
        resp = client.get("/api/multimodel/session")
        assert resp.json()["active"] is True


# ---------------------------------------------------------------------------
# /api/multimodel/next
# ---------------------------------------------------------------------------

class TestMultimodelNext:
    def test_returns_400_when_no_session_active(self, client):
        resp = client.post("/api/multimodel/next")
        assert resp.status_code == 400

    def test_generates_turn_with_mocked_model(self, client):
        client.post("/api/multimodel/start", json={
            "initial_prompt": "Begin.", "participants": VALID_PARTICIPANTS
        })
        fake_model, fake_tok = _fake_model_and_tokenizer()

        def _mock_ensure(model_name, adapter_path):
            return fake_model, fake_tok

        gen = make_response_generator("says")
        with patch.object(app_module.active_multimodel_conversation, "generate_next_turn",
                          wraps=lambda loader, resp_gen=None: \
                              app_module.active_multimodel_conversation.__class__.generate_next_turn(
                                  app_module.active_multimodel_conversation, _mock_ensure, gen
                              )):
            resp = client.post("/api/multimodel/next")

        assert resp.status_code == 200

    def test_returns_complete_state_when_already_done(self, client):
        client.post("/api/multimodel/start", json={
            "initial_prompt": "Begin.", "participants": VALID_PARTICIPANTS, "max_turns": 1
        })
        # Mark session complete via stop so /next returns immediately without model.
        app_module.active_multimodel_conversation.stop()
        resp = client.post("/api/multimodel/next")
        assert resp.status_code == 200
        assert resp.json()["is_complete"] is True


# ---------------------------------------------------------------------------
# /api/multimodel/stop
# ---------------------------------------------------------------------------

class TestMultimodelStop:
    def test_returns_idle_when_no_session(self, client):
        resp = client.post("/api/multimodel/stop")
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_stops_active_session(self, client):
        client.post("/api/multimodel/start", json={
            "initial_prompt": "Begin.", "participants": VALID_PARTICIPANTS
        })
        resp = client.post("/api/multimodel/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_stopped"] is True
        assert data["status"] == "stopped"


# ---------------------------------------------------------------------------
# /api/get_models
# ---------------------------------------------------------------------------

class TestGetModels:
    def test_returns_list(self, client):
        resp = client.get("/api/get_models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# /api/select_character
# ---------------------------------------------------------------------------

class TestSelectCharacter:
    def test_valid_character_returns_200(self, client):
        resp = client.get("/api/select_character", params={"character": "Hamlet", "work": "Hamlet"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["character"] == "Hamlet"
        assert data["work"] == "Hamlet"

    def test_empty_character_returns_400(self, client):
        resp = client.get("/api/select_character", params={"character": "", "work": "Hamlet"})
        assert resp.status_code == 400

    def test_empty_work_returns_400(self, client):
        resp = client.get("/api/select_character", params={"character": "Hamlet", "work": ""})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/refresh_chat
# ---------------------------------------------------------------------------

class TestRefreshChat:
    def test_returns_200_with_message(self, client):
        resp = client.get("/api/refresh_chat")
        assert resp.status_code == 200
        assert "message" in resp.json()


# ---------------------------------------------------------------------------
# /api/select_model
# ---------------------------------------------------------------------------

class TestSelectModel:
    def test_valid_model_and_adapter_returns_200(self, client):
        fake_model, fake_tok = _fake_model_and_tokenizer()
        with patch("app.ensure_loaded_model", return_value=(fake_model, fake_tok)):
            resp = client.get("/api/select_model", params={
                "model_name": "ModelA", "adapter_path": "__base__"
            })
        assert resp.status_code == 200
        assert resp.json()["model_name"] == "ModelA"

    def test_persists_selection_for_generate_endpoint(self, client):
        fake_model, fake_tok = _fake_model_and_tokenizer()
        with patch("app.ensure_loaded_model", return_value=(fake_model, fake_tok)):
            client.get("/api/select_model", params={
                "model_name": "ModelA", "adapter_path": "__base__"
            })
        assert app_module.selected_chat_model_name == "ModelA"

    def test_invalid_model_returns_400(self, client):
        with patch("app.ensure_loaded_model", side_effect=ValueError("not available")):
            resp = client.get("/api/select_model", params={
                "model_name": "Fake", "adapter_path": "__base__"
            })
        assert resp.status_code == 400

    def test_503_on_runtime_error(self, client):
        with patch("app.ensure_loaded_model", side_effect=RuntimeError("OOM")):
            resp = client.get("/api/select_model", params={
                "model_name": "ModelA", "adapter_path": "__base__"
            })
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_returns_400_when_no_model_selected(self, client):
        resp = client.get("/api/generate_response", params={"question": "Hello?"})
        assert resp.status_code == 400

    def test_returns_response_with_mocked_model(self, client):
        app_module.selected_chat_model_name = "ModelA"
        app_module.selected_chat_adapter_path = "__base__"

        fake_model, fake_tok = _fake_model_and_tokenizer()
        with patch("app.ensure_loaded_model", return_value=(fake_model, fake_tok)), \
             patch("app.generate_output", return_value="To be, or not to be."):
            resp = client.get("/api/generate_response", params={"question": "Hello?"})

        assert resp.status_code == 200
        assert resp.json()["response"] == "To be, or not to be."

    def test_passes_shakespeare_style_flag(self, client):
        app_module.selected_chat_model_name = "ModelA"
        app_module.selected_chat_adapter_path = "__base__"

        fake_model, fake_tok = _fake_model_and_tokenizer()
        with patch("app.ensure_loaded_model", return_value=(fake_model, fake_tok)), \
             patch("app.generate_output", return_value="reply") as mock_gen:
            client.get("/api/generate_response", params={
                "question": "Hello?", "shakespeare_style": "true"
            })

        mock_gen.assert_called_once()
        _, kwargs = mock_gen.call_args
        assert kwargs.get("apply_shakespeare_style") is True


# ---------------------------------------------------------------------------
# /api/voices
# ---------------------------------------------------------------------------

class TestListVoices:
    def test_returns_voice_options(self, client):
        resp = client.get("/api/voices")
        assert resp.status_code == 200
        payload = resp.json()
        assert "voices" in payload
        assert isinstance(payload["voices"], list)
        assert payload["voices"], "voice list should not be empty"
        first = payload["voices"][0]
        assert set(first.keys()) == {"name", "voice_id"}


# ---------------------------------------------------------------------------
# /api/tts — ElevenLabs synthesis endpoint
# ---------------------------------------------------------------------------

class TestTtsEndpoint:
    def test_returns_audio_bytes_with_media_type(self, client):
        with patch("app.generate_tts_audio",
                   return_value=(b"\x00\x01\x02", "audio/mpeg")) as mock_tts:
            resp = client.post("/api/tts", params={
                "text": "To be, or not to be.",
                "character": "Hamlet",
            })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/mpeg")
        assert resp.content == b"\x00\x01\x02"
        mock_tts.assert_called_once_with(
            "To be, or not to be.", character="Hamlet", voice=None,
        )

    def test_passes_voice_label_through(self, client):
        with patch("app.generate_tts_audio",
                   return_value=(b"data", "audio/mpeg")) as mock_tts:
            resp = client.post("/api/tts", params={
                "text": "Hi",
                "character": "Hamlet",
                "voice": "Will (Male)",
            })
        assert resp.status_code == 200
        mock_tts.assert_called_once_with(
            "Hi", character="Hamlet", voice="Will (Male)",
        )

    def test_defaults_character_to_hamlet(self, client):
        with patch("app.generate_tts_audio",
                   return_value=(b"data", "audio/mpeg")) as mock_tts:
            client.post("/api/tts", params={"text": "Hi"})
        _, kwargs = mock_tts.call_args
        assert kwargs["character"] == "Hamlet"

    def test_value_error_returns_400(self, client):
        with patch("app.generate_tts_audio", side_effect=ValueError("Text is required.")):
            resp = client.post("/api/tts", params={"text": " "})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Text is required."

    def test_runtime_error_returns_503(self, client):
        with patch("app.generate_tts_audio",
                   side_effect=RuntimeError("ELEVENLABS_API_KEY is not set.")):
            resp = client.post("/api/tts", params={"text": "Hi"})
        assert resp.status_code == 503
        assert "ELEVENLABS_API_KEY" in resp.json()["detail"]

    def test_propagates_non_mp3_media_type(self, client):
        with patch("app.generate_tts_audio",
                   return_value=(b"wavdata", "audio/wav")):
            resp = client.post("/api/tts", params={"text": "Hi"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")
