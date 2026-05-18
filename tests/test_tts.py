'''Tests for pipeline/tts.py — ElevenLabs voice resolution and audio synthesis.

The ElevenLabs SDK and network calls are stubbed; tests exercise pure logic
plus the boundary at which the SDK client is invoked.
'''

import sys
import pytest
from unittest.mock import MagicMock, patch

import pipeline.tts as tts_module
from pipeline.tts import (
    CHARACTER_VOICE_IDS,
    DEFAULT_AUDIO_MIME,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_VOICE_ID,
    DEFAULT_VOICE_KEY,
    _character_key,
    _output_format_mime,
    _resolve_model_id,
    _resolve_output_format,
    _resolve_voice_id,
    generate_tts_audio,
    get_voice_options,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_tts_state(monkeypatch):
    '''Clear the cached SDK client and strip env vars between tests.'''
    tts_module._client = None
    for var in [
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "ELEVENLABS_MODEL_ID",
        "ELEVENLABS_OUTPUT_FORMAT",
    ]:
        monkeypatch.delenv(var, raising=False)
    # Strip per-character overrides too.
    for key in list(__import__("os").environ.keys()):
        if key.startswith("ELEVENLABS_VOICE_ID_"):
            monkeypatch.delenv(key, raising=False)
    yield
    tts_module._client = None


def _install_fake_sdk(audio_chunks=(b"chunk1", b"chunk2"), raise_on_convert=None):
    '''Install a fake `elevenlabs.client` module that yields ``audio_chunks``.'''
    fake_module = MagicMock()
    fake_client = MagicMock()
    if raise_on_convert is not None:
        fake_client.text_to_speech.convert.side_effect = raise_on_convert
    else:
        fake_client.text_to_speech.convert.return_value = iter(audio_chunks)
    fake_module.ElevenLabs.return_value = fake_client
    sys.modules["elevenlabs.client"] = fake_module
    return fake_module, fake_client


# ---------------------------------------------------------------------------
# _character_key
# ---------------------------------------------------------------------------

class TestCharacterKey:
    def test_empty_string_returns_default(self):
        assert _character_key("") == "DEFAULT"

    def test_none_returns_default(self):
        assert _character_key(None) == "DEFAULT"

    def test_uppercases_and_replaces_non_alnum_with_underscore(self):
        assert _character_key("Lady Macbeth") == "LADY_MACBETH"
        assert _character_key("King Henry V") == "KING_HENRY_V"

    def test_strips_surrounding_whitespace(self):
        assert _character_key("  Hamlet  ") == "HAMLET"

    def test_non_alphanumeric_punctuation_becomes_underscore(self):
        assert _character_key("Mercutio-2!") == "MERCUTIO_2_"


# ---------------------------------------------------------------------------
# get_voice_options
# ---------------------------------------------------------------------------

class TestGetVoiceOptions:
    def test_returns_one_dict_per_configured_voice(self):
        options = get_voice_options()
        assert len(options) == len(CHARACTER_VOICE_IDS)

    def test_each_entry_has_name_and_voice_id(self):
        for entry in get_voice_options():
            assert set(entry.keys()) == {"name", "voice_id"}
            assert isinstance(entry["name"], str) and entry["name"]
            assert isinstance(entry["voice_id"], str) and entry["voice_id"]

    def test_includes_default_entry(self):
        names = {opt["name"] for opt in get_voice_options()}
        assert DEFAULT_VOICE_KEY in names


# ---------------------------------------------------------------------------
# _resolve_voice_id
# ---------------------------------------------------------------------------

class TestResolveVoiceId:
    def test_explicit_voice_label_wins(self):
        assert _resolve_voice_id("Hamlet", voice="Will (Male)") == CHARACTER_VOICE_IDS["Will (Male)"]

    def test_explicit_raw_voice_id_passes_through(self):
        raw = CHARACTER_VOICE_IDS["Cornelius (Male)"]
        assert _resolve_voice_id("Hamlet", voice=raw) == raw

    def test_per_character_env_override_used_when_no_explicit_voice(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_VOICE_ID_HAMLET", "custom-hamlet-voice")
        assert _resolve_voice_id("Hamlet") == "custom-hamlet-voice"

    def test_explicit_voice_overrides_env_override(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_VOICE_ID_HAMLET", "env-voice")
        assert _resolve_voice_id("Hamlet", voice="Will (Male)") == CHARACTER_VOICE_IDS["Will (Male)"]

    def test_character_name_matches_voice_key(self):
        # Use a character whose label literally matches a CHARACTER_VOICE_IDS key.
        label = "Will (Male)"
        assert _resolve_voice_id(label) == CHARACTER_VOICE_IDS[label]

    def test_falls_back_to_default_voice_key(self):
        assert _resolve_voice_id("Hamlet") == CHARACTER_VOICE_IDS[DEFAULT_VOICE_KEY]

    def test_falls_back_to_env_when_default_key_missing(self, monkeypatch):
        monkeypatch.setitem(CHARACTER_VOICE_IDS, "_saved", CHARACTER_VOICE_IDS.pop("default"))
        try:
            monkeypatch.setenv("ELEVENLABS_VOICE_ID", "fallback-voice")
            assert _resolve_voice_id("Hamlet") == "fallback-voice"
        finally:
            CHARACTER_VOICE_IDS["default"] = CHARACTER_VOICE_IDS.pop("_saved")

    def test_module_default_when_default_key_and_env_missing(self, monkeypatch):
        monkeypatch.setitem(CHARACTER_VOICE_IDS, "_saved", CHARACTER_VOICE_IDS.pop("default"))
        try:
            assert _resolve_voice_id("Unknown Character") == DEFAULT_VOICE_ID
        finally:
            CHARACTER_VOICE_IDS["default"] = CHARACTER_VOICE_IDS.pop("_saved")

    def test_whitespace_only_voice_does_not_match(self):
        # An empty/whitespace ``voice`` should be treated as not provided.
        assert _resolve_voice_id("Hamlet", voice="   ") == CHARACTER_VOICE_IDS[DEFAULT_VOICE_KEY]


# ---------------------------------------------------------------------------
# _resolve_model_id / _resolve_output_format
# ---------------------------------------------------------------------------

class TestResolveModelId:
    def test_default_when_no_env(self):
        assert _resolve_model_id() == DEFAULT_MODEL_ID

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_MODEL_ID", "eleven_custom_v9")
        assert _resolve_model_id() == "eleven_custom_v9"


class TestResolveOutputFormat:
    def test_default_when_no_env(self):
        assert _resolve_output_format() == DEFAULT_OUTPUT_FORMAT

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
        assert _resolve_output_format() == "pcm_16000"


# ---------------------------------------------------------------------------
# _output_format_mime
# ---------------------------------------------------------------------------

class TestOutputFormatMime:
    @pytest.mark.parametrize("fmt,expected", [
        ("mp3_44100_128", "audio/mpeg"),
        ("MP3_22050_32", "audio/mpeg"),
        ("pcm_16000", "audio/wav"),
        ("wav_44100", "audio/wav"),
        ("ulaw_8000", "audio/basic"),
        ("opus_48000_64", "audio/ogg"),
        ("flac_44100", "application/octet-stream"),
        ("", "application/octet-stream"),
    ])
    def test_known_and_unknown_prefixes(self, fmt, expected):
        assert _output_format_mime(fmt) == expected

    def test_default_format_maps_to_default_mime(self):
        assert _output_format_mime(DEFAULT_OUTPUT_FORMAT) == DEFAULT_AUDIO_MIME


# ---------------------------------------------------------------------------
# _get_client
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_raises_when_api_key_missing(self):
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY is not set"):
            tts_module._get_client()

    def test_raises_when_sdk_unavailable(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        with patch("pipeline.tts.importlib.import_module",
                   side_effect=ImportError("no elevenlabs")):
            with pytest.raises(RuntimeError, match="ElevenLabs SDK is unavailable"):
                tts_module._get_client()

    def test_returns_initialized_client(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        fake_module, fake_client = _install_fake_sdk()
        client = tts_module._get_client()
        fake_module.ElevenLabs.assert_called_once_with(api_key="key-123")
        assert client is fake_client

    def test_caches_client_across_calls(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        fake_module, _ = _install_fake_sdk()
        first = tts_module._get_client()
        second = tts_module._get_client()
        assert first is second
        # ElevenLabs constructor invoked exactly once despite two _get_client calls.
        fake_module.ElevenLabs.assert_called_once()


# ---------------------------------------------------------------------------
# generate_tts_audio
# ---------------------------------------------------------------------------

class TestGenerateTtsAudio:
    def test_raises_value_error_for_empty_text(self):
        with pytest.raises(ValueError, match="Text is required"):
            generate_tts_audio("")

    def test_raises_value_error_for_whitespace_only_text(self):
        with pytest.raises(ValueError, match="Text is required"):
            generate_tts_audio("   \n\t ")

    def test_returns_concatenated_audio_with_mime(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        _, fake_client = _install_fake_sdk(audio_chunks=(b"\x00\x01", b"\x02\x03"))

        audio, mime = generate_tts_audio("Hello", character="Hamlet")
        assert audio == b"\x00\x01\x02\x03"
        assert mime == DEFAULT_AUDIO_MIME
        fake_client.text_to_speech.convert.assert_called_once()
        kwargs = fake_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["text"] == "Hello"
        assert kwargs["model_id"] == DEFAULT_MODEL_ID
        assert kwargs["output_format"] == DEFAULT_OUTPUT_FORMAT

    def test_strips_whitespace_before_sending(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        _, fake_client = _install_fake_sdk()
        generate_tts_audio("  hello world  ")
        kwargs = fake_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["text"] == "hello world"

    def test_explicit_voice_label_routed_to_sdk(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        _, fake_client = _install_fake_sdk()
        generate_tts_audio("Hi", character="Hamlet", voice="Cornelius (Male)")
        kwargs = fake_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["voice_id"] == CHARACTER_VOICE_IDS["Cornelius (Male)"]

    def test_mime_reflects_output_format_env(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        monkeypatch.setenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
        _install_fake_sdk(audio_chunks=(b"abc",))
        _, mime = generate_tts_audio("Hi")
        assert mime == "audio/wav"

    def test_wraps_sdk_exception_as_runtime_error(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        _install_fake_sdk(raise_on_convert=Exception("boom"))
        with pytest.raises(RuntimeError, match="ElevenLabs TTS generation failed: boom"):
            generate_tts_audio("Hi")

    def test_raises_runtime_error_on_empty_audio(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "key-123")
        _install_fake_sdk(audio_chunks=())
        with pytest.raises(RuntimeError, match="returned no audio bytes"):
            generate_tts_audio("Hi")

    def test_missing_api_key_propagates_runtime_error(self):
        # No API key set; _get_client raises before SDK is reached.
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY is not set"):
            generate_tts_audio("Hi")