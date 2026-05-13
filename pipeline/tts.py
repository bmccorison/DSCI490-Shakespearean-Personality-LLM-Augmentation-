''' TTS with ElevenLabs API '''

import importlib
import os
import threading


DEFAULT_VOICE_ID = "KjWPwHJWLungxeiYigoM"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_AUDIO_MIME = "audio/mpeg"
DEFAULT_VOICE_KEY = "default"

CHARACTER_VOICE_IDS: dict[str, str] = {
    "default": "JBFqnCBsd6RMkjVDRZzb",
    "Monika Sogam (Female)": "6aO1exAR9bDruq155LzQ",
    "William (Male)": "fjnwTZkKtQOJaYzGLa6n",
    "Cornelius (Male)": "6sFKzaJr574YWVu4UuJF",
    "Will (Male)": "KjWPwHJWLungxeiYigoM",
    "Darryl Lim (Male)": "O8ykjWKd0RjX6e5EyDuE",
}

_client = None
_client_lock = threading.Lock()


def _character_key(character: str) -> str:
    '''Normalize a character name for env-variable lookup keys.'''
    if not character:
        return "DEFAULT"
    return "".join(ch if ch.isalnum() else "_" for ch in character.strip().upper())


def get_voice_options() -> list[dict]:
    '''Return the configured voice options for the frontend selector.'''
    return [
        {"name": name, "voice_id": voice_id}
        for name, voice_id in CHARACTER_VOICE_IDS.items()
    ]


def _resolve_voice_id(character: str, voice: str | None = None) -> str:
    '''Resolve an ElevenLabs voice ID.

    Resolution order:
      1. Explicit voice label (key in CHARACTER_VOICE_IDS) or raw voice_id.
      2. Per-character env override (ELEVENLABS_VOICE_ID_<CHARACTER>).
      3. Character name matched as a key in CHARACTER_VOICE_IDS.
      4. The "default" entry in CHARACTER_VOICE_IDS.
      5. ELEVENLABS_VOICE_ID env / module default.
    '''
    if voice:
        normalized_voice = voice.strip()
        if normalized_voice in CHARACTER_VOICE_IDS:
            return CHARACTER_VOICE_IDS[normalized_voice]
        if normalized_voice in CHARACTER_VOICE_IDS.values():
            return normalized_voice

    override = os.getenv(f"ELEVENLABS_VOICE_ID_{_character_key(character)}")
    if override:
        return override

    normalized_character = character.strip() if character else ""
    if normalized_character and normalized_character in CHARACTER_VOICE_IDS:
        return CHARACTER_VOICE_IDS[normalized_character]

    if DEFAULT_VOICE_KEY in CHARACTER_VOICE_IDS:
        return CHARACTER_VOICE_IDS[DEFAULT_VOICE_KEY]

    return os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)


def _resolve_model_id() -> str:
    '''Resolve the ElevenLabs TTS model ID.'''
    return os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID)


def _resolve_output_format() -> str:
    '''Resolve the ElevenLabs output format.'''
    return os.getenv("ELEVENLABS_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)


def _output_format_mime(output_format: str) -> str:
    '''Map an ElevenLabs output_format identifier to an HTTP Content-Type.'''
    prefix = output_format.split("_", 1)[0].lower()
    if prefix == "mp3":
        return "audio/mpeg"
    if prefix in {"pcm", "wav"}:
        return "audio/wav"
    if prefix == "ulaw":
        return "audio/basic"
    if prefix == "opus":
        return "audio/ogg"
    return "application/octet-stream"


def _get_client():
    '''Return a cached ElevenLabs client, importing the SDK lazily.'''
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. Configure it in the environment to enable TTS."
            )

        try:
            client_module = importlib.import_module("elevenlabs.client")
        except Exception as exc:
            raise RuntimeError(
                "ElevenLabs SDK is unavailable. Install the 'elevenlabs' package to enable TTS."
            ) from exc

        _client = client_module.ElevenLabs(api_key=api_key)
        return _client


def generate_tts_audio(
    text: str,
    character: str = "Hamlet",
    voice: str | None = None,
) -> tuple[bytes, str]:
    '''Synthesize speech for the given text using ElevenLabs.

    The optional ``voice`` argument selects a specific entry from
    ``CHARACTER_VOICE_IDS`` (by label) or accepts a raw ElevenLabs voice_id.

    Returns ``(audio_bytes, media_type)`` suitable for an HTTP response.
    '''
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text is required.")

    client = _get_client()
    output_format = _resolve_output_format()

    try:
        audio_stream = client.text_to_speech.convert(
            text=normalized_text,
            voice_id=_resolve_voice_id(character, voice=voice),
            model_id=_resolve_model_id(),
            output_format=output_format,
        )
        audio_bytes = b"".join(audio_stream)
    except Exception as exc:
        raise RuntimeError(f"ElevenLabs TTS generation failed: {exc}") from exc

    if not audio_bytes:
        raise RuntimeError("ElevenLabs returned no audio bytes.")

    return audio_bytes, _output_format_mime(output_format)
