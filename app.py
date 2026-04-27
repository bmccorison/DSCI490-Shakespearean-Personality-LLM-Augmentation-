''' Handle fastapi endpoints for the front-end interface. '''

import gc
import importlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import scipy.io.wavfile as wav
import uvicorn

from pipeline.lm_generation import (
    generate_output,
    refresh_chat_history,
    model_selection,
    get_model,
    set_character_context,
)
from pipeline.multimodel import (
    DEFAULT_MAX_TURNS as DEFAULT_MULTIMODEL_MAX_TURNS,
    HARD_MAX_TURNS as HARD_MULTIMODEL_MAX_TURNS,
    MAX_PARTICIPANTS as MAX_MULTIMODEL_PARTICIPANTS,
    MIN_PARTICIPANTS as MIN_MULTIMODEL_PARTICIPANTS,
    MultiModelConversation,
    MultiModelParticipant,
)

# from pipeline.rag import get_context  # TODO Implement

BARK_HISTORY_PROMPT = os.getenv("BARK_HISTORY_PROMPT", "v2/en_speaker_6")
BARK_CHARACTER_PROMPTS = {
    "hamlet": "v2/en_speaker_9",
}
ESPEAK_DEFAULT_VOICE = "en-gb+m3"
ESPEAK_DEFAULT_SPEED = "145"
ESPEAK_DEFAULT_PITCH = "42"
ESPEAK_DEFAULT_AMPLITUDE = "165"
ESPEAK_CHARACTER_VOICES = {
    "hamlet": "en-gb+m3",
}

_bark_generate_audio = None
_bark_sample_rate = None
_bark_preload_models = None
_bark_models_preloaded = False
_bark_load_error = None

# TODO: Refactor the support multiple chat histories and characters
app = FastAPI()  # Initialize the FastAPI app
default_cors_origins = "http://localhost:6969,http://127.0.0.1:6969"
configured_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", default_cors_origins)
allowed_origins = [origin.strip() for origin in configured_cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # Placeholder for the LLM model, to be loaded in by user
tokenizer = None
loaded_model_name = ""
loaded_adapter_path = ""
selected_chat_model_name = ""
selected_chat_adapter_path = ""
active_multimodel_conversation: MultiModelConversation | None = None
multimodel_default_max_turns = DEFAULT_MULTIMODEL_MAX_TURNS


class MultiModelParticipantRequest(BaseModel):
    '''Request payload for one model-to-model speaker.'''

    name: str
    character: str
    work: str = "Hamlet"
    model_name: str
    adapter_path: str


class MultiModelStartRequest(BaseModel):
    '''Request payload used to create a new multimodel conversation.'''

    initial_prompt: str
    participants: list[MultiModelParticipantRequest]
    max_turns: int | None = None
    shakespeare_style: bool = False


class MultiModelConfigRequest(BaseModel):
    '''Request payload for updating multimodel defaults.'''

    max_turns: int


def _release_loaded_model() -> None:
    '''Release the currently loaded model/tokenizer and clear accelerator caches.'''
    global model, tokenizer, loaded_model_name, loaded_adapter_path

    previous_model = model
    previous_tokenizer = tokenizer
    model = None
    tokenizer = None
    loaded_model_name = ""
    loaded_adapter_path = ""

    if previous_model is None and previous_tokenizer is None:
        return

    del previous_model, previous_tokenizer
    gc.collect()

    try:
        import torch
    except Exception:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available") or not cuda.is_available():
        return

    cuda.empty_cache()
    if hasattr(cuda, "ipc_collect"):
        cuda.ipc_collect()


def _ensure_loaded_model(model_name: str, adapter_path: str):
    '''Load the requested model/adapter only when it is not already active.'''
    global model, tokenizer, loaded_model_name, loaded_adapter_path

    normalized_model_name = model_name.strip()
    normalized_adapter_path = adapter_path.strip()
    if not normalized_model_name or not normalized_adapter_path:
        raise ValueError("Model name and adapter path are required.")

    if (
        model is not None
        and tokenizer is not None
        and loaded_model_name == normalized_model_name
        and loaded_adapter_path == normalized_adapter_path
    ):
        return model, tokenizer

    # Multimodel generation swaps participants sequentially, so the app keeps
    # exactly one heavyweight model resident at a time.
    _release_loaded_model()
    model, tokenizer = get_model(normalized_model_name, normalized_adapter_path)
    loaded_model_name = normalized_model_name
    loaded_adapter_path = normalized_adapter_path
    return model, tokenizer


def _resolve_bark_use_gpu() -> bool:
    '''Return whether Bark should attempt GPU execution.'''
    configured_value = os.getenv("BARK_USE_GPU")
    if configured_value is not None:
        return configured_value.strip().lower() in {"1", "true", "yes", "on"}

    try:
        import torch
    except Exception:
        return False

    return bool(torch.cuda.is_available())


def _load_bark_dependencies():
    '''Import Bark lazily so the API can still start when TTS is unavailable.'''
    global _bark_generate_audio, _bark_sample_rate, _bark_preload_models, _bark_models_preloaded, _bark_load_error

    if _bark_generate_audio is not None and _bark_sample_rate is not None:
        _preload_bark_models_if_needed()
        return _bark_generate_audio, _bark_sample_rate

    if _bark_load_error is not None:
        raise RuntimeError(_bark_load_error)

    try:
        bark_module = importlib.import_module("bark")
    except Exception as exc:
        _bark_load_error = (
            "TTS backend is unavailable on this host. "
            "Bark could not be imported, likely because the installed torchaudio build "
            "requires CUDA libraries that are not present. "
            f"Original error: {exc}"
        )
        raise RuntimeError(_bark_load_error) from exc

    _bark_generate_audio = bark_module.generate_audio
    _bark_sample_rate = bark_module.SAMPLE_RATE
    _bark_preload_models = getattr(bark_module, "preload_models", None)
    _preload_bark_models_if_needed()
    return _bark_generate_audio, _bark_sample_rate


def _preload_bark_models_if_needed():
    '''Preload Bark models once so CPU/GPU selection is applied via Bark's supported API.'''
    global _bark_models_preloaded, _bark_load_error

    if _bark_preload_models is None or _bark_models_preloaded:
        return

    try:
        use_gpu = _resolve_bark_use_gpu()
        _bark_preload_models(
            text_use_gpu=use_gpu,
            coarse_use_gpu=use_gpu,
            fine_use_gpu=use_gpu,
            codec_use_gpu=use_gpu,
        )
        _bark_models_preloaded = True
    except Exception as exc:
        _bark_load_error = (
            "TTS backend is unavailable on this host. "
            "Bark models could not be prepared. "
            f"Original error: {exc}"
        )
        raise RuntimeError(_bark_load_error) from exc


def _resolve_bark_history_prompt(character: str) -> str:
    '''Resolve Bark speaker preset for the selected character.'''
    normalized_character = character.strip().lower()
    override_key = f"BARK_HISTORY_PROMPT_{normalized_character.upper()}"
    configured_override = os.getenv(override_key)
    if configured_override:
        return configured_override

    return BARK_CHARACTER_PROMPTS.get(normalized_character, BARK_HISTORY_PROMPT)


def _resolve_tts_fallback_binary() -> str | None:
    '''Resolve an offline espeak binary available on this host.'''
    for candidate in ("espeak-ng", "espeak"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _character_key(character: str) -> str:
    '''Normalize character names for env-variable lookup keys.'''
    if not character:
        return "DEFAULT"
    return "".join(next_char if next_char.isalnum() else "_" for next_char in character.strip().upper())


def _resolve_character_espeak_voice(character: str) -> str:
    '''Resolve an espeak voice per character, with env var overrides.'''
    character_lookup_key = _character_key(character)
    configured_voice = os.getenv(f"ESPEAK_VOICE_{character_lookup_key}")
    if configured_voice:
        return configured_voice

    default_voice = os.getenv("ESPEAK_VOICE", ESPEAK_DEFAULT_VOICE)
    mapped_voice = ESPEAK_CHARACTER_VOICES.get(character.strip().lower())
    if mapped_voice:
        return mapped_voice
    return default_voice


def _resolve_piper_model_path(character: str) -> Path | None:
    '''Resolve an optional Piper model path (character-specific override first).'''
    character_lookup_key = _character_key(character)
    configured_path = os.getenv(f"PIPER_MODEL_PATH_{character_lookup_key}") or os.getenv("PIPER_MODEL_PATH")
    if not configured_path:
        return None

    resolved_path = Path(configured_path).expanduser()
    if not resolved_path.exists():
        return None
    return resolved_path


def _generate_piper_tts_audio(text: str, character: str) -> bytes:
    '''Generate WAV audio with Piper when binary and voice model are available.'''
    piper_binary = shutil.which("piper")
    if piper_binary is None:
        raise RuntimeError("Piper is not installed on this host.")

    model_path = _resolve_piper_model_path(character)
    if model_path is None:
        raise RuntimeError("No Piper voice model is configured. Set PIPER_MODEL_PATH.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        output_path = Path(temp_file.name)

    command = [piper_binary, "--model", str(model_path), "--output_file", str(output_path)]
    configured_speaker_id = os.getenv("PIPER_SPEAKER_ID")
    if configured_speaker_id:
        command.extend(["--speaker", configured_speaker_id])

    try:
        completed = subprocess.run(
            command,
            input=text,
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(stderr or f"piper failed with exit code {completed.returncode}.")
        return output_path.read_bytes()
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def _generate_espeak_tts_audio(text: str, character: str) -> bytes:
    '''Generate WAV audio with tuned espeak settings for a less robotic fallback voice.'''
    tts_binary = _resolve_tts_fallback_binary()
    if tts_binary is None:
        raise RuntimeError("No CPU espeak fallback was found. Install espeak or espeak-ng.")

    voice = _resolve_character_espeak_voice(character)
    speed = os.getenv("ESPEAK_SPEED", ESPEAK_DEFAULT_SPEED)
    pitch = os.getenv("ESPEAK_PITCH", ESPEAK_DEFAULT_PITCH)
    amplitude = os.getenv("ESPEAK_AMPLITUDE", ESPEAK_DEFAULT_AMPLITUDE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        output_path = Path(temp_file.name)

    try:
        completed = subprocess.run(
            [
                tts_binary,
                "-w",
                str(output_path),
                "-v",
                voice,
                "-s",
                speed,
                "-p",
                pitch,
                "-a",
                amplitude,
                text,
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(stderr or f"{Path(tts_binary).name} failed with exit code {completed.returncode}.")

        return output_path.read_bytes()
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def _generate_fallback_tts_audio(text: str, character: str = "Hamlet") -> bytes:
    '''Generate WAV audio from available CPU-safe TTS backends.'''
    fallback_errors = []

    try:
        return _generate_piper_tts_audio(text, character)
    except RuntimeError as piper_error:
        fallback_errors.append(f"piper: {piper_error}")

    try:
        return _generate_espeak_tts_audio(text, character)
    except RuntimeError as espeak_error:
        fallback_errors.append(f"espeak: {espeak_error}")

    raise RuntimeError(" ; ".join(fallback_errors))


@app.get("/api/generate_response")
def generate_response_endpoint(question: str, shakespeare_style: bool = False):
    ''' Endpoint to trigger the response pipeline given a user question. '''
    global selected_chat_model_name, selected_chat_adapter_path

    if not selected_chat_model_name or not selected_chat_adapter_path:
        raise HTTPException(status_code=400, detail="Model is not loaded. Call /api/select_model first.")

    # TODO: wire in RAG once vector store/context plumbing is implemented.
    rag_context = None

    try:
        active_model, active_tokenizer = _ensure_loaded_model(
            selected_chat_model_name,
            selected_chat_adapter_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Generate the response JSON from the LLM {response: str, confidence_score: int}
    response_text = generate_output(
        question,
        active_tokenizer,
        active_model,
        rag_context,
        apply_shakespeare_style=shakespeare_style,
    )
    
    return {"response": response_text}


@app.get("/api/refresh_chat")
def refresh_chat():
    ''' Endpoint to trigger the reset of the conversation history. '''
    refresh_chat_history()
    return {"message": "Chat history refreshed."}


@app.get("/api/select_character")
def select_character(character: str, work: str):
    ''' Endpoint to select the character and work for the system prompt. '''
    try:
        set_character_context(character, work)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Character context updated.",
        "character": character.strip(),
        "work": work.strip(),
    }


@app.get("/api/select_model")
def select_model(model_name: str, adapter_path: str):
    ''' Endpoint to select the specific LLM for response generation. '''
    global selected_chat_model_name, selected_chat_adapter_path

    normalized_model_name = model_name.strip()
    normalized_adapter_path = adapter_path.strip()
    try:
        _ensure_loaded_model(normalized_model_name, normalized_adapter_path)
        selected_chat_model_name = normalized_model_name
        selected_chat_adapter_path = normalized_adapter_path
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "message": "Model loaded.",
        "model_name": normalized_model_name,
        "adapter_path": normalized_adapter_path,
    }


@app.get("/api/get_models")
def get_models():
    ''' Endpoint to get the list of available models and adapters. '''
    return model_selection()  # Return the full model list as a JSON


def _validate_multimodel_max_turns(max_turns: int) -> int:
    '''Validate configurable multimodel turn limits before session creation.'''
    try:
        parsed_max_turns = int(max_turns)
    except (TypeError, ValueError) as exc:
        raise ValueError("Max turns must be an integer.") from exc

    if parsed_max_turns < 1 or parsed_max_turns > HARD_MULTIMODEL_MAX_TURNS:
        raise ValueError(f"Max turns must be between 1 and {HARD_MULTIMODEL_MAX_TURNS}.")
    return parsed_max_turns


def _empty_multimodel_session() -> dict[str, object]:
    '''Return a stable idle payload for frontend session polling.'''
    return {
        "active": False,
        "status": "idle",
        "is_stopped": False,
        "is_complete": True,
        "turn_count": 0,
        "turns": [],
        "last_turn": None,
        "next_speaker": None,
    }


@app.get("/api/multimodel/config")
def get_multimodel_config():
    '''Return defaults and hard limits for model-to-model conversations.'''
    return {
        "default_max_turns": multimodel_default_max_turns,
        "hard_max_turns": HARD_MULTIMODEL_MAX_TURNS,
        "min_participants": MIN_MULTIMODEL_PARTICIPANTS,
        "max_participants": MAX_MULTIMODEL_PARTICIPANTS,
    }


@app.post("/api/multimodel/config")
def update_multimodel_config(config: MultiModelConfigRequest):
    '''Update the default multimodel turn count used by new sessions.'''
    global multimodel_default_max_turns

    try:
        multimodel_default_max_turns = _validate_multimodel_max_turns(config.max_turns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return get_multimodel_config()


@app.post("/api/multimodel/start")
def start_multimodel_conversation(payload: MultiModelStartRequest):
    '''Create a new model-to-model conversation session without generating yet.'''
    global active_multimodel_conversation

    try:
        participants = [
            MultiModelParticipant(
                name=participant.name,
                character=participant.character,
                work=participant.work,
                model_name=participant.model_name,
                adapter_path=participant.adapter_path,
            )
            for participant in payload.participants
        ]
        max_turns = (
            multimodel_default_max_turns
            if payload.max_turns is None
            else _validate_multimodel_max_turns(payload.max_turns)
        )
        active_multimodel_conversation = MultiModelConversation(
            participants=participants,
            initial_prompt=payload.initial_prompt,
            max_turns=max_turns,
            shakespeare_style=payload.shakespeare_style,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return active_multimodel_conversation.to_dict()


@app.post("/api/multimodel/next")
def generate_multimodel_turn():
    '''Generate the next round-robin turn for the active multimodel session.'''
    if active_multimodel_conversation is None:
        raise HTTPException(status_code=400, detail="No multimodel session is active.")

    if active_multimodel_conversation.is_complete:
        return active_multimodel_conversation.to_dict()

    try:
        next_turn = active_multimodel_conversation.generate_next_turn(_ensure_loaded_model)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return active_multimodel_conversation.to_dict(last_turn=next_turn)


@app.post("/api/multimodel/stop")
def stop_multimodel_conversation():
    '''Stop the active model-to-model conversation before any later turn.'''
    if active_multimodel_conversation is None:
        return _empty_multimodel_session()

    active_multimodel_conversation.stop()
    return active_multimodel_conversation.to_dict()


@app.get("/api/multimodel/session")
def get_multimodel_session():
    '''Return the current model-to-model conversation session, if any.'''
    if active_multimodel_conversation is None:
        return _empty_multimodel_session()

    return active_multimodel_conversation.to_dict()


@app.post("/api/tts")
def generate_tts(text: str, character: str = "Hamlet"):
    ''' Endpoint to generate TTS audio from the given text. '''
    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Text is required.")

    try:
        generate_audio, sample_rate = _load_bark_dependencies()
    except RuntimeError as bark_error:
        try:
            fallback_audio = _generate_fallback_tts_audio(normalized_text, character=character)
        except RuntimeError as fallback_error:
            raise HTTPException(
                status_code=503,
                detail=f"{bark_error} Fallback TTS also failed: {fallback_error}",
            ) from fallback_error
        return Response(content=fallback_audio, media_type="audio/wav")

    # Generate the speech (TODO make seperate voices for each character)
    try:
        audio_array = generate_audio(
            normalized_text,
            history_prompt=_resolve_bark_history_prompt(character),
        )  # TODO: Refactor to use character specific voices (may need to switch TTS library to support voice cloning)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    # Save to buffer to send over HTTP
    buffer = io.BytesIO()
    wav.write(buffer, sample_rate, audio_array)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")

if __name__ == "__main__":
    # TODO initialize models, initialize vector stores, etc. here before starting the server
    backend_port = int(os.getenv("BACKEND_PORT", os.getenv("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=backend_port)
    
