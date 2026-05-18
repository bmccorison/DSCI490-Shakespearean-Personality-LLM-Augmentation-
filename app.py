''' Handle fastapi endpoints for the front-end interface. '''

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from pipeline.feedback_store import load_feedback, save_feedback
from pipeline.lm_generation import (
    generate_output,
    get_conversation_id,
    get_message_index,
    model_selection,
    refresh_chat_history,
    set_character_context,
)
from pipeline.multimodel import (
    DEFAULT_MAX_TURNS as DEFAULT_MULTIMODEL_MAX_TURNS,
    HARD_MAX_TURNS as HARD_MULTIMODEL_MAX_TURNS,
    MAX_PARTICIPANTS as MAX_MULTIMODEL_PARTICIPANTS,
    MIN_PARTICIPANTS as MIN_MULTIMODEL_PARTICIPANTS,
    MultiModelConversation,
    MultiModelParticipant,
    validate_max_turns as validate_multimodel_max_turns,
)
from pipeline.rag import get_context
from pipeline.tts import generate_tts_audio, get_voice_options
from pipeline.utils import (
    empty_multimodel_session,
    ensure_loaded_model,
    resolve_cors_origins,
    resolve_multimodel_persona,
)


app = FastAPI()
default_cors_origins = "http://localhost:6969,http://127.0.0.1:6969"
allowed_origins = resolve_cors_origins(
    os.getenv("CORS_ALLOW_ORIGINS", default_cors_origins),
    default_cors_origins,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-chat selection — persisted across /api/select_model calls.
selected_chat_model_name = ""
selected_chat_adapter_path = ""

active_multimodel_conversation: MultiModelConversation | None = None
multimodel_default_max_turns = DEFAULT_MULTIMODEL_MAX_TURNS


class MultiModelParticipantRequest(BaseModel):
    '''Request payload for one model-to-model speaker.'''

    name: str
    character: str | None = None
    work: str | None = None
    model_name: str
    adapter_path: str


class MultiModelStartRequest(BaseModel):
    '''Request payload used to create a new multimodel conversation.'''

    initial_prompt: str
    participants: list[MultiModelParticipantRequest]
    max_turns: int | None = None
    shakespeare_style: bool = False
    rag_enabled: bool = True


class MultiModelConfigRequest(BaseModel):
    '''Request payload for updating multimodel defaults.'''

    max_turns: int


class SpanFeedback(BaseModel):
    '''Span-level feedback for highlighted response text.'''

    text: str
    polarity: str


class MessageFeedback(BaseModel):
    '''Feedback payload for one generated assistant message.'''

    conversation_id: str
    message_index: int
    vote: str
    spans: list[SpanFeedback] = Field(default_factory=list)


@app.post("/api/feedback")
def submit_feedback(feedback: MessageFeedback):
    '''Endpoint to receive per-message votes and span highlights from the frontend.'''
    from pipeline.local_logging import DEFAULT_LOGGING_DIR

    conversation_id = feedback.conversation_id.strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required.")

    matching_files = list(DEFAULT_LOGGING_DIR.rglob(f"*{conversation_id}*.json"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="Conversation log not found.")

    log_file = matching_files[0]

    existing_feedback = load_feedback(log_file)
    existing_feedback = [
        record for record in existing_feedback
        if record.get("message_index") != feedback.message_index
    ]
    existing_feedback.append({
        "message_index": feedback.message_index,
        "vote": feedback.vote,
        "spans": [{"text": s.text, "polarity": s.polarity} for s in feedback.spans],
    })

    save_feedback(log_file, existing_feedback)
    return {"message": "Feedback saved.", "message_index": feedback.message_index}


@app.get("/api/feedback/{conversation_id}")
def get_feedback(conversation_id: str):
    '''Endpoint to retrieve saved feedback for a conversation.'''
    from pipeline.local_logging import DEFAULT_LOGGING_DIR

    normalized_conversation_id = conversation_id.strip()
    if not normalized_conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required.")

    matching_files = list(
        DEFAULT_LOGGING_DIR.rglob(f"*{normalized_conversation_id}*.json")
    )
    if not matching_files:
        raise HTTPException(status_code=404, detail="Conversation log not found.")

    return {"feedback": load_feedback(matching_files[0])}


@app.get("/api/generate_response")
def generate_response_endpoint(
    question: str,
    shakespeare_style: bool = False,
    rag_enabled: bool = True,
):
    ''' Endpoint to trigger the response pipeline given a user question. '''
    global selected_chat_model_name, selected_chat_adapter_path

    if not selected_chat_model_name or not selected_chat_adapter_path:
        raise HTTPException(status_code=400, detail="Model is not loaded. Call /api/select_model first.")

    rag_context = get_context(question) if rag_enabled else None

    try:
        active_model, active_tokenizer = ensure_loaded_model(
            selected_chat_model_name,
            selected_chat_adapter_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response_text = generate_output(
        question,
        active_tokenizer,
        active_model,
        rag_context,
        apply_shakespeare_style=shakespeare_style,
    )

    return {
        "response": response_text,
        "conversation_id": get_conversation_id(),
        "message_index": get_message_index(),
    }


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
        ensure_loaded_model(normalized_model_name, normalized_adapter_path)
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
    return model_selection()


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
        multimodel_default_max_turns = validate_multimodel_max_turns(config.max_turns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return get_multimodel_config()


@app.post("/api/multimodel/start")
def start_multimodel_conversation(payload: MultiModelStartRequest):
    '''Create a new model-to-model conversation session without generating yet.'''
    global active_multimodel_conversation

    try:
        participants = []
        for participant in payload.participants:
            character, work = resolve_multimodel_persona(
                participant.model_name,
                participant.adapter_path,
            )
            participants.append(
                MultiModelParticipant(
                    name=participant.name,
                    character=character,
                    work=work,
                    model_name=participant.model_name,
                    adapter_path=participant.adapter_path,
                )
            )
        max_turns = (
            multimodel_default_max_turns
            if payload.max_turns is None
            else payload.max_turns
        )
        rag_context = (
            get_context(payload.initial_prompt) if payload.rag_enabled else ""
        )
        active_multimodel_conversation = MultiModelConversation(
            participants=participants,
            initial_prompt=payload.initial_prompt,
            max_turns=max_turns,
            shakespeare_style=payload.shakespeare_style,
            rag_context=rag_context,
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
        next_turn = active_multimodel_conversation.generate_next_turn(ensure_loaded_model)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return active_multimodel_conversation.to_dict(last_turn=next_turn)


@app.post("/api/multimodel/stop")
def stop_multimodel_conversation():
    '''Stop the active model-to-model conversation before any later turn.'''
    if active_multimodel_conversation is None:
        return empty_multimodel_session()

    active_multimodel_conversation.stop()
    return active_multimodel_conversation.to_dict()


@app.get("/api/multimodel/session")
def get_multimodel_session():
    '''Return the current model-to-model conversation session, if any.'''
    if active_multimodel_conversation is None:
        return empty_multimodel_session()

    return active_multimodel_conversation.to_dict()


@app.get("/api/voices")
def list_voices():
    ''' Endpoint to list available ElevenLabs voice options. '''
    return {"voices": get_voice_options()}


@app.post("/api/tts")
def generate_tts(text: str, character: str = "Hamlet", voice: str | None = None):
    ''' Endpoint to generate TTS audio from the given text via ElevenLabs. '''
    try:
        audio_bytes, media_type = generate_tts_audio(
            text,
            character=character,
            voice=voice,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio_bytes, media_type=media_type)


FRONTEND_DIST = Path(__file__).resolve().parent / "interface" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    backend_port = int(os.getenv("BACKEND_PORT", os.getenv("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=backend_port)
