''' Handle fastapi endpoints for the front-end interface. '''

from fastapi import FastAPI, Response
import uvicorn
import pipeline
from pipeline.lm_generation import generate_response, refresh_chat_history, model_selection, get_model
from pipeline.rag import get_context
from pipeline.data_ingestion import data_ingestion
from bark import generate_audio, SAMPLE_RATE  # TTS (TODO may want to refactor to use voice cloning capable library)
import scipy.io.wavfile as wav
import io

# TODO: Refactor the support multiple chat histories and characters
app = FastAPI()  # Initialize the FastAPI app
model = None  # Placeholder for the LLM model, to be loaded in by user


@app.get("/generate_response")
def generate_response(question: str):
    ''' Endpoint to trigger the response pipeline given a user question. '''
    # Ingest data for RAG context
    rag_context = get_context()
    
    # Generate the response JSON from the LLM {response: str, confidence_score: int}
    response_json = generate_response(question, rag_context)
    
    return response_json


@app.get("/refresh_chat")
def refresh_chat():
    ''' Endpoint to trigger the reset of the conversation history. '''
    refresh_chat_history()
    return {"message": "Chat history refreshed."}


@app.get("/select_character")
def select_character(character: str, work: str):
    ''' Endpoint to select the character and work for the system prompt. '''
    # TODO


@app.get("/select_model")
def select_model(model_name: str, adapter_path: str):
    ''' Endpoint to select the specific LLM for response generation. '''
    model = get_model(model_name)  # Load the model and set it to the global variable


@app.get("/get_models"):
def get_models():
    ''' Endpoint to get the list of available models and adapters. '''
    return model_selection()  # Return the full model list as a JSON


@app.post("/tts")
def generate_tts(text: str, character: str = "Hamlet"):
    ''' Endpoint to generate TTS audio from the given text. '''
    # Generate the speech (TODO make seperate voices for each character)
    audio_array = generate_audio(text, history_prompt="v2/en_speaker_6", use_gpu=True)  # TODO: Refactor to use character specific voices (may need to switch TTS library to support voice cloning)

    # Save to buffer to send over HTTP
    buffer = io.BytesIO()
    wav.write(buffer, SAMPLE_RATE, audio_array)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")

if __name__ == "__main__":
    # TODO initialize models, initialize vector stores, etc. here before starting the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    