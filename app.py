''' Handle fastapi endpoints for the front-end interface. '''

from fastapi import FastAPI, Response
import uvicorn
import pipeline
from pipeline.lm_generation import generate_response, refresh_chat_history, model_selection
from pipeline.rag import get_context
from pipeline.data_ingestion import data_ingestion
import io
from TTS.api import TTS
import soundfile as sf

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
def select_model(model_name: str):
    ''' Endpoint to select the specific LLM for response generation. '''
    pass


@app.get("/get_models"):
def get_models():
    ''' Endpoint to get the list of available models and adapters. '''
    return model_selection()  # Return the full model list as a JSON


@app.post("/tts")
def generate_tts(text: str, character: str = "Hamlet"):
    ''' Endpoint to generate TTS audio from the given text. '''
    # Initialize the sound model (TODO make seperate voices for each character)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    audio = tts.tts(text)
    
    # Save the audio to a buffer and return it as a response
    buffer = io.BytesIO()
    sf.write(buffer, audio, 22050, format='WAV')
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
    # TODO initialize models, initialize vector stores, etc. here before starting the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    