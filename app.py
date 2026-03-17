''' Handle fastapi endpoints for the front-end interface. '''

from fastapi import FastAPI
import uvicorn
import pipeline
import pipeline.lm_generation as lm_generation
import pipeline.data_ingestion as data_ingestion

app = FastAPI()

@app.get("/generate_response")
def generate_response(question: str):
    ''' Endpoint to generate a response from the LLM given a user question. '''
    # Ingest data for RAG context
    rag_context = data_ingestion.ingest_data()
    
    # Generate the response from the LLM
    response = pipeline.lm_generation.generate_output(question, rag_context)
    
    return {"response": response}