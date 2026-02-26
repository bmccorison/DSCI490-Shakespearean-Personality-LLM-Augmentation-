''' Placeholder for RAG pipeline functions '''
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, size=512, overlap=50):
    ''' Placeholder for text chunking code. '''
    pass

def embed_chunks(chunks):
    ''' Returns the vector embeddings of the document '''
    return model.encode(chunks)

def retrieve(query, vector_store):
    ''' Placeholder for retrieval code. '''
    pass