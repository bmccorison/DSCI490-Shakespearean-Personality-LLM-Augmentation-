''' Placeholder for data ingestion code for initial RAG vector store. '''
import json

def extract_data(name="hamlet") -> dict:
    ''' Placeholder for data extraction code (character profile) and transformations. '''
    with open("data/character_profile.json", "r") as f:
        character_profile = json.load(f)
    if character_profile == None:
        raise ValueError("Character profile data not found. Try running the character_profile_parser.py script on the .txt file.")
    return character_profile


def data_ingestion():
    ''' Main function to orchestrate data ingestion of RAG context. '''
    extracted_json = extract_data()  # Extract the character profile JSON
    return extracted_json  # TODO: Convert to vector store and return the vector store for retrieval


def reverse_mapping() -> str:
    ''' Placeholder for reverse mapping code to help with vocabulary and response formatting to make it more Shakespearean. '''
    pass  # TODO: Implement with some sort of dictionary mapping (same as used in training pre-processing)
