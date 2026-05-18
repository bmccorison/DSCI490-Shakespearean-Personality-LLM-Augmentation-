import torch
from transformers import pipeline

''' 
Testing basic model generation using the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model. 
Ensure that all dependencies from requirements.txt are installed before running this script. 
'''

def generate_text(pipe, prompt, max_length=50):
    # Generate text using the pipeline
    result = pipe(prompt, max_length=max_length, do_sample=True, top_p=0.95)
    
    # Extract and return the generated text
    generated_text = result[0]['generated_text']
    
    return generated_text

if __name__ == "__main__":
    # Define model name
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Initialize the pipeline for text generation
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    
    # Define the prompt for text generation
    prompt = input("Enter a prompt for text generation: ")
    prompt += " Answer as if you are Hamlet from Shakespeare's play."  # NOTE: This would be refactored to be a system prompt later.
    
    # Generate and print the text
    generated_text = generate_text(pipe, prompt)
    print("Generated Text:\n", generated_text)
    
    # Loop to allow multiple generations
    askAgain = input("Do you want to generate more text? (yes/no): ")
    while askAgain.lower() == "yes":
        prompt = input("Enter a prompt for text generation: ")
        prompt += " Answer as if you are Hamlet from Shakespeare's play."
        generated_text = generate_text(pipe, prompt)
        print("Generated Text:\n", generated_text)
        askAgain = input("Do you want to generate more text? (yes/no): ")