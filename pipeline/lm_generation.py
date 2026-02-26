''' Placeholder for LLM pipeline functions '''

def prompt_builder(question, context=None):
    ''' Builds the prompt fed into the LLM '''
    return f"""
    System Prompt: {get_system_prompt()}
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    """

def get_system_prompt():
    ''' Placeholder for system prompt retrieval code. '''
    pass

def generate_response(prompt):
    ''' Placeholder for response generation code. '''
    pass

def post_processing(response):
    ''' Placeholder for response post-processing code (such as extracting specific response). '''
    pass

def generate_output(question, context=None):
    ''' Main function to orchestrate LLM generation. '''
    prompt = prompt_builder(question, context)
    response = generate_response(prompt)
    final_output = post_processing(response)
    return final_output