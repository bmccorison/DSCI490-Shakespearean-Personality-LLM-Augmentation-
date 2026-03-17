''' Placeholder for LLM pipeline functions '''

from pathlib import Path

from models.models import model_list

# Declare a messages object to hold conversation history, loaded with the initial system prompt.
# TODO: This will eventually need to be refactored to support multiple conversations and users, 
# but for now we can just use a global variable to hold the conversation history for simplicity.
messages: list[dict[str, str]] = [
    {"role": "system", "content": "You are Hamlet from Shakespeare's work Hamlet."}
]


def get_chat_template(tokenizer, usr_msg=None, context=None):
    ''' Returns the tokenized chat template with the conversation history and RAG context. '''
    # Add the context and user message to the conversation history
    add_chat_history(user_msg=f"Context: {context}\n\n{usr_msg}")
    
    # Return the tokenized chat template to be fed into the LLM
    return tokenizer.apply_chat_template(
        messages,
        tokenize=true,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    

# TODO refactor this to make it better (and support multiple conversations/users)
def get_system_prompt() -> str:
    ''' Returns the system prompt for the conversation. '''
    return (
        """
        You are Hamlet from Shakespear's works Hamlet. 
        Use the following retrieved context to answer the question as best as you can. 
        Always use all available information to answer the question as accurately as possible.
        """
    )

    
def add_chat_history(user_msg=None, model_response=None):
    ''' Add the user message and/or model response to the conversation history. '''
    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg})
    if model_response is not None:
        messages.append({"role": "assistant", "content": model_response})

def refresh_chat_history():
    ''' Called when a new conversation starts to clear the conversation history. '''
    messages.clear()


def generate_response(tokenized_chat, model) -> str:
    ''' Placeholder for response generation code. '''
    output = model.generate(tokenized_chat)
    return post_processing(output)


def post_processing(response) -> str:
    ''' Placeholder for response post-processing code (such as extracting specific response). '''
    # TODO: May want to reverse mapping to help vocabulary and response formatting to make it more Shakespearean
    pass


def generate_output(question, tokenizer, model, context=None) -> str:
    ''' Main function to orchestrate LLM generation. '''
    # Build the prompt and get the tokenized chat template
    tokenized_chat = get_chat_template(tokenizer, user_msg, context)
    
    # Generate a repsonse from the LLM and post-process it to extract the final response string
    final_response = generate_response(tokenized_chat, model)
    return final_response


def model_selection():
    ''' Placeholder for model selection code to return the list of available models and adapters. '''
    return model_list()  # Return the list of available models and adapters from the models module


def get_model(model_name: str, adapter_path: str):
    ''' Load the base Hugging Face model and apply the proper PEFT adapter, then return the model. '''
    
    # Get the path from the repo root
    repo_root = Path(__file__).resolve().parent.parent
    resolved_adapter_path = Path(adapter_path)
    if not resolved_adapter_path.is_absolute():
        resolved_adapter_path = repo_root / resolved_adapter_path
    if not resolved_adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {resolved_adapter_path}")

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Add the LoRA adapter and return
    model = PeftModel.from_pretrained(base_model, str(resolved_adapter_path))
    model.eval()
    return model
    
