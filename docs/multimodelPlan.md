# Architectural Plan for Multi-Model Conversation

Ultimately, the goal will be to have >2 models conversing to each other with one chat interface.

## Computational Considerations

Due to computation constraints (& speed), we will probably just want to have one base model and multiple adapters loaded in. 

## Three New "Functions"

In order to successfully do this, we will need a:

- **Message Bus** - A way to send messages between the models. This could be a simple in-memory queue or be handled using multiple model instances communicating over HTTP for our purposes. If we want to scale this with larger models, we will eventually want to move to the HTTP approach, but for now we can just use an in-memory message bus to keep things simple.
- **State Manager** - A way to keep track of the conversation state, including which model is currently active and the history of messages. This could be implemented as a simple class that maintains the conversation history and the current active model.
- **Orchestrator** - A component that decides which model should respond to a given message based on the conversation context and the capabilities of each model. This could probably just be in a round-robin approach, or we could define something different.

## Context Window Considerations

Since these models are (relatively) small, storing the entire conversation history in the context window might not be feasible. We will need to implement a strategy for summarizing or truncating the conversation history to fit within the context window of each model.

Some ideas for this could be:
- Re-injecting the system prompt every x messages
- Summarizing/compacting the conversation history after a certain number of messages to prevent excessive drift (could be done by the non-adapted base model, or some larger model via API or something)
- Only including the most recent messages in the context window, while keeping a separate log of the full conversation history for reference.

## Other Notes

When developing, we will want for each model to have clear identifiers and will want to customize the system prompts for each model, ensuring that each one understands that it is in a conversation.
