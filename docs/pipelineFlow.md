# Pipeline Flow Documentation

This document summarizes the interaction between `app.py` and the `pipeline` directory, explaining the high-level architecture and execution flow for model generation within the project.

## High-Level Architecture

The application is structured as a FastAPI web service (`app.py`) that acts as the entry point and orchestration layer. It depends heavily on the modules within the `pipeline` directory (primarily `pipeline.lm_generation`) to execute the heavy lifting of model loading, memory management, chat history tracking, and text generation.

### 1. `app.py` (The Routing & Orchestration Layer)
`app.py` defines multiple FastAPI endpoints. It maintains global state for the currently loaded `model` and `tokenizer` to prevent reloading heavy models into memory (VRAM/RAM) across requests. It also implements internal logic for generating Text-to-Speech (TTS) audio with graceful fallbacks (Bark, Piper, Espeak).

### 2. `pipeline` Directory (The Core Engine)
The `pipeline` directory encapsulates the AI logic. The most prominent module is `pipeline.lm_generation`, which handles device negotiation (CUDA/MPS/CPU), context window management, prompt templating, tokenization, model inference, and text post-processing.

---

## Endpoint to Pipeline Tool Mapping

Here is how each endpoint in `app.py` calls specific tools in the `pipeline` directory:

### Model Selection and Discovery
- **`GET /api/get_models`**
  - **Tool Called:** `pipeline.lm_generation.model_selection()`
  - **Flow:** Returns a list of available base models and their compatible LoRA adapters. Provides a configuration schema that the frontend uses to populate selection dropdowns.

- **`GET /api/select_model`**
  - **Tool Called:** `pipeline.lm_generation.get_model(model_name, adapter_path)`
  - **Flow:** User selects a model and adapter. `app.py` unloads any existing model via `_release_loaded_model()` (triggering garbage collection and CUDA cache clearing) and then uses `get_model()` to load the base model and dynamically apply the selected PEFT LoRA adapter. The `model` and `tokenizer` are saved globally.

### Setting Persona and Context
- **`GET /api/select_character`**
  - **Tool Called:** `pipeline.lm_generation.set_character_context(character, work)`
  - **Flow:** Configures the targeted Shakespearean character and play. This internally updates the system prompt definitions inside the `lm_generation` module and automatically clears chat history so the new persona takes full effect.

- **`GET /api/refresh_chat`**
  - **Tool Called:** `pipeline.lm_generation.refresh_chat_history()`
  - **Flow:** Clears the chat history (the internal `messages` state in `lm_generation.py`) and resets the conversation log with a fresh System Prompt.

### Generation Flow
- **`GET /api/generate_response`**
  - **Tool Called:** `pipeline.lm_generation.generate_output(question, tokenizer, model, context, apply_shakespeare_style)`
  - **Flow:** This is the primary execution path.
    1. **Context Parsing:** If RAG feature is implemented (`pipeline.rag.get_context`), it retrieves semantic chunks.
    2. **History Management:** `lm_generation.get_chat_template()` appends the user content to the `messages` history payload and safely bounds context sizes (`_trim_chat_history`). 
    3. **Tokenization:** Converts the whole conversation, wrapped in explicit role-tags (e.g. `<|user|>`), into tensor inputs.
    4. **Inference:** `generate_response()` runs inference via `model.generate()`. It handles fallbacks, limits max tokens, and utilizes anti-repetition defaults. If it detects a repetitive infinite loop, it attempts graceful retries with stricter rules or by temporarily disabling the LoRA adapter.
    5. **Post-Processing:** Resulting tokens are decoded. A formatting layer strips special tokens and optionally runs `_apply_shakespeare_dialogue_style()` to lightly modify words ("you" -> "thou").
    6. **State Update:** The resulting text is appended to the message history and returned back to the caller in `app.py`.

### Text-To-Speech (TTS) Flow
- **`POST /api/tts`**
  - **Flow:** Driven primarily by the logic defined within `app.py`. It uses a cascade approach, first attempting to lazily load and utilize `bark` for expressive audio, then failing over to `piper`, and finally settling on `espeak` CPU-safe binaries to guarantee an audio response is returned.

---

## Summary
The separation of concerns between `app.py` and `pipeline` allows the FastAPI application to remain lightweight and primarily focused on request parsing, dependency management, and fallback chaining (likeTTS logic). The `pipeline` isolates all of the deep learning complexities—such as PEFT adapter merging, PyTorch/Accelerate resource configuration, context window truncation, and generation retries—into dedicated, maintainable modules.
