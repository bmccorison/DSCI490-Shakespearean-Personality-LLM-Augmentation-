''' Placeholder for LLM pipeline functions '''

import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.models import model_list

# Resolve project-relative adapter paths from a stable repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Special adapter token meaning: use the base chat model without LoRA overlays.
BASE_MODEL_ADAPTER_PATH = "__base__"
# Default persona context injected into the system prompt.
DEFAULT_CHARACTER = "Hamlet"
DEFAULT_WORK = "Hamlet"
# Baseline generation controls; each can be overridden by environment variables.
DEFAULT_MAX_CHAT_HISTORY_TURNS = 4
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
SHAKESPEARE_STYLE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Simple phrase substitutions used by the optional style-polish step.
    (r"\byou are\b", "thou art"),
    (r"\byour\b", "thy"),
    (r"\byours\b", "thine"),
    (r"\byou\b", "thou"),
    (r"\boften\b", "oft"),
    (r"\bperhaps\b", "perchance"),
    (r"\bbefore\b", "ere"),
)


def _read_int_setting(name: str, default: int, minimum: int) -> int:
    '''Read a positive integer from the environment, falling back safely when invalid.'''
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default

    return max(minimum, parsed_value)


def _read_float_setting(name: str, default: float, minimum: float) -> float:
    '''Read a positive float from the environment, falling back safely when invalid.'''
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed_value = float(raw_value)
    except ValueError:
        return default

    return max(minimum, parsed_value)


MAX_CHAT_HISTORY_TURNS = _read_int_setting(
    "MAX_CHAT_HISTORY_TURNS",
    DEFAULT_MAX_CHAT_HISTORY_TURNS,
    minimum=1,
)
# These globals are resolved once at import so every request uses consistent limits.
MAX_NEW_TOKENS = _read_int_setting(
    "MAX_NEW_TOKENS",
    DEFAULT_MAX_NEW_TOKENS,
    minimum=1,
)
TEMPERATURE = _read_float_setting("GENERATION_TEMPERATURE", DEFAULT_TEMPERATURE, minimum=0.1)
TOP_P = _read_float_setting("GENERATION_TOP_P", DEFAULT_TOP_P, minimum=0.1)
REPETITION_PENALTY = _read_float_setting(
    "GENERATION_REPETITION_PENALTY",
    DEFAULT_REPETITION_PENALTY,
    minimum=1.0,
)
NO_REPEAT_NGRAM_SIZE = _read_int_setting(
    "GENERATION_NO_REPEAT_NGRAM_SIZE",
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    minimum=1,
)

# Declare a messages object to hold conversation history, loaded with the initial system prompt.
# TODO: This will eventually need to be refactored to support multiple conversations and users,
# but for now we can just use a global variable to hold the conversation history for simplicity.
messages: list[dict[str, str]] = []
current_character = DEFAULT_CHARACTER
current_work = DEFAULT_WORK


def get_chat_template(tokenizer, usr_msg=None, context=None):
    ''' Returns the tokenized chat template with the conversation history and RAG context. '''
    # Build one composite user payload from optional RAG context + direct user message.
    prompt_parts = []
    if context is not None:
        prompt_parts.append(f"Context: {context}")
    if usr_msg is not None:
        prompt_parts.append(str(usr_msg))

    # Add the context and user message to the conversation history
    add_chat_history(user_msg="\n\n".join(prompt_parts))

    # TinyLlama responds more consistently when prompted with explicit role tags.
    prompt = _render_prompt_messages(messages)
    return tokenizer(prompt, return_tensors="pt")
    

# TODO refactor this to make it better (and support multiple conversations/users)
def get_system_prompt() -> str:
    ''' Returns the system prompt for the conversation. '''
    return (
        f"You are {current_character}, a character from Shakespeare's work {current_work}. "
        f"You are speaking directly as {current_character}, not describing {current_character} from the outside. "
        "Answer every user message in first person from the character's perspective. "
        "Stay in character at all times and never refer to yourself as an AI assistant, chatbot, or language model. "
        "Use any retrieved context as background knowledge about the character and work, "
        "but write the final answer as the character's own words. "
        "If something is uncertain, say so in character instead of inventing facts."
    )


def set_character_context(character: str, work: str) -> None:
    ''' Update the active character context and reset the chat history. '''
    global current_character, current_work

    # Normalize incoming values from query params/UI controls.
    next_character = character.strip()
    next_work = work.strip()
    if not next_character or not next_work:
        raise ValueError("Character and work are required.")

    # Persist persona state globally and restart the chat memory with new prompt.
    current_character = next_character
    current_work = next_work
    refresh_chat_history()


def add_chat_history(user_msg=None, model_response=None):
    ''' Add the user message and/or model response to the conversation history. '''
    # Append whichever side of the turn was provided by caller.
    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg})
    if model_response is not None:
        messages.append({"role": "assistant", "content": model_response})
    # Keep prompt size bounded so response latency stays predictable.
    _trim_chat_history()

def refresh_chat_history():
    ''' Called when a new conversation starts to clear the conversation history. '''
    messages.clear()
    messages.append({"role": "system", "content": get_system_prompt()})


def _render_prompt_messages(prompt_messages: list[dict[str, str]]) -> str:
    '''Render the chat history in the explicit role-tag format used by TinyLlama chat models.'''
    prompt_sections = []

    for message in prompt_messages:
        # Skip malformed entries so one bad message cannot break prompting.
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue

        # TinyLlama chat format expects a role tag and end-of-segment token.
        prompt_sections.append(f"<|{role}|>\n{content}</s>\n")

    # End with assistant tag to cue the next generated response.
    prompt_sections.append("<|assistant|>\n")
    return "".join(prompt_sections)


def _trim_chat_history() -> None:
    '''Keep the system prompt and only the most recent user turns to bound prompt growth.'''
    if len(messages) <= 2:
        return

    # Ignore system prompt while calculating trim boundaries.
    conversation = messages[1:]
    start_index = 0
    user_messages_seen = 0

    # Walk backwards to keep only the newest N user turns (+ replies that follow them).
    for index in range(len(conversation) - 1, -1, -1):
        if conversation[index].get("role") == "user":
            user_messages_seen += 1
            if user_messages_seen > MAX_CHAT_HISTORY_TURNS:
                start_index = index + 1
                break

    trimmed_conversation = conversation[start_index:]
    # Ensure the trimmed sequence starts with a user turn for prompt coherence.
    while trimmed_conversation and trimmed_conversation[0].get("role") != "user":
        trimmed_conversation = trimmed_conversation[1:]

    # Reassemble complete message list with unchanged system prompt at index 0.
    messages[:] = [messages[0], *trimmed_conversation]


def _resolve_model_device(model):
    ''' Best-effort lookup for the active model device. '''
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _prepare_generation_inputs(tokenized_chat, model):
    ''' Normalize chat-template output into a form suitable for model.generate. '''
    model_device = _resolve_model_device(model)

    if hasattr(tokenized_chat, "items"):
        # Dict-style payload from tokenizer(..., return_tensors=...) path.
        generation_inputs = dict(tokenized_chat.items())
        if model_device is not None:
            # Move each tensor to active model device before generation.
            generation_inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in generation_inputs.items()
            }
        return generation_inputs, generation_inputs.get("input_ids")

    # Fallback path for legacy tokenizer outputs that return only `input_ids`.
    prompt_input_ids = tokenized_chat
    if model_device is not None and hasattr(prompt_input_ids, "to"):
        prompt_input_ids = prompt_input_ids.to(model_device)
    return prompt_input_ids, prompt_input_ids


def _extract_generated_tokens(output, prompt_input_ids):
    ''' Decode only the newly generated continuation when prompt length is known. '''
    generated_tokens = output[0]
    if prompt_input_ids is None or not hasattr(prompt_input_ids, "shape"):
        # If prompt length is unknown, decode full output sequence.
        return generated_tokens

    prompt_length = int(prompt_input_ids.shape[-1])
    try:
        return generated_tokens[prompt_length:]
    except (TypeError, IndexError):
        return generated_tokens


def _build_generation_kwargs(generation_inputs, tokenizer, retry: bool = False):
    '''Apply conservative generation defaults so request-time work stays bounded.'''
    # Accept either dict inputs or bare tensor inputs.
    if isinstance(generation_inputs, dict):
        generation_kwargs = dict(generation_inputs)
    else:
        generation_kwargs = {"input_ids": generation_inputs}

    # Stable, low-cost defaults tuned to reduce looping and latency.
    generation_kwargs["max_new_tokens"] = MAX_NEW_TOKENS
    generation_kwargs["do_sample"] = True
    generation_kwargs["num_beams"] = 1
    generation_kwargs["use_cache"] = True
    generation_kwargs["temperature"] = max(0.1, TEMPERATURE - 0.1) if retry else TEMPERATURE
    generation_kwargs["top_p"] = max(0.1, TOP_P - 0.05) if retry else TOP_P
    generation_kwargs["repetition_penalty"] = (
        max(REPETITION_PENALTY, REPETITION_PENALTY + 0.1) if retry else REPETITION_PENALTY
    )
    generation_kwargs["no_repeat_ngram_size"] = (
        max(NO_REPEAT_NGRAM_SIZE, NO_REPEAT_NGRAM_SIZE + 1) if retry else NO_REPEAT_NGRAM_SIZE
    )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    # Supply tokenizer token ids when available to avoid generation warnings/errors.
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    return generation_kwargs


def _looks_degenerate_response(response: str) -> bool:
    '''Detect obvious repetition loops such as "I I I I I".'''
    normalized = response.strip()
    if not normalized:
        return True

    # Token-level heuristics to catch repetitive or low-diversity output.
    word_tokens = re.findall(r"[A-Za-z']+", normalized.lower())
    if len(word_tokens) < 5:
        return False

    repeated_single_token = re.search(r"\b([a-z']+)(?:\s+\1\b){3,}", normalized, flags=re.IGNORECASE)
    if repeated_single_token is not None:
        return True

    unique_words = set(word_tokens)
    if len(unique_words) <= 2 and len(word_tokens) >= 6:
        return True

    most_common_count = max(word_tokens.count(word) for word in unique_words)
    return most_common_count / len(word_tokens) >= 0.5


def _cuda_available() -> bool:
    '''Return whether CUDA is available without assuming every torch build exposes it.'''
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return False
    return bool(cuda.is_available())


def _mps_available() -> bool:
    '''Return whether Apple's Metal backend is available when running on macOS.'''
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    return bool(mps.is_available())


def _model_load_config() -> tuple[str, dict]:
    '''Choose the best available device and dtype for local inference.'''
    # Highest-priority path: CUDA for GPU acceleration.
    if _cuda_available():
        model_load_kwargs = {
            "dtype": getattr(torch, "bfloat16", getattr(torch, "float16", None)),
            "device_map": "auto",
        }
        return "cuda", {key: value for key, value in model_load_kwargs.items() if value is not None}

    # Secondary path: Apple Metal acceleration on macOS.
    if _mps_available():
        model_load_kwargs = {
            "dtype": getattr(torch, "float16", getattr(torch, "float32", None)),
        }
        return "mps", {key: value for key, value in model_load_kwargs.items() if value is not None}

    # Safe fallback path for CPU-only environments.
    model_load_kwargs = {
        "dtype": getattr(torch, "float32", None),
    }
    return "cpu", {key: value for key, value in model_load_kwargs.items() if value is not None}


def generate_response(
    tokenized_chat,
    model,
    tokenizer,
    apply_shakespeare_style: bool = True,
) -> str:
    ''' Placeholder for response generation code. '''
    # Prepare tensors for model.generate regardless of tokenizer output shape.
    generation_inputs, prompt_input_ids = _prepare_generation_inputs(tokenized_chat, model)
    generation_kwargs = _build_generation_kwargs(generation_inputs, tokenizer)
    output = model.generate(**generation_kwargs)

    # Decode only newly generated continuation (not the prompt prefix).
    decoded = tokenizer.decode(
        _extract_generated_tokens(output, prompt_input_ids),
        skip_special_tokens=True,
    )
    response_text = post_processing(
        decoded,
        apply_shakespeare_style=apply_shakespeare_style,
    )

    # Fast path for normal responses.
    if not _looks_degenerate_response(response_text):
        return response_text

    # If a LoRA adapter is active, retry once with adapter disabled.
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            base_output = model.generate(**_build_generation_kwargs(generation_inputs, tokenizer, retry=True))
        base_decoded = tokenizer.decode(
            _extract_generated_tokens(base_output, prompt_input_ids),
            skip_special_tokens=True,
        )
        base_response_text = post_processing(
            base_decoded,
            apply_shakespeare_style=apply_shakespeare_style,
        )
        if not _looks_degenerate_response(base_response_text):
            return base_response_text

    # Last resort: regenerate with stricter anti-repetition settings.
    retry_output = model.generate(**_build_generation_kwargs(generation_inputs, tokenizer, retry=True))
    retry_decoded = tokenizer.decode(
        _extract_generated_tokens(retry_output, prompt_input_ids),
        skip_special_tokens=True,
    )
    retry_response_text = post_processing(
        retry_decoded,
        apply_shakespeare_style=apply_shakespeare_style,
    )

    if _looks_degenerate_response(retry_response_text) and len(response_text) >= len(retry_response_text):
        return response_text

    return retry_response_text


def _match_case(source_text: str, replacement_text: str) -> str:
    '''Apply the source token's case pattern to the replacement token.'''
    if not source_text:
        return replacement_text
    if source_text.isupper():
        return replacement_text.upper()
    if source_text[0].isupper():
        return replacement_text.capitalize()
    return replacement_text


def _apply_shakespeare_dialogue_style(response: str) -> str:
    '''Apply a light Shakespearean polish to plain modern phrasing.'''
    styled_response = response
    # Apply each phrase replacement while preserving source capitalization.
    for pattern, replacement in SHAKESPEARE_STYLE_PATTERNS:
        styled_response = re.sub(
            pattern,
            lambda match, next_replacement=replacement: _match_case(
                match.group(0),
                next_replacement,
            ),
            styled_response,
            flags=re.IGNORECASE,
        )
    return styled_response


def post_processing(response, apply_shakespeare_style: bool = True) -> str:
    ''' Placeholder for response post-processing code (such as extracting specific response). '''
    # TODO: May want to reverse mapping to help vocabulary and response formatting to make it more Shakespearean
    # Remove any leftover chat-template tokens and normalize whitespace.
    cleaned_response = str(response).replace("<|assistant|>", " ").replace("</s>", " ")
    cleaned_response = re.sub(r"\s+", " ", cleaned_response)
    cleaned_response = cleaned_response.strip()

    # Optional style layer can be toggled by frontend request flag.
    if apply_shakespeare_style:
        cleaned_response = _apply_shakespeare_dialogue_style(cleaned_response)
    return cleaned_response


def generate_output(
    question,
    tokenizer,
    model,
    context=None,
    apply_shakespeare_style: bool = True,
) -> str:
    ''' Main function to orchestrate LLM generation. '''
    # Build the prompt and get the tokenized chat template
    tokenized_chat = get_chat_template(tokenizer, question, context)
    
    # Generate a repsonse from the LLM and post-process it to extract the final response string
    final_response = generate_response(
        tokenized_chat,
        model,
        tokenizer,
        apply_shakespeare_style=apply_shakespeare_style,
    )
    add_chat_history(model_response=final_response)
    return final_response


def _configured_adapters(configured_model: dict) -> list[dict]:
    '''Normalize model adapters from either the current or legacy config shape.'''
    # Preferred shape used by current model metadata.
    adapters = configured_model.get("adapters")
    if isinstance(adapters, list):
        return [
            adapter
            for adapter in adapters
            if isinstance(adapter, dict)
        ]

    # Legacy shape fallback for backward compatibility with older payloads.
    legacy_adapters = configured_model.get("adapter_paths")
    if not isinstance(legacy_adapters, list):
        return []

    normalized_adapters = []
    for adapter in legacy_adapters:
        if not isinstance(adapter, dict):
            continue

        # Preserve adapter description while extracting the name->path pair.
        description = adapter.get("description", "")
        adapter_fields = [
            (key, value)
            for key, value in adapter.items()
            if key != "description" and isinstance(value, str)
        ]
        if not adapter_fields:
            continue

        adapter_name, adapter_path = adapter_fields[0]
        normalized_adapters.append(
            {
                "name": adapter_name,
                "path": adapter_path,
                "description": description,
            }
        )

    return normalized_adapters


def _is_base_model_adapter(adapter_path: str) -> bool:
    '''Return whether the selected adapter path refers to the raw base model.'''
    return adapter_path.strip() == BASE_MODEL_ADAPTER_PATH


def _normalize_model_id(model_id: str) -> str:
    '''Normalize model identifiers for robust equality checks.'''
    return model_id.strip().rstrip("/").lower()


def _adapter_base_model_name(adapter_path: Path) -> str | None:
    '''Return adapter-declared base model name from adapter_config.json when available.'''
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    base_model_name = config.get("base_model_name_or_path")
    if isinstance(base_model_name, str):
        normalized_name = base_model_name.strip()
        if normalized_name:
            return normalized_name

    return None


def _is_adapter_compatible_with_model(model_name: str, adapter_path: Path) -> bool:
    '''Validate adapter base model metadata against selected base model.'''
    adapter_base_model = _adapter_base_model_name(adapter_path)
    if adapter_base_model is None:
        # If metadata is unavailable, defer compatibility check to PEFT load.
        return True

    return _normalize_model_id(adapter_base_model) == _normalize_model_id(model_name)


def _base_model_adapter_entry(configured_model: dict) -> dict[str, str]:
    '''Build a synthetic adapter entry that targets the unmodified base model.'''
    return {
        "name": "base_model",
        "path": BASE_MODEL_ADAPTER_PATH,
        "description": configured_model.get(
            "base_description",
            "Base model without a LoRA adapter.",
        ),
    }


def model_selection():
    ''' Model selection code to return the list of available models and adapters. '''
    available_models = []

    # Validate each model's adapters and only return loadable options.
    for configured_model in model_list():
        adapters = []
        for adapter in _configured_adapters(configured_model):
            adapter_path = adapter.get("path", "").strip()
            if not adapter_path:
                continue

            if _is_base_model_adapter(adapter_path):
                # Always expose synthetic base adapter token as loadable.
                adapters.append(
                    {
                        "name": adapter.get("name", adapter_path),
                        "path": adapter_path,
                        "description": adapter.get("description", ""),
                    }
                )
                continue

            resolved_path = resolve_adapter_path(adapter_path)
            if not resolved_path.exists():
                # Hide broken adapter entries from the frontend selector.
                continue
            if not _is_adapter_compatible_with_model(configured_model["name"], resolved_path):
                # Hide adapters trained on a different base model to prevent runtime shape errors.
                continue

            adapters.append(
                {
                    "name": adapter.get("name", adapter_path),
                    "path": adapter_path,
                    "description": adapter.get("description", ""),
                }
            )

        if not adapters:
            # Keep configured base models testable even when no adapter checkpoints exist.
            adapters.append(_base_model_adapter_entry(configured_model))

        if adapters:
            # Only publish models that have at least one usable adapter target.
            default_adapter_path = str(configured_model.get("default_adapter_path", "")).strip()
            available_adapter_paths = {adapter["path"] for adapter in adapters}
            if not default_adapter_path or default_adapter_path not in available_adapter_paths:
                default_adapter_path = adapters[0]["path"]

            available_models.append(
                {
                    "name": configured_model["name"],
                    "description": configured_model.get("description", ""),
                    "default_adapter_path": default_adapter_path,
                    "adapters": adapters,
                }
            )

    return available_models


def get_model(model_name: str, adapter_path: str):
    ''' Load the base Hugging Face model/tokenizer and apply PEFT adapter, then return both. '''
    # Normalize user-selected names from query params/UI form fields.
    normalized_model_name = model_name.strip()
    normalized_adapter_path = adapter_path.strip()

    if not normalized_model_name:
        raise ValueError("Model name is required.")
    if not normalized_adapter_path:
        raise ValueError("Adapter path is required.")

    selected_model = next(
        (candidate for candidate in model_selection() if candidate["name"] == normalized_model_name),
        None,
    )
    if selected_model is None:
        raise ValueError(f"Model is not available: {normalized_model_name}")

    # Validate the adapter against the chosen model's current adapter list.
    selected_adapter = next(
        (candidate for candidate in selected_model["adapters"] if candidate["path"] == normalized_adapter_path),
        None,
    )
    if selected_adapter is None:
        raise ValueError(
            f"Adapter path '{normalized_adapter_path}' is not valid for model '{normalized_model_name}'."
        )

    selected_adapter_path = selected_adapter["path"]
    use_base_model_only = _is_base_model_adapter(selected_adapter_path)
    resolved_adapter_path = None
    if not use_base_model_only:
        # Resolve and verify on-disk adapter checkpoint path.
        resolved_adapter_path = resolve_adapter_path(selected_adapter_path)
        if not resolved_adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {resolved_adapter_path}")
        if not _is_adapter_compatible_with_model(normalized_model_name, resolved_adapter_path):
            adapter_base_model = _adapter_base_model_name(resolved_adapter_path)
            raise ValueError(
                "Adapter is incompatible with selected base model. "
                f"Adapter expects '{adapter_base_model}', but selected model is '{normalized_model_name}'."
            )

    # Prefer adapter-local tokenizer if available, otherwise fall back to base model tokenizer.
    tokenizer_source = (
        str(resolved_adapter_path)
        if resolved_adapter_path is not None and (resolved_adapter_path / "tokenizer.json").exists()
        else normalized_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        # Ensure padding works for generation helpers expecting a pad token.
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    preferred_device, model_load_kwargs = _model_load_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        normalized_model_name,
        **model_load_kwargs,
    )
    if preferred_device == "mps" and hasattr(base_model, "to"):
        # MPS models need explicit transfer after loading.
        base_model = base_model.to(preferred_device)

    if use_base_model_only:
        # Base-only path skips LoRA wrapping entirely.
        base_model.eval()
        return base_model, tokenizer

    # Add the LoRA adapter and return
    try:
        model = PeftModel.from_pretrained(base_model, str(resolved_adapter_path))
    except RuntimeError as exc:
        raise ValueError(
            "Failed to load adapter weights into the selected base model. "
            "This adapter checkpoint is likely incompatible with the selected model architecture."
        ) from exc
    model.eval()
    return model, tokenizer


def resolve_adapter_path(adapter_path: str) -> Path:
    ''' Resolve an adapter path relative to the repository root when needed. '''
    # Accept absolute paths as-is and map relative paths to repo root.
    resolved_adapter_path = Path(adapter_path)
    if not resolved_adapter_path.is_absolute():
        resolved_adapter_path = REPO_ROOT / resolved_adapter_path
    return resolved_adapter_path


# Initialize global message history with default system prompt at import time.
refresh_chat_history()
