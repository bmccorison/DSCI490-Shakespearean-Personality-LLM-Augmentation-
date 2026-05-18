DEFAULT_CHARACTER = "Hamlet"
DEFAULT_WORK = "Hamlet"


def model_list() -> list[dict]:
    """Return the configured base models and their adapters."""
    return [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "A smaller version of the LLaMA model, optimized for chat applications",
            "character": DEFAULT_CHARACTER,
            "work": DEFAULT_WORK,
            "default_adapter_path": "__base__",
            "adapters": [
                {
                    "name": "base_chat",
                    "path": "__base__",
                    "description": "Preferred stable TinyLlama chat model without a LoRA adapter.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "hamlet_lora_1",
                    "path": "models/lora_finetuned_model/checkpoint-270",
                    "description": "Early experimental LoRA adapter trained on Hamlet dialogue.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "hamlet_lora_2",
                    "path": "models/lora_finetuned_model1",
                    "description": "Preferred LoRA adapter trained on the Hamlet character profile.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
            ],
        },
        {
            "name": "LiquidAI/LFM2-2.6B",
            "description": "A 2.6B parameter model",
            "character": DEFAULT_CHARACTER,
            "work": DEFAULT_WORK,
            "default_adapter_path": "models/lora_hamlet_5_2",
            "adapters": [
                {
                    "name": "hamlet_lora_3",
                    "path": "models/lora_hamlet_3",
                    "description": "LoRA adapter trained on a mix of Hamlet dialogue and character profile.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "hamlet_lora_5",
                    "path": "models/lora_hamlet_5",
                    "description": "Context-aware LoRA adapter trained on speaker-aware message-style dialogue windows for Hamlet.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "hamlet_lora_5_2",
                    "path": "models/lora_hamlet_5_2",
                    "description": "Improved context-aware LoRA adapter — excludes Act 5 Scene 2 and enforces scene boundary context windows.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "hamlet_lora_5_3",
                    "path": "models/lora_hamlet_5_3",
                    "description": "Testing Model performance with simple dynamic system prompts",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
                {
                    "name": "macbeth_lora_1",
                    "path": "models/lora_macbeth_1",
                    "description": "Context-aware LoRA adapter trained on speaker-aware message-style dialogue windows for Macbeth.",
                    "character": "Macbeth",
                    "work": "Macbeth",
                },
            ],
        },
        {
            "name": "LiquidAI/LFM2-8B-A1B",
            "description": "Larger 8B model good with creative writing",
            "character": DEFAULT_CHARACTER,
            "work": DEFAULT_WORK,
            "default_adapter_path": "models/lora_hamlet_4",
            "adapters": [
                {
                    "name": "hamlet_lora_4",
                    "path": "models/lora_hamlet_4",
                    "description": "LoRA adapter trained on Hamlet speeches translated to modern English with the learned reverse translator.",
                    "character": DEFAULT_CHARACTER,
                    "work": DEFAULT_WORK,
                },
            ],
        },
    ]
