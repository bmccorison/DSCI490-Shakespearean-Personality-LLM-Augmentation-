def model_list() -> list[dict]:
    """Return the configured base models and their adapters."""
    return [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "A smaller version of the LLaMA model, optimized for chat applications",
            "default_adapter_path": "__base__",
            "adapters": [
                {
                    "name": "base_chat",
                    "path": "__base__",
                    "description": "Preferred stable TinyLlama chat model without a LoRA adapter.",
                },
                {
                    "name": "hamlet_lora_1",
                    "path": "models/lora_finetuned_model1",
                    "description": "Early experimental LoRA adapter trained on Hamlet dialogue.",
                },
                {
                    "name": "hamlet_lora_2",
                    "path": "models/lora_finetuned_model/checkpoint-270",
                    "description": "Preferred LoRA adapter trained on the Hamlet character profile.",
                },
            ],
        },
        {
            "name": "LiquidAI/LFM2-8B-A1B",
            "description": "Larger 8B model good with creative writing",
            "adapters": [],
        },
    ]
