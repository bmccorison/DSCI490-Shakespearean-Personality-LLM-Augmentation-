def model_list() -> list[dict]:
    """ Model List Appendix """
    return [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "description": "A smaller version of the LLaMA model, optimized for chat applications",
            "adapter_paths": [
                { "hamlet_lora_1": "models/lora_finetuned_model1", "description": "Lobotmized adapter trained on Hamlet dialouge"},
                { "hamlet_lora_2": "models/lora_finetuned_model1", "description": "Lightweight adapter trained on character profile"}
            ]
        },
        {
            "name": "LiquidAI/LFM2-8B-A1B",
            "description": "Larger 8B model good with creative writing",
            "adapter_paths": [
                None  # TODO: Train some adapters for this
            ]
        }
    ]
