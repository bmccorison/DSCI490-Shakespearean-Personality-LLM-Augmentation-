# training/rl_hamlet_ppo.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import full_play_translator
import hamlet_speaker_aware_to_message_style_prompt as message_builder

from rl_dataset import build_rl_prompts
from rl_rewards import compute_rewards


BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "models/lora_hamlet_5"
OUTPUT_PATH = "models/lora_hamlet_rl"


def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    # add value head (required for PPO)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def freeze_non_lora(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def main():
    repo_root = full_play_translator.resolve_repo_root()

    # 🔹 load your existing dataset builder
    message_records, _ = message_builder.load_or_build_message_records(
        repo_root=repo_root,
        message_input_file="data/hamlet_speaker_aware_messages.json",
        context_input_file="data/hamlet_speaker_aware_context.json",
        full_play_input_file="data/hamlet_full_play.txt",
        speaker="Hamlet",
        k=4,
        include_last_speaker_line=False,
        system_prompt=message_builder.DEFAULT_SYSTEM_PROMPT,
        encoding="utf-8",
    )

    prompts = build_rl_prompts(message_records, limit=500)

    print(f"Loaded {len(prompts)} RL prompts")

    model, tokenizer = load_model()
    freeze_non_lora(model)

    config = PPOConfig(
        batch_size=1,
        learning_rate=1e-5,
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    for step, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        rewards = compute_rewards(responses)

        ppo_trainer.step(
            list(inputs["input_ids"]),
            list(outputs),
            list(rewards),
        )

        if step % 10 == 0:
            print(f"Step {step} | Reward: {rewards.mean().item():.3f}")

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"Saved RL-tuned model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
