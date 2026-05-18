"""
lora_2_generate.py  —  Interactive prompt loop for the Hamlet LoRA adapter

Run from the repo root:
    python training/lora_2_generate.py

Or from the training/ directory:
    python lora_2_generate.py
"""

import os
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # training/testing/ -> repo root
ADAPTER_DIR = os.path.join(REPO_ROOT, "models", "lora_hamlet_profile")
MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = (
    "You are Hamlet, Prince of Denmark. You speak with philosophical depth, "
    "melancholic wit, and introspective honesty. Answer as Hamlet would. Never break character."
)

# ---------------------------------------------------------------------------
# PRE-FLIGHT
# ---------------------------------------------------------------------------
if not os.path.isdir(ADAPTER_DIR):
    print(f"ERROR: Adapter not found at {ADAPTER_DIR}")
    print("Run lora_2.py first to train and save the adapter.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "torch_dtype": (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    ),
}
if device == "cuda":
    model_kwargs["device_map"] = "auto"

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    adapter_name="hamlet_adapter",
)
if device == "cpu":
    model.to(device)
model.eval()
# Remove the model's default max_length so max_new_tokens is the sole limit
model.generation_config.max_length = None
print("Model ready.\n")

# ---------------------------------------------------------------------------
# GENERATION SETTINGS  (tweak these to change response style)
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS    = 350
TEMPERATURE       = 0.7
TOP_P             = 0.9  # Higher to allow more diverse word choices (0.9 is a common sweet spot)
REPETITION_PENALTY = 1.15

# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------
def generate(question: str) -> str:
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# BATCH FILE MODE
# ---------------------------------------------------------------------------
def run_file_mode():
    """Read questions from a file (one per line), generate answers, write to a new file."""
    input_path = input("  Input file path (one question per line): ").strip()
    if not input_path:
        print("  Aborted.")
        return

    if not os.path.isfile(input_path):
        print(f"  ERROR: File not found — {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("  No questions found in file.")
        return

    # Output file sits next to the input file, with _answers suffix
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_answers{ext or '.txt'}"

    print(f"  Found {len(questions)} question(s). Generating answers...")
    print(f"  Output → {output_path}\n")

    with open(output_path, "w", encoding="utf-8") as out:
        for i, question in enumerate(questions, 1):
            label = question[:80] + ("..." if len(question) > 80 else "")
            print(f"  [{i}/{len(questions)}] {label}")
            answer = generate(question)
            out.write(f"Q: {question}\n")
            out.write(f"A: {answer}\n")
            out.write("-" * 60 + "\n")

    print(f"\n  Done. Saved to: {output_path}\n")


# ---------------------------------------------------------------------------
# INTERACTIVE LOOP
# ---------------------------------------------------------------------------
print("=" * 60)
print("Hamlet LoRA — Interactive Chat")
print("Type your question and press Enter.")
print("Commands:  'quit' or 'exit' to stop")
print("           'settings' to view/change generation settings")
print("           'file'     to batch-answer questions from a text file")
print("=" * 60)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nFarewell.")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit"):
        print("Farewell.")
        break

    if user_input.lower() == "file":
        run_file_mode()
        continue

    if user_input.lower() == "settings":
        print(f"\n  max_new_tokens    = {MAX_NEW_TOKENS}")
        print(f"  temperature       = {TEMPERATURE}")
        print(f"  top_p             = {TOP_P}")
        print(f"  repetition_penalty= {REPETITION_PENALTY}")
        print()
        continue

    response = generate(user_input)
    print(f"\nHamlet: {response}\n")
