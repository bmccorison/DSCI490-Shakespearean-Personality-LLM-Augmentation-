"""
lora_2.py  —  LoRA fine-tuning on character_profile_hamlet.json

Run from the repo root:
    python training/lora_2.py

Or from the training/ directory:
    python lora_2.py
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECKS — fail fast before loading any heavy libraries
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DATA_PATH   = os.path.join(REPO_ROOT, "data", "character_profile_hamlet.json")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "models", "lora_hamlet_profile")

print("=" * 60)
print("PRE-FLIGHT CHECKS")
print("=" * 60)

# 1. Python version
print(f"[1] Python: {sys.version}")
assert sys.version_info >= (3, 9), "Python 3.9+ required"
print("    OK")

# 2. Data file
print(f"[2] Data file: {DATA_PATH}")
assert os.path.isfile(DATA_PATH), f"DATA FILE NOT FOUND: {DATA_PATH}"
print("    OK")

# 3. Required packages
required = ["torch", "transformers", "peft", "datasets", "accelerate"]
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"[3] {pkg}: OK")
    except ImportError:
        missing.append(pkg)
        print(f"[3] {pkg}: MISSING")

if missing:
    print(f"\nInstall missing packages with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

# 4. GPU / device info
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[4] Device: {device}")
if device == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 5. Output directory (create if needed)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[5] Output dir: {OUTPUT_DIR}  OK")

print("=" * 60)
print("All checks passed — starting training")
print("=" * 60)

# ---------------------------------------------------------------------------
# IMPORTS (deferred until after checks)
# ---------------------------------------------------------------------------
from peft import LoraConfig, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
SYSTEM_PROMPT = (
    "You are Hamlet, Prince of Denmark. You speak with philosophical depth, "
    "melancholic wit, and introspective honesty. Answer as Hamlet would."
)

# ---------------------------------------------------------------------------
# 1. LOAD CHARACTER PROFILE
# ---------------------------------------------------------------------------
print("\n[Step 1] Loading character profile...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    profile = json.load(f)

print(f"  Character : {profile['character']}")
print(f"  Keys      : {list(profile.keys())}")

# ---------------------------------------------------------------------------
# 2. BUILD INSTRUCTION–RESPONSE PAIRS
# ---------------------------------------------------------------------------
print("\n[Step 2] Building training pairs...")
pairs = []

# Background
pairs.append((
    "Who is Hamlet, and what is his background?",
    profile["background"].replace("\n", " ").strip()
))

# Core traits
for trait in profile["core_traits"]:
    pairs.append((
        f"Describe Hamlet's {trait['name'].lower()} trait.",
        trait["description"]
    ))
    pairs.append((
        f"How does Hamlet exhibit {trait['name'].lower()} behavior?",
        trait["description"]
    ))

# Internal conflicts
for conflict in profile["key_internal_conflicts"]:
    pairs.append((
        f"Explain Hamlet's internal conflict regarding {conflict['name'].lower()}.",
        conflict["description"]
    ))

# Relationships
for rel in profile["relationships"]:
    pairs.append((
        f"How does Hamlet relate to {rel['character']}?",
        rel["description"]
    ))
    pairs.append((
        f"What is {rel['character']}'s role in Hamlet's story?",
        f"{rel['role']}. {rel['description']}"
    ))

# Psychological arc & endurance
pairs.append((
    "Describe Hamlet's psychological arc throughout the play.",
    profile["psychological_arc"].replace("\n", " ").strip()
))
pairs.append((
    "Why does Hamlet endure as a literary figure?",
    profile["why_hamlet_endures"].replace("\n", " ").strip()
))

# Character analysis summary
pairs.append((
    "Provide a detailed character analysis of Hamlet.",
    profile["character_analysis"]["summary"].replace("\n", " ").strip()
))

# Bullet facts
for bullet in profile["character_analysis"]["bullets"]:
    pairs.append(("State one key fact about Hamlet.", bullet))

# First-person roleplay
pairs.extend([
    (
        "How do you feel about your father's death?",
        "My father's death weighs upon me like a cloak of lead I cannot shed. "
        "The world, once full of promise, has turned into an unweeded garden — "
        "rank and gross in nature. I cannot simply cast off grief as others counsel."
    ),
    (
        "What do you think of Claudius?",
        "He is the embodiment of corruption — a smiling villain who poisoned "
        "not only my father but the very state of Denmark itself. "
        "To me he is Hyperion to a satyr; my father to this."
    ),
    (
        "To be, or not to be — what do you mean by that?",
        "I wrestle with the endurance of suffering against the fear of what lies beyond death. "
        "Whether it is nobler to bear the slings and arrows of outrageous fortune, "
        "or to take arms against them — that is the question I cannot cease asking."
    ),
    (
        "Why do you delay your revenge?",
        "It is not cowardice but conscience. I must be certain — beyond the whisper of a ghost — "
        "that Claudius is guilty. To act rashly is to become the very evil I seek to correct."
    ),
])

print(f"  Total pairs: {len(pairs)}")

# ---------------------------------------------------------------------------
# 3. FORMAT WITH TINYLLAMA CHAT TEMPLATE
# ---------------------------------------------------------------------------
def format_chat(instruction: str, response: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{instruction}</s>\n"
        f"<|assistant|>\n{response}</s>"
    )

formatted_texts = [format_chat(q, a) for q, a in pairs]
print(f"\n[Step 3] Sample formatted text:\n{formatted_texts[0][:200]}...")

# ---------------------------------------------------------------------------
# 4. LOAD TOKENIZER & MODEL
# ---------------------------------------------------------------------------
print(f"\n[Step 4] Loading tokenizer & model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------------
# 5. ATTACH LORA ADAPTER
# ---------------------------------------------------------------------------
print("\n[Step 5] Attaching LoRA adapter...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,           # Reduced from 8 — smaller adapter, less impact on base model
    lora_alpha=8,  # Reduced from 32 — effective scaling = alpha/r = 2.0 (was 4.0)
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model.add_adapter(lora_config, adapter_name="hamlet_adapter")
model.set_adapter("hamlet_adapter")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ---------------------------------------------------------------------------
# 6. TOKENIZE DATASET
# ---------------------------------------------------------------------------
print("\n[Step 6] Tokenizing dataset...")
raw_dataset = Dataset.from_dict({"text": formatted_texts})

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(f"  Dataset size: {len(tokenized_dataset)} samples")

# ---------------------------------------------------------------------------
# 7. TRAIN
# ---------------------------------------------------------------------------
print("\n[Step 7] Training...")
BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,       # Increased from 5
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,        # Slightly lower LR for the longer run
    lr_scheduler_type="cosine",
    warmup_steps=20,           # Longer warmup to match increased steps
    logging_steps=10,
    save_strategy="epoch",
    bf16=BF16_SUPPORTED,       # TinyLlama uses bfloat16 weights — use bf16 scaler
    fp16=False,                # fp16 scaler is incompatible with bfloat16 weights
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

# ---------------------------------------------------------------------------
# 8. SAVE ADAPTER
# ---------------------------------------------------------------------------
print("\n[Step 8] Saving adapter...")
model.save_pretrained(OUTPUT_DIR, adapter_name="hamlet_adapter")
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 9. INFERENCE TEST
# ---------------------------------------------------------------------------
print("\n[Step 9] Running inference test...")

base_model    = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
inf_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
inf_tokenizer.pad_token = inf_tokenizer.eos_token

inf_model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR,
    adapter_name="hamlet_adapter"
)
inf_model.eval()

def ask_hamlet(question: str, max_new_tokens: int = 120) -> str:
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = inf_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = inf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=inf_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return inf_tokenizer.decode(new_tokens, skip_special_tokens=True)

test_questions = [
    "How do you feel about your father's death?",
    "Describe your relationship with Ophelia.",
    "What is your greatest internal struggle?",
]

print()
for q in test_questions:
    print(f"Q: {q}")
    print(f"A: {ask_hamlet(q)}")
    print("-" * 60)

print("\nDone.")
