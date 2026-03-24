# Training Scripts & Notebooks

Any fine-tuning methods used when adapting the local LLMs.

## `lora_3.ipynb`

Notebook-based LoRA training workflow built around raw Hamlet text at:
- `data/hamlet_onlyhamletraw.txt`

This notebook uses HuggingFace + PEFT and keeps the full preprocessing,
dataset-building, training, and inference workflow in one place.

### What it includes
- Raw text parsing to extract Hamlet speeches from speaker-formatted play text.
- A dedicated separate cell for Shakespearean-to-plain-English normalization.
- Roleplay dataset construction *after* normalization so the adapter still learns
  to answer as Hamlet rather than learning a translation task.
- Assistant-only supervised fine-tuning with LoRA.
- A final sanity-check cell that tests both held-out prompts and normal Hamlet
  questions.

### Output

Adapter output path after running training cells:
- `models/lora_hamlet_plain_roleplay`
