# Training Scripts

Executable training and preprocessing scripts for the Hamlet LoRA workflows.

## Data sources

- `data/hamlet_onlyhamletraw.txt`
  Hamlet-only raw dialogue in abbreviated speaker format.
- `data/hamlet_full_play.txt`
  Full Folger-formatted play text with speaker blocks, headings, and stage directions.

## Main scripts

### `raw_dialouge_translator.py`

Translates the extracted Hamlet-only raw dialogue into plain English.

Run from repo root:
- `python training/raw_dialouge_translator.py`

What it does:
- extracts Hamlet speeches from `data/hamlet_onlyhamletraw.txt`
- applies the Hamlet-specific regex normalization pass
- runs the reverse translator, with rule-based fallback when the checkpoint is degenerate
- writes one translated speech per paragraph

Default output:
- `data/hamlet_plain_english.txt`

### `full_play_translator.py`

Parses the full Folger play and translates all spoken turns while preserving play structure.

Run from repo root:
- `python training/full_play_translator.py`

What it does:
- parses `ACT`, `Scene`, stage directions, and uppercase speaker blocks
- preserves headings and stage directions in the output text
- sends only spoken turns through the translation pipeline
- writes the translated play back out as plain text

Default output:
- `data/hamlet_full_play_plain_english.txt`

### `speaker_aware_context_filtering.py`

Builds a JSON dataset of target-speaker responses paired with filtered dialogue context.

Run from repo root:
- `python training/speaker_aware_context_filtering.py`
- `python training/speaker_aware_context_filtering.py --k 3 --include-last-speaker-line`

What it does:
- parses the full play through the shared full-play parser
- keeps the last `k` non-target turns as context
- optionally keeps the last prior target-speaker turn
- writes one JSON record per target-speaker response

Default arguments:
- `--speaker Hamlet`
- `--k 3`

Default output:
- `data/hamlet_speaker_aware_context.json`

### `hamlet_speaker_aware_to_message_style_prompt.py`

Converts the speaker-aware context JSON into message-style chat records for LoRA training.

Run from repo root:
- `python training/hamlet_speaker_aware_to_message_style_prompt.py`
- `python training/hamlet_speaker_aware_to_message_style_prompt.py --k 4 --include-last-speaker-line`

What it does:
- loads `data/hamlet_speaker_aware_context.json`
- rebuilds that source JSON from the full play if it is missing or stale
- emits `messages` arrays with a system anchor, optional prior Hamlet assistant line, the last `k` non-Hamlet user lines, and the target Hamlet assistant line
- writes one message-style JSON record per target-speaker response

Default arguments:
- `--speaker Hamlet`
- `--k 4`

Default output:
- `data/hamlet_speaker_aware_messages.json`

### `lora_3.py`

Baseline LoRA workflow using Hamlet-only raw text plus rule-based normalization.

Run from repo root:
- `python training/lora_3.py`

Default output:
- `models/lora_hamlet_3`

### `lora_4.py`

LoRA workflow using Hamlet-only raw text plus the reverse translator.

Run from repo root:
- `python training/lora_4.py`

What it adds over `lora_3.py`:
- reverse-translator plain-English rewriting
- smoke-test fallback when the reverse checkpoint produces repetitive output

Default output:
- `models/lora_hamlet_4`

### `lora_5.py`

Context-aware LoRA workflow using the message-style speaker-aware JSON dataset.

Run from repo root:
- `python training/lora_5.py`

Useful smoke test:
- `python training/lora_5.py --dry-run --limit 32`

What it does:
- loads `data/hamlet_speaker_aware_messages.json`
- auto-builds that message-style JSON from `data/hamlet_speaker_aware_context.json`
- rebuilds stale source and message JSON automatically when the saved speaker-aware settings do not match the requested speaker, `k`, system prompt, or previous-speaker-line flag
- trains a LoRA adapter directly on message-style dialogue history -> Hamlet reply examples
- defaults to a lower-memory training configuration for larger context windows:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 4-bit `nf4` loading with
  `float16` compute, `max_seq_length=256`, batch size `1`, gradient accumulation
  `8`, and default accelerate auto-dispatch caps of `6GiB` GPU plus `48GiB`
  CPU with an offload folder at `<output-dir>/offload`
- requires a working bitsandbytes 4-bit path on CUDA; the script now fails fast
  instead of silently degrading to 8-bit or full-precision loading on tight GPUs
- still lets you override the default auto-dispatch cap with `--gpu-max-memory`
  and `--cpu-max-memory`

Default arguments:
- `--speaker Hamlet`
- `--k 4`

Default output:
- `models/lora_hamlet_5`

## Recommended full-play pipeline

1. Build or inspect the full-play translation:
   `python training/full_play_translator.py`
2. Build the speaker-aware context dataset:
   `python training/speaker_aware_context_filtering.py --k 3 --include-last-speaker-line`
3. Convert it to message-style training records:
   `python training/hamlet_speaker_aware_to_message_style_prompt.py --k 4 --include-last-speaker-line`
4. Smoke-test the contextual training pipeline:
   `python training/lora_5.py --dry-run --limit 32 --include-last-speaker-line`
5. Train the contextual adapter:
   `python training/lora_5.py --include-last-speaker-line`
