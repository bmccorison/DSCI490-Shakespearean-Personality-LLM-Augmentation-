"""
lora_5.py - Context-aware LoRA fine-tuning for Hamlet from message-style JSON.

Note: this workflow trains directly from speaker-aware message records and does
not run any Shakespeare-to-modern reverse translation during preprocessing.

Run from repo root:
    python training/lora_5.py

Useful smoke test:
    python training/lora_5.py --dry-run --limit 32
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prefer the expandable allocator for this larger context-aware training path.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import full_play_translator
import hamlet_speaker_aware_to_message_style_prompt as message_style_builder
import lora_3 as base_training

DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_EVAL_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUMULATION_STEPS = 8
DEFAULT_GPU_MAX_MEMORY = os.environ.get("HAMLET_GPU_MAX_MEMORY", "6GiB").strip()
DEFAULT_CPU_MAX_MEMORY = os.environ.get(
    "HAMLET_CPU_MAX_MEMORY",
    base_training.CPU_OFFLOAD_MAX_MEMORY,
).strip()


def _resolve_repo_relative_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a context-aware Hamlet LoRA adapter from message-style "
            "speaker-aware JSON."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "data" / "hamlet_speaker_aware_messages.json"),
        help="Path to the message-style speaker-aware JSON file.",
    )
    parser.add_argument(
        "--context-input-file",
        default=str(repo_root / "data" / "hamlet_speaker_aware_context.json"),
        help="Source speaker-aware context JSON used to build the message dataset if needed.",
    )
    parser.add_argument(
        "--full-play-input-file",
        default=str(repo_root / "data" / "hamlet_full_play.txt"),
        help="Fallback full-play text used to rebuild the source context JSON if needed.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "models" / "lora_hamlet_5_3"),
        help="Directory to save the trained LoRA adapter and tokenizer.",
    )
    parser.add_argument(
        "--speaker",
        default="Hamlet",
        help="Target speaker used when auto-building datasets. Default: Hamlet.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of most recent non-target context turns to keep. Default: 4.",
    )
    parser.add_argument(
        "--include-last-speaker-line",
        action="store_true",
        help="Include the most recent prior target-speaker line as assistant context.",
    )
    parser.add_argument(
        "--no-exclude-act5-scene2",
        dest="exclude_act5_scene2",
        action="store_false",
        help="Disable Act 5 Scene 2 exclusion (default: enabled).",
    )
    parser.set_defaults(exclude_act5_scene2=True)
    parser.add_argument(
        "--no-prevent-scene-bleed",
        dest="prevent_scene_bleed",
        action="store_false",
        help="Disable scene bleed prevention (default: enabled).",
    )
    parser.set_defaults(prevent_scene_bleed=True)
    parser.add_argument(
        "--dynamic-system-prompt",
        dest="use_dynamic_system_prompt",
        action="store_true",
        help=(
            "Resolve a relationship-aware system prompt per record based on who "
            "Hamlet is directly addressing (default: disabled)."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=message_style_builder.DEFAULT_SYSTEM_PROMPT,
        help="System anchor to prepend to each message-style training record.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of message records to use. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing and dataset assembly without training.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum token length for each message-style training example. Default: 512.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Per-device training batch size. Default: 1.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Per-device evaluation batch size. Default: 1.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUMULATION_STEPS,
        help="Gradient accumulation steps. Default: 8.",
    )
    parser.add_argument(
        "--offload-folder",
        default="",
        help="Optional directory for auto-dispatch CPU/disk offload state. Defaults to <output-dir>/offload.",
    )
    parser.add_argument(
        "--gpu-max-memory",
        default="",
        help=(
            "Optional hard GPU max_memory budget for accelerate auto-dispatch, "
            'for example "6GiB" or "7000MiB". Applied to each visible CUDA device. '
            f'Default when omitted on CUDA: "{DEFAULT_GPU_MAX_MEMORY}".'
        ),
    )
    parser.add_argument(
        "--cpu-max-memory",
        default="",
        help=(
            "Optional hard CPU max_memory budget for accelerate auto-dispatch, "
            f'for example "16GiB". Default when omitted on CUDA: "{DEFAULT_CPU_MAX_MEMORY}".'
        ),
    )
    return parser.parse_args()


def _coerce_messages(messages_object: object) -> list[dict[str, str]]:
    if isinstance(messages_object, list):
        messages: list[dict[str, str]] = []
        for message in messages_object:
            if not isinstance(message, dict):
                raise ValueError(
                    f"Expected each message to be a JSON object, got {type(message).__name__}."
                )
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            messages.append({"role": role, "content": content})
        return messages

    if isinstance(messages_object, dict):
        roles = messages_object.get("role")
        contents = messages_object.get("content")
        if isinstance(roles, list) and isinstance(contents, list) and len(roles) == len(contents):
            return [
                {"role": str(role).strip(), "content": str(content).strip()}
                for role, content in zip(roles, contents)
            ]

    raise ValueError(
        "Messages must be a list of {role, content} objects or a compatible "
        "dataset-structured representation."
    )


def render_message_history(
    messages_object: object,
    append_assistant_header: bool,
) -> str:
    messages = _coerce_messages(messages_object)
    rendered_parts: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        if not content:
            continue
        rendered_parts.append(f"<|{role}|>\n{content}</s>\n")

    if append_assistant_header:
        rendered_parts.append("<|assistant|>\n")

    return "".join(rendered_parts)


def build_message_style_examples(
    message_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []

    for record in message_records:
        messages = _coerce_messages(record["messages"])
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            continue

        response = messages[-1]["content"].strip()
        if not response:
            continue

        examples.append(
            {
                "messages": messages,
                "response": response,
                "source_kind": "speaker_aware_messages",
                "act": record.get("act"),
                "scene": record.get("scene"),
                "speaker": record.get("speaker"),
                "source_line": record.get("source_line"),
                "context_message_count": max(0, len(messages) - 2),
            }
        )

    return examples


def tokenize_message_style_example(
    example: dict[str, object],
    tokenizer,
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = _coerce_messages(example["messages"])
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_text = render_message_history(messages[:-1], append_assistant_header=True)
    response_text = messages[-1]["content"].strip() + "</s>"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    max_prompt_tokens = max_seq_length - base_training.MIN_RESPONSE_TOKENS
    if len(prompt_ids) > max_prompt_tokens:
        if max_prompt_tokens <= 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        prompt_ids = prompt_ids[:max_prompt_tokens]

    available_response_tokens = max_seq_length - len(prompt_ids)
    if available_response_tokens <= 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    response_ids = response_ids[:available_response_tokens]
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _format_messages_preview(messages_object: object) -> str:
    messages = _coerce_messages(messages_object)
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages
        if message["content"]
    )


def ask_hamlet_from_messages(
    history_messages_object: object,
    tokenizer,
    inference_model,
    device: str,
    max_new_tokens: int = 160,
) -> str:
    prompt = render_message_history(
        history_messages_object,
        append_assistant_header=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with base_training.torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _prepare_model_for_low_memory_kbit_training(model):
    _cleanup_cuda_memory()
    try:
        model = base_training.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        print(
            "Installed PEFT does not support `gradient_checkpointing_kwargs`; "
            "falling back to the legacy k-bit preparation path."
        )
        model = base_training.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    converted = _downcast_fp32_parameters_to_fp16(model)
    _cleanup_cuda_memory()
    if converted:
        print(
            "Downcasted float32 parameters back to float16 immediately after "
            f"k-bit preparation: {converted}"
        )
    return model


def _enable_low_memory_gradient_checkpointing(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def _downcast_fp32_parameters_to_fp16(model) -> int:
    converted = 0
    for param in model.parameters():
        if param.dtype == base_training.torch.float32 and param.device.type != "meta":
            param.data = param.data.to(base_training.torch.float16)
            converted += 1
    return converted


def _cleanup_cuda_memory() -> None:
    if not base_training._cuda_available():
        return
    base_training.torch.cuda.empty_cache()
    if hasattr(base_training.torch.cuda, "ipc_collect"):
        base_training.torch.cuda.ipc_collect()


def _print_cuda_memory_snapshot(label: str) -> None:
    if not base_training._cuda_available():
        return
    allocated_gb = base_training.torch.cuda.memory_allocated() / 1e9
    reserved_gb = base_training.torch.cuda.memory_reserved() / 1e9
    print(f"[CUDA memory] {label}")
    print(f"Allocated: {allocated_gb:.3f} GB")
    print(f"Reserved: {reserved_gb:.3f} GB")
    print(base_training.torch.cuda.memory_summary())


def _build_explicit_max_memory_map(
    gpu_max_memory: str,
    cpu_max_memory: str,
) -> dict[object, str] | None:
    cleaned_gpu_budget = gpu_max_memory.strip()
    cleaned_cpu_budget = cpu_max_memory.strip()
    if not cleaned_gpu_budget and not cleaned_cpu_budget:
        return None

    max_memory: dict[object, str] = {}
    if cleaned_cpu_budget:
        max_memory["cpu"] = cleaned_cpu_budget
    elif cleaned_gpu_budget:
        max_memory["cpu"] = base_training.CPU_OFFLOAD_MAX_MEMORY

    if cleaned_gpu_budget:
        if not base_training._cuda_available():
            raise ValueError("--gpu-max-memory was provided but CUDA is not available.")
        for device_index in range(base_training.torch.cuda.device_count()):
            max_memory[device_index] = cleaned_gpu_budget

    return max_memory if max_memory else None


def main() -> None:
    use_cuda = base_training._cuda_available()
    if use_cuda:
        _cleanup_cuda_memory()

    repo_root = full_play_translator.resolve_repo_root()
    args = parse_args(repo_root)

    if args.k < 0:
        raise ValueError("--k must be 0 or greater.")
    if args.limit < 0:
        raise ValueError("--limit must be 0 or greater.")
    if args.max_seq_length <= base_training.MIN_RESPONSE_TOKENS:
        raise ValueError(
            f"--max-seq-length must be greater than {base_training.MIN_RESPONSE_TOKENS}."
        )
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be greater than 0.")
    if args.per_device_eval_batch_size <= 0:
        raise ValueError("--per-device-eval-batch-size must be greater than 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than 0.")
    if args.gpu_max_memory.strip() and not use_cuda:
        raise ValueError("--gpu-max-memory requires CUDA.")

    output_dir = _resolve_repo_relative_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload_dir = (
        _resolve_repo_relative_path(repo_root, args.offload_folder)
        if args.offload_folder.strip()
        else output_dir / "offload"
    )

    message_records, input_path = message_style_builder.load_or_build_message_records(
        repo_root=repo_root,
        message_input_file=args.input_file,
        context_input_file=args.context_input_file,
        full_play_input_file=args.full_play_input_file,
        speaker=args.speaker,
        k=args.k,
        include_last_speaker_line=args.include_last_speaker_line,
        exclude_act5_scene2=args.exclude_act5_scene2,
        prevent_scene_bleed=args.prevent_scene_bleed,
        system_prompt=args.system_prompt,
        use_dynamic_system_prompt=args.use_dynamic_system_prompt,
        encoding=args.encoding,
    )
    if args.limit:
        message_records = message_records[: args.limit]
    if len(message_records) < 2:
        raise ValueError("Need at least two message records to create train/eval split.")

    print(f"Repository root: {repo_root}")
    print(f"Message-style JSON: {input_path}")
    print(f"LoRA output dir: {output_dir}")
    print(f"CUDA available: {use_cuda}")
    print(f"Message records selected: {len(message_records)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Per-device train batch size: {args.per_device_train_batch_size}")
    print(f"Per-device eval batch size: {args.per_device_eval_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    examples = build_message_style_examples(message_records)
    if len(examples) < 2:
        raise ValueError("Need at least two message-style examples to create train/eval split.")

    print("Sample message-style record:")
    print(_format_messages_preview(examples[0]["messages"])[:800] or "<empty>")
    print("Sample response:", str(examples[0]["response"])[:240] + "...")

    example_dataset = base_training.Dataset.from_list(examples).shuffle(
        seed=base_training.SEED
    )
    eval_size = max(1, int(round(len(example_dataset) * base_training.EVAL_SPLIT)))
    if eval_size >= len(example_dataset):
        raise ValueError("Need at least two message-style examples to create train/eval split.")

    example_split = example_dataset.train_test_split(
        test_size=eval_size,
        seed=base_training.SEED,
    )
    train_dataset = example_split["train"]
    eval_dataset = example_split["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    print("Example training conversation:")
    print(
        _format_messages_preview(train_dataset[0]["messages"])[:800] or "<empty>"
    )

    if args.dry_run:
        print("Dry run complete. Skipping model load and training.")
        return

    if use_cuda:
        model_dtype = base_training.torch.float16
    else:
        model_dtype = base_training.torch.float32

    accelerate_available = base_training.is_accelerate_available()
    bitsandbytes_available = (
        base_training.BitsAndBytesConfig is not None
        and base_training.is_bitsandbytes_available()
    )

    model_load_kwargs: dict[str, object] = {
        "dtype": model_dtype,
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    using_quantization = False
    using_device_map = False
    max_memory_budget = _build_explicit_max_memory_map(
        args.gpu_max_memory,
        args.cpu_max_memory,
    )
    if max_memory_budget is None and use_cuda and accelerate_available:
        max_memory_budget = _build_explicit_max_memory_map(
            DEFAULT_GPU_MAX_MEMORY,
            DEFAULT_CPU_MAX_MEMORY,
        )

    if use_cuda:
        if bitsandbytes_available:
            model_load_kwargs["device_map"] = (
                "auto"
                if accelerate_available
                else {"": base_training._active_cuda_device()}
            )
            using_device_map = True
            model_load_kwargs["quantization_config"] = (
                base_training._build_bnb_quantization_config(
                    load_in_4bit=True,
                    cpu_offload=accelerate_available,
                )
            )
            using_quantization = True
            if accelerate_available:
                offload_dir.mkdir(parents=True, exist_ok=True)
                model_load_kwargs["offload_folder"] = str(offload_dir)
                if max_memory_budget is not None:
                    model_load_kwargs["max_memory"] = max_memory_budget
                print(
                    "Using 4-bit bitsandbytes quantization with accelerate "
                    "auto-dispatch so the loader fills available GPU memory "
                    "before pushing the remainder to CPU."
                )
                print(f"Offload folder: {offload_dir}")
                if max_memory_budget is not None:
                    print(
                        "Auto-dispatch max_memory: "
                        f"{base_training._format_max_memory_map(max_memory_budget)}"
                    )
            else:
                print(
                    "Using 4-bit bitsandbytes quantization on the active CUDA "
                    "device because accelerate auto-dispatch is unavailable."
                )
        else:
            raise RuntimeError(
                "CUDA training in lora_5 requires a working bitsandbytes 4-bit "
                "setup. This script no longer falls back to full-precision or "
                "8-bit CPU-offloaded loading because those paths exceed tight "
                "VRAM budgets. Install or fix bitsandbytes, then retry."
            )
    else:
        print("CUDA not available. Loading CPU model without bitsandbytes quantization.")

    if use_cuda:
        _cleanup_cuda_memory()
        _print_cuda_memory_snapshot("before base model load")

    selected_model_name, tokenizer, model, model_load_kwargs = (
        base_training._load_tokenizer_and_model_with_fallbacks(
            base_training._candidate_base_model_names(),
            model_load_kwargs,
        )
    )
    print(f"Selected base model: {selected_model_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_cuda and not using_device_map and not using_quantization:
        _cleanup_cuda_memory()
        _print_cuda_memory_snapshot("before model.to(cuda)")
        model.to(device="cuda", dtype=base_training.torch.float16)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if using_quantization:
        model = _prepare_model_for_low_memory_kbit_training(model)
    else:
        _cleanup_cuda_memory()
        _enable_low_memory_gradient_checkpointing(model)
        if use_cuda:
            downcasted_params = _downcast_fp32_parameters_to_fp16(model)
            if downcasted_params:
                print(
                    "Downcasted remaining float32 parameters to float16 before "
                    f"LoRA wrapping: {downcasted_params}"
                )

    lora_config = base_training.LoraConfig(
        task_type=base_training.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = base_training.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_train_dataset = train_dataset.map(
        lambda row: tokenize_message_style_example(
            row,
            tokenizer,
            args.max_seq_length,
        ),
        remove_columns=train_dataset.column_names,
    ).filter(lambda row: len(row["input_ids"]) > 0)
    tokenized_eval_dataset = eval_dataset.map(
        lambda row: tokenize_message_style_example(
            row,
            tokenizer,
            args.max_seq_length,
        ),
        remove_columns=eval_dataset.column_names,
    ).filter(lambda row: len(row["input_ids"]) > 0)
    data_collator = base_training.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    print(f"Tokenized train rows: {len(tokenized_train_dataset)}")
    print(f"Tokenized eval rows: {len(tokenized_eval_dataset)}")

    training_args = base_training.TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=base_training.NUM_EPOCHS,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=base_training.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        fp16=use_cuda,
        report_to="none",
        seed=base_training.SEED,
        remove_unused_columns=False,
    )

    if use_cuda:
        _cleanup_cuda_memory()

    trainer = base_training.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    if use_cuda:
        _cleanup_cuda_memory()
        _print_cuda_memory_snapshot("before trainer.train()")

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Training metrics:", train_result.metrics)
    print(f"Saved LoRA adapter + tokenizer to: {output_dir}")

    del trainer
    if use_cuda:
        base_training.torch.cuda.empty_cache()

    inference_model = model
    inference_model.eval()

    device = base_training._model_input_device(inference_model)
    if device == "cpu" or (
        device.startswith("cuda") and not using_device_map and not using_quantization
    ):
        inference_model.to(device)

    sample_count = min(3, len(eval_dataset))
    for index in range(sample_count):
        example = eval_dataset[index]
        messages = _coerce_messages(example["messages"])
        print(f"\nHeld-out example {index + 1}")
        print("Conversation:")
        print(_format_messages_preview(messages[:-1]))
        print("Reference:", example["response"])
        print(
            "Model output:",
            ask_hamlet_from_messages(
                messages[:-1],
                tokenizer,
                inference_model,
                device,
            ),
        )

    freeform_histories = [
        [
            {
                "role": "system",
                "content": args.system_prompt,
            },
            {
                "role": "user",
                "content": "Claudius: Why do you still grieve?",
            },
            {
                "role": "user",
                "content": "Gertrude: All that lives must die.",
            },
        ],
        [
            {
                "role": "system",
                "content": args.system_prompt,
            },
            {
                "role": "user",
                "content": "Horatio: What did the ghost demand of you?",
            },
        ],
    ]
    for history in freeform_histories:
        print(f"\nContext prompt:\n{_format_messages_preview(history)}")
        print(
            "Hamlet:",
            ask_hamlet_from_messages(
                history,
                tokenizer,
                inference_model,
                device,
            ),
        )


if __name__ == "__main__":
    main()
