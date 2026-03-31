"""
lora_4.py - LoRA roleplay fine-tuning for Hamlet using the reverse translator.

Run from `training/`:
    uv run lora_4.py
"""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent
TRANSLATIONS_DIR = TRAINING_DIR / "translations"

if str(TRANSLATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATIONS_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(1, str(TRAINING_DIR))

import tensorflow as tf

import lora_3 as base_training
import translator as reverse_translator


TRANSLATOR_DEVICE = os.environ.get("HAMLET_TRANSLATOR_DEVICE", "cpu").strip().lower()
TRANSLATOR_MAX_INPUT_TOKENS = int(
    os.environ.get("HAMLET_TRANSLATOR_MAX_INPUT_TOKENS", "96")
)
TRANSLATOR_MAX_OUTPUT_TOKENS = int(
    os.environ.get("HAMLET_TRANSLATOR_MAX_OUTPUT_TOKENS", "160")
)
TRANSLATOR_BEAM_WIDTH = int(os.environ.get("HAMLET_TRANSLATOR_BEAM_WIDTH", "4"))
SPECIAL_TOKEN_RE = re.compile(r"\[(?:BOS|EOS|PAD|UNK)\]|</?s>")


def _path_from_env(name: str) -> Path | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    return Path(raw_value).expanduser().resolve()


def _find_latest_reverse_checkpoint_dir() -> Path | None:
    checkpoint_root = TRANSLATIONS_DIR / "checkpoints" / "train_reverse"
    if not checkpoint_root.is_dir():
        return None

    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir()
        and (path / "checkpoint").is_file()
        and (path / "config.json").is_file()
    ]
    if not candidates:
        return None

    preferred = [path for path in candidates if "smoke" not in path.name.lower()]
    ranked = preferred if preferred else candidates
    return sorted(ranked, key=lambda path: path.name)[-1]


def configure_tensorflow_for_translation() -> None:
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        print("TensorFlow translator device: cpu (no GPU detected)")
        return

    try:
        if TRANSLATOR_DEVICE in {"", "cpu"}:
            tf.config.set_visible_devices([], "GPU")
            print("TensorFlow translator device: cpu (GPU hidden from TensorFlow)")
            return

        if TRANSLATOR_DEVICE in {"gpu", "cuda"}:
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("TensorFlow translator device: gpu (memory growth enabled)")
            return

        raise ValueError(
            "HAMLET_TRANSLATOR_DEVICE must be either 'cpu' or 'gpu'."
        )
    except RuntimeError as error:
        print(
            "TensorFlow device configuration was already initialized; "
            f"continuing with the current device placement: {error}"
        )


def resolve_reverse_translator_paths() -> dict[str, str]:
    checkpoint_dir = _path_from_env("HAMLET_TRANSLATOR_CHECKPOINT_DIR")
    if checkpoint_dir is None:
        checkpoint_dir = _find_latest_reverse_checkpoint_dir()

    config_file = _path_from_env("HAMLET_TRANSLATOR_CONFIG_FILE")
    if config_file is None:
        if checkpoint_dir is not None and (checkpoint_dir / "config.json").is_file():
            config_file = checkpoint_dir / "config.json"
        else:
            config_file = TRANSLATIONS_DIR / "trained_models" / "model_config.json"

    weights_path = _path_from_env("HAMLET_TRANSLATOR_WEIGHTS_PATH")
    if weights_path is None:
        weights_path = TRANSLATIONS_DIR / "trained_models" / "translator_weights"

    inp_sp_model_file = _path_from_env("HAMLET_TRANSLATOR_INPUT_TOKENIZER")
    if inp_sp_model_file is None:
        inp_sp_model_file = TRANSLATIONS_DIR / "tokenizers" / "original2k.model"

    tar_sp_model_file = _path_from_env("HAMLET_TRANSLATOR_TARGET_TOKENIZER")
    if tar_sp_model_file is None:
        tar_sp_model_file = TRANSLATIONS_DIR / "tokenizers" / "modern2k.model"

    if not config_file.is_file():
        raise FileNotFoundError(f"Reverse translator config not found: {config_file}")
    if not inp_sp_model_file.is_file():
        raise FileNotFoundError(
            f"Reverse translator input tokenizer not found: {inp_sp_model_file}"
        )
    if not tar_sp_model_file.is_file():
        raise FileNotFoundError(
            f"Reverse translator target tokenizer not found: {tar_sp_model_file}"
        )
    if checkpoint_dir is None and not Path(f"{weights_path}.index").is_file():
        raise FileNotFoundError(
            "Reverse translator weights were not found and no checkpoint directory "
            f"was available. Expected either {weights_path}.index or a checkpoint."
        )

    return {
        "config_file": str(config_file),
        "weights_path": str(weights_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
        "inp_sp_model_file": str(inp_sp_model_file),
        "tar_sp_model_file": str(tar_sp_model_file),
    }


def load_reverse_translator():
    configure_tensorflow_for_translation()
    paths = resolve_reverse_translator_paths()

    print("Loading reverse translator...")
    print(f"Translator config: {paths['config_file']}")
    if paths["checkpoint_dir"]:
        print(f"Translator checkpoint dir: {paths['checkpoint_dir']}")
    else:
        print(f"Translator weights prefix: {paths['weights_path']}")

    model, _ = reverse_translator.load_model(
        config_file=paths["config_file"],
        weights_path=paths["weights_path"],
        checkpoint_dir=paths["checkpoint_dir"],
    )
    inp_tokenizer, tar_tokenizer = reverse_translator.load_tokenizers(
        inp_file=paths["inp_sp_model_file"],
        tar_file=paths["tar_sp_model_file"],
    )
    return model, inp_tokenizer, tar_tokenizer


def _token_count(text: str, tokenizer) -> int:
    return int(tf.size(tokenizer.tokenize(text)).numpy())


def _split_segment_by_words(
    segment: str,
    tokenizer,
    max_input_tokens: int,
) -> list[str]:
    words = segment.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate_words = [*current_words, word]
        candidate_text = " ".join(candidate_words)
        if current_words and _token_count(candidate_text, tokenizer) > max_input_tokens:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words = candidate_words

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def segment_text_for_translation(text: str, tokenizer) -> list[str]:
    cleaned = base_training._clean_text(text)
    if not cleaned:
        return []

    segments: list[str] = []
    for sentence in base_training.split_sentences(cleaned):
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            candidate = clause.strip()
            if not candidate:
                continue

            if _token_count(candidate, tokenizer) <= TRANSLATOR_MAX_INPUT_TOKENS:
                segments.append(candidate)
                continue

            segments.extend(
                _split_segment_by_words(
                    candidate,
                    tokenizer,
                    TRANSLATOR_MAX_INPUT_TOKENS,
                )
            )

    return segments if segments else [cleaned]


def clean_translated_text(text: str) -> str:
    cleaned = SPECIAL_TOKEN_RE.sub(" ", text)
    cleaned = cleaned.replace("\u2014", "-").replace("\u2019", "'")
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return base_training.WHITESPACE_RE.sub(" ", cleaned).strip()


def translate_segment(
    text: str,
    model,
    inp_tokenizer,
    tar_tokenizer,
    cache: dict[str, str],
) -> str:
    normalized = base_training._clean_text(text)
    if not normalized:
        return ""

    cached = cache.get(normalized)
    if cached is not None:
        return cached

    token_budget = _token_count(normalized, inp_tokenizer)
    max_output_tokens = min(
        TRANSLATOR_MAX_OUTPUT_TOKENS,
        max(32, token_budget * 2 + 16),
    )

    try:
        translations, _ = reverse_translator.eager_beam_search(
            normalized,
            inp_tokenizer,
            model,
            K=max(1, TRANSLATOR_BEAM_WIDTH),
            maxlen=max_output_tokens,
        )
        translated = tar_tokenizer.detokenize(translations[0]).numpy().decode("utf-8")
        translated = clean_translated_text(translated)
    except Exception as error:
        print(
            "Reverse translator failed on one segment; using the original text "
            f"instead. Segment: {normalized[:120]!r}. Error: {error}"
        )
        translated = normalized

    cache[normalized] = translated or normalized
    return cache[normalized]


def translate_speeches_with_reverse_model(
    speeches: list[str],
    model,
    inp_tokenizer,
    tar_tokenizer,
) -> list[str]:
    cache: dict[str, str] = {}
    translated_speeches: list[str] = []
    total_segments = 0

    for index, speech in enumerate(speeches, start=1):
        segments = segment_text_for_translation(speech, inp_tokenizer)
        total_segments += len(segments)
        translated_segments = [
            translate_segment(segment, model, inp_tokenizer, tar_tokenizer, cache)
            for segment in segments
        ]
        translated_speech = clean_translated_text(" ".join(translated_segments))
        translated_speeches.append(translated_speech or base_training._clean_text(speech))

        if index == 1 or index % 25 == 0 or index == len(speeches):
            print(
                f"Translated speeches: {index}/{len(speeches)} | "
                f"segments processed: {total_segments} | "
                f"unique translated segments: {len(cache)}"
            )

    return translated_speeches


def release_reverse_translator() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def main() -> None:
    use_cuda = base_training._cuda_available()
    if use_cuda:
        base_training.torch.cuda.empty_cache()

    repo_root = base_training.find_repo_root()
    raw_text_path = repo_root / "data" / "hamlet_onlyhamletraw.txt"
    output_dir = repo_root / "models" / "lora_hamlet_4"

    if not raw_text_path.is_file():
        raise FileNotFoundError(f"Missing raw text file: {raw_text_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {repo_root}")
    print(f"Raw Hamlet text: {raw_text_path}")
    print(f"LoRA output dir: {output_dir}")
    print(f"CUDA available: {use_cuda}")

    hamlet_speeches = base_training.extract_hamlet_speeches(
        raw_text_path.read_text(encoding="utf-8").splitlines()
    )
    if not hamlet_speeches:
        raise ValueError("No Hamlet speeches extracted. Check parser and input format.")

    average_words = sum(len(speech.split()) for speech in hamlet_speeches) / len(
        hamlet_speeches
    )
    print(f"Extracted Hamlet speeches: {len(hamlet_speeches)}")
    print(f"Average speech length: {average_words:.1f} words")
    print("Sample extracted speech:")
    print(hamlet_speeches[0][:260] + "...")

    reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer = load_reverse_translator()
    plain_english_speeches = translate_speeches_with_reverse_model(
        hamlet_speeches,
        reverse_model,
        reverse_inp_tokenizer,
        reverse_tar_tokenizer,
    )
    del reverse_model, reverse_inp_tokenizer, reverse_tar_tokenizer
    release_reverse_translator()

    changed_flags = [
        source.lower() != target.lower()
        for source, target in zip(hamlet_speeches, plain_english_speeches)
    ]

    normalized_corpus = base_training.Dataset.from_dict(
        {
            "original_text": hamlet_speeches,
            "plain_english_text": plain_english_speeches,
            "changed": changed_flags,
        }
    )
    changed_examples = normalized_corpus.filter(lambda row: row["changed"])

    print(f"Total translated speeches: {len(normalized_corpus)}")
    print(f"Speeches changed by translator: {len(changed_examples)}")
    example_row = (
        changed_examples[0] if len(changed_examples) > 0 else normalized_corpus[0]
    )
    print("Original speech:", example_row["original_text"][:200] + "...")
    print("Plain English speech:", example_row["plain_english_text"][:200] + "...")

    speech_corpus = normalized_corpus.shuffle(seed=base_training.SEED)
    eval_size = max(1, int(round(len(speech_corpus) * base_training.EVAL_SPLIT)))
    if eval_size >= len(speech_corpus):
        raise ValueError("Need at least two speeches to create train/eval split.")

    speech_split = speech_corpus.train_test_split(
        test_size=eval_size,
        seed=base_training.SEED,
    )
    train_speech_dataset = speech_split["train"]
    eval_speech_dataset = speech_split["test"]

    train_dataset = base_training.Dataset.from_list(
        base_training.build_roleplay_examples(train_speech_dataset)
    )
    eval_dataset = base_training.Dataset.from_list(
        base_training.build_roleplay_examples(eval_speech_dataset)
    )

    print(f"Training speeches: {len(train_speech_dataset)}")
    print(f"Evaluation speeches: {len(eval_speech_dataset)}")
    print(f"Training roleplay examples: {len(train_dataset)}")
    print(f"Evaluation roleplay examples: {len(eval_dataset)}")
    print("Example training pair:")
    print("Prompt:", train_dataset[0]["instruction"])
    print("Response:", train_dataset[0]["response"][:400] + "...")

    if use_cuda and base_training.torch.cuda.is_bf16_supported():
        model_dtype = base_training.torch.bfloat16
    elif use_cuda:
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
        "low_cpu_mem_usage": True,
    }
    using_quantization = False
    using_device_map = False
    auto_max_memory = (
        base_training._cuda_max_memory_map()
        if use_cuda and accelerate_available
        else None
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
                if auto_max_memory is not None:
                    model_load_kwargs["max_memory"] = auto_max_memory
                print(
                    "Using 4-bit bitsandbytes quantization with accelerate "
                    "auto-dispatch so the loader fills available GPU memory "
                    "before pushing the remainder to CPU."
                )
                if auto_max_memory is not None:
                    print(
                        "Auto-dispatch max_memory: "
                        f"{base_training._format_max_memory_map(auto_max_memory)}"
                    )
            else:
                print(
                    "Using 4-bit bitsandbytes quantization on the active CUDA "
                    "device because accelerate auto-dispatch is unavailable."
                )
        else:
            if accelerate_available:
                model_load_kwargs["device_map"] = "auto"
                using_device_map = True
                if auto_max_memory is not None:
                    model_load_kwargs["max_memory"] = auto_max_memory
                    print(
                        "Auto-dispatch max_memory: "
                        f"{base_training._format_max_memory_map(auto_max_memory)}"
                    )
            else:
                print(
                    "accelerate is missing or below the Transformers minimum version. "
                    "Loading without `device_map`; "
                    "automatic offload/sharding is disabled."
                )
            print(
                "bitsandbytes is missing or incompatible with this Transformers/CUDA "
                "setup. Loading full-precision CUDA model."
            )
    else:
        print("CUDA not available. Loading CPU model without bitsandbytes quantization.")

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
        model.to("cuda")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if using_quantization:
        model = base_training.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

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
        lambda row: base_training.tokenize_supervised_example(row, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda row: base_training.tokenize_supervised_example(row, tokenizer),
        remove_columns=eval_dataset.column_names,
    )
    data_collator = base_training.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    print(f"Tokenized train rows: {len(tokenized_train_dataset)}")
    print(f"Tokenized eval rows: {len(tokenized_eval_dataset)}")

    bf16_ok = use_cuda and base_training.torch.cuda.is_bf16_supported()
    training_args = base_training.TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=base_training.NUM_EPOCHS,
        per_device_train_batch_size=base_training.BATCH_SIZE,
        per_device_eval_batch_size=base_training.BATCH_SIZE,
        gradient_accumulation_steps=base_training.GRAD_ACCUM_STEPS,
        learning_rate=base_training.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=use_cuda and not bf16_ok,
        report_to="none",
        seed=base_training.SEED,
        remove_unused_columns=False,
    )

    trainer = base_training.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

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
        print(f"\nHeld-out example {index + 1}")
        print("Prompt:", example["instruction"])
        print("Reference:", example["response"])
        print(
            "Model output:",
            base_training.ask_hamlet(
                example["instruction"],
                tokenizer,
                inference_model,
                device,
            ),
        )

    freeform_questions = [
        "How do you feel about your father's death?",
        "What do you think of Claudius?",
        "Why do you hesitate when revenge lies before you?",
    ]
    for question in freeform_questions:
        print(f"\nFree-form question: {question}")
        print(
            "Hamlet:",
            base_training.ask_hamlet(question, tokenizer, inference_model, device),
        )


if __name__ == "__main__":
    main()
