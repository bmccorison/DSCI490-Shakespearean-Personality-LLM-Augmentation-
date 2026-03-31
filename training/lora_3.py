"""
lora_3.py - LoRA roleplay fine-tuning for Hamlet using raw play text.

Run from repo root:
    python training/lora_3.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# Prefer CUDA's async allocator unless the caller already configured one.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    modeling_utils as transformers_modeling_utils,
)
from transformers.utils import is_accelerate_available, is_bitsandbytes_available

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


BASE_MODEL_NAME = os.environ.get("HAMLET_BASE_MODEL", "LiquidAI/LFM2-8B-A1B")
FALLBACK_BASE_MODEL_NAME = os.environ.get(
    "HAMLET_FALLBACK_BASE_MODEL",
    "LiquidAI/LFM2-2.6B",
)
MAX_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
EVAL_SPLIT = 0.1
SEED = 42
MIN_WORDS_PER_SPEECH = 4
FORCE_HF_DOWNLOAD = os.environ.get("HAMLET_FORCE_HF_DOWNLOAD", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CPU_OFFLOAD_MAX_MEMORY = os.environ.get("HAMLET_CPU_OFFLOAD_MAX_MEMORY", "48GiB")
CUDA_MEMORY_RESERVE_GIB = float(
    os.environ.get("HAMLET_CUDA_MEMORY_RESERVE_GIB", "0.0")
)

SYSTEM_PROMPT_ROLEPLAY = (
    "You are Hamlet, Prince of Denmark. Speak in clear modern English while "
    "preserving Hamlet's introspection, melancholy, philosophical wit, and "
    "moral tension. Stay in character."
)

SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9'_-]*)\.\s*(.*)$")
STAGE_DIRECTION_RE = re.compile(r"\[[^\]]+\]")
WHITESPACE_RE = re.compile(r"\s+")

TRANSLATION_RULES: list[tuple[str, str]] = [
    (r"(?<![A-Za-z])i\s+prithee(?![A-Za-z])", "please"),
    (r"(?<![A-Za-z])prithee(?![A-Za-z])", "please"),
    (r"(?<![A-Za-z])methinks(?![A-Za-z])", "I think"),
    (r"(?<![A-Za-z])wherefore(?![A-Za-z])", "why"),
    (r"(?<![A-Za-z])ere\s+yet(?![A-Za-z])", "before"),
    (r"(?<![A-Za-z])'tis(?![A-Za-z])", "it is"),
    (r"(?<![A-Za-z])'twas(?![A-Za-z])", "it was"),
    (r"(?<![A-Za-z])thou(?![A-Za-z])", "you"),
    (r"(?<![A-Za-z])thee(?![A-Za-z])", "you"),
    (r"(?<![A-Za-z])thy(?![A-Za-z])", "your"),
    (r"(?<![A-Za-z])thine(?![A-Za-z])", "yours"),
    (r"(?<![A-Za-z])art(?![A-Za-z])", "are"),
    (r"(?<![A-Za-z])dost(?![A-Za-z])", "do"),
    (r"(?<![A-Za-z])doth(?![A-Za-z])", "does"),
    (r"(?<![A-Za-z])hast(?![A-Za-z])", "have"),
    (r"(?<![A-Za-z])hath(?![A-Za-z])", "has"),
    (r"(?<![A-Za-z])wilt(?![A-Za-z])", "will"),
    (r"(?<![A-Za-z])shalt(?![A-Za-z])", "shall"),
    (r"(?<![A-Za-z])canst(?![A-Za-z])", "can"),
    (r"(?<![A-Za-z])couldst(?![A-Za-z])", "could"),
    (r"(?<![A-Za-z])wouldst(?![A-Za-z])", "would"),
    (r"(?<![A-Za-z])shouldst(?![A-Za-z])", "should"),
    (r"(?<![A-Za-z])mayst(?![A-Za-z])", "may"),
    (r"(?<![A-Za-z])ere(?![A-Za-z])", "before"),
    (r"(?<![A-Za-z])whilst(?![A-Za-z])", "while"),
    (r"(?<![A-Za-z])ne'er(?![A-Za-z])", "never"),
    (r"(?<![A-Za-z])o'er(?![A-Za-z])", "over"),
    (r"(?<![A-Za-z])e'en(?![A-Za-z])", "even"),
    (r"i'\s*th'", "in the"),
]

TOPIC_PROMPT_RULES: list[tuple[str, re.Pattern[str], list[str]]] = [
    (
        "father",
        re.compile(r"\b(father|ghost|murder|murdered|memory)\b", re.IGNORECASE),
        [
            "How does your father's death weigh on you?",
            "What rises in you when you remember your father?",
        ],
    ),
    (
        "claudius",
        re.compile(r"\b(claudius|uncle|king)\b", re.IGNORECASE),
        [
            "What do you think of Claudius?",
            "How do you see the king who now rules Denmark?",
        ],
    ),
    (
        "gertrude",
        re.compile(r"\b(mother|queen|woman|gertrude)\b", re.IGNORECASE),
        [
            "How do you feel about your mother?",
            "What troubles you when you think of the queen?",
        ],
    ),
    (
        "ophelia",
        re.compile(r"\b(ophelia|love|lover|beauty|nunnery)\b", re.IGNORECASE),
        [
            "How do you think about Ophelia?",
            "What place does love have in your mind right now?",
        ],
    ),
    (
        "revenge",
        re.compile(
            r"\b(revenge|act|action|conscience|delay|hesitate)\b",
            re.IGNORECASE,
        ),
        [
            "Why do you hesitate when action is demanded of you?",
            "What stands between you and revenge?",
        ],
    ),
    (
        "mortality",
        re.compile(
            r"\b(death|dead|die|dying|grave|skull|dust|sleep|dream)\b",
            re.IGNORECASE,
        ),
        [
            "How do you think about death and what may follow it?",
            "What do thoughts of mortality do to your mind?",
        ],
    ),
]

DEFAULT_PROMPTS = [
    "What is on your mind at this moment?",
    "Speak your thoughts plainly.",
    "How do you see the world around you right now?",
]


def find_repo_root() -> Path:
    """Locate repo root when running from root or from training/."""
    search_roots: list[Path] = []
    if "__file__" in globals():
        search_roots.append(Path(__file__).resolve().parent)
    search_roots.append(Path.cwd())

    for start in search_roots:
        for candidate in (start, *start.parents):
            if (candidate / "data" / "hamlet_onlyhamletraw.txt").is_file():
                return candidate

    raise FileNotFoundError(
        "Could not find repository root. Run from repo root or training/."
    )


def _clean_text(text: str) -> str:
    text = STAGE_DIRECTION_RE.sub(" ", text)
    text = text.replace("\u2014", "-").replace("\u2019", "'")
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_hamlet_speeches(lines: Iterable[str]) -> list[str]:
    speeches: list[str] = []
    current_speech: list[str] = []
    collecting_hamlet = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        speaker_match = SPEAKER_PREFIX_RE.match(line)

        if speaker_match:
            if collecting_hamlet and current_speech:
                speeches.append(_clean_text(" ".join(current_speech)))

            speaker = speaker_match.group(1).lower()
            first_text = _clean_text(speaker_match.group(2))
            collecting_hamlet = speaker.startswith("ham")
            current_speech = [first_text] if collecting_hamlet and first_text else []
            continue

        if collecting_hamlet:
            cleaned = _clean_text(line)
            if cleaned:
                current_speech.append(cleaned)

    if collecting_hamlet and current_speech:
        speeches.append(_clean_text(" ".join(current_speech)))

    return [speech for speech in speeches if len(speech.split()) >= MIN_WORDS_PER_SPEECH]


def _match_case(source_text: str, replacement: str) -> str:
    letters = [char for char in source_text if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def replace_case_aware(text: str, pattern: str, replacement: str) -> str:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    return compiled.sub(
        lambda match: _match_case(match.group(0), replacement),
        text,
    )


def shakespeare_to_plain_english(text: str) -> str:
    normalized = text
    for pattern, replacement in TRANSLATION_RULES:
        normalized = replace_case_aware(normalized, pattern, replacement)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def choose_roleplay_prompt(text: str, index: int) -> tuple[str, str]:
    for topic, pattern, prompts in TOPIC_PROMPT_RULES:
        if pattern.search(text):
            return prompts[index % len(prompts)], topic
    return DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)], "general"


def split_sentences(text: str) -> list[str]:
    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", text)
        if segment.strip()
    ]
    return sentences if sentences else [text.strip()]


def build_roleplay_examples(speech_rows: Dataset) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for index, row in enumerate(speech_rows):
        speech = row["plain_english_text"]
        prompt, topic = choose_roleplay_prompt(speech, index)
        examples.append(
            {
                "instruction": prompt,
                "response": speech,
                "topic": topic,
                "source_kind": "full_speech",
            }
        )

        sentences = split_sentences(speech)
        if len(sentences) >= 2:
            opening = sentences[0]
            continuation = " ".join(sentences[1:]).strip()
            if len(continuation.split()) >= MIN_WORDS_PER_SPEECH:
                examples.append(
                    {
                        "instruction": (
                            "Continue this thought as Hamlet in clear modern English:\n"
                            f"{opening}"
                        ),
                        "response": continuation,
                        "topic": topic,
                        "source_kind": "continuation",
                    }
                )
    return examples


def format_roleplay_prompt(instruction: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT_ROLEPLAY}</s>\n"
        f"<|user|>\n{instruction}</s>\n"
        "<|assistant|>\n"
    )


def tokenize_supervised_example(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
) -> dict[str, list[int]]:
    prompt_text = format_roleplay_prompt(example["instruction"])
    response_text = example["response"] + "</s>"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    available_response_tokens = MAX_LENGTH - len(prompt_ids)
    if available_response_tokens <= 0:
        raise ValueError("Prompt length exceeded MAX_LENGTH before adding response.")

    response_ids = response_ids[:available_response_tokens]
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_roleplay_prompt(question: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT_ROLEPLAY}</s>\n"
        f"<|user|>\n{question}</s>\n"
        "<|assistant|>\n"
    )


def ask_hamlet(
    question: str,
    tokenizer: AutoTokenizer,
    inference_model: PeftModel,
    device: str,
    max_new_tokens: int = 160,
) -> str:
    prompt = build_roleplay_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
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


def _cuda_available() -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return False
    return bool(cuda.is_available())


def _active_cuda_device() -> int:
    return int(torch.cuda.current_device())


def _cuda_max_memory_map() -> dict[object, str] | None:
    if not _cuda_available() or not hasattr(torch.cuda, "mem_get_info"):
        return None

    reserve_bytes = int(CUDA_MEMORY_RESERVE_GIB * (1024**3))
    max_memory: dict[object, str] = {"cpu": CPU_OFFLOAD_MAX_MEMORY}

    for device_index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        budget_bytes = max(0, free_bytes - reserve_bytes)
        if budget_bytes <= 0:
            continue

        budget_mib = max(1, budget_bytes // (1024**2))
        max_memory[device_index] = f"{budget_mib}MiB"

    return max_memory if len(max_memory) > 1 else None


def _format_max_memory_map(max_memory: dict[object, str]) -> str:
    entries: list[str] = []
    for device, budget in max_memory.items():
        label = f"cuda:{device}" if isinstance(device, int) else str(device)
        entries.append(f"{label}={budget}")
    return ", ".join(entries)


def _candidate_base_model_names() -> list[str]:
    candidates = [BASE_MODEL_NAME, FALLBACK_BASE_MODEL_NAME]
    deduped_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped_candidates:
            deduped_candidates.append(candidate)
    return deduped_candidates


def _build_bnb_quantization_config(
    *,
    load_in_4bit: bool,
    cpu_offload: bool,
) -> BitsAndBytesConfig:
    config_kwargs: dict[str, object] = {
        "llm_int8_enable_fp32_cpu_offload": cpu_offload,
    }
    if load_in_4bit:
        config_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16,
            }
        )
    else:
        config_kwargs["load_in_8bit"] = True
    return BitsAndBytesConfig(**config_kwargs)


def _model_input_device(model: torch.nn.Module) -> str:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        return str(input_embeddings.weight.device)

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _is_retryable_params4bit_dispatch_error(error: TypeError) -> bool:
    message = str(error)
    return (
        "Params4bit.__new__()" in message
        and "_is_hf_initialized" in message
    )


def _is_retryable_nvml_allocator_warmup_error(error: RuntimeError) -> bool:
    message = str(error)
    return (
        "NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_()" in message
        and "CUDACachingAllocator.cpp" in message
    )


def _is_retryable_bnb_cpu_offload_error(error: ValueError) -> bool:
    message = str(error)
    return (
        "Some modules are dispatched on the CPU or the disk." in message
        and "llm_int8_enable_fp32_cpu_offload=True" in message
    )


def _install_transformers_nvml_warmup_guard() -> None:
    if getattr(transformers_modeling_utils, "_hamlet_nvml_warmup_guard_installed", False):
        return

    original_warmup = transformers_modeling_utils.caching_allocator_warmup

    def _guarded_caching_allocator_warmup(model, expanded_device_map, hf_quantizer):
        try:
            return original_warmup(model, expanded_device_map, hf_quantizer)
        except RuntimeError as error:
            if not _is_retryable_nvml_allocator_warmup_error(error):
                raise
            print(
                "Skipping Transformers CUDA allocator warmup because "
                "PyTorch could not initialize NVML on this machine."
            )
            return None

    transformers_modeling_utils.caching_allocator_warmup = _guarded_caching_allocator_warmup
    transformers_modeling_utils._hamlet_nvml_warmup_guard_installed = True


def _load_causal_lm_with_quantized_dispatch_fallback(
    model_name: str,
    model_load_kwargs: dict[str, object],
) -> tuple[torch.nn.Module, dict[str, object]]:
    if _cuda_available() and model_load_kwargs.get("device_map") is not None:
        _install_transformers_nvml_warmup_guard()

    try:
        return (
            AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs),
            model_load_kwargs,
        )
    except TypeError as error:
        if (
            not _is_retryable_params4bit_dispatch_error(error)
            or model_load_kwargs.get("quantization_config") is None
            or model_load_kwargs.get("device_map") is None
            or not _cuda_available()
        ):
            raise

        quantization_config = model_load_kwargs.get("quantization_config")
        if (
            quantization_config is not None
            and getattr(quantization_config, "load_in_4bit", False)
            and model_load_kwargs.get("device_map") == "auto"
            and model_load_kwargs.get("max_memory") is not None
        ):
            retry_kwargs = dict(model_load_kwargs)
            retry_kwargs["quantization_config"] = _build_bnb_quantization_config(
                load_in_4bit=False,
                cpu_offload=True,
            )
            print(
                "Retrying model load with 8-bit quantization and fp32 CPU "
                "offload because this bitsandbytes build cannot auto-dispatch "
                "4-bit Params4bit tensors."
            )
            torch.cuda.empty_cache()
            return (
                AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs),
                retry_kwargs,
            )

        retry_kwargs = dict(model_load_kwargs)
        retry_kwargs["device_map"] = {"": _active_cuda_device()}
        retry_kwargs.pop("max_memory", None)

        quantization_config = retry_kwargs.get("quantization_config")
        if quantization_config is not None and hasattr(
            quantization_config, "llm_int8_enable_fp32_cpu_offload"
        ):
            quantization_config.llm_int8_enable_fp32_cpu_offload = False

        print(
            "Retrying 4-bit model load on the active CUDA device because "
            "multi-device accelerate dispatch is incompatible with this "
            "bitsandbytes build."
        )
        return (
            AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs),
            retry_kwargs,
        )
    except ValueError as error:
        quantization_config = model_load_kwargs.get("quantization_config")
        if (
            not _is_retryable_bnb_cpu_offload_error(error)
            or quantization_config is None
            or not hasattr(quantization_config, "llm_int8_enable_fp32_cpu_offload")
        ):
            raise

        retry_kwargs = dict(model_load_kwargs)
        quantization_config.llm_int8_enable_fp32_cpu_offload = True
        retry_kwargs["quantization_config"] = quantization_config
        print(
            "Retrying quantized model load with CPU offload enabled for modules "
            "that auto-dispatch places outside GPU memory."
        )
        return (
            AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs),
            retry_kwargs,
        )
    except torch.OutOfMemoryError as error:
        if model_load_kwargs.get("quantization_config") is None or not _cuda_available():
            raise

        current_device_map = model_load_kwargs.get("device_map")
        if current_device_map == "auto" or not is_accelerate_available():
            raise RuntimeError(
                "CUDA ran out of memory while loading the base model. "
                "Free GPU memory or set HAMLET_BASE_MODEL to a smaller model "
                "such as LiquidAI/LFM2-2.6B."
            ) from error

        retry_kwargs = dict(model_load_kwargs)
        retry_kwargs["device_map"] = "auto"
        max_memory = _cuda_max_memory_map()
        if max_memory is not None:
            retry_kwargs["max_memory"] = max_memory
        print(
            "Retrying quantized model load with accelerate auto-dispatch "
            "because the active CUDA device ran out of memory."
        )
        torch.cuda.empty_cache()
        return (
            AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs),
            retry_kwargs,
        )


def _load_tokenizer_and_model_with_fallbacks(
    model_names: list[str],
    model_load_kwargs: dict[str, object],
) -> tuple[str, AutoTokenizer, torch.nn.Module, dict[str, object]]:
    last_error: BaseException | None = None

    for index, model_name in enumerate(model_names):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                force_download=FORCE_HF_DOWNLOAD,
                trust_remote_code=True,
            )
            model, resolved_model_load_kwargs = (
                _load_causal_lm_with_quantized_dispatch_fallback(
                    model_name,
                    dict(model_load_kwargs),
                )
            )
            return model_name, tokenizer, model, resolved_model_load_kwargs
        except Exception as error:
            last_error = error
            if _cuda_available():
                torch.cuda.empty_cache()

            if index == len(model_names) - 1:
                raise

            short_error = str(error).splitlines()[0]
            next_model_name = model_names[index + 1]
            print(
                f"Failed to load base model {model_name}: "
                f"{error.__class__.__name__}: {short_error}"
            )
            print(f"Falling back to smaller base model: {next_model_name}")

    if last_error is not None:
        raise last_error
    raise RuntimeError("No candidate base models were configured.")


def main() -> None:
    use_cuda = _cuda_available()
    if use_cuda:
        torch.cuda.empty_cache()

    repo_root = find_repo_root()
    raw_text_path = repo_root / "data" / "hamlet_onlyhamletraw.txt"
    output_dir = repo_root / "models" / "lora_hamlet_3"

    if not raw_text_path.is_file():
        raise FileNotFoundError(f"Missing raw text file: {raw_text_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {repo_root}")
    print(f"Raw Hamlet text: {raw_text_path}")
    print(f"LoRA output dir: {output_dir}")
    print(f"CUDA available: {use_cuda}")

    hamlet_speeches = extract_hamlet_speeches(
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

    plain_english_speeches = [
        shakespeare_to_plain_english(speech) for speech in hamlet_speeches
    ]
    changed_flags = [
        source.lower() != target.lower()
        for source, target in zip(hamlet_speeches, plain_english_speeches)
    ]

    normalized_corpus = Dataset.from_dict(
        {
            "original_text": hamlet_speeches,
            "plain_english_text": plain_english_speeches,
            "changed": changed_flags,
        }
    )
    changed_examples = normalized_corpus.filter(lambda row: row["changed"])

    print(f"Total translated speeches: {len(normalized_corpus)}")
    print(f"Speeches changed by normalization: {len(changed_examples)}")
    example_row = (
        changed_examples[0] if len(changed_examples) > 0 else normalized_corpus[0]
    )
    print("Original speech:", example_row["original_text"][:200] + "...")
    print("Plain English speech:", example_row["plain_english_text"][:200] + "...")

    speech_corpus = normalized_corpus.shuffle(seed=SEED)
    eval_size = max(1, int(round(len(speech_corpus) * EVAL_SPLIT)))
    if eval_size >= len(speech_corpus):
        raise ValueError("Need at least two speeches to create train/eval split.")

    speech_split = speech_corpus.train_test_split(test_size=eval_size, seed=SEED)
    train_speech_dataset = speech_split["train"]
    eval_speech_dataset = speech_split["test"]

    train_dataset = Dataset.from_list(build_roleplay_examples(train_speech_dataset))
    eval_dataset = Dataset.from_list(build_roleplay_examples(eval_speech_dataset))

    print(f"Training speeches: {len(train_speech_dataset)}")
    print(f"Evaluation speeches: {len(eval_speech_dataset)}")
    print(f"Training roleplay examples: {len(train_dataset)}")
    print(f"Evaluation roleplay examples: {len(eval_dataset)}")
    print("Example training pair:")
    print("Prompt:", train_dataset[0]["instruction"])
    print("Response:", train_dataset[0]["response"][:400] + "...")

    if use_cuda and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif use_cuda:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    accelerate_available = is_accelerate_available()
    bitsandbytes_available = (
        BitsAndBytesConfig is not None and is_bitsandbytes_available()
    )

    model_load_kwargs: dict[str, object] = {
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    using_quantization = False
    using_device_map = False
    auto_max_memory = _cuda_max_memory_map() if use_cuda and accelerate_available else None

    if use_cuda:
        if bitsandbytes_available:
            model_load_kwargs["device_map"] = (
                "auto" if accelerate_available else {"": _active_cuda_device()}
            )
            using_device_map = True
            model_load_kwargs["quantization_config"] = _build_bnb_quantization_config(
                load_in_4bit=True,
                cpu_offload=accelerate_available,
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
                    print(f"Auto-dispatch max_memory: {_format_max_memory_map(auto_max_memory)}")
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
                    print(f"Auto-dispatch max_memory: {_format_max_memory_map(auto_max_memory)}")
            else:
                print(
                    "accelerate is missing or below the Transformers minimum version. "
                    "Loading without `device_map`; "
                    "automatic offload/sharding is disabled."
                )
            print(
                "bitsandbytes is missing or incompatible with this Transformers/CUDA setup. "
                "Loading full-precision CUDA model."
            )
    else:
        print("CUDA not available. Loading CPU model without bitsandbytes quantization.")

    selected_model_name, tokenizer, model, model_load_kwargs = (
        _load_tokenizer_and_model_with_fallbacks(
            _candidate_base_model_names(),
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
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_train_dataset = train_dataset.map(
        lambda row: tokenize_supervised_example(row, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda row: tokenize_supervised_example(row, tokenizer),
        remove_columns=eval_dataset.column_names,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    print(f"Tokenized train rows: {len(tokenized_train_dataset)}")
    print(f"Tokenized eval rows: {len(tokenized_eval_dataset)}")

    bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=use_cuda and not bf16_ok,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
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
        torch.cuda.empty_cache()

    inference_model = model
    inference_model.eval()

    device = _model_input_device(inference_model)
    if device == "cpu" or (device.startswith("cuda") and not using_device_map and not using_quantization):
        inference_model.to(device)

    sample_count = min(3, len(eval_dataset))
    for index in range(sample_count):
        example = eval_dataset[index]
        print(f"\nHeld-out example {index + 1}")
        print("Prompt:", example["instruction"])
        print("Reference:", example["response"])
        print("Model output:", ask_hamlet(example["instruction"], tokenizer, inference_model, device))

    freeform_questions = [
        "How do you feel about your father's death?",
        "What do you think of Claudius?",
        "Why do you hesitate when revenge lies before you?",
    ]
    for question in freeform_questions:
        print(f"\nFree-form question: {question}")
        print("Hamlet:", ask_hamlet(question, tokenizer, inference_model, device))


if __name__ == "__main__":
    main()
