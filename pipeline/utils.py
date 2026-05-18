''' Random utilities used in app.py '''

import gc
import threading

from pipeline.lm_generation import (
    BASE_MODEL_ADAPTER_PATH,
    attach_named_adapter,
    load_base_model_and_tokenizer,
    model_selection,
    validate_and_resolve_adapter,
)


# Resident model state — one base model is kept loaded across adapter swaps.
# LoRA adapters are stored by name in a single PeftModel to avoid redundant base reloads.
_resident_model_name = ""
_resident_model = None
_resident_tokenizer = None
_resident_adapter_slots: dict[str, str] = {}  # adapter_path → named slot in PeftModel
_next_slot = 0
_model_lock = threading.Lock()


def _release_resident_model() -> None:
    '''Release the resident base model and all loaded adapter slots.'''
    global _resident_model_name, _resident_model, _resident_tokenizer
    global _resident_adapter_slots, _next_slot

    prev_model = _resident_model
    prev_tokenizer = _resident_tokenizer
    _resident_model = None
    _resident_tokenizer = None
    _resident_model_name = ""
    _resident_adapter_slots = {}
    _next_slot = 0

    if prev_model is None and prev_tokenizer is None:
        return

    del prev_model, prev_tokenizer
    gc.collect()

    try:
        import torch
    except Exception:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available") or not cuda.is_available():
        return

    cuda.empty_cache()
    if hasattr(cuda, "ipc_collect"):
        cuda.ipc_collect()


def ensure_loaded_model(model_name: str, adapter_path: str):
    '''Return the active model and tokenizer, loading or hot-swapping adapters as needed.

    The base model stays resident as long as model_name does not change. LoRA adapters are
    loaded by name into a single PeftModel, so switching between participants with the same
    base model costs only a set_adapter call rather than a full model reload.

    Reverting to the base-model-only pseudo-adapter requires a full reload because extracting
    the bare base from a PeftModel is not supported without reloading from disk.
    '''
    global _resident_model_name, _resident_model, _resident_tokenizer
    global _resident_adapter_slots, _next_slot

    norm_model = model_name.strip()
    norm_adapter = adapter_path.strip()
    if not norm_model or not norm_adapter:
        raise ValueError("Model name and adapter path are required.")

    with _model_lock:
        base_only = norm_adapter == BASE_MODEL_ADAPTER_PATH
        peft_model_loaded = hasattr(_resident_model, "set_adapter")

        # A base-model-only participant after adapter-bearing ones forces a reload because
        # we cannot cleanly strip PeftModel wrapping without touching disk.
        if norm_model != _resident_model_name or (base_only and peft_model_loaded):
            _release_resident_model()
            _resident_model, _resident_tokenizer = load_base_model_and_tokenizer(norm_model)
            _resident_model_name = norm_model

        if not base_only:
            if norm_adapter not in _resident_adapter_slots:
                resolved = validate_and_resolve_adapter(norm_model, norm_adapter)
                slot_name = f"slot_{_next_slot}"
                _next_slot += 1
                _resident_model = attach_named_adapter(_resident_model, resolved, slot_name)
                _resident_adapter_slots[norm_adapter] = slot_name
            else:
                _resident_model.set_adapter(_resident_adapter_slots[norm_adapter])

        return _resident_model, _resident_tokenizer


def empty_multimodel_session() -> dict[str, object]:
    '''Return a stable idle payload for frontend session polling.'''
    return {
        "active": False,
        "status": "idle",
        "is_stopped": False,
        "is_complete": True,
        "turn_count": 0,
        "turns": [],
        "last_turn": None,
        "next_speaker": None,
    }


def resolve_multimodel_persona(
    model_name: str,
    adapter_path: str,
) -> tuple[str, str]:
    '''Resolve a multimodel participant's character/work from published model config.'''
    selected_model = next(
        (model for model in model_selection() if model["name"] == model_name),
        None,
    )
    if selected_model is None:
        raise ValueError(f"Model is not available: {model_name}")

    selected_adapter = next(
        (
            adapter
            for adapter in selected_model["adapters"]
            if adapter["path"] == adapter_path
        ),
        None,
    )
    if selected_adapter is None:
        raise ValueError(f"Adapter is not available for {model_name}: {adapter_path}")

    character = str(
        selected_adapter.get("character") or selected_model.get("character") or ""
    ).strip()
    work = str(
        selected_adapter.get("work") or selected_model.get("work") or ""
    ).strip()
    if not character or not work:
        raise ValueError(
            f"Character metadata is missing for {model_name} with {adapter_path}."
        )
    return character, work


def resolve_cors_origins(configured_value: str, default_value: str) -> list[str]:
    '''Parse a CORS origins env value into a clean list of origins.'''
    raw_value = configured_value if configured_value is not None else default_value
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]
