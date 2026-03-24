# AGENT.md

## Project Summary

This repository is a local Shakespeare persona-LLM prototype centered on Hamlet. The current runnable path is:

React/Vite frontend -> FastAPI backend -> `pipeline/lm_generation.py` -> Hugging Face base model + local LoRA adapter.

RAG exists only as scaffolding right now and is not wired into live request handling.

## Fast Path

If you need to move quickly, read these files first:

1. `app.py`
2. `pipeline/lm_generation.py`
3. `models/models.py`
4. `interface/src/App.jsx`
5. `tests/test_app_contract.py`
6. `tests/test_lm_generation_contract.py`

## Repo Map

- `app.py`: FastAPI app and backend entrypoint.
- `runWebDemo.sh`: launches backend and frontend together with shared shutdown handling.
- `pipeline/lm_generation.py`: actual generation flow, chat state, character context, and model loading.
- `pipeline/rag.py`: placeholder retrieval helpers; not in the serving path today.
- `pipeline/data_ingestion.py`: placeholder ingestion code; currently out of sync with checked-in data filenames.
- `models/models.py`: registry of base models and adapter metadata surfaced to the UI.
- `models/lora_finetuned_model1`: checked-in LoRA adapter for TinyLlama.
- `models/lora_finetuned_model/checkpoint-270`: another TinyLlama LoRA checkpoint with tokenizer and trainer state.
- `interface/`: React 18 + Vite 5 + Tailwind 3 frontend.
- `tests/`: lightweight contract tests that stub heavy model dependencies.
- `data/`: Hamlet source material, parsed profile JSON, parser script, and raw text used for experiments.
- `training/`, `benchmarking/`, `misc/`, `docs/`: notebooks, experiments, and planning docs; useful context but not part of the live runtime path.
- `uv_config/`: separate experimental `uv` project; not the source of truth for the current demo workflow.

## Active Runtime Flow

1. `runWebDemo.sh` starts `python app.py` in the repo root and `npm run dev` in `interface/`.
2. `interface/src/App.jsx` initializes by calling:
   - `/api/refresh_chat`
   - `/api/select_character`
   - `/api/get_models`
   - `/api/select_model`
3. `app.py` delegates model loading and response generation to `pipeline/lm_generation.py`.
4. `pipeline/lm_generation.py` resolves available adapters from `models/models.py`.
5. Optional speech playback goes through `/api/tts`, with Bark imported lazily at request time.

## Commands

- Install Python deps: `pip install -r requirements.txt`
- Run backend only: `python app.py`
- Run frontend only: `cd interface && npm install && npm run dev`
- Run full demo: `./runWebDemo.sh`
- Build frontend: `cd interface && npm run build`
- Run tests: `python -m unittest discover -s tests`
- Syntax-check pipeline files: `python -m py_compile pipeline/*.py`
- Rebuild character profile JSON: `python data/character_profile_parser.py data/character_profile_hamlet.txt`

## API Surface

- `GET /api/get_models`
- `GET /api/select_model?model_name=...&adapter_path=...`
- `GET /api/select_character?character=...&work=...`
- `GET /api/generate_response?question=...`
- `GET /api/refresh_chat`
- `POST /api/tts?text=...&character=...`

## Model and Data Notes

- The only loadable model path exercised by the app and tests is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- `models/models.py` also lists `LiquidAI/LFM2-8B-A1B`, but it has no adapters, so `model_selection()` filters it out.
- Both checked-in adapters target TinyLlama:
  - `models/lora_finetuned_model1`
  - `models/lora_finetuned_model/checkpoint-270`
- `models/lora_finetuned_model/checkpoint-270` includes training artifacts and tokenizer files; `models/lora_finetuned_model1` is a smaller adapter-only directory.
- Main Hamlet data assets:
  - `data/hamlet_onlyhamletraw.txt`
  - `data/character_profile_hamlet.txt`
  - `data/character_profile_hamlet.json`
  - `data/Character.Profile_.Hamlet.pdf`
- `training/testing/questions.txt` and `training/testing/questions_answers.txt` are qualitative evaluation artifacts, not automated tests.

## Important Implementation Details

- `pipeline/lm_generation.py` uses module-level globals:
  - `messages`
  - `current_character`
  - `current_work`
- The app is effectively single-conversation state today. It is not multi-user safe.
- `refresh_chat_history()` clears the chat and re-seeds it with one system prompt from the current character/work context.
- `generate_output()` appends the user content during prompt assembly, then appends the assistant reply after generation.
- `app.py` will reject `/api/generate_response` until `/api/select_model` has successfully loaded a model and tokenizer.
- Bark TTS is intentionally lazy-loaded so importing `app.py` does not fail on machines without a usable Bark/torchaudio stack.
- Relevant env vars:
  - `BARK_HISTORY_PROMPT` default: `v2/en_speaker_6`
  - `BARK_USE_GPU`: optional truthy/falsey override

## Known Gaps and Footguns

- RAG is not implemented. `app.py` hardcodes `rag_context = None`.
- `pipeline/rag.py` imports `SentenceTransformer`, but `sentence-transformers` is not present in `requirements.txt`. Do not wire RAG into runtime without fixing dependencies.
- `pipeline/data_ingestion.py` tries to open `data/character_profile.json`, but the checked-in file is `data/character_profile_hamlet.json`.
- `training/testing/lora_2_generate.py` points at `models/lora_hamlet_profile`, which does not exist in this repository. Treat that script as stale until updated.
- `interface/src/App.jsx` currently hardcodes `CHARACTER_OPTIONS = ["Hamlet"]` and `DEFAULT_WORK = "Hamlet"`.
- CI uses Python `3.11` for syntax checks, while `uv_config/pyproject.toml` says `>=3.12` and local artifacts show other interpreter versions too. Keep Python changes compatible with CI unless the workflow is updated.
- `interface/node_modules`, `interface/dist`, `.venv`, `uv_config/.venv`, and `__pycache__` are generated artifacts and should not be edited.

## Testing and CI

- GitHub Actions currently does two things:
  - `python -m py_compile` over `pipeline/*.py`
  - `npm ci && npm run build` in `interface/`
- The most useful local regression checks are the contract tests in `tests/`:
  - `tests/test_app_contract.py`
  - `tests/test_lm_generation_contract.py`
- Those tests intentionally stub `torch`, `transformers`, `peft`, and pipeline imports so they stay fast and do not try to load large models.

## Change Guidance

- If you change backend endpoint names, params, or response shapes, update both `app.py` and `interface/src/App.jsx`, then rebuild the frontend.
- If you change model-selection behavior, update `models/models.py`, `pipeline/lm_generation.py`, and `tests/test_lm_generation_contract.py`.
- If you touch TTS behavior, preserve lazy Bark import and rerun `tests/test_app_contract.py`.
- If you add more characters or plays, update:
  - `models/models.py`
  - `pipeline/lm_generation.py`
  - `interface/src/App.jsx`
  - any tests that assume Hamlet defaults
- Prefer contract tests and stubs over real model loads during routine iteration.

## Notebook and Research Context

- `training/lora_1.ipynb`: early LoRA fine-tuning notebook using raw Hamlet text.
- `benchmarking/benchmark_development.ipynb`: benchmark-planning notebook around prompt/reference/model-response comparison.
- `misc/model_size_testing.ipynb`: compares candidate open-weight model sizes from small to large tiers.
- `docs/roughRoadmap.md`: planning artifact describing intended future phases; useful for intent, not current implementation truth.
