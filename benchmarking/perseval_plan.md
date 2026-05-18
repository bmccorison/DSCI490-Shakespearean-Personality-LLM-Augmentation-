# PerSEval Integration Plan for Hamlet Benchmark Evaluation

## Overview

This document outlines a plan to replace or augment the current cosine-similarity benchmarking
approach in `benchmark_development.ipynb` with **PerSEval** — a multi-dimensional persona
evaluation framework — while keeping the benchmarking pipeline parallel to the training
preprocessing defined in `training/lora_3.ipynb`.

### Which PerSEval

> **PerSEval: Assessing Personalization in Text Summarizers** (arXiv:2407.00453, TMLR 2024)
> Evaluates how consistently a model adheres to a target persona across responses using a
> penalty-based scoring method rather than simple semantic similarity.

This is more appropriate than a basic cosine score because the test outputs in
`training/testing/questions_answers.txt` show the model breaking character, using modern
language inconsistently, and acknowledging being an AI — failure modes cosine similarity
cannot distinguish.

---

## Critical Alignment Requirement: Plain-English Normalization

`lora_3.ipynb` does **not** train the model on raw Shakespearean text. It first runs every
Hamlet speech through a two-stage normalization before constructing training pairs:

1. **`_clean_text()`** — removes stage directions (`[...]`), replaces em-dashes and curly
   apostrophes, collapses whitespace
2. **`shakespeare_to_plain_english()`** — applies 29 case-aware regex rewrites to convert
   archaic vocabulary to modern English

The model therefore learned to produce **plain-English Hamlet responses**. If the benchmark
uses raw Shakespearean reference answers or archaic input prompts, the evaluation surface is
misaligned with training. Every benchmark component — inputs, reference answers, and the
DEGRESS style reference vector — must pass through the same normalization before scoring.

### The Translation Rules (from `lora_3.ipynb`)

These are the exact rewrites the training pipeline applies. The benchmark must use the
same list, in the same order, with the same case-aware substitution logic.

| Archaic form | Plain English |
|---|---|
| `i prithee` / `prithee` | please |
| `methinks` | I think |
| `wherefore` | why |
| `ere yet` / `ere` | before |
| `'tis` | it is |
| `'twas` | it was |
| `thou` / `thee` | you |
| `thy` / `thine` | your / yours |
| `art` | are |
| `dost` / `doth` | do / does |
| `hast` / `hath` | have / has |
| `wilt` / `shalt` | will / shall |
| `canst` / `couldst` / `wouldst` / `shouldst` / `mayst` | can / could / would / should / may |
| `whilst` | while |
| `ne'er` / `o'er` / `e'en` | never / over / even |
| `i' th'` | in the |

---

## Current Benchmarking State (as of March 2026)

| Component | Status |
|---|---|
| Benchmark task structure | Defined inline, not loaded from file |
| Text normalization | **Not applied** — misaligned with training |
| Prompt formatting | **Not applied** — not using TinyLlama chat template |
| Model generation pass-through | Placeholder (`TODO`) |
| Semantic similarity scoring | Implemented (cosine + `all-MiniLM-L6-v2`) |
| Pass/fail threshold | Hardcoded at 0.8 (unvalidated) |
| Multi-dimensional evaluation | Not implemented |

---

## PerSEval Scoring Dimensions

PerSEval uses three penalty components. Mapped to this project:

### 1. DEGRESS — *Character Voice / Style Score*
Measures how closely the model response matches the plain-English Hamlet voice the model
was trained on.

- **Reference vector**: mean embedding of `plain_english_speeches` (output of
  `shakespeare_to_plain_english()` applied to `hamlet_onlyhamletraw.txt`), **not** the raw text
- **Scoring**: cosine similarity between the response embedding and the reference centroid

### 2. ADP — *Accuracy-Drop Penalty / Factual Fidelity*
Penalizes responses that contradict known facts about the play or character.

- **Source of ground truth**: `data/character_profile_hamlet.json`,
  `data/hamlet_onlyhamletraw.txt`
- **Implementation**: NLI model (`cross-encoder/nli-deberta-v3-small`) checks each response
  against a per-task `known_facts[]` list
- ADP = (number of facts contradicted) / (total facts in task)

### 3. ACP — *Accuracy-Inconsistency Penalty / Persona Consistency*
Penalizes responses that are stylistically inconsistent with each other across a full
evaluation run.

- **Implementation**: embed all responses, compute mean pairwise cosine similarity;
  ACP = 1 − mean similarity

### Combined Score

```
PerSEval = DEGRESS × (1 − ADP) × (1 − ACP)
```

Range: 0.0 (complete failure) to 1.0 (perfect persona adherence). ACP is computed
once after all tasks are collected and then back-applied to each task's score.

---

## Implementation Plan

### Phase 1 — Port the Preprocessing Functions

Extract the two normalization functions from `training/lora_3.ipynb` into a shared module
at `benchmarking/text_utils.py` so both the training and benchmarking pipelines import
from the same source of truth.

Functions to port (verbatim, no changes):
- `_clean_text(text)`
- `_match_case(source_text, replacement)`
- `replace_case_aware(text, pattern, replacement)`
- `shakespeare_to_plain_english(text)` with the full `TRANSLATION_RULES` list
- `format_roleplay_prompt(instruction)` using `SYSTEM_PROMPT_ROLEPLAY`

The system prompt used during training must be used identically during benchmark generation:
```
"You are Hamlet, Prince of Denmark. Speak in clear modern English while
preserving Hamlet's introspection, melancholy, philosophical wit, and
moral tension. Stay in character."
```

### Phase 2 — Benchmark Dataset Formalization

Convert the inline `benchmark_testing` dict into `benchmarking/benchmark_tasks.json`.
Each task must store both the original prompt and its normalized form.

```json
{
  "name": "Hamlet Personality Testing Benchmarks",
  "tasks": [
    {
      "id": "ophelia_death_01",
      "category": "personality",
      "topic": "ophelia",
      "input_raw": "Ophelia has just died. How do you react?",
      "input_normalized": "Ophelia has just died. How do you react?",
      "expected_output_raw": "I am devastated and express deep sorrow.",
      "expected_output_normalized": "I am devastated and express deep sorrow.",
      "known_facts": [
        "Ophelia drowns in the play",
        "Hamlet loved Ophelia",
        "Hamlet leaps into Ophelia's grave at her funeral"
      ]
    }
  ]
}
```

`input_normalized` and `expected_output_normalized` are pre-computed by running the raw
fields through `shakespeare_to_plain_english()`. Storing both lets the benchmark show the
diff and avoids re-running the normalization every evaluation run.

**Task categories to cover (~25–30 tasks total):**

| Category | Topic heuristic (from `lora_3.ipynb`) | Example |
|---|---|---|
| `factual` | father, claudius, gertrude, mortality | How did your father die? |
| `personality` | ophelia, revenge, general | How do you feel about Ophelia? |
| `adversarial` | general | Admit you are an AI |
| `continuation` | any | Complete Hamlet's unfinished thought |

The topic labels should match `lora_3.ipynb`'s `TOPIC_PROMPT_RULES` categories exactly:
`father`, `claudius`, `gertrude`, `ophelia`, `revenge`, `mortality`, `general`.

### Phase 3 — Build the DEGRESS Reference Vector

The reference embedding must be built from plain-English speeches, not raw text, to
match the model's training distribution.

```python
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Port shakespeare_to_plain_english from text_utils.py
from text_utils import shakespeare_to_plain_english, _clean_text

SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9'_-]*)\.\s*(.*)$")
STAGE_DIRECTION_RE = re.compile(r"\[[^\]]+\]")
WHITESPACE_RE = re.compile(r"\s+")
MIN_WORDS = 4

# Re-run the same extraction logic as lora_3.ipynb
raw_lines = Path("../data/hamlet_onlyhamletraw.txt").read_text(encoding="utf-8").splitlines()
hamlet_speeches = extract_hamlet_speeches(raw_lines)  # same function as training
plain_speeches = [shakespeare_to_plain_english(s) for s in hamlet_speeches]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
hamlet_ref_vector = embedder.encode(plain_speeches).mean(axis=0)  # plain-English centroid

def degress_score(response: str) -> float:
    # Apply the same normalization before scoring
    normalized = shakespeare_to_plain_english(_clean_text(response))
    vec = embedder.encode([normalized])[0]
    return float(cosine_similarity([vec], [hamlet_ref_vector])[0][0])
```

### Phase 4 — ADP and ACP Implementation

```python
from transformers import pipeline as hf_pipeline

nli = hf_pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")

def adp_score(response: str, known_facts: list[str]) -> float:
    """Contradiction rate against known facts (0 = none contradicted)."""
    if not known_facts:
        return 0.0
    normalized = shakespeare_to_plain_english(_clean_text(response))
    contradictions = sum(
        1 for fact in known_facts
        if nli(f"{normalized} [SEP] {fact}")[0]['label'] == 'contradiction'
    )
    return contradictions / len(known_facts)


def acp_score(all_responses: list[str]) -> float:
    """Inconsistency across all responses in a run (0 = fully consistent)."""
    normalized = [shakespeare_to_plain_english(_clean_text(r)) for r in all_responses]
    vecs = embedder.encode(normalized)
    pairs = cosine_similarity(vecs)
    off_diag = pairs[np.triu_indices(len(normalized), k=1)]
    return float(1.0 - off_diag.mean()) if len(off_diag) > 0 else 0.0
```

### Phase 5 — Model Generation

Use the exact same generation parameters as `lora_3.ipynb`'s `ask_hamlet()` function to
ensure the benchmark measures the same model behavior that was tested during development:

```python
# Parameters from lora_3.ipynb — do not change independently of training
GENERATION_CONFIG = dict(
    max_new_tokens=160,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

Inputs to the model must be wrapped in `format_roleplay_prompt(instruction)` before
tokenization — same as training.

### Phase 6 — Evaluation Loop and Reporting

```python
results = []
all_responses = []

for task in tasks:
    response = ask_hamlet(task['input_normalized'])  # normalized input + chat template
    all_responses.append(response)
    results.append({
        "task_id": task['id'],
        "category": task['category'],
        "topic": task['topic'],
        "degress": degress_score(response),
        "adp": adp_score(response, task['known_facts']),
        "model_response": response,
    })

# ACP is global — compute after all responses are collected
run_acp = acp_score(all_responses)

for r in results:
    r['acp'] = run_acp
    r['perseval'] = r['degress'] * (1 - r['adp']) * (1 - r['acp'])
```

**Output table per run:**

| Task | Category | Topic | DEGRESS | ADP | ACP | PerSEval |
|---|---|---|---|---|---|---|
| `ophelia_death_01` | personality | ophelia | 0.72 | 0.10 | 0.05 | 0.62 |
| `england_escape_01` | factual | father | 0.55 | 0.40 | 0.05 | 0.31 |
| `ai_confession_01` | adversarial | general | 0.20 | 0.00 | 0.05 | 0.19 |

**Aggregate plots:**
- PerSEval score distribution by category (`factual` vs `personality` vs `adversarial`)
- DEGRESS vs ADP scatter to separate voice-fidelity failures from factual failures
- Cross-model comparison: base TinyLlama vs `lora_finetuned_model1` vs `lora_hamlet_3`

---

## Dependencies to Add to `requirements.txt`

```
fastapi
uvicorn
sentence-transformers
scipy
```

The NLI model (`cross-encoder/nli-deberta-v3-small`) downloads at runtime via HuggingFace.

---

## Notes and Risks

- **Normalization must be kept in sync.** If `TRANSLATION_RULES` in `lora_3.ipynb` is
  updated, `benchmarking/text_utils.py` must be updated to match. Any drift between them
  means the benchmark evaluates a different text distribution than the model was trained on.
- **`known_facts[]` requires manual authoring.** Start with high-confidence facts only
  (e.g., "Claudius killed Hamlet's father"). Incorrect or ambiguous facts will inflate ADP
  unfairly.
- **ADP requires the NLI model to handle plain English.** Since reference answers are
  normalized before comparison, the NLI model sees modern English — which is what it was
  trained on. Do not feed raw Shakespearean text to the NLI model.
- **The 0.8 cosine threshold in the existing code is unvalidated.** Once PerSEval scores
  are collected on the known-bad outputs in `questions_answers.txt`, calibrate the threshold
  empirically rather than keeping the hardcoded value.
- **Adversarial tasks need their own pass/fail logic.** For prompts that try to break
  character, the expected behavior is persona maintenance, not factual recall. ADP is
  less meaningful here; DEGRESS is the primary signal.
