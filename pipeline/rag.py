'''
RAG (Retrieval-Augmented Generation) pipeline.

get_context(query) is the main entry point: it lazily builds and caches a vector store
from the character profile and speaker-aware context JSON files, then returns the
top-k most semantically similar passages as a single string for LLM grounding.
'''

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Single source of truth for the repo root lives in lm_generation.
from pipeline.lm_generation import REPO_ROOT

DEFAULT_CHARACTER_PROFILE = REPO_ROOT / "data" / "character_profile_hamlet.json"
DEFAULT_SPEAKER_CONTEXT = REPO_ROOT / "data" / "hamlet_speaker_aware_context.json"

_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
_TOP_K = 3
_TRANSFORMERS_LOAD_REPORT_LOGGERS = (
    "transformers.modeling_utils",
    "transformers.utils.loading_report",
)

# Lazy-initialized so importing this module does not block server startup while the sentence-transformer weights are downloaded or loaded from disk.
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()

# Keyed by (profile_path, context_path) so swapping characters rebuilds the store rather than returning stale embeddings for the wrong character.
_vs_cache: dict[tuple[str, str], "VectorStore"] = {}
_vs_lock = threading.Lock()


def _load_embedding_model(device: str) -> SentenceTransformer:
    '''Load the embedding model while suppressing harmless Transformers load reports.'''
    load_report_loggers = [
        logging.getLogger(logger_name)
        for logger_name in _TRANSFORMERS_LOAD_REPORT_LOGGERS
    ]
    previous_levels = [
        (load_report_logger, load_report_logger.level)
        for load_report_logger in load_report_loggers
    ]
    for load_report_logger in load_report_loggers:
        load_report_logger.setLevel(logging.ERROR)
    try:
        return SentenceTransformer(_EMBEDDING_MODEL, device=device)
    finally:
        for load_report_logger, previous_level in previous_levels:
            load_report_logger.setLevel(previous_level)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Place the embedding model on GPU when available so batch encoding during build_vector_store does not serialize on the CPU.
                try:
                    import torch
                    device = "cuda" if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
                _model = _load_embedding_model(device)
    return _model


@dataclass
class VectorStore:
    '''
    In-memory vector store holding all text chunks and their unit-normed embeddings.

    chunks     — the raw text strings, one per row of embeddings.
    embeddings — shape (n_chunks, embedding_dim); pre-normalized to unit length at build time
                 so per-request cosine similarity is a single dot product with no extra work.
    '''
    chunks: list[str]
    embeddings: np.ndarray

    def search(self, query_vec: np.ndarray, top_k: int = _TOP_K) -> list[str]:
        '''
        Return the top_k chunks most similar to query_vec using cosine similarity.

        Stored embeddings are pre-normalized; only the query needs normalizing here,
        so passage length does not bias scores toward longer chunks.
        argpartition finds the top-k candidates in O(N) without a full sort,
        then only those k indices are sorted by score.
        '''
        q_norm = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)  # guard zero-norm query
        scores = self.embeddings @ q_norm
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self.chunks[i] for i in top_idx]


def chunk_text(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    '''
    Split a long text string into overlapping word-based chunks.

    size    — max words per chunk (approximates token count).
    overlap — words shared between adjacent chunks so boundary context is not lost.
    '''
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    '''Return a dense embedding matrix for a list of text chunks.'''
    return _get_model().encode(chunks, show_progress_bar=False)


def _profile_chunks(profile: dict) -> list[str]:
    '''
    Convert a character-profile JSON object into a flat list of labeled text chunks.

    Each field becomes its own chunk so retrieval can surface the most relevant aspect
    of the character independently rather than always returning the entire profile.
    '''
    chunks = []
    if background := profile.get("background"):
        chunks.append(f"Background: {background}")
    for trait in profile.get("core_traits", []):
        chunks.append(f"Trait - {trait['name']}: {trait['description']}")
    for conflict in profile.get("key_internal_conflicts", []):
        chunks.append(f"Internal conflict - {conflict['name']}: {conflict['description']}")
    for rel in profile.get("relationships", []):
        chunks.append(f"Relationship with {rel['character']} ({rel['role']}): {rel['description']}")
    if arc := profile.get("psychological_arc"):
        chunks.append(f"Psychological arc: {arc}")
    if endures := profile.get("why_hamlet_endures"):
        chunks.append(f"Why Hamlet endures: {endures}")
    # The summary is a long essay; split on blank lines so each paragraph embeds as its own focused chunk rather than one massive undifferentiated vector.
    for para in profile.get("character_analysis", {}).get("summary", "").split("\n\n"):
        para = para.strip()
        if para:
            chunks.append(para)
    return chunks


def _context_chunks(entries: list[dict]) -> list[str]:
    '''
    Convert speaker-aware context entries into retrieval chunks.

    Pairing the preceding dialogue with Hamlet's response means a query resembling
    something another character said can surface Hamlet's canonical reply as context.
    '''
    chunks = []
    for entry in entries:
        context = entry.get("context_text", "").strip()
        response = entry.get("response", "").strip()
        if not response:
            continue
        chunk = f"{context}\nHamlet: {response}" if context else f"Hamlet: {response}"
        chunks.append(chunk)
    return chunks


def build_vector_store(
    character_profile_path: str | Path = DEFAULT_CHARACTER_PROFILE,
    speaker_context_path: str | Path = DEFAULT_SPEAKER_CONTEXT,
) -> VectorStore:
    '''
    Load both source JSON files, chunk and embed their contents, and return a VectorStore.

    Embeddings are batch-encoded (faster than one-at-a-time) and pre-normalized to unit
    length so per-request cosine similarity requires only a dot product in search().
    '''
    with open(character_profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    with open(speaker_context_path, encoding="utf-8") as f:
        context_entries = json.load(f)

    chunks = _profile_chunks(profile) + _context_chunks(context_entries)
    raw = embed_chunks(chunks)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / np.maximum(norms, 1e-9)  # pre-normalize; guard against zero-norm rows
    return VectorStore(chunks=chunks, embeddings=embeddings)


def retrieve(query: str, vector_store: VectorStore, top_k: int = _TOP_K) -> list[str]:
    '''Embed the query and return the top_k most similar passages from the vector store.'''
    return vector_store.search(_get_model().encode(query), top_k=top_k)


def get_context(
    query: str,
    character_profile_path: str | Path = DEFAULT_CHARACTER_PROFILE,
    speaker_context_path: str | Path = DEFAULT_SPEAKER_CONTEXT,
    top_k: int = _TOP_K,
) -> str:
    '''
    Main entry point called by the inference pipeline before each LLM generation.

    Lazily builds and caches the vector store on the first call for a given pair of
    data files, then retrieves the top_k most relevant passages joined by blank lines.
    '''
    key = (str(character_profile_path), str(speaker_context_path))
    # Double-checked locking: skips the lock on the hot path once the store is built, while preventing two concurrent first requests from both triggering a full rebuild.
    if key not in _vs_cache:
        with _vs_lock:
            if key not in _vs_cache:
                _vs_cache[key] = build_vector_store(character_profile_path, speaker_context_path)
    passages = retrieve(query, _vs_cache[key], top_k=top_k)
    return "\n\n".join(passages)
