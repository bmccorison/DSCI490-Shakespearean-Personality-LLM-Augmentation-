# span_weights.py
from __future__ import annotations


def compute_token_weights(
    response_text: str,
    spans: list[dict],
    tokenizer,
    base_weight: float = 1.0,
    good_boost: float = 1.5,
    bad_penalty: float = 0.2,
) -> list[float]:
    """
    Given a response string and a list of highlighted spans, produce
    per-token weights. Tokens inside a 'good' span are boosted;
    tokens inside a 'bad' span are penalized.
    """
    token_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    weights = [base_weight] * len(token_ids)

    for span in spans:
        text = span.get("text", "").strip()
        polarity = span.get("polarity")
        if not text or polarity not in ("good", "bad"):
            continue

        # Find character offset of span in response
        start_char = response_text.find(text)
        if start_char == -1:
            continue
        end_char = start_char + len(text)

        # Map character range to token indices using offset mapping
        encoding = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"]

        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start_char or tok_start >= end_char:
                continue
            if polarity == "good":
                weights[token_idx] = good_boost
            else:
                weights[token_idx] = bad_penalty

    return weights
