"""Offline, logical prefix-overlap analysis for exact prompt token IDs."""

from collections.abc import Sequence


def analyze_prefixes(sequences, block_size=1):
    """Compare each token-ID sequence with all prior requests in input order.

    Results describe potential prefix reuse only; they do not imply a KV cache
    retained, hit, or reused these tokens. Prompt strings and memory identifiers
    are deliberately not accepted as substitutes for tokenizer token IDs.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    prompts = []
    for request_index, sequence in enumerate(sequences):
        if isinstance(sequence, (str, bytes)) or not isinstance(sequence, Sequence):
            raise TypeError(f"sequence {request_index} must be a sequence of token IDs")
        ids = tuple(sequence)
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in ids):
            raise TypeError(f"sequence {request_index} contains a non-integer token ID")
        prompts.append(ids)

    results = []
    cumulative_repeated = 0
    cumulative_total = 0
    prior = []
    for request_index, ids in enumerate(prompts):
        max_prefix = 0
        duplicate = False
        for previous in prior:
            shared = 0
            limit = min(len(ids), len(previous))
            while shared < limit and ids[shared] == previous[shared]:
                shared += 1
            max_prefix = max(max_prefix, shared)
            if ids == previous:
                duplicate = True

        repeated = max_prefix
        cumulative_repeated += repeated
        cumulative_total += len(ids)
        results.append({
            "request_index": request_index,
            "prompt_tokens": len(ids),
            "prior_max_prefix_tokens": max_prefix,
            "block_aligned_prefix_tokens": (max_prefix // block_size) * block_size,
            "full_prompt_duplicate": duplicate,
            "repeated_tokens_this_request": repeated,
            "cumulative_repeated_tokens": cumulative_repeated,
            "cumulative_total_tokens": cumulative_total,
            "cumulative_repeated_fraction": (
                cumulative_repeated / cumulative_total if cumulative_total else 0.0
            ),
        })
        prior.append(ids)
    return results
