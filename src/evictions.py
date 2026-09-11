from __future__ import annotations

from typing import Any

import torch


def _slice_cache_tensor(
    tensor,
    start: int,
    end: int | None = None,
):
    """
    Slice the sequence dimension of a KV-cache tensor.

    Qwen's KV tensors have the shape:

        [batch, kv_heads, sequence_length, head_dim]

    Therefore the sequence dimension is -2.
    """
    return tensor[..., start:end, :]


def sliding_window_evict(
    past_key_values: Any,
    window_size: int,
):
    """
    Keep only the most recent `window_size` tokens in a KV cache.

    The function accepts a legacy tuple-style cache:

        (
            (key_layer_0, value_layer_0),
            (key_layer_1, value_layer_1),
            ...
        )

    Each key/value tensor is expected to have shape:

        [batch, kv_heads, sequence_length, head_dim]

    Returns:
        A new tuple-style cache containing only the most recent
        `window_size` tokens.
    """

    if window_size <= 0:
        raise ValueError(
            f"window_size must be positive, got {window_size}"
        )

    if past_key_values is None:
        return None

    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "sliding_window_evict expects a legacy tuple/list "
            "KV cache. Convert modern Hugging Face Cache objects "
            "before calling this function."
        )

    if len(past_key_values) == 0:
        return tuple()

    first_layer = past_key_values[0]

    if not isinstance(first_layer, (tuple, list)):
        raise TypeError(
            "Each cache layer must contain (key, value)."
        )

    sequence_length = first_layer[0].shape[-2]

    # Nothing to evict.
    if sequence_length <= window_size:
        return tuple(
            (key, value)
            for key, value in past_key_values
        )

    start = sequence_length - window_size

    evicted_cache = []

    for layer in past_key_values:
        key, value = layer

        if key.shape[-2] != sequence_length:
            raise ValueError(
                "All cache layers must have the same sequence length."
            )

        if value.shape[-2] != sequence_length:
            raise ValueError(
                "Key/value sequence lengths must match."
            )

        new_key = _slice_cache_tensor(
            key,
            start,
            None,
        )

        new_value = _slice_cache_tensor(
            value,
            start,
            None,
        )

        evicted_cache.append(
            (new_key, new_value)
        )

    return tuple(evicted_cache)

def attention_sink_evict(
    past_key_values: Any,
    cache_budget: int,
    sink_tokens: int = 4,
):
    """
    Keep the first `sink_tokens` entries and the most recent entries
    required to satisfy `cache_budget`.

    Example:
        sequence_length = 10
        cache_budget = 6
        sink_tokens = 2

        Retained positions:
            [0, 1, 6, 7, 8, 9]

    This implements the structural StreamingLLM-style policy:
        initial attention sinks + recent local window.

    Correct RoPE/position handling after eviction is intentionally
    deferred to Stage 6.
    """
    if cache_budget <= 0:
        raise ValueError(
            f"cache_budget must be positive, got {cache_budget}"
        )

    if sink_tokens < 0:
        raise ValueError(
            f"sink_tokens must be non-negative, got {sink_tokens}"
        )

    if sink_tokens >= cache_budget:
        raise ValueError(
            "sink_tokens must be smaller than cache_budget."
        )

    if past_key_values is None:
        return None

    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "attention_sink_evict expects a legacy tuple/list "
            "KV cache."
        )

    if len(past_key_values) == 0:
        return tuple()

    sequence_length = (
        past_key_values[0][0].shape[-2]
    )

    if sequence_length <= cache_budget:
        return tuple(
            (key, value)
            for key, value in past_key_values
        )

    recent_tokens = cache_budget - sink_tokens
    recent_start = sequence_length - recent_tokens

    evicted_cache = []

    for layer in past_key_values:
        key, value = layer

        if key.shape[-2] != sequence_length:
            raise ValueError(
                "All cache layers must have the same sequence length."
            )

        if value.shape[-2] != sequence_length:
            raise ValueError(
                "Key/value sequence lengths must match."
            )

        sink_key = key[..., :sink_tokens, :]
        recent_key = key[..., recent_start:, :]

        sink_value = value[..., :sink_tokens, :]
        recent_value = value[..., recent_start:, :]

        new_key = torch.cat(
            [sink_key, recent_key],
            dim=-2,
        )

        new_value = torch.cat(
            [sink_value, recent_value],
            dim=-2,
        )

        evicted_cache.append(
            (new_key, new_value)
        )

    return tuple(evicted_cache)

def heavy_hitter_evict(
    past_key_values: Any,
    attention_scores: torch.Tensor,
    cache_budget: int,
    sink_tokens: int = 1,
    recent_window: int = 1,
):
    """
    Keep sinks, a recent window, and accumulated-attention heavy hitters.

    This implements the Stage 5 H2O-style heavy-hitter policy.

    Args:
        past_key_values:
            Legacy tuple/list KV cache:
                (
                    (key_layer_0, value_layer_0),
                    ...
                )

        attention_scores:
            One accumulated attention score per cached token.
            Expected shape:
                [sequence_length]
            or
                [1, sequence_length]

        cache_budget:
            Maximum number of tokens to retain.

    Returns:
        A tuple-style KV cache containing selected tokens in their original
        sequence order.
    """

    if cache_budget <= 0:
        raise ValueError(
            f"cache_budget must be positive, got {cache_budget}"
        )
    if sink_tokens < 0 or recent_window < 0:
        raise ValueError("sink_tokens and recent_window must be non-negative.")
    if sink_tokens + recent_window > cache_budget:
        raise ValueError(
            "sink_tokens + recent_window must not exceed cache_budget."
        )

    if past_key_values is None:
        return None

    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "heavy_hitter_evict expects a legacy tuple/list KV cache."
        )

    if len(past_key_values) == 0:
        return tuple()

    if not isinstance(attention_scores, torch.Tensor):
        raise TypeError(
            "attention_scores must be a torch.Tensor."
        )

    if attention_scores.dim() == 2:
        if attention_scores.shape[0] != 1:
            raise ValueError(
                "attention_scores with 2 dimensions must have "
                "shape [1, sequence_length]."
            )
        attention_scores = attention_scores[0]

    if attention_scores.dim() != 1:
        raise ValueError(
            "attention_scores must have shape "
            "[sequence_length] or [1, sequence_length]."
        )

    first_key, first_value = past_key_values[0]

    sequence_length = first_key.shape[-2]

    if attention_scores.shape[0] != sequence_length:
        raise ValueError(
            "Attention-score length must match the KV-cache "
            f"sequence length. Got scores={attention_scores.shape[0]}, "
            f"cache={sequence_length}."
        )

    # Nothing to evict.
    if sequence_length <= cache_budget:
        return tuple(
            (key, value)
            for key, value in past_key_values
        )

    # Deterministic priority: sinks, recent window, then the highest-scoring
    # non-reserved tokens. Final indices are restored to sequence order.
    sink = list(range(sink_tokens))
    recent_start = max(sequence_length - recent_window, sink_tokens)
    reserved = set(sink + list(range(recent_start, sequence_length)))
    heavy_slots = cache_budget - len(reserved)
    candidates = [index for index in range(sequence_length) if index not in reserved]
    ranked = sorted(
        candidates,
        key=lambda index: (-float(attention_scores[index].item()), index),
    )
    selected_indices = torch.tensor(
        sorted(reserved.union(ranked[:heavy_slots])),
        dtype=torch.long,
        device=attention_scores.device,
    )

    evicted_cache = []

    for key, value in past_key_values:
        if key.shape[-2] != sequence_length:
            raise ValueError(
                "All cache layers must have the same sequence length."
            )

        if value.shape[-2] != sequence_length:
            raise ValueError(
                "Key/value sequence lengths must match."
            )

        new_key = key.index_select(
            dim=-2,
            index=selected_indices.to(key.device),
        )

        new_value = value.index_select(
            dim=-2,
            index=selected_indices.to(value.device),
        )

        evicted_cache.append(
            (new_key, new_value)
        )

    return tuple(evicted_cache)

def cache_sequence_length(
    past_key_values: Any,
) -> int:
    """
    Return the cached sequence length for a legacy cache.
    """

    if past_key_values is None:
        return 0

    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "cache_sequence_length expects a tuple/list cache."
        )

    if len(past_key_values) == 0:
        return 0

    return past_key_values[0][0].shape[-2]