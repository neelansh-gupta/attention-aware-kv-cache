from __future__ import annotations

from typing import Any


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