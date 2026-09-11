"""Compatibility helpers for Hugging Face KV-cache representations."""

from __future__ import annotations

from typing import Any, Iterator

import torch


def iter_cache_layers(
    past_key_values: Any,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(key, value)`` tensors from current or legacy cache APIs."""

    if past_key_values is None:
        return

    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Each legacy cache layer must contain key and value tensors.")
            yield layer[0], layer[1]
        return

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                yield keys, values
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        yield from zip(key_cache, value_cache)
        return

    raise TypeError(f"Unsupported cache type: {type(past_key_values).__name__}")


def cache_as_tensor_tuple(
    past_key_values: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return a non-copying tensor view of any supported cache."""

    return tuple(iter_cache_layers(past_key_values))


def cache_sequence_length(past_key_values: Any) -> int:
    """Return the physical sequence length of a supported cache."""

    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    layers = cache_as_tensor_tuple(past_key_values)
    return 0 if not layers else int(layers[0][0].shape[-2])
