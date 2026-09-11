from __future__ import annotations

from typing import Any

import torch


def build_absolute_position_ids(
    start_position: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build absolute position IDs for a generation step.

    The returned tensor has shape:

        [1, sequence_length]

    Unlike a cache-relative position calculation, these positions
    continue increasing even when old KV-cache entries are evicted.

    Example:

        start_position = 10
        sequence_length = 1

        -> tensor([[10]])
    """

    if start_position < 0:
        raise ValueError(
            "start_position must be non-negative."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    return torch.arange(
        start_position,
        start_position + sequence_length,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)


def build_cache_position(
    start_position: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build cache positions for a generation step.

    The values represent the absolute positions of the input
    tokens in the original sequence.
    """

    if start_position < 0:
        raise ValueError(
            "start_position must be non-negative."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    return torch.arange(
        start_position,
        start_position + sequence_length,
        device=device,
        dtype=torch.long,
    )


def ensure_rope_cache_length(
    model: Any,
    required_length: int,
) -> None:
    """
    Ensure legacy Qwen2 RoPE cosine/sine caches are long enough
    for the requested absolute position.

    Older Transformers Qwen2 implementations use a precomputed
    cosine/sine table and index it with position_ids.

    Newer implementations compute RoPE directly from position_ids,
    so no action is required for them.

    This function therefore only touches rotary embeddings that
    expose the legacy `_set_cos_sin_cache()` API.
    """

    if required_length <= 0:
        return

    base_model = getattr(model, "model", model)

    rotary_embeddings = []

    # Newer Qwen2 implementations keep one rotary embedding on
    # the base model.
    base_rotary = getattr(
        base_model,
        "rotary_emb",
        None,
    )

    if base_rotary is not None:
        rotary_embeddings.append(base_rotary)

    # Older Qwen2 implementations keep rotary embeddings inside
    # every attention layer.
    layers = getattr(
        base_model,
        "layers",
        [],
    )

    for layer in layers:
        attention = getattr(
            layer,
            "self_attn",
            None,
        )

        rotary = getattr(
            attention,
            "rotary_emb",
            None,
        )

        if rotary is not None:
            rotary_embeddings.append(rotary)

    seen = set()

    for rotary in rotary_embeddings:
        rotary_id = id(rotary)

        if rotary_id in seen:
            continue

        seen.add(rotary_id)

        if not hasattr(
            rotary,
            "_set_cos_sin_cache",
        ):
            continue

        max_cached = getattr(
            rotary,
            "max_seq_len_cached",
            0,
        )

        if required_length <= max_cached:
            continue

        inv_freq = getattr(
            rotary,
            "inv_freq",
            None,
        )

        if inv_freq is None:
            continue

        try:
            model_dtype = next(
                model.parameters()
            ).dtype
        except StopIteration:
            model_dtype = torch.float32

        rotary._set_cos_sin_cache(
            seq_len=required_length,
            device=inv_freq.device,
            dtype=model_dtype,
        )