from __future__ import annotations

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