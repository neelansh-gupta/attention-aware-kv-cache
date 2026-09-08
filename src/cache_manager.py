from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CacheStats:
    sequence_length: int
    budget: int
    total_evictions: int


class SlidingWindowCacheManager:
    """
    Manage a Hugging Face DynamicCache under a fixed
    sliding-window budget.

    Stage 3:
        Keep only the most recent `window_size` KV entries.

    Note:
        Correct RoPE/position handling after eviction is
        intentionally deferred to Stage 6.
    """

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive, got {window_size}"
            )

        self.window_size = window_size
        self.total_evictions = 0

    def reset(self) -> None:
        self.total_evictions = 0

    def sequence_length(self, cache: Any) -> int:
        if cache is None:
            return 0

        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())

        if isinstance(cache, (tuple, list)):
            if len(cache) == 0:
                return 0

            return cache[0][0].shape[-2]

        raise TypeError(
            f"Unsupported cache type: {type(cache).__name__}"
        )

    def update(self, cache: Any):
        """
        Enforce the sliding-window budget.

        For modern Hugging Face DynamicCache, use its native
        crop() operation so the model can continue receiving
        a Cache object.
        """

        if cache is None:
            return None

        sequence_length = self.sequence_length(cache)

        if sequence_length <= self.window_size:
            return cache

        removed = sequence_length - self.window_size

        # Modern Hugging Face Cache API.
        if hasattr(cache, "crop"):
            cache.crop(self.window_size)

            self.total_evictions += removed

            return cache

        # Fallback for legacy tuple-style caches.
        if isinstance(cache, (tuple, list)):
            evicted_cache = []

            for key, value in cache:
                evicted_cache.append(
                    (
                        key[..., -self.window_size:, :],
                        value[..., -self.window_size:, :],
                    )
                )

            self.total_evictions += removed

            return tuple(evicted_cache)

        raise TypeError(
            f"Unsupported cache type: {type(cache).__name__}"
        )

    def stats(self, cache: Any) -> CacheStats:
        return CacheStats(
            sequence_length=self.sequence_length(cache),
            budget=self.window_size,
            total_evictions=self.total_evictions,
        )