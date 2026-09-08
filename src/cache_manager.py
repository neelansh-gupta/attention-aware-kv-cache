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

class AttentionSinkCacheManager:
    """
    Manage a KV cache using a StreamingLLM-style policy.

    The cache retains:
        1. A fixed number of initial sink tokens.
        2. The most recent tokens using the remaining budget.

    Example:
        budget = 8
        sink_tokens = 2

        retained = [first 2 tokens] + [latest 6 tokens]

    Correct RoPE/position handling after eviction is intentionally
    deferred to Stage 6.
    """

    def __init__(
        self,
        cache_budget: int,
        sink_tokens: int = 4,
    ):
        if cache_budget <= 0:
            raise ValueError(
                "cache_budget must be positive."
            )

        if sink_tokens < 0:
            raise ValueError(
                "sink_tokens must be non-negative."
            )

        if sink_tokens >= cache_budget:
            raise ValueError(
                "sink_tokens must be smaller than cache_budget."
            )

        self.cache_budget = cache_budget
        self.sink_tokens = sink_tokens
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

    @staticmethod
    def _evict_tensor(
        tensor: torch.Tensor,
        cache_budget: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        sequence_length = tensor.shape[-2]

        if sequence_length <= cache_budget:
            return tensor

        recent_tokens = cache_budget - sink_tokens
        recent_start = sequence_length - recent_tokens

        sink = tensor[..., :sink_tokens, :]
        recent = tensor[..., recent_start:, :]

        return torch.cat(
            [sink, recent],
            dim=-2,
        )

    def _update_dynamic_cache(self, cache: Any) -> Any:
        """
        Apply sink-aware eviction to modern Hugging Face cache objects.

        Supports:
            - modern Cache layers using `.layers`
            - older DynamicCache implementations using
              `.key_cache` / `.value_cache`
        """

        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if not hasattr(layer, "keys"):
                    continue

                layer.keys = self._evict_tensor(
                    layer.keys,
                    self.cache_budget,
                    self.sink_tokens,
                )

                layer.values = self._evict_tensor(
                    layer.values,
                    self.cache_budget,
                    self.sink_tokens,
                )

            return cache

        if hasattr(cache, "key_cache") and hasattr(
            cache, "value_cache"
        ):
            for layer_idx in range(len(cache.key_cache)):
                cache.key_cache[layer_idx] = (
                    self._evict_tensor(
                        cache.key_cache[layer_idx],
                        self.cache_budget,
                        self.sink_tokens,
                    )
                )

                cache.value_cache[layer_idx] = (
                    self._evict_tensor(
                        cache.value_cache[layer_idx],
                        self.cache_budget,
                        self.sink_tokens,
                    )
                )

            return cache

        raise TypeError(
            "Unsupported DynamicCache implementation."
        )

    def update(self, cache: Any):
        """
        Enforce the StreamingLLM-style cache budget.

        The native DynamicCache is preserved. We modify its stored
        key/value tensors instead of converting it to a legacy cache.
        """

        if cache is None:
            return None

        sequence_length = self.sequence_length(cache)

        if sequence_length <= self.cache_budget:
            return cache

        removed = sequence_length - self.cache_budget

        if isinstance(cache, (tuple, list)):
            evicted_cache = []

            recent_tokens = (
                self.cache_budget - self.sink_tokens
            )
            recent_start = (
                sequence_length - recent_tokens
            )

            for key, value in cache:
                sink_key = key[
                    ..., :self.sink_tokens, :
                ]
                recent_key = key[
                    ..., recent_start:, :
                ]

                sink_value = value[
                    ..., :self.sink_tokens, :
                ]
                recent_value = value[
                    ..., recent_start:, :
                ]

                evicted_cache.append(
                    (
                        torch.cat(
                            [sink_key, recent_key],
                            dim=-2,
                        ),
                        torch.cat(
                            [sink_value, recent_value],
                            dim=-2,
                        ),
                    )
                )

            self.total_evictions += removed
            return tuple(evicted_cache)

        result = self._update_dynamic_cache(cache)

        self.total_evictions += removed

        return result

    def stats(self, cache: Any) -> CacheStats:
        return CacheStats(
            sequence_length=self.sequence_length(cache),
            budget=self.cache_budget,
            total_evictions=self.total_evictions,
        )