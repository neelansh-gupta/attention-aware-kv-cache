import torch

from src.cache_manager import SlidingWindowCacheManager
from src.evictions import sliding_window_evict


def make_fake_cache(
    sequence_length: int,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 4,
):
    """
    Create a small synthetic KV cache.

    Shape:
        [batch, kv_heads, sequence_length, head_dim]
    """

    cache = []

    for layer in range(num_layers):
        key = torch.arange(
            sequence_length * num_kv_heads * head_dim,
            dtype=torch.float32,
        ).reshape(
            1,
            num_kv_heads,
            sequence_length,
            head_dim,
        )

        value = key + 1000 + layer

        cache.append(
            (key, value)
        )

    return tuple(cache)


def test_sliding_window_keeps_recent_tokens():
    cache = make_fake_cache(
        sequence_length=10
    )

    result = sliding_window_evict(
        cache,
        window_size=4,
    )

    assert len(result) == 2

    for key, value in result:
        assert key.shape[-2] == 4
        assert value.shape[-2] == 4

    # The final four original positions must remain.
    original_key = cache[0][0]

    expected = original_key[..., -4:, :]

    assert torch.equal(
        result[0][0],
        expected,
    )


def test_cache_manager_enforces_budget():
    cache = make_fake_cache(
        sequence_length=10
    )

    manager = SlidingWindowCacheManager(
        window_size=4
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 4

    stats = manager.stats(result)

    assert stats.sequence_length == 4
    assert stats.budget == 4
    assert stats.total_evictions == 6


def test_cache_manager_does_not_evict_small_cache():
    cache = make_fake_cache(
        sequence_length=3
    )

    manager = SlidingWindowCacheManager(
        window_size=4
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 3
    assert manager.total_evictions == 0

    assert torch.equal(
        result[0][0],
        cache[0][0],
    )


if __name__ == "__main__":
    test_sliding_window_keeps_recent_tokens()
    test_cache_manager_enforces_budget()
    test_cache_manager_does_not_evict_small_cache()

    print("Stage 3 sliding-window tests passed.")