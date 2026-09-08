import torch

from src.cache_manager import AttentionSinkCacheManager
from src.evictions import attention_sink_evict


def make_fake_cache(
    sequence_length: int,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 4,
):
    cache = []

    for layer in range(num_layers):
        key = torch.arange(
            sequence_length
            * num_kv_heads
            * head_dim,
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


def test_attention_sink_keeps_prefix_and_suffix():
    cache = make_fake_cache(
        sequence_length=10
    )

    result = attention_sink_evict(
        cache,
        cache_budget=6,
        sink_tokens=2,
    )

    for key, value in result:
        assert key.shape[-2] == 6
        assert value.shape[-2] == 6

    original_key = cache[0][0]

    expected = torch.cat(
        [
            original_key[..., :2, :],
            original_key[..., -4:, :],
        ],
        dim=-2,
    )

    assert torch.equal(
        result[0][0],
        expected,
    )


def test_attention_sink_manager_enforces_budget():
    cache = make_fake_cache(
        sequence_length=10
    )

    manager = AttentionSinkCacheManager(
        cache_budget=6,
        sink_tokens=2,
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 6

    stats = manager.stats(result)

    assert stats.sequence_length == 6
    assert stats.budget == 6
    assert stats.total_evictions == 4


def test_attention_sink_does_not_evict_small_cache():
    cache = make_fake_cache(
        sequence_length=5
    )

    manager = AttentionSinkCacheManager(
        cache_budget=6,
        sink_tokens=2,
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 5
    assert manager.total_evictions == 0

    assert torch.equal(
        result[0][0],
        cache[0][0],
    )


def test_attention_sink_preserves_all_layers():
    cache = make_fake_cache(
        sequence_length=12,
        num_layers=4,
    )

    result = attention_sink_evict(
        cache,
        cache_budget=8,
        sink_tokens=3,
    )

    assert len(result) == 4

    for key, value in result:
        assert key.shape[-2] == 8
        assert value.shape[-2] == 8


def test_invalid_sink_configuration():
    try:
        AttentionSinkCacheManager(
            cache_budget=4,
            sink_tokens=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError when sink_tokens >= cache_budget."
        )


if __name__ == "__main__":
    test_attention_sink_keeps_prefix_and_suffix()
    test_attention_sink_manager_enforces_budget()
    test_attention_sink_does_not_evict_small_cache()
    test_attention_sink_preserves_all_layers()
    test_invalid_sink_configuration()

    print(
        "Stage 4 attention-sink tests passed."
    )