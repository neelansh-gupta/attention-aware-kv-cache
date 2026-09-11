import torch

from src.cache_manager import SlidingWindowCacheManager
from src.cache_utils import cache_as_tensor_tuple
from src.evictions import sliding_window_evict
from src.model_wrapper import ModelWrapper


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


def test_retained_original_position_metadata():
    cache = make_fake_cache(sequence_length=10)
    manager = SlidingWindowCacheManager(window_size=4)
    result = manager.update(cache, token_positions=list(range(10)))
    assert manager.retained_positions.tolist() == [6, 7, 8, 9]
    assert torch.equal(result[0][0], cache[0][0][..., 6:10, :])
    assert torch.equal(result[0][1], cache[0][1][..., 6:10, :])


def test_reference_cache_and_generation():
    wrapper = ModelWrapper(device="cpu")
    inputs = wrapper.tokenize("zero one two three four five")
    with torch.no_grad():
        outputs = wrapper.model(**inputs, use_cache=True)
    reference = tuple(
        (key.clone(), value.clone())
        for key, value in cache_as_tensor_tuple(outputs.past_key_values)
    )
    manager = SlidingWindowCacheManager(window_size=inputs["input_ids"].shape[-1])
    unchanged = manager.update(
        outputs.past_key_values,
        token_positions=list(range(inputs["input_ids"].shape[-1])),
    )
    custom = cache_as_tensor_tuple(unchanged)
    assert len(reference) == len(custom)
    for (ref_key, ref_value), (key, value) in zip(reference, custom):
        assert torch.allclose(key, ref_key, atol=1e-6, rtol=1e-5)
        assert torch.allclose(value, ref_value, atol=1e-6, rtol=1e-5)

    result = wrapper.sliding_window_generate(
        "zero one two three four five six seven eight nine",
        max_new_tokens=2,
        window_size=4,
        return_diagnostics=True,
    )
    assert result.cache_length <= 4
    assert len(result.retained_positions) == result.cache_length
    assert result.retained_positions == sorted(result.retained_positions)
    assert len(result.generated_token_ids) > 0


if __name__ == "__main__":
    test_sliding_window_keeps_recent_tokens()
    test_cache_manager_enforces_budget()
    test_cache_manager_does_not_evict_small_cache()
    test_retained_original_position_metadata()
    test_reference_cache_and_generation()

    print("Stage 3 sliding-window tests passed.")