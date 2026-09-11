import torch

from src.cache_manager import HeavyHitterCacheManager
from src.evictions import heavy_hitter_evict
from src.model_wrapper import (
    ModelWrapper,
    aggregate_newest_attention,
    update_accumulated_scores,
)


def make_cache(
    sequence_length: int,
    num_layers: int = 2,
):
    cache = []

    for _ in range(num_layers):
        key = torch.arange(
            sequence_length,
            dtype=torch.float32,
        ).reshape(
            1,
            1,
            sequence_length,
            1,
        )

        value = (
            key + 100
        )

        cache.append(
            (key, value)
        )

    return tuple(cache)


def test_heavy_hitter_evict():
    cache = make_cache(8)

    attention_scores = torch.tensor(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.05,
        ]
    )

    result = heavy_hitter_evict(
        cache,
        attention_scores,
        cache_budget=4,
    )

    assert len(result) == 2

    retained = result[0][0][
        0,
        0,
        :,
        0,
    ]

    expected = torch.tensor(
        [0.0, 1.0, 3.0, 7.0]
    )

    assert torch.equal(
        retained,
        expected,
    )


def test_heavy_hitter_manager():
    cache = make_cache(8)

    attention_scores = torch.tensor(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.05,
        ]
    )

    manager = HeavyHitterCacheManager(
        cache_budget=4
    )

    updated_cache, updated_scores = (
        manager.update(
            cache,
            attention_scores,
        )
    )

    retained = updated_cache[0][0][
        0,
        0,
        :,
        0,
    ]

    expected_tokens = torch.tensor(
        [0.0, 1.0, 3.0, 7.0]
    )

    expected_scores = torch.tensor(
        [0.1, 0.9, 0.8, 0.05]
    )

    assert torch.equal(
        retained,
        expected_tokens,
    )

    assert torch.allclose(
        updated_scores,
        expected_scores,
    )

    assert (
        manager.total_evictions == 4
    )


def test_no_eviction_under_budget():
    cache = make_cache(4)

    attention_scores = torch.tensor(
        [0.1, 0.4, 0.2, 0.3]
    )

    manager = HeavyHitterCacheManager(
        cache_budget=8
    )

    updated_cache, updated_scores = (
        manager.update(
            cache,
            attention_scores,
        )
    )

    assert (
        updated_cache[0][0].shape[-2]
        == 4
    )

    assert torch.equal(
        updated_scores,
        attention_scores,
    )

    assert (
        manager.total_evictions == 0
    )


def test_scores_and_cache_length_must_match():
    cache = make_cache(8)

    attention_scores = torch.tensor(
        [0.1, 0.2, 0.3]
    )

    manager = HeavyHitterCacheManager(
        cache_budget=4
    )

    try:
        manager.update(
            cache,
            attention_scores,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for mismatched "
        "attention-score length."
    )


def test_reserved_sinks_recent_and_heavy_hitters():
    cache = make_cache(10)
    scores = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.5, 0.0, 0.0])
    manager = HeavyHitterCacheManager(
        cache_budget=5,
        sink_tokens=1,
        recent_window=2,
    )
    updated, updated_scores = manager.update(
        cache, scores, token_positions=list(range(10))
    )
    assert manager.retained_positions.tolist() == [0, 3, 5, 8, 9]
    assert len(set(manager.retained_positions.tolist())) == 5
    assert updated[0][0].shape[-2] == 5
    assert torch.equal(updated_scores, scores[[0, 3, 5, 8, 9]])


def test_attention_aggregation_and_accumulation():
    layer1 = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.3, 0.2, 0.1, 0.4]]]])
    layer2 = torch.tensor([[[[0.2, 0.2, 0.2, 0.4]], [[0.4, 0.2, 0.2, 0.2]]]])
    aggregated = aggregate_newest_attention((layer1, layer2))
    assert torch.allclose(aggregated, torch.tensor([0.25, 0.2, 0.2, 0.35]))
    first = update_accumulated_scores(None, aggregated)
    second = update_accumulated_scores(
        first, torch.tensor([0.1, 0.2, 0.3, 0.1, 0.3])
    )
    assert torch.allclose(second, torch.tensor([0.35, 0.4, 0.5, 0.45, 0.3]))


def test_generation_accumulates_attention_scores():
    wrapper = ModelWrapper(device="cpu")
    result = wrapper.heavy_hitter_generate(
        "zero one two three four five six seven eight nine ten",
        max_new_tokens=3,
        cache_budget=6,
        sink_tokens=1,
        recent_window=2,
        return_diagnostics=True,
    )
    assert result.score_updates == 3
    assert result.cache_length <= 6
    assert len(result.retained_positions) == result.cache_length
    assert len(set(result.retained_positions)) == result.cache_length
    assert result.retained_positions[0] == 0
    assert len(result.generated_token_ids) > 0


if __name__ == "__main__":
    test_heavy_hitter_evict()
    test_heavy_hitter_manager()
    test_no_eviction_under_budget()
    test_scores_and_cache_length_must_match()
    test_reserved_sinks_recent_and_heavy_hitters()
    test_attention_aggregation_and_accumulation()
    test_generation_accumulates_attention_scores()

    print(
        "Stage 5 heavy-hitter tests passed."
    )