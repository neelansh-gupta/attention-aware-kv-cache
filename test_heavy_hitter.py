import torch

from src.cache_manager import HeavyHitterCacheManager
from src.evictions import heavy_hitter_evict


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
        [1.0, 3.0, 5.0, 6.0]
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
        [1.0, 3.0, 5.0, 6.0]
    )

    expected_scores = torch.tensor(
        [0.9, 0.8, 0.7, 0.4]
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


if __name__ == "__main__":
    test_heavy_hitter_evict()
    test_heavy_hitter_manager()
    test_no_eviction_under_budget()
    test_scores_and_cache_length_must_match()

    print(
        "Stage 5 heavy-hitter tests passed."
    )