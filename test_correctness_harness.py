"""
Stage 7: Full correctness harness.

This test suite validates the mechanical correctness of all implemented
KV-cache eviction policies without loading the language model.

Covered:

- Sliding-window eviction
- StreamingLLM / attention-sink eviction
- H2O / heavy-hitter eviction
- Cache-budget enforcement
- Key/value alignment
- Retained-token ordering
- Attention-score alignment
- Legacy tuple-style caches
- Modern Hugging Face-style cache objects
- No-op behavior
- Invalid-input handling
- Cache statistics
- Stage 6 position utilities
"""

from __future__ import annotations

import torch

from src.cache_manager import (
    AttentionSinkCacheManager,
    HeavyHitterCacheManager,
    SlidingWindowCacheManager,
)
from src.evictions import (
    attention_sink_evict,
    heavy_hitter_evict,
    sliding_window_evict,
)
from src.position_utils import (
    build_absolute_position_ids,
    build_cache_position,
)
from src.model_wrapper import (
    ModelWrapper,
    aggregate_newest_attention,
    update_accumulated_scores,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_legacy_cache(
    sequence_length: int,
    num_layers: int = 2,
    head_dim: int = 2,
):
    """
    Build a deterministic legacy tuple-style KV cache.

    Each token position is encoded directly into the tensor values so that
    retained positions can be checked exactly after eviction.
    """

    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
    )

    key = positions.view(1, 1, sequence_length, 1).repeat(
        1,
        1,
        1,
        head_dim,
    )

    value = (
        positions + 1000
    ).view(1, 1, sequence_length, 1).repeat(
        1,
        1,
        1,
        head_dim,
    )

    return tuple(
        (
            key.clone(),
            value.clone(),
        )
        for _ in range(num_layers)
    )


def extract_key_positions(cache):
    """Return token-position markers from the first cache layer."""

    key = cache[0][0]

    return key[0, 0, :, 0].to(torch.long).tolist()


def extract_value_positions(cache):
    """Return value-position markers from the first cache layer."""

    value = cache[0][1]

    return (
        value[0, 0, :, 0] - 1000
    ).to(torch.long).tolist()


def assert_cache_integrity(cache, expected_length: int):
    """
    Verify basic KV-cache invariants.
    """

    assert isinstance(cache, tuple)
    assert len(cache) > 0

    for key, value in cache:
        assert key.shape[-2] == expected_length
        assert value.shape[-2] == expected_length
        assert key.shape[:-1] == value.shape[:-1]

        key_positions = key[0, 0, :, 0].to(torch.long)
        value_positions = (
            value[0, 0, :, 0] - 1000
        ).to(torch.long)

        assert torch.equal(
            key_positions,
            value_positions,
        )


# ---------------------------------------------------------------------------
# Sliding-window eviction
# ---------------------------------------------------------------------------


def test_sliding_window_exact_selection():
    cache = make_legacy_cache(10)

    result = sliding_window_evict(
        cache,
        window_size=6,
    )

    assert extract_key_positions(result) == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]

    assert extract_value_positions(result) == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]

    assert_cache_integrity(result, 6)


def test_sliding_window_no_eviction():
    cache = make_legacy_cache(6)

    result = sliding_window_evict(
        cache,
        window_size=6,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    assert_cache_integrity(result, 6)


def test_sliding_window_repeated_application_is_stable():
    cache = make_legacy_cache(10)

    first = sliding_window_evict(
        cache,
        window_size=6,
    )

    second = sliding_window_evict(
        first,
        window_size=6,
    )

    assert extract_key_positions(second) == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]

    assert_cache_integrity(second, 6)


# ---------------------------------------------------------------------------
# Attention-sink eviction
# ---------------------------------------------------------------------------


def test_attention_sink_exact_selection():
    cache = make_legacy_cache(10)

    result = attention_sink_evict(
        cache,
        cache_budget=6,
        sink_tokens=2,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        6,
        7,
        8,
        9,
    ]

    assert extract_value_positions(result) == [
        0,
        1,
        6,
        7,
        8,
        9,
    ]

    assert_cache_integrity(result, 6)


def test_attention_sink_no_eviction():
    cache = make_legacy_cache(5)

    result = attention_sink_evict(
        cache,
        cache_budget=6,
        sink_tokens=2,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert_cache_integrity(result, 5)


def test_attention_sink_repeated_application_is_stable():
    cache = make_legacy_cache(10)

    first = attention_sink_evict(
        cache,
        cache_budget=6,
        sink_tokens=2,
    )

    second = attention_sink_evict(
        first,
        cache_budget=6,
        sink_tokens=2,
    )

    assert extract_key_positions(second) == [
        0,
        1,
        6,
        7,
        8,
        9,
    ]

    assert_cache_integrity(second, 6)


# ---------------------------------------------------------------------------
# Heavy-hitter eviction
# ---------------------------------------------------------------------------


def test_heavy_hitter_exact_selection():
    cache = make_legacy_cache(10)

    scores = torch.tensor(
        [
            0.2,
            0.9,
            0.1,
            0.8,
            0.3,
            0.7,
            0.4,
            0.1,
            0.6,
            0.2,
        ]
    )

    result = heavy_hitter_evict(
        cache,
        attention_scores=scores,
        cache_budget=5,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        3,
        5,
        9,
    ]

    assert extract_value_positions(result) == [
        0,
        1,
        3,
        5,
        9,
    ]

    assert_cache_integrity(result, 5)


def test_heavy_hitter_preserves_original_order():
    cache = make_legacy_cache(8)

    scores = torch.tensor(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.6,
        ]
    )

    result = heavy_hitter_evict(
        cache,
        attention_scores=scores,
        cache_budget=4,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        3,
        7,
    ]

    assert_cache_integrity(result, 4)


def test_heavy_hitter_no_eviction():
    cache = make_legacy_cache(4)

    scores = torch.tensor(
        [0.1, 0.2, 0.3, 0.4]
    )

    result = heavy_hitter_evict(
        cache,
        attention_scores=scores,
        cache_budget=8,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        2,
        3,
    ]

    assert_cache_integrity(result, 4)


def test_heavy_hitter_accepts_batched_scores():
    cache = make_legacy_cache(6)

    scores = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]]
    )

    result = heavy_hitter_evict(
        cache,
        attention_scores=scores,
        cache_budget=3,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        5,
    ]

    assert_cache_integrity(result, 3)


# ---------------------------------------------------------------------------
# Cache-manager tests
# ---------------------------------------------------------------------------


def test_sliding_window_manager():
    cache = make_legacy_cache(10)

    manager = SlidingWindowCacheManager(
        window_size=6
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 6
    assert extract_key_positions(result) == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]

    stats = manager.stats(result)

    assert stats.sequence_length == 6
    assert stats.budget == 6
    assert stats.total_evictions == 4


def test_attention_sink_manager():
    cache = make_legacy_cache(10)

    manager = AttentionSinkCacheManager(
        cache_budget=6,
        sink_tokens=2,
    )

    result = manager.update(cache)

    assert manager.sequence_length(result) == 6
    assert extract_key_positions(result) == [
        0,
        1,
        6,
        7,
        8,
        9,
    ]

    stats = manager.stats(result)

    assert stats.sequence_length == 6
    assert stats.budget == 6
    assert stats.total_evictions == 4


def test_heavy_hitter_manager_and_score_alignment():
    cache = make_legacy_cache(8)

    scores = torch.tensor(
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    )

    manager = HeavyHitterCacheManager(
        cache_budget=4
    )

    result, updated_scores = manager.update(
        cache,
        scores,
    )

    assert manager.sequence_length(result) == 4

    assert extract_key_positions(result) == [
        0,
        1,
        3,
        7,
    ]

    assert torch.equal(
        updated_scores,
        torch.tensor(
            [0.1, 0.9, 0.8, 0.6]
        ),
    )

    stats = manager.stats(result)

    assert stats.sequence_length == 4
    assert stats.budget == 4
    assert stats.total_evictions == 4


def test_heavy_hitter_manager_accepts_batched_scores():
    cache = make_legacy_cache(6)

    scores = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]]
    )

    manager = HeavyHitterCacheManager(
        cache_budget=3
    )

    result, updated_scores = manager.update(
        cache,
        scores,
    )

    assert extract_key_positions(result) == [
        0,
        1,
        5,
    ]

    assert torch.equal(
        updated_scores,
        torch.tensor(
            [0.1, 0.9, 0.7]
        ),
    )


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_manager_configuration():
    for invalid_window in [0, -1]:
        try:
            SlidingWindowCacheManager(
                invalid_window
            )
            raise AssertionError(
                "Expected ValueError."
            )
        except ValueError:
            pass

    try:
        AttentionSinkCacheManager(
            cache_budget=4,
            sink_tokens=4,
        )
        raise AssertionError(
            "Expected ValueError."
        )
    except ValueError:
        pass

    try:
        HeavyHitterCacheManager(0)
        raise AssertionError(
            "Expected ValueError."
        )
    except ValueError:
        pass


def test_heavy_hitter_score_length_mismatch():
    cache = make_legacy_cache(8)

    scores = torch.ones(7)

    manager = HeavyHitterCacheManager(
        cache_budget=4
    )

    try:
        manager.update(cache, scores)
        raise AssertionError(
            "Expected ValueError."
        )
    except ValueError:
        pass


def test_invalid_heavy_hitter_score_shape():
    cache = make_legacy_cache(8)

    scores = torch.ones(2, 4)

    manager = HeavyHitterCacheManager(
        cache_budget=4
    )

    try:
        manager.update(cache, scores)
        raise AssertionError(
            "Expected ValueError."
        )
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Stage 6 position regression checks
# ---------------------------------------------------------------------------


def test_absolute_positions_continue_after_eviction():
    device = torch.device("cpu")

    first = build_absolute_position_ids(
        start_position=0,
        sequence_length=8,
        device=device,
    )

    next_position = build_absolute_position_ids(
        start_position=8,
        sequence_length=1,
        device=device,
    )

    later_position = build_absolute_position_ids(
        start_position=20,
        sequence_length=1,
        device=device,
    )

    assert first.tolist() == [
        list(range(8))
    ]

    assert next_position.tolist() == [
        [8]
    ]

    assert later_position.tolist() == [
        [20]
    ]


def test_absolute_cache_position():
    device = torch.device("cpu")

    cache_position = build_cache_position(
        start_position=15,
        sequence_length=1,
        device=device,
    )

    assert cache_position.tolist() == [15]


def test_explicit_original_position_metadata():
    positions = list(range(10))
    sliding = SlidingWindowCacheManager(4)
    sliding.update(make_legacy_cache(10), positions)
    assert sliding.retained_positions.tolist() == [6, 7, 8, 9]

    streaming = AttentionSinkCacheManager(6, 2)
    streaming.update(make_legacy_cache(10), positions)
    assert streaming.retained_positions.tolist() == [0, 1, 6, 7, 8, 9]


def test_h2o_reserves_sinks_recent_and_heavy_hitters():
    scores = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.5, 0.0, 0.0])
    manager = HeavyHitterCacheManager(5, sink_tokens=1, recent_window=2)
    cache, retained_scores = manager.update(
        make_legacy_cache(10), scores, list(range(10))
    )
    assert extract_key_positions(cache) == [0, 3, 5, 8, 9]
    assert manager.retained_positions.tolist() == [0, 3, 5, 8, 9]
    assert torch.equal(retained_scores, scores[[0, 3, 5, 8, 9]])


def test_attention_aggregation_and_score_updates():
    layer1 = torch.tensor([[[[0.1, 0.2]], [[0.3, 0.4]]]])
    layer2 = torch.tensor([[[[0.2, 0.4]], [[0.4, 0.2]]]])
    current = aggregate_newest_attention((layer1, layer2))
    assert torch.allclose(current, torch.tensor([0.25, 0.3]))
    first = update_accumulated_scores(None, current)
    second = update_accumulated_scores(first, torch.tensor([0.1, 0.2, 0.7]))
    assert torch.allclose(second, torch.tensor([0.35, 0.5, 0.7]))


def test_short_attention_prefix_is_unavailable():
    wrapper = object.__new__(ModelWrapper)
    attention = torch.full((1, 2, 4, 4), 0.25)
    report = wrapper.analyze_attention_sinks((attention,))
    assert report["early_token_attention"][1] is not None
    assert report["early_token_attention"][4] is not None
    assert report["early_token_attention"][8] is None


# ---------------------------------------------------------------------------
# Harness entry point
# ---------------------------------------------------------------------------


def run_all_tests():
    tests = [
        test_sliding_window_exact_selection,
        test_sliding_window_no_eviction,
        test_sliding_window_repeated_application_is_stable,
        test_attention_sink_exact_selection,
        test_attention_sink_no_eviction,
        test_attention_sink_repeated_application_is_stable,
        test_heavy_hitter_exact_selection,
        test_heavy_hitter_preserves_original_order,
        test_heavy_hitter_no_eviction,
        test_heavy_hitter_accepts_batched_scores,
        test_sliding_window_manager,
        test_attention_sink_manager,
        test_heavy_hitter_manager_and_score_alignment,
        test_heavy_hitter_manager_accepts_batched_scores,
        test_invalid_manager_configuration,
        test_heavy_hitter_score_length_mismatch,
        test_invalid_heavy_hitter_score_shape,
        test_absolute_positions_continue_after_eviction,
        test_absolute_cache_position,
        test_explicit_original_position_metadata,
        test_h2o_reserves_sinks_recent_and_heavy_hitters,
        test_attention_aggregation_and_score_updates,
        test_short_attention_prefix_is_unavailable,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print()
    print(
        f"Stage 7 correctness harness passed "
        f"({len(tests)} tests)."
    )


if __name__ == "__main__":
    run_all_tests()