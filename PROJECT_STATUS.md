# Attention-Aware KV Cache Compression

Postman AI/ML Recruitment Task 3 (Batch 25)

---

## Overall Status

| Stage | Description | Status |
|---|---|---|
| Stage 0 | Project foundation and scope | ✅ Complete |
| Stage 1 | Model loading and KV cache inspection | ✅ Complete |
| Stage 2 | Baseline generation and cache interface | ✅ Complete |
| Stage 3 | Sliding-window KV cache eviction | ✅ Complete |
| Stage 4 | StreamingLLM-style attention-sink-aware eviction | ✅ Complete |
| Stage 5 | H2O-style heavy-hitter eviction | ✅ Complete |
| Stage 6 | RoPE and position handling after eviction | ✅ Complete |
| Stage 7 | Full correctness harness | ✅ Complete |
| Stage 8 | Benchmarking and evaluation | ⏭️ Next |

---

# Stage 0 — Project Foundation

**Status: Complete**

Established the project structure, requirements, documentation, and initial scope.

---

# Stage 1 — Model Loading and KV Cache Inspection

**Status: Complete**

Implemented the model wrapper and verified that the selected language model can be loaded and used for generation.

The stage also established the interface used to inspect and work with the model's KV cache.

---

# Stage 2 — Baseline Generation and Cache Interface

**Status: Complete**

Established baseline generation behavior and the cache interfaces required by later eviction strategies.

The project can now perform generation while accessing and manipulating the KV cache.

---

# Stage 3 — Sliding-Window KV Cache Eviction

**Status: Complete**

Implemented a sliding-window eviction policy.

The policy keeps the most recent tokens in the KV cache and removes older entries once the cache exceeds the configured window size.

Implemented components include:

- Sliding-window eviction utility
- `SlidingWindowCacheManager`
- Generation integration
- Validation tests

Validation:

    Stage 3 sliding-window tests passed.

---

# Stage 4 — StreamingLLM Attention-Sink Eviction

**Status: Complete**

Implemented a StreamingLLM-style attention-sink-aware eviction policy.

The policy preserves:

- A fixed number of initial sink tokens
- The most recent tokens

Older middle tokens are removed when the cache exceeds the configured budget.

Implemented components include:

- Attention-sink eviction utility
- `AttentionSinkCacheManager`
- Generation integration
- Validation tests

Validation:

    Stage 4 attention-sink tests passed.

---

# Stage 5 — H2O Heavy-Hitter Eviction

**Status: Complete**

Implemented an H2O-style heavy-hitter KV cache policy.

The policy uses accumulated attention scores to identify tokens that should remain in the cache.

Implemented components include:

- Heavy-hitter eviction utility
- `HeavyHitterCacheManager`
- Attention-score accumulation
- Generation integration
- Batched attention-score handling
- Validation tests

The generation path:

1. Runs the model with attention outputs enabled.
2. Extracts attention for the newest query token.
3. Averages attention across heads and layers.
4. Accumulates token-level attention scores.
5. Selects the highest-scoring tokens.
6. Preserves their original sequence order.
7. Evicts the remaining tokens.

Validation:

    Stage 5 heavy-hitter tests passed.

A longer H2O generation test was also performed. The generation completed successfully, although the output became highly repetitive at a very small cache budget. This was not treated as a quality benchmark.

---

# Stage 6 — RoPE and Position Handling

**Status: Complete**

Fixed positional handling after KV cache eviction.

The physical length of the KV cache can decrease after eviction, but token positions must continue to represent their original absolute positions.

Implemented:

- Absolute `position_ids`
- Absolute `cache_position`
- Tracking of the next absolute token position
- RoPE cache extension when required
- `src/position_utils.py`
- Position-handling regression tests

The generation paths now maintain position information independently from the physical cache length.

Validation:

    Stage 6 position-handling tests passed.

Additional generation checks were performed for:

- H2O heavy-hitter generation
- Sliding-window generation
- StreamingLLM generation

The generation paths completed successfully after the position-handling changes.

---

# Stage 7 — Full Correctness Harness

**Status: Complete**

Implemented a deterministic, CPU-only correctness harness covering the cache eviction mechanisms and position-handling logic.

The goal of Stage 7 is to verify **correctness of cache manipulation**, rather than model quality or benchmark performance.

## Stage 7 Validation

The final correctness harness contains **19 tests**.

All tests passed:

    PASS: test_sliding_window_exact_selection
    PASS: test_sliding_window_no_eviction
    PASS: test_sliding_window_repeated_application_is_stable

    PASS: test_attention_sink_exact_selection
    PASS: test_attention_sink_no_eviction
    PASS: test_attention_sink_repeated_application_is_stable

    PASS: test_heavy_hitter_exact_selection
    PASS: test_heavy_hitter_preserves_original_order
    PASS: test_heavy_hitter_no_eviction
    PASS: test_heavy_hitter_accepts_batched_scores

    PASS: test_sliding_window_manager
    PASS: test_attention_sink_manager
    PASS: test_heavy_hitter_manager_and_score_alignment
    PASS: test_heavy_hitter_manager_accepts_batched_scores

    PASS: test_invalid_manager_configuration
    PASS: test_heavy_hitter_score_length_mismatch
    PASS: test_invalid_heavy_hitter_score_shape

    PASS: test_absolute_positions_continue_after_eviction
    PASS: test_absolute_cache_position

Final result:

    Stage 7 correctness harness passed (19 tests).

## What Stage 7 Verifies

### Sliding Window

- Exact retained-token selection
- Correct behavior when the cache is below budget
- Stable repeated application of eviction

### Attention Sink

- Exact preservation of sink and recent tokens
- Correct no-eviction behavior
- Stable repeated application

### H2O Heavy Hitter

- Correct top-k token selection
- Original sequence ordering is preserved
- Correct no-eviction behavior
- Batched attention-score support
- Attention-score and KV-cache alignment
- Invalid score lengths and shapes are rejected

### Cache Managers

- Correct manager behavior for all three policies
- Cache-budget enforcement
- Eviction statistics
- Stable repeated updates

### Position Handling

- Absolute token positions continue correctly after cache eviction
- `cache_position` remains independent of physical cache length

## Scope

Stage 7 does not attempt to measure:

- Perplexity
- Generation quality
- Throughput
- Memory consumption
- Latency

Those measurements are part of the later benchmarking and evaluation stage.

---

# Current Repository Components

    attention-aware-kv-cache/
    ├── results/
    ├── src/
    │   ├── cache_manager.py
    │   ├── evictions.py
    │   ├── model_wrapper.py
    │   └── position_utils.py
    ├── .gitignore
    ├── PROJECT_STATUS.md
    ├── README.md
    ├── requirements.txt
    ├── test_attention_sink.py
    ├── test_correctness_harness.py
    ├── test_heavy_hitter.py
    ├── test_position_handling.py
    ├── test_sliding_window.py
    └── visualise.py

---

# Next Stage

## Stage 8 — Benchmarking and Evaluation

The next stage will focus on quantitative evaluation of the different KV cache policies.

Planned areas include:

- Generation latency
- Memory/cache size
- Compression ratio
- Generation behavior
- Comparison across cache budgets
- Baseline vs compressed-cache performance
- Reproducible benchmark results
- Result collection and visualization